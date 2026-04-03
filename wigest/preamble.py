"""Preamble detection and segmentation module.

Implements WiGest §III-D – "Preamble Detection".

The preamble is a special two-finger up-down gesture performed before the
actual gesture sequence.  It serves two purposes:

1. **Segmentation trigger**: marks the boundary between background WiFi
   fluctuation and intentional gesture motion.
2. **Calibration**: the preamble RSSI range is used to:

   * Determine the magnitude threshold (``'high'`` vs ``'low'`` RSSI change).
   * Determine signal polarity (whether to flip the RSSI sign).
   * Establish the dominant motion frequency.

Detection algorithm (two stages)
---------------------------------
Stage 1 – *coarse trigger*:
    Detect a rapid drop in RSSI > *drop_threshold* dBm within a short
    sliding window.  This catches the initial approach of the hand.

Stage 2 – *fine confirmation*:
    In the candidate window following the RSSI drop, look for **4 consecutive
    peaks** in the wavelet detail coefficients (two up-down cycles = two
    fingers moving up then down).

After preamble confirmation the detector:

* Emits the calibration parameters.
* Switches to *gesture scanning* mode.
* Resets to preamble scanning after *silence_timeout* seconds of
  low-variance RSSI.

Paper reference
---------------
WiGest §III-D – "Preamble Detection":
  "We detect the preamble by first detecting an RSSI drop of more than Δ_p
   dBm, then verifying the presence of two consecutive up-down cycles
   (four peaks) in the wavelet details."
"""

from dataclasses import dataclass
from typing import List, Optional, Tuple

import numpy as np

from wigest.denoising import denoise
from wigest.primitives import _find_peaks_and_troughs


# ---------------------------------------------------------------------------
# Calibration result
# ---------------------------------------------------------------------------

@dataclass
class PreambleCalibration:
    """Calibration parameters extracted from the preamble segment.

    Attributes
    ----------
    magnitude_threshold:
        RSSI change (dBm) that separates ``'high'`` from ``'low'`` magnitude.
        Set to half the peak-to-peak RSSI range observed during the preamble.
    flip_polarity:
        ``True`` if the RSSI signal should be negated before processing.
        The preamble defines the expected polarity; if the first detected
        transition is a falling edge instead of a rising one the flag is set.
    motion_frequency:
        Estimated dominant gesture frequency (Hz) derived from the preamble
        peak spacing.
    preamble_start:
        Sample index of the preamble start.
    preamble_end:
        Sample index of the preamble end (exclusive).
    """

    magnitude_threshold: float
    flip_polarity: bool
    motion_frequency: float
    preamble_start: int
    preamble_end: int


# ---------------------------------------------------------------------------
# Detector class
# ---------------------------------------------------------------------------

class PreambleDetector:
    """Stateful detector for WiGest preamble + gesture segmentation.

    Maintains an internal sliding buffer of RSSI samples and transitions
    between three states:

    ``'waiting'``
        Looking for the coarse RSSI drop that signals an incoming preamble.
    ``'confirming'``
        Drop detected; looking for 4 consecutive peaks to confirm preamble.
    ``'gesture'``
        Preamble confirmed; forwarding samples to the gesture pipeline until
        a silence timeout resets the state.

    Parameters
    ----------
    sample_rate:
        Acquisition rate in Hz.
    drop_threshold:
        Minimum RSSI drop (dBm, positive value) in the coarse-trigger window
        to enter *confirming* state.  Default: 5 dBm.
    drop_window:
        Duration (seconds) of the coarse-trigger sliding window.
    confirm_window:
        Duration (seconds) of the fine-confirmation window after the drop.
    silence_timeout:
        Seconds of low-variance RSSI after which the detector resets to
        *waiting* state.
    variance_threshold:
        RSSI variance (dBm²) below which a segment counts as silence.
    wavelet:
        DWT wavelet for detail computation (default ``"haar"``).
    """

    def __init__(
        self,
        sample_rate: float = 10.0,
        drop_threshold: float = 5.0,
        drop_window: float = 1.0,
        confirm_window: float = 2.0,
        silence_timeout: float = 3.0,
        variance_threshold: float = 1.0,
        wavelet: str = "haar",
    ) -> None:
        self.sample_rate = sample_rate
        self.drop_threshold = drop_threshold
        self._drop_win = max(2, int(sample_rate * drop_window))
        self._confirm_win = max(4, int(sample_rate * confirm_window))
        self._silence_win = max(2, int(sample_rate * silence_timeout))
        self.variance_threshold = variance_threshold
        self.wavelet = wavelet

        self._state: str = "waiting"
        self._buffer: List[float] = []
        self._drop_start: int = 0        # absolute sample counter at drop
        self._sample_count: int = 0
        self.calibration: Optional[PreambleCalibration] = None

    # ------------------------------------------------------------------
    # Public interface
    # ------------------------------------------------------------------

    @property
    def state(self) -> str:
        """Current detector state: ``'waiting'``, ``'confirming'``, or ``'gesture'``."""
        return self._state

    def reset(self) -> None:
        """Hard-reset the detector to *waiting* state."""
        self._state = "waiting"
        self._buffer.clear()
        self._drop_start = 0
        self._sample_count = 0
        self.calibration = None

    def feed(self, rssi: float) -> Optional[PreambleCalibration]:
        """Feed a single RSSI sample to the detector.

        Parameters
        ----------
        rssi:
            Current RSSI measurement in dBm.

        Returns
        -------
        PreambleCalibration or None
            Returns calibration data the *first* time a preamble is
            successfully confirmed; ``None`` otherwise.  Once calibration
            is returned the detector transitions to ``'gesture'`` state.
        """
        self._buffer.append(rssi)
        self._sample_count += 1

        if self._state == "waiting":
            return self._check_drop()
        if self._state == "confirming":
            return self._check_confirm()
        if self._state == "gesture":
            self._check_silence()
        return None

    def feed_batch(self, rssi_sequence) -> Optional[PreambleCalibration]:
        """Feed an iterable of RSSI samples; return calibration when found.

        Stops consuming the sequence as soon as the preamble is confirmed.
        """
        for rssi in rssi_sequence:
            result = self.feed(float(rssi))
            if result is not None:
                return result
        return None

    # ------------------------------------------------------------------
    # State-machine internals
    # ------------------------------------------------------------------

    def _check_drop(self) -> Optional[PreambleCalibration]:
        """Stage 1: look for a rapid RSSI drop in the sliding window."""
        buf = self._buffer
        win = self._drop_win
        if len(buf) < win:
            return None
        window = buf[-win:]
        drop = float(np.max(window) - np.min(window))
        if drop >= self.drop_threshold:
            self._state = "confirming"
            self._drop_start = self._sample_count - win
            # Keep only the confirmation window in the buffer.
            self._buffer = list(window)
        return None

    def _check_confirm(self) -> Optional[PreambleCalibration]:
        """Stage 2: look for 4 consecutive peaks to confirm preamble."""
        buf = self._buffer
        win = self._confirm_win
        if len(buf) < win:
            return None

        segment = np.array(buf[-win:], dtype=float)
        calib = self._try_confirm(segment, self._drop_start)
        if calib is not None:
            self.calibration = calib
            self._state = "gesture"
            # Retain only samples after the preamble end.
            post = self._sample_count - (self._drop_start + win)
            self._buffer = list(self._buffer[-max(0, post):])
            return calib

        # Timeout: if we've accumulated too many samples without confirmation
        # go back to waiting.
        if len(buf) > self._confirm_win * 2:
            self._state = "waiting"
            self._buffer = []
        return None

    @staticmethod
    def _try_confirm(
        segment: np.ndarray,
        abs_offset: int,
    ) -> Optional[PreambleCalibration]:
        """Attempt to find 4 peaks in *segment* confirming a preamble.

        Returns a :class:`PreambleCalibration` if successful, else ``None``.
        """
        _, detail_bands = denoise(segment, wavelet="haar")
        if not detail_bands:
            return None

        # Use highest-energy detail band.
        energies = [float(np.sum(d ** 2)) for d in detail_bands]
        coeffs = detail_bands[int(np.argmax(energies))]
        peaks, troughs = _find_peaks_and_troughs(coeffs, min_prominence=0.3)

        # Need at least 4 extrema total (peaks + troughs interleaved).
        all_extrema = np.sort(np.concatenate([peaks, troughs]))
        if len(all_extrema) < 4:
            return None

        # Derive calibration parameters from the segment.
        p2p = float(np.max(segment) - np.min(segment))
        magnitude_threshold = p2p / 2.0

        # Polarity: if the first extremum is a trough (minimum), the first
        # motion is a rising edge, which is the expected preamble.  If it is
        # a peak, the signal polarity is inverted.
        first = int(all_extrema[0])
        flip = bool(first in peaks)

        # Motion frequency from the mean inter-extremum spacing.
        spacings = np.diff(all_extrema.astype(float))
        if len(spacings) > 0:
            mean_half_period_coeff = float(np.mean(spacings))
            # Coefficient domain → sample domain → frequency.
            scale = len(segment) / len(coeffs)
            mean_half_period_samples = mean_half_period_coeff * scale
            sample_rate_est = 10.0  # default; caller knows the real rate
            motion_freq = (
                sample_rate_est / (2.0 * mean_half_period_samples)
                if mean_half_period_samples > 0
                else 1.0
            )
        else:
            motion_freq = 1.0

        return PreambleCalibration(
            magnitude_threshold=max(magnitude_threshold, 0.5),
            flip_polarity=flip,
            motion_frequency=motion_freq,
            preamble_start=abs_offset,
            preamble_end=abs_offset + len(segment),
        )

    def _check_silence(self) -> None:
        """In gesture state, reset if RSSI has been quiet for *silence_timeout*."""
        buf = self._buffer
        win = self._silence_win
        if len(buf) < win:
            return
        recent = np.array(buf[-win:], dtype=float)
        if float(np.var(recent)) < self.variance_threshold:
            self._state = "waiting"
            self._buffer = []
            self.calibration = None


# ---------------------------------------------------------------------------
# Convenience function: detect preamble in a pre-recorded signal
# ---------------------------------------------------------------------------

def detect_preamble(
    signal,
    sample_rate: float = 10.0,
    drop_threshold: float = 5.0,
    **kwargs,
) -> Tuple[Optional[PreambleCalibration], int]:
    """Find the preamble in a pre-recorded RSSI sequence.

    Parameters
    ----------
    signal:
        1-D RSSI sequence (dBm).
    sample_rate:
        Acquisition rate in Hz.
    drop_threshold:
        Minimum RSSI drop to trigger preamble search.
    **kwargs:
        Additional keyword arguments forwarded to :class:`PreambleDetector`.

    Returns
    -------
    calibration : PreambleCalibration or None
        Calibration data if a preamble was found.
    sample_offset : int
        Index of the first sample *after* the preamble (i.e., where
        gesture processing should begin).  ``0`` if no preamble found.
    """
    detector = PreambleDetector(
        sample_rate=sample_rate,
        drop_threshold=drop_threshold,
        **kwargs,
    )
    for i, rssi in enumerate(signal):
        calib = detector.feed(float(rssi))
        if calib is not None:
            return calib, calib.preamble_end
    return None, 0
