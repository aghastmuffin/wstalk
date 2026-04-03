"""Main WiGest pipeline controller.

:class:`WiGest` ties all sub-modules together into a single high-level
interface:

1. **RSSI acquisition** via a :class:`~wigest.rssi.RSSISource`.
2. **Preamble detection** to segment and calibrate the signal.
3. **Denoising** (DWT/Haar + SURE thresholding).
4. **Primitive extraction** (edge detection, pause detection).
5. **Multi-AP voting** when more than one AP is present.
6. **Gesture matching** (pattern encoding + family lookup).
7. **Action dispatch** via an optional :class:`~wigest.actions.ActionMapper`.

Typical usage::

    from wigest import WiGest, MockRSSISource, ActionMapper

    source = MockRSSISource(my_rssi_array)
    mapper = ActionMapper()

    @mapper.on("push")
    def handle_push(result):
        print("Push detected!", result)

    wg = WiGest(source, action_mapper=mapper)
    for result in wg.run(max_gestures=5):
        print(result)

Paper reference
---------------
WiGest (INFOCOM 2015) – §III, full pipeline.
"""

import logging
from typing import Iterator, List, Optional

import numpy as np

from wigest.actions import ActionMapper
from wigest.gestures import GestureResult, match_gesture
from wigest.preamble import PreambleCalibration, PreambleDetector
from wigest.primitives import Primitive, extract_primitives
from wigest.rssi import RSSISource
from wigest.voting import majority_vote

logger = logging.getLogger(__name__)


class WiGest:
    """High-level WiGest gesture recognition pipeline.

    Parameters
    ----------
    source:
        :class:`~wigest.rssi.RSSISource` providing RSSI samples.
    action_mapper:
        Optional :class:`~wigest.actions.ActionMapper`.  When provided,
        :meth:`dispatch` is called automatically for each recognised gesture.
    sample_rate:
        Acquisition rate in Hz (must match the source's actual rate).
    buffer_duration:
        Seconds of RSSI data to accumulate before running primitive
        extraction.  A longer buffer captures slower gestures but increases
        latency.
    magnitude_threshold:
        Default RSSI change threshold (dBm) for magnitude classification.
        Overridden by preamble calibration when available.
    variance_threshold:
        RSSI variance (dBm²) used for pause / silence detection.
    preamble:
        If ``True`` (default), require a preamble before processing
        gestures.  If ``False``, start processing immediately.
    drop_threshold:
        Preamble coarse-trigger: minimum RSSI drop in dBm.
    wavelet:
        DWT wavelet name (default ``"haar"``).
    """

    def __init__(
        self,
        source: RSSISource,
        action_mapper: Optional[ActionMapper] = None,
        sample_rate: float = 10.0,
        buffer_duration: float = 3.0,
        magnitude_threshold: float = 3.0,
        variance_threshold: float = 1.0,
        preamble: bool = True,
        drop_threshold: float = 5.0,
        wavelet: str = "haar",
    ) -> None:
        self.source = source
        self.action_mapper = action_mapper
        self.sample_rate = sample_rate
        self._buf_len = max(4, int(sample_rate * buffer_duration))
        self._mag_threshold = magnitude_threshold
        self._var_threshold = variance_threshold
        self._use_preamble = preamble
        self._wavelet = wavelet

        self._preamble_detector = PreambleDetector(
            sample_rate=sample_rate,
            drop_threshold=drop_threshold,
            variance_threshold=variance_threshold,
            wavelet=wavelet,
        )
        self._calibration: Optional[PreambleCalibration] = None

        # Buffers per AP.
        self._ap_buffers: dict = {}

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def run(self, max_gestures: Optional[int] = None) -> Iterator[GestureResult]:
        """Stream gesture results from the RSSI source.

        This is the primary entry point.  It reads samples from
        :attr:`source`, accumulates them in a rolling buffer, and
        periodically runs the full pipeline to yield recognised gestures.

        Parameters
        ----------
        max_gestures:
            Stop after yielding this many gestures.  ``None`` runs
            indefinitely until the source is exhausted.

        Yields
        ------
        GestureResult
        """
        count = 0
        for snapshot in self.source.stream():
            for ap_id, rssi in snapshot.items():
                self._ap_buffers.setdefault(ap_id, []).append(rssi)
                # Keep buffer at most _buf_len samples.
                if len(self._ap_buffers[ap_id]) > self._buf_len:
                    self._ap_buffers[ap_id] = self._ap_buffers[ap_id][-self._buf_len:]

            # Only process once enough data is available.
            buf_ready = all(len(b) >= self._buf_len for b in self._ap_buffers.values())
            if not buf_ready:
                continue

            result = self.process()
            if result is not None:
                if self.action_mapper is not None:
                    self.action_mapper.dispatch(result)
                yield result
                count += 1
                if max_gestures is not None and count >= max_gestures:
                    return

    def process(self) -> Optional[GestureResult]:
        """Run one recognition cycle over the current AP buffers.

        This can be called manually when driving the pipeline from an
        external loop (e.g., when samples arrive asynchronously).

        Returns
        -------
        GestureResult or None
            ``None`` if preamble has not yet been confirmed or if no
            primitives could be extracted.
        """
        if not self._ap_buffers:
            return None

        # --- Preamble gating ---
        if self._use_preamble and self._calibration is None:
            # Feed the latest sample of the first AP to the preamble detector.
            first_ap = next(iter(self._ap_buffers))
            buf = self._ap_buffers[first_ap]
            calib = self._preamble_detector.feed(buf[-1])
            if calib is None:
                return None
            self._calibration = calib
            logger.info("Preamble confirmed: %s", calib)

        # Resolve effective magnitude threshold.
        mag_thresh = (
            self._calibration.magnitude_threshold
            if self._calibration is not None
            else self._mag_threshold
        )
        flip = self._calibration.flip_polarity if self._calibration else False

        # --- Primitive extraction per AP ---
        ap_primitives = {}
        total_samples = 0
        for ap_id, buf in self._ap_buffers.items():
            signal = np.array(buf, dtype=float)
            if flip:
                signal = -signal
            prims = extract_primitives(
                signal,
                sample_rate=self.sample_rate,
                magnitude_threshold=mag_thresh,
                variance_threshold=self._var_threshold,
                wavelet=self._wavelet,
            )
            ap_primitives[ap_id] = prims
            total_samples = max(total_samples, len(buf))

        # --- Multi-AP voting ---
        if len(ap_primitives) > 1:
            fused: List[Primitive] = majority_vote(
                ap_primitives,
                sample_rate=self.sample_rate,
                total_samples=total_samples,
            )
        else:
            fused = next(iter(ap_primitives.values())) if ap_primitives else []

        if not fused:
            return None

        # --- Gesture matching ---
        result = match_gesture(fused, sample_rate=self.sample_rate)
        return result

    def reset(self) -> None:
        """Reset the pipeline state (preamble + buffers)."""
        self._ap_buffers.clear()
        self._calibration = None
        self._preamble_detector.reset()

    # ------------------------------------------------------------------
    # Single-shot helpers
    # ------------------------------------------------------------------

    @classmethod
    def from_array(
        cls,
        rssi_array,
        sample_rate: float = 10.0,
        preamble: bool = False,
        **kwargs,
    ) -> "WiGest":
        """Create a WiGest instance pre-loaded with a numpy array.

        Convenience constructor for offline / batch processing.

        Parameters
        ----------
        rssi_array:
            1-D array of RSSI values, or a dict ``{ap_id: array}``.
        sample_rate:
            Acquisition rate in Hz.
        preamble:
            Whether to require preamble detection (default ``False`` for
            offline/batch use).
        **kwargs:
            Extra parameters forwarded to :class:`WiGest`.
        """
        from wigest.rssi import MockRSSISource

        source = MockRSSISource(rssi_array, sample_rate=sample_rate)
        return cls(source, sample_rate=sample_rate, preamble=preamble, **kwargs)
