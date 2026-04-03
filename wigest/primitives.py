"""Edge extraction and primitive detection module.

Implements WiGest §III-C – "Primitive Detection".

The paper identifies three *primitives*:

Rising edge (``'+'``)
    RSSI increases — hand moving toward the AP.
    Detected as a **local minimum** in the wavelet detail coefficients
    (the signal first drops then rises at the edge).

Falling edge (``'-'``)
    RSSI decreases — hand moving away from the AP.
    Detected as a **local maximum** in the wavelet detail coefficients.

Pause (``'0'``)
    RSSI stays approximately constant (variance < threshold) for at
    least 0.5 s (configurable via *pause_min_duration*).

Each primitive carries three attributes:

speed
    ``'high'``   edge duration < 0.75 s
    ``'medium'`` 0.75 s ≤ duration < 1.50 s
    ``'low'``    duration ≥ 1.50 s

magnitude
    ``'high'``  |ΔRSSI| ≥ *magnitude_threshold*
    ``'low'``   |ΔRSSI| < *magnitude_threshold*

direction
    ``'rising'``, ``'falling'``, or ``'pause'``

Paper reference
---------------
WiGest §III-C – "Primitive Detection and Classification":
  "We use the wavelet detail coefficients to locate edges.  A local minimum
   in the detail coefficients corresponds to a rising edge and a local
   maximum corresponds to a falling edge."
"""

from dataclasses import dataclass, field
from typing import List, Optional, Tuple

import numpy as np
from scipy.signal import find_peaks

from wigest.denoising import denoise


# ---------------------------------------------------------------------------
# Data class
# ---------------------------------------------------------------------------

@dataclass
class Primitive:
    """A single WiGest gesture primitive.

    Attributes
    ----------
    kind:
        ``'+'`` (rising), ``'-'`` (falling), or ``'0'`` (pause).
    speed:
        ``'high'``, ``'medium'``, or ``'low'``.
    magnitude:
        ``'high'`` or ``'low'``.
    start_idx:
        Sample index where the primitive begins.
    end_idx:
        Sample index where the primitive ends (exclusive).
    delta_rssi:
        Signed RSSI change (dBm) over the primitive.
    duration_s:
        Duration in seconds.
    """

    kind: str
    speed: str
    magnitude: str
    start_idx: int
    end_idx: int
    delta_rssi: float
    duration_s: float
    # Extra metadata for downstream use.
    extra: dict = field(default_factory=dict)

    def __str__(self) -> str:  # pragma: no cover
        return (
            f"Primitive({self.kind!r}, speed={self.speed!r}, "
            f"mag={self.magnitude!r}, Δ={self.delta_rssi:+.1f} dBm, "
            f"dur={self.duration_s:.2f}s)"
        )


# ---------------------------------------------------------------------------
# Speed classification
# ---------------------------------------------------------------------------

# Thresholds from WiGest §III-C (seconds).
_SPEED_HIGH_MAX = 0.75
_SPEED_MED_MAX = 1.50


def _classify_speed(duration_s: float) -> str:
    """Return speed class for an edge of given *duration_s*.

    Thresholds (WiGest §III-C):
    * high   < 0.75 s
    * medium 0.75 – 1.50 s
    * low    ≥ 1.50 s
    """
    if duration_s < _SPEED_HIGH_MAX:
        return "high"
    if duration_s < _SPEED_MED_MAX:
        return "medium"
    return "low"


# ---------------------------------------------------------------------------
# Spectrogram / frequency-level selection
# ---------------------------------------------------------------------------

def _select_detail_level(detail_bands: List[np.ndarray]) -> int:
    """Choose the decomposition level whose energy best captures motion.

    The paper selects the frequency sub-band whose detail coefficients
    show the highest energy (local maxima count) — corresponding to the
    dominant motion frequency.  We use total squared energy as a proxy,
    but require at least 4 coefficients for reliable peak detection.

    Parameters
    ----------
    detail_bands:
        List of detail coefficient arrays, finest level first
        (as returned by :func:`~wigest.denoising.denoise`).

    Returns
    -------
    int
        Index into *detail_bands* of the best level (0 = finest).
    """
    if not detail_bands:
        return 0
    # Only consider bands with enough elements for peak detection.
    min_len = 4
    candidate_energies = [
        float(np.sum(d ** 2)) if len(d) >= min_len else -1.0
        for d in detail_bands
    ]
    if max(candidate_energies) <= 0:
        # Fall back to the longest band.
        return int(np.argmax([len(d) for d in detail_bands]))
    return int(np.argmax(candidate_energies))


# ---------------------------------------------------------------------------
# Peak / trough detection helpers
# ---------------------------------------------------------------------------

def _find_peaks_and_troughs(
    coeffs: np.ndarray,
    min_prominence: float = 0.5,
) -> Tuple[np.ndarray, np.ndarray]:
    """Locate local maxima (peaks) and minima (troughs) in *coeffs*.

    Parameters
    ----------
    coeffs:
        1-D array of detail coefficients.
    min_prominence:
        Minimum prominence required for a peak/trough to be retained.
        Smaller values include more events; larger values filter noise.

    Returns
    -------
    peaks : np.ndarray
        Indices of local maxima (falling-edge candidates).
    troughs : np.ndarray
        Indices of local minima (rising-edge candidates).
    """
    peaks, _ = find_peaks(coeffs, prominence=min_prominence)
    troughs, _ = find_peaks(-coeffs, prominence=min_prominence)
    return peaks, troughs


# ---------------------------------------------------------------------------
# Pause detection
# ---------------------------------------------------------------------------

def _find_pauses(
    signal: np.ndarray,
    sample_rate: float,
    variance_threshold: float,
    pause_min_duration: float,
) -> List[Tuple[int, int]]:
    """Identify segments where the RSSI variance is below *variance_threshold*.

    A pause is a contiguous run of samples whose local variance (computed
    over a window of size ``int(sample_rate * pause_min_duration)``)
    remains below *variance_threshold*.

    Parameters
    ----------
    signal:
        Denoised RSSI signal.
    sample_rate:
        Acquisition rate in Hz.
    variance_threshold:
        RSSI variance (dBm²) below which a segment is considered a pause.
    pause_min_duration:
        Minimum pause duration in seconds.

    Returns
    -------
    list of (start, end) index tuples (exclusive end).
    """
    n = len(signal)
    window = max(2, int(sample_rate * pause_min_duration))
    pauses: List[Tuple[int, int]] = []
    i = 0
    while i + window <= n:
        seg = signal[i: i + window]
        if float(np.var(seg)) < variance_threshold:
            # Extend the pause as far as possible.
            j = i + window
            while j < n and float(np.var(signal[i:j + 1])) < variance_threshold:
                j += 1
            pauses.append((i, j))
            i = j
        else:
            i += 1
    return pauses


# ---------------------------------------------------------------------------
# Main public function
# ---------------------------------------------------------------------------

def extract_primitives(
    signal: np.ndarray,
    sample_rate: float = 10.0,
    magnitude_threshold: float = 3.0,
    variance_threshold: float = 1.0,
    pause_min_duration: float = 0.5,
    peak_prominence: float = 0.5,
    wavelet: str = "haar",
    dwt_level: Optional[int] = None,
) -> List[Primitive]:
    """Extract WiGest primitives from a 1-D RSSI signal.

    This function implements the full primitive-detection pipeline from
    WiGest §III-C:

    1. **Denoise** *signal* via DWT/Haar + SURE thresholding.
    2. **Compute rolling variance** over a window of *pause_min_duration*
       seconds to distinguish *active* (gesture) segments from *quiet*
       (pause) segments.
    3. **Classify active segments**: compute the net RSSI change (end minus
       start) to determine whether the segment is a rising (``'+'``) or
       falling (``'-'``) edge.  For segments with negligible net change,
       the majority gradient direction is used.
    4. **Classify quiet segments** of sufficient duration as pauses (``'0'``).
    5. **Assign speed and magnitude** to each primitive.

    This approach robustly handles the staircase-shaped denoised signal
    produced by the Haar wavelet by assessing activity over a multi-sample
    window rather than sample-by-sample.

    Parameters
    ----------
    signal:
        1-D array of RSSI samples in dBm.
    sample_rate:
        Acquisition rate in Hz (used for duration calculation).
    magnitude_threshold:
        RSSI change (dBm) separating ``'high'`` from ``'low'`` magnitude.
        Should be derived from the preamble; pass the preamble-calibrated
        value via :class:`~wigest.preamble.PreambleDetector`.
    variance_threshold:
        RSSI variance (dBm²) above which a sample is considered *active*
        (part of a gesture), below which it is considered *quiet* (pause).
    pause_min_duration:
        Minimum duration (seconds) for a quiet segment to be labelled a
        pause.  Shorter quiet gaps between edges are silently discarded.
    peak_prominence:
        Reserved; not currently used in the active-region approach but
        kept for API compatibility with the preamble module's internal
        :func:`_find_peaks_and_troughs` helper.
    wavelet:
        DWT wavelet name used for the denoising step (default ``"haar"``).
    dwt_level:
        Decomposition depth passed to :func:`~wigest.denoising.denoise`.

    Returns
    -------
    list of :class:`Primitive`
        Sorted by *start_idx*, non-overlapping.  Empty list if fewer than
        4 samples.
    """
    signal = np.asarray(signal, dtype=float)
    n = len(signal)
    if n < 4:
        return []

    # Step 1 – Denoise.
    denoised, _ = denoise(signal, wavelet=wavelet, level=dwt_level)

    # Step 2 – Rolling variance to separate active from quiet.
    # Window size = pause_min_duration in samples (at least 4).
    win = max(4, int(sample_rate * pause_min_duration))
    half = win // 2
    padded = np.pad(denoised, half, mode="edge")
    variances = np.array(
        [float(np.var(padded[i: i + win])) for i in range(n)]
    )
    active = variances > variance_threshold  # boolean mask

    # Steps 3 & 4 – Segment the signal and classify each segment.
    primitives: List[Primitive] = []
    i = 0
    while i < n:
        if active[i]:
            # --- Active segment: rising or falling edge ---
            j = i + 1
            while j < n and active[j]:
                j += 1
            seg = denoised[i:j]
            delta = float(seg[-1] - seg[0])

            # Determine edge direction from net RSSI change.
            if abs(delta) >= 0.5:
                kind = "+" if delta > 0 else "-"
            else:
                # Near-zero net change: use majority gradient direction.
                grad = np.gradient(seg)
                kind = "+" if np.sum(grad > 0) >= np.sum(grad < 0) else "-"

            duration_s = (j - i) / sample_rate
            primitives.append(
                Primitive(
                    kind=kind,
                    speed=_classify_speed(duration_s),
                    magnitude="high" if abs(delta) >= magnitude_threshold else "low",
                    start_idx=i,
                    end_idx=j,
                    delta_rssi=delta,
                    duration_s=duration_s,
                )
            )
            i = j

        else:
            # --- Quiet segment: potential pause ---
            j = i + 1
            while j < n and not active[j]:
                j += 1
            duration_s = (j - i) / sample_rate
            if duration_s >= pause_min_duration:
                primitives.append(
                    Primitive(
                        kind="0",
                        speed=_classify_speed(duration_s),
                        magnitude="low",
                        start_idx=i,
                        end_idx=j,
                        delta_rssi=0.0,
                        duration_s=duration_s,
                    )
                )
            i = j

    return primitives
