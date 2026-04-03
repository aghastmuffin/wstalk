"""Wavelet denoising module.

Implements the DWT-based RSSI denoising described in WiGest §III-B:

1. Decompose the signal with a 1-D Discrete Wavelet Transform (DWT) using
   the **Haar** wavelet basis.
2. Apply **soft thresholding** to the detail coefficients at every level,
   removing Gaussian noise.
3. Reconstruct the denoised signal via the Inverse DWT (IDWT).

The threshold is chosen via the **SURE** (Stein's Unbiased Risk Estimate)
criterion (Donoho & Johnstone, 1994), which is well-suited to Gaussian
noise.  For each decomposition level the threshold is computed independently
so that the denoising adapts to the local noise power.

Paper reference
---------------
WiGest §III-B – "Signal Pre-processing (Wavelet De-noising)":
  "We use the Haar wavelet with a soft-threshold scheme where the threshold
   is selected using SURE."
"""

from typing import List, Optional, Tuple

import numpy as np
import pywt


# ---------------------------------------------------------------------------
# SURE threshold computation
# ---------------------------------------------------------------------------

def sure_threshold(coefficients: np.ndarray) -> float:
    """Compute the SURE soft-threshold for a 1-D array of coefficients.

    Stein's Unbiased Risk Estimate selects the threshold *t* that minimises
    the expected mean-squared-error when soft-thresholding under the
    assumption of unit-variance Gaussian noise.  The noise variance is
    estimated from the median absolute deviation (MAD) of the finest-scale
    detail coefficients (robust estimator).

    The algorithm (Donoho & Johnstone, 1994, §3):

    1. Sort ``|d|^2`` in ascending order.
    2. For each candidate threshold index *k* compute the SURE risk:
       ``risk(k) = (n - 2k + sum_{i<=k} d^2[i] + (n-k)*d^2[k]) / n``
    3. Return ``sqrt(d^2[k*])``, where *k\\** minimises risk.

    Parameters
    ----------
    coefficients:
        1-D array of wavelet detail coefficients (already noise-scaled,
        i.e. divided by the estimated noise standard deviation).

    Returns
    -------
    float
        SURE-optimal soft-threshold (in the same units as *coefficients*).

    Notes
    -----
    When all coefficients are zero the function returns 0 to avoid
    division-by-zero issues.
    """
    n = len(coefficients)
    if n == 0:
        return 0.0

    d_sq = np.sort(coefficients ** 2)
    cumsum = np.cumsum(d_sq)
    k = np.arange(n)
    # SURE risk estimate for each candidate threshold index.
    risk = (n - 2.0 * (k + 1) + cumsum + (n - k - 1) * d_sq) / n
    best_k = int(np.argmin(risk))
    return float(np.sqrt(d_sq[best_k]))


# ---------------------------------------------------------------------------
# Noise-level estimation
# ---------------------------------------------------------------------------

def estimate_noise_std(detail_coefficients: np.ndarray) -> float:
    """Estimate noise standard deviation via MAD of detail coefficients.

    Uses the robust MAD estimator (Donoho & Johnstone, 1994):
        σ = median(|d|) / 0.6745

    Parameters
    ----------
    detail_coefficients:
        Finest-level detail coefficients from a DWT.

    Returns
    -------
    float
        Estimated noise standard deviation.  Returns a small positive
        value (1e-10) if all coefficients are zero to avoid numerical
        issues downstream.
    """
    mad = float(np.median(np.abs(detail_coefficients)))
    sigma = mad / 0.6745
    return max(sigma, 1e-10)


# ---------------------------------------------------------------------------
# Per-level SURE thresholding
# ---------------------------------------------------------------------------

def _soft_threshold(coefficients: np.ndarray, threshold: float) -> np.ndarray:
    """Apply soft (shrinkage) thresholding element-wise.

    Soft thresholding shrinks all coefficients toward zero:
        sign(d) * max(|d| - t, 0)

    Parameters
    ----------
    coefficients:
        Array of wavelet coefficients.
    threshold:
        Non-negative threshold value.

    Returns
    -------
    np.ndarray
        Thresholded coefficients.
    """
    return np.sign(coefficients) * np.maximum(np.abs(coefficients) - threshold, 0.0)


# ---------------------------------------------------------------------------
# Public denoising function
# ---------------------------------------------------------------------------

def denoise(
    signal: np.ndarray,
    wavelet: str = "haar",
    level: Optional[int] = None,
    mode: str = "periodization",
    n_fine_bands: int = 2,
) -> Tuple[np.ndarray, List[np.ndarray]]:
    """Denoise a 1-D RSSI signal using DWT with SURE thresholding.

    Pipeline (mirrors WiGest §III-B):

    1. DWT decomposition with *wavelet* (default: Haar) up to *level*
       octaves.
    2. Estimate the noise standard deviation σ from the **finest** detail
       sub-band via the robust MAD estimator (Donoho & Johnstone, 1994).
    3. Apply SURE-optimal soft thresholding to the *n_fine_bands* finest
       detail sub-bands (high-frequency bands where Gaussian noise
       dominates).  Coarser bands that capture gesture-scale signal
       features are kept intact.
    4. Reconstruct via IDWT.

    The noise estimate uses the finest band because:
    * RSSI noise is broadband; the finest detail captures the most noise.
    * Coarser bands contain both signal energy (from gesture motion) and
      noise, inflating per-level MAD estimates and causing over-thresholding.

    Parameters
    ----------
    signal:
        1-D NumPy array of RSSI samples (dBm).  Must contain at least 4
        samples; shorter inputs are returned unchanged.
    wavelet:
        PyWavelets wavelet name.  ``"haar"`` is the default as specified
        in the paper.
    level:
        Decomposition depth.  ``None`` (default) uses the maximum level
        supported by *wavelet* and the signal length.
    mode:
        DWT signal extension mode (passed to PyWavelets).  ``"periodization"``
        keeps the coefficient lengths equal to ``ceil(N/2)``.
    n_fine_bands:
        Number of finest detail sub-bands to threshold (default: 2).
        Only the highest-frequency bands are denoised; coarser bands that
        encode gesture edges are preserved.

    Returns
    -------
    denoised : np.ndarray
        Reconstructed signal after noise removal.
    detail_bands : list of np.ndarray
        All detail coefficient arrays after (selective) thresholding,
        **finest level first** (index 0 = finest, index -1 = coarsest).
        Used by downstream modules (edge detection, spectrogram).

    Examples
    --------
    >>> import numpy as np
    >>> from wigest.denoising import denoise
    >>> rng = np.random.default_rng(0)
    >>> clean = np.sin(np.linspace(0, 4 * np.pi, 128)) * 10 - 65
    >>> noisy = clean + rng.normal(0, 2, 128)
    >>> denoised, details = denoise(noisy)
    >>> float(np.mean((denoised - clean) ** 2)) < float(np.mean((noisy - clean) ** 2))
    True
    """
    signal = np.asarray(signal, dtype=float)
    n = len(signal)
    # Require at least 4 samples for a meaningful single-level DWT.
    if n < 4:
        return signal.copy(), []

    w = pywt.Wavelet(wavelet)
    max_level = pywt.dwt_max_level(n, w.dec_len)
    if level is None:
        level = max_level
    else:
        level = min(level, max_level)

    # --- Decompose ---
    coeffs = pywt.wavedec(signal, w, mode=mode, level=level)
    # pywt layout: coeffs[0] = approximation (cA_level),
    #              coeffs[1] = coarsest detail (cD_level),
    #              …
    #              coeffs[-1] = finest detail (cD_1).

    # --- Estimate global noise σ from the finest detail band ---
    sigma = estimate_noise_std(coeffs[-1])

    # --- Selectively threshold only the finest n_fine_bands detail bands ---
    # These indices in pywt's coeffs list are: -1 (finest), -2, -3, ...
    # In terms of the list: coeffs[-1], coeffs[-2], ..., coeffs[-n_fine_bands].
    # We must not threshold coeffs[0] (approximation).
    bands_to_threshold = min(n_fine_bands, len(coeffs) - 1)

    # Work on a mutable copy of the list.
    new_coeffs: List[np.ndarray] = list(coeffs)
    for i in range(1, bands_to_threshold + 1):
        # Index from the end: finest = -1, second-finest = -2, …
        d = coeffs[-i]
        d_norm = d / sigma
        t = sure_threshold(np.abs(d_norm))
        new_coeffs[-i] = _soft_threshold(d_norm, t) * sigma

    # --- Reconstruct ---
    denoised = pywt.waverec(new_coeffs, w, mode=mode)
    # waverec may produce one extra sample for odd-length inputs.
    denoised = denoised[:n]

    # --- Build detail_bands list: finest first (index 0) → coarsest last ---
    # coeffs[-1] is finest; we want index 0 = finest.
    thresholded_details: List[np.ndarray] = [new_coeffs[-i] for i in range(1, len(new_coeffs))]

    return denoised, thresholded_details
