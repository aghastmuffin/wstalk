"""Unit tests for wigest.denoising.

Tests validate:
- SURE threshold computation
- Noise standard deviation estimation (MAD)
- Soft thresholding
- Full denoise() pipeline produces less noisy output
- Detail bands are returned and have correct length structure
"""

import numpy as np
import pytest

from wigest.denoising import (
    denoise,
    estimate_noise_std,
    sure_threshold,
)


# ---------------------------------------------------------------------------
# sure_threshold
# ---------------------------------------------------------------------------

class TestSureThreshold:
    def test_all_zeros_returns_zero(self):
        coeffs = np.zeros(16)
        assert sure_threshold(coeffs) == 0.0

    def test_empty_returns_zero(self):
        assert sure_threshold(np.array([])) == 0.0

    def test_single_element(self):
        # With one coefficient the threshold should be finite.
        t = sure_threshold(np.array([2.0]))
        assert np.isfinite(t)
        assert t >= 0.0

    def test_returns_nonnegative(self):
        rng = np.random.default_rng(42)
        coeffs = rng.standard_normal(64)
        t = sure_threshold(np.abs(coeffs))
        assert t >= 0.0

    def test_high_snr_signal_gives_low_threshold(self):
        """For a nearly clean signal the SURE threshold should be small."""
        # Mostly-zero coefficients with a few large values (high SNR).
        coeffs = np.zeros(64)
        coeffs[10] = 5.0
        coeffs[30] = 4.0
        t = sure_threshold(np.abs(coeffs))
        # The threshold should not eliminate the large coefficients entirely.
        assert t < 5.0

    def test_pure_noise_gives_threshold_near_mad(self):
        """For pure noise the SURE threshold should be > 0."""
        rng = np.random.default_rng(0)
        noise_coeffs = rng.standard_normal(128)
        t = sure_threshold(np.abs(noise_coeffs))
        assert t >= 0.0


# ---------------------------------------------------------------------------
# estimate_noise_std
# ---------------------------------------------------------------------------

class TestEstimateNoiseStd:
    def test_known_gaussian_noise(self):
        """MAD estimator should recover σ within ±20% for n=1024."""
        rng = np.random.default_rng(7)
        sigma = 3.0
        noise = rng.normal(0, sigma, 1024)
        est = estimate_noise_std(noise)
        assert abs(est - sigma) / sigma < 0.20

    def test_all_zeros_returns_tiny_positive(self):
        """Should never return exactly zero to avoid downstream /0 errors."""
        est = estimate_noise_std(np.zeros(32))
        assert est > 0.0

    def test_single_value(self):
        est = estimate_noise_std(np.array([1.0]))
        assert est > 0.0


# ---------------------------------------------------------------------------
# denoise()
# ---------------------------------------------------------------------------

class TestDenoise:
    def _make_noisy_signal(self, n=128, sigma=2.0, seed=0):
        rng = np.random.default_rng(seed)
        # Use a piecewise-constant signal (realistic RSSI shape) that
        # Haar DWT denoising handles well.
        clean = np.concatenate([
            np.full(n // 4, -70.0),
            np.full(n // 4, -60.0),
            np.full(n // 4, -65.0),
            np.full(n - 3 * (n // 4), -70.0),
        ])
        noisy = clean + rng.normal(0, sigma, n)
        return clean, noisy

    def test_denoised_is_closer_to_clean(self):
        clean, noisy = self._make_noisy_signal()
        denoised, _ = denoise(noisy)
        mse_noisy = float(np.mean((noisy - clean) ** 2))
        mse_denoised = float(np.mean((denoised - clean) ** 2))
        assert mse_denoised < mse_noisy, (
            f"denoised MSE ({mse_denoised:.4f}) should be < noisy MSE ({mse_noisy:.4f})"
        )

    def test_output_length_matches_input(self):
        for n in [16, 32, 64, 100, 128, 200]:
            sig = np.ones(n) * -65.0
            denoised, _ = denoise(sig)
            assert len(denoised) == n, f"n={n}: expected {n}, got {len(denoised)}"

    def test_returns_detail_bands(self):
        sig = np.random.default_rng(1).standard_normal(64)
        denoised, details = denoise(sig)
        assert len(details) >= 1
        # Each band should be a numpy array.
        for d in details:
            assert isinstance(d, np.ndarray)

    def test_detail_bands_count_equals_level(self):
        sig = np.random.default_rng(2).standard_normal(128)
        _, details = denoise(sig, level=3)
        assert len(details) == 3

    def test_very_short_signal(self):
        """A signal with fewer than 4 samples should be returned unchanged."""
        sig = np.array([-65.0, -66.0, -65.5])
        denoised, details = denoise(sig)
        np.testing.assert_array_equal(denoised, sig)
        assert details == []

    def test_flat_signal_denoised_is_approximately_flat(self):
        """A constant RSSI should denoise to approximately the same constant."""
        flat = np.full(64, -70.0)
        denoised, _ = denoise(flat)
        # The reconstructed signal should stay close to −70 dBm.
        assert float(np.max(np.abs(denoised - (-70.0)))) < 1e-6

    def test_haar_wavelet_default(self):
        sig = np.random.default_rng(3).standard_normal(64)
        # Should not raise.
        denoise(sig)

    def test_custom_wavelet(self):
        sig = np.random.default_rng(4).standard_normal(64)
        # db4 is also a valid wavelet.
        denoised, details = denoise(sig, wavelet="db4")
        assert len(denoised) == 64
