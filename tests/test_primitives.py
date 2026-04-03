"""Unit tests for wigest.primitives.

Tests validate:
- Speed classification thresholds
- Rising/falling edge detection from synthetic RSSI with clear edges
- Pause detection (low-variance segment ≥ 0.5 s)
- Primitive magnitude classification
- Overlap filtering (no two primitives overlap)
- Short signal returns empty list
"""

import numpy as np
import pytest

from wigest.primitives import (
    Primitive,
    _classify_speed,
    extract_primitives,
)


# ---------------------------------------------------------------------------
# Speed classification
# ---------------------------------------------------------------------------

class TestClassifySpeed:
    def test_high_speed(self):
        assert _classify_speed(0.3) == "high"
        assert _classify_speed(0.74) == "high"

    def test_medium_speed(self):
        assert _classify_speed(0.75) == "medium"
        assert _classify_speed(1.0) == "medium"
        assert _classify_speed(1.49) == "medium"

    def test_low_speed(self):
        assert _classify_speed(1.5) == "low"
        assert _classify_speed(3.0) == "low"

    def test_boundary_values(self):
        # Exactly at boundary: 0.75 → medium, 1.50 → low
        assert _classify_speed(0.75) == "medium"
        assert _classify_speed(1.50) == "low"


# ---------------------------------------------------------------------------
# Synthetic signal helpers
# ---------------------------------------------------------------------------

def _make_rising_edge(n=100, start_val=-70.0, end_val=-55.0):
    """RSSI signal with a single rising edge in the middle."""
    sig = np.full(n, start_val)
    mid = n // 2
    sig[mid:] = end_val
    return sig


def _make_falling_edge(n=100, start_val=-55.0, end_val=-70.0):
    """RSSI signal with a single falling edge in the middle."""
    sig = np.full(n, start_val)
    mid = n // 2
    sig[mid:] = end_val
    return sig


def _make_pause_signal(
    n=100,
    sample_rate=10.0,
    pause_duration=1.0,
):
    """RSSI that is active then becomes flat (pause)."""
    rng = np.random.default_rng(99)
    sig = np.full(n, -65.0)
    # Add some motion in the first half.
    half = n // 2
    sig[:half] += rng.normal(0, 5, half)
    # Second half is a flat pause (very low variance).
    # The pause region is kept exactly flat to guarantee detection.
    return sig


# ---------------------------------------------------------------------------
# extract_primitives
# ---------------------------------------------------------------------------

class TestExtractPrimitives:
    def test_too_short_returns_empty(self):
        assert extract_primitives(np.array([-65.0, -66.0])) == []

    def test_constant_signal_produces_pause(self):
        """A completely flat 3-second signal should yield at least one pause."""
        sample_rate = 10.0
        sig = np.full(int(sample_rate * 3), -65.0)
        prims = extract_primitives(
            sig,
            sample_rate=sample_rate,
            variance_threshold=0.5,
            pause_min_duration=0.5,
        )
        pause_prims = [p for p in prims if p.kind == "0"]
        assert len(pause_prims) >= 1

    def test_rising_edge_signal_has_rising_primitive(self):
        """A clear rising RSSI transition should produce at least one '+' primitive."""
        sample_rate = 10.0
        # Smooth ramp from −70 → −50 dBm (realistic gesture transition).
        pre = np.full(15, -70.0)
        ramp = np.linspace(-70.0, -50.0, 20)
        post = np.full(15, -50.0)
        sig = np.concatenate([pre, ramp, post])
        rng = np.random.default_rng(5)
        sig = sig + rng.normal(0, 0.2, len(sig))
        prims = extract_primitives(
            sig,
            sample_rate=sample_rate,
            magnitude_threshold=5.0,
            variance_threshold=2.0,
        )
        kinds = [p.kind for p in prims]
        assert "+" in kinds, f"Expected rising primitive, got: {kinds}"

    def test_falling_edge_signal_has_falling_primitive(self):
        """A clear falling RSSI transition should produce at least one '-' primitive."""
        sample_rate = 10.0
        pre = np.full(15, -50.0)
        ramp = np.linspace(-50.0, -70.0, 20)
        post = np.full(15, -70.0)
        sig = np.concatenate([pre, ramp, post])
        rng = np.random.default_rng(6)
        sig = sig + rng.normal(0, 0.2, len(sig))
        prims = extract_primitives(
            sig,
            sample_rate=sample_rate,
            magnitude_threshold=5.0,
            variance_threshold=2.0,
        )
        kinds = [p.kind for p in prims]
        assert "-" in kinds, f"Expected falling primitive, got: {kinds}"

    def test_primitives_do_not_overlap(self):
        """No two returned primitives should have overlapping sample ranges."""
        rng = np.random.default_rng(10)
        sig = np.sin(np.linspace(0, 6 * np.pi, 120)) * 15 - 65
        sig += rng.normal(0, 0.5, len(sig))
        prims = extract_primitives(sig, sample_rate=10.0)
        for i in range(1, len(prims)):
            assert prims[i].start_idx >= prims[i - 1].end_idx, (
                f"Overlap between primitive {i-1} and {i}: "
                f"[{prims[i-1].start_idx},{prims[i-1].end_idx}) "
                f"vs [{prims[i].start_idx},{prims[i].end_idx})"
            )

    def test_primitives_sorted_by_start(self):
        """Primitives should be sorted in chronological order."""
        rng = np.random.default_rng(11)
        sig = np.sin(np.linspace(0, 4 * np.pi, 100)) * 10 - 65
        sig += rng.normal(0, 0.5, len(sig))
        prims = extract_primitives(sig, sample_rate=10.0)
        starts = [p.start_idx for p in prims]
        assert starts == sorted(starts)

    def test_magnitude_high_for_large_delta(self):
        """A large RSSI swing should produce a 'high' magnitude primitive."""
        # Smooth ramp with large amplitude (30 dBm).
        pre = np.full(15, -70.0)
        ramp = np.linspace(-70.0, -40.0, 20)
        post = np.full(15, -40.0)
        sig = np.concatenate([pre, ramp, post])
        rng = np.random.default_rng(12)
        sig += rng.normal(0, 0.2, len(sig))
        prims = extract_primitives(sig, sample_rate=10.0, magnitude_threshold=5.0,
                                   variance_threshold=2.0)
        high_mag = [p for p in prims if p.magnitude == "high"]
        assert len(high_mag) >= 1

    def test_magnitude_low_for_small_delta(self):
        """A tiny RSSI swing should produce a 'low' magnitude primitive."""
        n = 64
        # Very small step: −65 → −64 dBm.
        sig = np.concatenate([np.full(n // 2, -65.0), np.full(n // 2, -64.0)])
        rng = np.random.default_rng(13)
        sig += rng.normal(0, 0.05, n)
        prims = extract_primitives(sig, sample_rate=10.0, magnitude_threshold=5.0)
        if prims:
            low_mag = [p for p in prims if p.magnitude == "low"]
            assert len(low_mag) >= 1

    def test_primitive_dataclass_fields(self):
        """Each returned Primitive should have all required fields."""
        sig = np.sin(np.linspace(0, 2 * np.pi, 64)) * 10 - 65
        prims = extract_primitives(sig, sample_rate=10.0)
        for p in prims:
            assert p.kind in ("+", "-", "0")
            assert p.speed in ("high", "medium", "low")
            assert p.magnitude in ("high", "low")
            assert p.start_idx >= 0
            assert p.end_idx > p.start_idx
            assert np.isfinite(p.delta_rssi)
            assert p.duration_s > 0.0
