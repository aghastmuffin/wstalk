"""Unit tests for wigest.gestures.

Tests validate:
- encode_primitives: correct string encoding
- match_gesture: family detection for all defined families
- _count_repetitions: push_n / pull_n cycle counting
- Frequency (Hz) calculation
- Unknown pattern returns family='unknown'
- GestureResult fields are populated correctly
"""

import pytest

from wigest.gestures import (
    GestureResult,
    _count_repetitions,
    encode_primitives,
    match_gesture,
)
from wigest.primitives import Primitive


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _prim(kind: str, duration_s: float = 0.5, speed: str = "high", magnitude: str = "low"):
    """Build a minimal Primitive for testing."""
    n_samples = int(duration_s * 10)
    return Primitive(
        kind=kind,
        speed=speed,
        magnitude=magnitude,
        start_idx=0,
        end_idx=max(1, n_samples),
        delta_rssi=5.0 if kind == "+" else (-5.0 if kind == "-" else 0.0),
        duration_s=duration_s,
    )


def _prims(*kinds, duration_s=0.5):
    """Build a list of Primitives from kind characters."""
    return [_prim(k, duration_s=duration_s) for k in kinds]


# ---------------------------------------------------------------------------
# encode_primitives
# ---------------------------------------------------------------------------

class TestEncodePrimitives:
    def test_empty_list(self):
        assert encode_primitives([]) == ""

    def test_single_rising(self):
        assert encode_primitives([_prim("+")]) == "+"

    def test_single_falling(self):
        assert encode_primitives([_prim("-")]) == "-"

    def test_single_pause(self):
        assert encode_primitives([_prim("0")]) == "0"

    def test_push_pattern(self):
        assert encode_primitives(_prims("+", "-")) == "+-"

    def test_pull_pattern(self):
        assert encode_primitives(_prims("-", "+")) == "-+"

    def test_repeated_push(self):
        assert encode_primitives(_prims("+", "-", "+", "-")) == "+-+-"

    def test_left_swipe_pattern(self):
        assert encode_primitives(_prims("+", "-", "0", "-")) == "+-0-"


# ---------------------------------------------------------------------------
# match_gesture – family detection
# ---------------------------------------------------------------------------

class TestMatchGesture:
    def _match(self, *kinds, duration_s=0.5):
        prims = _prims(*kinds, duration_s=duration_s)
        return match_gesture(prims, sample_rate=10.0)

    def test_single_push(self):
        result = self._match("+", "-")
        assert result.family == "push"
        assert result.pattern == "+-"

    def test_single_pull(self):
        result = self._match("-", "+")
        assert result.family == "pull"
        assert result.pattern == "-+"

    def test_repeated_push(self):
        result = self._match("+", "-", "+", "-")
        assert result.family == "push_n"
        assert result.repetition_count == 2

    def test_repeated_pull(self):
        result = self._match("-", "+", "-", "+")
        assert result.family == "pull_n"
        assert result.repetition_count == 2

    def test_left_swipe(self):
        result = self._match("+", "-", "0", "-")
        assert result.family == "left"

    def test_right_swipe(self):
        result = self._match("-", "+", "0", "+")
        assert result.family == "right"

    def test_single_up(self):
        result = self._match("+")
        assert result.family == "up"

    def test_single_down(self):
        result = self._match("-")
        assert result.family == "down"

    def test_unknown_pattern(self):
        # A pattern that doesn't match any family.
        result = self._match("+", "0", "+", "0", "+")
        assert result.family == "unknown"
        assert result.pattern == "+0+0+"

    def test_empty_primitives(self):
        result = match_gesture([], sample_rate=10.0)
        assert result.family == "unknown"
        assert result.pattern == ""
        assert result.repetition_count == 1
        assert result.duration_s == 0.0


# ---------------------------------------------------------------------------
# Frequency and count
# ---------------------------------------------------------------------------

class TestFrequencyAndCount:
    def test_push_n_count(self):
        prims = _prims("+", "-", "+", "-", "+", "-")
        result = match_gesture(prims, sample_rate=10.0)
        assert result.family == "push_n"
        assert result.repetition_count == 3

    def test_frequency_is_count_over_duration(self):
        duration_s = 0.4
        prims = _prims("+", "-", "+", "-", duration_s=duration_s)
        result = match_gesture(prims, sample_rate=10.0)
        total_dur = sum(p.duration_s for p in prims)
        expected_freq = result.repetition_count / total_dur
        assert abs(result.frequency_hz - expected_freq) < 1e-9

    def test_frequency_zero_for_empty(self):
        result = match_gesture([], sample_rate=10.0)
        assert result.frequency_hz == 0.0

    def test_pull_n_count(self):
        prims = _prims("-", "+", "-", "+")
        result = match_gesture(prims, sample_rate=10.0)
        assert result.family == "pull_n"
        assert result.repetition_count == 2


# ---------------------------------------------------------------------------
# GestureResult fields
# ---------------------------------------------------------------------------

class TestGestureResultFields:
    def test_all_fields_present(self):
        prims = _prims("+", "-")
        result = match_gesture(prims, sample_rate=10.0)
        assert isinstance(result.family, str)
        assert isinstance(result.pattern, str)
        assert isinstance(result.repetition_count, int)
        assert isinstance(result.frequency_hz, float)
        assert isinstance(result.duration_s, float)
        assert isinstance(result.primitives, list)
        assert isinstance(result.extra, dict)

    def test_primitives_list_is_copy(self):
        """The result's primitives list should be independent of the input."""
        prims = _prims("+", "-")
        result = match_gesture(prims, sample_rate=10.0)
        prims.clear()
        assert len(result.primitives) == 2


# ---------------------------------------------------------------------------
# _count_repetitions
# ---------------------------------------------------------------------------

class TestCountRepetitions:
    def test_push_n(self):
        assert _count_repetitions("+-+-+-", "push_n") == 3

    def test_pull_n(self):
        assert _count_repetitions("-+-+", "pull_n") == 2

    def test_non_repeated_family(self):
        assert _count_repetitions("+-", "push") == 1

    def test_unknown_family(self):
        assert _count_repetitions("+0-", "unknown") == 1
