"""Unit tests for wigest.voting.

Tests validate:
- Empty input returns empty list
- Single AP passes through unchanged
- Two APs agree → that kind is returned
- Majority vote with tie-breaking logic
- Kind arrays are correctly aligned on sample grid
- Consensus attribute (speed, magnitude) derivation
"""

import pytest

from wigest.primitives import Primitive
from wigest.voting import majority_vote


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _prim(kind: str, start: int, end: int, speed="medium", magnitude="low", delta=0.0):
    return Primitive(
        kind=kind,
        speed=speed,
        magnitude=magnitude,
        start_idx=start,
        end_idx=end,
        delta_rssi=delta,
        duration_s=(end - start) / 10.0,
    )


# ---------------------------------------------------------------------------
# Basic behaviour
# ---------------------------------------------------------------------------

class TestMajorityVoteBasic:
    def test_empty_input_returns_empty(self):
        assert majority_vote({}) == []

    def test_no_primitives_returns_empty(self):
        result = majority_vote({"ap1": [], "ap2": []}, total_samples=50)
        assert result == []

    def test_single_ap_passes_through(self):
        prims = [_prim("+", 0, 10), _prim("-", 10, 20)]
        result = majority_vote({"ap1": prims}, total_samples=20)
        kinds = [p.kind for p in result]
        assert "+" in kinds
        assert "-" in kinds

    def test_total_samples_inferred(self):
        prims = [_prim("+", 0, 15)]
        result = majority_vote({"ap1": prims})
        assert len(result) >= 1


# ---------------------------------------------------------------------------
# Multi-AP voting
# ---------------------------------------------------------------------------

class TestMajorityVoteMultiAP:
    def test_two_aps_agree(self):
        """Both APs report '+' over the same window → result is '+'."""
        prims_a = [_prim("+", 0, 20)]
        prims_b = [_prim("+", 0, 20)]
        result = majority_vote({"a": prims_a, "b": prims_b}, total_samples=20)
        kinds = [p.kind for p in result]
        assert kinds == ["+"]

    def test_two_vs_one_majority(self):
        """Two APs vote '+', one votes '-' → '+' wins."""
        prims_a = [_prim("+", 0, 10)]
        prims_b = [_prim("+", 0, 10)]
        prims_c = [_prim("-", 0, 10)]
        result = majority_vote(
            {"a": prims_a, "b": prims_b, "c": prims_c},
            total_samples=10,
        )
        kinds = [p.kind for p in result]
        assert "+" in kinds

    def test_tie_breaking_prefers_motion(self):
        """Tie between '+' and '0' → '+' wins (motion preferred over pause)."""
        prims_a = [_prim("+", 0, 10)]
        prims_b = [_prim("0", 0, 10)]
        result = majority_vote({"a": prims_a, "b": prims_b}, total_samples=10)
        kinds = [p.kind for p in result]
        # Under tie-breaking rules, non-pause kind should win.
        assert "+" in kinds

    def test_non_overlapping_aps_merge(self):
        """APs covering disjoint time ranges should both appear in output."""
        prims_a = [_prim("+", 0, 10)]
        prims_b = [_prim("-", 10, 20)]
        result = majority_vote({"a": prims_a, "b": prims_b}, total_samples=20)
        kinds = {p.kind for p in result}
        assert "+" in kinds
        assert "-" in kinds


# ---------------------------------------------------------------------------
# Run-length encoding
# ---------------------------------------------------------------------------

class TestRunLengthEncoding:
    def test_consecutive_same_kind_merged(self):
        """Two adjacent '+' regions should merge into one primitive."""
        prims_a = [_prim("+", 0, 10)]
        prims_b = [_prim("+", 0, 10)]
        result = majority_vote({"a": prims_a, "b": prims_b}, total_samples=10)
        plus_prims = [p for p in result if p.kind == "+"]
        assert len(plus_prims) == 1

    def test_different_kinds_not_merged(self):
        """'+' then '-' must remain two separate primitives."""
        prims = [_prim("+", 0, 10), _prim("-", 10, 20)]
        result = majority_vote({"a": prims}, total_samples=20)
        kinds = [p.kind for p in result]
        assert kinds.count("+") == 1
        assert kinds.count("-") == 1


# ---------------------------------------------------------------------------
# Consensus attributes
# ---------------------------------------------------------------------------

class TestConsensusAttributes:
    def test_speed_majority(self):
        """Speed should reflect the consensus across APs."""
        prims_a = [_prim("+", 0, 10, speed="high")]
        prims_b = [_prim("+", 0, 10, speed="high")]
        prims_c = [_prim("+", 0, 10, speed="low")]
        result = majority_vote(
            {"a": prims_a, "b": prims_b, "c": prims_c},
            total_samples=10,
        )
        plus_prims = [p for p in result if p.kind == "+"]
        assert plus_prims[0].speed == "high"

    def test_magnitude_majority(self):
        prims_a = [_prim("+", 0, 10, magnitude="high")]
        prims_b = [_prim("+", 0, 10, magnitude="high")]
        prims_c = [_prim("+", 0, 10, magnitude="low")]
        result = majority_vote(
            {"a": prims_a, "b": prims_b, "c": prims_c},
            total_samples=10,
        )
        plus_prims = [p for p in result if p.kind == "+"]
        assert plus_prims[0].magnitude == "high"

    def test_primitive_duration_correct(self):
        prims = [_prim("+", 0, 20)]
        result = majority_vote({"a": prims}, total_samples=20, sample_rate=10.0)
        plus_prims = [p for p in result if p.kind == "+"]
        assert abs(plus_prims[0].duration_s - 2.0) < 1e-9
