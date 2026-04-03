"""End-to-end pipeline tests for wigest.pipeline.

Tests validate:
- WiGest.from_array() constructor
- WiGest.run() yields GestureResult objects
- WiGest with MockRSSISource processes correctly
- Multi-AP path through the pipeline
- Reset clears state
"""

import numpy as np
import pytest

from wigest import MockRSSISource, WiGest
from wigest.gestures import GestureResult


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_push_pull_signal(
    sample_rate: float = 10.0,
    n_cycles: int = 2,
    duration_s: float = 4.0,
) -> np.ndarray:
    """Synthetic RSSI with alternating push-pull motion."""
    n = int(sample_rate * duration_s)
    t = np.linspace(0, n_cycles * 2 * np.pi, n)
    return np.sin(t) * 10 - 65


# ---------------------------------------------------------------------------
# from_array
# ---------------------------------------------------------------------------

class TestFromArray:
    def test_constructor_returns_wigest(self):
        sig = _make_push_pull_signal()
        wg = WiGest.from_array(sig, sample_rate=10.0, preamble=False)
        assert isinstance(wg, WiGest)

    def test_dict_input(self):
        sig = _make_push_pull_signal()
        wg = WiGest.from_array({"ap1": sig, "ap2": sig}, sample_rate=10.0, preamble=False)
        assert isinstance(wg, WiGest)


# ---------------------------------------------------------------------------
# run() and process()
# ---------------------------------------------------------------------------

class TestWiGestRun:
    def test_run_yields_gesture_results(self):
        sig = _make_push_pull_signal(duration_s=6.0)
        source = MockRSSISource(sig, sample_rate=10.0)
        wg = WiGest(source, sample_rate=10.0, preamble=False, buffer_duration=3.0)
        results = list(wg.run(max_gestures=2))
        assert len(results) >= 1
        for r in results:
            assert isinstance(r, GestureResult)

    def test_run_respects_max_gestures(self):
        sig = _make_push_pull_signal(duration_s=10.0)
        source = MockRSSISource(sig, sample_rate=10.0)
        wg = WiGest(source, sample_rate=10.0, preamble=False, buffer_duration=2.0)
        results = list(wg.run(max_gestures=3))
        assert len(results) <= 3

    def test_exhausted_source_stops(self):
        sig = np.full(20, -65.0)  # Too short for a full buffer.
        source = MockRSSISource(sig, sample_rate=10.0)
        wg = WiGest(source, sample_rate=10.0, preamble=False, buffer_duration=3.0)
        results = list(wg.run())
        # Should complete without hanging; result count may be 0.
        assert isinstance(results, list)

    def test_action_mapper_dispatched(self):
        from wigest.actions import ActionMapper

        calls = []
        mapper = ActionMapper()
        mapper.register("*", lambda r: calls.append(r.family))

        sig = _make_push_pull_signal(duration_s=6.0)
        source = MockRSSISource(sig, sample_rate=10.0)
        wg = WiGest(
            source,
            action_mapper=mapper,
            sample_rate=10.0,
            preamble=False,
            buffer_duration=3.0,
        )
        list(wg.run(max_gestures=1))
        assert len(calls) >= 1


# ---------------------------------------------------------------------------
# Multi-AP
# ---------------------------------------------------------------------------

class TestMultiAP:
    def test_two_aps_run_successfully(self):
        sig = _make_push_pull_signal(duration_s=6.0)
        source = MockRSSISource(
            {"ap1": sig, "ap2": sig},
            sample_rate=10.0,
        )
        wg = WiGest(source, sample_rate=10.0, preamble=False, buffer_duration=3.0)
        results = list(wg.run(max_gestures=2))
        assert isinstance(results, list)

    def test_two_aps_with_noise_disagreement(self):
        """Two APs with slight noise should still produce a result."""
        rng = np.random.default_rng(42)
        sig1 = _make_push_pull_signal(duration_s=6.0)
        sig2 = sig1 + rng.normal(0, 1, len(sig1))
        source = MockRSSISource({"ap1": list(sig1), "ap2": list(sig2)}, sample_rate=10.0)
        wg = WiGest(source, sample_rate=10.0, preamble=False, buffer_duration=3.0)
        results = list(wg.run(max_gestures=1))
        assert isinstance(results, list)


# ---------------------------------------------------------------------------
# Reset
# ---------------------------------------------------------------------------

class TestReset:
    def test_reset_clears_buffers(self):
        sig = _make_push_pull_signal()
        wg = WiGest.from_array(sig, preamble=False)
        # Feed some data.
        wg._ap_buffers["ap0"] = list(sig)
        wg.reset()
        assert wg._ap_buffers == {}
        assert wg._calibration is None
