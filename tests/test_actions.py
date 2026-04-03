"""Unit tests for wigest.actions.

Tests validate:
- ActionMapper registers callbacks correctly
- on() decorator and register() both work
- dispatch() calls matching callbacks
- dispatch() respects min_count / max_count filters
- dispatch() respects speed / magnitude filters
- Wildcard family '*' matches any gesture
- Default callback called when no binding matches
- describe() returns binding metadata
- create_media_player_mapper() returns a configured mapper
"""

import pytest

from wigest.actions import ActionMapper, create_media_player_mapper
from wigest.gestures import GestureResult
from wigest.primitives import Primitive


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _result(family="push", pattern="+-", rep=1, freq=1.0, dur=0.5, prims=None):
    return GestureResult(
        family=family,
        pattern=pattern,
        repetition_count=rep,
        frequency_hz=freq,
        duration_s=dur,
        primitives=prims or [],
    )


def _prim(kind="+", speed="high", magnitude="low"):
    return Primitive(
        kind=kind,
        speed=speed,
        magnitude=magnitude,
        start_idx=0,
        end_idx=5,
        delta_rssi=5.0,
        duration_s=0.5,
    )


# ---------------------------------------------------------------------------
# Basic registration and dispatch
# ---------------------------------------------------------------------------

class TestActionMapperBasic:
    def test_decorator_registration(self):
        mapper = ActionMapper()
        called = []

        @mapper.on("push")
        def handler(r):
            called.append(r)

        mapper.dispatch(_result("push"))
        assert len(called) == 1

    def test_register_method(self):
        mapper = ActionMapper()
        called = []
        mapper.register("pull", lambda r: called.append(r))
        mapper.dispatch(_result("pull"))
        assert len(called) == 1

    def test_no_matching_binding_calls_nothing(self):
        mapper = ActionMapper()
        called = []
        mapper.register("push", lambda r: called.append(r))
        # Dispatch a 'pull' – no binding matches.
        count = mapper.dispatch(_result("pull"))
        assert count == 0
        assert called == []

    def test_returns_count_of_called_handlers(self):
        mapper = ActionMapper()
        mapper.register("push", lambda r: None)
        mapper.register("push", lambda r: None)
        count = mapper.dispatch(_result("push"))
        assert count == 2

    def test_all_matching_bindings_called(self):
        mapper = ActionMapper()
        calls = []
        mapper.register("push", lambda r: calls.append("first"))
        mapper.register("push", lambda r: calls.append("second"))
        mapper.dispatch(_result("push"))
        assert calls == ["first", "second"]

    def test_clear_removes_all_bindings(self):
        mapper = ActionMapper()
        called = []
        mapper.register("push", lambda r: called.append(r))
        mapper.clear()
        mapper.dispatch(_result("push"))
        assert called == []


# ---------------------------------------------------------------------------
# Default callback
# ---------------------------------------------------------------------------

class TestDefaultCallback:
    def test_default_called_when_no_match(self):
        default_calls = []
        mapper = ActionMapper(default_callback=lambda r: default_calls.append(r))
        mapper.dispatch(_result("unknown_gesture"))
        assert len(default_calls) == 1

    def test_default_not_called_when_match_exists(self):
        default_calls = []
        mapper = ActionMapper(default_callback=lambda r: default_calls.append(r))
        mapper.register("push", lambda r: None)
        mapper.dispatch(_result("push"))
        assert len(default_calls) == 0


# ---------------------------------------------------------------------------
# Wildcard family
# ---------------------------------------------------------------------------

class TestWildcardFamily:
    def test_wildcard_matches_any_family(self):
        mapper = ActionMapper()
        calls = []
        mapper.register("*", lambda r: calls.append(r.family))
        for family in ("push", "pull", "left", "right", "unknown"):
            mapper.dispatch(_result(family))
        assert calls == ["push", "pull", "left", "right", "unknown"]


# ---------------------------------------------------------------------------
# Attribute filters
# ---------------------------------------------------------------------------

class TestAttributeFilters:
    def test_min_count_filter(self):
        mapper = ActionMapper()
        called = []
        mapper.register("push_n", lambda r: called.append(r), min_count=3)
        # rep=2 → should NOT fire.
        mapper.dispatch(_result("push_n", rep=2))
        assert called == []
        # rep=3 → should fire.
        mapper.dispatch(_result("push_n", rep=3))
        assert len(called) == 1

    def test_max_count_filter(self):
        mapper = ActionMapper()
        called = []
        mapper.register("push_n", lambda r: called.append(r), max_count=2)
        mapper.dispatch(_result("push_n", rep=3))
        assert called == []
        mapper.dispatch(_result("push_n", rep=2))
        assert len(called) == 1

    def test_speed_filter_match(self):
        mapper = ActionMapper()
        called = []
        prims = [_prim("+", speed="high")]
        mapper.register("push", lambda r: called.append(r), speed="high")
        mapper.dispatch(_result("push", prims=prims))
        assert len(called) == 1

    def test_speed_filter_no_match(self):
        mapper = ActionMapper()
        called = []
        prims = [_prim("+", speed="low")]
        mapper.register("push", lambda r: called.append(r), speed="high")
        mapper.dispatch(_result("push", prims=prims))
        assert called == []

    def test_magnitude_filter_match(self):
        mapper = ActionMapper()
        called = []
        prims = [_prim("+", magnitude="high")]
        mapper.register("push", lambda r: called.append(r), magnitude="high")
        mapper.dispatch(_result("push", prims=prims))
        assert len(called) == 1


# ---------------------------------------------------------------------------
# describe()
# ---------------------------------------------------------------------------

class TestDescribe:
    def test_returns_list_of_dicts(self):
        mapper = ActionMapper()
        mapper.register("push", lambda r: None, description="Volume up")
        desc = mapper.describe()
        assert isinstance(desc, list)
        assert desc[0]["family"] == "push"
        assert desc[0]["description"] == "Volume up"

    def test_empty_mapper_returns_empty_list(self):
        assert ActionMapper().describe() == []


# ---------------------------------------------------------------------------
# create_media_player_mapper
# ---------------------------------------------------------------------------

class TestCreateMediaPlayerMapper:
    def test_returns_action_mapper(self):
        mapper = create_media_player_mapper()
        assert isinstance(mapper, ActionMapper)

    def test_push_dispatches_volume_up(self):
        calls = []
        mapper = create_media_player_mapper(on_volume_up=lambda r: calls.append("up"))
        mapper.dispatch(_result("push"))
        assert calls == ["up"]

    def test_pull_dispatches_volume_down(self):
        calls = []
        mapper = create_media_player_mapper(on_volume_down=lambda r: calls.append("dn"))
        mapper.dispatch(_result("pull"))
        assert calls == ["dn"]

    def test_right_dispatches_next_track(self):
        calls = []
        mapper = create_media_player_mapper(on_next_track=lambda r: calls.append("next"))
        mapper.dispatch(_result("right"))
        assert calls == ["next"]
