"""Action mapping layer.

Provides an application-facing :class:`ActionMapper` that binds WiGest
gesture families (and optional attribute filters) to Python callbacks.

This keeps gesture recognition decoupled from application logic: the
WiGest pipeline produces :class:`~wigest.gestures.GestureResult` objects
and the :class:`ActionMapper` dispatches them to the appropriate handler.

Example usage (media-player-style bindings)::

    from wigest.actions import ActionMapper

    mapper = ActionMapper()

    @mapper.on("push")
    def volume_up(result):
        print(f"Volume UP  (speed={result.extra.get('speed')!r})")

    @mapper.on("pull")
    def volume_down(result):
        print("Volume DOWN")

    @mapper.on("push_n", min_count=3)
    def skip_forward(result):
        print(f"Skip forward {result.repetition_count} steps")

    # In your gesture loop:
    # mapper.dispatch(gesture_result)

Paper reference
---------------
WiGest §IV – "Application Mapping":
  "We map each gesture family to an application action.  The mapping is
   kept configurable so that the same gesture engine can drive different
   applications."
"""

from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional

from wigest.gestures import GestureResult


# ---------------------------------------------------------------------------
# Binding entry
# ---------------------------------------------------------------------------

@dataclass
class _Binding:
    """Internal representation of a single action binding.

    Attributes
    ----------
    family:
        Gesture family name to match (e.g. ``'push'``).  ``'*'`` matches
        any family.
    callback:
        Callable invoked with the :class:`~wigest.gestures.GestureResult`
        when matched.
    min_count:
        Minimum repetition count required (inclusive).  ``None`` means
        no constraint.
    max_count:
        Maximum repetition count required (inclusive).  ``None`` means
        no constraint.
    speed:
        Required speed class (``'high'``, ``'medium'``, ``'low'``).
        ``None`` means any speed.
    magnitude:
        Required magnitude class (``'high'``, ``'low'``).
        ``None`` means any magnitude.
    description:
        Optional human-readable description of the action.
    """

    family: str
    callback: Callable[[GestureResult], Any]
    min_count: Optional[int] = None
    max_count: Optional[int] = None
    speed: Optional[str] = None
    magnitude: Optional[str] = None
    description: str = ""


# ---------------------------------------------------------------------------
# ActionMapper
# ---------------------------------------------------------------------------

class ActionMapper:
    """Binds gesture families to application callbacks.

    Bindings are evaluated in registration order; all matching bindings
    are called (not just the first), unless *exclusive* is set to
    ``True`` when registering.

    Parameters
    ----------
    default_callback:
        Called for any :class:`~wigest.gestures.GestureResult` that does
        not match any binding.  ``None`` (default) silently ignores
        unmatched gestures.
    """

    def __init__(
        self,
        default_callback: Optional[Callable[[GestureResult], Any]] = None,
    ) -> None:
        self._bindings: List[_Binding] = []
        self._default = default_callback

    # ------------------------------------------------------------------
    # Registration
    # ------------------------------------------------------------------

    def on(
        self,
        family: str,
        *,
        min_count: Optional[int] = None,
        max_count: Optional[int] = None,
        speed: Optional[str] = None,
        magnitude: Optional[str] = None,
        description: str = "",
    ) -> Callable:
        """Decorator / function that registers *callback* for *family*.

        Can be used as a decorator::

            @mapper.on("push")
            def handler(result): ...

        Or called directly::

            mapper.on("push")(my_handler)

        Parameters
        ----------
        family:
            Gesture family name.  Use ``'*'`` to catch all gestures.
        min_count, max_count:
            Optional repetition count range filter.
        speed:
            Optional speed class filter.
        magnitude:
            Optional magnitude class filter.
        description:
            Human-readable description shown in :meth:`describe`.

        Returns
        -------
        Callable
            Decorator that registers the wrapped function and returns it
            unchanged.
        """
        def _decorator(callback: Callable) -> Callable:
            self._bindings.append(
                _Binding(
                    family=family,
                    callback=callback,
                    min_count=min_count,
                    max_count=max_count,
                    speed=speed,
                    magnitude=magnitude,
                    description=description or getattr(callback, "__doc__", ""),
                )
            )
            return callback
        return _decorator

    def register(
        self,
        family: str,
        callback: Callable[[GestureResult], Any],
        **kwargs,
    ) -> None:
        """Imperatively register a binding (alternative to the decorator).

        Parameters
        ----------
        family:
            Gesture family name.
        callback:
            Handler callable.
        **kwargs:
            Forwarded to :meth:`on`.
        """
        self.on(family, **kwargs)(callback)

    # ------------------------------------------------------------------
    # Dispatch
    # ------------------------------------------------------------------

    def dispatch(self, result: GestureResult) -> int:
        """Dispatch *result* to all matching registered callbacks.

        Parameters
        ----------
        result:
            Recognised gesture result from the WiGest pipeline.

        Returns
        -------
        int
            Number of callbacks that were invoked.
        """
        called = 0
        for binding in self._bindings:
            if self._matches(binding, result):
                binding.callback(result)
                called += 1

        if called == 0 and self._default is not None:
            self._default(result)
            called = 1

        return called

    @staticmethod
    def _matches(binding: _Binding, result: GestureResult) -> bool:
        """Return True if *result* satisfies all constraints in *binding*."""
        if binding.family != "*" and binding.family != result.family:
            return False
        if binding.min_count is not None and result.repetition_count < binding.min_count:
            return False
        if binding.max_count is not None and result.repetition_count > binding.max_count:
            return False
        # Speed / magnitude checks look at the first primitive that has them.
        if binding.speed is not None:
            speeds = [p.speed for p in result.primitives if p.kind != "0"]
            if speeds and speeds[0] != binding.speed:
                return False
        if binding.magnitude is not None:
            mags = [p.magnitude for p in result.primitives if p.kind != "0"]
            if mags and mags[0] != binding.magnitude:
                return False
        return True

    # ------------------------------------------------------------------
    # Introspection
    # ------------------------------------------------------------------

    def describe(self) -> List[Dict[str, Any]]:
        """Return a list of binding descriptions for introspection / help.

        Returns
        -------
        list of dict
            Each dict has keys: ``family``, ``description``, ``min_count``,
            ``max_count``, ``speed``, ``magnitude``.
        """
        result = []
        for b in self._bindings:
            result.append({
                "family": b.family,
                "description": b.description,
                "min_count": b.min_count,
                "max_count": b.max_count,
                "speed": b.speed,
                "magnitude": b.magnitude,
            })
        return result

    def clear(self) -> None:
        """Remove all registered bindings."""
        self._bindings.clear()


# ---------------------------------------------------------------------------
# Built-in example mapper (media player)
# ---------------------------------------------------------------------------

def create_media_player_mapper(
    on_volume_up: Optional[Callable] = None,
    on_volume_down: Optional[Callable] = None,
    on_next_track: Optional[Callable] = None,
    on_prev_track: Optional[Callable] = None,
    on_play_pause: Optional[Callable] = None,
    on_stop: Optional[Callable] = None,
) -> ActionMapper:
    """Create a pre-configured media-player :class:`ActionMapper`.

    This function wires common media-player actions to WiGest gesture
    families as described in the WiGest paper (Table II).  All parameters
    are optional callbacks; pass ``None`` to leave an action unbound.

    Gesture-to-action mapping
    -------------------------
    * ``push``      → volume up (push hand toward AP)
    * ``pull``      → volume down (pull hand away from AP)
    * ``right``     → next track (right swipe)
    * ``left``      → previous track (left swipe)
    * ``push_n`` ≥2 → play/pause (double push)
    * ``down``      → stop

    Parameters
    ----------
    on_volume_up, on_volume_down, on_next_track, on_prev_track,
    on_play_pause, on_stop:
        Application callbacks.  Each receives the :class:`~wigest.gestures.GestureResult`.

    Returns
    -------
    ActionMapper
        Configured mapper ready to receive :meth:`~ActionMapper.dispatch` calls.
    """
    mapper = ActionMapper()

    def _noop(name: str) -> Callable:
        def _cb(result: GestureResult) -> None:  # pragma: no cover
            print(f"[WiGest] {name}")
        return _cb

    mapper.register(
        "push", on_volume_up or _noop("volume_up"),
        description="Volume up",
    )
    mapper.register(
        "pull", on_volume_down or _noop("volume_down"),
        description="Volume down",
    )
    mapper.register(
        "right", on_next_track or _noop("next_track"),
        description="Next track",
    )
    mapper.register(
        "left", on_prev_track or _noop("prev_track"),
        description="Previous track",
    )
    mapper.register(
        "push_n", on_play_pause or _noop("play_pause"),
        min_count=2, description="Play/pause",
    )
    mapper.register(
        "down", on_stop or _noop("stop"),
        description="Stop",
    )
    return mapper
