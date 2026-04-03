"""Gesture pattern encoding and family matching module.

Implements WiGest §III-F – "Gesture Recognition".

Encoding
--------
Each primitive is mapped to a single character:

* Rising edge  ``'+'`` → ``'+'``
* Falling edge ``'-'`` → ``'-'``
* Pause        ``'0'`` → ``'0'``

The primitive sequence is concatenated into a *gesture string* such as
``"+-+-"`` (two hand pushes) or ``"+0-"`` (push, pause, pull).

Gesture families
----------------
The paper defines the following families (Table I):

+---------+----------------------------------+-------------------------+
| Family  | Pattern (regex)                  | Description             |
+=========+==================================+=========================+
| push    | ``+-``                           | hand toward then away   |
+---------+----------------------------------+-------------------------+
| pull    | ``-+``                           | hand away then toward   |
+---------+----------------------------------+-------------------------+
| push_n  | ``(\\+-){2,}``                   | repeated push           |
+---------+----------------------------------+-------------------------+
| pull_n  | ``(-\\+){2,}``                   | repeated pull           |
+---------+----------------------------------+-------------------------+
| left    | ``+-0-``                         | push-pause-pull         |
+---------+----------------------------------+-------------------------+
| right   | ``-+0+``                         | pull-pause-push         |
+---------+----------------------------------+-------------------------+
| up      | ``+``                            | single rising           |
+---------+----------------------------------+-------------------------+
| down    | ``-``                            | single falling          |
+---------+----------------------------------+-------------------------+
| circle  | ``+-+-0-+`` or ``-+0+-+``        | circular motion         |
+---------+----------------------------------+-------------------------+

Attributes
----------
After matching, two additional attributes are computed:

repetition_count
    Number of full push/pull cycles in the pattern.
frequency_hz
    Mean cycle rate = repetition_count / total_duration_s.

Paper reference
---------------
WiGest §III-F – "Gesture Family Matching":
  "Primitives are encoded as a string and matched against family patterns
   using a longest-prefix strategy."
"""

import re
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

from wigest.primitives import Primitive


# ---------------------------------------------------------------------------
# Gesture families
# ---------------------------------------------------------------------------

# Each family entry: (name, regex_pattern, description)
# Patterns are tried in order; the first full match wins.
GESTURE_FAMILIES: List[Tuple[str, str, str]] = [
    # Repeated gestures first (longer patterns take priority).
    ("push_n", r"(\+-){2,}", "repeated push (N cycles)"),
    ("pull_n", r"(-\+){2,}", "repeated pull (N cycles)"),
    ("left",   r"\+-0-",     "push-pause-pull (left swipe)"),
    ("right",  r"-\+0\+",    "pull-pause-push (right swipe)"),
    ("circle", r"\+-\+-0-\+|-\+0\+-\+", "circular motion"),
    # Single gestures last (subsets of the above).
    ("push",   r"\+-",       "single push"),
    ("pull",   r"-\+",       "single pull"),
    ("up",     r"\+",        "single rising edge"),
    ("down",   r"-",         "single falling edge"),
]

# Pre-compile regexes for speed.
_COMPILED_FAMILIES: List[Tuple[str, re.Pattern, str]] = [
    (name, re.compile(pattern), desc)
    for name, pattern, desc in GESTURE_FAMILIES
]


# ---------------------------------------------------------------------------
# Result dataclass
# ---------------------------------------------------------------------------

@dataclass
class GestureResult:
    """Result of gesture recognition.

    Attributes
    ----------
    family:
        Matched gesture family name (e.g. ``'push'``), or ``'unknown'``.
    pattern:
        Encoded primitive string (e.g. ``'+-'``).
    repetition_count:
        Number of full push/pull cycles detected.
    frequency_hz:
        Estimated gesture frequency in Hz.
    duration_s:
        Total duration of the gesture sequence in seconds.
    primitives:
        The raw primitive list that was encoded.
    extra:
        Additional metadata (e.g., speed/magnitude of component primitives).
    """

    family: str
    pattern: str
    repetition_count: int
    frequency_hz: float
    duration_s: float
    primitives: List[Primitive] = field(default_factory=list)
    extra: dict = field(default_factory=dict)

    def __str__(self) -> str:  # pragma: no cover
        return (
            f"Gesture(family={self.family!r}, pattern={self.pattern!r}, "
            f"count={self.repetition_count}, freq={self.frequency_hz:.2f} Hz, "
            f"dur={self.duration_s:.2f}s)"
        )


# ---------------------------------------------------------------------------
# Encoding
# ---------------------------------------------------------------------------

def encode_primitives(primitives: List[Primitive]) -> str:
    """Encode a list of primitives as a gesture string.

    Each primitive maps to its :attr:`~wigest.primitives.Primitive.kind`
    character (``'+'``, ``'-'``, ``'0'``).

    Parameters
    ----------
    primitives:
        Ordered list of primitives (typically sorted by *start_idx*).

    Returns
    -------
    str
        Concatenated kind characters, e.g. ``"+-0+"`` or ``"+-+-"``.
    """
    return "".join(p.kind for p in primitives)


# ---------------------------------------------------------------------------
# Matching
# ---------------------------------------------------------------------------

def match_gesture(
    primitives: List[Primitive],
    sample_rate: float = 10.0,
    extra_context: Optional[dict] = None,
) -> GestureResult:
    """Match a primitive sequence to the closest WiGest gesture family.

    Strategy: for each family (tried longest-pattern-first) attempt a
    *full-string* regex match.  The first match wins.  If no family
    matches, returns ``family='unknown'``.

    Parameters
    ----------
    primitives:
        Primitive list to classify.
    sample_rate:
        Acquisition rate (Hz), used to compute frequency.
    extra_context:
        Optional dict attached to :attr:`GestureResult.extra`.

    Returns
    -------
    GestureResult
        Recognition result with family, attributes, and metadata.
    """
    pattern = encode_primitives(primitives)
    total_duration = sum(p.duration_s for p in primitives)

    matched_family = "unknown"
    rep_count = 1

    for name, regex, _ in _COMPILED_FAMILIES:
        if regex.fullmatch(pattern):
            matched_family = name
            rep_count = _count_repetitions(pattern, name)
            break

    freq = (rep_count / total_duration) if total_duration > 0 else 0.0

    return GestureResult(
        family=matched_family,
        pattern=pattern,
        repetition_count=rep_count,
        frequency_hz=freq,
        duration_s=total_duration,
        primitives=list(primitives),
        extra=extra_context or {},
    )


def _count_repetitions(pattern: str, family: str) -> int:
    """Count the number of full gesture cycles in *pattern*.

    For repeated families (``push_n``, ``pull_n``) this is the number of
    ``+-`` or ``-+`` pairs respectively.  For all other families it is 1.

    Parameters
    ----------
    pattern:
        Encoded primitive string.
    family:
        Matched gesture family name.

    Returns
    -------
    int
        Number of repetitions (always ≥ 1).
    """
    if family == "push_n":
        return max(1, pattern.count("+-"))
    if family == "pull_n":
        return max(1, pattern.count("-+"))
    return 1


# ---------------------------------------------------------------------------
# Batch helper
# ---------------------------------------------------------------------------

def match_gestures_stream(
    primitive_batches: List[List[Primitive]],
    sample_rate: float = 10.0,
) -> List[GestureResult]:
    """Recognise a list of pre-segmented primitive batches.

    Parameters
    ----------
    primitive_batches:
        Each element is a complete primitive sequence for one gesture.
    sample_rate:
        Acquisition rate in Hz.

    Returns
    -------
    list of GestureResult
    """
    return [match_gesture(batch, sample_rate=sample_rate) for batch in primitive_batches]
