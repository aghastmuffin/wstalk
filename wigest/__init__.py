"""WiGest: WiFi gesture recognition library.

Implements the WiGest system as described in:
  Abdelnasser, H., Youssef, M., & Harras, K. A. (2015).
  WiGest: A ubiquitous WiFi-based gesture recognition system.
  IEEE INFOCOM 2015.

Public API
----------
WiGest           -- main pipeline controller
RSSISource       -- abstract RSSI source (base class)
MockRSSISource   -- synthetic / unit-test source
LiveRSSISource   -- live polling via iwconfig / netsh
GestureResult    -- result returned by ``WiGest.process()``
ActionMapper     -- binds gesture families to callbacks
"""

from wigest.rssi import RSSISource, MockRSSISource, LiveRSSISource  # noqa: F401
from wigest.denoising import denoise  # noqa: F401
from wigest.primitives import (  # noqa: F401
    Primitive,
    extract_primitives,
)
from wigest.preamble import PreambleDetector  # noqa: F401
from wigest.voting import majority_vote  # noqa: F401
from wigest.gestures import (  # noqa: F401
    encode_primitives,
    match_gesture,
    GestureResult,
    GESTURE_FAMILIES,
)
from wigest.actions import ActionMapper  # noqa: F401
from wigest.pipeline import WiGest  # noqa: F401

__version__ = "0.1.0"
__all__ = [
    "WiGest",
    "RSSISource",
    "MockRSSISource",
    "LiveRSSISource",
    "GestureResult",
    "ActionMapper",
    "denoise",
    "Primitive",
    "extract_primitives",
    "PreambleDetector",
    "majority_vote",
    "encode_primitives",
    "match_gesture",
    "GESTURE_FAMILIES",
]
