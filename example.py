#!/usr/bin/env python3
"""WiGest runnable example.

Demonstrates the full WiGest pipeline with live or simulated RSSI:

- **Live mode** (default): polls the system WiFi interface at 10 Hz and
  prints recognised gestures in real time.
- **Simulate mode** (``--simulate``): runs on a synthetic RSSI sequence
  without any hardware so the script can be run anywhere.

Usage
-----
    # Live WiFi (Linux / Windows):
    python example.py

    # Simulated push-pull gestures (no hardware needed):
    python example.py --simulate

    # Media-player action bindings on simulated input:
    python example.py --simulate --media

Press Ctrl-C to stop.
"""

import argparse
import sys

import numpy as np


def build_simulated_source(sample_rate: float = 10.0):
    """Build a MockRSSISource with a repeating push-pull RSSI sequence."""
    from wigest import MockRSSISource

    t = np.linspace(0, 8 * np.pi, int(sample_rate * 8))
    # Two cycles of sinusoidal RSSI swings (push-pull gestures).
    sig = np.sin(t) * 12 - 65.0
    # Add small noise to make denoising non-trivial.
    rng = np.random.default_rng(2024)
    sig = sig + rng.normal(0, 0.5, len(sig))
    return MockRSSISource(sig, sample_rate=sample_rate, loop=True)


def build_live_source(sample_rate: float = 10.0, interface: str = None):
    """Build a LiveRSSISource for the current system."""
    from wigest import LiveRSSISource

    return LiveRSSISource(interface=interface, sample_rate=sample_rate)


def build_mapper(media: bool = False):
    """Build an ActionMapper with print callbacks."""
    from wigest.actions import ActionMapper, create_media_player_mapper

    if media:
        # Wire up named media-player callbacks.
        mapper = create_media_player_mapper(
            on_volume_up=lambda r: print(
                f"  🔊 Volume UP  (speed={(r.primitives[0].speed if r.primitives else '?')!r})"
            ),
            on_volume_down=lambda r: print("  🔉 Volume DOWN"),
            on_next_track=lambda r: print("  ⏭  Next track"),
            on_prev_track=lambda r: print("  ⏮  Previous track"),
            on_play_pause=lambda r: print(
                f"  ⏯  Play/Pause  (×{r.repetition_count})"
            ),
            on_stop=lambda r: print("  ⏹  Stop"),
        )
    else:
        mapper = ActionMapper(
            default_callback=lambda r: print(
                f"  [unbound] family={r.family!r}  pattern={r.pattern!r}"
            )
        )

        @mapper.on("*")
        def _print_gesture(result):
            print(
                f"  Gesture: family={result.family!r}  pattern={result.pattern!r}"
                f"  count={result.repetition_count}  freq={result.frequency_hz:.2f} Hz"
                f"  dur={result.duration_s:.2f}s"
            )

    return mapper


def main():
    parser = argparse.ArgumentParser(description="WiGest live gesture recognition")
    parser.add_argument(
        "--simulate",
        action="store_true",
        help="Use a synthetic RSSI sequence instead of live WiFi",
    )
    parser.add_argument(
        "--media",
        action="store_true",
        help="Use media-player action bindings",
    )
    parser.add_argument(
        "--interface",
        default=None,
        help="WiFi interface name (Linux: wlan0; Windows: Wi-Fi)",
    )
    parser.add_argument(
        "--sample-rate",
        type=float,
        default=10.0,
        help="RSSI sampling rate in Hz (default: 10)",
    )
    parser.add_argument(
        "--max-gestures",
        type=int,
        default=None,
        help="Stop after this many recognised gestures",
    )
    parser.add_argument(
        "--no-preamble",
        action="store_true",
        help="Skip preamble detection (process all RSSI immediately)",
    )
    args = parser.parse_args()

    from wigest import WiGest

    if args.simulate:
        print("=== WiGest – Simulated mode ===")
        source = build_simulated_source(sample_rate=args.sample_rate)
    else:
        print(f"=== WiGest – Live mode (interface: {args.interface or 'auto'}) ===")
        source = build_live_source(
            sample_rate=args.sample_rate,
            interface=args.interface,
        )

    mapper = build_mapper(media=args.media)

    wg = WiGest(
        source,
        action_mapper=mapper,
        sample_rate=args.sample_rate,
        preamble=not args.no_preamble,
    )

    print("Listening for gestures… (Ctrl-C to stop)\n")
    try:
        for gesture in wg.run(max_gestures=args.max_gestures):
            pass  # Mapper already printed the result.
    except KeyboardInterrupt:
        print("\nStopped.")
        sys.exit(0)


if __name__ == "__main__":
    main()
