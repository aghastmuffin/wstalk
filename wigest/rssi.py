"""RSSI acquisition module.

Provides an abstract :class:`RSSISource` base class plus two concrete
implementations:

* :class:`MockRSSISource`  – feeds a pre-recorded / synthetic numpy array;
  used for unit tests and offline replay.
* :class:`LiveRSSISource`  – polls the operating system for the current
  Received Signal Strength Indicator (RSSI) in dBm:

  - **Linux**: tries ``iw`` first, falls back to ``iwconfig``.
  - **Windows**: uses ``netsh wlan show interfaces``.

The abstraction makes it straightforward to swap in custom sources
(e.g. a TCP socket receiving samples from remote hardware) by
sub-classing :class:`RSSISource` and implementing :meth:`read`.

Paper reference
---------------
WiGest §III-A – "Signal Acquisition":
  RSSI is sampled at ~10 Hz from the NIC driver reports.  Multiple APs
  can be monitored simultaneously; each yields an independent stream.
"""

import abc
import platform
import re
import subprocess
import time
from typing import Dict, Iterator, List, Optional


class RSSISource(abc.ABC):
    """Abstract base class for RSSI data sources.

    Sub-classes must implement :meth:`read`.  The source is used as an
    iterator via :meth:`stream`, which calls :meth:`read` and inserts the
    configured inter-sample delay.

    Parameters
    ----------
    sample_rate:
        Target number of RSSI samples per second (Hz).  The default of
        ``10`` matches the acquisition rate used in the WiGest paper.
    """

    def __init__(self, sample_rate: float = 10.0) -> None:
        self.sample_rate = sample_rate
        self._interval = 1.0 / sample_rate

    @abc.abstractmethod
    def read(self) -> Dict[str, float]:
        """Return the current RSSI for each monitored AP.

        Returns
        -------
        dict
            Mapping ``{bssid_or_ssid: rssi_dbm}``.  At least one entry
            must be present; return an empty dict to signal end-of-stream.
        """

    def stream(self) -> Iterator[Dict[str, float]]:
        """Yield RSSI snapshots indefinitely, honouring *sample_rate*.

        Yields
        ------
        dict
            Same format as :meth:`read`.  Stops when :meth:`read` returns
            an empty dict.
        """
        while True:
            t_start = time.monotonic()
            snapshot = self.read()
            if not snapshot:
                return
            yield snapshot
            elapsed = time.monotonic() - t_start
            sleep_for = self._interval - elapsed
            if sleep_for > 0:
                time.sleep(sleep_for)


# ---------------------------------------------------------------------------
# Mock source – synthetic / pre-recorded data
# ---------------------------------------------------------------------------

class MockRSSISource(RSSISource):
    """Replays a pre-recorded or synthetically generated RSSI sequence.

    Useful for unit tests and offline algorithm development because no
    hardware or OS calls are required.

    Parameters
    ----------
    data:
        Either a 1-D list/array of RSSI values (treated as a single AP
        named ``"mock"``) **or** a dict mapping AP names to 1-D sequences.
    sample_rate:
        Playback rate in Hz.  Does *not* insert real-time delays unless
        *realtime* is ``True``.
    realtime:
        If ``True``, honour the inter-sample interval with ``time.sleep``.
        Defaults to ``False`` so that tests run at full speed.
    loop:
        Repeat the sequence indefinitely.
    """

    def __init__(
        self,
        data,
        sample_rate: float = 10.0,
        realtime: bool = False,
        loop: bool = False,
    ) -> None:
        super().__init__(sample_rate)
        if not isinstance(data, dict):
            data = {"mock": list(data)}
        # Ensure all sequences are plain Python lists for simple indexing.
        self._data: Dict[str, List[float]] = {k: list(v) for k, v in data.items()}
        self._realtime = realtime
        self._loop = loop
        self._index = 0
        # Determine total length (all sequences must be the same length).
        lengths = {k: len(v) for k, v in self._data.items()}
        if len(set(lengths.values())) > 1:
            raise ValueError(
                f"All AP sequences must have equal length; got {lengths}"
            )
        self._length: int = next(iter(lengths.values())) if lengths else 0

    def read(self) -> Dict[str, float]:
        """Return the next RSSI snapshot from the pre-recorded data."""
        if self._index >= self._length:
            if self._loop:
                self._index = 0
            else:
                return {}
        snapshot = {ap: seq[self._index] for ap, seq in self._data.items()}
        self._index += 1
        if self._realtime:
            time.sleep(self._interval)
        return snapshot

    def reset(self) -> None:
        """Rewind to the beginning of the recorded sequence."""
        self._index = 0


# ---------------------------------------------------------------------------
# Live source – OS-level polling
# ---------------------------------------------------------------------------

class LiveRSSISource(RSSISource):
    """Live RSSI polling via OS built-in commands.

    On **Linux** the source first attempts ``iw dev <iface> link`` and
    falls back to ``iwconfig <iface>``.  On **Windows** it uses
    ``netsh wlan show interfaces``.

    Parameters
    ----------
    interface:
        NIC interface name (e.g. ``"wlan0"`` on Linux,
        ``"Wi-Fi"`` on Windows).  Pass ``None`` to auto-detect.
    sample_rate:
        Target acquisition rate in Hz.
    bssid_filter:
        Optional list of BSSIDs / SSIDs to keep.  When ``None`` all
        visible APs are included (currently only the *associated* AP is
        reliably available without root privileges).
    """

    def __init__(
        self,
        interface: Optional[str] = None,
        sample_rate: float = 10.0,
        bssid_filter: Optional[List[str]] = None,
    ) -> None:
        super().__init__(sample_rate)
        self._os = platform.system()
        self._iface = interface or self._detect_interface()
        self._filter = set(bssid_filter) if bssid_filter else None

    # ------------------------------------------------------------------
    # Interface detection helpers
    # ------------------------------------------------------------------

    def _detect_interface(self) -> str:
        """Try to auto-detect the wireless interface name."""
        if self._os == "Linux":
            return self._detect_linux_iface()
        if self._os == "Windows":
            return "Wi-Fi"
        return "wlan0"

    @staticmethod
    def _detect_linux_iface() -> str:
        """Return the first wireless interface reported by ``iw dev``."""
        try:
            out = subprocess.check_output(
                ["iw", "dev"], stderr=subprocess.DEVNULL, text=True
            )
            m = re.search(r"Interface\s+(\S+)", out)
            if m:
                return m.group(1)
        except (FileNotFoundError, subprocess.CalledProcessError):
            pass
        # Fall back to the first ``wlan*`` interface.
        try:
            out = subprocess.check_output(
                ["iwconfig"], stderr=subprocess.STDOUT, text=True
            )
            m = re.search(r"^(\w+)\s+IEEE", out, re.MULTILINE)
            if m:
                return m.group(1)
        except (FileNotFoundError, subprocess.CalledProcessError):
            pass
        return "wlan0"

    # ------------------------------------------------------------------
    # Platform-specific RSSI parsers
    # ------------------------------------------------------------------

    def _read_linux(self) -> Dict[str, float]:
        """Parse RSSI on Linux using ``iw`` or ``iwconfig``."""
        # --- try iw first (more reliable) ---
        try:
            out = subprocess.check_output(
                ["iw", "dev", self._iface, "link"],
                stderr=subprocess.DEVNULL,
                text=True,
            )
            # "signal: -65 dBm" or "signal: -65/-95 dBm"
            m = re.search(r"signal:\s*([-\d]+)", out)
            if m:
                rssi = float(m.group(1))
                bssid_m = re.search(r"Connected to\s+([\da-fA-F:]+)", out)
                key = bssid_m.group(1) if bssid_m else self._iface
                return {key: rssi}
        except (FileNotFoundError, subprocess.CalledProcessError):
            pass

        # --- fall back to iwconfig ---
        try:
            out = subprocess.check_output(
                ["iwconfig", self._iface],
                stderr=subprocess.STDOUT,
                text=True,
            )
            # "Signal level=-65 dBm" or "Signal level=51/100"
            m = re.search(r"Signal level[=:]\s*([-\d]+)", out)
            if m:
                rssi = float(m.group(1))
                ap_m = re.search(r"Access Point[:\s]+([\dA-Fa-f:]+)", out)
                key = ap_m.group(1) if ap_m else self._iface
                return {key: rssi}
        except (FileNotFoundError, subprocess.CalledProcessError):
            pass

        return {}

    def _read_windows(self) -> Dict[str, float]:
        """Parse RSSI on Windows using ``netsh wlan show interfaces``."""
        try:
            out = subprocess.check_output(
                ["netsh", "wlan", "show", "interfaces"],
                stderr=subprocess.DEVNULL,
                text=True,
                encoding="utf-8",
                errors="replace",
            )
        except (FileNotFoundError, subprocess.CalledProcessError):
            return {}

        # Locate the block for the configured interface.
        blocks = re.split(r"\n\s*\n", out)
        for block in blocks:
            if self._iface.lower() in block.lower():
                # "Signal : 78%" → convert percentage to approximate dBm.
                pct_m = re.search(r"Signal\s*:\s*(\d+)\s*%", block)
                if pct_m:
                    pct = float(pct_m.group(1))
                    # Rough linear mapping: 100 % ≈ -50 dBm, 0 % ≈ -100 dBm.
                    rssi = (pct / 2.0) - 100.0
                    bssid_m = re.search(r"BSSID\s*:\s*([\da-fA-F:]+)", block)
                    key = bssid_m.group(1) if bssid_m else self._iface
                    return {key: rssi}
        return {}

    # ------------------------------------------------------------------
    # Public interface
    # ------------------------------------------------------------------

    def read(self) -> Dict[str, float]:
        """Poll the OS for current RSSI values.

        Returns
        -------
        dict
            ``{bssid: rssi_dbm}`` for the associated AP(s).
        """
        if self._os == "Linux":
            snapshot = self._read_linux()
        elif self._os == "Windows":
            snapshot = self._read_windows()
        else:
            snapshot = {}

        if self._filter and snapshot:
            snapshot = {k: v for k, v in snapshot.items() if k in self._filter}
        return snapshot
