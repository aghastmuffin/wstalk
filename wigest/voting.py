"""Multi-AP majority voting module.

Implements WiGest §III-E – "Multi-AP Support".

When RSSI streams from multiple Access Points (APs) are available, each AP
independently produces a list of gesture primitives.  The primitives are
then fused by **majority vote** to obtain a single robust primitive
sequence:

* At each time step, all APs vote for a primitive kind
  (``'+'``, ``'-'``, or ``'0'``).
* The kind receiving the most votes wins.
* Ties are broken in favour of the non-pause kind (motion is rarer and
  more informative than silence) and then alphabetically.

Paper reference
---------------
WiGest §III-E – "Multi-AP Fusion":
  "The final primitive at each time step is determined by majority vote
   over all APs.  This reduces the effect of AP-specific interference."
"""

from collections import Counter
from typing import Dict, List, Optional

from wigest.primitives import Primitive


def majority_vote(
    ap_primitives: Dict[str, List[Primitive]],
    sample_rate: float = 10.0,
    total_samples: Optional[int] = None,
) -> List[Primitive]:
    """Fuse per-AP primitive lists into one via majority vote.

    Each AP produces an independent list of primitives with sample-index
    boundaries.  This function aligns them on a common sample grid and
    selects, for every non-overlapping window, the primitive kind voted
    by the majority of APs.

    Algorithm
    ---------
    1. Build a per-sample *kind* array for every AP by filling in the
       primitive kind for samples covered by each primitive and ``None``
       for uncovered samples.
    2. For each sample position take the majority kind among APs that
       reported a kind (ignore ``None`` votes).
    3. Run-length-encode the majority kind sequence to collapse consecutive
       identical kinds into a single :class:`~wigest.primitives.Primitive`.
    4. Compute speed/magnitude/duration from the reconstructed segment.

    Parameters
    ----------
    ap_primitives:
        Mapping ``{ap_id: [Primitive, ...]}`` — one entry per AP.
    sample_rate:
        Acquisition rate in Hz (used for duration calculations).
    total_samples:
        Length of the common sample grid.  Inferred from the maximum
        ``end_idx`` across all primitives when ``None``.

    Returns
    -------
    list of :class:`~wigest.primitives.Primitive`
        Fused primitive sequence sorted by *start_idx*.  Empty list when
        *ap_primitives* is empty or contains no primitives.
    """
    if not ap_primitives:
        return []

    # Determine total sample length.
    if total_samples is None:
        total_samples = max(
            (p.end_idx for prims in ap_primitives.values() for p in prims),
            default=0,
        )
    if total_samples == 0:
        return []

    # Build per-AP kind arrays (length = total_samples).
    # kind_grid[ap][sample] = '+' | '-' | '0' | None
    kind_grids: Dict[str, List[Optional[str]]] = {}
    for ap_id, prims in ap_primitives.items():
        grid: List[Optional[str]] = [None] * total_samples
        for prim in prims:
            for idx in range(prim.start_idx, min(prim.end_idx, total_samples)):
                grid[idx] = prim.kind
        kind_grids[ap_id] = grid

    # Majority vote at each sample.
    voted: List[Optional[str]] = []
    grids = list(kind_grids.values())
    for i in range(total_samples):
        votes = [g[i] for g in grids if g[i] is not None]
        if not votes:
            voted.append(None)
        else:
            cnt = Counter(votes)
            # Break ties: prefer non-pause, then alphabetical.
            winner = max(
                cnt.keys(),
                key=lambda k: (cnt[k], k != "0", k),
            )
            voted.append(winner)

    # Run-length encode the voted sequence.
    fused: List[Primitive] = []
    i = 0
    while i < total_samples:
        kind = voted[i]
        if kind is None:
            i += 1
            continue
        # Find the end of this run.
        j = i + 1
        while j < total_samples and voted[j] == kind:
            j += 1
        duration_s = (j - i) / sample_rate
        # Compute consensus speed/magnitude from the first AP that has a
        # primitive overlapping this range.
        speed, magnitude, delta = _consensus_attrs(
            ap_primitives, kind, i, j
        )
        fused.append(
            Primitive(
                kind=kind,
                speed=speed,
                magnitude=magnitude,
                start_idx=i,
                end_idx=j,
                delta_rssi=delta,
                duration_s=duration_s,
            )
        )
        i = j

    return fused


def _consensus_attrs(
    ap_primitives: Dict[str, List[Primitive]],
    kind: str,
    start: int,
    end: int,
) -> tuple:
    """Return (speed, magnitude, delta_rssi) by majority from overlapping primitives."""
    speeds: List[str] = []
    magnitudes: List[str] = []
    deltas: List[float] = []

    for prims in ap_primitives.values():
        for p in prims:
            if p.kind == kind and p.start_idx < end and p.end_idx > start:
                speeds.append(p.speed)
                magnitudes.append(p.magnitude)
                deltas.append(p.delta_rssi)

    speed = Counter(speeds).most_common(1)[0][0] if speeds else "medium"
    magnitude = Counter(magnitudes).most_common(1)[0][0] if magnitudes else "low"
    delta = float(sum(deltas) / len(deltas)) if deltas else 0.0
    return speed, magnitude, delta
