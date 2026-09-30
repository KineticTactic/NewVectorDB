"""index/query.py — THE query path (shared by both maintenance arms).

DESIGN §6.5 rules:
  - This is the ONLY search function in the codebase.
  - Both maintenance arms call it unchanged.
  - Never prune within a posting — always exact-score every live member
    of every fetched posting.
  - Over-fetch: k' = ceil(k * (1 + overfetch)) to absorb tombstone loss.
  - Zero-candidate units are normal, not errors.
  - There must be NO strategy-specific branches here.
"""

from __future__ import annotations

import math
import numpy as np

from ..core.config import QueryConfig
from ..core.metrics import Counters
from ..core.distance import sim_one_to_many
from ..core.types import VectorId
from .hierarchy import descend, descend_global
from .segment import Segment
from .postings import fetch_posting


def search(
    q: np.ndarray,
    units: list[Segment],
    cfg: QueryConfig,
    counters: Counters,
) -> list[tuple[VectorId, float]]:
    """Search across all searchable units.

    Parameters
    ----------
    q       : float32 (D,) — L2-normalised query vector
    units   : list[Segment] — all searchable units (sealed + memtable)
    cfg     : QueryConfig
    counters: Counters — updated in-place

    Returns
    -------
    list of (VectorId, similarity) sorted descending by similarity, length <= k.
    Tombstoned vectors never appear.
    """
    q = np.asarray(q, dtype=np.float32)
    k_prime = math.ceil(cfg.k * (1.0 + cfg.overfetch))

    candidates: list[tuple[float, VectorId]] = []

    for unit in units:
        _search_unit(q, unit, cfg, counters, k_prime, candidates)

    # Sort descending by similarity, stable for determinism
    candidates.sort(key=lambda x: -x[0])

    # Deduplicate: a vector may appear in multiple units (out-of-place seals)
    seen: set[VectorId] = set()
    results: list[tuple[VectorId, float]] = []
    for sim, vid in candidates:
        if vid not in seen:
            seen.add(vid)
            results.append((vid, float(sim)))
        if len(results) >= cfg.k:
            break

    return results


def _search_unit(
    q: np.ndarray,
    unit: Segment,
    cfg: QueryConfig,
    counters: Counters,
    k_prime: int,
    out: list[tuple[float, VectorId]],
) -> None:
    """Search a single segment and append (sim, vid) candidates to out."""
    D = q.shape[0]

    # Descend to leaf postings
    if cfg.global_beam:
        posting_ids = descend_global(
            q, unit.levels,
            beam_width=cfg.beam_widths[-1],
            n_postings=cfg.n_postings,
            counters=counters,
        )
    else:
        posting_ids = descend(
            q, unit.levels,
            beam_widths=cfg.beam_widths,
            n_postings=cfg.n_postings,
            counters=counters,
        )

    if not posting_ids:
        return

    # Fetch and score each posting
    for pid in posting_ids:
        if pid >= len(unit.postings):
            continue
        posting = fetch_posting(unit.postings[pid], D, counters)

        members = posting.members
        if len(members) == 0:
            continue

        # Filter tombstones AFTER scoring (§6.4 / §6.5)
        # — collect all member vectors first
        member_vecs = unit.vectors[members]   # (M, D)

        # Exact-score EVERY member (no intra-posting pruning, §2)
        sims = sim_one_to_many(q, member_vecs, counters, "scan")
        counters.vectors_scanned += len(members)

        for local_i, vid in enumerate(members):
            vid_int = int(vid)
            if unit.is_alive(vid_int):
                out.append((float(sims[local_i]), vid_int))
