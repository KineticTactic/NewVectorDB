"""index/hierarchy.py — routing tree: build + beam descent.

DESIGN §6.3 rules:
  - Build is bottom-up; levels stored top-first (index 0 = root).
  - Depth 1 = no routing levels above the leaves — flat scan of all leaf centroids.
  - beam_widths is a survivor count per level (not branching factor).
  - Tie-breaking is always stable (deterministic argsort ascending by node id).
  - descend_global maintains a single cross-level frontier; default off.
"""

from __future__ import annotations

import math
import heapq
import numpy as np

from ..core.types import HierarchyLevel, PostingId
from ..core.metrics import Counters
from ..core.distance import sim_one_to_many
from ..core.clustering import balanced_kmeans


# ---------------------------------------------------------------------------
# Build
# ---------------------------------------------------------------------------

def build_hierarchy(
    leaf_centroids: np.ndarray,
    branching_factor: int,
    seed: int,
    depth: int | None = None,
    counters: Counters | None = None,
) -> list[HierarchyLevel]:
    """Build routing hierarchy above leaf centroids. Returns levels top-first."""
    from ..core.metrics import Counters as _C
    if counters is None:
        counters = _C()

    leaf_centroids = np.asarray(leaf_centroids, dtype=np.float32)
    n_leaves = leaf_centroids.shape[0]
    leaf_level = HierarchyLevel(centroids=leaf_centroids, children=[])

    if n_leaves <= 1 or depth == 1:
        return [leaf_level]

    if depth is None:
        levels_above = _build_loop(leaf_centroids, branching_factor, seed, counters)
    else:
        n_routing = depth - 1
        if n_routing <= 0:
            return [leaf_level]
        fan_out = max(2, math.ceil(n_leaves ** (1.0 / n_routing)))
        levels_above = _build_fixed(leaf_centroids, fan_out, n_routing, seed, counters)

    return levels_above + [leaf_level]


def _build_loop(bottom: np.ndarray, B: int, seed: int, counters: Counters) -> list[HierarchyLevel]:
    levels: list[HierarchyLevel] = []
    current = bottom
    level_seed = seed
    while len(current) > B:
        n_clusters = math.ceil(len(current) / B)
        centroids, labels, _ = balanced_kmeans(current, n_clusters, seed=level_seed, counters=counters)
        children = [np.where(labels == i)[0].astype(np.int64) for i in range(n_clusters)]
        levels.append(HierarchyLevel(centroids=centroids, children=children))
        current = centroids
        level_seed += 1
    levels.reverse()
    return levels


def _build_fixed(bottom: np.ndarray, fan_out: int, n_routing: int, seed: int, counters: Counters) -> list[HierarchyLevel]:
    levels: list[HierarchyLevel] = []
    current = bottom
    level_seed = seed
    for _ in range(n_routing):
        if len(current) <= 1:
            break
        n_clusters = max(1, math.ceil(len(current) / fan_out))
        centroids, labels, _ = balanced_kmeans(current, n_clusters, seed=level_seed, counters=counters)
        children = [np.where(labels == i)[0].astype(np.int64) for i in range(n_clusters)]
        levels.append(HierarchyLevel(centroids=centroids, children=children))
        current = centroids
        level_seed += 1
    levels.reverse()
    return levels


# ---------------------------------------------------------------------------
# Descent
# ---------------------------------------------------------------------------

def _stable_top_k(neg_sims: np.ndarray, k: int) -> np.ndarray:
    """Indices of k smallest values in neg_sims; ties broken by ascending index."""
    if k <= 0:
        return np.array([], dtype=np.int64)
    k = min(k, len(neg_sims))
    return np.argsort(neg_sims, kind="stable")[:k].astype(np.int64)


def descend(
    q: np.ndarray,
    levels: list[HierarchyLevel],
    beam_widths: tuple[int, ...] | list[int],
    n_postings: int,
    counters: Counters,
) -> list[PostingId]:
    """Restricted beam descent.  Returns leaf-level node indices (ascending)."""
    if not levels:
        return []
    beam_widths = list(beam_widths)

    def _bw(lvl: int) -> int:
        return beam_widths[min(lvl, len(beam_widths) - 1)]

    n_levels = len(levels)
    leaf_level = levels[-1]

    # ---- depth-1 fast path ------------------------------------------------
    if n_levels == 1:
        sims = sim_one_to_many(q, leaf_level.centroids, counters, "routing")
        k = min(n_postings, leaf_level.centroids.shape[0])
        return sorted(_stable_top_k(-sims, k).tolist())

    routing = levels[:-1]  # all levels except the leaf level

    # Score entire top level
    sims = sim_one_to_many(q, routing[0].centroids, counters, "routing")
    survivors = _stable_top_k(-sims, min(_bw(0), routing[0].centroids.shape[0]))

    # Descend through routing levels
    for li in range(1, len(routing)):
        w = _bw(li)
        child_set = sorted(set(
            c for s in survivors for c in routing[li - 1].children[s].tolist()
        ))
        if not child_set:
            survivors = np.array([], dtype=np.int64)
            break
        ca = np.array(child_set, dtype=np.int64)
        sims = sim_one_to_many(q, routing[li].centroids[ca], counters, "routing")
        survivors = ca[_stable_top_k(-sims, min(w, len(child_set)))]

    # Expand to leaf candidates
    leaf_cands = sorted(set(
        c for s in survivors for c in routing[-1].children[s].tolist()
    ))
    if not leaf_cands:
        return []
    lc = np.array(leaf_cands, dtype=np.int64)
    sims = sim_one_to_many(q, leaf_level.centroids[lc], counters, "routing")
    top = _stable_top_k(-sims, min(n_postings, len(leaf_cands)))
    return sorted(lc[top].tolist())


def descend_global(
    q: np.ndarray,
    levels: list[HierarchyLevel],
    beam_width: int,
    n_postings: int,
    counters: Counters,
) -> list[PostingId]:
    """Single-frontier global beam descent (experimental; default off)."""
    if not levels or n_postings == 0:
        return []

    leaf_level = levels[-1]

    if len(levels) == 1:
        sims = sim_one_to_many(q, leaf_level.centroids, counters, "routing")
        k = min(n_postings, leaf_level.centroids.shape[0])
        return sorted(_stable_top_k(-sims, k).tolist())

    # Seed with top-level nodes
    top = levels[0]
    sims = sim_one_to_many(q, top.centroids, counters, "routing")
    frontier = [(float(-s), 0, int(i)) for i, s in enumerate(sims)]
    heapq.heapify(frontier)

    leaves: list[int] = []
    while frontier and len(leaves) < n_postings:
        neg_s, lvl_idx, nidx = heapq.heappop(frontier)
        if lvl_idx == len(levels) - 1:
            leaves.append(nidx)
        else:
            children = levels[lvl_idx].children[nidx]
            csims = sim_one_to_many(q, levels[lvl_idx + 1].centroids[children], counters, "routing")
            for li, cs in enumerate(csims):
                heapq.heappush(frontier, (float(-cs), lvl_idx + 1, int(children[li])))
        if len(frontier) > beam_width:
            frontier = heapq.nsmallest(beam_width, frontier)
            heapq.heapify(frontier)

    return sorted(set(leaves[:n_postings]))
