"""index/segment.py — Segment = one hierarchy + its postings (a searchable unit).

A Segment wraps:
  - a list of HierarchyLevel (may be a single leaf level for flat segments)
  - a list of Posting (one per leaf)
  - a tombstone set
  - a reference to the global vector array (shared, never copied)

The flat MemTable used by the out-of-place maintainer is represented as a
Segment with a single-level hierarchy (one posting holding everything), so
query.py never branches on unit type.
"""

from __future__ import annotations

import numpy as np

from ..core.types import HierarchyLevel, Posting, PostingId, VectorId
from ..core.config import IndexConfig
from ..core.metrics import Counters
from .postings import assign, fetch_posting
from .hierarchy import build_hierarchy


class Segment:
    """A single searchable index unit."""

    def __init__(
        self,
        levels: list[HierarchyLevel],
        postings: list[Posting],
        vectors: np.ndarray,          # float32 (N_global, D) — shared view
        tombstones: set[VectorId] | None = None,
    ) -> None:
        self.levels = levels
        self.postings = postings          # index == PostingId
        self.vectors = vectors
        self.tombstones: set[VectorId] = tombstones or set()

    @property
    def n_postings(self) -> int:
        return len(self.postings)

    @property
    def n_vectors(self) -> int:
        return sum(len(p.owned) for p in self.postings)

    def clone(self) -> Segment:
        """Create an independent copy of this segment with distinct levels and postings."""
        new_levels = [
            HierarchyLevel(
                centroids=lvl.centroids.copy(),
                children=[c.copy() for c in lvl.children],
            )
            for lvl in self.levels
        ]
        new_postings = [
            Posting(
                id=p.id,
                centroid=p.centroid.copy(),
                members=p.members.copy(),
                owned=p.owned.copy(),
            )
            for p in self.postings
        ]
        return Segment(
            levels=new_levels,
            postings=new_postings,
            vectors=self.vectors,
            tombstones=set(self.tombstones),
        )

    # ------------------------------------------------------------------
    # Factory: build a sealed Segment from a set of vector ids
    # ------------------------------------------------------------------
    @classmethod
    def build(
        cls,
        vids: np.ndarray,         # int64 vector ids to include
        vectors: np.ndarray,      # float32 (N_global, D)
        cfg: IndexConfig,
        counters: Counters,
        tombstones: set[VectorId] | None = None,
    ) -> "Segment":
        """Cluster vids and build the full hierarchy."""
        X = vectors[vids].astype(np.float32)
        N = len(vids)
        D = vectors.shape[1]

        n_leaves = max(1, round(N / cfg.leaf_size))

        from ..core.clustering import balanced_kmeans
        if n_leaves == 1:
            centroids = X.mean(axis=0, keepdims=True).astype(np.float32)
            labels = np.zeros(N, dtype=np.int64)
        else:
            centroids, labels, _ = balanced_kmeans(
                X, n_leaves, seed=cfg.seed, counters=counters
            )

        # Build postings (with optional overlap)
        raw_postings, bloat = assign(
            X, centroids,
            delta=cfg.delta,
            rng_rule=cfg.rng_rule,
            counters=counters,
        )

        # Remap local indices → global VectorIds
        postings: list[Posting] = []
        for p in raw_postings:
            global_members = vids[p.members]
            global_owned   = vids[p.owned]
            postings.append(Posting(
                id=p.id,
                centroid=p.centroid,
                members=global_members,
                owned=global_owned,
            ))

        # Build routing hierarchy above leaf centroids
        levels = build_hierarchy(
            leaf_centroids=centroids,
            branching_factor=cfg.branching_factor,
            seed=cfg.seed,
            depth=cfg.depth,
            counters=counters,
        )

        return cls(levels=levels, postings=postings, vectors=vectors,
                   tombstones=tombstones or set())

    @classmethod
    def flat(
        cls,
        vids: np.ndarray,
        vectors: np.ndarray,
        tombstones: set[VectorId] | None = None,
    ) -> "Segment":
        """Flat single-posting segment for the MemTable.

        One posting owns all vids; one leaf level with a single centroid.
        """
        D = vectors.shape[1]
        if len(vids) == 0:
            centroid = np.zeros(D, dtype=np.float32)
        else:
            centroid = vectors[vids].mean(axis=0).astype(np.float32)

        posting = Posting(
            id=0,
            centroid=centroid,
            members=vids.astype(np.int64),
            owned=vids.astype(np.int64),
        )
        leaf_level = HierarchyLevel(centroids=centroid[np.newaxis], children=[])
        return cls(
            levels=[leaf_level],
            postings=[posting],
            vectors=vectors,
            tombstones=tombstones or set(),
        )

    # ------------------------------------------------------------------
    # Tombstone helpers
    # ------------------------------------------------------------------
    def mark_deleted(self, vid: VectorId) -> None:
        self.tombstones.add(vid)

    def is_alive(self, vid: VectorId) -> bool:
        return vid not in self.tombstones
