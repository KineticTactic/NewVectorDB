"""maintain/inplace.py — in-place maintenance strategy.

DESIGN §6.6: one segment mutated in place.
Insert → nearest leaf → append → split if over threshold.
After split: neighbourhood reassignment (not local-only — §2 critical rule).
Promote centroids upward; recurse on overflow; add level if root splits.
Records: splits, vectors_reassigned, npa_violations_detected, depth.
"""

from __future__ import annotations

import numpy as np

from ..core.types import VectorId, Posting
from ..core.config import IndexConfig, MaintainConfig
from ..core.metrics import Counters
from ..core.distance import sim_one_to_many
from ..core.clustering import balanced_kmeans
from ..index.hierarchy import build_hierarchy, _stable_top_k
from ..index.postings import posting_add_vector
from ..index.segment import Segment
from .base import IndexMaintainer


class InPlaceMaintainer(IndexMaintainer):

    def __init__(
        self,
        vectors: np.ndarray,
        cfg_index: IndexConfig,
        cfg_maintain: MaintainConfig,
        counters: Counters,
        initial_vids: np.ndarray | None = None,
        initial_segment: Segment | None = None,
    ) -> None:
        self._vectors = vectors
        self._cfg_index = cfg_index
        self._cfg_maintain = cfg_maintain
        self._counters = counters
        self._tombstones: set[VectorId] = set()

        if initial_segment is not None:
            self._segment = initial_segment.clone()
        elif initial_vids is not None and len(initial_vids) > 0:
            self._segment = Segment.build(
                initial_vids, vectors, cfg_index, counters, tombstones=set()
            )
        else:
            from ..core.types import HierarchyLevel
            centroid = np.zeros(cfg_index.dim, dtype=np.float32)
            empty_posting = Posting(
                id=0, centroid=centroid,
                members=np.array([], dtype=np.int64),
                owned=np.array([], dtype=np.int64),
            )
            leaf_level = HierarchyLevel(centroids=centroid[np.newaxis], children=[])
            self._segment = Segment(
                levels=[leaf_level], postings=[empty_posting],
                vectors=vectors, tombstones=set()
            )

    # ------------------------------------------------------------------
    def insert(self, vid: VectorId, vec: np.ndarray) -> None:
        leaf_level = self._segment.levels[-1]
        sims = sim_one_to_many(vec, leaf_level.centroids, self._counters, "routing")
        pid = int(_stable_top_k(-sims, 1)[0])
        p = self._segment.postings[pid]
        p = posting_add_vector(p, vid, owned=True)
        self._segment.postings[pid] = p
        if len(p.owned) > self._cfg_maintain.split_threshold:
            self._split(pid)

    def delete(self, vid: VectorId) -> None:
        self._tombstones.add(vid)
        self._segment.mark_deleted(vid)
        for pid, p in enumerate(self._segment.postings):
            if np.any(p.owned == vid):
                n_owned = len(p.owned)
                n_tomb = int(np.sum(np.isin(p.owned, list(self._tombstones))))
                if n_owned > 0 and n_tomb / n_owned >= self._cfg_maintain.tombstone_rebuild_frac:
                    self._rebuild_posting(pid)
                break

    def maintain(self) -> None:
        pass  # maintenance is eager

    def searchable_units(self) -> list[Segment]:
        return [self._segment]

    def stats(self) -> dict:
        return {
            "strategy": "inplace",
            "depth": len(self._segment.levels),
            "n_postings": len(self._segment.postings),
            "splits": self._counters.splits,
            "vectors_reassigned": self._counters.vectors_reassigned,
            "npa_violations_detected": self._counters.npa_violations_detected,
            "hierarchy_rebuilds": self._counters.hierarchy_rebuilds,
            "tombstones": len(self._tombstones),
        }

    # ------------------------------------------------------------------
    # Split
    # ------------------------------------------------------------------

    def _split(self, pid: int) -> None:
        p = self._segment.postings[pid]
        owned_vids = p.owned
        if len(owned_vids) < 2:
            return
        orig_centroid = p.centroid.copy()
        X_owned = self._vectors[owned_vids].astype(np.float32)
        try:
            centroids2, labels2, _ = balanced_kmeans(
                X_owned, 2,
                seed=self._cfg_index.seed + self._counters.splits,
                counters=self._counters
            )
        except Exception:
            return

        new_pid = len(self._segment.postings)
        own0 = owned_vids[labels2 == 0]
        own1 = owned_vids[labels2 == 1]

        # Distribute replica members
        replica_members = p.members[~np.isin(p.members, p.owned)]
        if len(replica_members) > 0:
            X_rep = self._vectors[replica_members].astype(np.float32)
            s0 = sim_one_to_many(centroids2[0], X_rep, self._counters, "build")
            s1 = sim_one_to_many(centroids2[1], X_rep, self._counters, "build")
            rep0 = replica_members[s0 >= s1]
            rep1 = replica_members[s0 < s1]
        else:
            rep0 = rep1 = np.array([], dtype=np.int64)

        p0 = Posting(pid, centroids2[0].astype(np.float32),
                     np.concatenate([own0, rep0]).astype(np.int64), own0)
        p1 = Posting(new_pid, centroids2[1].astype(np.float32),
                     np.concatenate([own1, rep1]).astype(np.int64), own1)
        self._segment.postings[pid] = p0
        self._segment.postings.append(p1)
        self._counters.splits += 1

        leaf_level = self._segment.levels[-1]
        new_row = centroids2[1].astype(np.float32)[np.newaxis]
        leaf_level.centroids = np.vstack(
            [leaf_level.centroids, new_row]
        ).astype(np.float32)
        leaf_level.centroids[pid] = centroids2[0].astype(np.float32)

        self._reassign_neighbourhood(orig_centroid, pid, new_pid)
        self._promote_split(pid, new_pid)

    def _reassign_neighbourhood(self, orig_centroid: np.ndarray, pid: int, new_pid: int) -> None:
        n_nbrs = self._cfg_maintain.n_reassign_neighbours
        leaf_centroids = self._segment.levels[-1].centroids
        K = len(leaf_centroids)
        n_search = min(n_nbrs + 2, K)
        sims_orig = sim_one_to_many(orig_centroid, leaf_centroids, self._counters, "build")
        sims_c0 = sim_one_to_many(leaf_centroids[pid], leaf_centroids, self._counters, "build")
        sims_c1 = sim_one_to_many(leaf_centroids[new_pid], leaf_centroids, self._counters, "build")
        top_pids = set(_stable_top_k(-sims_orig, n_search).tolist())
        top_pids.update(_stable_top_k(-sims_c0, n_search).tolist())
        top_pids.update(_stable_top_k(-sims_c1, n_search).tolist())
        affected = sorted(top_pids | {pid, new_pid})

        all_vids_set: set[int] = set()
        for ap in affected:
            if ap < len(self._segment.postings):
                for v in self._segment.postings[ap].owned:
                    all_vids_set.add(int(v))
        if not all_vids_set:
            return

        all_vids = np.array(sorted(all_vids_set), dtype=np.int64)
        aff_arr = np.array(affected, dtype=np.int64)
        aff_centroids = leaf_centroids[aff_arr]
        X_aff = self._vectors[all_vids].astype(np.float32)
        sims2 = X_aff @ aff_centroids.T
        self._counters.add_distance("build", len(all_vids) * len(affected))
        new_owner_pid = aff_arr[np.argmax(sims2, axis=1)]

        old_owner: dict[int, int] = {}
        for ap in affected:
            if ap < len(self._segment.postings):
                for v in self._segment.postings[ap].owned:
                    old_owner[int(v)] = ap

        new_owned: dict[int, list[int]] = {ap: [] for ap in affected}
        n_reassigned = 0
        for i, vid in enumerate(all_vids):
            vid_int, np_int = int(vid), int(new_owner_pid[i])
            if old_owner.get(vid_int, -1) != np_int:
                n_reassigned += 1
            if np_int in new_owned:
                new_owned[np_int].append(vid_int)
        self._counters.vectors_reassigned += n_reassigned

        for ap in affected:
            if ap >= len(self._segment.postings):
                continue
            p = self._segment.postings[ap]
            new_own_arr = np.array(new_owned.get(ap, []), dtype=np.int64)
            replicas = p.members[~np.isin(p.members, p.owned)]
            self._segment.postings[ap] = Posting(
                p.id, p.centroid,
                np.concatenate([new_own_arr, replicas]).astype(np.int64),
                new_own_arr
            )

    def _promote_split(self, pid: int, new_pid: int) -> None:
        if len(self._segment.levels) == 1:
            from ..core.types import HierarchyLevel
            lc = self._segment.levels[-1].centroids
            root_c = lc.mean(axis=0, keepdims=True).astype(np.float32)
            root_level = HierarchyLevel(
                centroids=root_c,
                children=[np.arange(len(lc), dtype=np.int64)]
            )
            self._segment.levels.insert(0, root_level)
            return

        routing = self._segment.levels[:-1]
        bottom = routing[-1]
        for node_idx in range(len(bottom.children)):
            if np.any(bottom.children[node_idx] == pid):
                bottom.children[node_idx] = np.append(
                    bottom.children[node_idx], new_pid
                ).astype(np.int64)
                if len(bottom.children[node_idx]) > self._cfg_index.branching_factor * 2:
                    self._rebuild_hierarchy()
                break

    def _rebuild_hierarchy(self) -> None:
        leaf_centroids = self._segment.levels[-1].centroids
        new_levels = build_hierarchy(
            leaf_centroids,
            branching_factor=self._cfg_index.branching_factor,
            seed=self._cfg_index.seed,
            depth=self._cfg_index.depth,
            counters=self._counters,
        )
        self._counters.hierarchy_rebuilds += 1
        self._segment.levels = new_levels

    def _rebuild_posting(self, pid: int) -> None:
        p = self._segment.postings[pid]
        tomb = list(self._tombstones)
        live_owned = p.owned[~np.isin(p.owned, tomb)]
        live_members = p.members[~np.isin(p.members, tomb)]
        new_p = Posting(p.id, p.centroid, live_members, live_owned)
        self._segment.postings[pid] = new_p
