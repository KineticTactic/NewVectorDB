"""maintain/outofplace.py — out-of-place maintenance strategy.

DESIGN §6.6:
  - Buffer inserts in a MemTable (flat segment, exhaustively scanned).
  - At memtable_limit vectors flush to a new sealed Segment.
  - At compaction_threshold sealed segments compact to one.
  - Records: compaction_events, vectors_reclustered, segment count.
"""

from __future__ import annotations

import numpy as np

from ..core.types import VectorId
from ..core.config import IndexConfig, MaintainConfig
from ..core.metrics import Counters
from ..index.segment import Segment
from .base import IndexMaintainer


class OutOfPlaceMaintainer(IndexMaintainer):

    def __init__(
        self,
        vectors: np.ndarray,          # float32 (N_global, D) — shared
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

        self._memtable: list[VectorId] = []          # buffered, not yet sealed
        self._tombstones: set[VectorId] = set()      # global tombstone set

        if initial_segment is not None:
            self._sealed: list[Segment] = [initial_segment.clone()]
            self._tombstones.update(self._sealed[0].tombstones)
        elif initial_vids is not None and len(initial_vids) > 0:
            seg = Segment.build(
                initial_vids, self._vectors, self._cfg_index, self._counters,
                tombstones=set()
            )
            self._sealed = [seg]
        else:
            self._sealed = []

    # ------------------------------------------------------------------
    def insert(self, vid: VectorId, vec: np.ndarray) -> None:
        # If the vector was previously marked deleted, re-inserting revives it
        self._tombstones.discard(vid)
        self._memtable.append(vid)
        if len(self._memtable) >= self._cfg_maintain.memtable_limit:
            self._flush()

    def delete(self, vid: VectorId) -> None:
        """Mark a vector as deleted (tombstone). Idempotent."""
        if vid in self._tombstones:
            return
        self._tombstones.add(vid)
        # Propagate to sealed segments
        for seg in self._sealed:
            seg.mark_deleted(vid)

    def maintain(self) -> None:
        """Perform maintenance: compact sealed segments on segment count or tombstone ratio."""
        if not self._sealed:
            return

        total_owned = sum(len(p.owned) for seg in self._sealed for p in seg.postings)
        total_dead = sum(len(seg.tombstones) for seg in self._sealed)
        tombstone_frac = (total_dead / total_owned) if total_owned > 0 else 0.0

        should_compact = (
            len(self._sealed) >= self._cfg_maintain.compaction_threshold
            or (
                total_dead > 0
                and tombstone_frac >= self._cfg_maintain.tombstone_rebuild_frac
            )
        )
        if should_compact:
            self._compact()

    def searchable_units(self) -> list[Segment]:
        units: list[Segment] = list(self._sealed)
        if self._memtable:
            vids = np.array(self._memtable, dtype=np.int64)
            mem_seg = Segment.flat(vids, self._vectors,
                                   tombstones=set(self._tombstones))
            units.append(mem_seg)
        return units

    def flush(self) -> None:
        """Flush pending MemTable entries to a new sealed segment."""
        self._flush()

    def stats(self) -> dict:
        actual_depth = max((len(s.levels) for s in self._sealed), default=0)
        total_postings = sum(len(s.postings) for s in self._sealed)
        if self._memtable:
            total_postings += 1
            if actual_depth == 0:
                actual_depth = 1
        return {
            "strategy": "outofplace",
            "memtable_size": len(self._memtable),
            "sealed_segments": len(self._sealed),
            "unit_count": len(self.searchable_units()),
            "depth": actual_depth,
            "n_postings": total_postings,
            "compaction_events": self._counters.compaction_events,
            "vectors_reclustered": self._counters.vectors_reclustered,
            "flushes": self._counters.flushes,
            "hierarchy_rebuilds": self._counters.hierarchy_rebuilds,
            "replicas_added": self._counters.replicas_added,
            "replicas_removed": self._counters.replicas_removed,
            "membership_vectors_recomputed": self._counters.membership_vectors_recomputed,
            "tombstones": len(self._tombstones),
        }

    # ------------------------------------------------------------------
    def _flush(self) -> None:
        """Flush MemTable to a new sealed Segment."""
        if not self._memtable:
            return
        vids = np.array(self._memtable, dtype=np.int64)
        # Remove tombstoned vectors before sealing
        live_mask = np.array([v not in self._tombstones for v in vids], dtype=bool)
        live_vids = vids[live_mask]

        # Purge any dead vectors that were pending in memtable from _tombstones
        dead_in_mem = set(int(v) for v in vids[~live_mask])
        self._tombstones -= dead_in_mem
        self._memtable = []

        if len(live_vids) == 0:
            return

        seg = Segment.build(
            live_vids, self._vectors, self._cfg_index, self._counters,
            tombstones=set()
        )
        self._sealed.append(seg)
        self._counters.flushes += 1
        replicas_in_seg = sum(len(p.members) - len(p.owned) for p in seg.postings)
        self._counters.replicas_added += replicas_in_seg

    def _compact(self) -> None:
        """Union live vectors across all sealed segments; re-cluster from scratch.

        Atomically rebuilds all sealed segments into a single unified segment,
        purging all tombstoned vectors and reclaiming dead storage.
        """
        saved_sealed = [s.clone() for s in self._sealed]
        saved_tombstones = set(self._tombstones)
        saved_counters = self._counters.snapshot()

        try:
            live_vids_set: set[int] = set()
            for seg in self._sealed:
                for p in seg.postings:
                    for vid in p.owned:
                        vi = int(vid)
                        if seg.is_alive(vi) and vi not in self._tombstones:
                            live_vids_set.add(vi)

            if not live_vids_set:
                self._sealed = []
                self._counters.compaction_events += 1
                # Preserve only tombstones for vectors still pending in memtable
                memtable_set = set(int(v) for v in self._memtable)
                self._tombstones = self._tombstones & memtable_set
                return

            live_vids = np.array(sorted(live_vids_set), dtype=np.int64)
            n = len(live_vids)

            new_seg = Segment.build(
                live_vids, self._vectors, self._cfg_index, self._counters,
                tombstones=set()
            )
            self._counters.compaction_events += 1
            self._counters.vectors_reclustered += n
            self._counters.hierarchy_rebuilds += 1
            self._sealed = [new_seg]

            # Tombstone garbage collection: tombstones that belonged to the old sealed
            # segments are now physically purged from the index. Only tombstones for
            # vectors currently pending in memtable (if any) need to be retained.
            memtable_set = set(int(v) for v in self._memtable)
            self._tombstones = self._tombstones & memtable_set
        except Exception:
            self._sealed = saved_sealed
            self._tombstones = saved_tombstones
            for name, value in vars(saved_counters).items():
                setattr(self._counters, name, value)
            raise
