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
    ) -> None:
        self._vectors = vectors
        self._cfg_index = cfg_index
        self._cfg_maintain = cfg_maintain
        self._counters = counters

        self._memtable: list[VectorId] = []          # buffered, not yet sealed
        self._sealed: list[Segment] = []             # sealed segments
        self._tombstones: set[VectorId] = set()      # global tombstone set

    # ------------------------------------------------------------------
    def insert(self, vid: VectorId, vec: np.ndarray) -> None:
        self._memtable.append(vid)
        if len(self._memtable) >= self._cfg_maintain.memtable_limit:
            self._flush()

    def delete(self, vid: VectorId) -> None:
        self._tombstones.add(vid)
        # propagate to sealed segments
        for seg in self._sealed:
            seg.mark_deleted(vid)

    def maintain(self) -> None:
        """Compact sealed segments if threshold reached."""
        if len(self._sealed) >= self._cfg_maintain.compaction_threshold:
            self._compact()

    def searchable_units(self) -> list[Segment]:
        units: list[Segment] = list(self._sealed)
        if self._memtable:
            vids = np.array(self._memtable, dtype=np.int64)
            mem_seg = Segment.flat(vids, self._vectors,
                                   tombstones=set(self._tombstones))
            units.append(mem_seg)
        return units

    def stats(self) -> dict:
        return {
            "strategy": "outofplace",
            "memtable_size": len(self._memtable),
            "sealed_segments": len(self._sealed),
            "compaction_events": self._counters.compaction_events,
            "vectors_reclustered": self._counters.vectors_reclustered,
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
        if len(live_vids) == 0:
            self._memtable = []
            return
        seg = Segment.build(
            live_vids, self._vectors, self._cfg_index, self._counters,
            tombstones=set()
        )
        self._sealed.append(seg)
        self._memtable = []

    def _compact(self) -> None:
        """Union live vectors across all sealed segments; re-cluster from scratch."""
        live_vids_set: set[int] = set()
        for seg in self._sealed:
            for p in seg.postings:
                for vid in p.owned:
                    if seg.is_alive(int(vid)):
                        live_vids_set.add(int(vid))

        if not live_vids_set:
            self._sealed = []
            self._counters.compaction_events += 1
            return

        live_vids = np.array(sorted(live_vids_set), dtype=np.int64)
        n = len(live_vids)

        new_seg = Segment.build(
            live_vids, self._vectors, self._cfg_index, self._counters,
            tombstones=set()
        )
        self._counters.compaction_events += 1
        self._counters.vectors_reclustered += n
        self._sealed = [new_seg]
