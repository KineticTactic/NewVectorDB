"""harness/live.py — continuous live-churn monitor for the in-place maintainer.

This module backs ``scripts/live_churn.py``.  It runs an *open-ended* balanced
churn cycle (one insert, then one delete) against the real
:class:`~vecstudy.maintain.inplace.InPlaceMaintainer`, recomputes recall@k
against the *currently live* vectors after **every** operation, prints one line
per operation, and can be stopped with Ctrl+C.

It is a monitoring/experimentation tool, not a new index.  It reuses the
production maintenance path, the production query path (``index.query.search``),
the project metric (inner product, ``distance = 1 - similarity``), the dynamic
ground-truth computation and the two independent auditors.  No maintenance or
query logic is reimplemented or altered.

Design notes
------------
* **Vector ids are rows** of one float32 matrix shared by the maintainer and its
  segment.  New ids are assigned monotonically and never reused.  Storage grows
  through :class:`GrowingVectorStore`, which reallocates the matrix and re-binds
  *both* the maintainer and its searchable units (see :meth:`GrowingVectorStore.bind`);
  it never swaps the array in one holder while leaving a stale reference in another.
* **New vectors** are copies of a uniformly random *initial* base vector (row in
  ``[0, initial_size)``), chosen by the seeded churn RNG.  Copying an existing
  value is acceptable for this script and keeps the live distribution stationary;
  each copy receives a distinct, never-reused id.
* **Recall** is recomputed from scratch after every mutation: exact brute-force
  top-k over only the live vectors (tombstones excluded), compared against the
  production search.  Ground truth is never cached or reused across operations.
* **Events** (splits/merges) are read from the real maintenance counters *and*
  observed directly by :class:`ObservedInPlaceMaintainer`, so ids come from the
  actual maintenance call, not inferred from a leaf-count delta.
* **Bounded output/history.**  One line per operation; a compact cumulative
  summary every ``summary_every`` ops; only a bounded window of recent recalls is
  retained.  Unbounded in memory by design: unique ids, tombstones and the
  vector matrix all grow by one row per insert (they are never freed).
"""

from __future__ import annotations

import argparse
import math
import sys
import time
from collections import deque
from dataclasses import dataclass, field
from typing import TextIO

import numpy as np

from ..core.config import IndexConfig, MaintainConfig, QueryConfig
from ..core.metrics import Counters
from ..core.types import VectorId
from ..index.query import search
from .churn import audit_maintainer, audit_membership, audit_outofplace_membership
from .datasets import make_synthetic, recall_at_k
from ..maintain.base import IndexMaintainer
from ..maintain.inplace import InPlaceMaintainer
from ..maintain.outofplace import OutOfPlaceMaintainer

# ---------------------------------------------------------------------------
# Reproducible, growing backing store
# ---------------------------------------------------------------------------

class GrowingVectorStore:
    """Append-only float32 matrix with a re-bindable backing array.

    Vector ids are row indices.  numpy cannot resize an array in place, so growth
    allocates a larger matrix and copies the live prefix.  Because the maintainer
    and its segment each hold their own reference to the array, every holder must
    be re-bound to :attr:`data` after a growth — that is what :meth:`bind` does.
    """

    def __init__(self, initial: np.ndarray, *, grow_factor: float = 2.0) -> None:
        data = np.array(initial, dtype=np.float32, copy=True)
        if data.ndim != 2:
            raise ValueError(f"initial vectors must be 2-D, got shape {data.shape}")
        if len(data) < 1:
            raise ValueError("initial vector pool must be non-empty")
        self._data = np.ascontiguousarray(data)
        self._size = len(self._data)
        self._grow_factor = float(grow_factor)
        self._n_grows = 0

    @property
    def data(self) -> np.ndarray:
        """The current backing matrix (reference changes after a growth)."""
        return self._data

    @property
    def size(self) -> int:
        """Number of valid rows, i.e. the next fresh id."""
        return self._size

    @property
    def dim(self) -> int:
        return int(self._data.shape[1])

    @property
    def capacity(self) -> int:
        return int(self._data.shape[0])

    @property
    def n_grows(self) -> int:
        return self._n_grows

    def vector(self, vid: VectorId) -> np.ndarray:
        if not (0 <= int(vid) < self._size):
            raise IndexError(f"vector id {vid} out of range [0, {self._size})")
        return self._data[int(vid)]

    def _ensure_capacity(self, n_rows: int) -> None:
        if n_rows <= len(self._data):
            return
        target = max(n_rows, int(math.ceil(len(self._data) * self._grow_factor)))
        grown = np.empty((target, self.dim), dtype=np.float32)
        grown[: self._size] = self._data[: self._size]
        self._data = grown
        self._n_grows += 1

    def append(self, vec: np.ndarray) -> VectorId:
        """Append *vec* and return its fresh, never-reused id."""
        v = np.asarray(vec, dtype=np.float32)
        if v.shape != (self.dim,):
            raise ValueError(f"expected a ({self.dim},) vector, got {v.shape}")
        self._ensure_capacity(self._size + 1)
        self._data[self._size] = v
        self._size += 1
        return self._size - 1

    def bind(self, maint: IndexMaintainer) -> None:
        """Re-point *maint* and every searchable unit at the current array.

        Called after a growth so no object keeps a stale (too-small) reference.
        """
        maint._vectors = self._data
        if hasattr(maint, "_sealed"):
            for seg in maint._sealed:
                seg.vectors = self._data
        for unit in maint.searchable_units():
            unit.vectors = self._data


# ---------------------------------------------------------------------------
# Structural-event observation
# ---------------------------------------------------------------------------

@dataclass
class StructuralEvent:
    """One real split, merge, flush, or compaction, captured as it is performed."""
    kind: str                       # "SPLIT" | "MERGE" | "FLUSH" | "COMPACT"
    pid: int | None = None          # split: the posting that was split
    new_pid: int | None = None      # split: the posting appended
    survivor: int | None = None     # merge: surviving posting
    removed: int | None = None      # merge: removed posting
    vectors_reassigned: int = 0
    hierarchy_rebuilds: int = 0
    segment_count: int = 0          # out-of-place: sealed segment count after event
    vectors_affected: int = 0       # out-of-place: vectors flushed or reclustered

    def format(self) -> str:
        if self.kind == "SPLIT":
            return (f"  SPLIT pid={self.pid} new_pid={self.new_pid} "
                    f"vectors_reassigned={self.vectors_reassigned} "
                    f"hierarchy_rebuilds={self.hierarchy_rebuilds}")
        if self.kind == "MERGE":
            return (f"  MERGE survivor={self.survivor} removed={self.removed} "
                    f"vectors_reassigned={self.vectors_reassigned} "
                    f"hierarchy_rebuilds={self.hierarchy_rebuilds}")
        if self.kind == "FLUSH":
            return (f"  FLUSH sealed_segments={self.segment_count} "
                    f"vectors={self.vectors_affected}")
        if self.kind == "COMPACT":
            return (f"  COMPACT sealed_segments={self.segment_count} "
                    f"vectors_reclustered={self.vectors_affected} "
                    f"hierarchy_rebuilds={self.hierarchy_rebuilds}")
        return f"  {self.kind}"


class ObservedInPlaceMaintainer(InPlaceMaintainer):
    """Real maintainer that records its structural events as they happen.

    Only ``_split`` and ``_merge_pair`` are overridden, purely to observe: each
    calls the production implementation unchanged and appends an event only when
    the corresponding counter actually advanced (a no-op attempt, e.g. a split of
    an indivisible posting, is not recorded).  It does not alter maintenance.
    """

    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.events: list[StructuralEvent] = []

    def drain_events(self) -> list[StructuralEvent]:
        events, self.events = self.events, []
        return events

    def _split(self, pid: int) -> None:
        new_pid = len(self._segment.postings)
        splits = self._counters.splits
        reassigned = self._counters.vectors_reassigned
        rebuilds = self._counters.hierarchy_rebuilds
        super()._split(pid)
        if self._counters.splits > splits:
            self.events.append(StructuralEvent(
                "SPLIT", pid=int(pid), new_pid=int(new_pid),
                vectors_reassigned=self._counters.vectors_reassigned - reassigned,
                hierarchy_rebuilds=self._counters.hierarchy_rebuilds - rebuilds,
            ))

    def _merge_pair(self, pid_a: int, pid_b: int) -> None:
        merges = self._counters.merges
        reassigned = self._counters.merge_vectors_reassigned
        rebuilds = self._counters.merge_hierarchy_rebuilds
        super()._merge_pair(pid_a, pid_b)
        if self._counters.merges > merges:
            survivor, removed = (pid_a, pid_b) if pid_a < pid_b else (pid_b, pid_a)
            self.events.append(StructuralEvent(
                "MERGE", survivor=int(survivor), removed=int(removed),
                vectors_reassigned=(
                    self._counters.merge_vectors_reassigned - reassigned
                ),
                hierarchy_rebuilds=(
                    self._counters.merge_hierarchy_rebuilds - rebuilds
                ),
            ))


class ObservedOutOfPlaceMaintainer(OutOfPlaceMaintainer):
    """Real out-of-place maintainer that records its structural events as they happen.

    Only ``_flush`` and ``_compact`` are overridden, purely to observe: each
    calls the production implementation unchanged and appends an event only when
    the corresponding counter actually advanced.
    """

    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.events: list[StructuralEvent] = []

    def drain_events(self) -> list[StructuralEvent]:
        events, self.events = self.events, []
        return events

    def _flush(self) -> None:
        flushes = self._counters.flushes
        mem_len = len(self._memtable)
        super()._flush()
        if self._counters.flushes > flushes:
            self.events.append(StructuralEvent(
                "FLUSH",
                segment_count=len(self._sealed),
                vectors_affected=mem_len,
            ))

    def _compact(self) -> None:
        compactions = self._counters.compaction_events
        reclustered = self._counters.vectors_reclustered
        rebuilds = self._counters.hierarchy_rebuilds
        super()._compact()
        if self._counters.compaction_events > compactions:
            self.events.append(StructuralEvent(
                "COMPACT",
                segment_count=len(self._sealed),
                vectors_affected=self._counters.vectors_reclustered - reclustered,
                hierarchy_rebuilds=self._counters.hierarchy_rebuilds - rebuilds,
            ))


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class LiveChurnConfig:
    """Resolved, validated monitor configuration."""
    strategy: str = "inplace"
    initial_size: int = 300
    k: int = 10
    query_count: int = 32
    dim: int = 128
    dataset: str = "synthetic"
    depth: int | None = None
    leaf_size: int = 40
    branching_factor: int = 16
    split_threshold: int = 40
    merge_threshold: int | None = None
    delta: float | None = None
    interval: float = 0.2
    seed: int = 7
    audit_every: int = 1
    summary_every: int = 50
    n_postings: int = 1
    beam: int = 8
    continue_on_audit_failure: bool = False
    memtable_limit: int = 50
    compaction_threshold: int = 4
    tombstone_rebuild_frac: float = 0.20

    def __post_init__(self) -> None:
        if self.strategy not in ("inplace", "outofplace"):
            raise ValueError(
                f"unsupported strategy {self.strategy!r}; must be 'inplace' or 'outofplace'"
            )
        if self.dataset != "synthetic":
            raise ValueError(
                f"unsupported dataset {self.dataset!r}; this monitor supports "
                "'synthetic' (see the report's limitations)"
            )
        if self.initial_size < 1:
            raise ValueError("initial_size must be >= 1")
        if self.k < 1:
            raise ValueError("k must be >= 1")
        if self.query_count < 1:
            raise ValueError("query_count must be >= 1")
        if self.dim < 2:
            raise ValueError("dim must be >= 2")
        if self.depth is not None and self.depth < 1:
            raise ValueError("depth must be >= 1 or None")
        if self.leaf_size < 1:
            raise ValueError("leaf_size must be >= 1")
        if self.branching_factor < 2:
            raise ValueError("branching_factor must be >= 2")
        if self.split_threshold < 1:
            raise ValueError("split_threshold must be >= 1")
        if self.merge_threshold is not None and self.merge_threshold < 0:
            raise ValueError("merge_threshold must be >= 0 or None")
        if self.delta is not None and self.delta < 0.0:
            raise ValueError("delta must be >= 0 or None")
        if self.interval < 0.0:
            raise ValueError("interval must be >= 0")
        if self.audit_every < 0:
            raise ValueError("audit_every must be >= 0 (0 disables audits)")
        if self.summary_every < 0:
            raise ValueError("summary_every must be >= 0 (0 disables)")
        if self.n_postings < 1:
            raise ValueError("n_postings must be >= 1")
        if self.beam < 1:
            raise ValueError("beam must be >= 1")
        if self.memtable_limit < 1:
            raise ValueError("memtable_limit must be >= 1")
        if self.compaction_threshold < 2:
            raise ValueError("compaction_threshold must be >= 2")
        if not (0.0 <= self.tombstone_rebuild_frac <= 1.0):
            raise ValueError("tombstone_rebuild_frac must be in [0.0, 1.0]")

    def to_dict(self) -> dict:
        return {
            "strategy": self.strategy,
            "dataset": self.dataset, "dim": self.dim, "seed": self.seed,
            "initial_size": self.initial_size, "k": self.k,
            "query_count": self.query_count, "depth": self.depth,
            "leaf_size": self.leaf_size, "branching_factor": self.branching_factor,
            "split_threshold": self.split_threshold,
            "merge_threshold": self.merge_threshold, "delta": self.delta,
            "interval": self.interval, "audit_every": self.audit_every,
            "summary_every": self.summary_every, "n_postings": self.n_postings,
            "beam": self.beam,
            "continue_on_audit_failure": self.continue_on_audit_failure,
            "memtable_limit": self.memtable_limit,
            "compaction_threshold": self.compaction_threshold,
            "tombstone_rebuild_frac": self.tombstone_rebuild_frac,
        }


# ---------------------------------------------------------------------------
# Per-operation record
# ---------------------------------------------------------------------------

@dataclass
class OperationRecord:
    index: int
    op: str                         # "INSERT" | "DELETE"
    vid: int
    live: int
    leaves: int
    splits_total: int
    merges_total: int
    recall: float | None
    elapsed_total: float
    elapsed_op: float
    source_row: int | None = None   # INSERT: base row that was copied
    events: list[StructuralEvent] = field(default_factory=list)
    deltas: dict[str, int] = field(default_factory=dict)
    npa: str = "SKIP"
    membership: str = "SKIP"
    detail: str = ""
    failed: bool = False
    flushes_total: int = 0
    compactions_total: int = 0
    sealed_segments: int = 0
    memtable_size: int = 0


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _fmt_elapsed(seconds: float) -> str:
    total_ms = max(0, int(round(seconds * 1000.0)))
    ms = total_ms % 1000
    s = (total_ms // 1000) % 60
    m = (total_ms // 60_000) % 60
    h = total_ms // 3_600_000
    return f"{h:02d}:{m:02d}:{s:02d}.{ms:03d}"


def parse_delta_arg(value: str) -> float | None:
    """argparse ``type`` for ``--delta``: a number, or 'none'/'off' to disable."""
    text = str(value).strip().lower()
    if text in ("", "none", "off", "null", "disable", "disabled"):
        return None
    try:
        return float(text)
    except ValueError as exc:
        raise argparse.ArgumentTypeError(
            f"delta must be a number or 'none'/'off', got {value!r}"
        ) from exc


# ---------------------------------------------------------------------------
# Monitor
# ---------------------------------------------------------------------------

class LiveChurnMonitor:
    """Open-ended balanced churn + per-operation recall monitoring.

    The default output stream is ``sys.stdout``; tests pass ``io.StringIO``.
    """

    def __init__(self, cfg: LiveChurnConfig, *, out: TextIO | None = None) -> None:
        self.cfg = cfg
        self._out = out if out is not None else sys.stdout

        self._cfg_index = IndexConfig(
            dim=cfg.dim, leaf_size=cfg.leaf_size,
            branching_factor=cfg.branching_factor, depth=cfg.depth,
            delta=cfg.delta, rng_rule=False, seed=cfg.seed,
        )
        self._cfg_maintain = MaintainConfig(
            strategy=cfg.strategy,
            split_threshold=cfg.split_threshold,
            merge_threshold=cfg.merge_threshold,
            memtable_limit=cfg.memtable_limit,
            compaction_threshold=cfg.compaction_threshold,
            tombstone_rebuild_frac=cfg.tombstone_rebuild_frac,
        )
        self._cfg_query = QueryConfig(
            k=cfg.k, beam_widths=(cfg.beam,), n_postings=cfg.n_postings,
        )

        # Reproducible dataset (base) and fixed evaluation queries.
        base = make_synthetic(n=cfg.initial_size, dim=cfg.dim, seed=cfg.seed)
        self._store = GrowingVectorStore(base)
        self._queries = make_synthetic(n=cfg.query_count, dim=cfg.dim,
                                       seed=cfg.seed + 1)

        self._counters = Counters()
        self._query_counters = Counters()
        initial_vids = np.arange(cfg.initial_size, dtype=np.int64)
        if cfg.strategy == "outofplace":
            self._maint = ObservedOutOfPlaceMaintainer(
                self._store.data, self._cfg_index, self._cfg_maintain,
                self._counters, initial_vids=initial_vids,
            )
        else:
            self._maint = ObservedInPlaceMaintainer(
                self._store.data, self._cfg_index, self._cfg_maintain,
                self._counters, initial_vids=initial_vids,
            )
        self._store.bind(self._maint)

        # Live bookkeeping.  ``_live`` is kept sorted ascending: fresh ids are
        # always the maximum, so an insert appends and a delete pops in place.
        self._live: list[int] = list(range(cfg.initial_size))
        self._live_set: set[int] = set(self._live)
        self._next_id = cfg.initial_size
        self._base_pool = cfg.initial_size

        # Seeded churn RNG (source-vector choice + delete-victim choice).
        self._rng = np.random.RandomState(cfg.seed)

        # Bounded history / running statistics.
        self._recent: deque[float] = deque(maxlen=max(1, cfg.summary_every or 50))
        self._recall_sum = 0.0
        self._recall_n = 0
        self._recall_best = -1.0
        self._recall_worst = 2.0
        self._last_recall: float | None = None
        self._last_audit = "NPA=SKIP MEMBERSHIP=SKIP"

        self._op_count = 0
        self._inserts = 0
        self._deletes = 0
        self._op_time_sum = 0.0
        self._t0 = time.perf_counter()
        self._stopped = False
        self._interrupted = False
        self._failed = False
        self._last_npa = "SKIP"
        self._last_membership = "SKIP"

    # -- output -----------------------------------------------------------

    def _print_line(self, text: str) -> None:
        print(text, file=self._out, flush=True)

    # -- recall -----------------------------------------------------------

    def _live_ground_truth(self) -> np.ndarray:
        """Exact brute-force top-k over the *live* vectors only (no cache).

        Metric matches production: float32 inner product.  Local offsets are
        mapped back to global ids.
        """
        live = np.array(self._live, dtype=np.int64)
        eff_k = min(self._cfg_query.k, len(live))
        gt = np.empty((len(self._queries), eff_k), dtype=np.int64)
        if eff_k == 0:
            return gt
        base = self._store.data[live].astype(np.float32)
        for i, q in enumerate(self._queries):
            sims = base @ q.astype(np.float32)
            top = np.argpartition(-sims, kth=eff_k - 1)[:eff_k]
            top = top[np.argsort(-sims[top], kind="stable")]
            gt[i] = live[top[:eff_k]]
        return gt

    def evaluate_recall(self) -> float | None:
        """Recall@k of the maintained index over the current live set.

        ``None`` when no live vectors remain (reported as ``N/A``).
        """
        if not self._live:
            return None
        gt = self._live_ground_truth()
        units = self._maint.searchable_units()
        results: list[list[int]] = []
        for q in self._queries:
            self._query_counters.reset_search()
            res = search(q, units, self._cfg_query, self._query_counters)
            results.append([int(v) for v, _ in res])
        return float(recall_at_k(results, gt, self._cfg_query.k))

    # -- churn ------------------------------------------------------------

    def _append_vector(self, vec: np.ndarray, vid: int) -> None:
        """Append to storage, re-bind holders, and insert into the maintainer."""
        if vid != self._store.size:
            raise AssertionError(
                f"id {vid} must be the next fresh id {self._store.size}"
            )
        self._store.append(vec)
        self._store.bind(self._maint)     # grow → re-bind maintainer + segment
        self._maint.insert(vid, self._store.data[vid])

    def _plan_insert(self) -> tuple[int, int]:
        """Reproducibly choose (source_row, fresh_vid); does not mutate state."""
        row = int(self._rng.randint(0, self._base_pool))
        return row, self._next_id

    def _choose_victim_index(self) -> int:
        """Index into ``_live`` of the delete victim (never the newest id).

        Excludes the last element, which is the id the current cycle just
        inserted, so the fresh vector gets a chance to participate.
        """
        n = len(self._live)
        if n <= 1:
            return 0
        return int(self._rng.randint(0, n - 1))

    def _next_op(self) -> str:
        if self._op_count % 2 == 0 or not self._live:
            return "INSERT"
        return "DELETE"

    def _counter_deltas(self, before: Counters, after: Counters) -> dict[str, int]:
        c = self._counters
        return {
            "splits": c.splits - before.splits,
            "merges": c.merges - before.merges,
            "flushes": c.flushes - before.flushes,
            "compactions": c.compaction_events - before.compaction_events,
            "vectors_reclustered": c.vectors_reclustered - before.vectors_reclustered,
            "hierarchy_rebuilds": c.hierarchy_rebuilds - before.hierarchy_rebuilds,
            "replicas_added": c.replicas_added - before.replicas_added,
            "replicas_removed": c.replicas_removed - before.replicas_removed,
            "vectors_reassigned": c.vectors_reassigned - before.vectors_reassigned,
            "merge_vectors_reassigned":
                c.merge_vectors_reassigned - before.merge_vectors_reassigned,
            "merge_candidates_examined":
                c.merge_candidates_examined - before.merge_candidates_examined,
            "membership_vectors_recomputed":
                c.membership_vectors_recomputed
                - before.membership_vectors_recomputed,
            "maint_similarity_evaluations":
                c.distance_computations.get("maintenance", 0)
                - before.distance_computations.get("maintenance", 0),
        }

    def _run_audits(self) -> tuple[str, str, str]:
        if self.cfg.audit_every == 0 or self._op_count % self.cfg.audit_every != 0:
            return "SKIP", "SKIP", ""
        npa = audit_maintainer(self._maint, expected_live_vids=self._live_set)
        if isinstance(self._maint, InPlaceMaintainer):
            mem = audit_membership(self._maint)
        else:
            mem = audit_outofplace_membership(self._maint)
        npa_status = "PASS" if npa.passed else "FAIL"
        mem_status = "PASS" if mem.passed else "FAIL"
        detail = ""
        if not npa.passed:
            detail = npa.summary()
            if npa.examples:
                detail += f" e.g. {npa.examples[:2]}"
        if not mem.passed:
            mem_detail = mem.summary()
            detail = (detail + " | " if detail else "") + mem_detail
            first = (mem.missing or mem.extra or mem.duplicates
                     or mem.missing_from_owner or mem.dead_members
                     or mem.ownership_conflicts)
            if first:
                detail += f" e.g. {first[0]}"
        return npa_status, mem_status, detail

    def run_one_operation(self) -> OperationRecord | None:
        """Execute exactly one churn operation, with recall and audits after."""
        op = self._next_op()
        self._maint.drain_events()
        before = self._counters.snapshot()
        t_op = time.perf_counter()
        source_row: int | None = None

        if op == "INSERT":
            row, vid = self._plan_insert()
            source_row = row
            vec = self._store.vector(row).copy()
            self._append_vector(vec, vid)
            self._live.append(vid)          # fresh id is the max → stays sorted
            self._live_set.add(vid)
            self._next_id += 1
            self._inserts += 1
        else:
            idx = self._choose_victim_index()
            vid = int(self._live.pop(idx))
            self._live_set.discard(vid)
            self._maint.delete(vid)
            self._deletes += 1

        self._maint.maintain()

        elapsed_op = time.perf_counter() - t_op
        self._op_time_sum += elapsed_op
        self._op_count += 1

        events = self._maint.drain_events()
        deltas = self._counter_deltas(before, self._counters)
        npa, mem, detail = self._run_audits()
        self._last_npa, self._last_membership = npa, mem
        if npa != "SKIP":
            self._last_audit = f"NPA={npa} MEMBERSHIP={mem}"
        failed = (npa == "FAIL") or (mem == "FAIL")
        if failed:
            self._failed = True

        try:
            recall = self.evaluate_recall()
        except Exception as exc:                # keep the monitor alive on summary
            recall = None
            detail = (detail + " | " if detail else "") + f"recall error: {exc!r}"

        if recall is not None:
            self._last_recall = recall
            self._recent.append(recall)
            self._recall_sum += recall
            self._recall_n += 1
            self._recall_best = max(self._recall_best, recall)
            self._recall_worst = min(self._recall_worst, recall)

        stats = self._maint.stats()
        leaves = stats.get("n_postings", 0)
        sealed = stats.get("sealed_segments", 0)
        mem_size = stats.get("memtable_size", 0)

        return OperationRecord(
            index=self._op_count, op=op, vid=vid, live=len(self._live),
            leaves=leaves,
            splits_total=self._counters.splits,
            merges_total=self._counters.merges,
            recall=recall,
            elapsed_total=time.perf_counter() - self._t0,
            elapsed_op=elapsed_op, source_row=source_row, events=events,
            deltas=deltas, npa=npa, membership=mem, detail=detail,
            failed=failed,
            flushes_total=self._counters.flushes,
            compactions_total=self._counters.compaction_events,
            sealed_segments=sealed,
            memtable_size=mem_size,
        )

    # -- formatting -------------------------------------------------------

    def format_record(self, r: OperationRecord) -> list[str]:
        recall = "N/A" if r.recall is None else f"{r.recall:.4f}"
        if self.cfg.strategy == "outofplace":
            lines = [
                f"[{_fmt_elapsed(r.elapsed_total)}] #{r.index:04d} {r.op:<6} "
                f"vid={r.vid} live={r.live} leaves={r.leaves} sealed={r.sealed_segments} "
                f"flushes={r.flushes_total} compactions={r.compactions_total} "
                f"recall@{self.cfg.k}={recall} NPA={r.npa} MEMBERSHIP={r.membership}"
            ]
        else:
            lines = [
                f"[{_fmt_elapsed(r.elapsed_total)}] #{r.index:04d} {r.op:<6} "
                f"vid={r.vid} live={r.live} leaves={r.leaves} "
                f"splits={r.splits_total} merges={r.merges_total} "
                f"recall@{self.cfg.k}={recall} NPA={r.npa} MEMBERSHIP={r.membership}"
            ]
        for ev in r.events:
            lines.append(ev.format())
        active = {k: v for k, v in r.deltas.items() if v}
        if active:
            lines.append("  delta: " + " ".join(
                f"{k}=+{v}" for k, v in active.items()
            ) + " prunes=not_instrumented")
        if r.detail:
            lines.append(f"  !! {r.detail}")
        return lines

    def print_record(self, r: OperationRecord) -> None:
        for line in self.format_record(r):
            self._print_line(line)

    def _print_header(self) -> None:
        c = self.cfg
        delta = "disabled" if c.delta is None else f"{c.delta:.2f}"
        stats = self._maint.stats()
        self._print_line(f"Live churn monitor — {c.strategy} maintenance")
        self._print_line(
            f"  dataset={c.dataset} dim={c.dim} seed={c.seed} "
            f"initial_size={c.initial_size} live={len(self._live)} "
            f"postings={stats.get('n_postings', 0)} "
            f"depth={stats.get('depth', 0)}"
        )
        if c.strategy == "inplace":
            merge_thr = (
                self._maint._effective_merge_threshold() if c.merge_threshold is None
                else c.merge_threshold
            )
            self._print_line(
                f"  leaf_size={c.leaf_size} branching_factor={c.branching_factor} "
                f"split_threshold={c.split_threshold} merge_threshold={merge_thr} "
                f"delta={delta}"
            )
        else:
            self._print_line(
                f"  leaf_size={c.leaf_size} branching_factor={c.branching_factor} "
                f"memtable_limit={c.memtable_limit} "
                f"compaction_threshold={c.compaction_threshold} "
                f"delta={delta}"
            )
        self._print_line(
            f"  k={c.k} query_count={c.query_count} interval={c.interval:.2f}s "
            f"audit_every={c.audit_every}"
        )
        self._print_line(
            "  (unique ids & tombstones grow without bound; matrix doubles "
            "on demand)"
        )
        initial = self.evaluate_recall()
        if initial is not None:
            self._last_recall = initial
            self._recent.append(initial)
            self._recall_sum += initial
            self._recall_n += 1
            self._recall_best = max(self._recall_best, initial)
            self._recall_worst = min(self._recall_worst, initial)
        recall = "N/A" if initial is None else f"{initial:.4f}"
        self._print_line(f"  initial recall@{c.k}={recall}")

    def _print_summary(self) -> None:
        recent = list(self._recent)
        mean_recent = sum(recent) / len(recent) if recent else float("nan")
        avg_op = self._op_time_sum / self._op_count if self._op_count else 0.0
        best = self._recall_best if self._recall_n else float("nan")
        worst = self._recall_worst if self._recall_n else float("nan")
        stats = self._maint.stats()
        leaves = stats.get("n_postings", 0)
        if self.cfg.strategy == "inplace":
            self._print_line(
                f"--- summary @ op {self._op_count}: inserts={self._inserts} "
                f"deletes={self._deletes} splits={self._counters.splits} "
                f"merges={self._counters.merges} live={len(self._live)} "
                f"postings={leaves} "
                f"recall mean={mean_recent:.4f} best={best:.4f} worst={worst:.4f} "
                f"avg_op={avg_op * 1000:.2f}ms"
            )
        else:
            self._print_line(
                f"--- summary @ op {self._op_count}: inserts={self._inserts} "
                f"deletes={self._deletes} flushes={self._counters.flushes} "
                f"compactions={self._counters.compaction_events} "
                f"sealed={stats.get('sealed_segments', 0)} "
                f"memtable={stats.get('memtable_size', 0)} "
                f"live={len(self._live)} postings={leaves} "
                f"recall mean={mean_recent:.4f} best={best:.4f} worst={worst:.4f} "
                f"avg_op={avg_op * 1000:.2f}ms"
            )

    def _print_final(self) -> None:
        latest = "N/A" if self._last_recall is None else f"{self._last_recall:.4f}"
        best = f"{self._recall_best:.4f}" if self._recall_n else "N/A"
        worst = f"{self._recall_worst:.4f}" if self._recall_n else "N/A"
        reason = "Ctrl+C" if self._interrupted else (
            "audit failure" if self._failed else "completed"
        )
        stats = self._maint.stats()
        leaves = stats.get("n_postings", 0)
        self._print_line("=" * 52)
        self._print_line("Final summary")
        self._print_line(
            f"  operations={self._op_count} inserts={self._inserts} "
            f"deletes={self._deletes}"
        )
        if self.cfg.strategy == "inplace":
            self._print_line(
                f"  splits={self._counters.splits} merges={self._counters.merges} "
                f"hierarchy_rebuilds={self._counters.hierarchy_rebuilds}"
            )
            self._print_line(
                f"  live={len(self._live)} "
                f"postings={leaves} "
                f"tombstones={len(self._maint._tombstones)}"
            )
        else:
            self._print_line(
                f"  flushes={self._counters.flushes} "
                f"compactions={self._counters.compaction_events} "
                f"hierarchy_rebuilds={self._counters.hierarchy_rebuilds}"
            )
            self._print_line(
                f"  live={len(self._live)} "
                f"sealed_segments={stats.get('sealed_segments', 0)} "
                f"memtable={stats.get('memtable_size', 0)} "
                f"postings={leaves} "
                f"tombstones={len(self._maint._tombstones)}"
            )
        self._print_line(
            f"  latest recall@{self.cfg.k}={latest} "
            f"best={best} worst={worst}"
        )
        self._print_line(f"  auditor: {self._last_audit}")
        self._print_line(f"  stopped: {reason}")
        self._print_line("=" * 52)

    # -- loop -------------------------------------------------------------

    def stop(self) -> None:
        self._stopped = True

    def run(
        self,
        max_operations: int | None = None,
        on_operation=None,
    ) -> "LiveChurnMonitor":
        """Run until Ctrl+C, :meth:`stop`, an audit failure, or *max_operations*.

        ``on_operation(record)`` may return ``False`` to stop.  A
        ``KeyboardInterrupt`` (Ctrl+C) is caught, a final summary is printed, and
        the method returns normally.
        """
        self._print_header()
        try:
            while not self._stopped:
                if max_operations is not None and self._op_count >= max_operations:
                    break
                record = self.run_one_operation()
                if record is None:
                    break
                self.print_record(record)
                if on_operation is not None and on_operation(record) is False:
                    break
                if (self.cfg.summary_every
                        and self._op_count % self.cfg.summary_every == 0):
                    self._print_summary()
                if record.failed and not self.cfg.continue_on_audit_failure:
                    self._stopped = True
                    break
                if self.cfg.interval > 0:
                    time.sleep(self.cfg.interval)
        except KeyboardInterrupt:
            self._interrupted = True
            self._print_line("")
            self._print_line("Interrupted (Ctrl+C) — shutting down.")
        self._print_final()
        return self


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="live_churn",
        description="Continuously churn an in-place vector index and print "
                    "recall after every operation. Runs until Ctrl+C.",
    )
    p.add_argument("--k", type=int, default=10,
                   help="recall cutoff (default: 10)")
    p.add_argument("--query-count", dest="query_count", type=int, default=32,
                   help="number of fixed evaluation queries (default: 32)")
    p.add_argument("--seed", type=int, default=7,
                   help="seed for dataset, queries and churn choices "
                        "(default: 7)")
    p.add_argument("--initial-size", dest="initial_size", type=int, default=300,
                   help="initial live-vector count (default: 300)")
    p.add_argument("--dataset", default="synthetic", choices=["synthetic"],
                   help="dataset selection (only 'synthetic' is supported)")
    p.add_argument("--dim", type=int, default=128,
                   help="vector dimension (default: 128)")
    p.add_argument("--depth", type=int, default=None,
                   help="hierarchy depth (default: project default = flat when "
                        "few leaves)")
    p.add_argument("--leaf-size", dest="leaf_size", type=int, default=40,
                   help="target vectors per posting during build (default: 40)")
    p.add_argument("--branching-factor", dest="branching_factor", type=int,
                   default=16, help="routing fan-out (default: 16)")
    p.add_argument("--split-threshold", dest="split_threshold", type=int,
                   default=None,
                   help="posting split threshold (default: = --leaf-size)")
    p.add_argument("--merge-threshold", dest="merge_threshold", type=int,
                   default=None,
                   help="posting merge underflow threshold "
                        "(default: max(1, split_threshold//2))")
    p.add_argument("--delta", type=parse_delta_arg, default=None,
                   help="fixed-Δ overlap value, or 'none'/'off' to disable "
                        "(default: disabled)")
    p.add_argument("--interval", type=float, default=0.2,
                   help="delay between operations in seconds (default: 0.2)")
    p.add_argument("--audit-every", dest="audit_every", type=int, default=1,
                   help="run both auditors every N ops; 0 disables "
                        "(default: 1)")
    p.add_argument("--summary-every", dest="summary_every", type=int, default=50,
                   help="print a cumulative summary every N ops; 0 disables "
                        "(default: 50)")
    p.add_argument("--n-postings", dest="n_postings", type=int, default=1,
                   help="postings fetched per unit during search (default: 1)")
    p.add_argument("--beam", type=int, default=8,
                   help="beam width during search (default: 8)")
    p.add_argument("--continue-on-audit-failure", dest="continue_on_audit_failure",
                   action="store_true",
                   help="keep running after an audit failure (default: stop so "
                        "the failure is visible and reproducible)")
    p.add_argument("--strategy", choices=["inplace", "outofplace"], default="inplace",
                   help="maintenance strategy: 'inplace' (SPFresh) or 'outofplace' (LSM-Vec) (default: inplace)")
    p.add_argument("--memtable-limit", dest="memtable_limit", type=int, default=50,
                   help="flush threshold for out-of-place (default: 50)")
    p.add_argument("--compaction-threshold", dest="compaction_threshold", type=int, default=4,
                   help="sealed segment count triggering compaction for out-of-place (default: 4)")
    p.add_argument("--max-operations", dest="max_operations", type=int,
                   default=None,
                   help="demo/testing aid: stop after N operations "
                        "(default: run until Ctrl+C)")
    return p


def resolve_config(args: argparse.Namespace) -> LiveChurnConfig:
    leaf_size = int(args.leaf_size)
    split_threshold = (
        int(args.split_threshold) if args.split_threshold is not None
        else leaf_size
    )
    return LiveChurnConfig(
        strategy=getattr(args, "strategy", "inplace"),
        initial_size=int(args.initial_size),
        k=int(args.k),
        query_count=int(args.query_count),
        dim=int(args.dim),
        dataset=str(args.dataset),
        depth=(None if args.depth is None else int(args.depth)),
        leaf_size=leaf_size,
        branching_factor=int(args.branching_factor),
        split_threshold=split_threshold,
        merge_threshold=(None if args.merge_threshold is None
                         else int(args.merge_threshold)),
        delta=(None if args.delta is None else float(args.delta)),
        interval=float(args.interval),
        seed=int(args.seed),
        audit_every=int(args.audit_every),
        summary_every=int(args.summary_every),
        n_postings=int(args.n_postings),
        beam=int(args.beam),
        continue_on_audit_failure=bool(args.continue_on_audit_failure),
        memtable_limit=int(getattr(args, "memtable_limit", 50)),
        compaction_threshold=int(getattr(args, "compaction_threshold", 4)),
    )


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    try:
        cfg = resolve_config(args)
    except ValueError as exc:
        parser.error(str(exc))
        return 2  # unreachable (parser.error exits)
    monitor = LiveChurnMonitor(cfg)
    monitor.run(max_operations=args.max_operations)
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
