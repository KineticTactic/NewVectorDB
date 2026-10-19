"""harness/structure.py — structural maintenance instrumentation.

Pure, observational metrics over the *shape* of the live index at each
checkpoint, plus thin subclasses of the two maintainers that time maintenance
mutations and count empty-leaf prunes.  Neither the metrics nor the subclasses
change any indexing, maintenance or query behaviour: the subclasses only wrap
existing hooks (`_split`, `_merge_after_delete`, `_prune_empty_leaves`,
`_flush`, `_compact`) and read state.

Why a separate table
--------------------
The paired (``compare.py``) and equal-work (``search_budget.py``) harnesses
report *query-side* outcomes (recall, work, latency) and *update-side counters*
(splits, merges, ...), but not the shape of the index those updates produced:
how large the leaf postings are, how far they sit above ``split_threshold``, or
how long maintenance actually took.  This module records that shape once per
``(strategy, config, seed, checkpoint)`` into ``results_structure_checkpoints``
so a threshold sweep can be interpreted without re-deriving it from the
per-preset raw rows.

Conventions (also written into ``run_metadata.json``)
-----------------------------------------------------
* Posting sizes count **live, canonical-owned** vectors only: tombstoned ids are
  excluded and each vector is counted in exactly one posting (its owner).
  Replica membership is reported separately (``member_list_size_*``), never
  folded into the canonical sizes.
* Percentiles use NumPy's default ``linear`` interpolation (``np.percentile``).
* Fractions are in ``[0, 1]``, not percent.
* "Over threshold" means strictly ``size > split_threshold`` — the same strict
  comparison the maintainers use to trigger a split.
* Interval counters are per-checkpoint deltas of the existing cumulative
  counters; production counters are never reset.  Interval at checkpoint 0 is 0.
* In the CSV, an empty string/``None`` (null in Parquet) means *not applicable
  to this strategy*; ``0`` means applicable but nothing happened; ``nan`` is
  used only for a statistic that is undefined because the structure is empty.
* Wall times are ``time.perf_counter_ns`` around maintenance mutations only
  (in-place split/merge/prune; out-of-place flush/compaction), excluding search
  and ground-truth work.  ``update_wall_ms_*`` additionally covers the whole
  insert/delete operation and is reported separately.
"""

from __future__ import annotations

import time
from typing import Any, Iterable, Sequence

import numpy as np

from ..maintain.inplace import InPlaceMaintainer
from ..maintain.outofplace import OutOfPlaceMaintainer

# ``None`` marks a metric that does not apply to a strategy.
NA = None

# Per-event ownership: which strategy the event can happen in.
_EVENT_STRATEGY = {
    "splits": "inplace",
    "merges": "inplace",
    "pruned_postings": "inplace",
    "flushes": "outofplace",
    "compactions": "outofplace",
    "compaction_vectors_reclustered": "outofplace",
}


# ---------------------------------------------------------------------------
# Pure summarizers
# ---------------------------------------------------------------------------

def _pct(arr: np.ndarray, q: float) -> float:
    return float(np.percentile(arr, q, method="linear"))


def posting_size_stats(sizes: Sequence[int], split_threshold: int) -> dict[str, Any]:
    """Distribution of live canonical-owned sizes over non-empty postings.

    Empty structure (no non-empty posting) -> count 0, size statistics ``nan``,
    fractions ``0.0`` (no division by zero).
    """
    arr = np.asarray(list(sizes), dtype=np.float64)
    n = int(arr.size)
    if n == 0:
        return {
            "live_posting_count": 0,
            "posting_size_median": float("nan"),
            "posting_size_mean": float("nan"),
            "posting_size_p90": float("nan"),
            "posting_size_p95": float("nan"),
            "posting_size_p99": float("nan"),
            "posting_size_max": float("nan"),
            "fraction_postings_over_split_threshold": 0.0,
            "fraction_vectors_in_overthreshold_postings": 0.0,
        }
    over = arr > float(split_threshold)
    total = float(arr.sum())
    return {
        "live_posting_count": n,
        "posting_size_median": float(np.median(arr)),
        "posting_size_mean": float(arr.mean()),
        "posting_size_p90": _pct(arr, 90.0),
        "posting_size_p95": _pct(arr, 95.0),
        "posting_size_p99": _pct(arr, 99.0),
        "posting_size_max": float(arr.max()),
        "fraction_postings_over_split_threshold": float(over.mean()),
        "fraction_vectors_in_overthreshold_postings": (
            float(arr[over].sum() / total) if total > 0.0 else 0.0
        ),
    }


def member_size_stats(sizes: Sequence[int]) -> dict[str, Any]:
    """Distribution of live member-list sizes (replicas included)."""
    arr = np.asarray(list(sizes), dtype=np.float64)
    n = int(arr.size)
    if n == 0:
        return {
            "member_list_size_count": 0,
            "member_list_size_median": float("nan"),
            "member_list_size_p95": float("nan"),
            "member_list_size_max": float("nan"),
        }
    return {
        "member_list_size_count": n,
        "member_list_size_median": float(np.median(arr)),
        "member_list_size_p95": _pct(arr, 95.0),
        "member_list_size_max": float(arr.max()),
    }


def live_mask_from(live: Iterable[int], n_total: int) -> np.ndarray:
    """Boolean mask indexed by vector id; True iff the id is live.

    Grows past ``n_total`` if a live id sits above it, so lookups never raise.
    """
    idx = np.fromiter((int(v) for v in live), dtype=np.int64)
    size = int(n_total)
    if idx.size:
        size = max(size, int(idx.max()) + 1)
    mask = np.zeros(max(0, size), dtype=bool)
    if idx.size:
        mask[idx] = True
    return mask


def _mask_has(mask: np.ndarray, vid: int) -> bool:
    vi = int(vid)
    return 0 <= vi < mask.shape[0] and bool(mask[vi])


def _live_count(ids: np.ndarray, live_mask: np.ndarray) -> int:
    if ids.size == 0:
        return 0
    return int(np.count_nonzero(live_mask[ids]))


def _posting_block(postings, live_mask: np.ndarray,
                   split_threshold: int) -> dict[str, Any]:
    """Canonical-owned size stats over *non-empty* postings + member stats."""
    sizes: list[int] = []
    member_sizes: list[int] = []
    for p in postings:
        oc = _live_count(p.owned, live_mask)
        if oc > 0:
            sizes.append(oc)
        mc = _live_count(p.members, live_mask)
        if mc > 0:
            member_sizes.append(mc)
    block = posting_size_stats(sizes, split_threshold)
    block.update(member_size_stats(member_sizes))
    return block


def inplace_structure(maint: InPlaceMaintainer, live_mask: np.ndarray,
                      split_threshold: int) -> dict[str, Any]:
    """Structure of the single in-place segment (live canonical-owned leaves)."""
    block = _posting_block(maint._segment.postings, live_mask, split_threshold)
    block["searchable_unit_count"] = len(maint.searchable_units())
    block["sealed_segment_count"] = NA
    block["memtable_live_vector_count"] = NA
    return block


def outofplace_structure(maint: OutOfPlaceMaintainer, live_mask: np.ndarray,
                         split_threshold: int) -> dict[str, Any]:
    """Structure of sealed segments; the MemTable is reported separately.

    Sealed-segment owned arrays may still carry tombstoned ids (lazy deletion),
    so liveness is filtered through ``live_mask``; the MemTable is *not* a
    posting and never contributes to the posting-size distribution.
    """
    postings = [p for seg in maint._sealed for p in seg.postings]
    block = _posting_block(postings, live_mask, split_threshold)
    block["searchable_unit_count"] = len(maint.searchable_units())
    block["sealed_segment_count"] = len(maint._sealed)
    block["memtable_live_vector_count"] = sum(
        1 for v in maint._memtable if _mask_has(live_mask, v)
    )
    return block


def structure_snapshot(*, strategy: str, maint, live: Iterable[int],
                       n_total: int, split_threshold: int) -> dict[str, Any]:
    """Dispatch to the strategy-specific, read-only structure snapshot."""
    live_mask = live_mask_from(live, n_total)
    if strategy == "inplace":
        return inplace_structure(maint, live_mask, split_threshold)
    return outofplace_structure(maint, live_mask, split_threshold)


# ---------------------------------------------------------------------------
# Event + timing blocks
# ---------------------------------------------------------------------------

def event_fields(*, strategy: str, cumulative: dict, interval: dict,
                 pruned_cumulative: int | None,
                 pruned_interval: int | None) -> dict[str, Any]:
    """Interval/cumulative event counts, with N/A for the other strategy.

    ``cumulative``/``interval`` are the existing maintenance-counter snapshots
    (``maint_cumulative``/``maint_delta``); nothing is reset.
    """
    src: dict[str, tuple[Any, Any]] = {
        "splits": (interval.get("splits", 0), cumulative.get("splits", 0)),
        "merges": (interval.get("merges", 0), cumulative.get("merges", 0)),
        "pruned_postings": (pruned_interval, pruned_cumulative),
        "flushes": (interval.get("flushes", 0), cumulative.get("flushes", 0)),
        "compactions": (interval.get("compaction_events", 0),
                        cumulative.get("compaction_events", 0)),
        "compaction_vectors_reclustered": (
            interval.get("vectors_reclustered", 0),
            cumulative.get("vectors_reclustered", 0),
        ),
    }
    out: dict[str, Any] = {}
    for name, applies in _EVENT_STRATEGY.items():
        iv, cum = src[name]
        if applies == strategy:
            out[f"{name}_interval"] = None if iv is None else int(iv)
            out[f"{name}_cumulative"] = None if cum is None else int(cum)
        else:
            out[f"{name}_interval"] = NA
            out[f"{name}_cumulative"] = NA
    return out


def timing_fields(*, maintenance_ns: int, prev_maintenance_ns: int,
                  update_ns: int, prev_update_ns: int) -> dict[str, float]:
    """Maintenance and full-update wall time, in milliseconds."""
    return {
        "maintenance_wall_ms_interval":
            (maintenance_ns - prev_maintenance_ns) / 1e6,
        "maintenance_wall_ms_cumulative": maintenance_ns / 1e6,
        "update_wall_ms_interval": (update_ns - prev_update_ns) / 1e6,
        "update_wall_ms_cumulative": update_ns / 1e6,
    }


def build_structure_row(*, identity: dict, experiment_id: str, run_index: int,
                        strategy: str, checkpoint: int, operations_processed: int,
                        live_vector_count: int, structure: dict, events: dict,
                        timing: dict, correctness_status: str,
                        checkpoint_correct: bool) -> dict[str, Any]:
    """Assemble one normalized structure row (pure; no I/O)."""
    row: dict[str, Any] = {
        "experiment_id": experiment_id,
        "run_index": run_index,
        "strategy": strategy,
        "checkpoint": checkpoint,
        "operations_processed": operations_processed,
        "live_vector_count": live_vector_count,
        "correctness_status": correctness_status,
        "checkpoint_correct": checkpoint_correct,
    }
    row.update(identity)
    row.update(structure)
    row.update(events)
    row.update(timing)
    return row


# ---------------------------------------------------------------------------
# Instrumented maintainers (observational wrappers)
# ---------------------------------------------------------------------------

class _MaintenanceTimerMixin:
    """Accumulate maintenance wall time with a reentrancy depth guard.

    Nested timed methods (e.g. ``_merge_after_delete`` -> ``_split_oversize``
    -> ``_split``) are counted once: only the outermost enter/exit pair adds to
    the total, so overlapping hooks never double count.
    """

    def _timer_init(self) -> None:
        self._maint_ns = 0
        self._maint_depth = 0
        self._maint_t0 = 0

    def _timer_enter(self) -> None:
        if self._maint_depth == 0:
            self._maint_t0 = time.perf_counter_ns()
        self._maint_depth += 1

    def _timer_exit(self) -> None:
        self._maint_depth -= 1
        if self._maint_depth == 0:
            self._maint_ns += time.perf_counter_ns() - self._maint_t0

    @property
    def maintenance_ns(self) -> int:
        return self._maint_ns


class InstrumentedInPlaceMaintainer(_MaintenanceTimerMixin, InPlaceMaintainer):
    """InPlaceMaintainer that times splits/merges and counts pruned leaves."""

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        super().__init__(*args, **kwargs)
        self._timer_init()
        self.pruned_postings = 0

    def _split(self, pid: int) -> None:
        self._timer_enter()
        try:
            super()._split(pid)
        finally:
            self._timer_exit()

    def _merge_after_delete(self) -> None:
        self._timer_enter()
        try:
            super()._merge_after_delete()
        finally:
            self._timer_exit()

    def _prune_empty_leaves(self) -> None:
        before = len(self._segment.postings)
        super()._prune_empty_leaves()
        removed = before - len(self._segment.postings)
        if removed > 0:
            self.pruned_postings += removed


class InstrumentedOutOfPlaceMaintainer(_MaintenanceTimerMixin, OutOfPlaceMaintainer):
    """OutOfPlaceMaintainer that times flush/compaction."""

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        super().__init__(*args, **kwargs)
        self._timer_init()
        self.pruned_postings = NA      # pruned-empty-leaf is in-place specific

    def _flush(self) -> None:
        self._timer_enter()
        try:
            super()._flush()
        finally:
            self._timer_exit()

    def _compact(self) -> None:
        self._timer_enter()
        try:
            super()._compact()
        finally:
            self._timer_exit()


def make_instrumented_maintainer(strategy: str, vectors: np.ndarray,
                                 cfg_index, cfg_maintain, counters,
                                 initial_segment):
    """Construct the instrumented maintainer for a strategy (drop-in)."""
    cls = (InstrumentedInPlaceMaintainer if strategy == "inplace"
           else InstrumentedOutOfPlaceMaintainer)
    return cls(vectors, cfg_index, cfg_maintain, counters,
               initial_segment=initial_segment.clone())


# ---------------------------------------------------------------------------
# Metadata for run_metadata.json
# ---------------------------------------------------------------------------

def structure_metadata() -> dict[str, Any]:
    return {
        "table": "results_structure_checkpoints.csv",
        "granularity": "one row per (strategy, base config, seed, checkpoint); "
                       "preset-independent (values repeat if copied into "
                       "per-preset rows)",
        "posting_size_basis": "live canonical-owned vectors per leaf posting "
                              "(tombstones excluded, replicas not counted)",
        "member_size_basis": "live members per posting, replicas included "
                             "(separate from canonical sizes)",
        "outofplace_basis": "sealed segments only; MemTable reported separately "
                            "as memtable_live_vector_count and excluded from "
                            "posting-size stats",
        "percentile_method": "numpy.percentile, linear interpolation",
        "fraction_units": "[0,1] not percent",
        "over_threshold_rule": "strict size > split_threshold",
        "empty_structure": "size stats are nan, counts 0, fractions 0.0",
        "events": "interval = per-checkpoint delta of the existing cumulative "
                  "maintenance counters; production counters are never reset; "
                  "checkpoint 0 interval is 0",
        "na_convention": "None/empty (null in Parquet) means not applicable to "
                         "the strategy; 0 means applicable but did not happen",
        "timer_scope": "perf_counter_ns around maintenance mutations only "
                       "(in-place split/merge/prune; out-of-place "
                       "flush/compaction); nested hooks counted once; search "
                       "and ground-truth excluded",
        "update_wall_ms": "whole insert/delete operation (includes maintenance), "
                          "reported separately from maintenance_wall_ms",
        "checkpoint_zero": "interval counts and wall-time intervals are 0 "
                           "(initial build excluded)",
        "unsupported_metrics": {
            "pruned_postings": "in-place only (empty-leaf pruning); n/a for "
                               "out-of-place",
            "sealed_segment_count": "out-of-place only",
            "memtable_live_vector_count": "out-of-place only",
            "posting_size_percentiles_oop": "computed over sealed segment leaf "
                                            "postings only; multi-level segment "
                                            "hierarchies are all included at "
                                            "their leaf level",
        },
    }
