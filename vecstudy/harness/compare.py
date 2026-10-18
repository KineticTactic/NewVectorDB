"""harness/compare.py — paired InPlace vs OutOfPlace maintenance benchmark.

This module backs ``scripts/compare_maintenance.py``.  It runs one dataset and
one deterministic churn workload against **both** maintenance strategies from an
**identical** initial :class:`~vecstudy.index.segment.Segment`, and reports
search quality, query latency, update latency, maintenance cost, storage
overhead and correctness side by side.

It is a measurement harness: it reuses the production maintainers, the single
production query path (:func:`vecstudy.index.query.search`), the project metric
(inner product, ``distance = 1 - similarity``), the deterministic churn trace
generator, and the two independent auditors.  No maintenance or query
algorithm is reimplemented, and no algorithm is changed to make the arms look
more comparable.

Paired-run invariants
---------------------
* Both arms start from ``Segment.build(...).clone()`` of **one** segment, so the
  initial centroids, hierarchy, canonical ownership and memberships are
  identical and mutably independent.
* Both arms receive the **same** :class:`~vecstudy.harness.churn.ChurnTrace`
  (operation order and vector ids) and the **same** fixed query set.
* The backing store is pre-allocated to ``initial_size + reserve_size`` rows and
  shared by both arms; an insert selects a reserve row by id, so no array is
  reallocated or re-bound mid-run (no stale matrix references are possible).
* Paired runs execute sequentially (never concurrently).  Which arm runs first
  is alternated by run index to reduce systematic order effects.

Measurement limitations (see also ``R1_T8_Report.md``)
------------------------------------------------------
1. **Per-unit posting budget.**  ``search`` descends each searchable unit with
   ``n_postings`` independently, so out-of-place search over ``S`` units may
   examine up to ``S`` times the postings of the single in-place unit at the
   same nominal configuration.  The engine has no global budget and is not
   modified here; actual work is measured and a work-normalized view is
   provided instead of assuming equality.
2. **Maintenance distance work is a lower bound.**  ``balanced_kmeans`` delegates
   its Lloyd iterations to scikit-learn, whose internal distances are uncounted
   by design.  Counters only see NumPy distance kernels.  Both arms call the same
   routine, so the undercount is symmetric but absolute counts understate work.
3. **Empty-leaf pruning has no counter** in the in-place arm; it is reported as
   ``n/a`` and never inferred from a leaf-count delta.
4. **Fixed-Δ membership audit is in-place only.**  ``audit_membership`` rejects
   the out-of-place maintainer, so its membership-correctness audit is ``n/a``;
   out-of-place NPA is checked within each sealed segment (documented semantics).
5. **RSS** is unavailable without ``psutil``; macOS ``ru_maxrss`` is *peak*, not
   current, and is labeled ``peak``.  Index-array bytes are an **estimate** from
   array shapes/dtypes, not measured physical storage.
6. **Maintenance time cannot be cleanly separated** from update time: in-place
   maintenance is eager (inside insert/delete) and out-of-place flushes happen
   inside insert.  Update latency therefore includes triggered maintenance; an
   attributed-maintenance wall time is reported separately with that caveat.
7. **Latency is wall-clock** and is not part of the algorithmic study (DESIGN
   §1); it is reported as requested with a warm-up pass and pinned BLAS threads.
"""

from __future__ import annotations

import argparse
import copy
import csv
import datetime
import json
import os
import platform
import sys
import time
import traceback
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np

from ..core.config import IndexConfig, MaintainConfig, QueryConfig
from ..core.metrics import Counters
from ..index.query import search
from ..index.segment import Segment
from ..maintain.base import IndexMaintainer
from ..maintain.inplace import InPlaceMaintainer
from ..maintain.outofplace import OutOfPlaceMaintainer
from .churn import (
    ChurnTrace,
    audit_maintainer,
    audit_membership,
    generate_churn_trace,
)
from .datasets import make_synthetic, recall_at_k

STRATEGIES = ("inplace", "outofplace")

# Buckets that count maintenance work (everything except query routing/scan).
_MAINT_BUCKETS = ("build", "maintenance")


# ---------------------------------------------------------------------------
# Environment metadata
# ---------------------------------------------------------------------------

def _git_sha() -> str:
    import subprocess
    try:
        r = subprocess.run(["git", "rev-parse", "HEAD"], capture_output=True,
                           text=True, timeout=5)
        if r.returncode == 0:
            return r.stdout.strip()
    except Exception:
        pass
    return "nogit"


def _threadpool_info() -> list[dict]:
    try:
        from threadpoolctl import threadpool_info
        return threadpool_info()
    except Exception:
        return []


def _rss_bytes() -> tuple[int | None, str]:
    """Best-effort process memory.  Returns ``(bytes, kind)``.

    ``psutil`` gives current RSS if installed; otherwise macOS/BSD
    ``ru_maxrss`` (bytes) or Linux ``ru_maxrss`` (KiB) is reported as *peak*.
    Returns ``(None, "unavailable")`` when nothing reliable is available.
    """
    try:
        import psutil  # type: ignore
        return int(psutil.Process().memory_info().rss), "current"
    except Exception:
        pass
    try:
        import resource
        usage = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
        if sys.platform == "darwin":
            return int(usage), "peak"
        return int(usage) * 1024, "peak"
    except Exception:
        return None, "unavailable"


def _environment_metadata(blas_threads: int | None) -> dict:
    return {
        "python": sys.version.split()[0],
        "numpy": np.__version__,
        "platform": platform.platform(),
        "machine": platform.machine(),
        "processor": platform.processor(),
        "blas_threads_requested": blas_threads,
        "threadpool_info": _threadpool_info(),
    }


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class CompareConfig:
    """One experiment cell: identical for both strategies."""
    dataset: str
    dim: int
    initial_size: int
    reserve_size: int
    operations: int
    workload: str
    insert_prob: float
    seed: int
    depth: int | None
    leaf_size: int
    branching_factor: int
    split_threshold: int
    merge_threshold: int | None
    delta: float | None
    memtable_limit: int
    compaction_threshold: int
    k: int
    recall_ks: tuple[int, ...]
    query_count: int
    n_postings: int
    beam_widths: tuple[int, ...]

    def to_dict(self) -> dict:
        return {
            "dataset": self.dataset, "dim": self.dim,
            "initial_size": self.initial_size, "reserve_size": self.reserve_size,
            "operations": self.operations, "workload": self.workload,
            "insert_prob": self.insert_prob, "seed": self.seed,
            "depth": self.depth, "leaf_size": self.leaf_size,
            "branching_factor": self.branching_factor,
            "split_threshold": self.split_threshold,
            "merge_threshold": self.merge_threshold, "delta": self.delta,
            "memtable_limit": self.memtable_limit,
            "compaction_threshold": self.compaction_threshold,
            "k": self.k, "recall_ks": list(self.recall_ks),
            "query_count": self.query_count, "n_postings": self.n_postings,
            "beam_widths": list(self.beam_widths),
        }

    @property
    def index_key(self) -> tuple:
        """Key for sharing one built initial Segment across cells/strategies."""
        return (self.dataset, self.dim, self.initial_size, self.leaf_size,
                self.branching_factor, self.depth, self.delta, False, self.seed)

    @property
    def pair_key(self) -> tuple:
        """Key identifying a paired comparison (config + seed), excluding the arm."""
        c = self.to_dict()
        c.pop("dim", None)
        c.pop("reserve_size", None)
        return tuple(sorted((k, str(v)) for k, v in c.items()))


@dataclass
class SweepSpec:
    dataset: str
    initial_size: int
    operations: int
    workload: str
    insert_prob: float
    dim: int
    leaf_size: int
    branching_factor: int
    split_thresholds: list[int]
    merge_thresholds: list[int | None]
    deltas: list[float | None]
    depths: list[int | None]
    n_postings_list: list[int]
    beam_list: list[int]
    seeds: list[int]
    k: int
    recall_ks: tuple[int, ...]
    query_count: int
    memtable_limit: int
    compaction_threshold: int
    checkpoint_interval: int
    flush_final: bool
    continue_on_audit_failure: bool
    blas_threads: int | None
    table_a_rows: str
    out_dir: Path
    overwrite: bool
    profile: str

    def reserve_size(self) -> int:
        return max(1, self.operations)

    def expand(self) -> list[CompareConfig]:
        cells: list[CompareConfig] = []
        for depth in self.depths:
            for st in self.split_thresholds:
                for mt in self.merge_thresholds:
                    for delta in self.deltas:
                        for npost in self.n_postings_list:
                            for beam in self.beam_list:
                                for seed in self.seeds:
                                    cells.append(CompareConfig(
                                        dataset=self.dataset, dim=self.dim,
                                        initial_size=self.initial_size,
                                        reserve_size=self.reserve_size(),
                                        operations=self.operations,
                                        workload=self.workload,
                                        insert_prob=self.insert_prob,
                                        seed=seed, depth=depth,
                                        leaf_size=self.leaf_size,
                                        branching_factor=self.branching_factor,
                                        split_threshold=st, merge_threshold=mt,
                                        delta=delta,
                                        memtable_limit=self.memtable_limit,
                                        compaction_threshold=self.compaction_threshold,
                                        k=self.k, recall_ks=self.recall_ks,
                                        query_count=self.query_count,
                                        n_postings=npost,
                                        beam_widths=(beam,),
                                    ))
        return cells

    def to_dict(self) -> dict:
        return {
            "dataset": self.dataset, "initial_size": self.initial_size,
            "operations": self.operations, "workload": self.workload,
            "insert_prob": self.insert_prob, "dim": self.dim,
            "leaf_size": self.leaf_size,
            "branching_factor": self.branching_factor,
            "split_thresholds": list(self.split_thresholds),
            "merge_thresholds": list(self.merge_thresholds),
            "deltas": list(self.deltas), "depths": list(self.depths),
            "n_postings": list(self.n_postings_list),
            "beam_widths": list(self.beam_list), "seeds": list(self.seeds),
            "k": self.k, "recall_ks": list(self.recall_ks),
            "query_count": self.query_count,
            "memtable_limit": self.memtable_limit,
            "compaction_threshold": self.compaction_threshold,
            "checkpoint_interval": self.checkpoint_interval,
            "flush_final": self.flush_final,
            "continue_on_audit_failure": self.continue_on_audit_failure,
            "blas_threads": self.blas_threads,
            "profile": self.profile,
        }


# ---------------------------------------------------------------------------
# Dataset / initial-state caching
# ---------------------------------------------------------------------------

class _DatasetCache:
    """Caches generated datasets, query sets and built initial segments."""

    def __init__(self) -> None:
        self._data: dict[tuple, np.ndarray] = {}
        self._queries: dict[tuple, np.ndarray] = {}
        self._segments: dict[tuple, Segment] = {}

    def data(self, cfg: CompareConfig) -> np.ndarray:
        key = (cfg.dataset, cfg.dim, cfg.initial_size + cfg.reserve_size, cfg.seed)
        if key not in self._data:
            self._data[key] = _load_dataset(
                cfg.dataset, cfg.initial_size + cfg.reserve_size, cfg.dim, cfg.seed
            )
        return self._data[key]

    def queries(self, cfg: CompareConfig) -> np.ndarray:
        key = (cfg.dataset, cfg.dim, cfg.query_count, cfg.seed)
        if key not in self._queries:
            self._queries[key] = _load_dataset(
                cfg.dataset, cfg.query_count, cfg.dim, cfg.seed + 1
            )
        return self._queries[key]

    def initial_segment(self, cfg: CompareConfig) -> Segment:
        key = cfg.index_key
        if key not in self._segments:
            vectors = self.data(cfg)
            initial_vids = np.arange(cfg.initial_size, dtype=np.int64)
            cfg_idx = _index_config(cfg)
            self._segments[key] = Segment.build(
                initial_vids, vectors, cfg_idx, Counters()
            )
        return self._segments[key]


def _load_dataset(dataset: str, n: int, dim: int, seed: int) -> np.ndarray:
    if dataset == "synthetic":
        return make_synthetic(n=n, dim=dim, seed=seed)
    raise ValueError(
        f"unsupported dataset {dataset!r}; supported: 'synthetic' "
        "(SIFT needs a download — see the report's limitations)"
    )


def _dataset_checksum(arr: np.ndarray) -> str:
    import hashlib
    return hashlib.sha256(np.ascontiguousarray(arr).tobytes()).hexdigest()


# ---------------------------------------------------------------------------
# Config builders
# ---------------------------------------------------------------------------

def _index_config(cfg: CompareConfig) -> IndexConfig:
    return IndexConfig(
        dim=cfg.dim, leaf_size=cfg.leaf_size,
        branching_factor=cfg.branching_factor, depth=cfg.depth,
        delta=cfg.delta, rng_rule=False, seed=cfg.seed,
    )


def _maintain_config(cfg: CompareConfig, strategy: str) -> MaintainConfig:
    return MaintainConfig(
        strategy=strategy, memtable_limit=cfg.memtable_limit,
        compaction_threshold=cfg.compaction_threshold,
        split_threshold=cfg.split_threshold, merge_threshold=cfg.merge_threshold,
    )


def _query_config(cfg: CompareConfig) -> QueryConfig:
    return QueryConfig(k=max(cfg.recall_ks), beam_widths=tuple(cfg.beam_widths),
                       n_postings=cfg.n_postings)


# ---------------------------------------------------------------------------
# Ground truth, work, storage helpers
# ---------------------------------------------------------------------------

def dynamic_ground_truth(queries: np.ndarray, vectors: np.ndarray,
                         live_vids: np.ndarray, k: int) -> np.ndarray:
    """Exact brute-force top-k over only the live vectors (recomputed each call).

    Metric matches production (float32 inner product).  Ties are broken by
    ascending global VectorId (stable).  Local row offsets are mapped back to
    stable global ids.  Returns an ``(Q, min(k, N_live))`` int64 array.
    """
    live = np.asarray(live_vids, dtype=np.int64)
    q = np.asarray(queries, dtype=np.float32)
    if len(live) == 0:
        return np.empty((len(q), 0), dtype=np.int64)
    eff_k = min(k, len(live))
    base = vectors[live].astype(np.float32)
    gt = np.empty((len(q), eff_k), dtype=np.int64)
    for i, qi in enumerate(q):
        sims = base @ qi.astype(np.float32)
        top = np.argpartition(-sims, kth=eff_k - 1)[:eff_k]
        order = np.lexsort((live[top], -sims[top]))
        gt[i] = live[top[order]]
    return gt


def _benchmark_queries(units: list[Segment], queries: np.ndarray,
                       cfg_q: QueryConfig) -> tuple[Counters, list[int], list[list[int]]]:
    """Warm-up (untimed) then a timed pass; returns total query counters."""
    warm = Counters()
    for q in queries:
        warm.reset_search()
        search(q, units, cfg_q, warm)

    total = Counters()
    latencies_ns: list[int] = []
    results: list[list[int]] = []
    for q in queries:
        warm.reset_search()
        t0 = time.perf_counter_ns()
        res = search(q, units, cfg_q, warm)
        latencies_ns.append(time.perf_counter_ns() - t0)
        total = total + warm.snapshot()
        results.append([int(v) for v, _ in res])
    return total, latencies_ns, results


def _stats_ms(values_ns: list[int]) -> dict:
    if not values_ns:
        return {"p50_ms": float("nan"), "p95_ms": float("nan"), "mean_ms": float("nan")}
    a = np.asarray(values_ns, dtype=np.float64) / 1e6
    return {
        "p50_ms": float(np.percentile(a, 50)),
        "p95_ms": float(np.percentile(a, 95)),
        "mean_ms": float(a.mean()),
    }


def _hierarchy_valid(unit: Segment) -> bool:
    levels = unit.levels
    if not levels:
        return False
    for li in range(len(levels) - 1):
        up, down = levels[li], levels[li + 1]
        if len(up.children) != up.centroids.shape[0]:
            return False
        n_child = down.centroids.shape[0]
        for ch in up.children:
            if len(ch) and (int(ch.min()) < 0 or int(ch.max()) >= n_child):
                return False
    if levels[-1].children:
        return False
    return True


def _index_bytes_estimate(units: list[Segment]) -> int:
    total = 0
    for u in units:
        for lvl in u.levels:
            total += int(lvl.centroids.nbytes)
            for ch in lvl.children:
                total += int(ch.nbytes)
        for p in u.postings:
            total += int(p.centroid.nbytes) + int(p.members.nbytes) + int(p.owned.nbytes)
    return total


# ---------------------------------------------------------------------------
# Per-strategy run
# ---------------------------------------------------------------------------

@dataclass
class RunResult:
    config: CompareConfig
    strategy: str
    run_index: int
    order_in_pair: int
    rows: list[dict] = field(default_factory=list)
    failed: bool = False
    failure_reason: str = ""
    error: str = ""

    @property
    def final_row(self) -> dict | None:
        return self.rows[-1] if self.rows else None


def run_strategy(
    cfg: CompareConfig,
    strategy: str,
    *,
    vectors: np.ndarray,
    queries: np.ndarray,
    trace: ChurnTrace,
    initial_segment: Segment,
    run_index: int,
    order_in_pair: int,
    checkpoint_interval: int,
    flush_final: bool,
    continue_on_audit_failure: bool,
    experiment_id: str,
    dataset_checksum: str,
    progress: bool = True,
) -> RunResult:
    """Run one strategy over the shared workload; return checkpoint rows."""
    cfg_idx = _index_config(cfg)
    cfg_m = _maintain_config(cfg, strategy)
    cfg_q = _query_config(cfg)
    result = RunResult(config=cfg, strategy=strategy, run_index=run_index,
                       order_in_pair=order_in_pair)

    try:
        counters = Counters()          # maintenance counters (never reset)
        if strategy == "inplace":
            maint: IndexMaintainer = InPlaceMaintainer(
                vectors, cfg_idx, cfg_m, counters,
                initial_segment=initial_segment.clone(),
            )
        else:
            maint = OutOfPlaceMaintainer(
                vectors, cfg_idx, cfg_m, counters,
                initial_segment=initial_segment.clone(),
            )
    except Exception as exc:                                    # pragma: no cover
        result.failed = True
        result.error = f"construction failed: {exc!r}"
        return result

    live: set[int] = set(range(cfg.initial_size))
    prev_maint = _maint_snapshot(counters)
    insert_ns: list[int] = []
    delete_ns: list[int] = []
    update_wall_ns = 0
    maintain_call_ns = 0
    maint_attributed_ns = 0
    maintain_call_events = 0
    cum_inserts = 0
    cum_deletes = 0
    cum_ops = 0
    cp_idx = 0

    def checkpoint(cp: int, ops: int) -> None:
        nonlocal prev_maint
        row, failed = _evaluate_checkpoint(
            cfg=cfg, strategy=strategy, cfg_q=cfg_q, cfg_idx=cfg_idx, cfg_m=cfg_m,
            maint=maint, counters=counters, prev_maint=prev_maint,
            units=None, live=live, queries=queries, vectors=vectors,
            insert_ns=insert_ns, delete_ns=delete_ns,
            update_wall_ns=update_wall_ns, maintain_call_ns=maintain_call_ns,
            maint_attributed_ns=maint_attributed_ns,
            maintain_call_events=maintain_call_events,
            checkpoint=cp, ops=ops, cum_inserts=cum_inserts,
            cum_deletes=cum_deletes,
            experiment_id=experiment_id, run_index=run_index,
            order_in_pair=order_in_pair, trace=trace,
            dataset_checksum=dataset_checksum,
        )
        result.rows.append(row)
        prev_maint = _maint_snapshot(counters)
        if failed:
            result.failed = True
            result.failure_reason = row.get("failure_reason", "audit failure")

    # Checkpoint 0 (both arms identical pre-churn)
    checkpoint(0, 0)
    if result.failed and not continue_on_audit_failure:
        return result

    total_ops = len(trace.operations)
    for i, (op_type, vid) in enumerate(trace.operations):
        before_op = _maint_snapshot(counters)
        t0 = time.perf_counter_ns()
        if op_type == "insert":
            maint.insert(int(vid), vectors[int(vid)])
            live.add(int(vid))
            cum_inserts += 1
        else:
            maint.delete(int(vid))
            live.discard(int(vid))
            cum_deletes += 1

        t1 = time.perf_counter_ns()
        t2 = time.perf_counter_ns()
        maint.maintain()
        t3 = time.perf_counter_ns()
        m1 = _maint_snapshot(counters)

        dur = t1 - t0
        update_wall_ns += dur
        maintain_call_ns += (t3 - t2)
        if t3 > t2:
            maintain_call_events += 1
        if _snapshot_advanced(before_op, m1):
            # best-effort attribution: the whole op (or maintain call) triggered
            # structural maintenance; we cannot split inside the call.
            maint_attributed_ns += dur + (t3 - t2)
        if op_type == "insert":
            insert_ns.append(dur)
        else:
            delete_ns.append(dur)
        cum_ops += 1

        is_checkpoint = (cum_ops % checkpoint_interval == 0) or (cum_ops == total_ops)
        if is_checkpoint:
            cp_idx += 1
            checkpoint(cp_idx, cum_ops)
            if progress:
                _print_progress(result.rows[-1])
            if result.failed and not continue_on_audit_failure:
                return result

    if flush_final and strategy == "outofplace":
        try:
            maint.flush()
        except Exception:
            pass

    return result


# ---------------------------------------------------------------------------
# Checkpoint evaluation
# ---------------------------------------------------------------------------

def _evaluate_checkpoint(
    *, cfg: CompareConfig, strategy: str, cfg_q: QueryConfig,
    cfg_idx: IndexConfig, cfg_m: MaintainConfig, maint: IndexMaintainer,
    counters: Counters, prev_maint: dict, units: list[Segment] | None,
    live: set[int], queries: np.ndarray, vectors: np.ndarray,
    insert_ns: list[int], delete_ns: list[int], update_wall_ns: int,
    maintain_call_ns: int, maint_attributed_ns: int, maintain_call_events: int,
    checkpoint: int, ops: int, cum_inserts: int, cum_deletes: int,
    experiment_id: str, run_index: int, order_in_pair: int, trace: ChurnTrace,
    dataset_checksum: str,
) -> tuple[dict, bool]:
    """Evaluate the maintainer at one checkpoint; return (row, failed)."""
    if units is None:
        units = maint.searchable_units()

    # -- correctness audits (read-only) -----------------------------------
    audit = audit_maintainer(maint, expected_live_vids=live)
    membership_status = "n/a"
    membership_violations = 0
    if strategy == "inplace":
        mem = audit_membership(maint)
        membership_status = "PASS" if mem.passed else "FAIL"
        membership_violations = mem.violation_count

    # -- dynamic ground truth + recall ------------------------------------
    live_arr = np.array(sorted(live), dtype=np.int64)
    n_live = len(live_arr)
    gt = dynamic_ground_truth(queries, vectors, live_arr, k=max(cfg.recall_ks))
    q_total, q_lat_ns, q_results = _benchmark_queries(units, queries, cfg_q)

    recall_fields: dict[str, Any] = {}
    for kk in cfg.recall_ks:
        if n_live == 0:
            recall_fields[f"recall_at_{kk}"] = float("nan")
        else:
            recall_fields[f"recall_at_{kk}"] = float(recall_at_k(q_results, gt, kk))
    primary_recall = recall_fields.get(f"recall_at_{cfg.k}", float("nan"))

    # -- deleted / non-live id leak ---------------------------------------
    returned: set[int] = set()
    for res in q_results:
        returned.update(res)
    leaked = sorted(returned - live)
    deleted_retrievals = len(leaked)

    # -- storage ----------------------------------------------------------
    live_mask = live
    m_live = 0                      # live membership references (incl. replicas)
    canonical_live = 0
    leaf_total = 0
    for u in units:
        for p in u.postings:
            leaf_total += 1
            for v in p.members.tolist():
                if int(v) in live_mask:
                    m_live += 1
            for v in p.owned.tolist():
                if int(v) in live_mask:
                    canonical_live += 1
    bloat_pct = (100.0 * (m_live - n_live) / n_live) if n_live else 0.0
    replica_count = m_live - canonical_live
    hierarchy_ok = all(_hierarchy_valid(u) for u in units)
    stats = maint.stats()
    pending_memtable = int(stats.get("memtable_size", 0))

    # -- work measured by the query pass ----------------------------------
    n_q = max(1, len(queries))
    routing = q_total.distance_computations.get("routing", 0)
    scan = q_total.distance_computations.get("scan", 0)

    # -- maintenance deltas since previous checkpoint ---------------------
    cur_maint = _maint_snapshot(counters)
    maint_delta = _snapshot_delta(prev_maint, cur_maint)

    qstats = _stats_ms(q_lat_ns)
    istats = _stats_ms(insert_ns)
    dstats = _stats_ms(delete_ns)
    total_mut_wall = update_wall_ns + maintain_call_ns
    throughput = (ops / (total_mut_wall / 1e9)) if total_mut_wall > 0 else float("nan")

    strategy_correct = (
        audit.passed
        and leaked == []
        and membership_status in ("PASS", "n/a")
        and hierarchy_ok
        and canonical_live == n_live
    )
    correctness_status = "PASSED" if strategy_correct else "FAILED"
    failure_reason = ""
    if not strategy_correct:
        bits = []
        if not audit.passed:
            bits.append(
                f"audit(passed={audit.passed}, npa={len(audit.npa_violations)}, "
                f"own={len(audit.ownership_errors)}, "
                f"cov={len(audit.coverage_errors)})"
            )
        if leaked:
            bits.append(f"{len(leaked)} non-live ids returned (e.g. {leaked[:5]})")
        if membership_status == "FAIL":
            bits.append(f"membership violations={membership_violations}")
        if not hierarchy_ok:
            bits.append("invalid hierarchy references")
        if canonical_live != n_live:
            bits.append(f"canonical owners {canonical_live} != live {n_live}")
        failure_reason = "; ".join(bits)

    row: dict[str, Any] = {
        # identity / config
        "experiment_id": experiment_id,
        "run_index": run_index,
        "order_in_pair": order_in_pair,
        "strategy": strategy,
        "dataset": cfg.dataset,
        "dataset_checksum": dataset_checksum,
        "seed": cfg.seed,
        "trace_seed": trace.seed,
        "trace_hash": trace.trace_hash,
        "workload": cfg.workload,
        "insert_prob": cfg.insert_prob,
        "depth_config": cfg.depth if cfg.depth is not None else -1,
        "depth_actual": max((len(u.levels) for u in units), default=0),
        "leaf_size": cfg.leaf_size,
        "branching_factor": cfg.branching_factor,
        "split_threshold": cfg.split_threshold,
        "merge_threshold": cfg.merge_threshold if cfg.merge_threshold is not None else -1,
        "delta": cfg.delta,
        "memtable_limit": cfg.memtable_limit,
        "compaction_threshold": cfg.compaction_threshold,
        "k": cfg.k,
        "beam_widths": str(list(cfg.beam_widths)),
        "n_postings": cfg.n_postings,
        "checkpoint": checkpoint,
        "operations_processed": ops,
        "cumulative_insertions": cum_inserts,
        "cumulative_deletions": cum_deletes,
        "live_vector_count": n_live,

        # search quality
        "recall": primary_recall,
        **recall_fields,
        "gt_recomputed": True,

        # query performance (measurement interval = this checkpoint)
        "query_count": len(queries),
        "query_latency_p50_ms": qstats["p50_ms"],
        "query_latency_p95_ms": qstats["p95_ms"],
        "query_latency_mean_ms": qstats["mean_ms"],
        "work_vectors_scanned_per_query": q_total.vectors_scanned / n_q,
        "work_scan_sims_per_query": scan / n_q,
        "work_routing_sims_per_query": routing / n_q,
        "work_postings_read_per_query": q_total.postings_read / n_q,
        "work_bytes_read_per_query": q_total.bytes_read / n_q,
        "work_vectors_scanned_total": q_total.vectors_scanned,
        "work_sim_evals_total": scan + routing,
        "units_searched": len(units),
        "postings_budget": cfg.n_postings * len(units),

        # update performance (cumulative-to-date)
        "insert_count": len(insert_ns),
        "insert_latency_p50_ms": istats["p50_ms"],
        "insert_latency_p95_ms": istats["p95_ms"],
        "insert_latency_mean_ms": istats["mean_ms"],
        "delete_count": len(delete_ns),
        "delete_latency_p50_ms": dstats["p50_ms"],
        "delete_latency_p95_ms": dstats["p95_ms"],
        "delete_latency_mean_ms": dstats["mean_ms"],
        "update_wall_ms": update_wall_ns / 1e6,
        "maintain_call_wall_ms": maintain_call_ns / 1e6,
        "maintenance_wall_ms": maint_attributed_ns / 1e6,
        "total_mutation_wall_ms": total_mut_wall / 1e6,
        "maintain_call_events": maintain_call_events,
        "mutation_throughput_ops_per_s": throughput,

        # storage / structure
        "leaf_posting_count_total": leaf_total,
        "searchable_segment_count": len(units),
        "total_live_membership_count": m_live,
        "canonical_owned_live_count": canonical_live,
        "replica_count": replica_count,
        "membership_bloat_pct": bloat_pct,
        "pending_memtable_entries": pending_memtable,
        "index_bytes_estimate": _index_bytes_estimate(units),

        # maintenance counters (cumulative) + deltas since last checkpoint
        **_maint_cumulative_fields(cur_maint),
        **{f"delta_{k}": v for k, v in maint_delta.items()},

        # correctness
        "correctness_status": correctness_status,
        "npa_violations": len(audit.npa_violations),
        "ownership_errors": len(audit.ownership_errors),
        "coverage_errors": len(audit.coverage_errors),
        "membership_status": membership_status,
        "membership_violations": membership_violations,
        "deleted_id_retrievals": deleted_retrievals,
        "hierarchy_valid": hierarchy_ok,
        "run_failed": (not strategy_correct),
        "failure_reason": failure_reason,
    }
    return row, (not strategy_correct)


# ---------------------------------------------------------------------------
# Counter snapshots / deltas
# ---------------------------------------------------------------------------

def _maint_snapshot(c: Counters) -> dict:
    snap = c.maintain_dict()
    snap["maint_dist"] = int(sum(c.distance_computations.values()))
    return snap


def _snapshot_delta(before: dict, after: dict) -> dict:
    return {k: int(after.get(k, 0) - before.get(k, 0)) for k in after}


def _snapshot_advanced(before: dict, after: dict) -> bool:
    for k, v in after.items():
        if k == "maint_dist":
            if v != before.get(k, 0):
                return True
        elif k in ("replicas_added", "replicas_removed", "vectors_reassigned",
                   "membership_vectors_recomputed", "splits", "merges",
                   "hierarchy_rebuilds", "compaction_events"):
            if v != before.get(k, 0):
                return True
    return False


def _maint_cumulative_fields(snap: dict) -> dict:
    return {
        "splits": snap["splits"],
        "merges": snap["merges"],
        "hierarchy_rebuilds": snap["hierarchy_rebuilds"],
        "vectors_reassigned": snap["vectors_reassigned"],
        "replicas_added": snap["replicas_added"],
        "replicas_removed": snap["replicas_removed"],
        "membership_vectors_recomputed": snap["membership_vectors_recomputed"],
        "merge_vectors_reassigned": snap["merge_vectors_reassigned"],
        "merge_candidates_examined": snap["merge_candidates_examined"],
        "compaction_events": snap["compaction_events"],
        "vectors_reclustered": snap["vectors_reclustered"],
        "flushes": snap["flushes"],
        "forced_checkpoint_flushes": snap["forced_checkpoint_flushes"],
        "prunes": float("nan"),            # not instrumented (n/a)
        "maintenance_similarity_evaluations": snap["maint_dist"],
    }


# ---------------------------------------------------------------------------
# Progress printing
# ---------------------------------------------------------------------------

def _print_progress(row: dict) -> None:
    print(
        f"[cmp] run={row['run_index']:<3} {row['strategy']:<10} "
        f"depth={row['depth_config']:<2} split={row['split_threshold']:<4} "
        f"delta={row['delta']!s:<4} seed={row['seed']:<3} "
        f"cp={row['checkpoint']:<2} op={row['operations_processed']:<4} "
        f"live={row['live_vector_count']:<4} recall@{row['k']}={row['recall']:.4f} "
        f"{row['correctness_status']}",
        flush=True,
    )


# ---------------------------------------------------------------------------
# Sweep execution
# ---------------------------------------------------------------------------

def run_sweep(spec: SweepSpec, *, progress: bool = True) -> list[RunResult]:
    """Run every cell for both strategies, alternating arm order per run."""
    cache = _DatasetCache()
    cells = spec.expand()
    results: list[RunResult] = []
    experiment_id = _new_experiment_id()

    if progress:
        print(f"Experiment id: {experiment_id}")
        print(f"Planned runs: {len(cells)} cells x 2 strategies = {len(cells) * 2}")
        print(f"Profile: {spec.profile}  dataset={spec.dataset} "
              f"initial={spec.initial_size} ops={spec.operations} "
              f"interval={spec.checkpoint_interval}")

    for run_index, cfg in enumerate(cells):
        vectors = cache.data(cfg)
        queries = cache.queries(cfg)
        initial_segment = cache.initial_segment(cfg)
        trace = generate_churn_trace(
            initial_n=cfg.initial_size, reserve_n=cfg.reserve_size,
            n_operations=cfg.operations, insert_prob=cfg.insert_prob,
            seed=cfg.seed, workload=cfg.workload,
        )
        dcs = _dataset_checksum(vectors)

        order = list(STRATEGIES)
        if run_index % 2 == 1:              # alternate arm order to limit bias
            order.reverse()

        for pos, strategy in enumerate(order):
            if progress:
                print(f"--- cell {run_index + 1}/{len(cells)} "
                      f"strategy={strategy} (order {pos + 1}/2) ---")
            try:
                rr = run_strategy(
                    cfg, strategy, vectors=vectors, queries=queries, trace=trace,
                    initial_segment=initial_segment, run_index=run_index,
                    order_in_pair=pos, checkpoint_interval=spec.checkpoint_interval,
                    flush_final=spec.flush_final,
                    continue_on_audit_failure=spec.continue_on_audit_failure,
                    experiment_id=experiment_id, dataset_checksum=dcs,
                    progress=progress,
                )
            except Exception as exc:                            # pragma: no cover
                rr = RunResult(config=cfg, strategy=strategy, run_index=run_index,
                               order_in_pair=pos, failed=True,
                               error=f"{exc!r}\n{traceback.format_exc()}")
                if progress:
                    print(f"!! run {run_index} {strategy} crashed: {exc!r}",
                          file=sys.stderr)
            results.append(rr)
    return results


def _new_experiment_id() -> str:
    stamp = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
    return f"cmp-{stamp}-{os.getpid()}"


# ---------------------------------------------------------------------------
# Aggregation
# ---------------------------------------------------------------------------

def rows_from_results(results: list[RunResult]) -> list[dict]:
    rows: list[dict] = []
    for r in results:
        rows.extend(r.rows)
    return rows


def build_paired(rows: list[dict]) -> list[dict]:
    """Side-by-side in-place vs out-of-place at each matching checkpoint.

    Sign convention (explicit): every ``*_diff`` is **in-place minus
    out-of-place**; every ``*_ratio`` is **in-place / out-of-place** (so > 1
    means in-place is slower/larger).  ``recall_diff_pp`` is in percentage
    points.
    """
    groups: dict[tuple, dict[str, dict]] = {}
    for row in rows:
        key = tuple(sorted((k, str(v)) for k, v in row.items()
                           if k in _PAIR_KEYS))
        groups.setdefault(key, {})[row["strategy"]] = row

    paired: list[dict] = []
    for key, by in groups.items():
        ip, oop = by.get("inplace"), by.get("outofplace")
        if ip is None or oop is None:
            continue
        pair = {k: v for k, v in key}
        pair.update({
            "seed": ip["seed"],
            "depth_config": ip["depth_config"],
            "split_threshold": ip["split_threshold"],
            "merge_threshold": ip["merge_threshold"],
            "delta": ip["delta"],
            "n_postings": ip["n_postings"],
            "beam_widths": ip["beam_widths"],
            "checkpoint": ip["checkpoint"],
            "operations_processed": ip["operations_processed"],
            "live_vector_count": ip["live_vector_count"],
            "inplace_recall": ip["recall"],
            "outofplace_recall": oop["recall"],
            "recall_diff_pp": (ip["recall"] - oop["recall"]) * 100.0,
            "inplace_query_p50_ms": ip["query_latency_p50_ms"],
            "outofplace_query_p50_ms": oop["query_latency_p50_ms"],
            "query_p50_ratio": _ratio(ip["query_latency_p50_ms"],
                                      oop["query_latency_p50_ms"]),
            "inplace_query_p95_ms": ip["query_latency_p95_ms"],
            "outofplace_query_p95_ms": oop["query_latency_p95_ms"],
            "query_p95_ratio": _ratio(ip["query_latency_p95_ms"],
                                      oop["query_latency_p95_ms"]),
            "inplace_insert_p50_ms": ip["insert_latency_p50_ms"],
            "outofplace_insert_p50_ms": oop["insert_latency_p50_ms"],
            "insert_p50_ratio": _ratio(ip["insert_latency_p50_ms"],
                                       oop["insert_latency_p50_ms"]),
            "inplace_delete_p50_ms": ip["delete_latency_p50_ms"],
            "outofplace_delete_p50_ms": oop["delete_latency_p50_ms"],
            "delete_p50_ratio": _ratio(ip["delete_latency_p50_ms"],
                                       oop["delete_latency_p50_ms"]),
            "inplace_maintenance_wall_ms": ip["maintenance_wall_ms"],
            "outofplace_maintenance_wall_ms": oop["maintenance_wall_ms"],
            "maintenance_wall_ratio": _ratio(ip["maintenance_wall_ms"],
                                             oop["maintenance_wall_ms"]),
            "inplace_vectors_scanned_per_query": ip["work_vectors_scanned_per_query"],
            "outofplace_vectors_scanned_per_query": oop["work_vectors_scanned_per_query"],
            "work_scan_ratio": _ratio(ip["work_vectors_scanned_per_query"],
                                      oop["work_vectors_scanned_per_query"]),
            "work_diff_vectors": (ip["work_vectors_scanned_per_query"]
                                  - oop["work_vectors_scanned_per_query"]),
            "inplace_bloat_pct": ip["membership_bloat_pct"],
            "outofplace_bloat_pct": oop["membership_bloat_pct"],
            "bloat_diff_pp": ip["membership_bloat_pct"] - oop["membership_bloat_pct"],
            "inplace_index_bytes": ip["index_bytes_estimate"],
            "outofplace_index_bytes": oop["index_bytes_estimate"],
            "index_bytes_diff": (ip["index_bytes_estimate"]
                                 - oop["index_bytes_estimate"]),
            "inplace_correctness": ip["correctness_status"],
            "outofplace_correctness": oop["correctness_status"],
        })
        paired.append(pair)
    paired.sort(key=lambda p: (str(p.get("seed")), str(p.get("depth")),
                               str(p.get("split_threshold")), str(p.get("delta")),
                               p["checkpoint"]))
    return paired


_PAIR_KEYS = (
    "dataset", "seed", "depth_config", "leaf_size", "split_threshold",
    "merge_threshold", "delta", "memtable_limit", "compaction_threshold",
    "k", "beam_widths", "n_postings", "checkpoint",
)


def _ratio(a, b) -> float:
    try:
        if b is None or b == 0 or np.isnan(b):
            return float("nan")
        return float(a) / float(b)
    except Exception:
        return float("nan")


def build_summary(rows: list[dict]) -> list[dict]:
    """Per-run aggregate and configuration-level (across-seed) aggregate rows.

    Each output row has ``row_kind``: ``run`` (one per strategy x config x seed,
    metrics aggregated across that run's checkpoints) or ``aggregate`` (across
    the successful repetitions of a config).  Failed runs are excluded from the
    aggregate but counted in ``n_failed``; they are never silently dropped.
    """
    # -- per-run rows ------------------------------------------------------
    run_groups: dict[tuple, list[dict]] = {}
    for row in rows:
        key = (row["run_index"], row["strategy"])
        run_groups.setdefault(key, []).append(row)

    run_rows: list[dict] = []
    for (run_index, strategy), cps in run_groups.items():
        cps = sorted(cps, key=lambda r: r["checkpoint"])
        final = cps[-1]
        run_rows.append({
            "row_kind": "run",
            "run_index": run_index,
            "strategy": strategy,
            "dataset": final["dataset"],
            "seed": final["seed"],
            "depth_config": final["depth_config"],
            "split_threshold": final["split_threshold"],
            "merge_threshold": final["merge_threshold"],
            "delta": final["delta"],
            "n_postings": final["n_postings"],
            "beam_widths": final["beam_widths"],
            "n_checkpoints": len(cps),
            "final_recall": final["recall"],
            "final_membership_bloat_pct": final["membership_bloat_pct"],
            "final_live_vector_count": final["live_vector_count"],
            "final_index_bytes_estimate": final["index_bytes_estimate"],
            "insert_latency_p50_ms": final["insert_latency_p50_ms"],
            "delete_latency_p50_ms": final["delete_latency_p50_ms"],
            "query_latency_p50_ms": final["query_latency_p50_ms"],
            "maintenance_wall_ms": final["maintenance_wall_ms"],
            "total_mutation_wall_ms": final["total_mutation_wall_ms"],
            "maintenance_similarity_evaluations":
                final["maintenance_similarity_evaluations"],
            "final_correctness_status": final["correctness_status"],
            "run_failed": final["run_failed"],
        })

    # -- configuration-level aggregate (across seeds, final checkpoint) ----
    agg_groups: dict[tuple, list[dict]] = {}
    for row in rows:
        if row["checkpoint"] != max(
            r["checkpoint"] for r in run_groups[(row["run_index"], row["strategy"])]
        ):
            continue
        key = (row["dataset"], row["depth_config"], row["split_threshold"],
               row["merge_threshold"], row["delta"], row["strategy"],
               row["n_postings"], row["beam_widths"])
        agg_groups.setdefault(key, []).append(row)

    agg_rows: list[dict] = []
    for key, rs in agg_groups.items():
        (dataset, depth, split, merge, delta, strategy, npost, beam) = key
        ok = [r for r in rs if not r["run_failed"] and r["correctness_status"] == "PASSED"]
        n_failed = len(rs) - len(ok)
        base = {
            "row_kind": "aggregate",
            "dataset": dataset, "depth_config": depth,
            "split_threshold": split, "merge_threshold": merge,
            "delta": delta, "strategy": strategy, "n_postings": npost,
            "beam_widths": beam, "n_reps": len(rs), "n_ok": len(ok),
            "n_failed": n_failed,
        }
        base.update(_mean_std(ok, "recall", "recall"))
        base.update(_mean_std(ok, "query_latency_p50_ms", "query_p50_ms"))
        base.update(_mean_std(ok, "insert_latency_p50_ms", "insert_p50_ms"))
        base.update(_mean_std(ok, "delete_latency_p50_ms", "delete_p50_ms"))
        base.update(_mean_std(ok, "maintenance_wall_ms", "maintenance_wall_ms"))
        base.update(_mean_std(ok, "membership_bloat_pct", "membership_bloat_pct"))
        base.update(_mean_std(ok, "work_vectors_scanned_per_query", "vectors_scanned_per_query"))
        agg_rows.append(base)

    return run_rows + agg_rows


def _mean_std(rows: list[dict], key: str, prefix: str) -> dict:
    vals = [float(r[key]) for r in rows
            if r.get(key) is not None and not (isinstance(r[key], float)
                                               and np.isnan(r[key]))]
    if not vals:
        return {f"{prefix}_mean": float("nan"), f"{prefix}_std": float("nan")}
    a = np.asarray(vals, dtype=np.float64)
    return {f"{prefix}_mean": float(a.mean()),
            f"{prefix}_std": float(a.std(ddof=0))}


# ---------------------------------------------------------------------------
# Terminal tables
# ---------------------------------------------------------------------------

def _fmt_cell(value: Any, width: int, fmt: str) -> str:
    if value is None:
        s = "n/a"
    elif isinstance(value, float) and np.isnan(value):
        s = "n/a"
    elif fmt == "f4":
        s = f"{value:.4f}"
    elif fmt == "f2":
        s = f"{value:.2f}"
    elif fmt == "f1":
        s = f"{value:.1f}"
    elif fmt == "s":
        s = str(value)
    else:
        s = str(value)
    return s[:width].rjust(width)


def render_table(title: str, columns: list[tuple[str, str, int, str]],
                 rows: list[dict]) -> str:
    """Render a simple fixed-width text table.  columns = (header, key, width, fmt)."""
    lines = [title, "-" * (sum(c[2] for c in columns) + len(columns))]
    lines.append(" ".join(h.rjust(w) for h, _, w, _ in columns))
    for r in rows:
        lines.append(" ".join(_fmt_cell(r.get(key), w, fmt)
                              for _, key, w, fmt in columns))
    return "\n".join(lines)


_TABLE_A_COLUMNS = [
    ("run", "run_index", 4, "s"),
    ("strategy", "strategy", 10, "s"),
    ("depth", "depth_config", 5, "s"),
    ("split", "split_threshold", 5, "s"),
    ("delta", "delta", 5, "s"),
    ("seed", "seed", 4, "s"),
    ("cp", "checkpoint", 3, "s"),
    ("ops", "operations_processed", 4, "s"),
    ("live", "live_vector_count", 5, "s"),
    ("recall", "recall", 7, "f4"),
    ("q_p50ms", "query_latency_p50_ms", 8, "f2"),
    ("q_p95ms", "query_latency_p95_ms", 8, "f2"),
    ("ins_p50", "insert_latency_p50_ms", 8, "f2"),
    ("del_p50", "delete_latency_p50_ms", 8, "f2"),
    ("mut_ms", "total_mutation_wall_ms", 9, "f1"),
    ("scan/q", "work_vectors_scanned_per_query", 8, "f1"),
    ("bloat%", "membership_bloat_pct", 7, "f1"),
    ("split#", "splits", 6, "s"),
    ("merge#", "merges", 6, "s"),
    ("flush/cp", "compaction_events", 8, "s"),
    ("npa", "npa_violations", 4, "s"),
    ("mem", "membership_status", 4, "s"),
    ("status", "correctness_status", 7, "s"),
]

_TABLE_B_COLUMNS = [
    ("seed", "seed", 4, "s"),
    ("depth", "depth_config", 5, "s"),
    ("split", "split_threshold", 5, "s"),
    ("delta", "delta", 5, "s"),
    ("cp", "checkpoint", 3, "s"),
    ("ip_recall", "inplace_recall", 9, "f4"),
    ("oop_recall", "outofplace_recall", 9, "f4"),
    ("dRec_pp", "recall_diff_pp", 8, "f2"),
    ("q_p50_ratio", "query_p50_ratio", 10, "f2"),
    ("ins_ratio", "insert_p50_ratio", 9, "f2"),
    ("del_ratio", "delete_p50_ratio", 9, "f2"),
    ("mnt_ratio", "maintenance_wall_ratio", 10, "f2"),
    ("scan_ratio", "work_scan_ratio", 10, "f2"),
    ("dBloat_pp", "bloat_diff_pp", 9, "f2"),
    ("dBytes", "index_bytes_diff", 8, "s"),
    ("ip_corr", "inplace_correctness", 8, "s"),
    ("oop_corr", "outofplace_correctness", 9, "s"),
]

_AGG_COLUMNS = [
    ("strategy", "strategy", 10, "s"),
    ("depth", "depth_config", 5, "s"),
    ("split", "split_threshold", 5, "s"),
    ("delta", "delta", 5, "s"),
    ("nq", "n_reps", 3, "s"),
    ("nok", "n_ok", 3, "s"),
    ("nfail", "n_failed", 5, "s"),
    ("recall_mean", "recall_mean", 11, "f4"),
    ("recall_std", "recall_std", 10, "f4"),
    ("q_p50_mean", "query_p50_ms_mean", 11, "f2"),
    ("ins_p50_mean", "insert_p50_ms_mean", 12, "f2"),
    ("del_p50_mean", "delete_p50_ms_mean", 12, "f2"),
    ("mnt_ms_mean", "maintenance_wall_ms_mean", 11, "f2"),
    ("bloat_mean", "membership_bloat_pct_mean", 10, "f2"),
    ("scan_mean", "vectors_scanned_per_query_mean", 10, "f1"),
]


# ---------------------------------------------------------------------------
# Output persistence
# ---------------------------------------------------------------------------

def _jsonify_rows(rows: list[dict]) -> list[dict]:
    out = []
    for r in rows:
        d = {}
        for k, v in r.items():
            if isinstance(v, (np.integer,)):
                v = int(v)
            elif isinstance(v, (np.floating,)):
                v = None if np.isnan(v) else float(v)
            elif isinstance(v, float) and np.isnan(v):
                v = None
            elif isinstance(v, (tuple, list)):
                v = str(v)
            d[k] = v
        out.append(d)
    return out


def write_csv(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    keys: list[str] = []
    seen: set[str] = set()
    for r in rows:
        for k in r:
            if k not in seen:
                seen.add(k)
                keys.append(k)
    with open(path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=keys)
        w.writeheader()
        for r in rows:
            w.writerow({k: ("" if r.get(k) is None else r.get(k)) for k in keys})


def write_parquet(path: Path, rows: list[dict]) -> bool:
    try:
        import pyarrow as pa
        import pyarrow.parquet as pq
    except Exception:
        return False
    if not rows:
        return False
    table = pa.table(_columnar(rows))
    pq.write_table(table, str(path))
    return True


def _columnar(rows: list[dict]) -> dict:
    keys: list[str] = []
    seen: set[str] = set()
    for r in rows:
        for k in r:
            if k not in seen:
                seen.add(k)
                keys.append(k)
    cols: dict[str, list] = {}
    for k in keys:
        vals = [r.get(k) for r in rows]
        nonnull = [v for v in vals if v is not None
                   and not (isinstance(v, float) and np.isnan(v))]
        if nonnull and all(isinstance(v, bool) for v in nonnull):
            cols[k] = [None if v is None else bool(v) for v in vals]
        elif nonnull and all(isinstance(v, (int, np.integer)) and not isinstance(v, bool)
                             for v in nonnull):
            cols[k] = [None if v is None else int(v) for v in vals]
        elif nonnull and all(isinstance(v, (int, float, np.integer, np.floating))
                             and not isinstance(v, bool) for v in nonnull):
            cols[k] = [None if (v is None or (isinstance(v, float) and np.isnan(v)))
                       else float(v) for v in vals]
        else:
            cols[k] = [None if v is None else str(v) for v in vals]
    return cols


def write_outputs(out_dir: Path, raw_rows: list[dict], paired: list[dict],
                  summary: list[dict], metadata: dict) -> dict:
    out_dir.mkdir(parents=True, exist_ok=True)
    ck = out_dir / "results_checkpoints.csv"
    raw = out_dir / "results_raw.csv"
    pair = out_dir / "results_paired.csv"
    summ = out_dir / "results_summary.csv"
    meta = out_dir / "run_metadata.json"

    write_csv(ck, raw_rows)
    write_csv(raw, raw_rows)
    write_csv(pair, paired)
    write_csv(summ, summary)
    with open(meta, "w") as f:
        json.dump(metadata, f, indent=2, default=str)

    written = {
        "results_checkpoints.csv": len(raw_rows),
        "results_raw.csv": len(raw_rows),
        "results_paired.csv": len(paired),
        "results_summary.csv": len(summary),
        "run_metadata.json": 1,
    }
    if write_parquet(out_dir / "results_checkpoints.parquet", raw_rows):
        written["results_checkpoints.parquet"] = len(raw_rows)
    if write_parquet(out_dir / "results_paired.parquet", paired):
        written["results_paired.parquet"] = len(paired)
    return written


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def _parse_int_list(text: str | None, default: list[int]) -> list[int]:
    if text is None or text == "":
        return list(default)
    return [int(x) for x in str(text).split(",") if x != ""]


def _parse_delta_list(text: str | None, default: list[float | None]) -> list[float | None]:
    if text is None or text == "":
        return list(default)
    out: list[float | None] = []
    for part in str(text).split(","):
        p = part.strip().lower()
        if p in ("", "none", "off", "null"):
            out.append(None)
        else:
            out.append(float(p))
    return out


def _parse_merge_list(text: str | None, default: list[int | None]) -> list[int | None]:
    if text is None or text == "":
        return list(default)
    out: list[int | None] = []
    for part in str(text).split(","):
        p = part.strip().lower()
        if p in ("", "none", "off", "null"):
            out.append(None)
        else:
            out.append(int(p))
    return out


_PROFILE_DEFAULTS = {
    "smoke": dict(initial_size=120, operations=60, checkpoint_interval=20,
                  depths=[1, 2], split_thresholds=[24], deltas=[None],
                  n_postings=[2], beam=[8], seeds=[0]),
    "quick": dict(initial_size=400, operations=400, checkpoint_interval=50,
                  depths=[1, 2, 3, 4], split_thresholds=[24, 48],
                  deltas=[None, 0.3], n_postings=[4], beam=[8], seeds=[0]),
    "research": dict(initial_size=400, operations=400, checkpoint_interval=50,
                     depths=[1, 2, 3, 4], split_thresholds=[24, 48],
                     deltas=[None, 0.3], n_postings=[4], beam=[8],
                     seeds=[0, 1, 2]),
}


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="compare_maintenance",
        description="Paired InPlace vs OutOfPlace maintenance benchmark. "
                    "Identical initial index and workload; measured search, "
                    "query/update latency, maintenance and storage.",
    )
    p.add_argument("--profile", choices=["smoke", "quick", "research"],
                   default="quick", help="preset sweep size (default: quick)")
    p.add_argument("--dataset", default="synthetic", choices=["synthetic"])
    p.add_argument("--initial-size", dest="initial_size", type=int, default=None)
    p.add_argument("--operations", type=int, default=None)
    p.add_argument("--workload", default="mixed",
                   choices=["mixed", "insert_only", "delete_only"])
    p.add_argument("--insert-prob", dest="insert_prob", type=float, default=0.5)
    p.add_argument("--dim", type=int, default=128)
    p.add_argument("--leaf-size", dest="leaf_size", type=int, default=40)
    p.add_argument("--branching-factor", dest="branching_factor", type=int, default=16)
    p.add_argument("--depths", default=None,
                   help="comma list of depths, e.g. 1,2,3,4 (or 'none')")
    p.add_argument("--split-thresholds", dest="split_thresholds", default=None,
                   help="comma list, e.g. 24,48")
    p.add_argument("--merge-thresholds", dest="merge_thresholds", default=None,
                   help="comma list incl. 'none', e.g. none,10")
    p.add_argument("--deltas", default=None,
                   help="comma list incl. 'none', e.g. none,0.3")
    p.add_argument("--n-postings", dest="n_postings", default=None,
                   help="comma list, e.g. 1,4,8")
    p.add_argument("--beam-widths", dest="beam_widths", default=None,
                   help="comma list, e.g. 4,8")
    p.add_argument("--seeds", default=None, help="comma list, e.g. 0,1,2")
    p.add_argument("--k", type=int, default=10, help="primary recall cutoff")
    p.add_argument("--recall-ks", dest="recall_ks", default=None,
                   help="comma list of recall cutoffs, e.g. 1,10,100 "
                        "(default: --k only); search k = max(list)")
    p.add_argument("--query-count", dest="query_count", type=int, default=32)
    p.add_argument("--checkpoint-interval", dest="checkpoint_interval",
                   type=int, default=None)
    p.add_argument("--memtable-limit", dest="memtable_limit", type=int, default=100)
    p.add_argument("--compaction-threshold", dest="compaction_threshold",
                   type=int, default=3)
    p.add_argument("--out", default="results/compare")
    p.add_argument("--overwrite", action="store_true",
                   help="write directly into --out instead of a unique subdir")
    p.add_argument("--blas-threads", dest="blas_threads", type=int, default=None)
    p.add_argument("--table-a-rows", dest="table_a_rows",
                   choices=["final", "all", "none"], default="final",
                   help="terminal Table A detail (full rows always saved)")
    p.add_argument("--flush-final", dest="flush_final", action="store_true",
                   help="flush the out-of-place MemTable once at the very end "
                        "(recorded; never forced at checkpoints)")
    p.add_argument("--continue-on-audit-failure", dest="continue_on_audit_failure",
                   action="store_true",
                   help="keep a run going after a correctness failure "
                        "(the run is still marked failed)")
    return p


def _depths_from_text(text: str | None, default: list[int | None]) -> list[int | None]:
    if text is None or text == "":
        return list(default)
    out: list[int | None] = []
    for part in str(text).split(","):
        p = part.strip().lower()
        out.append(None if p in ("", "none", "null") else int(p))
    return out


def resolve_sweep(args: argparse.Namespace) -> SweepSpec:
    prof = _PROFILE_DEFAULTS[args.profile]
    initial_size = args.initial_size if args.initial_size is not None else prof["initial_size"]
    operations = args.operations if args.operations is not None else prof["operations"]
    checkpoint_interval = (args.checkpoint_interval if args.checkpoint_interval is not None
                           else prof["checkpoint_interval"])
    depths = _depths_from_text(args.depths, prof["depths"])
    split_thresholds = _parse_int_list(args.split_thresholds, prof["split_thresholds"])
    merge_thresholds = _parse_merge_list(args.merge_thresholds, [None])
    deltas = _parse_delta_list(args.deltas, prof["deltas"])
    n_postings = _parse_int_list(args.n_postings, prof["n_postings"])
    beam = _parse_int_list(args.beam_widths, prof["beam"])
    seeds = _parse_int_list(args.seeds, prof["seeds"])
    recall_ks = tuple(_parse_int_list(args.recall_ks, [args.k]))
    if args.k not in recall_ks:
        recall_ks = tuple(sorted(set(recall_ks) | {args.k}))
    if operations < 1:
        raise ValueError("operations must be >= 1")
    if checkpoint_interval < 1:
        raise ValueError("checkpoint_interval must be >= 1")
    return SweepSpec(
        dataset=args.dataset, initial_size=initial_size, operations=operations,
        workload=args.workload, insert_prob=args.insert_prob, dim=args.dim,
        leaf_size=args.leaf_size, branching_factor=args.branching_factor,
        split_thresholds=split_thresholds, merge_thresholds=merge_thresholds,
        deltas=deltas, depths=depths, n_postings_list=n_postings, beam_list=beam,
        seeds=seeds, k=args.k, recall_ks=recall_ks, query_count=args.query_count,
        memtable_limit=args.memtable_limit,
        compaction_threshold=args.compaction_threshold,
        checkpoint_interval=checkpoint_interval, flush_final=args.flush_final,
        continue_on_audit_failure=args.continue_on_audit_failure,
        blas_threads=args.blas_threads, table_a_rows=args.table_a_rows,
        out_dir=Path(args.out), overwrite=args.overwrite, profile=args.profile,
    )


def _unique_out_dir(base: Path, experiment_id: str, overwrite: bool) -> Path:
    if overwrite:
        return base
    return base / experiment_id


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    try:
        spec = resolve_sweep(args)
    except ValueError as exc:
        parser.error(str(exc))
        return 2

    experiment_id = _new_experiment_id()
    out_dir = _unique_out_dir(spec.out_dir, experiment_id, spec.overwrite)

    cm = None
    if spec.blas_threads is not None:
        try:
            from threadpoolctl import threadpool_limits
            cm = threadpool_limits(limits=spec.blas_threads)
            cm.__enter__()
        except Exception:
            cm = None

    try:
        results = run_sweep(spec, progress=True)
    finally:
        if cm is not None:
            cm.__exit__(None, None, None)

    rows = rows_from_results(results)
    paired = build_paired(rows)
    summary = build_summary(rows)

    metadata = {
        "invocation_id": experiment_id,
        "timestamp": datetime.datetime.now(datetime.timezone.utc).isoformat(),
        "git_sha": _git_sha(),
        "environment": _environment_metadata(spec.blas_threads),
        "rss_bytes": _rss_bytes()[0],
        "rss_kind": _rss_bytes()[1],
        "sweep": spec.to_dict(),
        "flush_policy": ("automatic: out-of-place flushes at memtable_limit via "
                         "insert; maintain() compacts at compaction_threshold; "
                         "no benchmark-only checkpoint flushes"
                         + ("; one final out-of-place flush requested"
                            if spec.flush_final else "")),
        "n_checkpoints_per_run": spec.operations // spec.checkpoint_interval + 1,
        "n_queries": spec.query_count,
        "runs": [
            {"run_index": r.run_index, "strategy": r.strategy,
             "order_in_pair": r.order_in_pair, "failed": r.failed,
             "failure_reason": r.failure_reason, "error": r.error,
             **r.config.to_dict()}
            for r in results
        ],
        "measurement_limitations": [
            "search() uses n_postings per searchable unit; no global posting budget",
            "balanced_kmeans internal distances uncounted (maintenance work is a lower bound)",
            "empty-leaf pruning not instrumented (n/a)",
            "audit_membership is in-place only (out-of-place membership audit n/a)",
            "rss is peak when psutil is unavailable; index bytes are an estimate",
            "update latency includes triggered maintenance; attribution is best effort",
        ],
    }

    written = write_outputs(out_dir, rows, paired, summary, metadata)

    # ---- terminal tables -------------------------------------------------
    print()
    print("=" * 100)
    print(f"Output directory: {out_dir}")
    print(f"Files: {', '.join(f'{k} ({v} rows)' for k, v in written.items())}")

    if spec.table_a_rows != "none" and rows:
        table_a = _select_table_a(rows, spec.table_a_rows)
        print()
        print(render_table("Table A — per-run results (in-place and out-of-place)",
                           _TABLE_A_COLUMNS, table_a))
        print("Sign note: latency/work/bloat are per strategy; 'status' is the "
              "correctness gate (PASSED = coverage + unique ownership + NPA).")

    if paired:
        print()
        print(render_table("Table B — paired comparison (diff = in-place minus "
                           "out-of-place; ratio = in-place / out-of-place)",
                           _TABLE_B_COLUMNS, paired))
        print("Positive dRec_pp favours in-place; ratio > 1 means in-place is "
              "slower/larger (worse).")

    agg = [r for r in summary if r["row_kind"] == "aggregate"]
    if agg:
        print()
        print(render_table("Aggregate across seeds (successful runs only; "
                           "mean ± std)", _AGG_COLUMNS, agg))

    n_failed = sum(1 for r in results if r.failed)
    if n_failed:
        print()
        print(f"WARNING: {n_failed} run(s) failed the correctness gate and were "
              f"excluded from aggregates; see results_raw.csv (run_failed) and "
              f"run_metadata.json.")
    return 0


def _select_table_a(rows: list[dict], mode: str) -> list[dict]:
    if mode == "all":
        return rows
    # "final": last checkpoint of each run
    final_by_run: dict[tuple, dict] = {}
    for r in rows:
        key = (r["run_index"], r["strategy"])
        curr = final_by_run.get(key)
        if curr is None or r["checkpoint"] > curr["checkpoint"]:
            final_by_run[key] = r
    return sorted(final_by_run.values(), key=lambda r: (r["run_index"], r["strategy"]))


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
