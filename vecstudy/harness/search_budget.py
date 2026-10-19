"""harness/search_budget.py — equal-search-work comparison harness.

Backs ``scripts/compare_search_budget.py``.  It answers a question the paired
maintenance benchmark (``harness/compare.py``) cannot: *at matched measured query
work*, how do the two maintenance strategies compare on recall and latency?

Why a separate harness
----------------------
``compare.py`` fixes one ``QueryConfig`` for both arms, so in-place and
out-of-place are compared at *equal nominal* settings, which (because
out-of-place searches several units and in-place searches one) are *not* equal
work.  This harness instead:

1. replays the churn trace **once** per ``(base config, strategy)`` and, at each
   checkpoint, clones the live index state;
2. executes the whole ``beam_width × n_postings`` search-preset grid against that
   *same* cloned checkpoint state, so no churn is ever replayed per preset;
3. matches in-place and out-of-place presets by **observed**
   ``vectors_scanned_per_query``.

Search semantics used here (verified against the code, not assumed)
-------------------------------------------------------------------
* ``QueryConfig.beam_widths`` is a **per-level survivor list**:
  ``hierarchy.descend`` indexes it by routing level and clamps to its last
  element.  A scalar beam width ``B`` therefore translates to the valid
  configuration ``beam_widths=(B,)`` ("``B`` survivors at every routing level").
* At ``depth == 1`` there are no routing levels: ``descend`` takes the depth-1
  fast path and the beam width is **unused** — only ``n_postings`` changes work.
* ``QueryConfig`` is fully separate from ``IndexConfig``/``MaintainConfig``, so
  none of these settings can influence maintenance.  The harness additionally
  asserts this at runtime (search presets must not change any structural
  counter).

Audit asymmetry
---------------
``audit_membership`` is in-place only, but the repository does provide a
suitable out-of-place auditor (:func:`churn.audit_outofplace_membership`), so
this harness wires it in and both strategies get a real fixed-Δ membership
audit (unlike ``compare.py``, which reported ``n/a`` for out-of-place).

Measurement proxy caveat
------------------------
``vectors_scanned_per_query`` is kept as the primary budget metric because it is
already reported by the engine, but it is only a *proxy* for query cost:
segment fan-out, routing distance work and memory locality differ.  Query p50/p95
latency is reported alongside it, and this is an **in-memory prototype** — these
numbers say nothing about disk/NVMe behaviour.
"""

from __future__ import annotations

import argparse
import datetime
import json
import os
import platform
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np

from ..core.config import QueryConfig
from ..core.metrics import Counters
from ..index.segment import Segment
from .churn import (
    generate_churn_trace,
    audit_maintainer,
    audit_membership,
    audit_outofplace_membership,
)
from .datasets import recall_at_k
from .compare import (
    CompareConfig,
    STRATEGIES,
    _DatasetCache,
    _dataset_checksum,
    _index_config,
    _maintain_config,
    _depths_from_text,
    _parse_int_list,
    _parse_delta_list,
    _parse_merge_list,
    dynamic_ground_truth,
    _benchmark_queries,
    _stats_ms,
    _hierarchy_valid,
    _maint_snapshot,
    _snapshot_delta,
    _snapshot_advanced,
    _maint_cumulative_fields,
    _index_bytes_estimate,
    _environment_metadata,
    _rss_bytes,
    _git_sha,
    render_table,
    write_csv,
    write_parquet,
)
from .structure import (
    build_structure_row,
    event_fields,
    make_instrumented_maintainer,
    structure_metadata,
    structure_snapshot,
    timing_fields,
)

BEAM_DEFAULT = [2, 4, 8, 16, 32]
NPOST_DEFAULT = [1, 2, 4, 8]

# Base-group identity for pairing/matching (order fixed, used as an index key).
_BASE_GROUP_FIELDS = (
    "dataset", "workload", "trace_hash", "seed", "depth_config",
    "split_threshold", "merge_threshold", "delta", "checkpoint",
)


# ---------------------------------------------------------------------------
# Search presets
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class SearchPreset:
    """One point in the search grid: scalar beam width + postings per unit."""
    beam_scalar: int
    n_postings: int

    @property
    def beam_widths(self) -> tuple[int, ...]:
        """Valid ``QueryConfig`` translation of the scalar beam width."""
        return (self.beam_scalar,)

    @property
    def label(self) -> str:
        return f"b{self.beam_scalar}p{self.n_postings}"

    @property
    def nominal_budget(self) -> int:
        return self.beam_scalar * self.n_postings

    def to_dict(self) -> dict:
        return {
            "beam_scalar": self.beam_scalar,
            "beam_widths": str(list(self.beam_widths)),
            "n_postings": self.n_postings,
            "label": self.label,
        }


def build_presets(beam_values: list[int], n_postings_values: list[int]) -> list[SearchPreset]:
    return [SearchPreset(b, p) for b in beam_values for p in n_postings_values]


# ---------------------------------------------------------------------------
# Spec / base configs
# ---------------------------------------------------------------------------

@dataclass
class BudgetSpec:
    phase: str
    dataset: str
    dim: int
    initial_size: int
    operations: int
    workload: str
    insert_prob: float
    leaf_size: int
    branching_factor: int
    depths: list[int | None]
    split_thresholds: list[int]
    merge_thresholds: list[int | None]
    deltas: list[float | None]
    seeds: list[int]
    beam_values: list[int]
    n_postings_values: list[int]
    k: int
    recall_ks: tuple[int, ...]
    query_count: int
    checkpoint_interval: int
    memtable_limit: int
    compaction_threshold: int
    match_within: float
    match_approx: float
    out_dir: Path
    overwrite: bool
    calibration_results: Path | None
    blas_threads: int | None

    def reserve_size(self) -> int:
        return max(1, self.operations)

    def presets(self) -> list[SearchPreset]:
        return build_presets(self.beam_values, self.n_postings_values)

    def base_configs(self) -> list[CompareConfig]:
        cells: list[CompareConfig] = []
        for depth in self.depths:
            for st in self.split_thresholds:
                for mt in self.merge_thresholds:
                    for delta in self.deltas:
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
                                # inert placeholders — presets drive search
                                n_postings=1, beam_widths=(1,),
                            ))
        return cells

    def to_dict(self) -> dict:
        return {
            "phase": self.phase, "dataset": self.dataset, "dim": self.dim,
            "initial_size": self.initial_size, "operations": self.operations,
            "workload": self.workload, "insert_prob": self.insert_prob,
            "leaf_size": self.leaf_size,
            "branching_factor": self.branching_factor,
            "depths": [d for d in self.depths],
            "split_thresholds": list(self.split_thresholds),
            "merge_thresholds": list(self.merge_thresholds),
            "deltas": list(self.deltas), "seeds": list(self.seeds),
            "beam_width_values": list(self.beam_values),
            "n_postings_values": list(self.n_postings_values),
            "k": self.k, "recall_ks": list(self.recall_ks),
            "query_count": self.query_count,
            "checkpoint_interval": self.checkpoint_interval,
            "memtable_limit": self.memtable_limit,
            "compaction_threshold": self.compaction_threshold,
            "match_within_pct": self.match_within,
            "match_approx_pct": self.match_approx,
            "blas_threads": self.blas_threads,
            "calibration_results": (str(self.calibration_results)
                                    if self.calibration_results else None),
        }


# ---------------------------------------------------------------------------
# Checkpoint snapshot + per-preset evaluation
# ---------------------------------------------------------------------------

@dataclass
class CheckpointSnapshot:
    checkpoint: int
    ops: int
    live: list[int]
    live_count: int
    units: list[Segment]                      # clones (isolated from maint)
    units_searched: int
    index_bytes_estimate: int
    hierarchy_valid: bool
    canonical_owned_live_count: int
    total_live_membership_count: int
    replica_count: int
    membership_bloat_pct: float
    audit_passed: bool
    npa_violations: int
    ownership_errors: int
    coverage_errors: int
    membership_status: str
    membership_violations: int
    checkpoint_correct: bool
    checkpoint_failure_reason: str
    gt: np.ndarray
    maint_cumulative: dict
    maint_delta: dict
    insert_ns: list[int] = field(default_factory=list)
    delete_ns: list[int] = field(default_factory=list)


def _capture_snapshot(*, checkpoint: int, ops: int, live: set[int], maint,
                      counters: Counters, prev_maint: dict, queries: np.ndarray,
                      vectors: np.ndarray, cfg: CompareConfig, strategy: str,
                      insert_ns: list[int], delete_ns: list[int]) -> CheckpointSnapshot:
    live_arr = np.array(sorted(live), dtype=np.int64)
    n_live = len(live_arr)

    real_units = maint.searchable_units()
    units = [u.clone() for u in real_units]      # checkpointed, isolated state

    audit = audit_maintainer(maint, expected_live_vids=live)
    if strategy == "inplace":
        mem = audit_membership(maint)
    else:
        mem = audit_outofplace_membership(maint)
    membership_status = "PASS" if mem.passed else "FAIL"

    m_live = 0
    canonical_live = 0
    for u in units:
        for p in u.postings:
            for v in p.members.tolist():
                if int(v) in live:
                    m_live += 1
            for v in p.owned.tolist():
                if int(v) in live:
                    canonical_live += 1

    hierarchy_ok = all(_hierarchy_valid(u) for u in units)
    bloat_pct = (100.0 * (m_live - n_live) / n_live) if n_live else 0.0

    correct = (audit.passed and mem.passed and hierarchy_ok
               and canonical_live == n_live)
    reason = ""
    if not correct:
        bits = []
        if not audit.passed:
            bits.append(f"NPA/ownership/coverage: npa={len(audit.npa_violations)} "
                        f"own={len(audit.ownership_errors)} "
                        f"cov={len(audit.coverage_errors)}")
        if not mem.passed:
            bits.append(f"membership violations={mem.violation_count}")
        if not hierarchy_ok:
            bits.append("invalid hierarchy references")
        if canonical_live != n_live:
            bits.append(f"canonical owners {canonical_live} != live {n_live}")
        reason = "; ".join(bits)

    gt = dynamic_ground_truth(queries, vectors, live_arr, k=max(cfg.recall_ks))

    cur_maint = _maint_snapshot(counters)
    return CheckpointSnapshot(
        checkpoint=checkpoint, ops=ops, live=[int(v) for v in live_arr],
        live_count=n_live, units=units, units_searched=len(units),
        index_bytes_estimate=_index_bytes_estimate(units),
        hierarchy_valid=hierarchy_ok,
        canonical_owned_live_count=canonical_live,
        total_live_membership_count=m_live,
        replica_count=m_live - canonical_live, membership_bloat_pct=bloat_pct,
        audit_passed=audit.passed, npa_violations=len(audit.npa_violations),
        ownership_errors=len(audit.ownership_errors),
        coverage_errors=len(audit.coverage_errors),
        membership_status=membership_status,
        membership_violations=mem.violation_count,
        checkpoint_correct=correct, checkpoint_failure_reason=reason,
        gt=gt, maint_cumulative=_maint_cumulative_fields(cur_maint),
        maint_delta=_snapshot_delta(prev_maint, cur_maint),
        insert_ns=list(insert_ns), delete_ns=list(delete_ns),
    )


def _evaluate_preset(snapshot: CheckpointSnapshot, preset: SearchPreset,
                     queries: np.ndarray, cfg: CompareConfig, *,
                     strategy: str, experiment_id: str, run_index: int,
                     order_in_pair: int, cfg_identity: dict) -> dict:
    """Search one preset against the frozen checkpoint units (fresh counters)."""
    cfg_q = QueryConfig(k=max(cfg.recall_ks), beam_widths=preset.beam_widths,
                        n_postings=preset.n_postings)
    q_total, lat_ns, results = _benchmark_queries(snapshot.units, queries, cfg_q)

    recall_fields: dict[str, Any] = {}
    for kk in cfg.recall_ks:
        if snapshot.live_count == 0:
            recall_fields[f"recall_at_{kk}"] = float("nan")
        else:
            recall_fields[f"recall_at_{kk}"] = float(
                recall_at_k(results, snapshot.gt, kk))
    primary = recall_fields.get(f"recall_at_{cfg.k}", float("nan"))

    returned: set[int] = set()
    for res in results:
        returned.update(res)
    live_set = set(snapshot.live)
    leaked = sorted(returned - live_set)

    preset_correct = snapshot.checkpoint_correct and not leaked
    if not snapshot.checkpoint_correct:
        reason = snapshot.checkpoint_failure_reason
    elif leaked:
        reason = f"{len(leaked)} non-live ids returned (e.g. {leaked[:5]})"
    else:
        reason = ""

    n_q = max(1, len(queries))
    scan = q_total.distance_computations.get("scan", 0)
    routing = q_total.distance_computations.get("routing", 0)
    qstats = _stats_ms(lat_ns)

    row = dict(cfg_identity)
    row.update({
        "experiment_id": experiment_id,
        "run_index": run_index,
        "order_in_pair": order_in_pair,
        "strategy": strategy,
        "checkpoint": snapshot.checkpoint,
        "operations_processed": snapshot.ops,
        "live_vector_count": snapshot.live_count,

        # search preset (actual settings used by this strategy)
        "search_preset": preset.label,
        "beam_scalar": preset.beam_scalar,
        "beam_widths": str(list(preset.beam_widths)),
        "n_postings": preset.n_postings,
        "nominal_search_budget": preset.nominal_budget,
        "units_searched": snapshot.units_searched,
        "postings_budget": preset.n_postings * snapshot.units_searched,

        # search quality
        "recall": primary,
        **recall_fields,
        "gt_recomputed": True,

        # measured query work (per query) — primary budget metric
        "vectors_scanned_per_query": q_total.vectors_scanned / n_q,
        "vectors_scanned_total": q_total.vectors_scanned,
        "work_scan_sims_per_query": scan / n_q,
        "work_routing_sims_per_query": routing / n_q,
        "work_sim_evals_per_query": (scan + routing) / n_q,
        "work_postings_read_per_query": q_total.postings_read / n_q,
        "work_bytes_read_per_query": q_total.bytes_read / n_q,

        # query latency (wall clock)
        "query_count": len(queries),
        "query_latency_p50_ms": qstats["p50_ms"],
        "query_latency_p95_ms": qstats["p95_ms"],
        "query_latency_mean_ms": qstats["mean_ms"],

        # storage / structure at the checkpoint
        "index_bytes_estimate": snapshot.index_bytes_estimate,
        "total_live_membership_count": snapshot.total_live_membership_count,
        "canonical_owned_live_count": snapshot.canonical_owned_live_count,
        "replica_count": snapshot.replica_count,
        "membership_bloat_pct": snapshot.membership_bloat_pct,

        # correctness (checkpoint audits + preset-specific leak check)
        "audit_passed": snapshot.audit_passed,
        "npa_violations": snapshot.npa_violations,
        "ownership_errors": snapshot.ownership_errors,
        "coverage_errors": snapshot.coverage_errors,
        "membership_status": snapshot.membership_status,
        "membership_violations": snapshot.membership_violations,
        "hierarchy_valid": snapshot.hierarchy_valid,
        "deleted_id_retrievals": len(leaked),
        "correctness_status": "PASSED" if preset_correct else "FAILED",
        "run_failed": (not preset_correct),
        "failure_reason": reason,

        # maintenance trajectory (proves search presets did not alter updates)
        **snapshot.maint_cumulative,
        **{f"delta_{k}": v for k, v in snapshot.maint_delta.items()},

        "insert_latency_p50_ms": _stats_ms(snapshot.insert_ns)["p50_ms"],
        "delete_latency_p50_ms": _stats_ms(snapshot.delete_ns)["p50_ms"],
    })
    return row


# ---------------------------------------------------------------------------
# Run one (base config, strategy): replay churn once, evaluate grid at checkpoints
# ---------------------------------------------------------------------------

@dataclass
class RunInfo:
    run_index: int
    strategy: str
    rows: int = 0
    failed: bool = False
    error: str = ""
    immutability_violations: int = 0
    structure_rows: list[dict] = field(default_factory=list)


def run_base_config(cfg: CompareConfig, strategy: str, presets: list[SearchPreset], *,
                    vectors: np.ndarray, queries: np.ndarray, trace, initial_segment,
                    checkpoint_interval: int, run_index: int, order_in_pair: int,
                    experiment_id: str, dataset_checksum: str) -> tuple[list[dict], RunInfo]:
    info = RunInfo(run_index=run_index, strategy=strategy)
    rows: list[dict] = []

    cfg_idx = _index_config(cfg)
    cfg_m = _maintain_config(cfg, strategy)
    counters = Counters()
    try:
        maint = make_instrumented_maintainer(
            strategy, vectors, cfg_idx, cfg_m, counters, initial_segment)
    except Exception as exc:                                    # pragma: no cover
        info.failed = True
        info.error = f"construction failed: {exc!r}"
        return rows, info

    identity = {
        "dataset": cfg.dataset, "dataset_checksum": dataset_checksum,
        "dim": cfg.dim,
        "initial_size": cfg.initial_size, "reserve_size": cfg.reserve_size,
        "workload": cfg.workload, "insert_prob": cfg.insert_prob,
        "seed": cfg.seed, "trace_seed": trace.seed, "trace_hash": trace.trace_hash,
        "depth_config": cfg.depth if cfg.depth is not None else -1,
        "leaf_size": cfg.leaf_size, "branching_factor": cfg.branching_factor,
        "split_threshold": cfg.split_threshold,
        "merge_threshold": cfg.merge_threshold if cfg.merge_threshold is not None else -1,
        "delta": cfg.delta, "k": cfg.k,
        "memtable_limit": cfg.memtable_limit,
        "compaction_threshold": cfg.compaction_threshold,
    }

    live: set[int] = set(range(cfg.initial_size))
    prev_maint = _maint_snapshot(counters)
    insert_ns: list[int] = []
    delete_ns: list[int] = []
    cp_idx = 0

    # Structural instrumentation state (observational only).
    prev_maint_ns = 0
    prev_pruned = 0
    prev_update_ns = 0
    total_update_ns = 0
    n_total = int(vectors.shape[0])

    def checkpoint(cp: int, ops: int) -> None:
        nonlocal prev_maint, prev_maint_ns, prev_pruned, prev_update_ns
        before = _maint_snapshot(counters)
        snap = _capture_snapshot(
            checkpoint=cp, ops=ops, live=live, maint=maint, counters=counters,
            prev_maint=prev_maint, queries=queries, vectors=vectors, cfg=cfg,
            strategy=strategy, insert_ns=insert_ns, delete_ns=delete_ns,
        )
        structure = structure_snapshot(
            strategy=strategy, maint=maint, live=live, n_total=n_total,
            split_threshold=cfg.split_threshold,
        )
        maint_ns = maint.maintenance_ns
        pruned = maint.pruned_postings
        events = event_fields(
            strategy=strategy, cumulative=snap.maint_cumulative,
            interval=snap.maint_delta,
            pruned_cumulative=pruned,
            pruned_interval=(None if pruned is None else pruned - prev_pruned),
        )
        timing = timing_fields(
            maintenance_ns=maint_ns, prev_maintenance_ns=prev_maint_ns,
            update_ns=total_update_ns, prev_update_ns=prev_update_ns,
        )
        info.structure_rows.append(build_structure_row(
            identity=identity, experiment_id=experiment_id, run_index=run_index,
            strategy=strategy, checkpoint=cp, operations_processed=ops,
            live_vector_count=snap.live_count, structure=structure, events=events,
            timing=timing,
            correctness_status=("PASSED" if snap.checkpoint_correct else "FAILED"),
            checkpoint_correct=snap.checkpoint_correct,
        ))
        prev_maint_ns = maint_ns
        if pruned is not None:
            prev_pruned = pruned
        prev_update_ns = total_update_ns

        for preset in presets:
            rows.append(_evaluate_preset(
                snap, preset, queries, cfg, strategy=strategy,
                experiment_id=experiment_id, run_index=run_index,
                order_in_pair=order_in_pair, cfg_identity=identity,
            ))
        after = _maint_snapshot(counters)
        if _snapshot_advanced(before, after):
            info.immutability_violations += 1
        if any(r["run_failed"] for r in rows[-len(presets):]):
            info.failed = True
        prev_maint = after

    checkpoint(0, 0)

    total_ops = len(trace.operations)
    done = 0
    for (op_type, vid) in trace.operations:
        t0 = time.perf_counter_ns()
        if op_type == "insert":
            maint.insert(int(vid), vectors[int(vid)])
            live.add(int(vid))
        else:
            maint.delete(int(vid))
            live.discard(int(vid))
        t1 = time.perf_counter_ns()
        maint.maintain()
        total_update_ns += time.perf_counter_ns() - t0
        (insert_ns if op_type == "insert" else delete_ns).append(t1 - t0)
        done += 1
        if done % checkpoint_interval == 0 or done == total_ops:
            cp_idx += 1
            checkpoint(cp_idx, done)

    info.rows = len(rows)
    return rows, info


# ---------------------------------------------------------------------------
# Matching: relative scan gap
# ---------------------------------------------------------------------------

def relative_scan_gap(ip_scan: float, oop_scan: float) -> float:
    """100 * |ip - oop| / ((ip + oop) / 2).  NaN if both are zero-ish."""
    denom = (float(ip_scan) + float(oop_scan)) / 2.0
    if denom <= 0.0:
        return 0.0 if float(ip_scan) == float(oop_scan) else float("nan")
    return 100.0 * abs(float(ip_scan) - float(oop_scan)) / denom


def match_status(gap: float, within: float, approx: float) -> str:
    if gap is None or (isinstance(gap, float) and np.isnan(gap)):
        return "unmatched"
    if gap <= within:
        return "matched"
    if gap <= approx:
        return "approximate"
    return "unmatched"


def _base_group_key(row: dict) -> tuple:
    return tuple(str(row.get(f)) for f in _BASE_GROUP_FIELDS)


def _group_rows(rows: list[dict]) -> dict[tuple, dict[str, dict[str, dict]]]:
    """group key -> strategy -> preset label -> row."""
    groups: dict[tuple, dict[str, dict[str, dict]]] = {}
    for r in rows:
        key = _base_group_key(r)
        groups.setdefault(key, {}).setdefault(r["strategy"], {})[r["search_preset"]] = r
    return groups


# ---------------------------------------------------------------------------
# Calibration selection
# ---------------------------------------------------------------------------

@dataclass
class SelectionPick:
    bucket: str
    kind: str                       # "depth" | "global_fallback"
    ip: SearchPreset
    oop: SearchPreset
    median_gap: float
    p25_gap: float
    p75_gap: float
    frac_within_5: float
    frac_within_10: float
    mean_q_p50_ip: float
    mean_q_p50_oop: float
    n_groups: int

    def to_row(self) -> dict:
        return {
            "bucket": self.bucket, "selection_kind": self.kind,
            "ip_preset": self.ip.label, "ip_beam": self.ip.beam_scalar,
            "ip_n_postings": self.ip.n_postings,
            "oop_preset": self.oop.label, "oop_beam": self.oop.beam_scalar,
            "oop_n_postings": self.oop.n_postings,
            "median_scan_gap_pct": self.median_gap,
            "p25_scan_gap_pct": self.p25_gap, "p75_scan_gap_pct": self.p75_gap,
            "frac_within_5pct": self.frac_within_5,
            "frac_within_10pct": self.frac_within_10,
            "mean_q_p50_ms_ip": self.mean_q_p50_ip,
            "mean_q_p50_ms_oop": self.mean_q_p50_oop,
            "n_groups": self.n_groups,
        }


def _select_for_groups(groups: list[dict], presets: list[SearchPreset], *,
                       within: float, approx: float, bucket: str,
                       kind: str) -> SelectionPick | None:
    """Minimize median relative scan gap; tie-break latency then nominal budget."""
    scored: list[tuple] = []
    for ip in presets:
        for oop in presets:
            gaps: list[float] = []
            lat_ip: list[float] = []
            lat_oop: list[float] = []
            for g in groups:
                iprow = g.get("inplace", {}).get(ip.label)
                ooprow = g.get("outofplace", {}).get(oop.label)
                if iprow is None or ooprow is None:
                    continue
                if iprow["run_failed"] or ooprow["run_failed"]:
                    continue
                gaps.append(relative_scan_gap(
                    iprow["vectors_scanned_per_query"],
                    ooprow["vectors_scanned_per_query"]))
                lat_ip.append(iprow["query_latency_p50_ms"])
                lat_oop.append(ooprow["query_latency_p50_ms"])
            if not gaps:
                continue
            ga = np.asarray(gaps, dtype=np.float64)
            med = float(np.median(ga))
            mean_lat = float(np.nanmean(np.asarray(lat_ip + lat_oop, dtype=np.float64)))
            scored.append((
                med, mean_lat, ip.nominal_budget + oop.nominal_budget,
                ip.label, oop.label, gaps, lat_ip, lat_oop,
            ))
    if not scored:
        return None
    scored.sort(key=lambda t: (t[0], t[1], t[2], t[3], t[4]))
    med, mean_lat, _bud, ipl, oopl, gaps, lat_ip, lat_oop = scored[0]
    ga = np.asarray(gaps, dtype=np.float64)
    return SelectionPick(
        bucket=bucket, kind=kind,
        ip=_preset_lookup(presets, ipl), oop=_preset_lookup(presets, oopl),
        median_gap=med,
        p25_gap=float(np.percentile(ga, 25)), p75_gap=float(np.percentile(ga, 75)),
        frac_within_5=float((ga <= within).mean()),
        frac_within_10=float((ga <= approx).mean()),
        mean_q_p50_ip=float(np.nanmean(np.asarray(lat_ip, dtype=np.float64))),
        mean_q_p50_oop=float(np.nanmean(np.asarray(lat_oop, dtype=np.float64))),
        n_groups=len(gaps),
    )


def _preset_lookup(presets: list[SearchPreset], label: str) -> SearchPreset:
    for p in presets:
        if p.label == label:
            return p
    raise KeyError(label)


def select_pairs(raw_rows: list[dict], presets: list[SearchPreset], *,
                 within: float, approx: float) -> list[SelectionPick]:
    """Calibration selection: one pick per depth, plus a global fallback.

    Selection granularity is deliberately **depth** (not the full
    split/delta/seed group) because beam/n_postings effects on scan work are
    dominated by depth and unit count, while the absolute split/delta settings
    differ between the calibration and evaluation phases.  Held-out groups with
    no calibration depth still resolve via the global fallback.
    """
    all_groups = list(_group_rows(raw_rows).values())
    picks: list[SelectionPick] = []

    depths = sorted({int(r["depth_config"]) for r in raw_rows})
    for d in depths:
        depth_groups = list(_group_rows(
            [r for r in raw_rows if int(r["depth_config"]) == d]).values())
        pick = _select_for_groups(depth_groups, presets, within=within,
                                  approx=approx, bucket=str(d), kind="depth")
        if pick is not None:
            picks.append(pick)

    glob = _select_for_groups(all_groups, presets, within=within, approx=approx,
                              bucket="__global__", kind="global_fallback")
    if glob is not None:
        picks.append(glob)
    return picks


def write_selection(path: Path, picks: list[SelectionPick]) -> None:
    write_csv(path, [p.to_row() for p in picks])


def read_selection(path: Path) -> dict[str, SelectionPick]:
    import csv
    picks: dict[str, SelectionPick] = {}
    with open(path) as f:
        for row in csv.DictReader(f):
            ip = SearchPreset(int(row["ip_beam"]), int(row["ip_n_postings"]))
            oop = SearchPreset(int(row["oop_beam"]), int(row["oop_n_postings"]))
            picks[row["bucket"]] = SelectionPick(
                bucket=row["bucket"], kind=row["selection_kind"], ip=ip, oop=oop,
                median_gap=float(row["median_scan_gap_pct"]),
                p25_gap=float(row["p25_scan_gap_pct"]),
                p75_gap=float(row["p75_scan_gap_pct"]),
                frac_within_5=float(row["frac_within_5pct"]),
                frac_within_10=float(row["frac_within_10pct"]),
                mean_q_p50_ip=float(row["mean_q_p50_ms_ip"]),
                mean_q_p50_oop=float(row["mean_q_p50_ms_oop"]),
                n_groups=int(row["n_groups"]),
            )
    return picks


# ---------------------------------------------------------------------------
# Matched rows
# ---------------------------------------------------------------------------

def build_matched(raw_rows: list[dict], selection: dict[str, SelectionPick], *,
                  within: float, approx: float, source: str,
                  phase: str) -> list[dict]:
    primary_k = _primary_k(raw_rows)
    out: list[dict] = []
    for key, strat in sorted(_group_rows(raw_rows).items()):
        head = next(iter(next(iter(strat.values())).values()))
        depth_bucket = str(head["depth_config"])
        pick = selection.get(depth_bucket) or selection.get("__global__")
        bucket_used = depth_bucket if depth_bucket in selection else "__global__"
        if pick is None:
            continue
        iprow = strat.get("inplace", {}).get(pick.ip.label)
        ooprow = strat.get("outofplace", {}).get(pick.oop.label)

        base = {
            "phase": phase, "selection_source": source,
            "selection_bucket": bucket_used, "selection_kind": pick.kind,
            "k": primary_k,
            "dataset": head["dataset"], "workload": head["workload"],
            "trace_hash": head["trace_hash"], "seed": head["seed"],
            "depth_config": head["depth_config"],
            "split_threshold": head["split_threshold"],
            "merge_threshold": head["merge_threshold"], "delta": head["delta"],
            "checkpoint": head["checkpoint"],
            "operations_processed": head["operations_processed"],
            "live_vector_count": head["live_vector_count"],
            "ip_preset": pick.ip.label, "ip_beam": pick.ip.beam_scalar,
            "ip_n_postings": pick.ip.n_postings,
            "oop_preset": pick.oop.label, "oop_beam": pick.oop.beam_scalar,
            "oop_n_postings": pick.oop.n_postings,
        }
        if iprow is None or ooprow is None:
            base.update({
                "ip_scan_per_query": float("nan"),
                "oop_scan_per_query": float("nan"),
                "relative_scan_gap_pct": float("nan"),
                "match_status": "missing",
                f"ip_recall_at_{primary_k}": float("nan"),
                f"oop_recall_at_{primary_k}": float("nan"),
                "recall_diff_pp": float("nan"),
                "ip_q_p50_ms": float("nan"), "oop_q_p50_ms": float("nan"),
                "ip_q_p95_ms": float("nan"), "oop_q_p95_ms": float("nan"),
                "ip_correctness_status": "n/a", "oop_correctness_status": "n/a",
                "ip_failed": True, "oop_failed": True,
            })
            out.append(base)
            continue

        if iprow["run_failed"] or ooprow["run_failed"]:
            status = "failed"
            gap = float("nan")
        else:
            gap = relative_scan_gap(iprow["vectors_scanned_per_query"],
                                    ooprow["vectors_scanned_per_query"])
            status = match_status(gap, within, approx)

        ipr = iprow.get(f"recall_at_{primary_k}", iprow["recall"])
        oopr = ooprow.get(f"recall_at_{primary_k}", ooprow["recall"])
        base.update({
            "ip_scan_per_query": iprow["vectors_scanned_per_query"],
            "oop_scan_per_query": ooprow["vectors_scanned_per_query"],
            "relative_scan_gap_pct": gap,
            "match_status": status,
            f"ip_recall_at_{primary_k}": ipr,
            f"oop_recall_at_{primary_k}": oopr,
            "recall_diff_pp": 100.0 * (ipr - oopr),
            "ip_q_p50_ms": iprow["query_latency_p50_ms"],
            "oop_q_p50_ms": ooprow["query_latency_p50_ms"],
            "ip_q_p95_ms": iprow["query_latency_p95_ms"],
            "oop_q_p95_ms": ooprow["query_latency_p95_ms"],
            "ip_correctness_status": iprow["correctness_status"],
            "oop_correctness_status": ooprow["correctness_status"],
            "ip_failed": iprow["run_failed"], "oop_failed": ooprow["run_failed"],
        })
        out.append(base)
    return out


def build_oracle(raw_rows: list[dict], presets: list[SearchPreset], *,
                 within: float, approx: float, phase: str) -> list[dict]:
    """Per-checkpoint nearest-work pair — DESCRIPTIVE/oracle, not primary."""
    primary_k = _primary_k(raw_rows)
    out: list[dict] = []
    for key, strat in sorted(_group_rows(raw_rows).items()):
        head = next(iter(next(iter(strat.values())).values()))
        best = None
        for ip in presets:
            for oop in presets:
                iprow = strat.get("inplace", {}).get(ip.label)
                ooprow = strat.get("outofplace", {}).get(oop.label)
                if not iprow or not ooprow:
                    continue
                if iprow["run_failed"] or ooprow["run_failed"]:
                    continue
                gap = relative_scan_gap(iprow["vectors_scanned_per_query"],
                                        ooprow["vectors_scanned_per_query"])
                cand = (gap, ip.nominal_budget + oop.nominal_budget, ip.label,
                        oop.label, iprow, ooprow)
                if best is None or cand[:4] < best[:4]:
                    best = cand
        if best is None:
            continue
        gap, _b, ipl, oopl, iprow, ooprow = best
        out.append({
            "diagnostic": "oracle_per_checkpoint",
            "phase": phase, "k": primary_k,
            "dataset": head["dataset"], "workload": head["workload"],
            "trace_hash": head["trace_hash"], "seed": head["seed"],
            "depth_config": head["depth_config"],
            "split_threshold": head["split_threshold"],
            "merge_threshold": head["merge_threshold"], "delta": head["delta"],
            "checkpoint": head["checkpoint"],
            "ip_preset": ipl, "oop_preset": oopl,
            "ip_scan_per_query": iprow["vectors_scanned_per_query"],
            "oop_scan_per_query": ooprow["vectors_scanned_per_query"],
            "relative_scan_gap_pct": gap,
            "match_status": match_status(gap, within, approx),
            f"ip_recall_at_{primary_k}": iprow.get(f"recall_at_{primary_k}"),
            f"oop_recall_at_{primary_k}": ooprow.get(f"recall_at_{primary_k}"),
        })
    return out


def _primary_k(rows: list[dict]) -> int:
    for r in rows:
        if "k" in r:
            return int(r["k"])
    return 10


# ---------------------------------------------------------------------------
# Summary + Pareto
# ---------------------------------------------------------------------------

_WORK_BANDS = [(0.0, 128.0), (128.0, 256.0), (256.0, 512.0),
               (512.0, 1024.0), (1024.0, float("inf"))]


def _work_band(scan: float) -> str:
    if scan is None or (isinstance(scan, float) and np.isnan(scan)):
        return "n/a"
    for lo, hi in _WORK_BANDS:
        if lo <= scan < hi:
            return f"[{int(lo)},{int(hi) if np.isfinite(hi) else 'inf'})"
    return "n/a"


def _ms(rows: list[dict], key: str) -> tuple[float, float]:
    vals = [float(r[key]) for r in rows
            if r.get(key) is not None
            and not (isinstance(r[key], float) and np.isnan(r[key]))]
    if not vals:
        return float("nan"), float("nan")
    a = np.asarray(vals, dtype=np.float64)
    return float(a.mean()), float(a.std(ddof=0))


def build_summary(matched: list[dict]) -> list[dict]:
    primary_k = _primary_k(matched)
    groups: dict[tuple, list[dict]] = {}
    for r in matched:
        if r["match_status"] in ("missing",):
            continue
        pair_scan = np.nanmean([r["ip_scan_per_query"], r["oop_scan_per_query"]])
        key = (r["depth_config"], r["split_threshold"], r["delta"],
               _work_band(float(pair_scan)))
        groups.setdefault(key, []).append(r)

    out: list[dict] = []
    for key, rows in sorted(groups.items(), key=lambda kv: str(kv[0])):
        depth, split, delta, band = key
        n_matched = sum(1 for r in rows if r["match_status"] == "matched")
        n_approx = sum(1 for r in rows if r["match_status"] == "approximate")
        n_unmatched = sum(1 for r in rows if r["match_status"] == "unmatched")
        n_failed = sum(1 for r in rows if r["match_status"] == "failed")
        ip_r_m, ip_r_s = _ms(rows, f"ip_recall_at_{primary_k}")
        oop_r_m, oop_r_s = _ms(rows, f"oop_recall_at_{primary_k}")
        dr_m, dr_s = _ms(rows, "recall_diff_pp")
        ips_m, ips_s = _ms(rows, "ip_scan_per_query")
        oops_m, oops_s = _ms(rows, "oop_scan_per_query")
        gap_m, gap_s = _ms(rows, "relative_scan_gap_pct")
        ipq_m, _ = _ms(rows, "ip_q_p50_ms")
        oopq_m, _ = _ms(rows, "oop_q_p50_ms")
        out.append({
            "row_kind": "summary",
            "depth_config": depth, "split_threshold": split, "delta": delta,
            "target_work_band_scans_per_query": band,
            "n_obs": len(rows), "n_matched": n_matched, "n_approx": n_approx,
            "n_unmatched": n_unmatched, "n_failed": n_failed,
            f"ip_recall_at_{primary_k}_mean": ip_r_m,
            f"ip_recall_at_{primary_k}_std": ip_r_s,
            f"oop_recall_at_{primary_k}_mean": oop_r_m,
            f"oop_recall_at_{primary_k}_std": oop_r_s,
            "recall_diff_pp_mean": dr_m, "recall_diff_pp_std": dr_s,
            "ip_scan_mean": ips_m, "ip_scan_std": ips_s,
            "oop_scan_mean": oops_m, "oop_scan_std": oops_s,
            "relative_scan_gap_pct_mean": gap_m,
            "relative_scan_gap_pct_std": gap_s,
            "ip_q_p50_ms_mean": ipq_m, "oop_q_p50_ms_mean": oopq_m,
        })
    return out


def build_pareto(raw_rows: list[dict]) -> list[dict]:
    """Recall-vs-work Pareto frontier per (strategy, depth) and pooled."""
    primary_k = _primary_k(raw_rows)
    rk = f"recall_at_{primary_k}"

    def _points(rows: list[dict]) -> list[dict]:
        by_preset: dict[str, list[dict]] = {}
        for r in rows:
            if r["run_failed"]:
                continue
            by_preset.setdefault(r["search_preset"], []).append(r)
        pts: list[dict] = []
        for label, rs in by_preset.items():
            scan_m, scan_s = _ms(rs, "vectors_scanned_per_query")
            rec_m, rec_s = _ms(rs, rk)
            q_m, _ = _ms(rs, "query_latency_p50_ms")
            first = rs[0]
            pts.append({
                "search_preset": label, "beam_scalar": first["beam_scalar"],
                "n_postings": first["n_postings"],
                "mean_vectors_scanned_per_query": scan_m,
                "mean_recall": rec_m, "recall_std": rec_s,
                "mean_q_p50_ms": q_m, "n_obs": len(rs),
            })
        # mark dominated (lower scan + higher recall preferred)
        for p in pts:
            dom = False
            for q in pts:
                if q is p:
                    continue
                if (q["mean_vectors_scanned_per_query"] <= p["mean_vectors_scanned_per_query"]
                        and q["mean_recall"] >= p["mean_recall"]
                        and (q["mean_vectors_scanned_per_query"]
                             < p["mean_vectors_scanned_per_query"]
                             or q["mean_recall"] > p["mean_recall"])):
                    dom = True
                    break
            p["dominated"] = dom
        pts.sort(key=lambda p: (p["mean_vectors_scanned_per_query"], -p["mean_recall"]))
        return pts

    out: list[dict] = []
    for strategy in STRATEGIES:
        srows = [r for r in raw_rows if r["strategy"] == strategy]
        for p in _points(srows):
            out.append({"strategy": strategy, "depth_config": "all", **p})
        for depth in sorted({int(r["depth_config"]) for r in srows}):
            drows = [r for r in srows if int(r["depth_config"]) == depth]
            for p in _points(drows):
                out.append({"strategy": strategy, "depth_config": depth, **p})
    return out


# ---------------------------------------------------------------------------
# Rendering helpers
# ---------------------------------------------------------------------------

def _render_selection(picks: list[SelectionPick]) -> str:
    cols = [
        ("bucket", "bucket", 12, "s"), ("kind", "selection_kind", 16, "s"),
        ("ip_preset", "ip_preset", 10, "s"), ("oop_preset", "oop_preset", 10, "s"),
        ("medGap%", "median_scan_gap_pct", 8, "f2"),
        ("p25", "p25_scan_gap_pct", 7, "f2"), ("p75", "p75_scan_gap_pct", 7, "f2"),
        ("f<=5%", "frac_within_5pct", 7, "f2"),
        ("f<=10%", "frac_within_10pct", 8, "f2"),
        ("ip_q50", "mean_q_p50_ms_ip", 8, "f3"),
        ("oop_q50", "mean_q_p50_ms_oop", 8, "f3"),
        ("ngrp", "n_groups", 5, "s"),
    ]
    return render_table("Calibration selection (minimize median relative scan gap)",
                        cols, [p.to_row() for p in picks])


def _render_match_counts(matched: list[dict]) -> str:
    counts: dict[str, int] = {}
    for r in matched:
        counts[r["match_status"]] = counts.get(r["match_status"], 0) + 1
    parts = ", ".join(f"{k}={v}" for k, v in sorted(counts.items()))
    return f"Matched pair status counts: {parts}"


# ---------------------------------------------------------------------------
# Metadata
# ---------------------------------------------------------------------------

def _experiment_id(prefix: str = "sb") -> str:
    stamp = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
    return f"{prefix}-{stamp}-{os.getpid()}"


def _unique_out_dir(base: Path, experiment_id: str, overwrite: bool) -> Path:
    return base if overwrite else base / experiment_id


def _resolve_calibration_dir(path: Path) -> Path:
    """Locate a calibration output dir containing results_budget_selection.csv.

    Accepts either the run directory itself or a parent that holds one or more
    timestamped run sub-directories (the newest is used).
    """
    if (path / "results_budget_selection.csv").exists():
        return path
    candidates = sorted(
        (p.parent for p in path.glob("*/results_budget_selection.csv")),
        key=lambda p: p.name,
    )
    if not candidates:
        raise FileNotFoundError(
            f"no results_budget_selection.csv found directly under {path} "
            f"or in any immediate sub-directory")
    return candidates[-1]


def _output_schema(files: dict[str, list[dict]]) -> dict:
    schema = {}
    for name, rows in files.items():
        if rows:
            schema[name] = list(rows[0].keys())
        else:
            schema[name] = []
    return schema


# ---------------------------------------------------------------------------
# Execution
# ---------------------------------------------------------------------------

def run_experiment(spec: BudgetSpec, *, progress: bool = True) -> dict[str, list[dict]]:
    cache = _DatasetCache()
    presets = spec.presets()
    experiment_id = _experiment_id()
    cells = spec.base_configs()
    raw: list[dict] = []
    infos: list[RunInfo] = []
    structure_rows: list[dict] = []

    if progress:
        print(f"Experiment id: {experiment_id}")
        print(f"Phase: {spec.phase}  presets={len(presets)} "
              f"(beam {spec.beam_values} x n_postings {spec.n_postings_values})")
        print(f"Base configs: {len(cells)} x 2 strategies = {len(cells) * 2} runs; "
              f"checkpoint interval={spec.checkpoint_interval}")

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
        if run_index % 2 == 1:
            order.reverse()
        for pos, strategy in enumerate(order):
            rows, info = run_base_config(
                cfg, strategy, presets, vectors=vectors, queries=queries,
                trace=trace, initial_segment=initial_segment,
                checkpoint_interval=spec.checkpoint_interval, run_index=run_index,
                order_in_pair=pos, experiment_id=experiment_id,
                dataset_checksum=dcs,
            )
            raw.extend(rows)
            structure_rows.extend(info.structure_rows)
            infos.append(info)
            if progress:
                print(f"[sb] run={run_index:<3} {strategy:<10} "
                      f"depth={cfg.depth} split={cfg.split_threshold} "
                      f"delta={cfg.delta} seed={cfg.seed} rows={info.rows} "
                      f"failed={info.failed}", flush=True)

    immut = sum(i.immutability_violations for i in infos)
    errors = [i.error for i in infos if i.error]

    selection_picks: list[SelectionPick] = []
    if spec.phase == "calibration":
        selection_picks = select_pairs(raw, presets, within=spec.match_within,
                                       approx=spec.match_approx)
        sel_map = {p.bucket: p for p in selection_picks}
        matched = build_matched(raw, sel_map, within=spec.match_within,
                                approx=spec.match_approx,
                                source="calibration_in_sample",
                                phase=spec.phase)
    else:
        if spec.calibration_results is None:
            raise ValueError("evaluate phase requires --calibration-results")
        calib_dir = _resolve_calibration_dir(Path(spec.calibration_results))
        sel_path = calib_dir / "results_budget_selection.csv"
        sel_map = read_selection(sel_path)
        selection_picks = list(sel_map.values())
        matched = build_matched(raw, sel_map, within=spec.match_within,
                                approx=spec.match_approx,
                                source="calibration_applied", phase=spec.phase)

    summary = build_summary(matched)
    pareto = build_pareto(raw)
    oracle = (build_oracle(raw, presets, within=spec.match_within,
                           approx=spec.match_approx, phase=spec.phase)
              if spec.phase == "evaluate" else [])

    files = {
        "results_budget_raw.csv": raw,
        "results_budget_matched.csv": matched,
        "results_budget_summary.csv": summary,
        "results_budget_pareto.csv": pareto,
        "results_budget_selection.csv":
            [p.to_row() for p in selection_picks] if selection_picks else [],
        "results_budget_oracle.csv": oracle,
        "results_structure_checkpoints.csv": structure_rows,
    }
    meta = {
        "invocation_id": experiment_id,
        "timestamp": datetime.datetime.now().isoformat(),
        "git_sha": _git_sha(),
        "environment": _environment_metadata(spec.blas_threads),
        "platform": platform.platform(),
        "rss_bytes": _rss_bytes()[0], "rss_kind": _rss_bytes()[1],
        "phase": spec.phase,
        "preset_grid": {
            "beam_width_values": list(spec.beam_values),
            "n_postings_values": list(spec.n_postings_values),
            "presets": [p.to_dict() for p in presets],
        },
        "seeds": {
            "calibration": (list(spec.seeds) if spec.phase == "calibration" else []),
            "evaluation": (list(spec.seeds) if spec.phase == "evaluate" else []),
            "used": list(spec.seeds),
        },
        "workload": spec.to_dict(),
        "search_semantics": {
            "beam_widths_translation": "scalar beam B -> beam_widths=(B,) "
                                       "(per-level survivor count, clamped to last)",
            "depth1_note": "depth==1 takes the flat fast path; beam width is "
                           "unused, only n_postings changes work",
        },
        "matching_rule": {
            "primary_metric": "vectors_scanned_per_query",
            "relative_scan_gap_pct": "100*|ip-oop|/((ip+oop)/2)",
            "matched_within_pct": spec.match_within,
            "approximate_within_pct": spec.match_approx,
            "selection_granularity": "per depth, with a global fallback bucket",
            "objective": "minimize median relative scan gap across checkpoints",
            "tie_breakers": ["lower mean query p50 (both arms)",
                             "lower combined nominal search budget",
                             "lexicographic preset label"],
            "selection_is_out_of_sample": spec.phase == "evaluate",
        },
        "efficiency": {
            "churn_replayed_per": "one (base config, strategy); preset grid is "
                                  "evaluated on cloned checkpoint state",
            "clone_isolation": "checkpoint units are cloned before search",
        },
        "audit_asymmetry": {
            "oop_membership_auditor": "churn.audit_outofplace_membership",
            "note": "both strategies receive a fixed-Delta membership audit here "
                    "(compare.py reports n/a for out-of-place)",
        },
        "immutability_violations": immut,
        "run_errors": errors,
        "n_raw_rows": len(raw),
        "n_structure_rows": len(structure_rows),
        "n_matched_rows": len(matched),
        "structure_metrics": structure_metadata(),
        "calibration_results": (str(spec.calibration_results)
                                if spec.calibration_results else None),
        "output_schema": _output_schema(files),
    }

    out_dir = _unique_out_dir(spec.out_dir, experiment_id, spec.overwrite)
    out_dir.mkdir(parents=True, exist_ok=True)
    written: dict[str, int] = {}
    for name, rows in files.items():
        write_csv(out_dir / name, rows)
        written[name] = len(rows)
    for name in ("results_budget_raw.csv", "results_budget_matched.csv",
                 "results_structure_checkpoints.csv"):
        rows = files[name]
        if write_parquet(out_dir / name.replace(".csv", ".parquet"), rows):
            written[name.replace(".csv", ".parquet")] = len(rows)
    with open(out_dir / "run_metadata.json", "w") as f:
        json.dump(meta, f, indent=2, default=str)

    if progress:
        print()
        print(f"Output directory: {out_dir}")
        print("Files: " + ", ".join(f"{n} ({c} rows)" for n, c in written.items()))
        print()
        if selection_picks:
            print(_render_selection(selection_picks))
            print()
        print(_render_match_counts(matched))
        if immut:
            print(f"WARNING: {immut} checkpoint(s) changed structural counters "
                  f"during search — investigate.")
    return written


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="compare_search_budget",
        description="Equal-search-work comparison of InPlace vs OutOfPlace "
                    "maintenance: one churn replay per (base config, strategy), "
                    "whole search-preset grid evaluated on cloned checkpoints, "
                    "pairs matched by observed vectors_scanned_per_query.",
    )
    p.add_argument("--phase", choices=["calibration", "evaluate"],
                   default="calibration")
    p.add_argument("--dataset", default="synthetic", choices=["synthetic"])
    p.add_argument("--dim", type=int, default=128)
    p.add_argument("--initial-size", dest="initial_size", type=int, default=400)
    p.add_argument("--operations", type=int, default=400)
    p.add_argument("--workload", default="mixed",
                   choices=["mixed", "insert_only", "delete_only"])
    p.add_argument("--insert-prob", dest="insert_prob", type=float, default=0.5)
    p.add_argument("--leaf-size", dest="leaf_size", type=int, default=40)
    p.add_argument("--branching-factor", dest="branching_factor", type=int, default=16)
    p.add_argument("--depths", default="1,2,3")
    p.add_argument("--split-thresholds", dest="split_thresholds", default="24,48")
    p.add_argument("--merge-thresholds", dest="merge_thresholds", default=None)
    p.add_argument("--deltas", default="none,0.3")
    p.add_argument("--calibration-seeds", dest="calibration_seeds", default="0,1,2")
    p.add_argument("--evaluation-seeds", dest="evaluation_seeds",
                   default="3,4,5,6,7,8,9")
    p.add_argument("--beam-width-values", dest="beam_width_values",
                   default=",".join(str(b) for b in BEAM_DEFAULT),
                   help="scalar beam widths; each maps to beam_widths=(B,)")
    p.add_argument("--n-postings-values", dest="n_postings_values",
                   default=",".join(str(n) for n in NPOST_DEFAULT))
    p.add_argument("--k", type=int, default=10)
    p.add_argument("--recall-ks", dest="recall_ks", default="1,10,100")
    p.add_argument("--query-count", dest="query_count", type=int, default=32)
    p.add_argument("--checkpoint-interval", dest="checkpoint_interval",
                   type=int, default=50)
    p.add_argument("--memtable-limit", dest="memtable_limit", type=int, default=100)
    p.add_argument("--compaction-threshold", dest="compaction_threshold",
                   type=int, default=3)
    p.add_argument("--match-within", dest="match_within", type=float, default=5.0,
                   help="scan-gap %% below which a pair is 'matched'")
    p.add_argument("--match-approx", dest="match_approx", type=float, default=10.0,
                   help="scan-gap %% below which a pair is 'approximate'")
    p.add_argument("--calibration-results", dest="calibration_results", default=None,
                   help="calibration output dir (required for --phase evaluate)")
    p.add_argument("--out", default="results/search_budget")
    p.add_argument("--overwrite", action="store_true")
    p.add_argument("--blas-threads", dest="blas_threads", type=int, default=None)
    return p


def resolve_spec(args: argparse.Namespace) -> BudgetSpec:
    if args.phase == "calibration":
        seeds = _parse_int_list(args.calibration_seeds, [0, 1, 2])
    else:
        seeds = _parse_int_list(args.evaluation_seeds, [3, 4, 5, 6, 7, 8, 9])
    recall_ks = tuple(_parse_int_list(args.recall_ks, [1, 10, 100]))
    if args.k not in recall_ks:
        recall_ks = tuple(sorted(set(recall_ks) | {args.k}))
    if args.operations < 1:
        raise ValueError("operations must be >= 1")
    if args.checkpoint_interval < 1:
        raise ValueError("checkpoint_interval must be >= 1")
    if args.match_within < 0 or args.match_approx < args.match_within:
        raise ValueError("require 0 <= match_within <= match_approx")
    beam_values = _parse_int_list(args.beam_width_values, BEAM_DEFAULT)
    n_posts = _parse_int_list(args.n_postings_values, NPOST_DEFAULT)
    if not beam_values or not n_posts:
        raise ValueError("beam/n_postings value lists must be non-empty")
    if any(b < 1 for b in beam_values) or any(n < 1 for n in n_posts):
        raise ValueError("beam/n_postings values must be >= 1")
    calib = Path(args.calibration_results) if args.calibration_results else None
    if args.phase == "evaluate" and calib is None:
        raise ValueError("--phase evaluate requires --calibration-results DIR")
    return BudgetSpec(
        phase=args.phase, dataset=args.dataset, dim=args.dim,
        initial_size=args.initial_size, operations=args.operations,
        workload=args.workload, insert_prob=args.insert_prob,
        leaf_size=args.leaf_size, branching_factor=args.branching_factor,
        depths=_depths_from_text(args.depths, [1, 2, 3]),
        split_thresholds=_parse_int_list(args.split_thresholds, [24, 48]),
        merge_thresholds=_parse_merge_list(args.merge_thresholds, [None]),
        deltas=_parse_delta_list(args.deltas, [None, 0.3]),
        seeds=seeds, beam_values=beam_values, n_postings_values=n_posts,
        k=args.k, recall_ks=recall_ks, query_count=args.query_count,
        checkpoint_interval=args.checkpoint_interval,
        memtable_limit=args.memtable_limit,
        compaction_threshold=args.compaction_threshold,
        match_within=args.match_within, match_approx=args.match_approx,
        out_dir=Path(args.out), overwrite=args.overwrite,
        calibration_results=calib, blas_threads=args.blas_threads,
    )


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    try:
        spec = resolve_spec(args)
    except ValueError as exc:
        parser.error(str(exc))
        return 2

    cm = None
    if spec.blas_threads is not None:
        try:
            from threadpoolctl import threadpool_limits
            cm = threadpool_limits(limits=spec.blas_threads)
            cm.__enter__()
        except Exception:
            cm = None
    try:
        run_experiment(spec)
    finally:
        if cm is not None:
            cm.__exit__(None, None, None)
    return 0
