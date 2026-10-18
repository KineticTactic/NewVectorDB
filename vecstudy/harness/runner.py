"""harness/runner.py — CLI entry points (DESIGN §10).

  python -m vecstudy.harness.runner sweep --config sweeps/depth_study.yaml --out results/
  python -m vecstudy.harness.runner single --dataset synthetic --strategy inplace --out results/
  python -m vecstudy.harness.runner churn --config sweeps/churn_depth_study.yaml --out results/churn/
"""

from __future__ import annotations

import argparse
import copy
import datetime
import os
import sys
from pathlib import Path

import numpy as np
import yaml

# Pin single-threaded BLAS — determinism requirement (§1.3) + §3
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("NUMEXPR_NUM_THREADS", "1")

from ..core.config import IndexConfig, QueryConfig, MaintainConfig, ExperimentConfig
from ..core.metrics import Counters
from ..core.types import VectorId
from ..index.segment import Segment
from ..index.query import search
from ..maintain.base import IndexMaintainer
from ..maintain.outofplace import OutOfPlaceMaintainer
from ..maintain.inplace import InPlaceMaintainer
from ..harness.datasets import (
    make_synthetic, read_fvecs, read_ivecs, fetch_sift,
    l2_normalize, compute_ground_truth, recall_at_k, dataset_checksum,
)
from ..harness.results import make_result_row, append_result, _git_sha
from ..harness.churn import (
    ChurnTrace, generate_churn_trace, compute_dynamic_ground_truth,
    audit_maintainer, AuditResult,
)


def run_experiment(exp: ExperimentConfig, out_dir: str | Path) -> dict:
    """Run one static experiment; write result row to Parquet; return row dict."""
    out_dir = Path(out_dir)
    base, queries, gt = _load_dataset(exp.dataset, exp.query.k)
    dim = base.shape[1]
    dcs = dataset_checksum(base)
    cfg_idx = IndexConfig(dim=dim, leaf_size=exp.index.leaf_size,
        branching_factor=exp.index.branching_factor, depth=exp.index.depth,
        delta=exp.index.delta, rng_rule=exp.index.rng_rule, seed=exp.index.seed)
    cfg_q, cfg_m = exp.query, exp.maintain
    counters = Counters()
    vids = np.arange(len(base), dtype=np.int64)
    if cfg_m.strategy == "outofplace":
        maint = OutOfPlaceMaintainer(base, cfg_idx, cfg_m, counters)
        for vid in vids:
            maint.insert(int(vid), base[vid])
            maint.maintain()
        if maint._memtable:
            maint.flush()
    else:
        maint = InPlaceMaintainer(base, cfg_idx, cfg_m, counters, initial_vids=vids)
    units = maint.searchable_units()
    all_res: list[list[int]] = []
    total_q = Counters()
    for q_vec in queries:
        counters.reset_search()
        res = search(q_vec, units, cfg_q, counters)
        all_res.append([v for v, _ in res])
        total_q = total_q + counters.snapshot()
    recall = recall_at_k(all_res, gt, cfg_q.k)
    sizes = np.array([len(p.owned) for u in units for p in u.postings] or [0], dtype=np.float64)
    psd = {"mean": float(sizes.mean()), "p50": float(np.percentile(sizes, 50)),
           "p99": float(np.percentile(sizes, 99)), "max": int(sizes.max())}
    depth = len(units[0].levels) if units else 0
    tot_owned = sum(len(p.owned) for u in units for p in u.postings)
    tot_mem   = sum(len(p.members) for u in units for p in u.postings)
    bloat = (tot_mem / tot_owned - 1.0) if tot_owned else 0.0
    row = make_result_row(
        config_dict=exp.to_dict(),
        counters_dict={**total_q.search_dict(), **counters.maintain_dict()},
        recall=recall, bloat=bloat, posting_size_dist=psd,
        depth=depth, unit_count=len(units), dataset_checksum=dcs)
    append_result(row, out_dir)
    return row


def _load_dataset(dataset: str, k: int):
    if dataset == "synthetic":
        base = make_synthetic(n=10_000, dim=128, seed=0)
        queries = make_synthetic(n=100, dim=128, seed=1)
        gt = compute_ground_truth(queries, base, k=k, dataset_key="synthetic")
        return base, queries, gt
    if dataset == "sift1m":
        sift_dir = fetch_sift()
        base    = l2_normalize(read_fvecs(sift_dir / "sift_base.fvecs"))
        queries = l2_normalize(read_fvecs(sift_dir / "sift_query.fvecs"))
        gt      = read_ivecs(sift_dir / "sift_groundtruth.ivecs")
        return base, queries, gt
    if dataset == "siftsmall":
        d = Path("D:/projects/VectorDatabaseTutorial/data")
        base    = l2_normalize(read_fvecs(d / "siftsmall_base.fvecs"))
        queries = l2_normalize(read_fvecs(d / "siftsmall_query.fvecs"))
        gt      = read_ivecs(d / "siftsmall_groundtruth.ivecs")
        return base, queries, gt
    raise ValueError(f"Unknown dataset: {dataset!r}")


def _load_churn_dataset(dataset: str, initial_size: int, reserve_size: int, seed: int = 0):
    total_n = initial_size + reserve_size
    if dataset == "synthetic":
        vectors = make_synthetic(n=total_n, dim=128, seed=seed)
        queries = make_synthetic(n=100, dim=128, seed=seed + 1)
        return vectors, queries
    if dataset == "sift1m":
        sift_dir = fetch_sift()
        base = l2_normalize(read_fvecs(sift_dir / "sift_base.fvecs"))
        queries = l2_normalize(read_fvecs(sift_dir / "sift_query.fvecs"))
        if len(base) < total_n:
            raise ValueError(f"SIFT1M base has {len(base)} vectors, but {total_n} requested")
        return base[:total_n], queries
    if dataset == "siftsmall":
        d = Path("D:/projects/VectorDatabaseTutorial/data")
        base = l2_normalize(read_fvecs(d / "siftsmall_base.fvecs"))
        queries = l2_normalize(read_fvecs(d / "siftsmall_query.fvecs"))
        return base[:total_n], queries
    raise ValueError(f"Unknown dataset: {dataset!r}")


def _evaluate_checkpoint(
    *,
    maint: IndexMaintainer,
    cfg_q: QueryConfig,
    queries: np.ndarray,
    vectors: np.ndarray,
    current_live_vids: set[int],
    cp_idx: int,
    cum_ops: int,
    cum_inserts: int,
    cum_deletes: int,
    initial_size: int,
    cfg_idx: IndexConfig,
    cfg_m: MaintainConfig,
    dataset_name: str,
    dcs: str,
    trace: ChurnTrace,
    counters: Counters,
    prev_counters: Counters,
) -> dict:
    """Evaluate maintainer at a checkpoint, audit invariants, and construct result row."""
    # 1. Audit structural invariants and NPA
    audit = audit_maintainer(maint, expected_live_vids=current_live_vids)
    # PASSED means coverage + unique canonical ownership + NPA all hold. It is
    # not an overlap-correctness claim: this auditor never re-checks the Δ
    # replica rule. Membership has its own audit (`audit_membership`), which the
    # test suite runs after every insert/split/delete; it is not folded into
    # `correctness_status`, whose meaning stays ownership/NPA-only.
    correctness_status = "PASSED" if audit.passed else "FAILED"

    # 2. Compute exact ground truth on currently live vectors
    live_vids_arr = np.array(sorted(current_live_vids), dtype=np.int64)
    gt_current = compute_dynamic_ground_truth(queries, vectors, live_vids_arr, k=cfg_q.k)

    # 3. Search query set across searchable units
    units = maint.searchable_units()
    all_res: list[list[int]] = []
    total_q = Counters()
    for q_vec in queries:
        counters.reset_search()
        res = search(q_vec, units, cfg_q, counters)
        all_res.append([v for v, _ in res])
        total_q = total_q + counters.snapshot()

    # 4. Recall
    recall = recall_at_k(all_res, gt_current, cfg_q.k)

    # 5. Index structure statistics
    sizes = np.array([len(p.owned) for u in units for p in u.postings] or [0], dtype=np.float64)
    psd = {
        "mean": float(sizes.mean()),
        "p50": float(np.percentile(sizes, 50)),
        "p99": float(np.percentile(sizes, 99)),
        "max": int(sizes.max()),
    }
    actual_depth = max((len(u.levels) for u in units), default=0)
    min_depth = min((len(u.levels) for u in units), default=0)
    max_depth = actual_depth

    tot_owned = sum(len(p.owned) for u in units for p in u.postings)
    tot_mem = sum(len(p.members) for u in units for p in u.postings)
    bloat = (tot_mem / tot_owned - 1.0) if tot_owned else 0.0

    stats = maint.stats()
    pending_memtable_size = stats.get("memtable_size", 0)

    # 6. Maintenance deltas since previous checkpoint
    delta_splits = counters.splits - prev_counters.splits
    delta_vectors_reassigned = counters.vectors_reassigned - prev_counters.vectors_reassigned
    delta_hierarchy_rebuilds = counters.hierarchy_rebuilds - prev_counters.hierarchy_rebuilds
    delta_compaction_events = counters.compaction_events - prev_counters.compaction_events
    delta_vectors_reclustered = counters.vectors_reclustered - prev_counters.vectors_reclustered
    delta_flushes = counters.flushes - prev_counters.flushes
    delta_forced_flushes = counters.forced_checkpoint_flushes - prev_counters.forced_checkpoint_flushes

    # Distances in maintenance
    total_dist = counters.distance_computations.get("build", 0)
    prev_dist = prev_counters.distance_computations.get("build", 0)
    delta_maintain_dist = total_dist - prev_dist

    churn_frac = (cum_inserts + cum_deletes) / float(initial_size) if initial_size else 0.0

    row = {
        # Experiment identification
        "dataset": dataset_name,
        "dataset_checksum": dcs,
        "strategy": cfg_m.strategy,
        "workload": trace.workload,
        "seed": cfg_idx.seed,
        "trace_seed": trace.seed,
        "trace_hash": trace.trace_hash,

        # Configurations
        "initial_depth": cfg_idx.depth if cfg_idx.depth is not None else actual_depth,
        "configured_depth": cfg_idx.depth if cfg_idx.depth is not None else -1,
        "depth": actual_depth,
        "actual_depth": actual_depth,
        "min_depth": min_depth,
        "max_depth": max_depth,
        "branching_factor": cfg_idx.branching_factor,
        "k": cfg_q.k,
        "beam_widths": str(list(cfg_q.beam_widths)),
        "n_postings": cfg_q.n_postings,

        # Checkpoint markers
        "checkpoint": cp_idx,
        "operations_processed": cum_ops,
        "cumulative_insertions": cum_inserts,
        "cumulative_deletions": cum_deletes,
        "churn_fraction": churn_frac,
        "live_vector_count": len(current_live_vids),

        # Search performance
        "recall": float(recall),
        "dist_routing": total_q.distance_computations.get("routing", 0),
        "dist_scan": total_q.distance_computations.get("scan", 0),
        "dist_build": total_q.distance_computations.get("build", 0),
        "postings_read_estimated": total_q.postings_read,
        "bytes_read_estimated": total_q.bytes_read,
        "vectors_scanned": total_q.vectors_scanned,

        # Structural & storage metrics
        "unit_count": len(units),
        "searchable_segment_count": len(units),
        "pending_memtable_size": pending_memtable_size,
        "posting_size_mean": psd["mean"],
        "posting_size_p50": psd["p50"],
        "posting_size_p99": psd["p99"],
        "posting_size_max": psd["max"],
        "bloat": float(bloat),

        # Cumulative maintenance counters
        "splits": counters.splits,
        "vectors_reassigned": counters.vectors_reassigned,
        "hierarchy_rebuilds": counters.hierarchy_rebuilds,
        "compaction_events": counters.compaction_events,
        "vectors_reclustered": counters.vectors_reclustered,
        "flushes": counters.flushes,
        "forced_checkpoint_flushes": counters.forced_checkpoint_flushes,
        "npa_violations_detected": counters.npa_violations_detected,

        # Maintenance deltas
        "delta_splits": delta_splits,
        "delta_vectors_reassigned": delta_vectors_reassigned,
        "delta_hierarchy_rebuilds": delta_hierarchy_rebuilds,
        "delta_compaction_events": delta_compaction_events,
        "delta_vectors_reclustered": delta_vectors_reclustered,
        "delta_flushes": delta_flushes,
        "delta_forced_checkpoint_flushes": delta_forced_flushes,
        "delta_maintain_dist": delta_maintain_dist,

        # Correctness & invariants
        "correctness_status": correctness_status,
        "npa_violations": len(audit.npa_violations),
        "ownership_errors": len(audit.ownership_errors),
        "coverage_errors": len(audit.coverage_errors),

        # Metadata
        "git_sha": _git_sha(),
        "timestamp": datetime.datetime.now(datetime.timezone.utc).isoformat(),
    }
    return row


def run_churn_experiment(
    *,
    dataset_name: str,
    cfg_idx: IndexConfig,
    cfg_q: QueryConfig,
    cfg_m: MaintainConfig,
    initial_size: int,
    reserve_size: int,
    trace: ChurnTrace,
    checkpoint_interval: int,
    flush_memtable_at_checkpoint: bool = False,
    out_dir: str | Path,
    vectors: np.ndarray | None = None,
    queries: np.ndarray | None = None,
    initial_segment: Segment | None = None,
) -> list[dict]:
    """Execute a single churn experiment arm across successive checkpoints."""
    out_dir = Path(out_dir)

    # 1. Load data if not supplied
    if vectors is None or queries is None:
        vectors, queries = _load_churn_dataset(dataset_name, initial_size, reserve_size, seed=cfg_idx.seed)
    dim = vectors.shape[1]
    dcs = dataset_checksum(vectors)

    # Update cfg_idx dim
    if cfg_idx.dim == 0:
        cfg_idx = IndexConfig(
            dim=dim,
            leaf_size=cfg_idx.leaf_size,
            branching_factor=cfg_idx.branching_factor,
            depth=cfg_idx.depth,
            delta=cfg_idx.delta,
            rng_rule=cfg_idx.rng_rule,
            seed=cfg_idx.seed,
        )

    # 2. Build or clone the common initial index
    initial_vids = np.arange(initial_size, dtype=np.int64)
    if initial_segment is None:
        build_counters = Counters()
        initial_segment = Segment.build(initial_vids, vectors, cfg_idx, build_counters)

    counters = Counters()
    if cfg_m.strategy == "inplace":
        maint = InPlaceMaintainer(vectors, cfg_idx, cfg_m, counters, initial_segment=initial_segment.clone())
    else:
        maint = OutOfPlaceMaintainer(vectors, cfg_idx, cfg_m, counters, initial_segment=initial_segment.clone())

    current_live_vids: set[int] = set(range(initial_size))
    prev_counters = counters.snapshot()
    rows: list[dict] = []

    # 3. Checkpoint 0 (before churn)
    cp0_row = _evaluate_checkpoint(
        maint=maint,
        cfg_q=cfg_q,
        queries=queries,
        vectors=vectors,
        current_live_vids=current_live_vids,
        cp_idx=0,
        cum_ops=0,
        cum_inserts=0,
        cum_deletes=0,
        initial_size=initial_size,
        cfg_idx=cfg_idx,
        cfg_m=cfg_m,
        dataset_name=dataset_name,
        dcs=dcs,
        trace=trace,
        counters=counters,
        prev_counters=prev_counters,
    )
    append_result(cp0_row, out_dir)
    rows.append(cp0_row)
    prev_counters = counters.snapshot()

    print(
        f"[churn] cp=0/{len(trace.operations)//checkpoint_interval} op=0/{len(trace.operations)} "
        f"strat={cfg_m.strategy:<10} depth={cp0_row['depth']} live={cp0_row['live_vector_count']} "
        f"recall={cp0_row['recall']:.4f} npa_viol={cp0_row['npa_violations']}"
    )

    # 4. Process churn workload in batches
    cum_ops = 0
    cum_inserts = 0
    cum_deletes = 0
    cp_idx = 1
    total_ops = len(trace.operations)

    for i, (op_type, vid) in enumerate(trace.operations):
        if op_type == "insert":
            maint.insert(vid, vectors[vid])
            maint.maintain()
            current_live_vids.add(vid)
            cum_inserts += 1
        elif op_type == "delete":
            maint.delete(vid)
            maint.maintain()
            current_live_vids.remove(vid)
            cum_deletes += 1
        cum_ops += 1

        # Check if checkpoint reached
        is_checkpoint = (cum_ops % checkpoint_interval == 0) or (cum_ops == total_ops)
        if is_checkpoint:
            if flush_memtable_at_checkpoint and hasattr(maint, "flush"):
                maint.flush()
                counters.forced_checkpoint_flushes += 1

            cp_row = _evaluate_checkpoint(
                maint=maint,
                cfg_q=cfg_q,
                queries=queries,
                vectors=vectors,
                current_live_vids=current_live_vids,
                cp_idx=cp_idx,
                cum_ops=cum_ops,
                cum_inserts=cum_inserts,
                cum_deletes=cum_deletes,
                initial_size=initial_size,
                cfg_idx=cfg_idx,
                cfg_m=cfg_m,
                dataset_name=dataset_name,
                dcs=dcs,
                trace=trace,
                counters=counters,
                prev_counters=prev_counters,
            )
            append_result(cp_row, out_dir)
            rows.append(cp_row)
            prev_counters = counters.snapshot()

            print(
                f"[churn] cp={cp_idx} op={cum_ops}/{total_ops} "
                f"strat={cfg_m.strategy:<10} depth={cp_row['depth']} live={cp_row['live_vector_count']} "
                f"recall={cp_row['recall']:.4f} npa_viol={cp_row['npa_violations']}"
            )
            cp_idx += 1

    return rows


def run_churn_sweep(spec: dict, out_dir: str | Path) -> list[dict]:
    """Run an entire sweep of churn configurations, sharing dataset and common initial index."""
    out_dir = Path(out_dir)
    dataset_name = spec.get("dataset", "synthetic")
    initial_size = int(spec.get("initial_size", 5000))
    reserve_size = int(spec.get("reserve_size", 5000))

    base_idx = spec.get("index", {})
    base_qry = spec.get("query", {})
    base_mnt = spec.get("maintain", {})
    churn_cfg = spec.get("churn", {})

    seed = int(base_idx.get("seed", 0))
    vectors, queries = _load_churn_dataset(dataset_name, initial_size, reserve_size, seed=seed)
    dim = vectors.shape[1]

    # Pre-generate deterministic churn trace
    trace = generate_churn_trace(
        initial_n=initial_size,
        reserve_n=reserve_size,
        n_operations=int(churn_cfg.get("operations", 2000)),
        insert_prob=float(churn_cfg.get("insert_prob", 0.6)),
        seed=int(churn_cfg.get("seed", 42)),
        workload=str(churn_cfg.get("workload", "mixed")),
    )
    checkpoint_interval = int(churn_cfg.get("checkpoint_interval", 500))
    flush_memtable = bool(churn_cfg.get("flush_memtable_at_checkpoint", False))

    # Cache initial segments per (leaf_size, branching_factor, depth, delta, rng_rule, seed)
    initial_segments_cache: dict[tuple, Segment] = {}
    initial_vids = np.arange(initial_size, dtype=np.int64)

    all_rows: list[dict] = []
    grid = spec.get("grid", [{}])

    for cell in grid:
        idx_kw = {**base_idx, **cell.get("index", {})}
        qry_kw = {**base_qry, **cell.get("query", {})}
        mnt_kw = {**base_mnt, **cell.get("maintain", {})}

        beam_raw = qry_kw.pop("beam_widths", [8])
        qry_kw["beam_widths"] = tuple([beam_raw] if isinstance(beam_raw, int) else beam_raw)

        cfg_idx = IndexConfig(dim=dim, **idx_kw)
        cfg_q = QueryConfig(**qry_kw)
        cfg_m = MaintainConfig(**mnt_kw)

        seg_key = (
            cfg_idx.dim, cfg_idx.leaf_size, cfg_idx.branching_factor,
            cfg_idx.depth, cfg_idx.delta, cfg_idx.rng_rule, cfg_idx.seed,
        )
        if seg_key not in initial_segments_cache:
            build_counters = Counters()
            initial_segments_cache[seg_key] = Segment.build(
                initial_vids, vectors, cfg_idx, build_counters
            )

        base_seg = initial_segments_cache[seg_key]
        rows = run_churn_experiment(
            dataset_name=dataset_name,
            cfg_idx=cfg_idx,
            cfg_q=cfg_q,
            cfg_m=cfg_m,
            initial_size=initial_size,
            reserve_size=reserve_size,
            trace=trace,
            checkpoint_interval=checkpoint_interval,
            flush_memtable_at_checkpoint=flush_memtable,
            out_dir=out_dir,
            vectors=vectors,
            queries=queries,
            initial_segment=base_seg.clone(),
        )
        all_rows.extend(rows)

    return all_rows


def cmd_single(args) -> None:
    beam = tuple(int(b) for b in args.beam.split(","))
    cfg = ExperimentConfig(
        index=IndexConfig(dim=0, branching_factor=args.branching_factor,
                          depth=args.depth, delta=args.delta, seed=args.seed),
        query=QueryConfig(k=args.k, beam_widths=beam, n_postings=args.n_postings),
        maintain=MaintainConfig(strategy=args.strategy),
        dataset=args.dataset,
    )
    row = run_experiment(cfg, args.out)
    print(f"recall@{args.k}={row['recall']:.4f}  bloat={row['bloat']:.3f}")


def cmd_sweep(args) -> None:
    with open(args.config) as f:
        spec = yaml.safe_load(f)
    base_idx = spec.get("index", {})
    base_qry = spec.get("query", {})
    base_mnt = spec.get("maintain", {})
    dataset  = spec.get("dataset", "synthetic")
    for cell in spec.get("grid", [{}]):
        idx_kw = {**base_idx, **cell.get("index", {})}
        qry_kw = {**base_qry, **cell.get("query", {})}
        mnt_kw = {**base_mnt, **cell.get("maintain", {})}
        beam_raw = qry_kw.pop("beam_widths", [8])
        qry_kw["beam_widths"] = tuple([beam_raw] if isinstance(beam_raw, int) else beam_raw)
        cfg = ExperimentConfig(index=IndexConfig(dim=0, **idx_kw),
                               query=QueryConfig(**qry_kw),
                               maintain=MaintainConfig(**mnt_kw), dataset=dataset)
        try:
            row = run_experiment(cfg, args.out)
            print(f"[sweep] recall={row['recall']:.4f}  {cell}")
        except Exception as e:
            print(f"[sweep] ERROR {cell}: {e}", file=sys.stderr)


def cmd_churn(args) -> None:
    with open(args.config) as f:
        spec = yaml.safe_load(f)
    out_dir = Path(args.out)
    rows = run_churn_sweep(spec, out_dir)
    print(f"\n[churn complete] Generated {len(rows)} checkpoint rows across all sweep cells. Written to {out_dir}")


def main() -> None:
    parser = argparse.ArgumentParser(prog="vecstudy.harness.runner")
    sub = parser.add_subparsers(dest="cmd", required=True)

    sp = sub.add_parser("single")
    sp.add_argument("--dataset", default="synthetic")
    sp.add_argument("--strategy", required=True, choices=["inplace", "outofplace"])
    sp.add_argument("--depth", type=int, default=None)
    sp.add_argument("--beam", default="8")
    sp.add_argument("--delta", type=float, default=None)
    sp.add_argument("--out", default="results/")
    sp.add_argument("--k", type=int, default=10)
    sp.add_argument("--n-postings", dest="n_postings", type=int, default=1)
    sp.add_argument("--branching-factor", dest="branching_factor", type=int, default=16)
    sp.add_argument("--seed", type=int, default=0)
    sp.set_defaults(func=cmd_single)

    sw = sub.add_parser("sweep")
    sw.add_argument("--config", required=True)
    sw.add_argument("--out", default="results/")
    sw.set_defaults(func=cmd_sweep)

    ch = sub.add_parser("churn")
    ch.add_argument("--config", required=True)
    ch.add_argument("--out", default="results/churn/")
    ch.set_defaults(func=cmd_churn)

    args = parser.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
