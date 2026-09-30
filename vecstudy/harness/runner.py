"""harness/runner.py — CLI entry points (DESIGN §10).

  python -m vecstudy.harness.runner sweep --config sweeps/depth_study.yaml --out results/
  python -m vecstudy.harness.runner single --dataset synthetic --strategy inplace --out results/
"""

from __future__ import annotations

import argparse
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
from ..index.segment import Segment
from ..index.query import search
from ..maintain.outofplace import OutOfPlaceMaintainer
from ..maintain.inplace import InPlaceMaintainer
from ..harness.datasets import (
    make_synthetic, read_fvecs, read_ivecs, fetch_sift,
    l2_normalize, compute_ground_truth, recall_at_k, dataset_checksum,
)
from ..harness.results import make_result_row, append_result


def run_experiment(exp: ExperimentConfig, out_dir: str | Path) -> dict:
    """Run one experiment; write result row to Parquet; return row dict."""
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
            maint._flush()
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
    args = parser.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
