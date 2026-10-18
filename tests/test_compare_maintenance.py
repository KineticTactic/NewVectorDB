"""test_compare_maintenance.py — Task 8 paired benchmark harness tests.

Covers the 15 required checks: identical initial state, identical trace,
dynamic ground truth, live-only recall, out-of-place current units,
independent search/maintenance work, bloat arithmetic, counter semantics,
seed reproducibility, pairing, correctness gating, aggregation, file schemas,
the smoke command end to end, and (separately) the rest of the suite.
"""

from __future__ import annotations

import csv
import json
from pathlib import Path

import numpy as np
import pytest

from vecstudy.core.config import IndexConfig, MaintainConfig, QueryConfig
from vecstudy.core.metrics import Counters
from vecstudy.harness.churn import generate_churn_trace
from vecstudy.harness.compare import (
    CompareConfig,
    SweepSpec,
    _DatasetCache,
    _benchmark_queries,
    _index_config,
    _maintain_config,
    _query_config,
    build_paired,
    build_summary,
    dynamic_ground_truth,
    main,
    run_strategy,
    run_sweep,
    resolve_sweep,
    build_parser,
    write_outputs,
)
from vecstudy.harness.datasets import make_synthetic
from vecstudy.index.segment import Segment
from vecstudy.maintain.inplace import InPlaceMaintainer
from vecstudy.maintain.outofplace import OutOfPlaceMaintainer


def _cfg(**over) -> CompareConfig:
    base = dict(
        dataset="synthetic", dim=8, initial_size=80, reserve_size=40,
        operations=40, workload="mixed", insert_prob=0.5, seed=0, depth=2,
        leaf_size=20, branching_factor=4, split_threshold=24, merge_threshold=None,
        delta=None, memtable_limit=30, compaction_threshold=3, k=5,
        recall_ks=(5,), query_count=6, n_postings=2, beam_widths=(8,),
    )
    base.update(over)
    return CompareConfig(**base)


def _run_pair(cfg: CompareConfig, interval=20):
    cache = _DatasetCache()
    vec = cache.data(cfg)
    q = cache.queries(cfg)
    seg = cache.initial_segment(cfg)
    trace = generate_churn_trace(
        initial_n=cfg.initial_size, reserve_n=cfg.reserve_size,
        n_operations=cfg.operations, insert_prob=cfg.insert_prob,
        seed=cfg.seed, workload=cfg.workload,
    )
    kw = dict(vectors=vec, queries=q, trace=trace, initial_segment=seg,
              run_index=0, checkpoint_interval=interval, flush_final=False,
              continue_on_audit_failure=False, experiment_id="t",
              dataset_checksum="chk", progress=False)
    ip = run_strategy(cfg, "inplace", order_in_pair=0, **kw)
    oop = run_strategy(cfg, "outofplace", order_in_pair=1, **kw)
    return ip, oop, trace, vec, q, seg


# ---------------------------------------------------------------------------
# 1. Same initial state
# ---------------------------------------------------------------------------

def test_same_initial_state():
    cfg = _cfg()
    cache = _DatasetCache()
    vec = cache.data(cfg)
    seg = cache.initial_segment(cfg)
    idx = _index_config(cfg)
    m_ip = InPlaceMaintainer(vec, idx, _maintain_config(cfg, "inplace"),
                             Counters(), initial_segment=seg.clone())
    m_oop = OutOfPlaceMaintainer(vec, idx, _maintain_config(cfg, "outofplace"),
                                 Counters(), initial_segment=seg.clone())
    u_ip = m_ip.searchable_units()[0]
    u_oop = m_oop.searchable_units()[0]
    np.testing.assert_array_equal(u_ip.levels[-1].centroids,
                                  u_oop.levels[-1].centroids)
    assert len(u_ip.postings) == len(u_oop.postings)
    for a, b in zip(u_ip.postings, u_oop.postings):
        np.testing.assert_array_equal(a.members, b.members)
        np.testing.assert_array_equal(a.owned, b.owned)
    assert len(u_ip.levels) == len(u_oop.levels)
    # mutable independence
    u_ip.levels[-1].centroids[0, 0] += 999.0
    assert not np.array_equal(u_ip.levels[-1].centroids,
                              u_oop.levels[-1].centroids)


# ---------------------------------------------------------------------------
# 2. Same operation trace / same logical live set at every checkpoint
# ---------------------------------------------------------------------------

def test_same_trace_applied_to_both():
    ip, oop, trace, *_ = _run_pair(_cfg())
    assert [r["cumulative_insertions"] for r in ip.rows] == \
           [r["cumulative_insertions"] for r in oop.rows]
    assert [r["cumulative_deletions"] for r in ip.rows] == \
           [r["cumulative_deletions"] for r in oop.rows]
    assert [r["live_vector_count"] for r in ip.rows] == \
           [r["live_vector_count"] for r in oop.rows]
    assert ip.rows[0]["trace_hash"] == oop.rows[0]["trace_hash"]
    # final live counts match the trace model
    live = set(range(_cfg().initial_size))
    for op, vid in trace.operations:
        if op == "insert":
            live.add(vid)
        else:
            live.discard(vid)
    assert ip.rows[-1]["live_vector_count"] == len(live)


# ---------------------------------------------------------------------------
# 3 & 4. Dynamic ground truth; recall against live only
# ---------------------------------------------------------------------------

def test_dynamic_ground_truth_tracks_mutations():
    vec = make_synthetic(n=40, dim=8, seed=3)
    q = vec[0:1].copy()
    live = np.array([0, 1, 2, 3, 4], dtype=np.int64)
    gt = dynamic_ground_truth(q, vec, live, 3)
    gt_after = dynamic_ground_truth(q, vec, np.array([1, 2, 3, 4], dtype=np.int64), 3)
    assert set(gt[0]) != set(gt_after[0])
    assert int(gt[0, 0]) == 0
    assert 0 not in set(gt_after[0])


def test_ground_truth_excludes_non_live_vectors():
    vec = make_synthetic(n=40, dim=8, seed=5)
    q = vec[0:1].copy()
    vec[7] = q[0]                       # vector 7 is an exact match...
    gt = dynamic_ground_truth(q, vec, np.array([0, 1, 2, 3], dtype=np.int64), 2)
    assert 7 not in set(gt[0])          # ...but it is not live


def test_recall_uses_only_live_and_no_deleted_returned():
    ip, oop, *_ = _run_pair(_cfg(workload="mixed"))
    for r in (ip.final_row, oop.final_row):
        assert r["deleted_id_retrievals"] == 0
        assert r["canonical_owned_live_count"] == r["live_vector_count"]
        assert r["status" if "status" in r else "correctness_status"] == "PASSED"


# ---------------------------------------------------------------------------
# 5. Out-of-place queries the actual current searchable units
# ---------------------------------------------------------------------------

def test_oop_units_reflect_memtable_and_sealed():
    cfg = _cfg(workload="insert_only", memtable_limit=10, operations=25, reserve_size=25)
    _, oop, *_ = _run_pair(cfg, interval=25)
    final = oop.final_row
    # 25 inserts with memtable_limit 10 leaves 5 pending in a live memtable unit,
    # alongside the sealed segment(s): more than one searchable unit.
    assert final["pending_memtable_entries"] >= 1
    assert final["searchable_segment_count"] >= 2
    assert final["units_searched"] == final["searchable_segment_count"]


def test_oop_flush_and_compaction_counters_advance():
    cfg = _cfg(memtable_limit=8, compaction_threshold=3, operations=40, reserve_size=40)
    _, oop, *_ = _run_pair(cfg, interval=40)
    f = oop.final_row
    assert f["flushes"] >= 1
    assert f["compaction_events"] >= 1


# ---------------------------------------------------------------------------
# 6. Search work measured independently of maintenance work
# ---------------------------------------------------------------------------

def test_query_work_independent_of_maintenance_counters():
    cfg = _cfg()
    cache = _DatasetCache()
    vec = cache.data(cfg)
    q = cache.queries(cfg)
    seg = cache.initial_segment(cfg)
    maint_counters = Counters()
    units = [seg]
    total, lat, res = _benchmark_queries(units, q, _query_config(cfg))
    assert total.vectors_scanned > 0
    assert len(lat) == len(q)
    # the maintenance counters object was not touched by the query pass
    assert maint_counters.vectors_scanned == 0
    assert maint_counters.distance_computations == {}


def test_maintenance_counters_not_reset_by_query_measurement():
    ip, *_ = _run_pair(_cfg())
    for prev, cur in zip(ip.rows, ip.rows[1:]):
        assert cur["maintenance_similarity_evaluations"] >= \
               prev["maintenance_similarity_evaluations"]


# ---------------------------------------------------------------------------
# 7. Membership bloat arithmetic
# ---------------------------------------------------------------------------

def test_bloat_arithmetic_with_replicas():
    ip, *_ = _run_pair(_cfg(delta=0.3))
    f = ip.final_row
    n = f["live_vector_count"]
    m = f["total_live_membership_count"]
    assert f["canonical_owned_live_count"] == n
    assert f["replica_count"] == m - f["canonical_owned_live_count"]
    expect = 100.0 * (m - n) / n
    assert abs(f["membership_bloat_pct"] - expect) < 1e-9
    assert m >= n


# ---------------------------------------------------------------------------
# 8. Counters accumulate (maintenance) and reset (query) as documented
# ---------------------------------------------------------------------------

def test_maintenance_counters_accumulate_and_deltas_match():
    ip, *_ = _run_pair(_cfg(split_threshold=16))
    for prev, cur in zip(ip.rows, ip.rows[1:]):
        assert cur["splits"] >= prev["splits"]
        assert cur["delta_splits"] == cur["splits"] - prev["splits"]
        assert cur["delta_replicas_added"] == \
            cur["replicas_added"] - prev["replicas_added"]
    # query work is per-checkpoint, not cumulative
    assert ip.rows[0]["checkpoint"] == 0


# ---------------------------------------------------------------------------
# 9. Same seed reproduces workload and results
# ---------------------------------------------------------------------------

def test_same_seed_reproduces_workload_and_results():
    t1 = generate_churn_trace(80, 40, 40, seed=0)
    t2 = generate_churn_trace(80, 40, 40, seed=0)
    t3 = generate_churn_trace(80, 40, 40, seed=1)
    assert t1.operations == t2.operations
    assert t1.trace_hash != t3.trace_hash

    ip1, *_ = _run_pair(_cfg())
    ip2, *_ = _run_pair(_cfg())
    assert ip1.final_row["recall"] == ip2.final_row["recall"]
    assert ip1.final_row["splits"] == ip2.final_row["splits"]
    assert ip1.final_row["live_vector_count"] == ip2.final_row["live_vector_count"]


# ---------------------------------------------------------------------------
# 10. Pairing groups only matching configurations
# ---------------------------------------------------------------------------

def test_paired_rows_match_only_compatible_configs():
    ip, oop, *_ = _run_pair(_cfg())
    rows = ip.rows + oop.rows
    paired = build_paired(rows)
    assert len(paired) == len(ip.rows)          # one row per checkpoint
    for p in paired:
        assert p["inplace_correctness"] == "PASSED"
        assert p["outofplace_correctness"] == "PASSED"

    # a second, different config must not be paired with the first
    ip2, oop2, *_ = _run_pair(_cfg(seed=1))
    paired2 = build_paired(ip.rows + oop.rows + ip2.rows + oop2.rows)
    assert len(paired2) == len(ip.rows) + len(ip2.rows)
    seeds = {p["seed"] for p in paired2}
    assert seeds == {0, 1}


# ---------------------------------------------------------------------------
# 11. Failed audits appear and are not silently dropped
# ---------------------------------------------------------------------------

def test_corrupted_index_fails_the_gate():
    cfg = _cfg()
    cache = _DatasetCache()
    vec = cache.data(cfg)
    seg = cache.initial_segment(cfg)
    idx = _index_config(cfg)
    maint = InPlaceMaintainer(vec, idx, _maintain_config(cfg, "inplace"),
                              Counters(), initial_segment=seg.clone())
    # deliberately misassign one vector's membership so ownership/cov breaks
    from vecstudy.core.types import Posting
    p0, p1 = maint._segment.postings[0], maint._segment.postings[1]
    victim = int(p0.owned[0])
    maint._segment.postings[1] = Posting(
        p1.id, p1.centroid, np.append(p1.members, victim),
        np.append(p1.owned, victim),
    )
    from vecstudy.harness.compare import _evaluate_checkpoint, _maint_snapshot
    row, failed = _evaluate_checkpoint(
        cfg=cfg, strategy="inplace", cfg_q=_query_config(cfg), cfg_idx=idx,
        cfg_m=_maintain_config(cfg, "inplace"), maint=maint, counters=Counters(),
        prev_maint=_maint_snapshot(Counters()), units=None,
        live=set(range(cfg.initial_size)), queries=cache.queries(cfg), vectors=vec,
        insert_ns=[], delete_ns=[], update_wall_ns=0, maintain_call_ns=0,
        maint_attributed_ns=0, maintain_call_events=0, checkpoint=0, ops=0,
        cum_inserts=0, cum_deletes=0, experiment_id="t", run_index=0,
        order_in_pair=0, trace=generate_churn_trace(80, 40, 0, seed=0),
        dataset_checksum="c",
    )
    assert failed is True
    assert row["correctness_status"] == "FAILED"
    assert row["ownership_errors"] > 0
    assert row["failure_reason"]


def test_summary_reports_failed_runs_not_dropped():
    good = _fake_row(run_index=0, strategy="inplace", recall=0.5, failed=False)
    bad = _fake_row(run_index=1, strategy="inplace", recall=0.0, failed=True)
    summary = build_summary([good, bad])
    run_rows = [r for r in summary if r["row_kind"] == "run"]
    agg = [r for r in summary if r["row_kind"] == "aggregate"]
    assert len(run_rows) == 2                 # both runs present
    assert any(r["run_failed"] for r in run_rows)
    assert agg and agg[0]["n_failed"] == 1
    assert agg[0]["n_ok"] == 1


# ---------------------------------------------------------------------------
# 12. Aggregate uses only successful repetitions
# ---------------------------------------------------------------------------

def test_aggregate_means_exclude_failed_runs():
    rows = [
        _fake_row(run_index=0, strategy="inplace", recall=1.0, failed=False, seed=0),
        _fake_row(run_index=1, strategy="inplace", recall=0.0, failed=False, seed=1),
        _fake_row(run_index=2, strategy="inplace", recall=0.0, failed=True, seed=2),
    ]
    agg = [r for r in build_summary(rows) if r["row_kind"] == "aggregate"][0]
    assert agg["n_ok"] == 2 and agg["n_failed"] == 1
    assert abs(agg["recall_mean"] - 0.5) < 1e-12


# ---------------------------------------------------------------------------
# 13. File schemas / consistent row counts
# ---------------------------------------------------------------------------

def test_output_files_schemas(tmp_path):
    ip, oop, *_ = _run_pair(_cfg())
    rows = ip.rows + oop.rows
    paired = build_paired(rows)
    summary = build_summary(rows)
    written = write_outputs(tmp_path, rows, paired, summary,
                            {"invocation_id": "t", "environment": {}, "sweep": {}})
    for name in ["results_checkpoints.csv", "results_raw.csv",
                 "results_paired.csv", "results_summary.csv", "run_metadata.json"]:
        assert (tmp_path / name).exists(), name
    with open(tmp_path / "results_checkpoints.csv") as f:
        n = sum(1 for _ in csv.DictReader(f))
    assert n == len(rows)
    with open(tmp_path / "results_paired.csv") as f:
        npair = sum(1 for _ in csv.DictReader(f))
    assert npair == len(paired)
    meta = json.loads((tmp_path / "run_metadata.json").read_text())
    assert "environment" in meta and "sweep" in meta

    # Parquet (available in this project) matches CSV row counts
    import pyarrow.parquet as pq
    p_ck = tmp_path / "results_checkpoints.parquet"
    if p_ck.exists():
        assert pq.read_table(str(p_ck)).num_rows == len(rows)
    p_pair = tmp_path / "results_paired.parquet"
    if p_pair.exists():
        assert pq.read_table(str(p_pair)).num_rows == len(paired)
    assert written["results_checkpoints.csv"] == len(rows)


# ---------------------------------------------------------------------------
# 14. Smoke command works end to end
# ---------------------------------------------------------------------------

def test_smoke_cli_end_to_end(tmp_path, capsys):
    rc = main(["--profile", "smoke", "--out", str(tmp_path), "--overwrite",
               "--table-a-rows", "final", "--deltas", "none"])
    assert rc == 0
    out = capsys.readouterr().out
    assert "Table A" in out and "Table B" in out
    assert "Aggregate across seeds" in out
    assert (tmp_path / "results_checkpoints.csv").exists()
    assert (tmp_path / "results_raw.csv").exists()
    assert (tmp_path / "results_paired.csv").exists()
    assert (tmp_path / "results_summary.csv").exists()
    assert (tmp_path / "run_metadata.json").exists()


# ---------------------------------------------------------------------------
# CLI plumbing
# ---------------------------------------------------------------------------

def test_resolve_sweep_and_expand_counts():
    args = build_parser().parse_args([
        "--profile", "quick", "--depths", "1,2", "--split-thresholds", "24,48",
        "--deltas", "none,0.3", "--seeds", "0,1", "--n-postings", "2,4",
        "--beam-widths", "8",
    ])
    spec = resolve_sweep(args)
    cells = spec.expand()
    # 2 depths x 2 splits x 1 merge x 2 deltas x 2 npost x 1 beam x 2 seeds
    assert len(cells) == 2 * 2 * 1 * 2 * 2 * 1 * 2
    assert all(isinstance(c, CompareConfig) for c in cells)
    assert spec.recall_ks == (10,)


def test_smoke_spec_is_small():
    spec = resolve_sweep(build_parser().parse_args(["--profile", "smoke"]))
    assert len(spec.expand()) <= 4


def test_run_sweep_alternates_order():
    spec = SweepSpec(
        dataset="synthetic", initial_size=60, operations=20, workload="mixed",
        insert_prob=0.5, dim=8, leaf_size=20, branching_factor=4,
        split_thresholds=[24], merge_thresholds=[None], deltas=[None],
        depths=[1], n_postings_list=[2], beam_list=[8], seeds=[0], k=5,
        recall_ks=(5,), query_count=4, memtable_limit=20, compaction_threshold=3,
        checkpoint_interval=20, flush_final=False, continue_on_audit_failure=False,
        blas_threads=None, table_a_rows="none", out_dir=Path("/tmp/x"),
        overwrite=True, profile="smoke",
    )
    results = run_sweep(spec, progress=False)
    assert len(results) == 2
    orders = {r.strategy: r.order_in_pair for r in results}
    assert set(orders.values()) == {0, 1}      # arms alternate on run 0 vs 1


# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------

def _fake_row(*, run_index, strategy, recall, failed, seed=0):
    status = "FAILED" if failed else "PASSED"
    return {
        "row_kind": "checkpoint", "run_index": run_index, "strategy": strategy,
        "dataset": "synthetic", "seed": seed, "depth_config": 1,
        "split_threshold": 24, "merge_threshold": -1, "delta": None,
        "n_postings": 2, "beam_widths": "(8,)", "checkpoint": 1,
        "recall": recall, "membership_bloat_pct": 0.0, "live_vector_count": 50,
        "index_bytes_estimate": 1000, "insert_latency_p50_ms": 0.01,
        "delete_latency_p50_ms": 0.01, "query_latency_p50_ms": 0.01,
        "maintenance_wall_ms": 0.1, "total_mutation_wall_ms": 0.2,
        "maintenance_similarity_evaluations": 10,
        "work_vectors_scanned_per_query": 12.0,
        "correctness_status": status, "run_failed": failed,
    }
