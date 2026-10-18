"""test_search_budget.py — equal-search-work harness tests.

Covers: beam_width semantics (per-level, depth-1 irrelevant), search-preset
immutability (no structural counter drift, same checkpoint state for all
presets), same live-only ground truth, relative scan-gap matching arithmetic,
unmatched visibility, calibration/evaluation seed separation, calibration-only
pair selection, Pareto dominance, output schemas, and an end-to-end CLI smoke.
"""

from __future__ import annotations

import csv
import json

import numpy as np
import pytest

from vecstudy.core.metrics import Counters
from vecstudy.harness.churn import generate_churn_trace
from vecstudy.harness.compare import CompareConfig, _DatasetCache, _dataset_checksum
from vecstudy.harness.search_budget import (
    BEAM_DEFAULT,
    NPOST_DEFAULT,
    SearchPreset,
    build_matched,
    build_parser,
    build_pareto,
    build_presets,
    build_summary,
    match_status,
    read_selection,
    relative_scan_gap,
    resolve_spec,
    run_base_config,
    run_experiment,
    select_pairs,
)


# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------

def _tiny_cfg(**over) -> CompareConfig:
    base = dict(
        dataset="synthetic", dim=8, initial_size=80, reserve_size=40,
        operations=40, workload="mixed", insert_prob=0.5, seed=0, depth=2,
        leaf_size=20, branching_factor=4, split_threshold=24,
        merge_threshold=None, delta=None, memtable_limit=30,
        compaction_threshold=3, k=5, recall_ks=(1, 5), query_count=6,
        n_postings=1, beam_widths=(1,),
    )
    base.update(over)
    return CompareConfig(**base)


def _run_tiny(cfg: CompareConfig, strategy: str, presets: list[SearchPreset],
              interval: int = 20):
    cache = _DatasetCache()
    vectors = cache.data(cfg)
    queries = cache.queries(cfg)
    seg = cache.initial_segment(cfg)
    trace = generate_churn_trace(
        initial_n=cfg.initial_size, reserve_n=cfg.reserve_size,
        n_operations=cfg.operations, insert_prob=cfg.insert_prob,
        seed=cfg.seed, workload=cfg.workload,
    )
    return run_base_config(
        cfg, strategy, presets, vectors=vectors, queries=queries, trace=trace,
        initial_segment=seg, checkpoint_interval=interval, run_index=0,
        order_in_pair=0, experiment_id="t",
        dataset_checksum=_dataset_checksum(vectors),
    )


def _cli(*args):
    return build_parser().parse_args(list(args))


# ---------------------------------------------------------------------------
# 1. beam_width semantics / translation
# ---------------------------------------------------------------------------

def test_scalar_beam_translates_to_single_element_tuple():
    assert SearchPreset(8, 4).beam_widths == (8,)
    assert SearchPreset(2, 1).beam_widths == (2,)
    assert build_presets([2, 4], [1, 2]) == [
        SearchPreset(2, 1), SearchPreset(2, 2), SearchPreset(4, 1), SearchPreset(4, 2)]


def test_depth1_beam_width_is_irrelevant():
    """depth==1 uses the flat fast path: beam width must not change work."""
    cfg = _tiny_cfg(depth=1)
    rows, _ = _run_tiny(cfg, "inplace", build_presets([2, 4, 8, 16], [2]))
    by_beam = {}
    for r in rows:
        if r["n_postings"] == 2:
            by_beam[r["beam_scalar"]] = r["vectors_scanned_per_query"]
    assert len(set(by_beam.values())) == 1, by_beam


# ---------------------------------------------------------------------------
# 2. matching arithmetic
# ---------------------------------------------------------------------------

def test_relative_scan_gap_formula():
    assert relative_scan_gap(100.0, 100.0) == 0.0
    # |110-100| / 105 * 100
    assert abs(relative_scan_gap(110.0, 100.0) - (10.0 / 105.0 * 100.0)) < 1e-9
    # symmetric
    assert relative_scan_gap(100.0, 110.0) == relative_scan_gap(110.0, 100.0)
    assert not np.isnan(relative_scan_gap(0.0, 0.0))
    assert relative_scan_gap(0.0, 0.0) == 0.0


def test_match_status_thresholds():
    assert match_status(0.0, 5.0, 10.0) == "matched"
    assert match_status(5.0, 5.0, 10.0) == "matched"
    assert match_status(7.5, 5.0, 10.0) == "approximate"
    assert match_status(10.0, 5.0, 10.0) == "approximate"
    assert match_status(12.0, 5.0, 10.0) == "unmatched"
    assert match_status(float("nan"), 5.0, 10.0) == "unmatched"


# ---------------------------------------------------------------------------
# 3. same checkpoint state for every preset + immutability
# ---------------------------------------------------------------------------

def test_all_presets_share_the_same_checkpoint_state():
    cfg = _tiny_cfg()
    presets = build_presets([2, 8], [1, 4])
    rows, _ = _run_tiny(cfg, "inplace", presets)
    groups: dict[tuple, list[dict]] = {}
    for r in rows:
        key = (r["seed"], r["depth_config"], r["split_threshold"],
               r["merge_threshold"], r["delta"], r["checkpoint"])
        groups.setdefault(key, []).append(r)
    assert groups
    for key, rs in groups.items():
        assert len(rs) == len(presets)
        for field in ("operations_processed", "live_vector_count",
                      "units_searched", "splits", "merges", "flushes",
                      "searchable_segment_count" if "searchable_segment_count" in rs[0]
                      else "total_live_membership_count"):
            vals = {r[field] for r in rs}
            assert len(vals) == 1, (key, field, vals)


def test_search_presets_do_not_change_structural_counters():
    cfg = _tiny_cfg()
    one, info_one = _run_tiny(cfg, "inplace", [SearchPreset(2, 1)])
    many, info_many = _run_tiny(cfg, "inplace", build_presets([2, 4, 8], [1, 2, 4]))
    assert info_one.immutability_violations == 0
    assert info_many.immutability_violations == 0

    def trajectory(rows):
        by_cp: dict[int, dict] = {}
        for r in rows:
            by_cp.setdefault(r["checkpoint"], r)
        return {cp: (r["splits"], r["merges"], r["flushes"],
                     r["operations_processed"], r["live_vector_count"])
                for cp, r in by_cp.items()}

    assert trajectory(one) == trajectory(many)


# ---------------------------------------------------------------------------
# 4. same live-only ground truth
# ---------------------------------------------------------------------------

def test_no_deleted_ids_returned_any_preset():
    cfg = _tiny_cfg(workload="mixed")
    rows, _ = _run_tiny(cfg, "inplace", build_presets([2, 8], [1, 4]))
    assert rows
    assert all(r["deleted_id_retrievals"] == 0 for r in rows)
    # correctness gate is checkpoint+preset derived and surfaces failures
    assert all(r["correctness_status"] in ("PASSED", "FAILED") for r in rows)


def test_recall_uses_only_live_ground_truth():
    from vecstudy.harness.compare import dynamic_ground_truth
    from vecstudy.harness.datasets import make_synthetic
    vec = make_synthetic(n=30, dim=8, seed=3)
    q = vec[0:1].copy()
    vec[9] = q[0]                       # exact match but not live
    gt = dynamic_ground_truth(q, vec, np.array([0, 1, 2, 3], dtype=np.int64), 2)
    assert 9 not in set(gt[0])


# ---------------------------------------------------------------------------
# 5. calibration / evaluation seed separation
# ---------------------------------------------------------------------------

def test_calibration_and_evaluation_seeds_are_separate():
    cal = resolve_spec(_cli("--phase", "calibration"))
    ev = resolve_spec(_cli("--phase", "evaluate",
                           "--calibration-results", "/tmp/none"))
    assert cal.seeds == [0, 1, 2]
    assert ev.seeds == [3, 4, 5, 6, 7, 8, 9]
    assert set(cal.seeds).isdisjoint(set(ev.seeds))


def test_evaluate_requires_calibration_results():
    with pytest.raises(ValueError):
        resolve_spec(_cli("--phase", "evaluate"))


def test_custom_seed_split_respected():
    cal = resolve_spec(_cli("--phase", "calibration", "--calibration-seeds", "0,1"))
    ev = resolve_spec(_cli("--phase", "evaluate", "--calibration-results", "x",
                           "--evaluation-seeds", "10,11"))
    assert cal.seeds == [0, 1]
    assert ev.seeds == [10, 11]


# ---------------------------------------------------------------------------
# 6. selection is a function of calibration rows only
# ---------------------------------------------------------------------------

def _synth_rows(depth, seed, checkpoint, ip_scan, oop_scan, strategy, label,
                recall=0.5, failed=False):
    beam, npost = label[1:].split("p")
    return {
        "dataset": "synthetic", "workload": "mixed", "trace_hash": f"th{seed}",
        "seed": seed, "depth_config": depth, "split_threshold": 24,
        "merge_threshold": -1, "delta": None, "k": 10, "checkpoint": checkpoint,
        "operations_processed": checkpoint * 10, "live_vector_count": 100,
        "strategy": strategy, "search_preset": label,
        "beam_scalar": int(beam), "n_postings": int(npost),
        "recall": recall, "recall_at_10": recall,
        "vectors_scanned_per_query": ip_scan if strategy == "inplace" else oop_scan,
        "query_latency_p50_ms": 0.01, "query_latency_p95_ms": 0.02,
        "correctness_status": "FAILED" if failed else "PASSED",
        "run_failed": failed,
    }


def test_select_pairs_minimizes_median_gap():
    presets = build_presets([2, 4], [1, 2])
    rows = []
    for seed in (0, 1, 2):
        for cp in (0, 1):
            # in-place b2p2 == out-of-place b4p1 == 100 scans -> perfect match
            for label, base in (("b2p1", 50.0), ("b2p2", 100.0),
                                ("b4p1", 100.0), ("b4p2", 200.0)):
                rows.append(_synth_rows(1, seed, cp, base, 0, "inplace", label))
            for label, base in (("b2p1", 50.0), ("b2p2", 100.0),
                                ("b4p1", 100.0), ("b4p2", 200.0)):
                rows.append(_synth_rows(1, seed, cp, 0, base, "outofplace", label))
    picks = select_pairs(rows, presets, within=5.0, approx=10.0)
    depth_pick = next(p for p in picks if p.bucket == "1")
    # a zero-gap pair exists; tie-break prefers the lowest combined budget
    assert depth_pick.median_gap < 1e-9
    assert depth_pick.ip.label == "b2p1"
    assert depth_pick.oop.label == "b2p1"


def test_select_pairs_ignores_rows_it_is_not_given():
    """Selection depends only on the rows passed (calibration-only contract)."""
    presets = build_presets([2], [1, 2])
    cal_rows = []
    for seed in (0, 1, 2):
        for cp in (0, 1):
            cal_rows.append(_synth_rows(1, seed, cp, 100, 0, "inplace", "b2p1"))
            cal_rows.append(_synth_rows(1, seed, cp, 200, 0, "inplace", "b2p2"))
            cal_rows.append(_synth_rows(1, seed, cp, 0, 200, "outofplace", "b2p1"))
            cal_rows.append(_synth_rows(1, seed, cp, 0, 200, "outofplace", "b2p2"))
    picks_a = select_pairs(cal_rows, presets, within=5.0, approx=10.0)
    # appending held-out (evaluation) rows is a *different* input; the pure
    # function on calibration rows is unchanged
    picks_b = select_pairs(list(cal_rows), presets, within=5.0, approx=10.0)
    assert [p.to_row() for p in picks_a] == [p.to_row() for p in picks_b]
    depth_pick = next(p for p in picks_a if p.bucket == "1")
    assert depth_pick.median_gap < 1e-9


# ---------------------------------------------------------------------------
# 7. unmatched rows stay visible
# ---------------------------------------------------------------------------

def test_build_matched_keeps_unmatched_rows():
    presets = build_presets([2, 4], [1, 2])
    rows = []
    for cp in (0, 1):
        for label, base in (("b2p1", 10.0), ("b2p2", 200.0),
                            ("b4p1", 20.0), ("b4p2", 400.0)):
            rows.append(_synth_rows(1, 0, cp, base, 0, "inplace", label))
        for label, base in (("b2p1", 500.0), ("b2p2", 500.0),
                            ("b4p1", 500.0), ("b4p2", 500.0)):
            rows.append(_synth_rows(1, 0, cp, 0, base, "outofplace", label))
    pick = select_pairs(rows, presets, within=5.0, approx=10.0)
    sel = {p.bucket: p for p in pick}
    matched = build_matched(rows, sel, within=5.0, approx=10.0,
                            source="calibration_in_sample", phase="calibration")
    assert len(matched) == 2                 # one per checkpoint, none dropped
    assert all(r["match_status"] == "unmatched" for r in matched)
    assert all(not np.isnan(r["relative_scan_gap_pct"]) for r in matched)


def test_missing_preset_pair_is_marked():
    rows = [_synth_rows(1, 0, 0, 10.0, 0, "inplace", "b2p1")]
    sel = {"1": select_pairs(
        [_synth_rows(1, 0, 0, 10.0, 0, "inplace", "b2p1"),
         _synth_rows(1, 0, 0, 0, 10.0, "outofplace", "b2p1")],
        build_presets([2], [1]), within=5.0, approx=10.0)[0]}
    matched = build_matched(rows, sel, within=5.0, approx=10.0,
                            source="calibration_in_sample", phase="calibration")
    assert matched[0]["match_status"] == "missing"
    assert matched[0]["ip_failed"] and matched[0]["oop_failed"]


# ---------------------------------------------------------------------------
# 8. summary + Pareto
# ---------------------------------------------------------------------------

def test_summary_counts_matched_approx_unmatched():
    matched = [
        {"match_status": "matched", "depth_config": 1, "split_threshold": 24,
         "delta": None, "ip_scan_per_query": 100.0, "oop_scan_per_query": 100.0,
         "recall_diff_pp": 1.0, "ip_recall_at_10": 0.8, "oop_recall_at_10": 0.79,
         "ip_q_p50_ms": 0.01, "oop_q_p50_ms": 0.01, "relative_scan_gap_pct": 0.0},
        {"match_status": "approximate", "depth_config": 1, "split_threshold": 24,
         "delta": None, "ip_scan_per_query": 108.0, "oop_scan_per_query": 100.0,
         "recall_diff_pp": 0.0, "ip_recall_at_10": 0.8, "oop_recall_at_10": 0.8,
         "ip_q_p50_ms": 0.01, "oop_q_p50_ms": 0.01,
         "relative_scan_gap_pct": relative_scan_gap(108.0, 100.0)},
    ]
    summary = build_summary(matched)
    assert len(summary) == 1
    row = summary[0]
    assert row["n_obs"] == 2
    assert row["n_matched"] == 1
    assert row["n_approx"] == 1
    assert row["n_unmatched"] == 0
    assert abs(row["recall_diff_pp_mean"] - 0.5) < 1e-9


def test_pareto_dominance():
    rows = []
    # three in-place presets at depth 1: (scan, recall)
    for label, scan, rec in (("b2p1", 50.0, 0.5), ("b2p2", 100.0, 0.7),
                             ("b2p4", 200.0, 0.6)):
        for _ in range(2):
            r = _synth_rows(1, 0, 0, scan, 0, "inplace", label, recall=rec)
            rows.append(r)
    pareto = build_pareto(rows)
    ip_depth1 = {r["search_preset"]: r for r in pareto
                 if r["strategy"] == "inplace" and r["depth_config"] == 1}
    assert ip_depth1["b2p1"]["dominated"] is False
    assert ip_depth1["b2p2"]["dominated"] is False
    assert ip_depth1["b2p4"]["dominated"] is True   # higher scan, lower recall


# ---------------------------------------------------------------------------
# 9. end-to-end CLI smoke (calibration -> evaluate) and file schemas
# ---------------------------------------------------------------------------

def _tiny_cli_args(out, phase, extra=()):
    args = [
        "--phase", phase, "--initial-size", "120", "--operations", "60",
        "--query-count", "8", "--checkpoint-interval", "20",
        "--depths", "1,2", "--split-thresholds", "24", "--deltas", "none",
        "--beam-width-values", "2,8", "--n-postings-values", "1,4",
        "--blas-threads", "1", "--out", str(out), "--overwrite",
    ]
    return args + list(extra)


def test_cli_smoke_calibration_then_evaluate(tmp_path):
    cal_dir = tmp_path / "cal"
    rc = run_experiment(resolve_spec(_cli(*_tiny_cli_args(
        cal_dir, "calibration", ["--calibration-seeds", "0"]))))
    assert rc  # returns written-file counts dict (non-empty)
    for name in ("results_budget_raw.csv", "results_budget_matched.csv",
                 "results_budget_summary.csv", "results_budget_pareto.csv",
                 "results_budget_selection.csv", "run_metadata.json"):
        assert (cal_dir / name).exists(), name

    # calibration selection pick for depth 1 must exist
    sel = read_selection(cal_dir / "results_budget_selection.csv")
    assert "1" in sel and "__global__" in sel

    ev_dir = tmp_path / "eval"
    run_experiment(resolve_spec(_cli(*_tiny_cli_args(
        ev_dir, "evaluate",
        ["--evaluation-seeds", "1", "--calibration-results", str(cal_dir)]))))
    assert (ev_dir / "results_budget_oracle.csv").exists()
    matched = list(csv.DictReader(open(ev_dir / "results_budget_matched.csv")))
    assert matched
    # selection came from calibration, not from evaluation outcomes
    assert all(r["selection_source"] == "calibration_applied" for r in matched)
    cal_pick = sel["1"]
    depth1 = [r for r in matched if r["depth_config"] == "1"]
    assert depth1
    assert all(r["ip_preset"] == cal_pick.ip.label for r in depth1)
    assert all(r["oop_preset"] == cal_pick.oop.label for r in depth1)

    meta = json.loads((ev_dir / "run_metadata.json").read_text())
    assert meta["immutability_violations"] == 0
    assert meta["matching_rule"]["matched_within_pct"] == 5.0
    assert meta["seeds"]["evaluation"] == [1]


def test_default_grid_sizes():
    spec = resolve_spec(_cli("--phase", "calibration"))
    assert len(spec.presets()) == len(BEAM_DEFAULT) * len(NPOST_DEFAULT) == 20


def test_beam_and_npost_validation():
    with pytest.raises(ValueError):
        resolve_spec(_cli("--phase", "calibration", "--beam-width-values", "0,4"))
    with pytest.raises(ValueError):
        resolve_spec(_cli("--phase", "calibration", "--n-postings-values", "0"))
    with pytest.raises(ValueError):
        resolve_spec(_cli("--phase", "calibration", "--match-within", "20",
                          "--match-approx", "10"))
