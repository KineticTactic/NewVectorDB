"""Tests for vecstudy.harness.frontier (final per-stratum frontier analysis)."""

from __future__ import annotations

import numpy as np
import pytest

from vecstudy.harness import frontier as fr


def _row(strategy, preset, beam, npost, work, recall, *, depth=1, split=48,
         delta=None, seed=1, cp=0, checksum="c1", trace="t1", ops=100,
         live=100, p50=0.01, p95=0.02, failed=False):
    return {
        "dataset": "synthetic", "dataset_checksum": checksum, "trace_hash": trace,
        "workload": "mixed", "insert_prob": 0.5,
        "depth_config": depth, "split_threshold": split, "merge_threshold": -1,
        "delta": delta, "seed": seed, "checkpoint": cp,
        "operations_processed": ops, "live_vector_count": live,
        "strategy": strategy, "search_preset": preset,
        "beam_scalar": beam, "n_postings": npost, "k": 10,
        "vectors_scanned_per_query": work, "recall_at_10": recall,
        "query_latency_p50_ms": p50, "query_latency_p95_ms": p95,
        "work_scan_sims_per_query": work, "run_failed": failed,
    }


def _stratum(depth=1, split=48, delta=None, seed=1, cp=0, ip=None, oop=None,
             **kw):
    rows = []
    for preset, beam, npost, work, rec in (ip or []):
        rows.append(_row("inplace", preset, beam, npost, work, rec, depth=depth,
                         split=split, delta=delta, seed=seed, cp=cp, **kw))
    for preset, beam, npost, work, rec in (oop or []):
        rows.append(_row("outofplace", preset, beam, npost, work, rec, depth=depth,
                         split=split, delta=delta, seed=seed, cp=cp, **kw))
    return rows


IP_A = [("b2p1", 2, 1, 10.0, 0.50), ("b2p2", 2, 2, 20.0, 0.70),
        ("b2p4", 2, 4, 30.0, 0.90), ("b4p2", 4, 2, 25.0, 0.60),
        ("b2p8", 2, 8, 30.0, 0.80)]
OOP_A = [("b2p1", 2, 1, 12.0, 0.45), ("b2p2", 2, 2, 22.0, 0.68),
         ("b2p4", 2, 4, 33.0, 0.88)]


# ---------------------------------------------------------------------------
# 1. strata are grouped separately
# ---------------------------------------------------------------------------

def test_strata_grouped_separately():
    raw = (_stratum(depth=1, seed=1, cp=0, ip=IP_A, oop=OOP_A)
           + _stratum(depth=2, seed=1, cp=0, ip=IP_A, oop=OOP_A)
           + _stratum(depth=1, seed=2, cp=0, ip=IP_A, oop=OOP_A)
           + _stratum(depth=1, seed=1, cp=1, ip=IP_A, oop=OOP_A)
           + _stratum(depth=1, seed=1, cp=0, split=192, ip=IP_A, oop=OOP_A)
           + _stratum(depth=1, seed=1, cp=0, delta=0.3, ip=IP_A, oop=OOP_A))
    groups = fr.group_strata(raw)
    assert len(groups) == 6
    ids = {r["stratum_id"] for r in fr.build_frontier_by_stratum(raw)}
    assert len(ids) == 6


# ---------------------------------------------------------------------------
# 2. budget picker never exceeds the budget
# ---------------------------------------------------------------------------

def test_picker_never_exceeds_budget():
    pts = [{"search_preset": p, "beam_scalar": b, "n_postings": n, "work": w,
            "recall": r, "q_p50": 0.01, "q_p95": 0.02}
           for p, b, n, w, r in IP_A]
    for b in np.linspace(9.0, 31.0, 40):
        pick = fr.pick_at_budget(pts, float(b))
        if pick is not None:
            assert pick["work"] <= b


def test_picker_prefers_higher_recall_then_lower_work():
    pick = fr.pick_at_budget(
        [{"search_preset": p, "beam_scalar": b, "n_postings": n, "work": w,
          "recall": r, "q_p50": 0.01, "q_p95": 0.02}
         for p, b, n, w, r in IP_A], 30.0)
    assert pick["recall"] == 0.90 and pick["search_preset"] == "b2p4"


# ---------------------------------------------------------------------------
# 3. no common interval -> explicit status, no invented match
# ---------------------------------------------------------------------------

def test_no_common_work_range_is_explicit():
    ip = [("b2p1", 2, 1, 10.0, 0.5), ("b2p4", 2, 4, 50.0, 0.9)]
    oop = [("b2p1", 2, 1, 100.0, 0.4), ("b2p4", 2, 4, 200.0, 0.9)]
    raw = _stratum(ip=ip, oop=oop)
    ov = fr.build_work_overlap(raw)
    assert ov[0]["has_common_range"] is False
    assert ov[0]["reason"] == "no_common_work_range"
    paired = fr.build_frontier_paired(raw, within=5.0, approx=10.0)
    assert all(r["match_status"] == "no_common_work_range" for r in paired)
    summary = fr.build_frontier_summary(paired, "recall_at_10")
    assert summary[0]["n_no_overlap"] >= 1
    assert summary[0]["n_matched"] == 0


# ---------------------------------------------------------------------------
# 4. matching uses measured work, not identical nominal presets
# ---------------------------------------------------------------------------

def test_matching_uses_measured_work_not_nominal():
    # two in-place presets with SAME nominal budget (beam*n_postings == 8) but
    # very different measured work; the picker must use measured work.
    ip = [("b2p4", 2, 4, 30.0, 0.90), ("b4p2", 4, 2, 20.0, 0.85)]
    oop = [("b2p2", 2, 2, 18.0, 0.78), ("b2p4", 2, 4, 26.0, 0.82)]
    raw = _stratum(ip=ip, oop=oop)
    paired = fr.build_frontier_paired(raw, within=5.0, approx=10.0)
    low = [r for r in paired if r["budget_position"] == "low"][0]
    # budget is far below 30 -> in-place must pick the measured-work point
    assert low["ip_actual_work_scans_per_query"] <= low["target_budget_scans_per_query"]
    assert low["ip_preset"] == "b4p2"


# ---------------------------------------------------------------------------
# 5. dominated points removed from frontier, retained in raw
# ---------------------------------------------------------------------------

def test_dominated_removed_from_frontier_only():
    raw = _stratum(ip=IP_A, oop=OOP_A)
    frows = fr.build_frontier_by_stratum(raw)
    ip_labels = {r["search_preset"] for r in frows if r["strategy"] == "inplace"}
    assert "b2p8" not in ip_labels          # (30,0.8) dominated by (30,0.9)
    assert "b2p4" in ip_labels
    # raw is untouched and still contains the dominated point
    assert any(r["search_preset"] == "b2p8" for r in raw)
    assert all(r["dominated"] is False for r in frows)


def test_frontier_points_helper():
    pts = [{"search_preset": p, "beam_scalar": b, "n_postings": n, "work": w,
            "recall": r, "q_p50": 0.0, "q_p95": 0.0}
           for p, b, n, w, r in IP_A]
    f = fr.frontier_points(pts)
    labels = {p["search_preset"] for p in f}
    assert "b2p8" not in labels
    works = [p["work"] for p in f]
    assert works == sorted(works)


# ---------------------------------------------------------------------------
# 6. deterministic tie-break
# ---------------------------------------------------------------------------

def test_tie_break_is_deterministic():
    pts = [{"search_preset": "b8p8", "beam_scalar": 8, "n_postings": 8, "work": 20.0,
            "recall": 0.8, "q_p50": 0.02, "q_p95": 0.0},
           {"search_preset": "b2p2", "beam_scalar": 2, "n_postings": 2, "work": 20.0,
            "recall": 0.8, "q_p50": 0.02, "q_p95": 0.0}]
    picks = {fr.pick_at_budget(pts, 25.0)["search_preset"] for _ in range(5)}
    assert picks == {"b2p2"}          # lexicographic label tie-break


def test_tie_break_prefers_lower_latency():
    pts = [{"search_preset": "b2p2", "beam_scalar": 2, "n_postings": 2, "work": 20.0,
            "recall": 0.8, "q_p50": 0.05, "q_p95": 0.0},
           {"search_preset": "b4p2", "beam_scalar": 4, "n_postings": 2, "work": 20.0,
            "recall": 0.8, "q_p50": 0.01, "q_p95": 0.0}]
    assert fr.pick_at_budget(pts, 25.0)["search_preset"] == "b4p2"


# ---------------------------------------------------------------------------
# 7. both arms scored on the same basis (validation)
# ---------------------------------------------------------------------------

def test_validation_accepts_paired_basis():
    raw = _stratum(ip=IP_A, oop=OOP_A)
    report = fr.validate_raw(raw)
    assert report["ok"] is True
    assert report["n_strata"] == 1
    assert report["n_unpaired_strata"] == 0


def test_validation_flags_basis_mismatch():
    raw = _stratum(ip=IP_A, oop=OOP_A, checksum="c1")
    for r in raw:
        if r["strategy"] == "outofplace":
            r["dataset_checksum"] = "other"
    report = fr.validate_raw(raw)
    assert report["ok"] is False
    assert report["n_basis_mismatches"] >= 1


def test_validation_flags_unpaired_stratum():
    raw = _stratum(ip=IP_A, oop=OOP_A)
    raw = [r for r in raw if r["strategy"] != "outofplace"]
    report = fr.validate_raw(raw)
    assert report["n_unpaired_strata"] == 1


# ---------------------------------------------------------------------------
# 8. paired summaries use matching keys, report unmatched counts
# ---------------------------------------------------------------------------

def test_summary_reports_unmatched_counts():
    # in-place 10..30, out-of-place 200..300 -> no overlap -> unmatched visible
    raw = _stratum(ip=[("b2p1", 2, 1, 10.0, 0.5), ("b2p4", 2, 4, 30.0, 0.9)],
                   oop=[("b2p1", 2, 1, 200.0, 0.5), ("b2p4", 2, 4, 300.0, 0.9)])
    paired = fr.build_frontier_paired(raw, within=5.0, approx=10.0)
    summary = fr.build_frontier_summary(paired, "recall_at_10")
    assert summary[0]["n_no_overlap"] == len(paired)
    assert summary[0]["n_matched"] == 0


# ---------------------------------------------------------------------------
# 9. confidence intervals resample seeds, not rows/checkpoints
# ---------------------------------------------------------------------------

def test_summary_ci_resamples_seeds_not_rows():
    paired = []
    # seed 1: many checkpoints all at diff 0; seed 2: one checkpoint at diff 10
    for cp in range(5):
        paired.append({"depth_config": 1, "split_threshold": 48, "delta": "none",
                       "budget_position": "low", "seed": 1, "checkpoint": cp,
                       "match_status": "matched", "recall_diff_pp": 0.0,
                       "ip_recall_at_10": 0.5, "oop_recall_at_10": 0.5,
                       "ip_actual_work_scans_per_query": 10.0,
                       "oop_actual_work_scans_per_query": 10.0,
                       "relative_scan_gap_pct": 0.0,
                       "target_budget_scans_per_query": 10.0})
    paired.append({"depth_config": 1, "split_threshold": 48, "delta": "none",
                   "budget_position": "low", "seed": 2, "checkpoint": 0,
                   "match_status": "matched", "recall_diff_pp": 10.0,
                   "ip_recall_at_10": 0.6, "oop_recall_at_10": 0.5,
                   "ip_actual_work_scans_per_query": 10.0,
                   "oop_actual_work_scans_per_query": 10.0,
                   "relative_scan_gap_pct": 0.0,
                   "target_budget_scans_per_query": 10.0})
    summary = fr.build_frontier_summary(paired, "recall_at_10")
    row = summary[0]
    assert row["n_seeds"] == 2
    # mean over rows would be 10/6 = 1.67; seed-level mean-of-means is 5.0
    assert row["recall_diff_pp_seedmean"] == pytest.approx(5.0)
    assert row["recall_diff_pp_mean"] == pytest.approx(10.0 / 6.0)
    assert row["recall_diff_pp_ci95_lo"] <= 5.0 <= row["recall_diff_pp_ci95_hi"]


def test_single_seed_ci_is_degenerate():
    paired = [{"depth_config": 1, "split_threshold": 48, "delta": "none",
               "budget_position": "low", "seed": 1, "checkpoint": cp,
               "match_status": "matched", "recall_diff_pp": 2.0,
               "ip_recall_at_10": 0.5, "oop_recall_at_10": 0.48,
               "ip_actual_work_scans_per_query": 10.0,
               "oop_actual_work_scans_per_query": 10.0,
               "relative_scan_gap_pct": 0.0,
               "target_budget_scans_per_query": 10.0} for cp in range(3)]
    row = fr.build_frontier_summary(paired, "recall_at_10")[0]
    assert row["recall_diff_pp_ci95_lo"] == row["recall_diff_pp_ci95_hi"] == 2.0


# ---------------------------------------------------------------------------
# 10. budget schedule + end-to-end on a file
# ---------------------------------------------------------------------------

def test_budget_schedule_positions_within_interval():
    sched = fr.budget_schedule(100.0, 200.0)
    assert sched["low"] == pytest.approx(125.0)
    assert sched["middle"] == pytest.approx(150.0)
    assert sched["high"] == pytest.approx(175.0)


def test_end_to_end_writes_all_files(tmp_path):
    import csv
    raw = (_stratum(depth=1, seed=1, cp=0, ip=IP_A, oop=OOP_A)
           + _stratum(depth=2, seed=1, cp=0, ip=IP_A, oop=OOP_A)
           + _stratum(depth=1, seed=2, cp=0, ip=IP_A, oop=OOP_A))
    raw_csv = tmp_path / "results_budget_raw.csv"
    fr.write_csv(raw_csv, raw)
    out = tmp_path / "frontier"
    written = fr.run_analysis(raw_csv, out, within=5.0, approx=10.0,
                              n_boot=200, rng_seed=1, progress=False)
    assert fr.FRONTIER_FILE in written and written[fr.FRONTIER_FILE] > 0
    assert fr.OVERLAP_FILE in written
    assert fr.PAIRED_FILE in written and written[fr.PAIRED_FILE] == 9  # 3 strata x 3
    assert (out / "run_metadata.json").exists()
    import json
    meta = json.loads((out / "run_metadata.json").read_text())
    assert meta["validation"]["ok"] is True
    assert meta["budget_rule"]["bootstrap"]["resample_unit"] == "seed/run"
    assert meta["claim_scope"]["kind"].startswith("empirical_frontier")


def test_analysis_does_not_mutate_raw():
    raw = _stratum(ip=IP_A, oop=OOP_A)
    before = [dict(r) for r in raw]
    fr.build_frontier_by_stratum(raw)
    fr.build_work_overlap(raw)
    fr.build_frontier_paired(raw, within=5.0, approx=10.0)
    assert raw == before
