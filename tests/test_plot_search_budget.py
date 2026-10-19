"""Tests for vecstudy.harness.plot_search_budget (figures/report generator).

These cover the figures-task acceptance criteria: Pareto dominance handling,
final-checkpoint-only headline data, best-recall-under-budget picking (ties and
no-eligible), paired-key grouping that cannot combine seeds/configurations,
visible missing rows, exclusion of the diagnostic calibration/matched/oracle
tables, and the CLI writing a report + figures on a tiny fixture while failing
loudly on a malformed schema.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from vecstudy.harness import plot_search_budget as psb


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

def _rec(strategy, preset, beam, npost, work, rec, *, depth=1, split=48,
         delta=0.3, seed=1, cp=100, failed=False, dataset="synthetic",
         checksum="c1", trace="t1", live=100):
    return {
        "dataset": dataset, "dataset_checksum": checksum, "trace_hash": trace,
        "workload": "mixed", "insert_prob": 0.5, "dim": 32, "initial_size": 200,
        "depth_config": depth, "split_threshold": split, "merge_threshold": -1,
        "delta": delta, "seed": seed, "checkpoint": cp,
        "operations_processed": cp, "live_vector_count": live,
        "strategy": strategy, "search_preset": preset, "beam_scalar": beam,
        "n_postings": npost, "k": 10,
        "vectors_scanned_per_query": work, "recall_at_10": rec,
        "query_latency_p50_ms": 0.01, "query_latency_p95_ms": 0.02,
        "query_count": 32, "run_failed": failed,
    }


# A small, non-dominated in-place curve and an out-of-place curve below it.
IP = [("b2p1", 2, 1, 10.0, 0.50), ("b2p2", 2, 2, 20.0, 0.70),
      ("b2p4", 2, 4, 30.0, 0.90), ("b4p2", 4, 2, 25.0, 0.60),
      ("b2p8", 2, 8, 30.0, 0.80)]
OOP = [("b2p1", 2, 1, 12.0, 0.45), ("b2p2", 2, 2, 22.0, 0.68),
       ("b2p4", 2, 4, 33.0, 0.88)]


def _points_to_df(points, **kw):
    return pd.DataFrame([_rec(s, p, b, n, w, r, **kw)
                         for s, (p, b, n, w, r) in
                         (("inplace", t) for t in points)])


def _full_df():
    rows = []
    for seed in (1, 2):
        for depth in (1, 2):
            for split in (48, 192):
                for preset, beam, npost, work, rec in IP:
                    rows.append(_rec("inplace", preset, beam, npost, work, rec,
                                     depth=depth, split=split, seed=seed))
                for preset, beam, npost, work, rec in OOP:
                    rows.append(_rec("outofplace", preset, beam, npost, work,
                                     rec, depth=depth, split=split, seed=seed))
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# 1. Pareto dominance: dominated points dropped from the frontier only
# ---------------------------------------------------------------------------

def test_dominated_point_excluded_from_frontier():
    df = _points_to_df(IP)
    pts = psb.points_from_df(df, "recall_at_10")
    front = psb.frontier_points(pts)
    labels = {p["search_preset"] for p in front}
    assert "b2p8" not in labels            # (30,0.80) dominated by (30,0.90)
    assert "b2p4" in labels
    # raw derivation still contains the dominated preset
    assert "b2p8" in set(df["search_preset"])


def test_frontier_sorted_by_increasing_work():
    df = _points_to_df(IP)
    works = [p["work"] for p in
             psb.frontier_points(psb.points_from_df(df, "recall_at_10"))]
    assert works == sorted(works)


# ---------------------------------------------------------------------------
# 2. headline data uses the final checkpoint only
# ---------------------------------------------------------------------------

def test_final_checkpoint_selection():
    early = [_rec("inplace", p, b, n, w, 0.99) for p, b, n, w, _ in IP]
    final = [_rec("inplace", p, b, n, w, r) for p, b, n, w, r in IP]
    for r in early:
        r["operations_processed"] = 50
        r["checkpoint"] = 0
    df = pd.DataFrame(early + final)
    assert psb.final_operations(df) == 100
    fr = psb.final_rows(df)
    assert set(fr["operations_processed"]) == {100}
    assert fr["recall_at_10"].max() == pytest.approx(0.90)


def test_budget_pairs_ignore_non_final_rows():
    early = [_rec("inplace", p, b, n, w, 0.99) for p, b, n, w, _ in IP]
    early += [_rec("outofplace", p, b, n, w, 0.99) for p, b, n, w, _ in OOP]
    for r in early:
        r["operations_processed"] = 50
    final = [_rec("inplace", p, b, n, w, r) for p, b, n, w, r in IP]
    final += [_rec("outofplace", p, b, n, w, r) for p, b, n, w, r in OOP]
    df = pd.DataFrame(early + final)
    pairs = psb.build_budget_pairs(psb.final_rows(df), "recall_at_10",
                                   (25.0,))
    elig = [r for r in pairs if r["eligible"]]
    assert elig, "expected an eligible pair at budget 25"
    # recall values must come from the final rows, never the 0.99 decoys
    assert max(r["ip_recall"] for r in elig) < 0.99


# ---------------------------------------------------------------------------
# 3. best-recall-under-budget: ties and no eligible preset
# ---------------------------------------------------------------------------

def _pts(rows):
    return [{"search_preset": p, "beam_scalar": b, "n_postings": n, "work": w,
             "recall": r, "q_p50": q, "q_p95": 0.0}
            for p, b, n, w, r, q in rows]


def test_picker_none_when_budget_too_small():
    pts = _pts([(p, b, n, w, r, 0.01) for p, b, n, w, r in IP])
    assert psb.best_recall_under_budget(pts, 5.0) is None


def test_picker_never_exceeds_budget_and_maximizes_recall():
    pts = _pts([(p, b, n, w, r, 0.01) for p, b, n, w, r in IP])
    pick = psb.best_recall_under_budget(pts, 30.0)
    assert pick["work"] <= 30.0 and pick["recall"] == pytest.approx(0.90)


def test_picker_tie_prefers_lower_work_then_label():
    pts = _pts([("b8p8", 8, 8, 20.0, 0.8, 0.01),
                ("b2p2", 2, 2, 20.0, 0.8, 0.01),
                ("b1p1", 1, 1, 25.0, 0.8, 0.01)])
    # same work + same recall -> lexicographically smallest label wins
    assert psb.best_recall_under_budget(pts, 20.0)["search_preset"] == "b2p2"
    # a larger cap still prefers the lower-work tie over b1p1's higher work
    assert psb.best_recall_under_budget(pts, 30.0)["search_preset"] == "b2p2"


# ---------------------------------------------------------------------------
# 4. paired keys: seeds and configurations must never be combined
# ---------------------------------------------------------------------------

def test_pairs_are_split_per_seed_and_config():
    df = _full_df()
    pairs = psb.build_budget_pairs(df, "recall_at_10", (25.0, 35.0))
    # 2 seeds x 2 depths x 2 splits x 2 budgets
    assert len(pairs) == 2 * 2 * 2 * 2
    combos = {(r["seed"], r["depth_config"], r["split_threshold"], r["budget"])
              for r in pairs}
    assert len(combos) == len(pairs)
    for r in pairs:
        assert isinstance(r["seed"], (int, np.integer))


def test_pairs_do_not_combine_seeds_with_different_checksums():
    a = _full_df()
    b = _full_df()
    b["dataset_checksum"] = "other"
    df = pd.concat([a, b], ignore_index=True)
    pairs = psb.build_budget_pairs(df, "recall_at_10", (25.0,))
    assert len(pairs) == 2 * 2 * 2 * 1 * 2     # extra factor of 2 for checksum


# ---------------------------------------------------------------------------
# 5. missing rows stay visible (ineligible), never silently dropped
# ---------------------------------------------------------------------------

def test_groups_missing_a_strategy_are_ineligible_not_dropped():
    rows = [_rec("inplace", p, b, n, w, r) for p, b, n, w, r in IP]
    rows += [_rec("outofplace", p, b, n, w, r) for p, b, n, w, r in OOP]
    # add a config/seed that only has the in-place arm
    for p, b, n, w, r in IP:
        rows.append(_rec("inplace", p, b, n, w, r, seed=99))
    df = pd.DataFrame(rows)
    pairs = psb.build_budget_pairs(df, "recall_at_10", (25.0,))
    orphan = [r for r in pairs if r["seed"] == 99]
    assert orphan, "orphan group must still produce rows"
    assert all(r["eligible"] is False for r in orphan)
    assert all(np.isnan(r["delta_recall_pp"]) for r in orphan)


def test_analysis_summary_reports_ineligible_counts():
    rows = [_rec("inplace", p, b, n, w, r) for p, b, n, w, r in IP]
    rows += [_rec("outofplace", p, b, n, w, r) for p, b, n, w, r in OOP]
    for p, b, n, w, r in IP:
        rows.append(_rec("inplace", p, b, n, w, r, seed=99))
    pairs = psb.build_budget_pairs(pd.DataFrame(rows), "recall_at_10", (25.0,))
    summary = psb.summarize_budget_pairs(pairs, n_boot=100, rng_seed=1)
    assert sum(s["ineligible_pairs"] for s in summary) >= 1


# ---------------------------------------------------------------------------
# 6. diagnostic tables (matched/oracle) never enter headline numbers
# ---------------------------------------------------------------------------

def _write_run_dir(root: Path, *, with_diagnostics: bool):
    root.mkdir(parents=True, exist_ok=True)
    df = _full_df()
    df.to_csv(root / "results_budget_raw.csv", index=False)
    if with_diagnostics:
        # decoy oracle with absurd recall that would move any summary if used
        oracle = df[df["strategy"] == "inplace"].copy()
        oracle["recall_at_10"] = 5.0
        oracle.to_csv(root / "results_budget_oracle.csv", index=False)
        # decoy matched pairing that points at a different, wrong preset
        matched = df.copy()
        matched["match_status"] = "matched"
        matched["recall_at_10"] = -1.0
        matched.to_csv(root / "results_budget_matched.csv", index=False)
        sel = pd.DataFrame([{"depth_config": 1, "search_preset": "ghost"}])
        sel.to_csv(root / "results_budget_selection.csv", index=False)


def test_oracle_and_matched_tables_excluded_from_headline(tmp_path):
    clean = tmp_path / "clean"
    dirty = tmp_path / "dirty"
    _write_run_dir(clean, with_diagnostics=False)
    _write_run_dir(dirty, with_diagnostics=True)
    psb.run(clean, budgets=(25.0, 35.0), make_figures=False, progress=False)
    psb.run(dirty, budgets=(25.0, 35.0), make_figures=False, progress=False)
    a = (clean / "analysis_summary.csv").read_text()
    b = (dirty / "analysis_summary.csv").read_text()
    assert a == b
    # decoy oracle recalled 5.0 and decoy matched -1.0; neither may leak in
    summary = pd.read_csv(clean / "analysis_summary.csv")
    assert summary["ip_recall_mean"].max() <= 1.0
    assert summary["oop_recall_mean"].max() <= 1.0


# ---------------------------------------------------------------------------
# 7. bootstrap interval resamples seeds, not rows
# ---------------------------------------------------------------------------

def test_summary_bootstrap_resamples_seeds():
    pairs = []
    for _ in range(5):
        pairs.append({"seed": 1, "depth_config": 1, "split_threshold": 48,
                      "delta": "0.3", "budget": 25.0, "eligible": True,
                      "delta_recall_pp": 0.0, "ip_recall": 0.5,
                      "oop_recall": 0.5, "ip_scan": 20.0, "oop_scan": 20.0,
                      "scan_gap_pct": 0.0})
    pairs.append({"seed": 2, "depth_config": 1, "split_threshold": 48,
                  "delta": "0.3", "budget": 25.0, "eligible": True,
                  "delta_recall_pp": 10.0, "ip_recall": 0.6, "oop_recall": 0.5,
                  "ip_scan": 20.0, "oop_scan": 20.0, "scan_gap_pct": 0.0})
    summary = psb.summarize_budget_pairs(pairs, n_boot=500, rng_seed=1)
    assert summary[0]["seed_count"] == 2
    # row-mean would be 10/6; seed-level mean-of-means is 5
    assert summary[0]["delta_recall_pp_seedmean"] == pytest.approx(5.0)
    assert summary[0]["delta_recall_pp_mean"] == pytest.approx(10.0 / 6.0)
    assert summary[0]["ci95_lo"] <= 5.0 <= summary[0]["ci95_hi"]


# ---------------------------------------------------------------------------
# 8. maintenance figure skipped (not fabricated) without a wall-time column
# ---------------------------------------------------------------------------

def test_fig3_skipped_without_maintenance_column(tmp_path):
    root = tmp_path / "run"
    _write_run_dir(root, with_diagnostics=False)
    assert psb.maintenance_wall_column(pd.read_csv(
        root / "results_budget_raw.csv")) is None
    result = psb.run(root, budgets=(25.0, 35.0), make_figures=False,
                     progress=False)
    assert result["maintenance_column"] is None
    assert "fig03" not in result["figures"]


def test_fig3_data_uses_named_wall_column():
    df = _full_df()
    df["maintenance_wall_ms"] = 1.0
    assert psb.maintenance_wall_column(df) == "maintenance_wall_ms"


# ---------------------------------------------------------------------------
# 9. CLI: writes fixtures/report, and fails loudly on a bad schema
# ---------------------------------------------------------------------------

def test_cli_writes_report_and_figures(tmp_path):
    root = tmp_path / "run"
    _write_run_dir(root, with_diagnostics=False)
    rc = psb.main(["--input", str(root), "--budgets", "25,35",
                   "--bootstrap", "200"])
    assert rc == 0
    fig_dir = root / "figures"
    assert (fig_dir / "research_figures_report.md").exists()
    assert (root / "analysis_summary.csv").exists()
    assert (fig_dir / "figure_data" / "fig01_recall_work_frontier.csv").exists()
    assert (fig_dir / "figure_data" / "fig02_recall_gain_by_work_budget.csv").exists()
    for stem in ("fig01_recall_work_frontier", "fig02_recall_gain_by_work_budget"):
        for ext in ("png", "pdf", "svg"):
            assert (fig_dir / f"{stem}.{ext}").exists()
    assert not (fig_dir / "fig03_maintenance_cost.png").exists()


def test_cli_fails_on_missing_columns(tmp_path):
    root = tmp_path / "bad"
    root.mkdir()
    _full_df().drop(columns=["vectors_scanned_per_query"]).to_csv(
        root / "results_budget_raw.csv", index=False)
    with pytest.raises(ValueError, match="schema validation failed"):
        psb.run(root, make_figures=False, progress=False)


def test_cli_missing_input_dir_raises():
    with pytest.raises(FileNotFoundError):
        psb.load_run(Path("/nonexistent/does/not/exist"))


def test_cli_rejects_empty_budgets(tmp_path):
    root = tmp_path / "run"
    _write_run_dir(root, with_diagnostics=False)
    with pytest.raises(SystemExit):
        psb.main(["--input", str(root), "--budgets", ""])
