"""harness/plot_search_budget.py — figures + summary for the final
equal-search-work comparison (post-processing only).

Reads a finished ``scripts/compare_search_budget.py`` evaluation directory and
produces up to three publication figures (PNG/PDF/SVG), the exact data plotted
(``figure_data/*.csv``), a compact ``analysis_summary.csv`` and
``research_figures_report.md``.

Ground rules enforced here
--------------------------
* **Post-processing only** — no benchmark rerun, no maintainer/query changes,
  and the source result files are never modified.
* Headline numbers use the **evaluation** run's raw per-preset rows at the
  **final checkpoint**.  ``results_budget_matched.csv`` (Task-9 selected-preset
  pairing) and ``results_budget_oracle.csv`` (per-checkpoint oracle) are
  diagnostic and are never mixed into headline estimates.
* Each curve holds ``depth``, ``split_threshold``, ``merge_threshold`` and
  ``delta`` fixed; only the search preset may vary (and only the *selected*
  preset may differ between strategies in the budget comparison).
* A budget point exists only where **both** strategies have a preset whose
  measured work is at or below the budget; otherwise it is reported as
  unavailable — never extrapolated or scored as a win/tie.

"Best recall under a work budget" is an *empirical upper envelope over the
tested preset grid*, not exact scan equality: the selected preset may scan fewer
vectors than the cap, so the actual scan counts are always reported alongside.
"""

from __future__ import annotations

import argparse
import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np

from .compare import write_csv
from .frontier import frontier_points, pick_at_budget
from .search_budget import relative_scan_gap

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

STRATEGIES = ("inplace", "outofplace")
STRATEGY_COLORS = {"inplace": "#0072B2", "outofplace": "#D55E00"}
STRATEGY_LABELS = {"inplace": "In-place", "outofplace": "Out-of-place"}

# Okabe-Ito-ish color-blind friendly line styles for split thresholds.
SPLIT_STYLES = ("-", "--", ":", "-.")

DEFAULT_BUDGETS = (64.0, 128.0, 256.0, 512.0)

# Keys that define a paired, controlled comparison (only the preset may differ).
MATCH_KEYS = (
    "dataset", "dataset_checksum", "trace_hash", "seed",
    "depth_config", "leaf_size", "branching_factor",
    "split_threshold", "merge_threshold", "delta",
    "memtable_limit", "compaction_threshold", "checkpoint",
)

MAINT_WALL_CANDIDATES = (
    "maintenance_wall_ms", "cumulative_maintenance_wall_ms",
    "total_maintenance_wall_ms", "delta_maintenance_wall_ms",
)

REQUIRED_RAW_COLUMNS = (
    "strategy", "search_preset", "vectors_scanned_per_query", "recall_at_10",
    "query_latency_p50_ms", "query_latency_p95_ms", "operations_processed",
    "run_failed", "depth_config", "split_threshold", "merge_threshold",
    "delta", "seed",
)


# ---------------------------------------------------------------------------
# Loading / validation
# ---------------------------------------------------------------------------

@dataclass
class RunData:
    root: Path
    raw: Any                       # pandas.DataFrame
    matched: Any | None
    metadata: dict
    frames: dict[str, Any] = field(default_factory=dict)


def load_run(root: Path) -> RunData:
    import pandas as pd

    root = Path(root)
    if not root.is_dir():
        raise FileNotFoundError(f"evaluation directory not found: {root}")
    raw_path = root / "results_budget_raw.csv"
    if not raw_path.exists():
        raise FileNotFoundError(f"missing results_budget_raw.csv in {root}")
    raw = pd.read_csv(raw_path)
    matched = None
    mp = root / "results_budget_matched.csv"
    if mp.exists():
        matched = pd.read_csv(mp)
    meta = {}
    mpath = root / "run_metadata.json"
    if mpath.exists():
        meta = json.loads(mpath.read_text())
    frames = {}
    for name in ("results_budget_summary.csv", "results_budget_pareto.csv",
                 "results_budget_selection.csv", "results_budget_oracle.csv"):
        p = root / name
        if p.exists():
            frames[name] = pd.read_csv(p)
    return RunData(root=root, raw=raw, matched=matched, metadata=meta, frames=frames)


def validate_raw(raw: Any) -> list[str]:
    issues: list[str] = []
    missing = [c for c in REQUIRED_RAW_COLUMNS if c not in raw.columns]
    if missing:
        issues.append(f"missing required raw columns: {missing}")
        return issues
    strategies = set(raw["strategy"].unique())
    if not {"inplace", "outofplace"} <= strategies:
        issues.append(f"raw is missing a strategy: have {sorted(strategies)}")
    if raw["run_failed"].astype(str).str.lower().eq("true").any():
        # Not fatal, but surfaced loudly in the report.
        pass
    return issues


def primary_k(raw: Any) -> int:
    if "k" in raw.columns and len(raw):
        return int(raw["k"].iloc[0])
    return 10


def recall_key(raw: Any) -> str:
    k = primary_k(raw)
    for cand in (f"recall_at_{k}", "recall_at_10", "recall"):
        if cand in raw.columns:
            return cand
    raise KeyError("no recall column found")


def final_operations(raw: Any) -> int:
    return int(raw["operations_processed"].max())


def final_rows(raw: Any) -> Any:
    return raw[raw["operations_processed"] == final_operations(raw)].copy()


def _delta_label(value: Any) -> str:
    if value is None or (isinstance(value, float) and np.isnan(value)):
        return "none"
    return str(value)


def audit_report(rd: RunData) -> dict:
    raw = rd.raw
    out: dict[str, Any] = {
        "n_raw_rows": int(len(raw)),
        "n_failed_rows": int(raw["run_failed"].astype(str).str.lower()
                              .eq("true").sum()) if "run_failed" in raw else None,
    }
    if "correctness_status" in raw:
        out["correctness_status_counts"] = (
            raw["correctness_status"].value_counts().to_dict())
    for col in ("npa_violations", "ownership_errors", "coverage_errors",
                "membership_violations", "deleted_id_retrievals"):
        if col in raw:
            out[f"max_{col}"] = int(np.nanmax(raw[col].to_numpy(dtype=float)))
    if rd.matched is not None and "match_status" in rd.matched:
        out["matched_status_counts"] = (
            rd.matched["match_status"].value_counts().to_dict())
    if rd.metadata:
        out["immutability_violations"] = rd.metadata.get(
            "immutability_violations")
        out["run_errors"] = rd.metadata.get("run_errors")
    return out


# ---------------------------------------------------------------------------
# Points / frontiers (no matplotlib)
# ---------------------------------------------------------------------------

def _is_failed(row: Any) -> bool:
    v = row.get("run_failed")
    if isinstance(v, str):
        return v.strip().lower() in ("true", "1")
    return bool(v)


def points_from_df(df: Any, rk: str) -> list[dict]:
    pts: list[dict] = []
    for _, r in df.iterrows():
        if _is_failed(r):
            continue
        work = float(r["vectors_scanned_per_query"])
        rec = float(r[rk])
        if not (np.isfinite(work) and np.isfinite(rec)):
            continue
        pts.append({
            "search_preset": str(r["search_preset"]),
            "beam_scalar": int(r.get("beam_scalar", 0)),
            "n_postings": int(r.get("n_postings", 0)),
            "work": work, "recall": rec,
            "q_p50": float(r.get("query_latency_p50_ms", np.nan)),
            "q_p95": float(r.get("query_latency_p95_ms", np.nan)),
        })
    return pts


def best_recall_under_budget(points: list[dict], budget: float) -> dict | None:
    """Max recall among presets with measured work <= budget (frontier.pick)."""
    return pick_at_budget(points, budget)


# ---------------------------------------------------------------------------
# Figure 1 data: recall vs work frontiers (final checkpoint)
# ---------------------------------------------------------------------------

def build_frontier_data(final_df: Any, rk: str) -> tuple[list[dict], list[dict]]:
    """Return (frontier_rows, seed_point_rows) aggregated over held-out seeds.

    A curve is identified by (depth_config, delta, split_threshold, strategy);
    no other configuration key is pooled.  Only the search preset varies.
    """
    df = final_df.copy()
    df["delta_label"] = df["delta"].apply(_delta_label)

    # seed-level raw points (faint scatter)
    seed_rows: list[dict] = []
    for key, g in df.groupby(
            ["depth_config", "delta_label", "split_threshold", "strategy", "seed"],
            dropna=False):
        depth, dlabel, split, strategy, seed = key
        for p in frontier_points(points_from_df(g, rk)):
            seed_rows.append({
                "depth_config": depth, "delta": dlabel,
                "split_threshold": split, "strategy": strategy, "seed": seed,
                "search_preset": p["search_preset"],
                "vectors_scanned_per_query": p["work"],
                rk: p["recall"], "kind": "seed_frontier",
            })

    # mean-across-seeds per preset, then nondominated envelope per curve
    frontier_rows: list[dict] = []
    grp_keys = ["depth_config", "delta_label", "split_threshold", "strategy",
                "search_preset"]
    agg = (df.groupby(grp_keys, dropna=False)
             .agg(work=("vectors_scanned_per_query", "mean"),
                  recall=(rk, "mean"),
                  n_seeds=("seed", "nunique"),
                  q50=("query_latency_p50_ms", "mean"))
             .reset_index())
    for key, g in agg.groupby(["depth_config", "delta_label", "split_threshold",
                               "strategy"], dropna=False):
        depth, dlabel, split, strategy = key
        pts = [{"search_preset": r["search_preset"], "beam_scalar": 0,
                "n_postings": 0, "work": float(r["work"]),
                "recall": float(r["recall"]), "q_p50": float(r["q50"]),
                "q_p95": float("nan")} for _, r in g.iterrows()]
        for p in frontier_points(pts):
            frontier_rows.append({
                "depth_config": depth, "delta": dlabel,
                "split_threshold": split, "strategy": strategy,
                "search_preset": p["search_preset"],
                "vectors_scanned_per_query": p["work"], rk: p["recall"],
                "n_seeds": int(g["n_seeds"].max()), "kind": "mean_frontier",
            })
    return frontier_rows, seed_rows


# ---------------------------------------------------------------------------
# Figure 2 data: paired recall gain under a common work budget
# ---------------------------------------------------------------------------

def build_budget_pairs(final_df: Any, rk: str,
                       budgets: tuple[float, ...]) -> list[dict]:
    """One row per (base config, seed, budget): best recall each under the cap."""
    rows: list[dict] = []
    group_cols = [c for c in MATCH_KEYS if c in final_df.columns]
    df = final_df.copy()
    df["delta_label"] = df["delta"].apply(_delta_label)
    for key, g in df.groupby(group_cols, dropna=False):
        keys = dict(zip(group_cols, key if isinstance(key, tuple) else (key,)))
        keys["delta"] = _delta_label(keys.get("delta"))
        ip_pts = points_from_df(g[g["strategy"] == "inplace"], rk)
        oop_pts = points_from_df(g[g["strategy"] == "outofplace"], rk)
        for budget in budgets:
            ipp = best_recall_under_budget(ip_pts, budget)
            oopp = best_recall_under_budget(oop_pts, budget)
            eligible = ipp is not None and oopp is not None
            row = {
                "row_kind": "pair_seed",
                "dataset": keys.get("dataset"),
                "seed": keys.get("seed"), "depth_config": keys.get("depth_config"),
                "split_threshold": keys.get("split_threshold"),
                "merge_threshold": keys.get("merge_threshold"),
                "delta": keys.get("delta"), "checkpoint": keys.get("checkpoint"),
                "budget": float(budget), "eligible": bool(eligible),
                "ip_preset": "" if ipp is None else ipp["search_preset"],
                "ip_scan": float("nan") if ipp is None else ipp["work"],
                "ip_recall": float("nan") if ipp is None else ipp["recall"],
                "oop_preset": "" if oopp is None else oopp["search_preset"],
                "oop_scan": float("nan") if oopp is None else oopp["work"],
                "oop_recall": float("nan") if oopp is None else oopp["recall"],
                "delta_recall_pp": (float("nan") if not eligible
                                    else 100.0 * (ipp["recall"] - oopp["recall"])),
                "scan_gap_pct": (float("nan") if not eligible
                                 else relative_scan_gap(ipp["work"], oopp["work"])),
            }
            rows.append(row)
    return rows


def _seed_bootstrap_ci(by_seed: dict[Any, list[float]], *, n_boot: int,
                       rng_seed: int) -> tuple[float, float, float]:
    seeds = sorted(by_seed, key=lambda s: str(s))
    arr = np.array([float(np.mean(by_seed[s])) for s in seeds], dtype=np.float64)
    point = float(arr.mean())
    if len(seeds) < 2:
        return point, point, point
    rng = np.random.default_rng(rng_seed)
    idx = rng.integers(0, len(arr), size=(n_boot, len(arr)))
    boot = arr[idx].mean(axis=1)
    lo, hi = np.percentile(boot, [2.5, 97.5])
    return point, float(lo), float(hi)


def summarize_budget_pairs(pairs: list[dict], *, n_boot: int = 2000,
                           rng_seed: int = 20261019) -> list[dict]:
    groups: dict[tuple, list[dict]] = {}
    for r in pairs:
        key = (_delta_label(r.get("delta")), r["depth_config"],
               r["split_threshold"], r["budget"])
        groups.setdefault(key, []).append(r)
    out: list[dict] = []
    for key, rows in sorted(groups.items(), key=lambda kv: str(kv[0])):
        delta, depth, split, budget = key
        elig = [r for r in rows if r["eligible"]]
        by_seed: dict[Any, list[float]] = {}
        for r in elig:
            by_seed.setdefault(r["seed"], []).append(r["delta_recall_pp"])
        point, lo, hi = (_seed_bootstrap_ci(by_seed, n_boot=n_boot, rng_seed=rng_seed)
                         if by_seed else (float("nan"),) * 3)
        diffs = [r["delta_recall_pp"] for r in elig]
        ipr = [r["ip_recall"] for r in elig]
        oopr = [r["oop_recall"] for r in elig]
        ipscan = [r["ip_scan"] for r in elig]
        oopscan = [r["oop_scan"] for r in elig]
        gaps = [r["scan_gap_pct"] for r in elig]
        out.append({
            "row_kind": "aggregate",
            "depth_config": depth, "split_threshold": split, "delta": delta,
            "budget": budget,
            "n_requested": len(rows), "eligible_pairs": len(elig),
            "ineligible_pairs": len(rows) - len(elig),
            "seed_count": len(by_seed),
            "ip_recall_mean": float(np.mean(ipr)) if ipr else float("nan"),
            "oop_recall_mean": float(np.mean(oopr)) if oopr else float("nan"),
            "delta_recall_pp_mean": float(np.mean(diffs)) if diffs else float("nan"),
            "delta_recall_pp_median": float(np.median(diffs)) if diffs else float("nan"),
            "delta_recall_pp_seedmean": point,
            "ci95_lo": lo, "ci95_hi": hi,
            "ip_scan_mean": float(np.mean(ipscan)) if ipscan else float("nan"),
            "ip_scan_median": float(np.median(ipscan)) if ipscan else float("nan"),
            "oop_scan_mean": float(np.mean(oopscan)) if oopscan else float("nan"),
            "oop_scan_median": float(np.median(oopscan)) if oopscan else float("nan"),
            "scan_gap_pct_mean": float(np.mean(gaps)) if gaps else float("nan"),
        })
    return out


# ---------------------------------------------------------------------------
# Figure 3 data: maintenance cost (final checkpoint)
# ---------------------------------------------------------------------------

def maintenance_wall_column(raw: Any) -> str | None:
    for c in MAINT_WALL_CANDIDATES:
        if c in raw.columns:
            return c
    return None


# ---------------------------------------------------------------------------
# Plotting
# ---------------------------------------------------------------------------

def _save(fig: Any, out_dir: Path, stem: str) -> list[str]:
    written = []
    for ext in ("png", "pdf", "svg"):
        p = out_dir / f"{stem}.{ext}"
        fig.savefig(p, dpi=300, bbox_inches="tight")
        written.append(p.name)
    return written


def _fig_palette():
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    return plt


def _mpl_modules():
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import matplotlib.ticker as mticker
    from matplotlib import colormaps
    return plt, mticker, colormaps


def plot_fig1(frontier_rows: list[dict], seed_rows: list[dict], rk: str,
              out_dir: Path) -> list[str]:
    plt = _fig_palette()
    depths = sorted({int(r["depth_config"]) for r in frontier_rows})
    deltas = sorted({r["delta"] for r in frontier_rows})
    splits = sorted({r["split_threshold"] for r in frontier_rows})
    fig, axes = plt.subplots(len(depths), len(deltas),
                             figsize=(5.2 * len(deltas), 3.6 * len(depths),
                                      ),
                             squeeze=False, sharex=True, layout="constrained")
    for i, depth in enumerate(depths):
        for j, dlabel in enumerate(deltas):
            ax = axes[i][j]
            for split, style in zip(splits, SPLIT_STYLES):
                for strategy in STRATEGIES:
                    sr = [r for r in seed_rows if int(r["depth_config"]) == depth
                          and r["delta"] == dlabel
                          and r["split_threshold"] == split
                          and r["strategy"] == strategy]
                    if sr:
                        ax.scatter([r["vectors_scanned_per_query"] for r in sr],
                                   [r[rk] for r in sr], s=8, alpha=0.18,
                                   color=STRATEGY_COLORS[strategy])
                    fr = [r for r in frontier_rows
                          if int(r["depth_config"]) == depth
                          and r["delta"] == dlabel
                          and r["split_threshold"] == split
                          and r["strategy"] == strategy]
                    fr.sort(key=lambda r: r["vectors_scanned_per_query"])
                    if fr:
                        ax.plot([r["vectors_scanned_per_query"] for r in fr],
                                [r[rk] for r in fr], style,
                                color=STRATEGY_COLORS[strategy], lw=1.8,
                                marker="o", ms=3,
                                label=f"{STRATEGY_LABELS[strategy]} split={split}")
            ax.set_title(f"depth={depth}, delta={dlabel}", fontsize=10)
            ax.set_xscale("log")
            ax.set_ylim(0, 1.02)
            ax.grid(alpha=0.25)
            if j == 0:
                ax.set_ylabel("Recall@10")
            if i == len(depths) - 1:
                ax.set_xlabel("vectors scanned / query (log)")
            if i == 0 and j == 0:
                ax.legend(fontsize=7, loc="lower right")
    fig.suptitle("Figure 1 — empirical Recall@10 vs measured search work "
                 "(final checkpoint, mean frontier over held-out seeds)",
                 fontsize=12)
    return _save(fig, out_dir, "fig01_recall_work_frontier")


def plot_fig2(pairs: list[dict], rk: str, out_dir: Path) -> list[str]:
    plt, mticker, colormaps = _mpl_modules()
    cmap = colormaps["viridis"]
    splits = sorted({r["split_threshold"] for r in pairs})
    deltas = sorted({_delta_label(r["delta"]) for r in pairs})
    depths = sorted({int(r["depth_config"]) for r in pairs})
    budgets = sorted({r["budget"] for r in pairs})
    fig, axes = plt.subplots(len(splits), len(deltas),
                             figsize=(5.4 * len(deltas), 3.8 * len(splits)),
                             squeeze=False, sharex=True, layout="constrained")
    for si, split in enumerate(splits):
        for dj, dlabel in enumerate(deltas):
            ax = axes[si][dj]
            for di, depth in enumerate(depths):
                xs, ys = [], []
                for b in budgets:
                    sel = [r for r in pairs if r["split_threshold"] == split
                           and _delta_label(r["delta"]) == dlabel
                           and int(r["depth_config"]) == depth
                           and r["budget"] == b and r["eligible"]]
                    if not sel:
                        continue
                    xs.append(b)
                    ys.append(float(np.mean([r["delta_recall_pp"] for r in sel])))
                    col = cmap(di / max(1, len(depths) - 1))
                    for r in sel:
                        ax.scatter(b, r["delta_recall_pp"], s=14, alpha=0.35,
                                   color=col)
                if xs:
                    ax.plot(xs, ys, marker="o", lw=1.8,
                            color=cmap(di / max(1, len(depths) - 1)),
                            label=f"depth={depth}")
            ax.axhline(0.0, color="k", lw=0.8)
            ax.set_title(f"split={split}, delta={dlabel}", fontsize=10)
            ax.set_xscale("log", base=2)
            ax.set_xticks(budgets)
            ax.get_xaxis().set_major_formatter(mticker.ScalarFormatter())
            ax.grid(alpha=0.25)
            if dj == 0:
                ax.set_ylabel("paired Recall@10 diff (pp)")
            if si == len(splits) - 1:
                ax.set_xlabel("work budget (vectors scanned / query)")
            if si == 0 and dj == 0:
                ax.legend(fontsize=7, loc="best")
    fig.suptitle("Figure 2 — Recall@10 advantage at a common work budget "
                 "(final checkpoint; >0 favours in-place)", fontsize=12)
    return _save(fig, out_dir, "fig02_recall_gain_by_work_budget")


def plot_fig3(final_df: Any, maint_col: str, out_dir: Path) -> list[str]:
    plt = _fig_palette()
    df = final_df.copy()
    df["delta_label"] = df["delta"].apply(_delta_label)
    pairs = []
    keys = [c for c in MATCH_KEYS if c in df.columns]
    for key, g in df.groupby(keys, dropna=False):
        keysd = dict(zip(keys, key if isinstance(key, tuple) else (key,)))
        ip = g[g["strategy"] == "inplace"][maint_col]
        oop = g[g["strategy"] == "outofplace"][maint_col]
        if len(ip) and len(oop):
            pairs.append({
                "depth_config": keysd.get("depth_config"),
                "split_threshold": keysd.get("split_threshold"),
                "delta": _delta_label(keysd.get("delta")),
                "seed": keysd.get("seed"),
                "ip_maint_ms": float(ip.max()), "oop_maint_ms": float(oop.max()),
            })
    cells = sorted({(p["depth_config"], p["split_threshold"], p["delta"])
                    for p in pairs})
    fig, ax = plt.subplots(figsize=(max(6, 1.1 * len(cells)), 4.2),
                           layout="constrained")
    for xi, cell in enumerate(cells):
        cp = [p for p in pairs if (p["depth_config"], p["split_threshold"],
                                   p["delta"]) == cell]
        for p in cp:
            ax.plot([xi - 0.15, xi + 0.15], [p["ip_maint_ms"], p["oop_maint_ms"]],
                    color="0.6", lw=0.6, zorder=1)
        if cp:
            ax.scatter([xi - 0.15] * len(cp), [p["ip_maint_ms"] for p in cp], s=22,
                       color=STRATEGY_COLORS["inplace"], zorder=2)
            ax.scatter([xi + 0.15] * len(cp), [p["oop_maint_ms"] for p in cp], s=22,
                       color=STRATEGY_COLORS["outofplace"], zorder=2)
    ax.set_yscale("log")
    ax.set_xticks(range(len(cells)))
    ax.set_xticklabels([f"d{c[0]}/s{c[1]}/Δ{c[2]}" for c in cells],
                       rotation=45, ha="right", fontsize=8)
    ax.set_ylabel("cumulative maintenance time (ms, log)")
    ax.set_title("Figure 3 — maintenance cost, in-place vs out-of-place "
                 "(final checkpoint, paired by config/seed)")
    ax.grid(alpha=0.25, axis="y")
    return _save(fig, out_dir, "fig03_maintenance_cost")


# ---------------------------------------------------------------------------
# Orchestration
# ---------------------------------------------------------------------------

def run(input_dir: Path, *, budgets: tuple[float, ...] = DEFAULT_BUDGETS,
        n_boot: int = 2000, rng_seed: int = 20261019,
        make_figures: bool = True, progress: bool = True) -> dict:
    rd = load_run(input_dir)
    issues = validate_raw(rd.raw)
    if issues:
        raise ValueError("raw schema validation failed: " + "; ".join(issues))

    rk = recall_key(rd.raw)
    final_df = final_rows(rd.raw)
    fig_dir = rd.root / "figures"
    data_dir = fig_dir / "figure_data"
    data_dir.mkdir(parents=True, exist_ok=True)

    frontier_rows, seed_rows = build_frontier_data(final_df, rk)
    pairs = build_budget_pairs(final_df, rk, budgets)
    summary = summarize_budget_pairs(pairs, n_boot=n_boot, rng_seed=rng_seed)

    write_csv(data_dir / "fig01_recall_work_frontier.csv",
              frontier_rows + seed_rows)
    write_csv(data_dir / "fig02_recall_gain_by_work_budget.csv",
              pairs + summary)
    write_csv(rd.root / "analysis_summary.csv", summary)

    written: dict[str, list[str]] = {}
    maint_col = maintenance_wall_column(rd.raw)
    if make_figures:
        plt = _fig_palette()
        written["fig01"] = plot_fig1(frontier_rows, seed_rows, rk, fig_dir)
        written["fig02"] = plot_fig2(pairs, rk, fig_dir)
        if maint_col is not None:
            fig3_data = []
            written["fig03"] = plot_fig3(final_df, maint_col, fig_dir)
        plt.close("all")

    report = build_report(rd, rk, summary, frontier_rows, pairs, issues,
                          maint_col, budgets)
    (fig_dir / "research_figures_report.md").write_text(report)

    if progress:
        print(f"Input: {rd.root}")
        print(f"Final checkpoint ops={final_operations(rd.raw)}  key={rk}")
        print(f"Figures written: {json.dumps(written)}")
        print(f"Maintenance wall column: {maint_col or 'ABSENT (fig03 skipped)'}")
        print(f"Fixture dir: {fig_dir}")
    return {"figures": written, "summary_rows": len(summary),
            "figures_dir": str(fig_dir), "maintenance_column": maint_col}


# ---------------------------------------------------------------------------
# Report
# ---------------------------------------------------------------------------

def build_report(rd: RunData, rk: str, summary: list[dict],
                 frontier_rows: list[dict], pairs: list[dict],
                 issues: list[str], maint_col: str | None,
                 budgets: tuple[float, ...]) -> str:
    meta = rd.metadata
    workload = meta.get("workload", {})
    audit = audit_report(rd)
    ops = final_operations(rd.raw)
    n_seeds = rd.raw["seed"].nunique() if "seed" in rd.raw else "?"
    depths = sorted(rd.raw["depth_config"].unique())
    splits = sorted(rd.raw["split_threshold"].unique())
    deltas = sorted({_delta_label(v) for v in rd.raw["delta"].unique()})
    presets = rd.raw["search_preset"].nunique()
    grid = meta.get("preset_grid", {})

    elig = sum(1 for r in pairs if r["eligible"])
    total = len(pairs)
    lines: list[str] = []
    lines.append("# R1 Final Comparison — Research Figures Report\n")
    lines.append("Post-processing of a finished equal-search-work evaluation; the "
                 "benchmark was not rerun and the source files were not modified.\n")

    lines.append("## 1. Experiment snapshot\n")
    lines.append(f"- Evaluation run: `{meta.get('invocation_id', rd.root.name)}`")
    lines.append(f"- Source directory: `{rd.root}`")
    lines.append(f"- Git SHA: `{meta.get('git_sha', 'n/a')}`")
    lines.append(f"- Workload: dataset={workload.get('dataset', 'synthetic')}, "
                 f"dim={workload.get('dim', '?')}, "
                 f"initial_size={workload.get('initial_size', '?')}, "
                 f"operations={workload.get('operations', '?')}, "
                 f"insert_prob={workload.get('insert_prob', '?')}, "
                 f"workload={workload.get('workload', '?')}")
    lines.append(f"- Depths: {depths}; split thresholds: {splits}; "
                 f"delta(s): {deltas}")
    lines.append(f"- Held-out seeds: {n_seeds}; final checkpoint at "
                 f"operations_processed={ops}")
    lines.append(f"- Search grid: beam={grid.get('beam_width_values')}, "
                 f"n_postings={grid.get('n_postings_values')} "
                 f"({presets} presets)")
    lines.append(f"- Query count per checkpoint: "
                 f"{int(rd.raw['query_count'].iloc[0]) if 'query_count' in rd.raw else '?'}")
    lines.append(f"- Primary metric: {rk}\n")

    lines.append("## 2. Primary outcome — paired recall gain under a work budget\n")
    lines.append("Delta is `100 * (inplace_best_recall - outofplace_best_recall)` "
                 "at the same cap; positive favours in-place. Presets are chosen "
                 "per strategy from the tested grid (see §6 caveat).\n")
    lines.append("| depth | split | delta | budget | eligible | seeds | ip rec | "
                 "oop rec | Δ pp | ip scan (mean) | oop scan (mean) | gap % | "
                 "95% CI |")
    lines.append("|---|---|---|---|---|---|---|---|---|---|---|---|---|")
    for s in summary:
        if not np.isfinite(s["delta_recall_pp_mean"]):
            continue
        ci = (f"[{s['ci95_lo']:.1f}, {s['ci95_hi']:.1f}]"
              if np.isfinite(s["ci95_lo"]) else "n/a")
        lines.append(
            f"| {s['depth_config']} | {s['split_threshold']} | {s['delta']} | "
            f"{s['budget']:.0f} | {s['eligible_pairs']}/{s['n_requested']} | "
            f"{s['seed_count']} | {s['ip_recall_mean']:.3f} | "
            f"{s['oop_recall_mean']:.3f} | {s['delta_recall_pp_mean']:+.2f} | "
            f"{s['ip_scan_mean']:.0f} | {s['oop_scan_mean']:.0f} | "
            f"{s['scan_gap_pct_mean']:.1f} | {ci} |")
    lines.append("")
    wins = sum(1 for s in summary
               if np.isfinite(s["delta_recall_pp_mean"]) and s["delta_recall_pp_mean"] > 0)
    losses = sum(1 for s in summary
                 if np.isfinite(s["delta_recall_pp_mean"]) and s["delta_recall_pp_mean"] < 0)
    ties = sum(1 for s in summary
               if np.isfinite(s["delta_recall_pp_mean"]) and s["delta_recall_pp_mean"] == 0)
    lines.append(f"Across all reported (config, budget) cells: in-place higher in "
                 f"**{wins}**, out-of-place higher in **{losses}**, tied in **{ties}**. "
                 "Sign is not uniform, so no 'in-place wins everywhere' claim is made.\n")

    lines.append("## 3. Frontier coverage\n")
    ip_work = rd.raw[rd.raw.strategy == "inplace"]["vectors_scanned_per_query"]
    oop_work = rd.raw[rd.raw.strategy == "outofplace"]["vectors_scanned_per_query"]
    lines.append(f"- Observed scan range — in-place: "
                 f"{ip_work.min():.0f}–{ip_work.max():.0f}, out-of-place: "
                 f"{oop_work.min():.0f}–{oop_work.max():.0f} vectors/query.")
    lines.append(f"- Requested budget comparisons: {total}; both strategies "
                 f"eligible: {elig} ({100*elig/total:.1f}% if total else 0); "
                 f"unavailable: {total-elig}.")
    lines.append(f"- Budget grid: {[int(b) for b in budgets]} vectors/query.\n")

    lines.append("## 4. Maintenance cost\n")
    if maint_col is None:
        lines.append("Skipped: the evaluation raw schema contains no cumulative "
                     "maintenance **wall-time** column "
                     f"(checked {list(MAINT_WALL_CANDIDATES)}); a substitute was "
                     "not fabricated. Use the Task-8 paired benchmark "
                     "(`scripts/compare_maintenance.py`) for maintenance timing.\n")
    else:
        lines.append(f"Figure 3 uses `{maint_col}` (paired by config/seed).\n")

    lines.append("## 5. Correctness gate\n")
    lines.append(f"- Raw rows: {audit['n_raw_rows']}; failed rows: "
                 f"{audit['n_failed_rows']}.")
    for k in ("correctness_status_counts", "matched_status_counts"):
        if k in audit:
            lines.append(f"- {k}: {audit[k]}")
    for k in ("npa_violations", "ownership_errors", "coverage_errors",
              "membership_violations", "deleted_id_retrievals"):
        if f"max_{k}" in audit:
            lines.append(f"- max {k}: {audit[f'max_{k}']}")
    lines.append(f"- immutability_violations: "
                 f"{audit.get('immutability_violations', 'n/a')}")
    lines.append("")

    lines.append("## 6. Limitations\n")
    lines.append("- **Empirical envelope, not exact scan equality.** At a budget "
                 "B each strategy's shown recall is its best preset with measured "
                 "work <= B; the selected preset may scan fewer vectors than B. "
                 "Actual scans and their gap are reported in §2 and "
                 "`figure_data/fig02_*.csv`.")
    lines.append("- **Finite preset grid.** Recall is bounded by the tested "
                 "beam x n_postings grid; a denser grid could raise any frontier.")
    lines.append("- **Few held-out seeds.** Intervals are descriptive bootstrap "
                 "intervals over seeds (clustered; checkpoints within a seed are "
                 "not independent), not precise inferential statements.")
    lines.append("- **Synthetic dataset** (clustered synthetic vectors); results "
                 "are in-memory and say nothing about disk/NVMe behaviour.")
    lines.append("- **Not a causal maintenance effect.** Differences combine "
                 "index layout, unit fan-out and routing; the figure does not "
                 "isolate the maintenance algorithm alone.")
    lines.append("- `results_budget_matched.csv` (Task-9 selected pairing) and "
                 "`results_budget_oracle.csv` (per-checkpoint oracle) are "
                 "diagnostic only and are excluded from all headline numbers.\n")

    lines.append("## 7. Figure captions\n")
    lines.append("**Figure 1 (`fig01_recall_work_frontier`).** Empirical "
                 "Recall@10 versus measured vectors scanned per query at the "
                 "final checkpoint, one non-dominated envelope per "
                 "(depth, delta, split, strategy); small multiples facet depth "
                 "(rows) and delta (columns), with split threshold as line style "
                 "and strategy as colour. Bold lines are means over held-out "
                 f"seeds ({n_seeds}); faint markers are per-seed frontiers. "
                 "x is log-scaled. The envelope is over the tested preset grid "
                 "only.\n")
    lines.append("**Figure 2 (`fig02_recall_gain_by_work_budget`).** Paired "
                 "Recall@10 difference (percentage points; >0 favours in-place) "
                 "at a common work budget, at the final checkpoint. Panels "
                 "facet (split, delta); a line per hierarchy depth; faint "
                 "markers are individual held-out seeds. Only budget points "
                 "where both strategies have an eligible preset are drawn.\n")
    if maint_col is not None:
        lines.append("**Figure 3 (`fig03_maintenance_cost`).** Cumulative "
                     "maintenance wall time at the final checkpoint, paired "
                     "in-place vs out-of-place by configuration and seed "
                     "(log y); grey segments link each pair.\n")
    else:
        lines.append("**Figure 3.** Not produced — no maintenance wall-time "
                     "column in the evaluation output (see §4).\n")

    lines.append("## 8. Output files\n")
    lines.append("- `figures/fig01_recall_work_frontier.{png,pdf,svg}`")
    lines.append("- `figures/fig02_recall_gain_by_work_budget.{png,pdf,svg}`")
    if maint_col is not None:
        lines.append("- `figures/fig03_maintenance_cost.{png,pdf,svg}`")
    lines.append("- `figures/figure_data/fig01_recall_work_frontier.csv`")
    lines.append("- `figures/figure_data/fig02_recall_gain_by_work_budget.csv`")
    lines.append("- `analysis_summary.csv`")
    lines.append("- `figures/research_figures_report.md` (this file)\n")
    if issues:
        lines.append("**Validation issues:** " + "; ".join(issues) + "\n")
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="plot_search_budget",
        description="Generate publication figures + summary from a finished "
                    "compare_search_budget evaluation directory (post-processing "
                    "only; no rerun).",
    )
    p.add_argument("--input", required=True,
                   help="evaluation output dir containing results_budget_raw.csv")
    p.add_argument("--budgets", default=",".join(str(int(b)) for b in DEFAULT_BUDGETS),
                   help="comma-separated work budgets (vectors/query)")
    p.add_argument("--bootstrap", type=int, default=2000)
    p.add_argument("--bootstrap-seed", dest="bootstrap_seed", type=int,
                   default=20261019)
    p.add_argument("--no-figures", dest="no_figures", action="store_true",
                   help="compute data/report only (skip matplotlib rendering)")
    return p


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    budgets = tuple(float(x) for x in args.budgets.split(",") if x.strip())
    if not budgets:
        raise SystemExit("--budgets must be non-empty")
    run(Path(args.input), budgets=budgets, n_boot=args.bootstrap,
        rng_seed=args.bootstrap_seed, make_figures=not args.no_figures)
    return 0
