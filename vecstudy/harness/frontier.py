"""harness/frontier.py — per-stratum recall-vs-work frontiers.

Backs ``scripts/compare_frontier.py`` and finishes the in-place vs out-of-place
comparison started in ``harness/search_budget.py``.

Why a second analysis
---------------------
Task 9's pooled frontier mixed different depths/configurations and its
selected-preset pairing could pair *different* presets whose recall difference is
not attributable to the maintenance strategy alone.  This module instead treats
**observed search work** (``vectors_scanned_per_query``, measured with a
dedicated query counter) as the budget and compares strategies **within each
experimental stratum**:

* A stratum is one base configuration x seed x checkpoint (the live vector set,
  query set and brute-force ground truth are shared by both arms — verified via
  ``dataset_checksum``/``trace_hash``/``live_vector_count``).
* For each stratum we keep each strategy's measured ``(work, recall)`` points and
  derive its nondominated frontier.
* We take the *common observed work interval* of the two frontiers and pick
  data-derived low/middle/high budgets inside it.  If the interval is empty the
  stratum is marked ``no_common_work_range`` — no fabricated budget.
* At a budget ``B`` the best attainable recall for a strategy is
  ``max(recall over presets whose measured work <= B)``.

Scope of the claim
------------------
This is an **attainable-performance** comparison over the tested preset grid
(``best_observed_at_budget`` / ``empirical_frontier``).  Choosing the preset from
the same measured checkpoint makes it an in-sample frontier, not a frozen
production preset, and it does **not** isolate the maintenance algorithm from
index-layout and routing differences.
"""

from __future__ import annotations

import argparse
import datetime
import json
import os
import platform
from pathlib import Path
from typing import Any

import numpy as np

from .compare import (
    STRATEGIES,
    write_csv,
    write_parquet,
    render_table,
    _environment_metadata,
    _rss_bytes,
    _git_sha,
)

PRIMARY_K_DEFAULT = 10

BUDGET_POSITIONS = ("low", "middle", "high")
BUDGET_FRACTIONS = {"low": 0.25, "middle": 0.50, "high": 0.75}

# Canonical stratum identity.  Fields absent from a raw row resolve to "".
STRATUM_FIELDS = (
    "dataset", "dim", "initial_size", "workload", "insert_prob",
    "leaf_size", "branching_factor", "depth_config", "split_threshold",
    "merge_threshold", "delta", "memtable_limit", "compaction_threshold",
    "seed", "checkpoint",
)

# Identifiers that must agree between the two arms of a paired checkpoint.
BASIS_FIELDS = (
    "dataset_checksum", "trace_hash", "operations_processed",
    "live_vector_count",
)

REQUIRED_COLUMNS = (
    "strategy", "search_preset", "beam_scalar", "n_postings",
    "vectors_scanned_per_query", "recall_at_10", "query_latency_p50_ms",
    "query_latency_p95_ms", "run_failed", "depth_config", "split_threshold",
    "delta", "seed", "checkpoint",
)

FRONTIER_FILE = "results_budget_frontier_by_stratum.csv"
PAIRED_FILE = "results_budget_frontier_paired.csv"
SUMMARY_FILE = "results_budget_frontier_summary.csv"
OVERLAP_FILE = "results_budget_work_overlap.csv"


# ---------------------------------------------------------------------------
# Small helpers
# ---------------------------------------------------------------------------

def _key_text(value: Any) -> str:
    """Normalize a value for use in a stratum key (NaN/None -> '')."""
    if value is None:
        return ""
    if isinstance(value, float) and np.isnan(value):
        return ""
    return str(value)


def _delta_label(value: Any) -> str:
    if value is None or (isinstance(value, float) and np.isnan(value)):
        return "none"
    return str(value)


def _primary_k(raw: list[dict]) -> int:
    for r in raw:
        if r.get("k") not in (None, ""):
            try:
                return int(r["k"])
            except (TypeError, ValueError):
                continue
    return PRIMARY_K_DEFAULT


def _stratum_key(row: dict) -> tuple:
    return tuple(_key_text(row.get(f)) for f in STRATUM_FIELDS)


def _is_failed(row: dict) -> bool:
    v = row.get("run_failed")
    if isinstance(v, str):
        return v.strip().lower() in ("true", "1")
    return bool(v)


def _num(row: dict, key: str) -> float:
    v = row.get(key)
    if v is None or v == "":
        return float("nan")
    return float(v)


# ---------------------------------------------------------------------------
# Validation
# ---------------------------------------------------------------------------

def validate_raw(raw: list[dict]) -> dict:
    """Return a validation report; ``ok`` is False only on hard failures."""
    issues: list[str] = []
    if not raw:
        return {"ok": False, "issues": ["raw row list is empty"],
                "n_rows": 0, "n_strata": 0, "n_unpaired_strata": 0}

    missing_cols = [c for c in REQUIRED_COLUMNS
                    if not any(c in r for r in raw)]
    if missing_cols:
        issues.append(f"missing required columns: {missing_cols}")

    has_query_work = any(k.startswith("work_") for k in raw[0])
    has_maint_work = any(k.startswith("delta_") for k in raw[0])
    if not has_query_work:
        issues.append("no dedicated query-work counters (work_* columns)")

    groups = group_strata(raw)
    n_unpaired = 0
    basis_mismatch = 0
    for key, st in groups.items():
        ip = st.get("inplace") or {}
        oop = st.get("outofplace") or {}
        if not ip or not oop:
            n_unpaired += 1
            continue
        ipr = next(iter(ip.values()))
        oopr = next(iter(oop.values()))
        for f in BASIS_FIELDS:
            if _key_text(ipr.get(f)) != _key_text(oopr.get(f)):
                basis_mismatch += 1
                issues.append(
                    f"stratum {key}: basis field {f!r} differs "
                    f"({ipr.get(f)!r} vs {oopr.get(f)!r})")
                break
    if n_unpaired:
        issues.append(f"{n_unpaired} stratum/strata missing one strategy")

    seeds = sorted({_key_text(r.get("seed")) for r in raw})
    return {
        "ok": not issues,
        "issues": issues,
        "n_rows": len(raw),
        "n_strata": len(groups),
        "n_unpaired_strata": n_unpaired,
        "n_basis_mismatches": basis_mismatch,
        "has_dedicated_query_counters": has_query_work,
        "has_maintenance_counters": has_maint_work,
        "seeds": seeds,
    }


def group_strata(raw: list[dict]) -> dict[tuple, dict[str, dict]]:
    """stratum key -> {"head": row, "inplace": {preset: row}, "outofplace": ...}."""
    groups: dict[tuple, dict[str, dict]] = {}
    for r in raw:
        key = _stratum_key(r)
        bucket = groups.setdefault(key, {"head": r, "inplace": {}, "outofplace": {}})
        strat = str(r.get("strategy"))
        bucket.setdefault(strat, {})[str(r.get("search_preset"))] = r
    return groups


# ---------------------------------------------------------------------------
# Frontier + overlap + budget
# ---------------------------------------------------------------------------

def _point(row: dict, rk: str) -> dict:
    return {
        "search_preset": row.get("search_preset"),
        "beam_scalar": row.get("beam_scalar"),
        "n_postings": row.get("n_postings"),
        "work": _num(row, "vectors_scanned_per_query"),
        "recall": _num(row, rk),
        "q_p50": _num(row, "query_latency_p50_ms"),
        "q_p95": _num(row, "query_latency_p95_ms"),
    }


def strategy_points(strat: dict[str, dict], strategy: str, rk: str) -> list[dict]:
    rows = strat.get(strategy) or {}
    pts = [_point(r, rk) for r in rows.values() if not _is_failed(r)]
    pts = [p for p in pts if np.isfinite(p["work"]) and np.isfinite(p["recall"])]
    return pts


def _dominates(a: dict, b: dict) -> bool:
    no_more_work = a["work"] <= b["work"]
    at_least_recall = a["recall"] >= b["recall"]
    strict = a["work"] < b["work"] or a["recall"] > b["recall"]
    return no_more_work and at_least_recall and strict


def frontier_points(points: list[dict]) -> list[dict]:
    """Remove dominated points; return sorted by (work, -recall, label).

    Equivalent presets that produce identical measured ``(work, recall, p50)``
    (e.g. different beam widths at depth 1, where the beam is unused) are
    collapsed to the lexicographically smallest label.
    """
    keep = []
    for p in points:
        if not any(_dominates(q, p) for q in points if q is not p):
            keep.append(p)
    keep.sort(key=lambda p: (p["work"], -p["recall"], str(p["search_preset"])))
    deduped: list[dict] = []
    seen: set[tuple] = set()
    for p in keep:
        sig = (round(p["work"], 6), round(p["recall"], 12))
        if sig in seen:
            continue
        seen.add(sig)
        deduped.append(p)
    return deduped


def work_overlap(ip_pts: list[dict], oop_pts: list[dict]) -> dict:
    if not ip_pts or not oop_pts:
        return {"has_common_range": False, "reason": "no_points",
                "common_lo": float("nan"), "common_hi": float("nan")}
    ip_lo = min(p["work"] for p in ip_pts)
    ip_hi = max(p["work"] for p in ip_pts)
    oop_lo = min(p["work"] for p in oop_pts)
    oop_hi = max(p["work"] for p in oop_pts)
    lo = max(ip_lo, oop_lo)
    hi = min(ip_hi, oop_hi)
    if hi <= lo:
        return {"has_common_range": False, "reason": "no_common_work_range",
                "ip_min": ip_lo, "ip_max": ip_hi,
                "oop_min": oop_lo, "oop_max": oop_hi,
                "common_lo": float("nan"), "common_hi": float("nan")}
    return {"has_common_range": True, "reason": "",
            "ip_min": ip_lo, "ip_max": ip_hi,
            "oop_min": oop_lo, "oop_max": oop_hi,
            "common_lo": lo, "common_hi": hi}


def budget_schedule(lo: float, hi: float) -> dict[str, float]:
    width = hi - lo
    return {pos: lo + frac * width for pos, frac in BUDGET_FRACTIONS.items()}


def pick_at_budget(points: list[dict], budget: float) -> dict | None:
    """Best recall among presets with work <= budget; deterministic tie-break."""
    feasible = [p for p in points if p["work"] <= budget]
    if not feasible:
        return None
    feasible.sort(key=lambda p: (-p["recall"], p["work"], p["q_p50"],
                                 str(p["search_preset"])))
    return feasible[0]


# ---------------------------------------------------------------------------
# Output builders
# ---------------------------------------------------------------------------

def _stratum_meta(head: dict) -> dict:
    meta = {f: head.get(f) for f in STRATUM_FIELDS}
    meta["delta"] = _delta_label(head.get("delta"))
    for f in BASIS_FIELDS:
        meta[f] = head.get(f)
    return meta


def build_frontier_by_stratum(raw: list[dict]) -> list[dict]:
    rk = f"recall_at_{_primary_k(raw)}"
    out: list[dict] = []
    groups = group_strata(raw)
    for idx, key in enumerate(sorted(groups)):
        strat = groups[key]
        meta = _stratum_meta(strat["head"])
        for strategy in STRATEGIES:
            pts = strategy_points(strat, strategy, rk)
            for p in frontier_points(pts):
                out.append({
                    "stratum_id": idx, "strategy": strategy, **meta,
                    "search_preset": p["search_preset"],
                    "beam_scalar": p["beam_scalar"],
                    "n_postings": p["n_postings"],
                    "vectors_scanned_per_query": p["work"],
                    rk: p["recall"],
                    "query_latency_p50_ms": p["q_p50"],
                    "query_latency_p95_ms": p["q_p95"],
                    "dominated": False,
                })
    return out


def build_work_overlap(raw: list[dict]) -> list[dict]:
    rk = f"recall_at_{_primary_k(raw)}"
    out: list[dict] = []
    groups = group_strata(raw)
    for idx, key in enumerate(sorted(groups)):
        strat = groups[key]
        meta = _stratum_meta(strat["head"])
        ip_pts = strategy_points(strat, "inplace", rk)
        oop_pts = strategy_points(strat, "outofplace", rk)
        ov = work_overlap(ip_pts, oop_pts)
        out.append({
            "stratum_id": idx, **meta,
            "n_ip_presets": len(ip_pts), "n_oop_presets": len(oop_pts),
            "ip_min_work": ov.get("ip_min", float("nan")),
            "ip_max_work": ov.get("ip_max", float("nan")),
            "oop_min_work": ov.get("oop_min", float("nan")),
            "oop_max_work": ov.get("oop_max", float("nan")),
            "common_lo": ov.get("common_lo", float("nan")),
            "common_hi": ov.get("common_hi", float("nan")),
            "common_width": (ov["common_hi"] - ov["common_lo"]
                             if ov["has_common_range"] else float("nan")),
            "has_common_range": ov["has_common_range"],
            "reason": ov["reason"] if not ov["has_common_range"] else "",
        })
    return out


def build_frontier_paired(raw: list[dict], *, within: float, approx: float) -> list[dict]:
    from .search_budget import relative_scan_gap, match_status

    rk = f"recall_at_{_primary_k(raw)}"
    out: list[dict] = []
    groups = group_strata(raw)
    for idx, key in enumerate(sorted(groups)):
        strat = groups[key]
        meta = _stratum_meta(strat["head"])
        ip_pts = strategy_points(strat, "inplace", rk)
        oop_pts = strategy_points(strat, "outofplace", rk)
        ov = work_overlap(ip_pts, oop_pts)
        if not ov["has_common_range"]:
            out.append({
                "stratum_id": idx, "result_kind": "empirical_frontier", **meta,
                "budget_position": "none",
                "target_budget_scans_per_query": float("nan"),
                "ip_preset": "", "ip_actual_work_scans_per_query": float("nan"),
                f"ip_{rk}": float("nan"),
                "ip_q_p50_ms": float("nan"), "ip_q_p95_ms": float("nan"),
                "oop_preset": "", "oop_actual_work_scans_per_query": float("nan"),
                f"oop_{rk}": float("nan"),
                "oop_q_p50_ms": float("nan"), "oop_q_p95_ms": float("nan"),
                "recall_diff_pp": float("nan"),
                "relative_scan_gap_pct": float("nan"),
                "match_status": ov["reason"],
            })
            continue
        sched = budget_schedule(ov["common_lo"], ov["common_hi"])
        for pos in BUDGET_POSITIONS:
            budget = sched[pos]
            ip_p = pick_at_budget(ip_pts, budget)
            oop_p = pick_at_budget(oop_pts, budget)
            if ip_p is None or oop_p is None:
                status = "no_feasible_preset"
                gap = float("nan")
            else:
                gap = relative_scan_gap(ip_p["work"], oop_p["work"])
                status = match_status(gap, within, approx)
            out.append({
                "stratum_id": idx, "result_kind": "empirical_frontier", **meta,
                "budget_position": pos,
                "target_budget_scans_per_query": budget,
                "ip_preset": "" if ip_p is None else ip_p["search_preset"],
                "ip_actual_work_scans_per_query":
                    float("nan") if ip_p is None else ip_p["work"],
                f"ip_{rk}": float("nan") if ip_p is None else ip_p["recall"],
                "ip_q_p50_ms": float("nan") if ip_p is None else ip_p["q_p50"],
                "ip_q_p95_ms": float("nan") if ip_p is None else ip_p["q_p95"],
                "oop_preset": "" if oop_p is None else oop_p["search_preset"],
                "oop_actual_work_scans_per_query":
                    float("nan") if oop_p is None else oop_p["work"],
                f"oop_{rk}": float("nan") if oop_p is None else oop_p["recall"],
                "oop_q_p50_ms": float("nan") if oop_p is None else oop_p["q_p50"],
                "oop_q_p95_ms": float("nan") if oop_p is None else oop_p["q_p95"],
                "recall_diff_pp": (float("nan") if ip_p is None or oop_p is None
                                   else 100.0 * (ip_p["recall"] - oop_p["recall"])),
                "relative_scan_gap_pct": gap,
                "match_status": status,
            })
    return out


# ---------------------------------------------------------------------------
# Across-seed summary
# ---------------------------------------------------------------------------

def _seed_bootstrap_ci(by_seed: dict[str, list[float]], *, n_boot: int,
                       rng_seed: int) -> tuple[float, float, float]:
    """Mean-of-seed-means and a 95% CI from resampling seeds (not rows)."""
    seeds = sorted(by_seed)
    seed_means = np.array([float(np.mean(by_seed[s])) for s in seeds],
                          dtype=np.float64)
    point = float(seed_means.mean())
    if len(seeds) < 2:
        return point, point, point
    rng = np.random.default_rng(rng_seed)
    idx = rng.integers(0, len(seed_means), size=(n_boot, len(seed_means)))
    boot = seed_means[idx].mean(axis=1)
    lo, hi = np.percentile(boot, [2.5, 97.5])
    return point, float(lo), float(hi)


def _summarize_rows(rows: list[dict], rk: str, *, n_boot: int,
                    rng_seed: int) -> dict:
    diffs = [r["recall_diff_pp"] for r in rows
             if np.isfinite(r["recall_diff_pp"])]
    by_seed: dict[str, list[float]] = {}
    for r in rows:
        if np.isfinite(r["recall_diff_pp"]):
            by_seed.setdefault(_key_text(r.get("seed")), []).append(r["recall_diff_pp"])
    mean, lo, hi = (_seed_bootstrap_ci(by_seed, n_boot=n_boot, rng_seed=rng_seed)
                    if by_seed else (float("nan"), float("nan"), float("nan")))
    arr = np.asarray(diffs, dtype=np.float64)

    def _m(key: str) -> float:
        vals = [r[key] for r in rows if np.isfinite(r.get(key, float("nan")))]
        return float(np.mean(vals)) if vals else float("nan")

    return {
        "n_obs": len(diffs),
        "n_seeds": len(by_seed),
        "n_matched": sum(1 for r in rows if r["match_status"] == "matched"),
        "n_approx": sum(1 for r in rows if r["match_status"] == "approximate"),
        "n_unmatched": sum(1 for r in rows if r["match_status"] == "unmatched"),
        "n_no_overlap": sum(1 for r in rows
                            if r["match_status"] == "no_common_work_range"),
        "recall_diff_pp_mean": float(arr.mean()) if len(arr) else float("nan"),
        "recall_diff_pp_median": float(np.median(arr)) if len(arr) else float("nan"),
        "recall_diff_pp_std": float(arr.std(ddof=0)) if len(arr) else float("nan"),
        "recall_diff_pp_seedmean": mean,
        "recall_diff_pp_ci95_lo": lo,
        "recall_diff_pp_ci95_hi": hi,
        "ip_recall_mean": _m(f"ip_{rk}"),
        "oop_recall_mean": _m(f"oop_{rk}"),
        "ip_work_mean": _m("ip_actual_work_scans_per_query"),
        "oop_work_mean": _m("oop_actual_work_scans_per_query"),
        "relative_scan_gap_pct_mean": _m("relative_scan_gap_pct"),
        "target_budget_mean": _m("target_budget_scans_per_query"),
    }


def build_frontier_summary(paired: list[dict], rk: str, *, n_boot: int = 2000,
                           rng_seed: int = 20261019) -> list[dict]:
    out: list[dict] = []

    # 1) per-cell summary (depth x split x delta x budget position)
    cells: dict[tuple, list[dict]] = {}
    for r in paired:
        key = (_key_text(r.get("depth_config")), _key_text(r.get("split_threshold")),
               _delta_label(r.get("delta")), r.get("budget_position"))
        cells.setdefault(key, []).append(r)
    for key, rows in sorted(cells.items(), key=lambda kv: str(kv[0])):
        depth, split, delta, pos = key
        stats = _summarize_rows(rows, rk, n_boot=n_boot, rng_seed=rng_seed)
        out.append({
            "row_kind": "cell", "group_scope": "depth_split_delta_budget",
            "depth_config": depth, "split_threshold": split, "delta": delta,
            "budget_position": pos, "n_cells": len(rows), **stats,
        })

    # 2) per-depth macro (macro-average over split/delta cells within a depth+budget)
    macros: dict[tuple, list[dict]] = {}
    for r in paired:
        key = (_key_text(r.get("depth_config")), r.get("budget_position"))
        macros.setdefault(key, []).append(r)
    cell_stats: dict[tuple, list[float]] = {}
    for key, rows in cells.items():
        depth, split, delta, pos = key
        m = _summarize_rows(rows, rk, n_boot=n_boot, rng_seed=rng_seed)
        cell_stats.setdefault((depth, pos), []).append(m["recall_diff_pp_mean"])
    for key, rows in sorted(macros.items(), key=lambda kv: str(kv[0])):
        depth, pos = key
        stats = _summarize_rows(rows, rk, n_boot=n_boot, rng_seed=rng_seed)
        cell_means = cell_stats.get((depth, pos), [])
        out.append({
            "row_kind": "macro", "group_scope": "depth_macro",
            "depth_config": depth, "split_threshold": "all", "delta": "all",
            "budget_position": pos, "n_cells": len(cell_means),
            "recall_diff_pp_cellmacro": float(np.mean(cell_means))
                if cell_means else float("nan"),
            "included_cells": ";".join(sorted({k[1] + "/" + k[2]
                                               for k in cells
                                               if k[0] == depth and k[3] == pos})),
            **stats,
        })
    return out


# ---------------------------------------------------------------------------
# Reading / writing
# ---------------------------------------------------------------------------

def read_raw(path: Path) -> list[dict]:
    """Read a Task-9 raw table (CSV or Parquet).  ``path`` may be a directory."""
    if path.is_dir():
        cand = path / "results_budget_raw.parquet"
        if not cand.exists():
            cand = path / "results_budget_raw.csv"
        path = cand
    if not path.exists():
        raise FileNotFoundError(f"raw results not found: {path}")
    import pandas as pd
    if path.suffix == ".parquet":
        df = pd.read_parquet(path)
    else:
        df = pd.read_csv(path)
    return df.to_dict("records")


def _render_paired(paired: list[dict]) -> str:
    cols = [
        ("depth", "depth_config", 5, "s"), ("split", "split_threshold", 5, "s"),
        ("delta", "delta", 5, "s"), ("cp", "checkpoint", 3, "s"),
        ("bud", "budget_position", 6, "s"),
        ("target", "target_budget_scans_per_query", 8, "f0"),
        ("ip_work", "ip_actual_work_scans_per_query", 8, "f0"),
        ("oop_work", "oop_actual_work_scans_per_query", 8, "f0"),
        ("gap%", "relative_scan_gap_pct", 7, "f1"),
        ("status", "match_status", 12, "s"),
        ("dRec_pp", "recall_diff_pp", 8, "f2"),
    ]
    return render_table("Per-stratum empirical frontier (first rows)",
                        cols, paired[:24])


def write_outputs(out_dir: Path, tables: dict[str, list[dict]]) -> dict[str, int]:
    out_dir.mkdir(parents=True, exist_ok=True)
    written: dict[str, int] = {}
    for name, rows in tables.items():
        write_csv(out_dir / name, rows)
        written[name] = len(rows)
    for name in (FRONTIER_FILE, PAIRED_FILE):
        rows = tables.get(name, [])
        pq = name.replace(".csv", ".parquet")
        if write_parquet(out_dir / pq, rows):
            written[pq] = len(rows)
    return written


def _experiment_id(prefix: str = "fr") -> str:
    stamp = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
    return f"{prefix}-{stamp}-{os.getpid()}"


def _unique_out_dir(base: Path, experiment_id: str, overwrite: bool) -> Path:
    return base if overwrite else base / experiment_id


# ---------------------------------------------------------------------------
# Execution
# ---------------------------------------------------------------------------

def run_analysis(raw_path: Path, out_dir: Path, *, within: float, approx: float,
                 n_boot: int, rng_seed: int, progress: bool = True) -> dict:
    raw = read_raw(raw_path)
    validation = validate_raw(raw)
    rk = f"recall_at_{_primary_k(raw)}"

    by_stratum = build_frontier_by_stratum(raw)
    overlap = build_work_overlap(raw)
    paired = build_frontier_paired(raw, within=within, approx=approx)
    summary = build_frontier_summary(paired, rk, n_boot=n_boot, rng_seed=rng_seed)

    no_overlap = sum(1 for r in overlap if not _truthy(r["has_common_range"]))

    tables = {
        FRONTIER_FILE: by_stratum,
        OVERLAP_FILE: overlap,
        PAIRED_FILE: paired,
        SUMMARY_FILE: summary,
    }
    meta = {
        "invocation_id": _experiment_id(),
        "timestamp": datetime.datetime.now().isoformat(),
        "git_sha": _git_sha(),
        "environment": _environment_metadata(None),
        "platform": platform.platform(),
        "rss_bytes": _rss_bytes()[0], "rss_kind": _rss_bytes()[1],
        "source_raw": str(raw_path),
        "n_raw_rows": len(raw),
        "validation": validation,
        "budget_rule": {
            "metric": "vectors_scanned_per_query (dedicated query counters)",
            "frontier_definition": "best_observed_at_budget: max recall over "
                                   "presets whose measured work <= B",
            "common_interval": "max(strategy minima) .. min(strategy maxima)",
            "budget_positions": {k: f"{int(v*100)}% through common interval"
                                 for k, v in BUDGET_FRACTIONS.items()},
            "empty_interval_status": "no_common_work_range",
            "thresholds": {"matched_within_pct": within,
                           "approximate_within_pct": approx},
            "bootstrap": {"resample_unit": "seed/run",
                          "n_boot": n_boot, "rng_seed": rng_seed,
                          "ci": "95% percentile of resampled seed means"},
        },
        "config_keys": list(STRATUM_FIELDS),
        "basis_keys": list(BASIS_FIELDS),
        "primary_k": _primary_k(raw),
        "seeds": validation.get("seeds", []),
        "n_strata": validation.get("n_strata"),
        "n_no_overlap_strata": int(no_overlap),
        "search_semantics": {
            "metric": "inner product; distance = 1 - similarity; L2-normalized",
            "beam_widths_translation": "scalar beam B -> beam_widths=(B,)",
            "depth1_note": "depth==1 flat fast path; beam unused, only n_postings",
        },
        "claim_scope": {
            "kind": "empirical_frontier / best_observed_at_budget",
            "note": "attainable performance over the tested preset grid within a "
                    "stratum; not a frozen production preset and not a causal "
                    "isolate of the maintenance algorithm from index layout.",
        },
        "output_schema": {name: (list(rows[0].keys()) if rows else [])
                          for name, rows in tables.items()},
    }

    written = write_outputs(out_dir, tables)
    with open(out_dir / "run_metadata.json", "w") as f:
        json.dump(meta, f, indent=2, default=str)

    if progress:
        print(f"Source raw: {raw_path}")
        print(f"Validation ok={validation['ok']} strata={validation['n_strata']} "
              f"unpaired={validation['n_unpaired_strata']} "
              f"issues={len(validation['issues'])}")
        print(f"No-overlap strata: {no_overlap}")
        print(f"Output directory: {out_dir}")
        print("Files: " + ", ".join(f"{n} ({c} rows)" for n, c in written.items()))
        print()
        print(_render_paired(paired))
    return written


def _truthy(v: Any) -> bool:
    if isinstance(v, str):
        return v.strip().lower() in ("true", "1")
    return bool(v)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="compare_frontier",
        description="Per-stratum recall-vs-work frontiers for the final "
                    "in-place vs out-of-place comparison.  Reads an existing "
                    "Task-9 results_budget_raw table; does not rerun the "
                    "experiment.",
    )
    p.add_argument("--raw", required=True,
                   help="Task-9 output dir or results_budget_raw.{csv,parquet}")
    p.add_argument("--out", default="results/search_budget/frontier")
    p.add_argument("--overwrite", action="store_true")
    p.add_argument("--match-within", dest="match_within", type=float, default=5.0)
    p.add_argument("--match-approx", dest="match_approx", type=float, default=10.0)
    p.add_argument("--bootstrap", type=int, default=2000)
    p.add_argument("--bootstrap-seed", dest="bootstrap_seed", type=int,
                   default=20261019)
    return p


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    if args.match_within < 0 or args.match_approx < args.match_within:
        parser.error("require 0 <= match_within <= match_approx")
    if args.bootstrap < 0:
        parser.error("bootstrap must be >= 0")
    run_analysis(
        Path(args.raw), _unique_out_dir(Path(args.out), _experiment_id(),
                                        args.overwrite),
        within=args.match_within, approx=args.match_approx,
        n_boot=args.bootstrap, rng_seed=args.bootstrap_seed,
    )
    return 0
