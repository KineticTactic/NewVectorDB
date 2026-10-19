"""structure_report.py — one structural diagnostics figure + its data table.

Reads ``results_structure_checkpoints.csv`` (written by the equal-work harness)
and produces:

* ``structure_threshold_series.csv`` — posting-size p50/p95/max aggregated per
  ``(strategy, split_threshold)`` (median over checkpoints/seeds), and
* at most one figure, ``fig_structure_posting_sizes.*`` — posting-size
  percentiles versus ``split_threshold`` with one panel per strategy.

This is deliberately a single diagnostics figure; it never touches the primary
recall-vs-work figures produced by ``plot_search_budget``.
"""

from __future__ import annotations

import csv
import statistics
from pathlib import Path
from typing import Any

import numpy as np

_FIG_NAME = "fig_structure_posting_sizes"


def _f(v: Any) -> float:
    if v is None or v == "":
        return float("nan")
    try:
        return float(v)
    except (TypeError, ValueError):
        return float("nan")


def load_structure_rows(path: Path) -> list[dict]:
    with open(path) as f:
        return list(csv.DictReader(f))


def threshold_series(rows: list[dict]) -> list[dict]:
    """Aggregate posting-size stats per (strategy, split_threshold).

    Each percentile is the median over all checkpoints/seeds/configs at that
    threshold (``nan`` entries excluded).
    """
    groups: dict[tuple[str, float], list[dict]] = {}
    for r in rows:
        key = (str(r.get("strategy")), _f(r.get("split_threshold")))
        groups.setdefault(key, []).append(r)

    out: list[dict] = []
    for (strategy, split), grp in sorted(groups.items(),
                                         key=lambda kv: (kv[0][0], kv[0][1])):
        rec: dict[str, Any] = {
            "strategy": strategy,
            "split_threshold": split,
            "n_observations": len(grp),
        }
        for name, col in (("posting_size_p50", "posting_size_median"),
                          ("posting_size_p90", "posting_size_p90"),
                          ("posting_size_p95", "posting_size_p95"),
                          ("posting_size_max", "posting_size_max")):
            vals = [v for v in (_f(r.get(col)) for r in grp) if not np.isnan(v)]
            rec[name] = float(statistics.median(vals)) if vals else float("nan")
        live = [_f(r.get("live_posting_count")) for r in grp]
        live = [v for v in live if not np.isnan(v)]
        rec["live_posting_count_median"] = (
            float(statistics.median(live)) if live else float("nan"))
        out.append(rec)
    return out


def plot_structure_diagnostics(structure_csv: Path, out_dir: Path, *,
                               make_figures: bool = True) -> dict[str, Any]:
    """Write the threshold-series CSV and (optionally) the diagnostics figure."""
    out_dir.mkdir(parents=True, exist_ok=True)
    rows = load_structure_rows(Path(structure_csv))
    series = threshold_series(rows)

    series_path = out_dir / "structure_threshold_series.csv"
    with open(series_path, "w", newline="") as f:
        if series:
            w = csv.DictWriter(f, fieldnames=list(series[0].keys()))
            w.writeheader()
            w.writerows(series)

    written = [series_path.name]
    if make_figures and series:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        strategies = ["inplace", "outofplace"]
        present = [s for s in strategies if any(x["strategy"] == s for x in series)]
        fig, axes = plt.subplots(1, max(1, len(present)),
                                 figsize=(5 * max(1, len(present)), 4),
                                 squeeze=False)
        for ax, strat in zip(axes[0], present):
            sub = [x for x in series if x["strategy"] == strat]
            xs = [x["split_threshold"] for x in sub]
            for name, marker in (("posting_size_p50", "o"),
                                 ("posting_size_p90", "s"),
                                 ("posting_size_p95", "^"),
                                 ("posting_size_max", "d")):
                ys = [x[name] for x in sub]
                ax.plot(xs, ys, marker=marker, label=name.replace("posting_size_", ""))
            ax.plot(xs, xs, ls="--", color="grey", label="split_threshold")
            ax.set_title(strat)
            ax.set_xlabel("split_threshold")
            ax.set_ylabel("live posting size")
            ax.legend(fontsize=8)
        fig.suptitle("Posting-size distribution vs split_threshold")
        fig.tight_layout()
        for ext in ("png", "pdf", "svg"):
            p = out_dir / f"{_FIG_NAME}.{ext}"
            fig.savefig(p, dpi=150)
            written.append(p.name)
        plt.close(fig)

    return {"series": series, "written": written}
