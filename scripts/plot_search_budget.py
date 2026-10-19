#!/usr/bin/env python
"""scripts/plot_search_budget.py — figures for the final equal-search-work run.

Thin entry point.  Makes the checkout importable and forwards to
``vecstudy.harness.plot_search_budget.main``.

Post-processing only: reads a finished ``compare_search_budget`` evaluation
directory and writes figures, plotted data, ``analysis_summary.csv`` and
``research_figures_report.md`` under ``<input>/figures/``.  It does not rerun the
benchmark or modify the source result files.

Examples
--------
    .venv/bin/python scripts/plot_search_budget.py \
        --input results/search_budget/evaluation/<run>
    .venv/bin/python scripts/plot_search_budget.py \
        --input results/search_budget/evaluation/<run> --budgets 64,128,256,512
"""

from __future__ import annotations

import os
import sys

os.environ.setdefault("MPLBACKEND", "Agg")

_REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)

from vecstudy.harness.plot_search_budget import main  # noqa: E402

if __name__ == "__main__":
    raise SystemExit(main())
