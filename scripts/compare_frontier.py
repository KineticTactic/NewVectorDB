#!/usr/bin/env python
"""scripts/compare_frontier.py — per-stratum recall-vs-work frontiers.

Thin entry point.  Makes the checkout importable and forwards to
``vecstudy.harness.frontier.main``.

This is the final-comparison analysis: it reads an **existing** Task-9
``results_budget_raw`` table (CSV or Parquet) and, per experimental stratum
(base config x seed x checkpoint), derives an independent Recall@10-vs-work
frontier for each strategy and compares them at common observed scan budgets.
It does not rerun the experiment.

Examples
--------
    .venv/bin/python scripts/compare_frontier.py \
        --raw results/search_budget/evaluation/sb-<stamp> \
        --out results/search_budget/frontier

See ``R1_Final_Report.md`` for the interpretation rules and caveats.
"""

from __future__ import annotations

import os
import sys

os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("NUMEXPR_NUM_THREADS", "1")

_REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)

from vecstudy.harness.frontier import main  # noqa: E402

if __name__ == "__main__":
    raise SystemExit(main())
