#!/usr/bin/env python
"""scripts/compare_search_budget.py — equal-search-work comparison harness.

Thin entry point.  Pins single-threaded BLAS before NumPy is imported, makes the
checkout importable, and forwards to ``vecstudy.harness.search_budget.main``.

Two phases:
  * ``--phase calibration`` replays each base config once per strategy, evaluates
    the whole ``beam_width x n_postings`` grid on cloned checkpoints, and selects
    an in-place/out-of-place preset pair per depth (plus a global fallback) that
    minimizes the median relative scan gap.
  * ``--phase evaluate`` applies that fixed selection to held-out seeds and emits
    the matched, summary and (descriptive) oracle result files.

Examples
--------
    .venv/bin/python scripts/compare_search_budget.py --help

    # calibration
    .venv/bin/python scripts/compare_search_budget.py --phase calibration \
        --initial-size 400 --operations 400 --query-count 32 \
        --checkpoint-interval 50 --depths 1,2,3 --split-thresholds 24,48 \
        --deltas none,0.3 --calibration-seeds 0,1,2 \
        --beam-width-values 2,4,8,16,32 --n-postings-values 1,2,4,8 \
        --blas-threads 1 --out results/search_budget/calibration

    # held-out evaluation
    .venv/bin/python scripts/compare_search_budget.py --phase evaluate \
        --initial-size 2000 --operations 2000 --query-count 64 \
        --checkpoint-interval 250 --depths 1,2,3,4 \
        --split-thresholds 48,96,192 --deltas none,0.15,0.3 \
        --evaluation-seeds 3,4,5,6,7,8,9 \
        --calibration-results results/search_budget/calibration \
        --blas-threads 1 --out results/search_budget/evaluation

See ``R1_T9_Report.md`` for sample output and measurement caveats.
"""

from __future__ import annotations

import os
import sys

# Pin single-threaded BLAS before importing numpy (determinism / fairness).
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("NUMEXPR_NUM_THREADS", "1")

_REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)

from vecstudy.harness.search_budget import main  # noqa: E402

if __name__ == "__main__":
    raise SystemExit(main())
