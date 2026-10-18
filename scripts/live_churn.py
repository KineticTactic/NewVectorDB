#!/usr/bin/env python
"""scripts/live_churn.py — entry point for the continuous live-churn monitor.

Continuously inserts and deletes vectors on the real in-place maintainer and
prints recall@k after every operation, until Ctrl+C.  All logic lives in
``vecstudy.harness.live`` (so it is unit-testable); this file only pins the BLAS
thread count for determinism, makes the repo importable when run from a checkout,
and forwards to ``main``.

Run from the repository root:

    .venv/bin/python scripts/live_churn.py

Configurable example:

    .venv/bin/python scripts/live_churn.py \
        --initial-size 300 --k 10 --query-count 32 --depth 2 \
        --split-threshold 40 --delta 0.3 --interval 0.2 --seed 7
"""

import os
import sys
from pathlib import Path

# Pin single-threaded BLAS before numpy is imported (determinism + parity with
# the test suite and the harness runners).
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("NUMEXPR_NUM_THREADS", "1")

# Allow running as a plain script from a checkout that is not pip-installed.
_REPO_ROOT = Path(__file__).resolve().parent.parent
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from vecstudy.harness.live import main  # noqa: E402

if __name__ == "__main__":
    raise SystemExit(main())
