#!/usr/bin/env python
"""scripts/compare_maintenance.py — paired InPlace vs OutOfPlace benchmark.

Thin entry point.  Pins single-threaded BLAS before NumPy is imported, makes the
checkout importable, and forwards to ``vecstudy.harness.compare.main``.

Examples
--------
    .venv/bin/python scripts/compare_maintenance.py --help
    .venv/bin/python scripts/compare_maintenance.py --profile smoke
    .venv/bin/python scripts/compare_maintenance.py --profile quick
    .venv/bin/python scripts/compare_maintenance.py --profile research \
        --out results/compare-research

See ``R1_T8_Report.md`` for the launch commands, sample output, and the
measurement limitations.
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

from vecstudy.harness.compare import main  # noqa: E402

if __name__ == "__main__":
    raise SystemExit(main())
