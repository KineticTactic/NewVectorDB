#!/usr/bin/env python
"""plot_structure.py — render the single structural diagnostics figure.

Reads ``results_structure_checkpoints.csv`` from the equal-search-work harness
and writes ``structure_threshold_series.csv`` plus one figure (posting-size
p50/p90/p95/max vs ``split_threshold``, one panel per strategy).

Usage:
    python scripts/plot_structure.py --input <run_dir> [--out <dir>]
                                     [--no-figures]
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from vecstudy.harness.structure_report import plot_structure_diagnostics  # noqa: E402


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(prog="plot_structure",
                                description=__doc__)
    p.add_argument("--input", required=True,
                   help="run directory or results_structure_checkpoints.csv")
    p.add_argument("--out", default=None,
                   help="output directory (default: alongside the input)")
    p.add_argument("--no-figures", action="store_true",
                   help="write only the threshold-series CSV")
    return p


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    src = Path(args.input)
    if src.is_dir():
        src = src / "results_structure_checkpoints.csv"
    if not src.exists():
        print(f"error: {src} not found", file=sys.stderr)
        return 2
    out = Path(args.out) if args.out else src.parent / "structure"
    res = plot_structure_diagnostics(src, out, make_figures=not args.no_figures)
    print(f"wrote {len(res['series'])} threshold-series rows to {out}")
    print("files: " + ", ".join(res["written"]))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
