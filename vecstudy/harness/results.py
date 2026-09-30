"""harness/results.py — result storage (Parquet via pyarrow).

Each result row = full config + all counters + recall + bloat +
posting-size distribution + depth + unit count + git SHA + timestamp +
dataset checksum.

Git SHA degrades to "nogit" if not in a repository (§Q2 decision).
"""

from __future__ import annotations

import datetime
import subprocess
from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq


def _git_sha() -> str:
    try:
        result = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            capture_output=True, text=True, timeout=5
        )
        if result.returncode == 0:
            return result.stdout.strip()
    except Exception:
        pass
    return "nogit"


def make_result_row(
    *,
    config_dict: dict,
    counters_dict: dict,
    recall: float,
    bloat: float,
    posting_size_dist: dict,  # mean, p50, p99, max
    depth: int,
    unit_count: int,
    dataset_checksum: str,
    extra: dict | None = None,
) -> dict:
    """Assemble a flat result dict ready for Parquet serialisation."""
    row: dict = {}
    row.update(config_dict)
    row.update(counters_dict)
    row["recall"] = float(recall)
    row["bloat"] = float(bloat)
    row["posting_size_mean"] = float(posting_size_dist.get("mean", 0.0))
    row["posting_size_p50"]  = float(posting_size_dist.get("p50", 0.0))
    row["posting_size_p99"]  = float(posting_size_dist.get("p99", 0.0))
    row["posting_size_max"]  = int(posting_size_dist.get("max", 0))
    row["depth"] = int(depth)
    row["unit_count"] = int(unit_count)
    row["dataset_checksum"] = dataset_checksum
    row["git_sha"] = _git_sha()
    row["timestamp"] = datetime.datetime.utcnow().isoformat()
    if extra:
        row.update(extra)
    return row


def append_result(row: dict, out_dir: str | Path) -> Path:
    """Write a single result row to Parquet (appends or creates)."""
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / "results.parquet"

    table = _row_to_table(row)

    if out_path.exists():
        existing = pq.read_table(str(out_path))
        combined = pa.concat_tables([existing, table], promote_options="default")
        pq.write_table(combined, str(out_path))
    else:
        pq.write_table(table, str(out_path))

    return out_path


def _row_to_table(row: dict) -> pa.Table:
    """Convert a flat dict row to a single-row PyArrow table."""
    arrays = {}
    for k, v in row.items():
        if isinstance(v, bool):
            arrays[k] = pa.array([v], type=pa.bool_())
        elif isinstance(v, int):
            arrays[k] = pa.array([v], type=pa.int64())
        elif isinstance(v, float):
            arrays[k] = pa.array([v], type=pa.float64())
        elif isinstance(v, str):
            arrays[k] = pa.array([v], type=pa.string())
        elif isinstance(v, list):
            arrays[k] = pa.array([str(v)], type=pa.string())
        else:
            arrays[k] = pa.array([str(v)], type=pa.string())
    return pa.table(arrays)
