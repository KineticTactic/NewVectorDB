"""core/config.py — all configuration dataclasses.

Every config is frozen and can be serialised to a flat dict for result rows.
"""

from __future__ import annotations

from dataclasses import dataclass, asdict
from typing import Literal


@dataclass(frozen=True)
class IndexConfig:
    """Static index construction parameters."""
    dim: int
    leaf_size: int = 800            # target vectors per posting
    branching_factor: int = 16
    # None → derive depth from branching_factor via §6.3 loop
    # int  → build exactly this many levels (leaf counted as depth 1)
    depth: int | None = None
    delta: float | None = None      # None → no overlap; relative threshold §6.4
    rng_rule: bool = False          # SPANN relative-neighbourhood rule
    seed: int = 0

    def to_dict(self) -> dict:
        return asdict(self)


@dataclass(frozen=True)
class QueryConfig:
    """Per-query search parameters."""
    k: int = 10
    # per-level survivor counts; last value repeats if shorter than depth
    beam_widths: tuple[int, ...] = (8,)
    n_postings: int = 1             # postings fetched per searchable unit
    overfetch: float = 0.2         # k' = ceil(k * (1 + overfetch))
    global_beam: bool = False       # use descend_global instead of descend

    def to_dict(self) -> dict:
        d = asdict(self)
        d["beam_widths"] = list(self.beam_widths)  # JSON/Parquet friendly
        return d


@dataclass(frozen=True)
class MaintainConfig:
    """Index maintenance parameters."""
    strategy: Literal["inplace", "outofplace"]
    memtable_limit: int = 10_000       # out-of-place: flush threshold
    compaction_threshold: int = 4      # out-of-place: compact at m sealed segments
    split_threshold: int = 1_600       # in-place: 2 × leaf_size
    # In-place: legacy neighbourhood width. Split reassignment no longer ranks
    # postings by centroid proximity (that heuristic could miss a vector sitting
    # far from its own centroid); candidate selection is now decided by the
    # per-posting bounding radius. Kept so existing configs/sweeps still load.
    n_reassign_neighbours: int = 8
    tombstone_rebuild_frac: float = 0.2  # in-place: rebuild posting fraction
    # In-place posting-merge underflow trigger. A posting whose live canonical
    # owned count is strictly below this value becomes merge-eligible after a
    # delete. None → derive as max(1, split_threshold // 2), i.e. half the split
    # threshold, which keeps merges from immediately re-splitting. Replicas in
    # `members` never count toward occupancy.
    merge_threshold: int | None = None

    def to_dict(self) -> dict:
        return asdict(self)


@dataclass(frozen=True)
class ExperimentConfig:
    """Top-level experiment configuration (all three sub-configs combined)."""
    index: IndexConfig
    query: QueryConfig
    maintain: MaintainConfig
    dataset: str = "synthetic"         # dataset name / path key

    def to_dict(self) -> dict:
        return {
            **{f"index_{k}": v for k, v in self.index.to_dict().items()},
            **{f"query_{k}": v for k, v in self.query.to_dict().items()},
            **{f"maintain_{k}": v for k, v in self.maintain.to_dict().items()},
            "dataset": self.dataset,
        }
