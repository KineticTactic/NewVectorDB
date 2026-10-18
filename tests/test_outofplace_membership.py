"""tests/test_outofplace_membership.py — Fixed-Δ replica membership and ownership tests for OutOfPlaceMaintainer.

Covers:
- Independent audit_membership support for OutOfPlaceMaintainer
- Cross-segment duplicate canonical ownership detection by audit_maintainer
- Fixed-Δ replica assignment across multiple sealed segments
- Dead vectors excluded from membership across sealed segments and MemTable
- Injected membership faults detected by the independent oracle
"""

from __future__ import annotations

import numpy as np
import pytest

from vecstudy.core.config import IndexConfig, MaintainConfig, QueryConfig
from vecstudy.core.metrics import Counters
from vecstudy.core.types import Posting
from vecstudy.harness.churn import (
    audit_maintainer,
    audit_membership,
    audit_outofplace_membership,
)
from vecstudy.harness.datasets import make_synthetic
from vecstudy.index.segment import Segment
from vecstudy.maintain.outofplace import OutOfPlaceMaintainer


def _unit(angle_deg: float) -> np.ndarray:
    rad = np.deg2rad(angle_deg)
    return np.array([np.cos(rad), np.sin(rad)], dtype=np.float32)


def test_oop_audit_membership_clean_segments():
    """audit_membership succeeds on a clean OutOfPlaceMaintainer with multiple sealed segments."""
    X = make_synthetic(n=120, dim=16, seed=10)
    cfg_idx = IndexConfig(
        dim=16, leaf_size=15, branching_factor=4, seed=0, depth=1, delta=0.5
    )
    cfg_m = MaintainConfig(strategy="outofplace", memtable_limit=30, compaction_threshold=5)
    counters = Counters()

    maint = OutOfPlaceMaintainer(X, cfg_idx, cfg_m, counters)
    # Insert 60 vectors => 2 sealed segments of 30 vectors
    for i in range(60):
        maint.insert(i, X[i])

    assert len(maint._sealed) == 2
    assert len(maint._memtable) == 0

    audit_res = audit_outofplace_membership(maint)
    assert audit_res.passed is True, audit_res.summary()
    assert audit_res.live_vectors_checked == 60
    assert audit_res.violation_count == 0


def test_oop_audit_membership_with_memtable():
    """audit_membership correctly accounts for live vectors in MemTable without replicas."""
    X = make_synthetic(n=80, dim=16, seed=11)
    cfg_idx = IndexConfig(
        dim=16, leaf_size=15, branching_factor=4, seed=0, depth=1, delta=0.4
    )
    cfg_m = MaintainConfig(strategy="outofplace", memtable_limit=30, compaction_threshold=5)
    counters = Counters()

    maint = OutOfPlaceMaintainer(X, cfg_idx, cfg_m, counters)
    # Insert 45 vectors => 1 sealed segment (30) + 15 in MemTable
    for i in range(45):
        maint.insert(i, X[i])

    assert len(maint._sealed) == 1
    assert len(maint._memtable) == 15

    audit_res = audit_outofplace_membership(maint)
    assert audit_res.passed is True, audit_res.summary()
    assert audit_res.live_vectors_checked == 45


def test_oop_audit_maintainer_catches_cross_segment_duplicate_ownership():
    """audit_maintainer flags an ownership error if a vector is owned by two sealed segments."""
    X = make_synthetic(n=60, dim=16, seed=12)
    cfg_idx = IndexConfig(dim=16, leaf_size=15, branching_factor=4, seed=0, depth=1)
    cfg_m = MaintainConfig(strategy="outofplace", memtable_limit=20, compaction_threshold=5)
    counters = Counters()

    maint = OutOfPlaceMaintainer(X, cfg_idx, cfg_m, counters)
    for i in range(40):
        maint.insert(i, X[i])

    assert len(maint._sealed) == 2
    clean_audit = audit_maintainer(maint, expected_live_vids=set(range(40)))
    assert clean_audit.passed is True

    # Deliberately inject vector 0 as owned into sealed[1]'s posting 0
    seg1 = maint._sealed[1]
    p0 = seg1.postings[0]
    new_owned = np.append(p0.owned, 0)
    new_members = np.append(p0.members, 0)
    seg1.postings[0] = Posting(p0.id, p0.centroid, new_members, new_owned)

    dirty_audit = audit_maintainer(maint, expected_live_vids=set(range(40)))
    assert dirty_audit.passed is False
    assert any("multiple owners across units" in err for err in dirty_audit.ownership_errors)


def test_oop_audit_maintainer_catches_memtable_sealed_duplicate():
    """audit_maintainer flags duplicate ownership between sealed segment and MemTable."""
    X = make_synthetic(n=50, dim=16, seed=13)
    cfg_idx = IndexConfig(dim=16, leaf_size=15, branching_factor=4, seed=0, depth=1)
    cfg_m = MaintainConfig(strategy="outofplace", memtable_limit=30, compaction_threshold=5)
    counters = Counters()

    maint = OutOfPlaceMaintainer(X, cfg_idx, cfg_m, counters)
    # 30 flushed to sealed[0]
    for i in range(30):
        maint.insert(i, X[i])

    # Re-insert 0 into memtable without deleting it first
    maint._memtable.append(0)

    audit = audit_maintainer(maint)
    assert audit.passed is False
    assert any("multiple owners across units" in err for err in audit.ownership_errors)


def test_oop_oracle_detects_missing_replica():
    """The membership oracle detects if a required replica is omitted from a sealed segment."""
    X = make_synthetic(n=50, dim=16, seed=14)
    cfg_idx = IndexConfig(
        dim=16, leaf_size=10, branching_factor=4, seed=0, depth=1, delta=0.5
    )
    cfg_m = MaintainConfig(strategy="outofplace", memtable_limit=30)
    counters = Counters()

    maint = OutOfPlaceMaintainer(X, cfg_idx, cfg_m, counters)
    for i in range(30):
        maint.insert(i, X[i])

    seg = maint._sealed[0]
    # Find a posting with a replica (member but not owner)
    victim_vid = None
    victim_pid = None
    for p in seg.postings:
        replicas = set(p.members.tolist()) - set(p.owned.tolist())
        if replicas:
            victim_vid = next(iter(replicas))
            victim_pid = p.id
            break

    assert victim_vid is not None, "Need at least one replica in the fixture"

    # Remove the replica
    p = seg.postings[victim_pid]
    seg.postings[victim_pid] = Posting(
        p.id, p.centroid,
        np.array([v for v in p.members if v != victim_vid], dtype=np.int64),
        p.owned,
    )

    audit = audit_outofplace_membership(maint)
    assert audit.passed is False
    assert len(audit.missing) >= 1
    assert any(m["vid"] == victim_vid for m in audit.missing)


def test_oop_membership_persists_across_compaction():
    """Fixed-Δ membership holds after sealed segments are compacted."""
    X = make_synthetic(n=120, dim=16, seed=15)
    cfg_idx = IndexConfig(
        dim=16, leaf_size=12, branching_factor=4, seed=0, depth=2, delta=0.4
    )
    cfg_m = MaintainConfig(
        strategy="outofplace", memtable_limit=25, compaction_threshold=3
    )
    counters = Counters()

    maint = OutOfPlaceMaintainer(X, cfg_idx, cfg_m, counters)
    # 75 inserts => 3 flushes => compaction fires
    for i in range(75):
        maint.insert(i, X[i])
        maint.maintain()

    assert len(maint._sealed) == 1
    assert counters.compaction_events == 1

    audit_res = audit_outofplace_membership(maint)
    assert audit_res.passed is True, audit_res.summary()
    assert audit_res.live_vectors_checked == 75
    # Total members exceeds live vectors because delta > 0 produced replicas
    total_members = sum(len(p.members) for p in maint._sealed[0].postings)
    assert total_members > 75
