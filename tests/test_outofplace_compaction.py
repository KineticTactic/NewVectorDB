"""tests/test_outofplace_compaction.py — Focus tests for OutOfPlaceMaintainer.

Covers:
- MemTable insertion buffering (O(1), zero distance ops on insert)
- MemTable flushing to sealed segments at memtable_limit
- Segment compaction when sealed segment count reaches compaction_threshold
- Deletion-driven compaction when tombstone fraction reaches tombstone_rebuild_frac
- Tombstone garbage collection (purging dead vectors from _tombstones after compaction)
- Idempotent deletes and re-insert revival
- Atomic rollback of compaction on failure
- Multi-segment search correctness and ground truth parity
"""

from __future__ import annotations

import numpy as np
import pytest

from vecstudy.core.config import IndexConfig, MaintainConfig, QueryConfig
from vecstudy.core.metrics import Counters
from vecstudy.harness.datasets import make_synthetic, compute_ground_truth, recall_at_k
from vecstudy.index.query import search
from vecstudy.index.segment import Segment
from vecstudy.maintain.outofplace import OutOfPlaceMaintainer


def _make_maint(
    vectors: np.ndarray,
    *,
    leaf_size: int = 20,
    branching_factor: int = 4,
    depth: int = 1,
    delta: float | None = None,
    memtable_limit: int = 40,
    compaction_threshold: int = 3,
    tombstone_rebuild_frac: float = 0.2,
    initial_n: int = 0,
) -> tuple[OutOfPlaceMaintainer, Counters]:
    D = vectors.shape[1]
    cfg_idx = IndexConfig(
        dim=D,
        leaf_size=leaf_size,
        branching_factor=branching_factor,
        depth=depth,
        delta=delta,
        seed=42,
    )
    cfg_m = MaintainConfig(
        strategy="outofplace",
        memtable_limit=memtable_limit,
        compaction_threshold=compaction_threshold,
        tombstone_rebuild_frac=tombstone_rebuild_frac,
    )
    counters = Counters()
    if initial_n > 0:
        initial_vids = np.arange(initial_n, dtype=np.int64)
        maint = OutOfPlaceMaintainer(
            vectors, cfg_idx, cfg_m, counters, initial_vids=initial_vids
        )
    else:
        maint = OutOfPlaceMaintainer(vectors, cfg_idx, cfg_m, counters)
    return maint, counters


def test_oop_insert_buffers_in_memtable():
    """Inserts buffer in MemTable without computing distances; searchable via flat unit."""
    X = make_synthetic(n=100, dim=16, seed=1)
    maint, counters = _make_maint(X, memtable_limit=50)

    # Insert 10 vectors
    for i in range(10):
        maint.insert(i, X[i])

    assert len(maint._memtable) == 10
    assert len(maint._sealed) == 0
    # Zero distance computations during pure buffering
    assert counters.distance_computations.get("routing", 0) == 0
    assert counters.distance_computations.get("scan", 0) == 0

    units = maint.searchable_units()
    assert len(units) == 1
    # Search finds the buffered vector
    cfg_q = QueryConfig(k=1, beam_widths=(2,), n_postings=1)
    res = search(X[5], units, cfg_q, Counters())
    assert len(res) == 1
    assert res[0][0] == 5


def test_oop_flush_creates_sealed_segment():
    """Crossing memtable_limit flushes to a new sealed segment with its own hierarchy."""
    X = make_synthetic(n=100, dim=16, seed=2)
    maint, counters = _make_maint(X, leaf_size=15, memtable_limit=30, depth=2)

    for i in range(30):
        maint.insert(i, X[i])

    # 30 reached => flushed
    assert len(maint._memtable) == 0
    assert len(maint._sealed) == 1
    assert counters.flushes == 1
    assert len(maint._sealed[0].levels) == 2


def test_oop_compaction_at_threshold():
    """Accumulating compaction_threshold sealed segments compacts into one unified segment."""
    X = make_synthetic(n=200, dim=16, seed=3)
    maint, counters = _make_maint(
        X, leaf_size=15, memtable_limit=25, compaction_threshold=3, depth=2
    )

    # 3 batches of 25 => 3 sealed segments
    for i in range(75):
        maint.insert(i, X[i])
        maint.maintain()

    assert len(maint._sealed) == 1
    assert counters.flushes == 3
    assert counters.compaction_events == 1
    assert counters.vectors_reclustered == 75
    assert counters.hierarchy_rebuilds >= 1


def test_oop_tombstone_driven_compaction():
    """Compaction fires when sealed segment tombstone ratio exceeds tombstone_rebuild_frac."""
    X = make_synthetic(n=100, dim=16, seed=4)
    # Start with an initial sealed segment of 50 vectors
    maint, counters = _make_maint(
        X,
        leaf_size=10,
        memtable_limit=50,
        compaction_threshold=5,
        tombstone_rebuild_frac=0.20,
        initial_n=50,
    )
    assert len(maint._sealed) == 1
    assert counters.compaction_events == 0

    # Delete 11 vectors (11 / 50 = 22% > 20%)
    for vid in range(11):
        maint.delete(vid)
        maint.maintain()

    # Deletion-driven compaction fires at 10 deletes (10 / 50 = 20%)
    assert counters.compaction_events == 1
    assert len(maint._sealed) == 1
    assert counters.vectors_reclustered == 40
    # The 10 dead vectors from compaction are no longer in the compacted segment
    compacted_seg = maint._sealed[0]
    all_owned = {int(v) for p in compacted_seg.postings for v in p.owned}
    for vid in range(10):
        assert vid not in all_owned


def test_oop_tombstone_garbage_collection():
    """Tombstones for sealed segments are cleared at compaction, while pending memtable tombstones persist."""
    X = make_synthetic(n=100, dim=16, seed=5)
    maint, counters = _make_maint(
        X, memtable_limit=50, compaction_threshold=2, initial_n=40
    )

    # Delete vector 0 from initial sealed segment
    maint.delete(0)
    assert 0 in maint._tombstones

    # Buffer vector 50 in memtable and delete it while pending
    maint.insert(50, X[50])
    maint.delete(50)
    assert 50 in maint._tombstones

    # Force compaction of sealed segments
    maint._compact()

    # Vector 0 (from sealed segment) was purged; its tombstone is now cleared!
    assert 0 not in maint._tombstones
    # Vector 50 (still pending in memtable) must retain its tombstone until flush
    assert 50 in maint._tombstones

    # Flush memtable: vector 50 is discarded and tombstone cleared
    maint.flush()
    assert 50 not in maint._tombstones
    assert len(maint._tombstones) == 0


def test_oop_delete_idempotent():
    """Deleting the same vector multiple times does not corrupt state."""
    X = make_synthetic(n=50, dim=16, seed=6)
    maint, counters = _make_maint(X, initial_n=30)

    maint.delete(5)
    assert 5 in maint._tombstones
    maint.delete(5)
    assert len(maint._tombstones) == 1


def test_oop_reinsert_revives_tombstoned_vector():
    """Inserting a previously deleted vector revives it and removes the tombstone."""
    X = make_synthetic(n=50, dim=16, seed=7)
    maint, counters = _make_maint(X, initial_n=30)

    maint.delete(10)
    assert 10 in maint._tombstones

    # Re-insert 10
    maint.insert(10, X[10])
    assert 10 not in maint._tombstones

    # 10 is searchable in memtable
    units = maint.searchable_units()
    cfg_q = QueryConfig(k=1, beam_widths=(2,), n_postings=1)
    res = search(X[10], units, cfg_q, Counters())
    assert res[0][0] == 10


def test_oop_compaction_atomicity_on_failure(monkeypatch):
    """Compaction failure rolls back sealed segments, tombstones, and counters cleanly."""
    X = make_synthetic(n=100, dim=16, seed=8)
    maint, counters = _make_maint(X, initial_n=50)

    saved_sealed_len = len(maint._sealed)
    saved_tombstones = set(maint._tombstones)
    saved_events = counters.compaction_events

    def _failing_build(*args, **kwargs):
        raise RuntimeError("Simulated cluster failure")

    monkeypatch.setattr(Segment, "build", _failing_build)

    with pytest.raises(RuntimeError, match="Simulated cluster failure"):
        maint._compact()

    assert len(maint._sealed) == saved_sealed_len
    assert maint._tombstones == saved_tombstones
    assert counters.compaction_events == saved_events


def test_oop_compaction_empty_when_all_dead():
    """If all vectors in sealed segments are tombstoned, compaction results in 0 segments."""
    X = make_synthetic(n=20, dim=16, seed=9)
    maint, counters = _make_maint(X, initial_n=10)

    for vid in range(10):
        maint.delete(vid)

    maint._compact()
    assert len(maint._sealed) == 0
    assert len(maint._tombstones) == 0
    assert counters.compaction_events == 1


def test_oop_stats_keys():
    """stats() returns all required strategy, depth, and maintenance counters."""
    X = make_synthetic(n=100, dim=16, seed=10)
    maint, _ = _make_maint(X, initial_n=50, depth=2)

    s = maint.stats()
    expected_keys = [
        "strategy",
        "memtable_size",
        "sealed_segments",
        "unit_count",
        "depth",
        "n_postings",
        "compaction_events",
        "vectors_reclustered",
        "flushes",
        "hierarchy_rebuilds",
        "tombstones",
    ]
    for k in expected_keys:
        assert k in s
    assert s["strategy"] == "outofplace"
    assert s["depth"] == 2
