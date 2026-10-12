"""test_churn.py — Comprehensive tests for the churn harness and fairness requirements."""

from __future__ import annotations

import tempfile
from pathlib import Path
import numpy as np
import pytest
import pyarrow.parquet as pq

from vecstudy.core.config import IndexConfig, QueryConfig, MaintainConfig, ExperimentConfig
from vecstudy.core.metrics import Counters
from vecstudy.core.types import Posting
from vecstudy.index.segment import Segment
from vecstudy.index.query import search
from vecstudy.maintain.inplace import InPlaceMaintainer
from vecstudy.maintain.outofplace import OutOfPlaceMaintainer
from vecstudy.harness.datasets import make_synthetic, recall_at_k
from vecstudy.harness.churn import (
    generate_churn_trace, compute_dynamic_ground_truth, audit_maintainer,
)
from vecstudy.harness.runner import run_churn_experiment, run_experiment


# ---------------------------------------------------------------------------
# Test 1: Trace determinism
# ---------------------------------------------------------------------------

def test_trace_determinism():
    """Same seed produces identical operations and trace hash; different seeds differ."""
    t1 = generate_churn_trace(initial_n=200, reserve_n=200, n_operations=100, seed=42)
    t2 = generate_churn_trace(initial_n=200, reserve_n=200, n_operations=100, seed=42)
    t3 = generate_churn_trace(initial_n=200, reserve_n=200, n_operations=100, seed=43)

    assert t1.operations == t2.operations
    assert t1.trace_hash == t2.trace_hash
    assert t1.trace_hash != t3.trace_hash

    # Validate operation semantics:
    # - Inserts use IDs >= initial_n and < initial_n + reserve_n
    # - No ID is inserted more than once
    # - Deletes only target currently live IDs
    live = set(range(200))
    inserted_ids = set()

    for op, vid in t1.operations:
        if op == "insert":
            assert vid >= 200 and vid < 400
            assert vid not in inserted_ids, f"ID {vid} inserted twice"
            assert vid not in live
            inserted_ids.add(vid)
            live.add(vid)
        elif op == "delete":
            assert vid in live, f"ID {vid} deleted while not live"
            live.remove(vid)
        else:
            pytest.fail(f"Unknown op {op}")


# ---------------------------------------------------------------------------
# Test 2: Same initial state
# ---------------------------------------------------------------------------

def test_same_initial_state(synthetic_200):
    """Both arms begin with identical live IDs, index structure, and search results at checkpoint zero."""
    X = synthetic_200
    D = X.shape[1]
    cfg_idx = IndexConfig(dim=D, leaf_size=40, branching_factor=4, seed=0, depth=2)
    cfg_m_ip = MaintainConfig(strategy="inplace", split_threshold=80)
    cfg_m_oop = MaintainConfig(strategy="outofplace", memtable_limit=50, compaction_threshold=4)

    initial_vids = np.arange(len(X), dtype=np.int64)
    base_seg = Segment.build(initial_vids, X, cfg_idx, Counters())

    m_ip = InPlaceMaintainer(X, cfg_idx, cfg_m_ip, Counters(), initial_segment=base_seg.clone())
    m_oop = OutOfPlaceMaintainer(X, cfg_idx, cfg_m_oop, Counters(), initial_segment=base_seg.clone())

    # Live sets
    live_ip = {int(v) for u in m_ip.searchable_units() for p in u.postings for v in p.owned if u.is_alive(int(v))}
    live_oop = {int(v) for u in m_oop.searchable_units() for p in u.postings for v in p.owned if u.is_alive(int(v))}
    assert live_ip == live_oop == set(range(len(X)))

    # Structure comparison
    u_ip = m_ip.searchable_units()[0]
    u_oop = m_oop.searchable_units()[0]
    assert len(u_ip.levels) == len(u_oop.levels) == 2
    assert len(u_ip.postings) == len(u_oop.postings)
    np.testing.assert_allclose(u_ip.levels[-1].centroids, u_oop.levels[-1].centroids)

    # State independence (not sharing mutable references)
    u_ip.levels[-1].centroids[0, 0] += 999.0
    assert u_oop.levels[-1].centroids[0, 0] != u_ip.levels[-1].centroids[0, 0]


# ---------------------------------------------------------------------------
# Test 3: Same workload replay
# ---------------------------------------------------------------------------

def test_same_workload():
    """Both arms receive identical operations in exact order; live sets match trace model."""
    vectors = make_synthetic(n=500, dim=32, seed=0)
    cfg_idx = IndexConfig(dim=32, leaf_size=40, branching_factor=4, seed=0, depth=2)
    cfg_m_ip = MaintainConfig(strategy="inplace", split_threshold=60, n_reassign_neighbours=8)
    cfg_m_oop = MaintainConfig(strategy="outofplace", memtable_limit=30, compaction_threshold=3)

    initial_vids = np.arange(300, dtype=np.int64)
    base_seg = Segment.build(initial_vids, vectors, cfg_idx, Counters())

    m_ip = InPlaceMaintainer(vectors, cfg_idx, cfg_m_ip, Counters(), initial_segment=base_seg.clone())
    m_oop = OutOfPlaceMaintainer(vectors, cfg_idx, cfg_m_oop, Counters(), initial_segment=base_seg.clone())

    trace = generate_churn_trace(initial_n=300, reserve_n=200, n_operations=100, seed=42)

    model_live: set[int] = set(range(300))

    for op, vid in trace.operations:
        if op == "insert":
            m_ip.insert(vid, vectors[vid])
            m_ip.maintain()
            m_oop.insert(vid, vectors[vid])
            m_oop.maintain()
            model_live.add(vid)
        elif op == "delete":
            m_ip.delete(vid)
            m_ip.maintain()
            m_oop.delete(vid)
            m_oop.maintain()
            model_live.remove(vid)

    # In-place live set
    live_ip = {int(v) for u in m_ip.searchable_units() for p in u.postings for v in p.owned if u.is_alive(int(v))}
    # Out-of-place live set (sealed + memtable)
    live_oop = set()
    for seg in m_oop._sealed:
        for p in seg.postings:
            for v in p.owned:
                vi = int(v)
                if seg.is_alive(vi) and vi not in m_oop._tombstones:
                    live_oop.add(vi)
    for v in m_oop._memtable:
        vi = int(v)
        if vi not in m_oop._tombstones:
            live_oop.add(vi)

    assert live_ip == model_live
    assert live_oop == model_live


# ---------------------------------------------------------------------------
# Test 4: Dynamic ground truth
# ---------------------------------------------------------------------------

def test_dynamic_ground_truth(tmp_path):
    """Ground truth strictly reflects current live vectors and maps local offsets to stable global IDs."""
    vectors = make_synthetic(n=100, dim=16, seed=10)
    queries = make_synthetic(n=5, dim=16, seed=20)

    # Non-contiguous live set (e.g. 10, 25, 42, 67, 88)
    live_vids = np.array([10, 25, 42, 67, 88], dtype=np.int64)
    gt = compute_dynamic_ground_truth(queries, vectors, live_vids, k=3, cache_dir=tmp_path)

    # All returned IDs must be in live_vids
    for row in gt:
        assert set(row) <= set(live_vids)
        # Check that local row offsets (0..4) are NOT mistakenly returned
        # (e.g. 0, 1, 2 are not in live_vids)
        assert 0 not in set(row)
        assert 1 not in set(row)

    # Make vector 99 identical to query 0; insert 99 into live_vids
    vectors[99] = queries[0]
    live_vids_updated = np.array([10, 25, 42, 67, 88, 99], dtype=np.int64)
    gt_updated = compute_dynamic_ground_truth(queries, vectors, live_vids_updated, k=3, cache_dir=tmp_path)

    # Vector 99 must be top-1 for query 0
    assert gt_updated[0, 0] == 99


# ---------------------------------------------------------------------------
# Test 5: MemTable visibility
# ---------------------------------------------------------------------------

def test_memtable_visibility():
    """Live inserts trapped in MemTable are searchable at checkpoints; public flush is counted."""
    vectors = make_synthetic(n=100, dim=16, seed=0)
    query = vectors[42].copy()  # Query will target vector 42

    cfg_idx = IndexConfig(dim=16, leaf_size=20, branching_factor=2, seed=0, depth=1)
    cfg_m = MaintainConfig(strategy="outofplace", memtable_limit=100, compaction_threshold=4)
    counters = Counters()

    # Initial segment with 40 vectors
    base_seg = Segment.build(np.arange(40, dtype=np.int64), vectors, cfg_idx, Counters())
    maint = OutOfPlaceMaintainer(vectors, cfg_idx, cfg_m, counters, initial_segment=base_seg.clone())

    # Insert vector 42 (goes to MemTable, memtable_limit=100 so it will NOT auto-flush)
    maint.insert(42, vectors[42])
    assert 42 in maint._memtable
    assert len(maint._sealed) == 1

    # Search should find vector 42 because searchable_units() exposes MemTable as flat segment
    units = maint.searchable_units()
    assert len(units) == 2  # sealed + flat
    cfg_q = QueryConfig(k=1, beam_widths=(2,), n_postings=1)
    res = search(query, units, cfg_q, Counters())
    assert len(res) > 0
    assert res[0][0] == 42, "Vector in MemTable must be visible to search"

    # Test public flush
    assert counters.flushes == 0
    maint.flush()
    assert counters.flushes == 1
    assert len(maint._memtable) == 0
    assert len(maint._sealed) == 2

    # Still searchable after flush
    res_after = search(query, maint.searchable_units(), cfg_q, Counters())
    assert res_after[0][0] == 42


# ---------------------------------------------------------------------------
# Test 6: NPA and structural correctness
# ---------------------------------------------------------------------------

def test_npa_auditor_catches_deliberate_misassignment(synthetic_200):
    """The global NPA auditor must catch a deliberately misassigned vector."""
    X = synthetic_200
    D = X.shape[1]
    cfg_idx = IndexConfig(dim=D, leaf_size=40, branching_factor=4, seed=0, depth=1)
    cfg_m = MaintainConfig(strategy="inplace", split_threshold=80)
    maint = InPlaceMaintainer(X, cfg_idx, cfg_m, Counters(), initial_vids=np.arange(len(X), dtype=np.int64))

    # Audit initial clean state
    audit_clean = audit_maintainer(maint, expected_live_vids=set(range(len(X))))
    assert audit_clean.passed is True
    assert len(audit_clean.npa_violations) == 0

    # Deliberately move a vector from posting 0 into posting 1
    seg = maint._segment
    p0 = seg.postings[0]
    p1 = seg.postings[1]
    victim_vid = p0.owned[0]

    # Reassign victim_vid to p1
    new_p0_owned = p0.owned[1:]
    new_p1_owned = np.append(p1.owned, victim_vid)
    new_p1_members = np.append(p1.members, victim_vid)

    seg.postings[0] = Posting(p0.id, p0.centroid, new_p0_owned, new_p0_owned)
    seg.postings[1] = Posting(p1.id, p1.centroid, new_p1_members, new_p1_owned)

    audit_dirty = audit_maintainer(maint, expected_live_vids=set(range(len(X))))
    assert audit_dirty.passed is False
    assert len(audit_dirty.npa_violations) > 0
    assert any(v["vid"] == victim_vid for v in audit_dirty.npa_violations)


def test_split_neighboring_postings_reassignment():
    """Splits reassign neighboring postings and maintain global NPA."""
    X = make_synthetic(n=500, dim=32, seed=42)
    cfg_idx = IndexConfig(dim=32, leaf_size=30, branching_factor=4, seed=42, depth=2)
    cfg_m = MaintainConfig(strategy="inplace", split_threshold=50, n_reassign_neighbours=8)
    c = Counters()
    maint = InPlaceMaintainer(X, cfg_idx, cfg_m, c, initial_vids=np.arange(200, dtype=np.int64))

    # Insert remaining vectors causing multiple splits
    for vid in range(200, 500):
        maint.insert(vid, X[vid])

    assert c.splits > 0
    assert c.vectors_reassigned > 0

    audit = audit_maintainer(maint, expected_live_vids=set(range(500)))
    assert audit.passed is True, f"NPA violations after splits: {audit.npa_violations[:5]}"


# ---------------------------------------------------------------------------
# Test 7: Result output schema and values
# ---------------------------------------------------------------------------

def test_result_output(tmp_path):
    """Harness emits one row per checkpoint with all required metrics."""
    vectors = make_synthetic(n=400, dim=32, seed=0)
    queries = make_synthetic(n=10, dim=32, seed=1)

    cfg_idx = IndexConfig(dim=32, leaf_size=50, branching_factor=4, seed=0, depth=2)
    cfg_q = QueryConfig(k=5, beam_widths=(4,), n_postings=1)
    cfg_m = MaintainConfig(strategy="inplace", split_threshold=80)

    trace = generate_churn_trace(initial_n=200, reserve_n=200, n_operations=50, seed=42)

    rows = run_churn_experiment(
        dataset_name="synthetic",
        cfg_idx=cfg_idx,
        cfg_q=cfg_q,
        cfg_m=cfg_m,
        initial_size=200,
        reserve_size=200,
        trace=trace,
        checkpoint_interval=25,
        out_dir=tmp_path,
        vectors=vectors,
        queries=queries,
    )

    # Checkpoints: 0, 25, 50 => 3 rows
    assert len(rows) == 3

    required_keys = [
        "strategy", "initial_depth", "checkpoint", "operations_processed",
        "cumulative_insertions", "cumulative_deletions", "live_vector_count",
        "recall", "trace_hash", "correctness_status", "npa_violations",
        "splits", "delta_splits", "flushes", "delta_flushes",
        "dist_routing", "dist_scan", "postings_read_estimated",
    ]
    for row in rows:
        for key in required_keys:
            assert key in row, f"Missing required key: {key}"
        assert row["correctness_status"] == "PASSED"
        assert row["trace_hash"] == trace.trace_hash

    # Verify parquet file written
    pq_path = tmp_path / "results.parquet"
    assert pq_path.exists()
    tbl = pq.read_table(str(pq_path))
    assert tbl.num_rows == 3


# ---------------------------------------------------------------------------
# Test 8: Backward compatibility
# ---------------------------------------------------------------------------

def test_backward_compatibility(tmp_path):
    """Existing static run_experiment works as before."""
    cfg = ExperimentConfig(
        index=IndexConfig(dim=0, branching_factor=4, depth=1, seed=0),
        query=QueryConfig(k=5, beam_widths=(4,), n_postings=1),
        maintain=MaintainConfig(strategy="inplace"),
        dataset="synthetic",
    )
    row = run_experiment(cfg, tmp_path)
    assert "recall" in row
    assert "bloat" in row
    assert (tmp_path / "results.parquet").exists()

