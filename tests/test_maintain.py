"""test_maintain.py — Stage 6/7 checkpoints: both maintainer strategies."""

from __future__ import annotations
import numpy as np
import pytest

from vecstudy.core.config import IndexConfig, QueryConfig, MaintainConfig
from vecstudy.core.metrics import Counters
from vecstudy.harness.datasets import compute_ground_truth, recall_at_k
from vecstudy.index.query import search
from vecstudy.maintain.outofplace import OutOfPlaceMaintainer
from vecstudy.maintain.inplace import InPlaceMaintainer


def _run_oop(X, leaf_size=40, memtable_limit=50, compaction_threshold=3):
    D = X.shape[1]
    cfg_idx = IndexConfig(dim=D, leaf_size=leaf_size, branching_factor=4, seed=0)
    cfg_m = MaintainConfig(strategy="outofplace",
                           memtable_limit=memtable_limit,
                           compaction_threshold=compaction_threshold)
    c = Counters()
    maint = OutOfPlaceMaintainer(X, cfg_idx, cfg_m, c)
    for vid in range(len(X)):
        maint.insert(vid, X[vid])
        maint.maintain()
    if maint._memtable:
        maint._flush()
    return maint, c


def _run_ip(X, leaf_size=40, split_threshold=80, use_initial=False):
    """Build in-place index.
    use_initial=True → bulk build (fast, no splits fired via insert).
    use_initial=False → insert one-by-one (splits triggered normally).
    """
    D = X.shape[1]
    cfg_idx = IndexConfig(dim=D, leaf_size=leaf_size, branching_factor=4, seed=0)
    cfg_m = MaintainConfig(strategy="inplace", split_threshold=split_threshold,
                           n_reassign_neighbours=8)
    c = Counters()
    if use_initial:
        vids = np.arange(len(X), dtype=np.int64)
        maint = InPlaceMaintainer(X, cfg_idx, cfg_m, c, initial_vids=vids)
    else:
        maint = InPlaceMaintainer(X, cfg_idx, cfg_m, c)
        for vid in range(len(X)):
            maint.insert(vid, X[vid])
    return maint, c


# ---------------------------------------------------------------------------
# Out-of-place
# ---------------------------------------------------------------------------

def test_oop_compaction_events(synthetic_200):
    """Compaction must fire when sealed segment count >= threshold."""
    maint, c = _run_oop(synthetic_200, memtable_limit=40, compaction_threshold=3)
    assert c.compaction_events >= 1, "Expected at least one compaction"


def test_oop_vectors_reclustered(synthetic_200):
    maint, c = _run_oop(synthetic_200, memtable_limit=40, compaction_threshold=3)
    assert c.vectors_reclustered > 0


def test_oop_recall_reasonable(synthetic_200):
    X = synthetic_200
    maint, _ = _run_oop(X, memtable_limit=40, compaction_threshold=3)
    units = maint.searchable_units()
    n_leaves = max(len(u.postings) for u in units)
    cfg_q = QueryConfig(k=10, beam_widths=(n_leaves,), n_postings=n_leaves)
    queries = X[:10]
    gt = compute_ground_truth(queries, X, k=10, dataset_key="oop_recall")
    results = [[v for v,_ in search(q, units, cfg_q, Counters())] for q in queries]
    rc = recall_at_k(results, gt, 10)
    assert rc >= 0.7, f"OOP recall {rc:.3f} surprisingly low"


def test_oop_tombstone_cleared_at_compaction(synthetic_200):
    X = synthetic_200
    maint, _ = _run_oop(X, memtable_limit=40, compaction_threshold=3)
    # Delete vector 0 and compact
    maint.delete(0)
    maint._compact()
    units = maint.searchable_units()
    n_leaves = max(len(u.postings) for u in units)
    cfg_q = QueryConfig(k=10, beam_widths=(n_leaves,), n_postings=n_leaves)
    for q in X[10:15]:
        res = search(q, units, cfg_q, Counters())
        assert 0 not in {v for v,_ in res}


# ---------------------------------------------------------------------------
# In-place
# ---------------------------------------------------------------------------

def test_ip_splits_occur(synthetic_1000):
    X = synthetic_1000
    maint, c = _run_ip(X, leaf_size=50, split_threshold=80, use_initial=False)
    assert c.splits > 0, "Expected splits with small split_threshold"


def test_ip_tombstone_delete(synthetic_200):
    X = synthetic_200
    maint, _ = _run_ip(X, leaf_size=40, split_threshold=80, use_initial=True)
    maint.delete(0)
    units = maint.searchable_units()
    n_leaves = len(units[0].postings)
    cfg_q = QueryConfig(k=10, beam_widths=(n_leaves,), n_postings=n_leaves)
    for q in X[10:15]:
        res = search(q, units, cfg_q, Counters())
        assert 0 not in {v for v,_ in res}


def test_ip_stats_keys(synthetic_200):
    maint, _ = _run_ip(synthetic_200, use_initial=True)
    stats = maint.stats()
    for key in ["splits", "vectors_reassigned", "depth", "n_postings"]:
        assert key in stats
