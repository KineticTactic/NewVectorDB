"""test_query.py — Stage 4 checkpoint: invariant 10 (exhaustive recall = 1.0)."""

import numpy as np
import pytest

from vecstudy.core.config import IndexConfig, QueryConfig, MaintainConfig
from vecstudy.core.metrics import Counters
from vecstudy.harness.datasets import compute_ground_truth, recall_at_k
from vecstudy.index.segment import Segment
from vecstudy.index.query import search


def _build_segment(X, leaf_size=50, seed=0):
    N, D = X.shape
    cfg = IndexConfig(dim=D, leaf_size=leaf_size, branching_factor=4, seed=seed)
    c = Counters()
    vids = np.arange(N, dtype=np.int64)
    return Segment.build(vids, X, cfg, c), c


def test_exhaustive_recall_1(synthetic_200):
    """Invariant 10: n_postings = n_leaves with full beam gives recall@10 = 1.0."""
    X = synthetic_200
    seg, _ = _build_segment(X, leaf_size=20, seed=0)
    n_leaves = len(seg.postings)
    cfg_q = QueryConfig(k=10, beam_widths=(n_leaves,), n_postings=n_leaves)

    queries = X[:20]
    gt = compute_ground_truth(queries, X, k=10, dataset_key="test_exhaustive")

    results = []
    for q in queries:
        c = Counters()
        res = search(q, [seg], cfg_q, c)
        results.append([v for v, _ in res])

    r = recall_at_k(results, gt, k=10)
    assert r == 1.0, f"Exhaustive search recall = {r:.4f}, expected 1.0"


def test_probe_monotonicity(synthetic_200):
    """Invariant 5: recall@10 is non-decreasing in n_postings."""
    X = synthetic_200
    seg, _ = _build_segment(X, leaf_size=20, seed=0)
    n_leaves = len(seg.postings)
    queries = X[:15]
    gt = compute_ground_truth(queries, X, k=10, dataset_key="test_probe_mono")

    prev_recall = 0.0
    for n_post in [1, 2, 4, n_leaves]:
        cfg_q = QueryConfig(k=10, beam_widths=(n_leaves,), n_postings=n_post)
        results = []
        for q in queries:
            c = Counters()
            res = search(q, [seg], cfg_q, c)
            results.append([v for v, _ in res])
        r = recall_at_k(results, gt, k=10)
        assert r >= prev_recall - 1e-9, \
            f"Recall dropped: n_postings={n_post} recall={r:.4f} < prev={prev_recall:.4f}"
        prev_recall = r


def test_tombstone_invisible(synthetic_200):
    """Invariant 8: deleted vectors never appear in results."""
    X = synthetic_200
    seg, _ = _build_segment(X, leaf_size=30, seed=0)
    n_leaves = len(seg.postings)

    deleted = {0, 1, 2, 5}
    for d in deleted:
        seg.mark_deleted(d)

    cfg_q = QueryConfig(k=10, beam_widths=(n_leaves,), n_postings=n_leaves)
    for q in X[10:20]:
        c = Counters()
        res = search(q, [seg], cfg_q, c)
        result_vids = {v for v, _ in res}
        assert result_vids.isdisjoint(deleted), \
            f"Deleted vectors appeared in results: {result_vids & deleted}"


def test_scan_bucket_incremented(synthetic_200):
    """search must increment 'scan' counter."""
    X = synthetic_200
    seg, _ = _build_segment(X, leaf_size=30, seed=0)
    n_leaves = len(seg.postings)
    cfg_q = QueryConfig(k=10, beam_widths=(n_leaves,), n_postings=1)
    c = Counters()
    search(X[0], [seg], cfg_q, c)
    assert c.distance_computations.get("scan", 0) > 0


def test_zero_candidate_unit_ok(synthetic_200):
    """A unit contributing zero candidates should not raise."""
    X = synthetic_200
    seg, _ = _build_segment(X[:10], leaf_size=5, seed=0)
    # Mark all vectors deleted
    for v in range(10):
        seg.mark_deleted(v)
    n_leaves = len(seg.postings)
    cfg_q = QueryConfig(k=5, beam_widths=(n_leaves,), n_postings=n_leaves)
    c = Counters()
    res = search(X[0], [seg], cfg_q, c)
    assert res == []
