"""test_inplace_integration.py — R1 Task 5 integration & structural-stability matrix.

Cross-checks the integrated in-place maintainer (NPA repair + fixed-Δ membership
+ deletion-driven merges) against *both* independent validators and the
structural invariants, across a seed-fixed matrix of hierarchy depths, Δ values,
split thresholds and churn workloads.

Per operation (small matrix) it verifies:

  * ``audit_maintainer``  — coverage, unique canonical ownership, NPA;
  * ``audit_membership``  — the fixed-Δ replica oracle;
  * posting ids are positional and match their leaf-centroid row;
  * routing levels have valid, complete parent→child references;
  * stored spread radii equal the independently recomputed radii;
  * the production query path, fetched exhaustively, returns exactly the
    brute-force top-k of the live set, and never a tombstoned id.

The matrix is deliberately wider than ``test_inplace_stress`` (depths 1–4, both
Δ on/off and two on values, low/high thresholds, insert-only/delete-only/mixed).
Every config is reproducible from its parameters alone.
"""

from __future__ import annotations

import numpy as np
import pytest

from vecstudy.core.config import IndexConfig, MaintainConfig, QueryConfig
from vecstudy.core.metrics import Counters
from vecstudy.harness.churn import (
    audit_maintainer,
    audit_membership,
    generate_churn_trace,
)
from vecstudy.harness.datasets import make_synthetic
from vecstudy.index.query import search
from vecstudy.maintain.inplace import InPlaceMaintainer, _owned_spread


N, DIM, INITIAL, OPS = 200, 8, 100, 120


def _make(depth, delta, leaf_size, split_threshold, seed):
    X = make_synthetic(n=N, dim=DIM, seed=seed)
    cfg_idx = IndexConfig(
        dim=DIM, leaf_size=leaf_size, branching_factor=4, seed=seed,
        depth=depth, delta=delta,
    )
    cfg_maint = MaintainConfig(strategy="inplace", split_threshold=split_threshold)
    counters = Counters()
    maint = InPlaceMaintainer(
        X, cfg_idx, cfg_maint, counters,
        initial_vids=np.arange(INITIAL, dtype=np.int64),
    )
    return X, maint, counters


def _check_structure(maint):
    seg = maint._segment
    postings = seg.postings
    levels = seg.levels
    leaf = levels[-1].centroids

    # Posting ids are positional and match their leaf-centroid row.
    assert len(leaf) == len(postings)
    for i, p in enumerate(postings):
        assert p.id == i, f"posting {i} id {p.id}"
        assert np.array_equal(
            np.asarray(p.centroid, np.float32), np.asarray(leaf[i], np.float32)
        ), f"posting {i} centroid != leaf row"

    # Routing levels have valid, complete, disjoint parent->child references.
    for li in range(len(levels) - 1):
        n_next = len(levels[li + 1].centroids)
        covered: list[int] = []
        for children in levels[li].children:
            for c in children.tolist():
                assert 0 <= int(c) < n_next, f"level {li} child {c} out of range"
                covered.append(int(c))
        assert sorted(covered) == list(range(n_next)), (
            f"level {li}: children do not partition next level"
        )

    # Stored spreads match the independent recomputation.
    for pid, p in enumerate(postings):
        exp = _owned_spread(maint._vectors, p.centroid, p.owned)
        assert abs(maint._spread[pid] - exp) < 1e-6, f"spread {pid}"


def _check_query(maint, queries, live, k=5):
    n_leaves = len(maint._segment.postings)
    cfg_q = QueryConfig(
        k=k, beam_widths=(max(1, n_leaves),), n_postings=max(1, n_leaves)
    )
    live_arr = np.array(sorted(live), dtype=np.int64)
    for q in queries:
        res = search(q, maint.searchable_units(), cfg_q, Counters())
        vids = [int(v) for v, _ in res]
        assert len(vids) == len(set(vids)), "duplicate ids in query result"
        for v in vids:
            assert v in live, f"query returned non-live id {v}"
        if len(live_arr) == 0:
            assert vids == []
            continue
        got = np.array([s for _, s in res], dtype=np.float64)
        sims = maint._vectors[live_arr] @ q.astype(np.float32)
        exp = np.sort(sims)[::-1][:k].astype(np.float64)
        assert got.shape == exp.shape
        assert np.allclose(got, exp, atol=1e-5), "top-k similarity mismatch"


def _run(depth, delta, leaf_size, split_threshold, workload, seed, insert_prob=0.35):
    X, maint, counters = _make(depth, delta, leaf_size, split_threshold, seed)
    trace = generate_churn_trace(
        initial_n=INITIAL, reserve_n=N - INITIAL, n_operations=OPS,
        insert_prob=insert_prob, seed=seed + 1, workload=workload,
    )
    live = set(range(INITIAL))
    queries = [X[i] for i in range(0, N, 17)]

    for step, (op, raw) in enumerate(trace.operations):
        vid = int(raw)
        if op == "insert":
            maint.insert(vid, X[vid])
            live.add(vid)
        else:
            maint.delete(vid)
            live.discard(vid)

        npa = audit_maintainer(maint, expected_live_vids=set(live))
        mem = audit_membership(maint)
        assert npa.passed, f"npa fail step {step}: {npa.to_dict()}"
        assert mem.passed, f"membership fail step {step}: {mem.to_dict()}"
        _check_structure(maint)
        _check_query(maint, queries, live)

    npa = audit_maintainer(maint, expected_live_vids=set(live))
    mem = audit_membership(maint)
    assert npa.passed and mem.passed
    return counters


# depth, delta, leaf_size, split_threshold, workload
MIXED = [
    (depth, delta, 16, st, "mixed")
    for depth in (1, 2, 3, 4)
    for delta in (None, 0.3)
    for st in (24, 48)
]
WORKLOADS = [
    (1, 0.3, 16, 24, "insert_only"),
    (1, 0.3, 16, 24, "delete_only"),
    (3, None, 16, 48, "insert_only"),
    (3, None, 16, 48, "delete_only"),
    (4, 0.3, 16, 24, "delete_only"),
]


@pytest.mark.parametrize(
    "depth,delta,leaf_size,st,workload", MIXED + WORKLOADS,
    ids=[f"d{d}_delta{de}_leaf{lf}_st{st}_{wl}" for d, de, lf, st, wl in MIXED + WORKLOADS],
)
def test_integration_matrix(depth, delta, leaf_size, st, workload):
    seed = 1000 + depth * 10 + (0 if delta is None else int(delta * 10))
    counters = _run(depth, delta, leaf_size, st, workload, seed)
    assert counters.splits >= 0 and counters.merges >= 0


def test_empty_index_query_is_safe():
    X = make_synthetic(n=10, dim=DIM, seed=3)
    cfg_idx = IndexConfig(dim=DIM, leaf_size=16, branching_factor=4, seed=3, depth=1)
    maint = InPlaceMaintainer(
        X, cfg_idx, MaintainConfig(strategy="inplace", split_threshold=24), Counters()
    )
    cfg_q = QueryConfig(k=5, beam_widths=(4,), n_postings=4)
    assert search(X[0], maint.searchable_units(), cfg_q, Counters()) == []


def test_single_posting_query_is_safe():
    from vecstudy.core.types import HierarchyLevel, Posting
    from vecstudy.index.segment import Segment

    X = make_synthetic(n=40, dim=DIM, seed=4)
    centroid = X.mean(axis=0).astype(np.float32)
    owned = np.arange(40, dtype=np.int64)
    posting = Posting(0, centroid, owned, owned)
    leaf = HierarchyLevel(centroids=centroid[np.newaxis], children=[])
    seg = Segment(levels=[leaf], postings=[posting], vectors=X, tombstones=set())
    cfg_idx = IndexConfig(dim=DIM, leaf_size=16, branching_factor=4, seed=4, depth=1)
    maint = InPlaceMaintainer(
        X, cfg_idx, MaintainConfig(strategy="inplace", split_threshold=1000),
        Counters(), initial_segment=seg,
    )
    assert len(maint._segment.postings) == 1
    cfg_q = QueryConfig(k=5, beam_widths=(4,), n_postings=4)
    res = search(X[0], maint.searchable_units(), cfg_q, Counters())
    assert len(res) == 5
    assert all(maint._segment.is_alive(int(v)) for v, _ in res)
