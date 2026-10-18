"""test_outofplace_integration.py — Integration & structural-stability matrix for OutOfPlaceMaintainer.

Cross-checks OutOfPlaceMaintainer against both independent validators (audit_maintainer,
audit_outofplace_membership) and structural invariants across a seed-fixed matrix of
hierarchy depths (1-4), Δ values (None, 0.3), compaction thresholds, and churn workloads.

Per operation it verifies:
  * audit_maintainer — coverage, unique canonical ownership across units, NPA per segment;
  * audit_outofplace_membership — fixed-Δ replica membership in sealed segments and flat MemTable;
  * structural invariants — valid levels, positional posting IDs, complete parent->child trees;
  * query invariants — exhaustive search matches brute force on currently live vectors,
    no dead vectors returned, no duplicates.
"""

from __future__ import annotations

import numpy as np
import pytest

from vecstudy.core.config import IndexConfig, MaintainConfig, QueryConfig
from vecstudy.core.metrics import Counters
from vecstudy.harness.churn import (
    audit_maintainer,
    audit_outofplace_membership,
    generate_churn_trace,
)
from vecstudy.harness.datasets import make_synthetic
from vecstudy.index.query import search
from vecstudy.maintain.outofplace import OutOfPlaceMaintainer

N, DIM, INITIAL, OPS = 160, 8, 80, 80


def _make(depth, delta, leaf_size, memtable_limit, compaction_threshold, seed):
    X = make_synthetic(n=N, dim=DIM, seed=seed)
    cfg_idx = IndexConfig(
        dim=DIM, leaf_size=leaf_size, branching_factor=4, seed=seed,
        depth=depth, delta=delta,
    )
    cfg_maint = MaintainConfig(
        strategy="outofplace",
        memtable_limit=memtable_limit,
        compaction_threshold=compaction_threshold,
        tombstone_rebuild_frac=0.25,
    )
    counters = Counters()
    maint = OutOfPlaceMaintainer(
        X, cfg_idx, cfg_maint, counters,
        initial_vids=np.arange(INITIAL, dtype=np.int64),
    )
    return X, maint, counters


def _check_structure(maint: OutOfPlaceMaintainer):
    for s_idx, seg in enumerate(maint._sealed):
        postings = seg.postings
        levels = seg.levels
        leaf = levels[-1].centroids

        # Posting ids are positional and match their leaf-centroid row.
        assert len(leaf) == len(postings)
        for i, p in enumerate(postings):
            assert p.id == i, f"sealed[{s_idx}] posting {i} id {p.id}"
            assert np.array_equal(
                np.asarray(p.centroid, np.float32), np.asarray(leaf[i], np.float32)
            ), f"sealed[{s_idx}] posting {i} centroid != leaf row"

        # Routing levels have valid, complete, disjoint parent->child references.
        for li in range(len(levels) - 1):
            n_next = len(levels[li + 1].centroids)
            covered: list[int] = []
            for children in levels[li].children:
                for c in children.tolist():
                    assert 0 <= int(c) < n_next, f"sealed[{s_idx}] level {li} child {c} out of range"
                    covered.append(int(c))
            assert sorted(covered) == list(range(n_next)), (
                f"sealed[{s_idx}] level {li}: children do not partition next level"
            )


def _check_query(maint: OutOfPlaceMaintainer, queries: list[np.ndarray], live: set[int], k: int = 5):
    units = maint.searchable_units()
    max_postings = max((len(u.postings) for u in units), default=1)
    cfg_q = QueryConfig(
        k=k, beam_widths=(max(1, max_postings),), n_postings=max(1, max_postings)
    )
    live_arr = np.array(sorted(live), dtype=np.int64)
    for q in queries:
        res = search(q, units, cfg_q, Counters())
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


def _run(depth, delta, leaf_size, memtable_limit, compaction_threshold, workload, seed, insert_prob=0.35):
    X, maint, counters = _make(depth, delta, leaf_size, memtable_limit, compaction_threshold, seed)
    trace = generate_churn_trace(
        initial_n=INITIAL, reserve_n=N - INITIAL, n_operations=OPS,
        insert_prob=insert_prob, seed=seed + 1, workload=workload,
    )
    live = set(range(INITIAL))
    queries = [X[i] for i in range(0, N, 19)]

    for step, (op, raw) in enumerate(trace.operations):
        vid = int(raw)
        if op == "insert":
            maint.insert(vid, X[vid])
            live.add(vid)
        else:
            maint.delete(vid)
            live.discard(vid)
        maint.maintain()

        npa = audit_maintainer(maint, expected_live_vids=set(live))
        mem = audit_outofplace_membership(maint)
        assert npa.passed, f"npa fail step {step}: {npa.to_dict()}"
        assert mem.passed, f"membership fail step {step}: {mem.to_dict()}"
        _check_structure(maint)
        _check_query(maint, queries, live)

    npa = audit_maintainer(maint, expected_live_vids=set(live))
    mem = audit_outofplace_membership(maint)
    assert npa.passed and mem.passed
    return counters


# Matrix of configurations: depth 1..4, delta None/0.3, memtable_limit, compaction_threshold, workload
MATRIX = [
    (depth, delta, 16, m_lim, c_thr, "mixed")
    for depth in (1, 2, 3, 4)
    for delta in (None, 0.3)
    for m_lim, c_thr in ((15, 3), (25, 4))
]
WORKLOADS = [
    (1, 0.3, 16, 15, 3, "insert_only"),
    (1, 0.3, 16, 15, 3, "delete_only"),
    (3, None, 16, 20, 3, "insert_only"),
    (3, None, 16, 20, 3, "delete_only"),
    (4, 0.3, 16, 15, 3, "delete_only"),
]


@pytest.mark.parametrize(
    "depth,delta,leaf_size,m_lim,c_thr,workload", MATRIX + WORKLOADS,
    ids=[f"d{d}_delta{de}_mlim{ml}_cthr{ct}_{wl}" for d, de, lf, ml, ct, wl in MATRIX + WORKLOADS],
)
def test_outofplace_integration_matrix(depth, delta, leaf_size, m_lim, c_thr, workload):
    seed = 2000 + depth * 10 + (0 if delta is None else int(delta * 10))
    counters = _run(depth, delta, leaf_size, m_lim, c_thr, workload, seed)
    assert counters.flushes >= 0 and counters.compaction_events >= 0
