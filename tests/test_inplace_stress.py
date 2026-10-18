"""test_inplace_stress.py — deterministic mixed-churn stress for the in-place arm.

R1 Task 3.  After every configured maintenance checkpoint this runs the two
*independent* validators on the in-place index:

  * ``audit_maintainer``  — coverage, unique canonical ownership and NPA
    (owner == ``argmax`` over live leaf centroids, tombstones excluded);
  * ``audit_membership``  — the fixed-Δ replica oracle, derived from the leaf
    centroids + owner + Δ with the ratio form, never from a production helper.

The matrix spans hierarchy depths (1 and 2), leaf sizes and Δ, and workloads mix
inserts, deletes and the splits they trigger.  Every config has a fixed seed, so
any failure is reproducible from the parameters alone.
"""

from __future__ import annotations

import numpy as np
import pytest

from vecstudy.core.config import IndexConfig, MaintainConfig
from vecstudy.core.metrics import Counters
from vecstudy.harness.churn import (
    audit_maintainer,
    audit_membership,
    generate_churn_trace,
)
from vecstudy.harness.datasets import make_synthetic
from vecstudy.maintain.inplace import InPlaceMaintainer


# depth, delta, leaf_size, split_threshold, seed
CONFIGS = [
    (1, None, 25, 30, 1),
    (2, None, 25, 30, 2),
    (1, 0.3, 40, 45, 3),
    (2, 0.6, 20, 25, 4),
    (1, 0.5, 30, 35, 5),
    (2, 0.2, 50, 55, 6),
]


@pytest.mark.parametrize(
    "depth,delta,leaf_size,split_threshold,seed",
    CONFIGS,
    ids=[f"d{d}_delta{de}_leaf{lf}_thr{th}_s{s}" for d, de, lf, th, s in CONFIGS],
)
def test_mixed_churn_keeps_npa_and_membership(
    depth, delta, leaf_size, split_threshold, seed
):
    n, dim, initial, n_ops, checkpoint = 800, 12, 200, 400, 50
    X = make_synthetic(n=n, dim=dim, seed=seed)

    cfg_idx = IndexConfig(
        dim=dim, leaf_size=leaf_size, branching_factor=4, seed=seed,
        depth=depth, delta=delta,
    )
    cfg_maint = MaintainConfig(strategy="inplace", split_threshold=split_threshold)
    counters = Counters()
    maint = InPlaceMaintainer(
        X, cfg_idx, cfg_maint, counters,
        initial_vids=np.arange(initial, dtype=np.int64),
    )

    trace = generate_churn_trace(
        initial_n=initial, reserve_n=n - initial, n_operations=n_ops,
        insert_prob=0.7, seed=seed, workload="mixed",
    )

    live = set(range(initial))
    for i, (op, raw_vid) in enumerate(trace.operations):
        vid = int(raw_vid)
        if op == "insert":
            maint.insert(vid, X[vid])
            live.add(vid)
        else:
            maint.delete(vid)
            live.discard(vid)

        if (i + 1) % checkpoint == 0:
            npa = audit_maintainer(maint, expected_live_vids=set(live))
            mem = audit_membership(maint)
            assert npa.passed, f"NPA failed at op {i + 1}: {npa.to_dict()}"
            assert mem.passed, f"membership failed at op {i + 1}: {mem.to_dict()}"
            # Tombstones must be gone from members independently of Δ.
            tomb = set(maint._segment.tombstones)
            for p in maint._segment.postings:
                assert not (set(int(v) for v in p.members) & tomb)

    npa = audit_maintainer(maint, expected_live_vids=set(live))
    mem = audit_membership(maint)
    assert npa.passed, npa.to_dict()
    assert mem.passed, mem.to_dict()
    assert npa.live_vectors_checked == len(live)
    assert counters.splits > 0, "config did not exercise splits"


def test_a_far_posting_is_untouched_by_a_split():
    """A split must change neither ownership nor membership of a far posting.

    Guards the over-correction direction: widening the candidate filter must not
    move vectors the centroid layout did not contest.  Geometry (2-D unit
    vectors): A owns {0 deg}, F owns {180 deg}; inserting 4 deg splits A into
    {0 deg, 4 deg}.  F's single vector equals its centroid, so the filter's
    bound for F against either new centroid is clearly negative and F is left
    exactly as it was.
    """
    from vecstudy.core.types import HierarchyLevel, Posting
    from vecstudy.index.segment import Segment

    def unit(deg):
        rad = np.deg2rad(deg)
        return np.array([np.cos(rad), np.sin(rad)], dtype=np.float32)

    X = np.stack([unit(0.0), unit(180.0), unit(4.0)]).astype(np.float32)
    postings = [
        Posting(0, unit(0.0), np.array([0], np.int64), np.array([0], np.int64)),
        Posting(1, unit(180.0), np.array([1], np.int64), np.array([1], np.int64)),
    ]
    leaf = HierarchyLevel(
        centroids=np.stack([p.centroid for p in postings]), children=[]
    )
    seg0 = Segment(levels=[leaf], postings=postings, vectors=X)
    cfg_idx = IndexConfig(dim=2, leaf_size=100, branching_factor=4, seed=0, depth=1)
    counters = Counters()
    maint = InPlaceMaintainer(
        X, cfg_idx, MaintainConfig(strategy="inplace", split_threshold=1),
        counters, initial_segment=seg0,
    )
    seg = maint._segment

    maint.insert(2, X[2])
    assert counters.splits == 1

    # F (pid 1) keeps exactly its owned set and membership.
    assert seg.postings[1].owned.tolist() == [1]
    assert seg.postings[1].members.tolist() == [1]
    npa = audit_maintainer(maint, expected_live_vids={0, 1, 2})
    mem = audit_membership(maint)
    assert npa.passed, npa.to_dict()
    assert mem.passed, mem.to_dict()
