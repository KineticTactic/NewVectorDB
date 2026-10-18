"""test_inplace_merge.py — deletion-driven posting merges (in-place arm).

R1 Task 4.  A merge is the structural inverse of a split: after a delete pushes
a leaf posting below the underflow threshold, it is fused with a sibling leaf
whose combined live occupancy still fits under ``split_threshold``.  The suite
below pins the policy (trigger, candidate selection, survivor, centroid),
the structural invariants (positional posting ids, leaf-centroid sync, hierarchy
links), the ownership repair (sound ball screen + exact ``argmax``), the
fixed-Δ membership re-derivation, spread radii, and failure atomicity.

Every test that mutates the index runs BOTH independent validators:
``audit_maintainer`` (coverage, unique ownership, NPA) and ``audit_membership``
(the fixed-Δ oracle, derived without any production helper).  Geometry is 2-D
unit vectors, so similarity/distance order is angle order.
"""

from __future__ import annotations

import numpy as np
import pytest

from vecstudy.core.config import IndexConfig, MaintainConfig, QueryConfig
from vecstudy.core.metrics import Counters
from vecstudy.core.types import HierarchyLevel, Posting
from vecstudy.harness.churn import (
    audit_maintainer,
    audit_membership,
    generate_churn_trace,
)
from vecstudy.index.query import search
from vecstudy.index.segment import Segment
from vecstudy.maintain.inplace import InPlaceMaintainer

import vecstudy.maintain.inplace as inplace_mod


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _unit(angle_deg) -> np.ndarray:
    rad = np.deg2rad(np.asarray(angle_deg, dtype=np.float64))
    return np.array([np.cos(rad), np.sin(rad)], dtype=np.float32)


def _means(X, lists):
    return [X[np.array(v, dtype=np.int64)].mean(axis=0).astype(np.float32) for v in lists]


def _leaf_level(centroids):
    return HierarchyLevel(
        centroids=np.stack(centroids).astype(np.float32), children=[]
    )


def _hier(leaf_centroids, depth, groups=None):
    """Build a routing hierarchy of the requested depth above leaf centroids.

    ``groups`` partitions leaf indices into immediate-parent groups and is used
    at depth >= 3, where sibling relationships are not "all leaves".
    """
    leaf_centroids = np.asarray(leaf_centroids, dtype=np.float32)
    K = len(leaf_centroids)
    leaf = _leaf_level(leaf_centroids)
    if depth <= 1:
        return [leaf]
    if depth == 2:
        root = HierarchyLevel(
            centroids=leaf_centroids.mean(axis=0, keepdims=True).astype(np.float32),
            children=[np.arange(K, dtype=np.int64)],
        )
        return [root, leaf]
    if groups is None:
        groups = [list(range(K))]
    mid_c = np.stack([leaf_centroids[np.array(g)].mean(axis=0) for g in groups]
                     ).astype(np.float32)
    mid = HierarchyLevel(
        centroids=mid_c,
        children=[np.array(g, dtype=np.int64) for g in groups],
    )
    if depth == 3:
        root = HierarchyLevel(
            centroids=mid_c.mean(axis=0, keepdims=True).astype(np.float32),
            children=[np.arange(len(groups), dtype=np.int64)],
        )
        return [root, mid, leaf]
    # depth == 4: root -> l2 -> mid -> leaf
    l2 = HierarchyLevel(
        centroids=mid_c.mean(axis=0, keepdims=True).astype(np.float32),
        children=[np.arange(len(groups), dtype=np.int64)],
    )
    root = HierarchyLevel(
        centroids=mid_c.mean(axis=0, keepdims=True).astype(np.float32),
        children=[np.array([0], dtype=np.int64)],
    )
    return [root, l2, mid, leaf]


def _segment(X, centroids, owned, levels=None, tombstones=None):
    postings = []
    for pid, (c, vids) in enumerate(zip(centroids, owned)):
        oa = np.array(vids, dtype=np.int64)
        postings.append(Posting(pid, np.asarray(c, dtype=np.float32), oa, oa))
    leaf_c = [np.asarray(c, dtype=np.float32) for c in centroids]
    lv = levels if levels is not None else [_leaf_level(leaf_c)]
    return Segment(levels=lv, postings=postings, vectors=X,
                   tombstones=tombstones or set())


def _maintainer(X, seg, *, depth=1, delta=None, split_threshold=10,
                merge_threshold=None, branching_factor=4):
    cfg = IndexConfig(
        dim=X.shape[1], leaf_size=100, branching_factor=branching_factor,
        seed=0, depth=depth, delta=delta,
    )
    mt = MaintainConfig(
        strategy="inplace", split_threshold=split_threshold,
        merge_threshold=merge_threshold,
    )
    counters = Counters()
    maint = InPlaceMaintainer(X, cfg, mt, counters, initial_segment=seg)
    return maint, counters


def _snapshot(maint):
    seg = maint._segment
    return {
        "centroids": seg.levels[-1].centroids.copy(),
        "postings": [(p.id, p.centroid.copy(), p.members.copy(), p.owned.copy())
                     for p in seg.postings],
        "tombstones": set(seg.tombstones),
        "spread": list(maint._spread),
    }


def _assert_audits(maint, live=None):
    npa = audit_maintainer(maint, expected_live_vids=live)
    assert npa.passed, f"NPA: {npa.summary()} {npa.examples}"
    mem = audit_membership(maint)
    assert mem.passed, f"membership: {mem.to_dict()}"


def _owner_of(maint, vid):
    return [p.id for p in maint._segment.postings if vid in p.owned.tolist()]


def _owners_consistent(maint):
    seen = {}
    for p in maint._segment.postings:
        for v in p.owned.tolist():
            seen[int(v)] = seen.get(int(v), 0) + 1
    return all(c == 1 for c in seen.values()), seen


# ---------------------------------------------------------------------------
# 1 & 10: trigger boundary, structure, centroid
# ---------------------------------------------------------------------------

def _three_even_postings():
    """pid0/1/2, each owning 3 live vectors in well-separated clusters."""
    angles = [0.0, 0.5, 1.0,  20.0, 20.5, 21.0,  90.0, 90.5, 91.0]
    X = np.stack([_unit(a) for a in angles]).astype(np.float32)
    owned = [[0, 1, 2], [3, 4, 5], [6, 7, 8]]
    return X, _segment(X, _means(X, owned), owned)


def test_no_merge_when_owning_count_stays_at_threshold():
    """A posting with live_owned == threshold is not underfilled (strict <)."""
    X, seg = _three_even_postings()
    # Give pid0 a 4th vector so deleting one leaves it exactly at 3.
    angles = [0.0, 0.5, 1.0, 1.5, 20.0, 20.5, 21.0, 90.0, 90.5, 91.0]
    X = np.stack([_unit(a) for a in angles]).astype(np.float32)
    owned = [[0, 1, 2, 3], [4, 5, 6], [7, 8, 9]]
    seg = _segment(X, _means(X, owned), owned)
    maint, counters = _maintainer(X, seg, split_threshold=10, merge_threshold=3)

    before = _snapshot(maint)
    maint.delete(3)  # pid0: 4 -> 3, not below threshold

    assert counters.merges == 0
    assert len(maint._segment.postings) == 3
    np.testing.assert_array_equal(
        maint._segment.levels[-1].centroids, before["centroids"]
    )
    _assert_audits(maint, {0, 1, 2, 4, 5, 6, 7, 8, 9})


def test_delete_below_threshold_triggers_merge_and_centroid_is_live_mean():
    """The underfilled posting merges with its sibling; centroid = live mean."""
    X, seg = _three_even_postings()
    maint, counters = _maintainer(X, seg, split_threshold=10, merge_threshold=3)

    maint.delete(2)  # pid0: 3 -> 2 (< 3) -> underflow

    assert counters.merges == 1
    assert len(maint._segment.postings) == 2  # pid0+pid1 fused; pid2 remains
    posts = maint._segment.postings
    assert posts[0].owned.tolist() == [0, 1, 3, 4, 5]
    assert posts[1].owned.tolist() == [6, 7, 8]
    merged = posts[0]
    expected = X[[0, 1, 3, 4, 5]].mean(axis=0).astype(np.float32)
    np.testing.assert_allclose(merged.centroid, expected, rtol=0, atol=1e-6)
    assert merged.id == 0
    np.testing.assert_array_equal(
        maint._segment.levels[-1].centroids[0], maint._segment.postings[0].centroid
    )
    _assert_audits(maint, {0, 1, 3, 4, 5, 6, 7, 8})


# ---------------------------------------------------------------------------
# 3 & 4: no eligible sibling
# ---------------------------------------------------------------------------

def test_no_merge_without_compatible_sibling():
    """Underflow but the only sibling's combined size exceeds split_threshold."""
    angles = [0.0, 0.5, 1.0,  20.0, 20.5, 21.0, 21.5, 22.0]
    X = np.stack([_unit(a) for a in angles]).astype(np.float32)
    owned = [[0, 1, 2], [3, 4, 5, 6, 7]]
    seg = _segment(X, _means(X, owned), owned)
    maint, counters = _maintainer(X, seg, split_threshold=5, merge_threshold=3)

    before = _snapshot(maint)
    maint.delete(2)  # pid0: 3 -> 2 (< 3); pid0+pid1 = 7 > 5

    assert counters.merges == 0
    assert len(maint._segment.postings) == 2
    np.testing.assert_array_equal(
        maint._segment.levels[-1].centroids, before["centroids"]
    )
    _assert_audits(maint, {0, 1, 3, 4, 5, 6, 7})


def test_no_merge_when_combined_exceeds_split_threshold():
    """The combined-size hysteresis is what blocks the merge here."""
    angles = [0.0, 0.5, 1.0,  10.0, 10.5, 11.0]
    X = np.stack([_unit(a) for a in angles]).astype(np.float32)
    owned = [[0, 1, 2], [3, 4, 5]]
    seg = _segment(X, _means(X, owned), owned)
    # threshold 3: pid0 -> 2 is underfilled; 2 + 3 = 5 > split_threshold 4.
    maint, counters = _maintainer(X, seg, split_threshold=4, merge_threshold=3)

    maint.delete(1)
    assert counters.merges == 0
    assert len(maint._segment.postings) == 2


# ---------------------------------------------------------------------------
# 5 & 6: partner selection
# ---------------------------------------------------------------------------

def test_lowest_ward_cost_partner_is_selected():
    """Among eligible siblings, the smallest Ward cost wins."""
    angles = [0.0, 0.5, 1.0,   12.0, 12.5, 13.0,   80.0, 80.5, 81.0]
    X = np.stack([_unit(a) for a in angles]).astype(np.float32)
    owned = [[0, 1, 2], [3, 4, 5], [6, 7, 8]]
    seg = _segment(X, _means(X, owned), owned)
    maint, counters = _maintainer(X, seg, split_threshold=10, merge_threshold=3)

    maint.delete(2)  # pid0 -> {0,1}

    assert counters.merges == 1
    # pid1 (12 deg) is nearer than pid2 (80 deg); both were eligible.
    assert maint._segment.postings[0].owned.tolist() == [0, 1, 3, 4, 5]
    assert maint._segment.postings[1].owned.tolist() == [6, 7, 8]
    _assert_audits(maint, {0, 1, 3, 4, 5, 6, 7, 8})


def test_equal_costs_resolve_deterministically_to_lower_id():
    """Mirror-image partners have equal cost; the lower posting id is chosen."""
    angles = [0.0, 0.5, 1.0,   10.0, 10.5, 11.0,   -10.0, -10.5, -11.0]
    X = np.stack([_unit(a) for a in angles]).astype(np.float32)
    owned = [[0, 1, 2], [3, 4, 5], [6, 7, 8]]
    seg = _segment(X, _means(X, owned), owned)
    maint, counters = _maintainer(X, seg, split_threshold=10, merge_threshold=3)

    maint.delete(2)
    assert counters.merges == 1
    # pid1 (id 1) beats pid2 (id 2) on the cost tie.
    assert maint._segment.postings[0].owned.tolist() == [0, 1, 3, 4, 5]
    first = _snapshot(maint)

    # Re-run from scratch: identical outcome.
    maint2, _ = _maintainer(X, seg, split_threshold=10, merge_threshold=3)
    maint2.delete(2)
    second = _snapshot(maint2)
    assert first["postings"][0][3].tolist() == second["postings"][0][3].tolist()
    np.testing.assert_array_equal(first["centroids"], second["centroids"])


# ---------------------------------------------------------------------------
# 7 & 8: hierarchy depth / parent-child
# ---------------------------------------------------------------------------

def test_depth1_has_implicit_root_and_merges_siblings():
    X, seg = _three_even_postings()
    maint, counters = _maintainer(X, seg, depth=1, split_threshold=10,
                                  merge_threshold=3)
    assert maint._leaf_parent_of() == {}
    maint.delete(2)
    assert counters.merges == 1


def _parent_fixture(depth, groups=None):
    angles = [0.0, 0.5, 1.0,   30.0, 30.5, 31.0, 31.5, 32.0, 32.5,
              150.0, 152.0, 154.0]
    X = np.stack([_unit(a) for a in angles]).astype(np.float32)
    owned = [[0, 1, 2], [3, 4, 5, 6, 7, 8], [9, 10, 11]]
    cents = _means(X, owned)
    levels = _hier(cents, depth, groups)
    seg = _segment(X, cents, owned, levels=levels)
    return X, seg


@pytest.mark.parametrize("depth", [3, 4])
def test_merge_only_within_same_parent(depth):
    """At depth >= 3 a cross-parent sibling is never merged, even if small.

    Leaves 0 and 1 share a parent; leaf 2 is under a different parent.  After
    the delete, leaf 0 is underfilled, its same-parent sibling (leaf 1) is too
    large to merge, and the only size-compatible partner (leaf 2) is a
    different parent — so no merge may occur.
    """
    X, seg = _parent_fixture(depth, groups=[[0, 1], [2]])
    maint, counters = _maintainer(X, seg, depth=depth, split_threshold=6,
                                  merge_threshold=3)

    parent_of = maint._leaf_parent_of()
    assert parent_of[0] == parent_of[1] and parent_of[2] != parent_of[0]

    before = _snapshot(maint)
    maint.delete(2)  # leaf0 -> {0,1}; leaf0+leaf1 = 8 > 6

    assert counters.merges == 0
    assert len(maint._segment.postings) == 3
    np.testing.assert_array_equal(
        maint._segment.levels[-1].centroids, before["centroids"]
    )
    _assert_audits(maint, {0, 1, 3, 4, 5, 6, 7, 8, 9, 10, 11})


@pytest.mark.parametrize("depth", [1, 2])
def test_depth1_and_2_treat_all_leaves_as_siblings(depth):
    """Without a partition above the leaves, the cross-group partner merges."""
    X, seg = _parent_fixture(depth, groups=[[0, 1], [2]] if depth > 2 else None)
    maint, counters = _maintainer(X, seg, depth=depth, split_threshold=6,
                                  merge_threshold=3)

    maint.delete(2)  # leaf0 -> {0,1}; the only size-compatible partner is leaf2

    # At depth 1/2 every leaf is a sibling, so the cross-group leaf2 is a legal
    # partner and a merge fires (contrast the depth 3/4 test, where it is not).
    assert counters.merges == 1
    _assert_audits(maint, {0, 1, 3, 4, 5, 6, 7, 8, 9, 10, 11})


@pytest.mark.parametrize("depth", [1, 2, 3, 4])
def test_hierarchy_is_valid_after_merge(depth):
    X, seg = _parent_fixture(depth, groups=[[0, 1], [2]])
    maint, counters = _maintainer(X, seg, depth=depth, split_threshold=10,
                                  merge_threshold=3)
    maint.delete(6)  # leaf1 (owns 6) -> 5, still >= 3
    maint.delete(2)  # leaf0 -> {0,1} (< 3) -> underflow drives a merge

    assert counters.merges >= 1
    levels = maint._segment.levels
    assert 1 <= len(levels) <= depth
    assert len(levels[-1].centroids) == len(maint._segment.postings)
    assert levels[-1].children == []
    for up, down in zip(levels[:-1], levels[1:]):
        covered = set()
        for ch in up.children:
            assert len(ch) > 0
            assert int(ch.max()) < len(down.centroids)
            covered.update(int(c) for c in ch.tolist())
        assert covered == set(range(len(down.centroids)))
    _assert_audits(maint, {0, 1, 3, 4, 5, 7, 8, 9, 10, 11})


# ---------------------------------------------------------------------------
# 9 & 10: centroid and id/positional invariants
# ---------------------------------------------------------------------------

def test_higher_id_removed_and_ids_stay_contiguous():
    X, seg = _three_even_postings()
    maint, counters = _maintainer(X, seg, split_threshold=10, merge_threshold=3)
    maint.delete(2)
    postings = maint._segment.postings
    assert [p.id for p in postings] == list(range(len(postings)))
    leaf = maint._segment.levels[-1].centroids
    assert len(leaf) == len(postings)
    for i, p in enumerate(postings):
        np.testing.assert_allclose(leaf[i], p.centroid, rtol=0, atol=0)


# ---------------------------------------------------------------------------
# 12 & 13 & 14 & 15: ownership repair
# ---------------------------------------------------------------------------

def test_vector_from_removed_posting_changes_owner():
    """A vector owned by the removed leaf is won by an unrelated posting.

    pid 0 (0 deg) is pushed under threshold by the delete and merges with its
    nearest eligible sibling pid 1 (45 deg, owning 25 and 65 deg).  The merged
    centroid lands near 30 deg, so the removed leaf's own 65 deg vector is now
    closer to the big posting at 85 deg and must change owner.
    """
    angles = [0.0, 0.3, 25.0, 65.0, 80.0, 85.0, 90.0]
    X = np.stack([_unit(a) for a in angles]).astype(np.float32)
    owned = [[0, 1], [2, 3], [4, 5, 6]]
    seg = _segment(X, _means(X, owned), owned)
    maint, counters = _maintainer(X, seg, split_threshold=10, merge_threshold=2)

    maint.delete(1)  # pid0 -> {0}; candidates (0,1) and (0,2); (0,1) is nearer

    assert counters.merges == 1
    assert len(maint._segment.postings) == 2
    # vid 3 (65 deg) belonged to the removed leaf pid 1; the merged centroid is
    # far from it, so the posting around 85 deg must own it.
    assert _owner_of(maint, 3) == [1]
    assert _owner_of(maint, 2) == [0]  # the 25 deg vector stays with the merge
    _assert_audits(maint, {0, 2, 3, 4, 5, 6})


def test_ownership_equals_global_argmax_after_merge():
    """Repaired ownership equals a brute-force argmax over the final centroids.

    This is the exact membership/ownership rule (inner product, ties -> lowest
    posting id) applied to every live vector against every surviving leaf
    centroid, so it subsumes both "a vector moved" and "a vector stayed".
    """
    angles = [0.0, 0.5, 1.0,   14.0, 16.0,   78.0, 80.0, 82.0]
    X = np.stack([_unit(a) for a in angles]).astype(np.float32)
    owned = [[0, 1, 2], [3, 4], [5, 6, 7]]
    seg = _segment(X, _means(X, owned), owned)
    maint, counters = _maintainer(X, seg, split_threshold=10, merge_threshold=3)

    maint.delete(2)  # pid0 -> {0,1} (< 3); pid0+pid1 (nearest) fuses
    assert counters.merges == 1

    live = {0, 1, 3, 4, 5, 6, 7}
    leaf = maint._segment.levels[-1].centroids
    owner_map = {}
    for p in maint._segment.postings:
        for v in p.owned.tolist():
            owner_map[int(v)] = p.id
    for vid in live:
        sims = leaf @ X[vid]
        expect = int(np.argmax(sims))
        assert owner_map[vid] == expect, (vid, owner_map[vid], expect)
    assert set(owner_map) == live
    _assert_audits(maint, live)


def test_untouched_posting_keeps_vectors_and_spread():
    """A far posting is provably unaffected: owned/members/spread unchanged."""
    angles = [0.0, 0.5, 3.0, 180.0, 182.0]
    X = np.stack([_unit(a) for a in angles]).astype(np.float32)
    owned = [[0, 1], [2], [3, 4]]
    cents = _means(X, owned)
    seg = _segment(X, cents, owned)
    maint, counters = _maintainer(X, seg, split_threshold=10, merge_threshold=2)

    far_before = (maint._segment.postings[2].owned.copy(),
                  maint._segment.postings[2].members.copy(),
                  maint._segment.postings[2].centroid.copy(),
                  maint._spread[2])
    maint.delete(1)

    assert counters.merges == 1
    far = maint._segment.postings[1]  # pid2 is renumbered to 1
    np.testing.assert_array_equal(far.owned, far_before[0])
    np.testing.assert_array_equal(far.members, far_before[1])
    np.testing.assert_array_equal(far.centroid, far_before[2])
    assert maint._spread[1] == pytest.approx(far_before[3])


def test_exact_tie_follows_lowest_posting_id():
    """A sel vector scores the merged and an unchanged centroid equally.

    Depth-3 layout: pid 0 (C) is alone under its parent, so it can never merge;
    pid 1 (B, two copies of b at 60 deg) and pid 2 (A, two copies of y=(1,0))
    share a parent.  Deleting one y leaves A underfilled, so B+A fuse and the
    survivor centroid is exactly (2/3, *).  y's similarity to it is exactly
    2/3 — the same as to C's centroid (whose x-component is 2/3).  The merged
    posting has the higher id, so the tie must resolve to C (pid 0).
    """
    y = np.array([1.0, 0.0], dtype=np.float32)
    b = _unit(60.0)
    cx = np.float32(2.0 / 3.0)
    c = np.array([cx, -np.float32(np.sqrt(1.0 - float(cx) ** 2))], dtype=np.float32)
    # pid0 = C (4 copies of c), pid1 = B (2 copies of b), pid2 = A (2 copies of y).
    X = np.stack([c, c, c, c,   b, b,   y, y]).astype(np.float32)
    owned = [[0, 1, 2, 3], [4, 5], [6, 7]]
    levels = _hier([c, b, y], depth=3, groups=[[0], [1, 2]])
    seg = _segment(X, [c, b, y], owned, levels=levels)
    maint, counters = _maintainer(X, seg, depth=3, split_threshold=10,
                                  merge_threshold=2)

    maint.delete(7)  # A (pid2) -> {6}; A runs under its parent with B (pid1)

    assert counters.merges == 1
    sims = maint._segment.levels[-1].centroids @ y
    best = float(sims.max())
    tied = sorted(int(p) for p in np.flatnonzero(sims == best))
    assert best == np.float32(2.0 / 3.0)
    assert tied == [0, 1], f"expected a tie between pid0 and the merged pid1, got {tied}"
    assert _owner_of(maint, 6) == [min(tied)] == [0]
    _assert_audits(maint, {0, 1, 2, 3, 4, 5, 6})


def test_merge_leaves_no_zero_or_multiple_owners():
    X, seg = _three_even_postings()
    maint, _ = _maintainer(X, seg, split_threshold=10, merge_threshold=3)
    maint.delete(2)
    ok, seen = _owners_consistent(maint)
    assert ok
    assert set(seen) == {0, 1, 3, 4, 5, 6, 7, 8}


# ---------------------------------------------------------------------------
# 17 & 18 & 19 & 20: deletions, replicas, membership
# ---------------------------------------------------------------------------

def test_deleted_vector_is_absent_from_all_live_arrays_after_merge():
    X, seg = _three_even_postings()
    maint, counters = _maintainer(X, seg, split_threshold=10, merge_threshold=3)
    maint.delete(2)
    for p in maint._segment.postings:
        assert 2 not in p.owned.tolist()
        assert 2 not in p.members.tolist()


def test_deleting_multi_replica_vector_clears_stale_references():
    """A vector in two postings loses both memberships on delete."""
    angles = [0.0, 0.5,   6.0,   10.0, 10.5, 11.0]
    X = np.stack([_unit(a) for a in angles]).astype(np.float32)
    owned = [[0, 1, 2], [3, 4, 5]]
    cents = _means(X, owned)
    seg = _segment(X, cents, owned)
    maint, counters = _maintainer(X, seg, delta=1.0, split_threshold=10,
                                  merge_threshold=3)
    maint.rebuild_membership()  # derive replicas for the supplied layout

    # vid 2 (6 deg) sits between the owners, so it should hold a replica.
    assert 2 in maint._segment.postings[1].members.tolist()
    removed_before = counters.replicas_removed

    maint.delete(2)
    assert counters.replicas_removed > removed_before
    for p in maint._segment.postings:
        assert 2 not in p.owned.tolist()
        assert 2 not in p.members.tolist()
    _assert_audits(maint, {0, 1, 3, 4, 5})


def test_merge_membership_matches_oracle_with_overlap():
    """With Δ>0 the final memberships match the independent oracle exactly."""
    angles = [0.0, 0.5, 1.0,   8.0, 8.5, 9.0,   40.0, 40.5, 41.0]
    X = np.stack([_unit(a) for a in angles]).astype(np.float32)
    owned = [[0, 1, 2], [3, 4, 5], [6, 7, 8]]
    seg = _segment(X, _cents := _means(X, owned), owned)
    maint, counters = _maintainer(X, seg, delta=1.5, split_threshold=10,
                                  merge_threshold=3)

    maint.delete(2)
    assert counters.merges == 1
    mem = audit_membership(maint)
    assert mem.passed, mem.to_dict()
    _assert_audits(maint, {0, 1, 3, 4, 5, 6, 7, 8})


def test_unchanged_owner_replica_is_recomputed():
    """A merge can add/remove a replica for a vector whose owner did not change."""
    # pid2 (40 deg) keeps its owner; the merged centroid moves toward it.
    angles = [0.0, 0.5, 1.0,   30.0, 30.5, 31.0,   38.0]
    X = np.stack([_unit(a) for a in angles]).astype(np.float32)
    owned = [[0, 1, 2], [3, 4, 5], [6]]
    seg = _segment(X, _means(X, owned), owned)
    maint, counters = _maintainer(X, seg, delta=None, split_threshold=10,
                                  merge_threshold=3)
    maint.delete(2)
    # Ownership of vid 6 is unchanged (still the far-ish posting), and both
    # audits pass — membership was re-derived, not carried forward.
    _assert_audits(maint, {0, 1, 3, 4, 5, 6})


# ---------------------------------------------------------------------------
# 22: spread radii
# ---------------------------------------------------------------------------

def test_spread_radii_match_independent_recomputation():
    angles = [0.0, 0.5, 1.0,   10.0, 10.5, 11.0,   90.0, 90.5, 91.0]
    X = np.stack([_unit(a) for a in angles]).astype(np.float32)
    owned = [[0, 1, 2], [3, 4, 5], [6, 7, 8]]
    seg = _segment(X, _means(X, owned), owned)
    maint, _ = _maintainer(X, seg, split_threshold=10, merge_threshold=3)
    maint.delete(2)

    for pid, p in enumerate(maint._segment.postings):
        if len(p.owned) == 0:
            assert maint._spread[pid] == 0.0
        else:
            expected = float(np.max(
                np.linalg.norm(X[p.owned] - p.centroid, axis=1)
            ))
            assert maint._spread[pid] == pytest.approx(expected, abs=1e-6)


# ---------------------------------------------------------------------------
# 23 - 27: repeated operations
# ---------------------------------------------------------------------------

def test_split_then_delete_merge_is_consistent():
    """A split, then a delete that underflows, then a merge — all audits clean."""
    angles = [0.0, 1.0,   40.0, 41.0,   0.4]
    X = np.stack([_unit(a) for a in angles]).astype(np.float32)
    owned = [[0, 1], [2, 3]]
    seg = _segment(X, _means(X, owned), owned)
    maint, counters = _maintainer(X, seg, split_threshold=2, merge_threshold=2)
    live = {0, 1, 2, 3}

    maint.insert(4, X[4])  # pid0 grows to 3 > 2 -> production split
    live.add(4)
    assert counters.splits >= 1
    _assert_audits(maint, live)

    # Now push the near-0 postings to underflow and fuse them again.
    for vid in (0, 1, 4):
        maint.delete(vid)
        live.discard(vid)
    assert counters.merges >= 1
    _assert_audits(maint, live)


def test_repeated_delete_merge_accumulates_no_stale_or_duplicates():
    angles = [float(a) for a in range(12)]
    X = np.stack([_unit(a) for a in angles]).astype(np.float32)
    owned = [[0, 1, 2], [3, 4, 5], [6, 7, 8], [9, 10, 11]]
    seg = _segment(X, _means(X, owned), owned)
    maint, counters = _maintainer(X, seg, split_threshold=8, merge_threshold=3)
    live = set(range(len(X)))

    for vid in (1, 2, 4, 7, 10):
        maint.delete(vid)
        live.discard(vid)
        _assert_audits(maint, live)
        for p in maint._segment.postings:
            assert len(set(p.members.tolist())) == len(p.members)
            assert set(p.owned.tolist()) <= set(p.members.tolist())
    assert counters.merges >= 1


def test_mixed_churn_keeps_both_audits():
    X = np.stack([_unit(a) for a in np.linspace(0, 350, 240)]).astype(np.float32)
    cfg = IndexConfig(dim=2, leaf_size=20, branching_factor=4, seed=0, depth=2,
                      delta=0.4)
    mt = MaintainConfig(strategy="inplace", split_threshold=20,
                        merge_threshold=8)
    counters = Counters()
    maint = InPlaceMaintainer(
        X, cfg, mt, counters, initial_vids=np.arange(120, dtype=np.int64)
    )
    trace = generate_churn_trace(
        initial_n=120, reserve_n=120, n_operations=300, insert_prob=0.6,
        seed=7, workload="mixed",
    )
    live = set(range(120))
    for op, vid in trace.operations:
        if op == "insert":
            maint.insert(int(vid), X[int(vid)])
            live.add(int(vid))
        else:
            maint.delete(int(vid))
            live.discard(int(vid))
    _assert_audits(maint, live)
    assert counters.merges > 0 and counters.splits > 0


def test_single_remaining_posting_stays_valid():
    X, seg = _three_even_postings()
    maint, _ = _maintainer(X, seg, split_threshold=10, merge_threshold=3)
    # Delete until at most one posting remains; it must stay consistent.
    for vid in (2, 5, 8):
        maint.delete(vid)
    assert len(maint._segment.postings) == 1
    p = maint._segment.postings[0]
    assert p.id == 0
    assert len(maint._segment.levels[-1].centroids) == 1
    np.testing.assert_allclose(maint._segment.levels[-1].centroids[0], p.centroid)


def test_no_live_vectors_is_query_safe():
    angles = [0.0, 0.5, 1.0, 5.0, 5.5, 6.0]
    X = np.stack([_unit(a) for a in angles]).astype(np.float32)
    owned = [[0, 1, 2], [3, 4, 5]]
    seg = _segment(X, _means(X, owned), owned)
    maint, _ = _maintainer(X, seg, split_threshold=10, merge_threshold=3)

    for vid in range(len(X)):
        maint.delete(vid)

    assert len(maint._segment.postings) >= 1
    for p in maint._segment.postings:
        assert p.owned.tolist() == []
        assert p.members.tolist() == []
    _assert_audits(maint, set())
    n = len(maint._segment.postings)
    cfg_q = QueryConfig(k=3, beam_widths=(max(1, n),), n_postings=max(1, n))
    res = search(_unit(0.0), maint.searchable_units(), cfg_q, Counters())
    assert res == []


# ---------------------------------------------------------------------------
# 28 & 29: determinism and failure atomicity
# ---------------------------------------------------------------------------

def test_failed_merge_attempt_leaves_state_unchanged():
    angles = [0.0, 0.5, 1.0,   20.0, 20.5, 21.0, 21.5, 22.0]
    X = np.stack([_unit(a) for a in angles]).astype(np.float32)
    owned = [[0, 1, 2], [3, 4, 5, 6, 7]]
    seg = _segment(X, _means(X, owned), owned)
    maint, counters = _maintainer(X, seg, split_threshold=5, merge_threshold=3)

    maint.delete(2)
    after = _snapshot(maint)
    # A second (idempotent) delete must not change the structure.
    maint.delete(2)
    again = _snapshot(maint)
    assert counters.merges == 0
    np.testing.assert_array_equal(after["centroids"], again["centroids"])
    for (_, _, bm, bo), (_, _, am, ao) in zip(after["postings"], again["postings"]):
        np.testing.assert_array_equal(bm, am)
        np.testing.assert_array_equal(bo, ao)


def test_exception_during_merge_rolls_back(monkeypatch):
    """A failure while rebuilding the hierarchy must not half-apply the merge."""
    X, seg = _three_even_postings()
    maint, counters = _maintainer(X, seg, split_threshold=10, merge_threshold=3)
    before = _snapshot(maint)
    merges_before = counters.merges

    def boom(*a, **k):
        raise RuntimeError("injected hierarchy failure")

    monkeypatch.setattr(inplace_mod, "build_hierarchy", boom)
    with pytest.raises(RuntimeError, match="injected"):
        maint.delete(2)

    after = _snapshot(maint)
    assert counters.merges == merges_before
    # The delete itself is not rolled back (it is not part of the atomic merge),
    # but the merge must not have altered anything beyond it.
    assert after["tombstones"] == {2}
    assert len(after["postings"]) == 3  # no leaf was removed
    assert after["postings"][0][3].tolist() == [0, 1]  # vid 2 dropped by delete
    for (bid, bc, bm, bo), (aid, ac, am, ao) in zip(
        before["postings"], after["postings"]
    ):
        assert bid == aid
        np.testing.assert_array_equal(bc, ac)
    for pid in range(3):
        np.testing.assert_array_equal(
            after["centroids"][pid], before["centroids"][pid]
        )
