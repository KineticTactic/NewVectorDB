"""test_membership.py — fixed-Δ replica membership for the in-place maintainer.

Two independent concerns, kept apart on purpose:

  * The oracle  — `harness.churn.audit_membership` must find a missing replica,
    an obsolete replica, a duplicate membership, a dead membership, a hole in
    the owner's members, and an inconsistent ownership.  Each of those is
    injected by hand into an otherwise valid fixture, so a passing oracle on
    the production path means something.
  * The maintainer — membership must be correct *immediately* after insert, and
    re-derived after a split or a delete, including for vectors whose canonical
    owner did not change.

Policy under test (DESIGN §6.4, R2 fixed-Δ experiments): similarity is the
inner product, distance is ``1 - similarity``, ``d1``/``d2`` are the distances
to the owner and to the nearest other leaf centroid, and a vector receives
exactly one extra membership iff ``d2/d1 - 1 <= delta``.  Canonical ownership
is unchanged by this feature: the owner always stays the owner and a vector can
only ever hold two memberships.

Geometry is 2-D unit vectors; angles are in degrees and, because both
similarity and distance are monotone in the angle, every ordering below can be
read off the angles.  All fixtures have explicit centroids and vector
coordinates — no test depends on random clustering to produce a transition.
"""

from __future__ import annotations

import numpy as np
import pytest

from vecstudy.core.config import IndexConfig, MaintainConfig, QueryConfig
from vecstudy.core.metrics import Counters
from vecstudy.core.types import HierarchyLevel, Posting
from vecstudy.harness.churn import audit_maintainer, audit_membership
from vecstudy.harness.datasets import make_synthetic
from vecstudy.index.segment import Segment
from vecstudy.index.query import search
from vecstudy.maintain.inplace import InPlaceMaintainer


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _unit(angle_deg) -> np.ndarray:
    """Unit vector at *angle_deg* (scalar or array), float32."""
    rad = np.deg2rad(np.asarray(angle_deg, dtype=np.float64))
    return np.array([np.cos(rad), np.sin(rad)], dtype=np.float32)


def _angle_of(vec: np.ndarray) -> float:
    """Angle of a 2-D direction in degrees, in (-180, 180]."""
    return float(np.rad2deg(np.arctan2(float(vec[1]), float(vec[0]))))


def _centroid_midpoint(angles) -> np.ndarray:
    """Area-style mean of unit vectors, i.e. the centroid k-means would form."""
    return np.mean(np.stack([_unit(a) for a in angles]), axis=0).astype(np.float32)


def _segment(X: np.ndarray, centroids, owned, members=None, tombstones=None) -> Segment:
    """Explicit segment: one posting per centroid, each owning *owned[i]*.

    ``members`` may differ from ``owned`` to express replica memberships;
    it defaults to ``owned`` (no overlap).
    """
    postings = []
    for pid, (centroid, vids) in enumerate(zip(centroids, owned)):
        c = np.asarray(centroid, dtype=np.float32)
        c = c if c.ndim == 1 else _unit(float(centroid))
        oa = np.array(vids, dtype=np.int64)
        ma = np.array(members[pid] if members is not None else vids, dtype=np.int64)
        postings.append(Posting(pid, c, ma, oa))
    leaf = HierarchyLevel(
        centroids=np.stack([p.centroid for p in postings]), children=[]
    )
    return Segment(
        levels=[leaf], postings=postings, vectors=X, tombstones=tombstones
    )


def _maintainer(
    X: np.ndarray,
    segment: Segment,
    *,
    delta: float | None,
    split_threshold: int,
    merge_threshold: int | None = None,
) -> tuple[InPlaceMaintainer, Counters]:
    cfg_index = IndexConfig(
        dim=X.shape[1], leaf_size=100, branching_factor=4, seed=0,
        depth=1, delta=delta,
    )
    cfg_maintain = MaintainConfig(
        strategy="inplace", split_threshold=split_threshold,
        merge_threshold=merge_threshold,
    )
    counters = Counters()
    maint = InPlaceMaintainer(
        X, cfg_index, cfg_maintain, counters, initial_segment=segment
    )
    return maint, counters


def _member_pids(seg: Segment, vid: int) -> tuple[int, ...]:
    return tuple(int(p.id) for p in seg.postings if vid in p.members.tolist())


def _owner_pids(seg: Segment, vid: int) -> tuple[int, ...]:
    return tuple(int(p.id) for p in seg.postings if vid in p.owned.tolist())


def _pid_at(maint: InPlaceMaintainer, angle_deg: float, tol: float = 0.15) -> int:
    """Posting id whose centroid sits at *angle_deg* (fixture self-check)."""
    for p in maint._segment.postings:
        if abs(_angle_of(p.centroid) - angle_deg) <= tol:
            return int(p.id)
    raise AssertionError(f"no centroid near {angle_deg} deg")


def _assert_audits(
    maint: InPlaceMaintainer, live_vids: set[int] | None = None
) -> None:
    """Both audits must pass: NPA/ownership/coverage *and* membership."""
    npa = audit_maintainer(maint, expected_live_vids=live_vids)
    assert npa.passed, (
        f"NPA audit failed: {npa.summary()} "
        f"{npa.npa_violations[:3]} {npa.ownership_errors[:3]} "
        f"{npa.coverage_errors[:3]}"
    )
    mem = audit_membership(maint)
    assert mem.passed, f"membership audit failed: {mem.to_dict()}"


def _snapshot(maint: InPlaceMaintainer) -> dict:
    seg = maint._segment
    return {
        "centroids": seg.levels[-1].centroids.copy(),
        "postings": [
            (p.id, p.centroid.copy(), p.members.copy(), p.owned.copy())
            for p in seg.postings
        ],
        "tombstones": set(seg.tombstones),
    }


# ---------------------------------------------------------------------------
# Fixture A — a 2-point posting that splits into its own two vectors.
#
#   pid 0   centroid 39.5 deg (mean of 36 and 43), owns nothing yet
#   pid 1   centroid 30 deg, owns {2: 30 deg, 3: 29 deg}
#   pid 2   centroid 80 deg, owns {4: 80 deg}
#
# Inserting vid 0 (36 deg) makes pid 0 own it.  The second-nearest posting is
# pid 1 at 6 deg, giving r = d(6 deg)/d(3.5 deg) - 1 = 0.42 <= delta = 0.5, so
# the vector must also be a member of pid 1.
#
# Inserting vid 1 (43 deg) pushes pid 0 to two owned vectors with
# split_threshold = 1, so pid 0 splits into singletons at 36 and 43 degrees.
# Afterwards:
#   * vid 0 keeps canonical owner pid 0 (36 deg) but its d1 collapses to 0, so
#     r -> inf and the replica in pid 1 must be REMOVED.
#   * vid 1 moves to the new posting (ownership change) and holds no replica.
# ---------------------------------------------------------------------------

VID_A_36, VID_A_43, VID_A_Q, VID_A_Q2, VID_A_R = 0, 1, 2, 3, 4
VID_A_RESERVE = 5  # 36.5 deg, inserted by the delete-then-split test
VID_A_RESERVE2 = 6  # 37.0 deg


def _fixture_a(
    delta: float | None = 0.5, merge_threshold: int | None = None
) -> tuple[InPlaceMaintainer, np.ndarray, Counters]:
    angles = (36.0, 43.0, 30.0, 29.0, 80.0, 36.5, 37.0)
    X = np.stack([_unit(a) for a in angles]).astype(np.float32)
    seg = _segment(
        X,
        [_centroid_midpoint([36.0, 43.0]), 30.0, 80.0],
        [[], [VID_A_Q, VID_A_Q2], [VID_A_R]],
    )
    maint, counters = _maintainer(
        X, seg, delta=delta, split_threshold=1, merge_threshold=merge_threshold
    )
    return maint, X, counters


def _fixture_a_live(*, inserted_43: bool = True) -> set[int]:
    live = {VID_A_36, VID_A_Q, VID_A_Q2, VID_A_R}
    if inserted_43:
        live.add(VID_A_43)
    return live


# ---------------------------------------------------------------------------
# Fixture B — a 4-point posting split into two 2-point clusters.
#
#   pid 0   centroid = mean of {4.0, 4.4, -4.0} deg, owns {0, 1, 2}
#   pid 1   centroid = mean of {8.0, 13.0} deg, owns {4: 8 deg, 5: 13 deg}
#
# Inserting vid 3 (-4.4 deg) gives pid 0 four owned vectors and splits it into
# {4.0, 4.4} and {-4.0, -4.4}, whose centroids are the (non-unit) means of the
# pairs, i.e. at +4.2 and -4.2 degrees.
#
# vid 4 (8 deg) is owned by pid 1 throughout.  Before the split its
# second-nearest posting is pid 0 at 1.47 deg (the 3-point mean), far enough
# that r > delta.  After the split the +4.2 piece is only 3.8 deg away, so
# r = d(3.8)/d(2.5) - 1 = 0.16 <= delta = 0.2 and the vector must gain exactly
# one replica — with its canonical owner unchanged.
# ---------------------------------------------------------------------------

VID_B_40, VID_B_44, VID_B_M40, VID_B_M44 = 0, 1, 2, 3
VID_B_Q8, VID_B_Q13 = 4, 5


def _fixture_b() -> tuple[InPlaceMaintainer, np.ndarray, Counters]:
    angles = (4.0, 4.4, -4.0, -4.4, 8.0, 13.0)
    X = np.stack([_unit(a) for a in angles]).astype(np.float32)
    seg = _segment(
        X,
        [_centroid_midpoint([4.0, 4.4, -4.0]), _centroid_midpoint([8.0, 13.0])],
        [[VID_B_40, VID_B_44, VID_B_M40], [VID_B_Q8, VID_B_Q13]],
    )
    maint, counters = _maintainer(X, seg, delta=0.2, split_threshold=3)
    return maint, X, counters


def _fixture_b_live() -> set[int]:
    return {VID_B_40, VID_B_44, VID_B_M40, VID_B_M44, VID_B_Q8, VID_B_Q13}


# ---------------------------------------------------------------------------
# Initial membership
# ---------------------------------------------------------------------------

def test_no_overlap_index_has_exactly_owner_membership():
    """delta=None keeps the single-membership behaviour and audits clean."""
    maint, X, counters = _fixture_a(delta=None)
    maint.insert(VID_A_36, X[VID_A_36])

    assert _member_pids(maint._segment, VID_A_36) == (0,)
    assert counters.replicas_added == 0
    _assert_audits(maint, _fixture_a_live(inserted_43=False))


def test_insert_creates_required_replica_immediately():
    """A Δ-eligible vector is a member of two postings right after insert."""
    maint, X, counters = _fixture_a()
    maint.insert(VID_A_36, X[VID_A_36])

    assert _owner_pids(maint._segment, VID_A_36) == (0,)
    assert _member_pids(maint._segment, VID_A_36) == (0, 1)
    assert counters.replicas_added == 1
    assert counters.replicas_removed == 0
    _assert_audits(maint, _fixture_a_live(inserted_43=False))


def test_insert_without_replica_stays_single_membership():
    """A vector whose runner-up is far away gains no replica."""
    maint, X, counters = _fixture_a()
    maint.insert(VID_A_43, X[VID_A_43])

    assert _owner_pids(maint._segment, VID_A_43) == (0,)
    assert _member_pids(maint._segment, VID_A_43) == (0,)
    # r = d(13 deg)/d(3.5 deg) - 1 ≈ 5.9, far above delta = 0.5
    assert counters.replicas_added == 0
    _assert_audits(maint, _fixture_a_live() - {VID_A_36})


def test_exact_tie_resolves_to_lowest_posting_id():
    """Two equidistant postings: the lower id owns, the other replicates.

    The similarities are bitwise equal (both centroids are exact float32
    mirror images of the probe), so this is a true tie, not an epsilon.
    """
    X = np.stack([_unit(0.0), _unit(10.0), _unit(-10.0)]).astype(np.float32)
    seg = _segment(X, [10.0, -10.0], [[], []])
    maint, counters = _maintainer(X, seg, delta=0.0, split_threshold=100)

    sims = maint._segment.levels[-1].centroids @ X[0]
    assert sims[0] == sims[1], "fixture must produce an exact tie"

    maint.insert(0, X[0])
    assert _owner_pids(maint._segment, 0) == (0,)
    assert _member_pids(maint._segment, 0) == (0, 1)
    assert counters.replicas_added == 1
    _assert_audits(maint, {0})


def test_zero_nearest_distance_grants_no_replica():
    """d1 == 0 makes the ratio undefined: only d2 == 0 may qualify."""
    X = np.stack([_unit(0.0), _unit(20.0)]).astype(np.float32)
    seg = _segment(X, [0.0, 20.0], [[], [1]])
    maint, counters = _maintainer(X, seg, delta=1.0, split_threshold=100)

    maint.insert(0, X[0])
    assert _member_pids(maint._segment, 0) == (0,)
    assert counters.replicas_added == 0
    _assert_audits(maint, {0, 1})


def test_coincident_centroids_tie_and_replicate():
    """Coincident centroids give d1 = d2 = 0: a tie the lower id wins."""
    X = np.stack([_unit(0.0)]).astype(np.float32)
    seg = _segment(X, [0.0, 0.0], [[], []])
    maint, counters = _maintainer(X, seg, delta=0.0, split_threshold=100)

    maint.insert(0, X[0])
    assert _owner_pids(maint._segment, 0) == (0,)
    assert _member_pids(maint._segment, 0) == (0, 1)
    assert counters.replicas_added == 1
    _assert_audits(maint, {0})


# ---------------------------------------------------------------------------
# The oracle must be able to fail
# ---------------------------------------------------------------------------

def _pair_fixture() -> tuple[InPlaceMaintainer, np.ndarray]:
    """Two postings with one replica-qualifying vector (vid 0) and one not."""
    X = np.stack([_unit(36.0), _unit(43.0), _unit(30.0)]).astype(np.float32)
    # vid 0 (36 deg) is eligible (r = 0.42 <= 0.5), so pid 1 holds it as a
    # replica while pid 0 owns it; vid 1 (43 deg) is not eligible.
    seg = _segment(
        X, [_centroid_midpoint([36.0, 43.0]), 30.0], [[0, 1], [2]],
        members=[[0, 1], [2, 0]],
    )
    maint, _ = _maintainer(X, seg, delta=0.5, split_threshold=100)
    return maint, X


def test_oracle_accepts_the_hand_built_state():
    maint, _ = _pair_fixture()
    assert _member_pids(maint._segment, 0) == (0, 1)
    audit = audit_membership(maint)
    assert audit.passed and audit.live_vectors_checked == 3


def test_oracle_detects_missing_replica():
    maint, _ = _pair_fixture()
    seg = maint._segment
    p = seg.postings[1]
    seg.postings[1] = Posting(p.id, p.centroid, p.members[p.members != 0], p.owned)

    audit = audit_membership(maint)
    assert not audit.passed
    assert [rec["vid"] for rec in audit.missing] == [0]
    rec = audit.missing[0]
    assert rec["owner_pid"] == 0
    assert rec["expected_pids"] == [0, 1]
    assert rec["actual_pids"] == [0]
    assert rec["missing_pids"] == [1]


def test_oracle_detects_obsolete_replica():
    """vid 1 is 13 deg from its owner: r ≈ 5.9, so pid 1 must not hold it."""
    maint, _ = _pair_fixture()
    seg = maint._segment
    p = seg.postings[1]
    seg.postings[1] = Posting(
        p.id, p.centroid, np.append(p.members, 1).astype(np.int64), p.owned
    )

    audit = audit_membership(maint)
    assert not audit.passed
    assert [rec["vid"] for rec in audit.extra] == [1]
    assert audit.extra[0]["extra_pids"] == [1]
    assert audit.extra[0]["expected_pids"] == [0]


def test_oracle_detects_duplicate_membership():
    maint, _ = _pair_fixture()
    seg = maint._segment
    p = seg.postings[0]
    seg.postings[0] = Posting(
        p.id, p.centroid, np.append(p.members, 0).astype(np.int64), p.owned
    )

    audit = audit_membership(maint)
    assert not audit.passed
    assert audit.duplicates == [{"vid": 0, "pid": 0, "count": 2}]


def test_oracle_detects_dead_membership():
    maint, X = _pair_fixture()
    maint._tombstones.add(2)
    maint._segment.mark_deleted(2)

    audit = audit_membership(maint)
    assert not audit.passed
    assert audit.dead_members == [{"vid": 2, "pid": 1}]
    assert audit.live_vectors_checked == 2


def test_oracle_detects_missing_from_owner():
    maint, _ = _pair_fixture()
    seg = maint._segment
    p = seg.postings[0]
    seg.postings[0] = Posting(
        p.id, p.centroid, p.members[p.members != 1], p.owned
    )

    audit = audit_membership(maint)
    assert not audit.passed
    assert [rec["vid"] for rec in audit.missing_from_owner] == [1]
    assert audit.missing_from_owner[0]["owner_pid"] == 0


def test_oracle_detects_ownership_conflict():
    maint, _ = _pair_fixture()
    seg = maint._segment
    p = seg.postings[1]
    seg.postings[1] = Posting(
        p.id, p.centroid, np.append(p.members, 1).astype(np.int64),
        np.append(p.owned, 1).astype(np.int64),
    )

    audit = audit_membership(maint)
    assert not audit.passed
    assert [rec["vid"] for rec in audit.ownership_conflicts] == [1]
    assert audit.ownership_conflicts[0]["owner_pids"] == [0, 1]


def test_oracle_does_not_mutate_the_index():
    maint, _ = _pair_fixture()
    before = _snapshot(maint)
    audit_membership(maint)
    after = _snapshot(maint)

    assert np.array_equal(before["centroids"], after["centroids"])
    assert all(
        b[0] == a[0] and np.array_equal(b[1], a[1])
        and np.array_equal(b[2], a[2]) and np.array_equal(b[3], a[3])
        for b, a in zip(before["postings"], after["postings"])
    )
    assert before["tombstones"] == after["tombstones"]


def test_oracle_rejects_a_non_inplace_maintainer():
    from vecstudy.maintain.outofplace import OutOfPlaceMaintainer

    X = make_synthetic(n=20, dim=8, seed=0)
    cfg_index = IndexConfig(dim=8, leaf_size=10, seed=0, delta=0.2)
    maint = OutOfPlaceMaintainer(
        X, cfg_index, MaintainConfig(strategy="outofplace"), Counters()
    )
    with pytest.raises(TypeError, match="InPlaceMaintainer"):
        audit_membership(maint)


# ---------------------------------------------------------------------------
# Splits
# ---------------------------------------------------------------------------

def test_split_removes_replica_when_owner_centroid_tightens():
    """vid 0 keeps owner pid 0 but loses the replica: d1 collapses to 0."""
    maint, X, counters = _fixture_a()
    maint.insert(VID_A_36, X[VID_A_36])
    assert _member_pids(maint._segment, VID_A_36) == (0, 1)

    maint.insert(VID_A_43, X[VID_A_43])  # triggers the split

    assert counters.splits == 1
    assert _owner_pids(maint._segment, VID_A_36) == (0,)
    assert _member_pids(maint._segment, VID_A_36) == (0,)
    assert counters.replicas_removed == 1
    _assert_audits(maint, _fixture_a_live())


def test_split_moves_vector_to_new_posting():
    """The 43 deg vector's nearest centroid becomes the new singleton."""
    maint, X, counters = _fixture_a()
    maint.insert(VID_A_36, X[VID_A_36])
    maint.insert(VID_A_43, X[VID_A_43])

    new_pid = len(maint._segment.postings) - 1
    assert counters.splits == 1
    assert _pid_at(maint, 36.0) == 0
    assert _pid_at(maint, 43.0) == new_pid
    assert _owner_pids(maint._segment, VID_A_43) == (new_pid,)
    assert _member_pids(maint._segment, VID_A_43) == (new_pid,)
    assert maint._segment.postings[0].owned.tolist() == [VID_A_36]
    _assert_audits(maint, _fixture_a_live())


def test_split_grants_replica_to_unchanged_owner():
    """vid 4 keeps owner pid 1 and gains the +4.2 deg piece as a replica."""
    maint, X, counters = _fixture_b()
    assert _member_pids(maint._segment, VID_B_Q8) == (1,)
    _assert_audits(maint, _fixture_b_live() - {VID_B_M44})

    maint.insert(VID_B_M44, X[VID_B_M44])  # triggers the split

    plus_pid = _pid_at(maint, 4.2)
    minus_pid = _pid_at(maint, -4.2)
    assert counters.splits == 1
    assert plus_pid != minus_pid

    assert _owner_pids(maint._segment, VID_B_Q8) == (1,)
    assert _member_pids(maint._segment, VID_B_Q8) == (1, plus_pid)
    assert counters.replicas_added == 1
    assert counters.replicas_removed == 0
    # The split posting's own vectors now sit on their centroid: d1 = 0.
    for vid in (VID_B_40, VID_B_44):
        assert _owner_pids(maint._segment, vid) == (plus_pid,)
        assert _member_pids(maint._segment, vid) == (plus_pid,)
    for vid in (VID_B_M40, VID_B_M44):
        assert _owner_pids(maint._segment, vid) == (minus_pid,)
    _assert_audits(maint, _fixture_b_live())


def test_split_recompute_visits_every_live_vector():
    """Membership after a split comes from a full pass, not a local patch."""
    maint, X, counters = _fixture_a()
    maint.insert(VID_A_36, X[VID_A_36])
    maint.insert(VID_A_43, X[VID_A_43])

    live = len(_fixture_a_live())
    assert counters.membership_vectors_recomputed == live


def test_recompute_is_idempotent():
    maint, X, counters = _fixture_b()
    maint.insert(VID_B_M44, X[VID_B_M44])
    after_split = [_snapshot(maint)["postings"][i] for i in range(3)]
    added, removed = counters.replicas_added, counters.replicas_removed
    recomputed = counters.membership_vectors_recomputed

    maint.rebuild_membership()

    assert counters.replicas_added == added
    assert counters.replicas_removed == removed
    assert counters.membership_vectors_recomputed == recomputed + len(_fixture_b_live())
    for pid, (_, centroid, members, owned) in enumerate(after_split):
        p = maint._segment.postings[pid]
        assert np.array_equal(p.members, members)
        assert np.array_equal(p.owned, owned)
    _assert_audits(maint, _fixture_b_live())


def test_membership_distances_are_not_charged_to_query_buckets():
    maint, X, counters = _fixture_b()
    counters.reset_search()
    maint.insert(VID_B_M44, X[VID_B_M44])

    assert counters.distance_computations["maintenance"] > 0
    assert "scan" not in counters.distance_computations
    assert counters.postings_read == 0 and counters.bytes_read == 0


# ---------------------------------------------------------------------------
# Deletions and rebuilds
# ---------------------------------------------------------------------------

def test_delete_removes_every_membership():
    maint, X, counters = _fixture_a()
    maint.insert(VID_A_36, X[VID_A_36])
    assert _member_pids(maint._segment, VID_A_36) == (0, 1)

    maint.delete(VID_A_36)

    assert _member_pids(maint._segment, VID_A_36) == ()
    assert _owner_pids(maint._segment, VID_A_36) == ()
    assert counters.replicas_removed == 1
    assert counters.replicas_added == 1  # unchanged by the delete
    _assert_audits(maint, _fixture_a_live(inserted_43=False) - {VID_A_36})


def test_repeated_delete_is_idempotent():
    maint, X, counters = _fixture_a()
    maint.insert(VID_A_36, X[VID_A_36])
    maint.delete(VID_A_36)
    removed = counters.replicas_removed

    maint.delete(VID_A_36)
    assert counters.replicas_removed == removed
    _assert_audits(maint, _fixture_a_live(inserted_43=False) - {VID_A_36})


def test_deleted_vector_is_not_returned_by_search():
    maint, X, counters = _fixture_a()
    maint.insert(VID_A_36, X[VID_A_36])
    maint.delete(VID_A_36)

    n_postings = len(maint._segment.postings)
    cfg_q = QueryConfig(k=5, beam_widths=(n_postings,), n_postings=n_postings)
    for query in (X[VID_A_36], X[VID_A_Q], X[VID_A_R]):
        found = {vid for vid, _ in search(query, maint.searchable_units(), cfg_q, counters)}
        assert VID_A_36 not in found


def test_external_tombstones_are_excluded_from_live_membership():
    """Tombstones already on a supplied segment must not stay as members.

    vid 4 (8 deg, owned by pid 1) is tombstoned before the maintainer is
    constructed.  A later split must neither keep it in `members` nor re-add
    it; the membership oracle would flag it as a dead membership otherwise.
    """
    X = np.stack([_unit(a) for a in (4.0, 4.4, -4.0, -4.4, 8.0, 13.0)]).astype(np.float32)
    seg = _segment(
        X,
        [_centroid_midpoint([4.0, 4.4, -4.0]), _centroid_midpoint([8.0, 13.0])],
        [[VID_B_40, VID_B_44, VID_B_M40], [VID_B_Q8, VID_B_Q13]],
        tombstones={VID_B_Q8},
    )
    maint, counters = _maintainer(X, seg, delta=0.2, split_threshold=3)
    assert maint._tombstones == {VID_B_Q8}

    maint.insert(VID_B_M44, X[VID_B_M44])  # triggers a split
    assert counters.splits == 1

    for p in maint._segment.postings:
        assert VID_B_Q8 not in p.members.tolist()
        assert VID_B_Q8 not in p.owned.tolist()
    _assert_audits(maint, _fixture_b_live() - {VID_B_Q8})


def test_split_after_delete_does_not_resurrect_the_vector():
    """A later split recomputes membership from live owned sets only."""
    # Merges are disabled (merge_threshold=0) so this test isolates the split
    # path: the emptied posting must stay in place and a later split must not
    # bring the tombstoned id back.  Merge behaviour is covered separately.
    maint, X, counters = _fixture_a(merge_threshold=0)
    maint.insert(VID_A_36, X[VID_A_36])
    maint.delete(VID_A_36)
    assert maint._segment.postings[0].owned.tolist() == []

    maint.insert(VID_A_RESERVE, X[VID_A_RESERVE])
    maint.insert(VID_A_RESERVE2, X[VID_A_RESERVE2])
    # The deleted vector's insert filled the posting but never split it, so the
    # two reserve inserts produce exactly one split — the one that must not
    # resurrect the tombstoned id.
    assert counters.splits == 1

    for p in maint._segment.postings:
        assert VID_A_36 not in p.members.tolist()
        assert VID_A_36 not in p.owned.tolist()

    live = {VID_A_Q, VID_A_Q2, VID_A_R, VID_A_RESERVE, VID_A_RESERVE2}
    _assert_audits(maint, live)

    n_postings = len(maint._segment.postings)
    cfg_q = QueryConfig(k=10, beam_widths=(n_postings,), n_postings=n_postings)
    found = {vid for vid, _ in search(X[VID_A_36], maint.searchable_units(), cfg_q, counters)}
    assert VID_A_36 not in found


# ---------------------------------------------------------------------------
# The production build path must follow the same policy
# ---------------------------------------------------------------------------

def test_rng_rule_is_a_build_path_option_only():
    """In-place membership is Δ-only; SPANN RNG filtering stays a build option.

    Geometry where the two rules disagree: postings at 30 and 33 degrees, a
    vector at 28 (owned by 30).  d2 = d(5 deg) is *larger* than the
    centroid-to-centroid distance d(3 deg), so `assign(..., rng_rule=True)`
    skips the replica while the fixed-Δ rule (r = 5.2 <= delta = 10) requires
    it.  Maintenance applies the Δ rule, which DESIGN §6.4 documents as the
    scope boundary rather than a silent metric switch.
    """
    from vecstudy.index.postings import assign

    X = np.stack([_unit(28.0), _unit(30.0), _unit(33.0)]).astype(np.float32)
    centroids = np.stack([_unit(30.0), _unit(33.0)]).astype(np.float32)
    seg = _segment(X, [30.0, 33.0], [[1], [2]])
    maint, counters = _maintainer(X, seg, delta=10.0, split_threshold=100)

    maint.insert(0, X[0])
    assert _member_pids(maint._segment, 0) == (0, 1)
    assert counters.replicas_added == 1
    _assert_audits(maint, {0, 1, 2})

    build_postings, _ = assign(
        X[:1], centroids, delta=10.0, rng_rule=True, counters=Counters()
    )
    assert [len(p.members) for p in build_postings] == [1, 0]


def test_segment_build_membership_matches_the_oracle():
    """`Segment.build`'s Δ assignment and the maintainer's rule agree."""
    X = make_synthetic(n=300, dim=16, seed=11)
    cfg_index = IndexConfig(
        dim=16, leaf_size=25, branching_factor=4, seed=0, depth=1, delta=0.5
    )
    counters = Counters()
    seg = Segment.build(np.arange(len(X), dtype=np.int64), X, cfg_index, counters)
    maint = InPlaceMaintainer(
        X, cfg_index, MaintainConfig(strategy="inplace", split_threshold=1000),
        counters, initial_segment=seg,
    )

    audit = audit_membership(maint)
    assert audit.passed, audit.to_dict()
    assert audit.live_vectors_checked == len(X)
    # The fixture must actually exercise overlap, or the test proves nothing.
    total_members = sum(len(p.members) for p in maint._segment.postings)
    assert total_members > len(X)
    _assert_audits(maint, set(range(len(X))))
