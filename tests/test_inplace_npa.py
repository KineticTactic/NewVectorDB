"""test_inplace_npa.py — NPA auditor diagnostics and split-reassignment regression.

Two independent concerns are tested here on purpose:

  * Auditor correctness  — the global NPA audit must find a *canonical-owner*
    violation when one is injected by hand, must ignore replicate membership,
    and must not mutate the index while checking.
  * Maintenance correctness — a production split of posting A can invalidate
    NPA for a vector owned by an unrelated posting B.  The fixture below is
    built so that B is outside the centroid-neighbourhood candidate set used by
    the old reassignment heuristic while one of A's new centroids is strictly
    closer to B's vector than B's own centroid.

The geometry (2-D unit vectors, angles in degrees, similarity = inner product,
distance = 1 - similarity, so angular order == similarity order):

    angle   role
    -----   ---------------------------------------------------------------
      0     centroid of posting A before the split (A owns 3 deg, 5 deg)
      3     A member (group 0)
      5     A member (group 0)
     43     A member (group 1)
     47     pending A member (group 1); inserting it triggers the split
    -1 .. -14.5   ten distractor postings, one vector each, on the far side
           of A from the probe so they never steal A's members or the probe
    120     centroid of posting B (B owns nothing yet)
     65     probe vector v, inserted into B through InPlaceMaintainer.insert

Before the split (all distances in degrees):

    v (65) -> B (120) = 55      <- owner, globally nearest
    v (65) -> A (0)   = 65
    v (65) -> d_i     >= 66

The split of A yields c0 = 4 deg and c1 = 45 deg (k-means over {3, 5} and
{43, 47}), so afterwards:

    v (65) -> c1 (45) = 20      <- strictly nearest, unique
    v (65) -> B (120) = 55      <- current owner: NPA violated
    v (65) -> d_i     >= 66

B never enters the neighbourhood heuristic: the ten distractors (plus A itself)
are all closer to A's old centroid, to c0 and to c1 than B is, so the old
`top n_reassign_neighbours` selection never looks at B's owned vector.
"""

from __future__ import annotations

import numpy as np

from vecstudy.core.config import IndexConfig, MaintainConfig
from vecstudy.core.metrics import Counters
from vecstudy.core.types import HierarchyLevel, Posting
from vecstudy.harness.churn import audit_maintainer
from vecstudy.harness.datasets import make_synthetic
from vecstudy.index.hierarchy import _stable_top_k
from vecstudy.index.segment import Segment
from vecstudy.maintain.inplace import InPlaceMaintainer, _ball_may_switch

# ---------------------------------------------------------------------------
# Geometry
# ---------------------------------------------------------------------------

A_OLD_ANGLE = 0.0
A_GROUP0_ANGLES = (3.0, 5.0)
A_GROUP1_ANGLES = (43.0, 47.0)
DISTRACTOR_ANGLES = (-1.0, -2.5, -4.0, -5.5, -7.0, -8.5, -10.0, -11.5, -13.0, -14.5)
B_ANGLE = 120.0
PROBE_ANGLE = 65.0
EXTRA_ANGLES = (35.0, 40.0, 44.0, 46.0, 50.0, 52.0, 58.0)

# Vector ids are row indices into the canonical vector store.  Rows for
# EXTRA_ANGLES are present in the store from the start but are only inserted
# into the index later, exactly as the churn harness keeps a reserve pool.
VID_A_GROUP0 = (0, 1)
VID_A_GROUP1_INDEXED = 2
VID_TRIGGER = 3
VID_DISTRACTORS = tuple(range(4, 4 + len(DISTRACTOR_ANGLES)))
VID_PROBE = 4 + len(DISTRACTOR_ANGLES)
VID_EXTRA = tuple(range(VID_PROBE + 1, VID_PROBE + 1 + len(EXTRA_ANGLES)))
PID_A = 0
PID_B = 1 + len(DISTRACTOR_ANGLES)
NEW_PID_AFTER_SPLIT = PID_B + 1  # postings are appended by _split


def _unit(angle_deg: float) -> np.ndarray:
    rad = np.deg2rad(angle_deg)
    return np.array([np.cos(rad), np.sin(rad)], dtype=np.float32)


def _angle_between(a: np.ndarray, b: np.ndarray) -> float:
    """Absolute angle in degrees between two 2-D unit vectors.

    Uses atan2 rather than arccos: close to 1 the cosine is flat in float32 and
    arccos amplifies the rounding error, which would make the fixture geometry
    look off by ~0.1 deg.
    """
    cross = float(a[0]) * float(b[1]) - float(a[1]) * float(b[0])
    dot = float(a[0]) * float(b[0]) + float(a[1]) * float(b[1])
    ang = abs(float(np.rad2deg(np.arctan2(cross, dot))))
    return min(ang, 360.0 - ang)


def _fixture_vectors() -> np.ndarray:
    angles = [
        *A_GROUP0_ANGLES,
        *A_GROUP1_ANGLES,
        *DISTRACTOR_ANGLES,
        PROBE_ANGLE,
        *EXTRA_ANGLES,
    ]
    return np.stack([_unit(a) for a in angles]).astype(np.float32)


def _fixture_segment(X: np.ndarray) -> Segment:
    """Explicit pre-split segment: A, ten distractors, one empty posting B."""
    postings: list[Posting] = [
        Posting(
            id=PID_A,
            centroid=_unit(A_OLD_ANGLE),
            members=np.array([*VID_A_GROUP0, VID_A_GROUP1_INDEXED], dtype=np.int64),
            owned=np.array([*VID_A_GROUP0, VID_A_GROUP1_INDEXED], dtype=np.int64),
        )
    ]
    for offset, (angle, vid) in enumerate(zip(DISTRACTOR_ANGLES, VID_DISTRACTORS)):
        postings.append(
            Posting(
                id=1 + offset,
                centroid=_unit(angle),
                members=np.array([vid], dtype=np.int64),
                owned=np.array([vid], dtype=np.int64),
            )
        )
    postings.append(
        Posting(
            id=PID_B,
            centroid=_unit(B_ANGLE),
            members=np.array([], dtype=np.int64),
            owned=np.array([], dtype=np.int64),
        )
    )
    leaf = HierarchyLevel(
        centroids=np.stack([p.centroid for p in postings]).astype(np.float32),
        children=[],
    )
    return Segment(levels=[leaf], postings=postings, vectors=X)


def _make_maintainer(
    *,
    split_threshold: int = 3,
    n_reassign_neighbours: int = 8,
) -> tuple[InPlaceMaintainer, np.ndarray, Counters]:
    X = _fixture_vectors()
    cfg_index = IndexConfig(
        dim=X.shape[1], leaf_size=100, branching_factor=4, seed=0, depth=1
    )
    cfg_maintain = MaintainConfig(
        strategy="inplace",
        split_threshold=split_threshold,
        n_reassign_neighbours=n_reassign_neighbours,
    )
    counters = Counters()
    maint = InPlaceMaintainer(
        X, cfg_index, cfg_maintain, counters, initial_segment=_fixture_segment(X)
    )
    return maint, X, counters


def _live_vids(include_trigger: bool) -> set[int]:
    vids = set(VID_A_GROUP0) | {VID_A_GROUP1_INDEXED, *VID_DISTRACTORS, VID_PROBE}
    if include_trigger:
        vids.add(VID_TRIGGER)
    return vids


def _prepare_probe(maint: InPlaceMaintainer, X: np.ndarray) -> None:
    """Insert the probe through the production path; it must land in B."""
    maint.insert(VID_PROBE, X[VID_PROBE])
    assert _owner_pids(maint._segment, VID_PROBE) == [PID_B], (
        "probe should start in posting B"
    )


def _nearest_pid(seg: Segment, vid: int) -> tuple[int, float, float]:
    """(nearest posting id, nearest similarity, runner-up similarity)."""
    centroids = seg.levels[-1].centroids
    sims = centroids @ seg.vectors[vid].astype(np.float32)
    order = np.argsort(-sims, kind="stable")
    return int(order[0]), float(sims[order[0]]), float(sims[order[1]])


def _owner_pids(seg: Segment, vid: int) -> list[int]:
    return [p.id for p in seg.postings if vid in p.owned]


def _move_ownership(seg: Segment, vid: int, src_pid: int, dst_pid: int) -> None:
    """Hand *vid* from one posting to another, keeping owned ⊆ members."""
    src = seg.postings[src_pid]
    dst = seg.postings[dst_pid]
    seg.postings[src_pid] = Posting(
        src.id, src.centroid,
        src.members[src.members != vid], src.owned[src.owned != vid],
    )
    seg.postings[dst_pid] = Posting(
        dst.id, dst.centroid,
        np.append(dst.members, vid).astype(np.int64),
        np.append(dst.owned, vid).astype(np.int64),
    )


# ---------------------------------------------------------------------------
# Auditor correctness
# ---------------------------------------------------------------------------


def _segment_snapshot(seg: Segment) -> dict:
    """Immutable snapshot used to prove the auditor does not mutate the index."""
    return {
        "centroids": seg.levels[-1].centroids.copy(),
        "postings": [
            (p.id, p.centroid.copy(), p.members.copy(), p.owned.copy())
            for p in seg.postings
        ],
        "tombstones": set(seg.tombstones),
        "levels": len(seg.levels),
    }


def test_auditor_accepts_valid_index():
    """A correctly built in-place index audits clean and reports full coverage."""
    maint, X, _ = _make_maintainer()
    _prepare_probe(maint, X)

    audit = audit_maintainer(maint, expected_live_vids=_live_vids(include_trigger=False))

    assert audit.passed is True
    assert audit.violation_count == 0
    assert audit.npa_violations == []
    assert audit.examples == []
    assert audit.live_vectors_checked == len(_live_vids(include_trigger=False))
    assert audit.postings_checked == PID_B + 1
    assert audit.ownership_errors == []
    assert audit.coverage_errors == []


def test_auditor_detects_deliberately_invalid_owner():
    """The auditor reports a hand-injected wrong canonical owner."""
    maint, X, _ = _make_maintainer()
    _prepare_probe(maint, X)

    victim = VID_DISTRACTORS[0]
    src_pid = 1
    _move_ownership(maint._segment, victim, src_pid=src_pid, dst_pid=PID_B)

    audit = audit_maintainer(maint, expected_live_vids=_live_vids(include_trigger=False))

    assert audit.passed is False
    assert audit.violation_count == len(audit.npa_violations) == 1
    assert audit.live_vectors_checked == len(_live_vids(include_trigger=False))
    assert audit.ownership_errors == []  # ownership is still unique and ⊆ members
    assert audit.coverage_errors == []

    example = audit.examples[0]
    assert example["vid"] == victim
    assert example["owned_pid"] == PID_B
    assert example["nearest_pid"] == src_pid
    assert example["best_sim"] > example["own_sim"]
    assert example["nearest_distance"] < example["own_distance"]
    assert example["margin"] > 0.1

    as_dict = audit.to_dict()
    assert as_dict["passed"] is False
    assert as_dict["violation_count"] == 1
    assert as_dict["live_vectors_checked"] == len(_live_vids(include_trigger=False))
    assert as_dict["postings_checked"] == PID_B + 1


def test_auditor_does_not_mutate_index():
    """Auditing an invalid index must report, not repair."""
    maint, X, _ = _make_maintainer()
    _prepare_probe(maint, X)
    seg = maint._segment
    _move_ownership(seg, VID_DISTRACTORS[0], src_pid=1, dst_pid=PID_B)

    before = _segment_snapshot(seg)
    audit_maintainer(maint, expected_live_vids=_live_vids(include_trigger=False))
    after = _segment_snapshot(seg)

    assert before["tombstones"] == after["tombstones"]
    assert before["levels"] == after["levels"]
    np.testing.assert_array_equal(before["centroids"], after["centroids"])
    assert len(before["postings"]) == len(after["postings"])
    for (bid, bc, bm, bo), (aid, ac, am, ao) in zip(
        before["postings"], after["postings"]
    ):
        assert bid == aid
        np.testing.assert_array_equal(bc, ac)
        np.testing.assert_array_equal(bm, am)
        np.testing.assert_array_equal(bo, ao)


def test_auditor_uses_ownership_not_membership():
    """A replicate member that is owned elsewhere is not an NPA violation."""
    maint, X, _ = _make_maintainer()
    _prepare_probe(maint, X)
    seg = maint._segment

    # Give posting A a replicate copy of the probe: member, not owner.
    p_a = seg.postings[PID_A]
    assert VID_PROBE not in p_a.members
    seg.postings[PID_A] = Posting(
        p_a.id, p_a.centroid,
        np.append(p_a.members, VID_PROBE).astype(np.int64),
        p_a.owned.copy(),
    )
    assert VID_PROBE in seg.postings[PID_A].members
    assert _owner_pids(seg, VID_PROBE) == [PID_B]

    audit = audit_maintainer(maint, expected_live_vids=_live_vids(include_trigger=False))

    assert audit.passed is True
    assert audit.violation_count == 0


def test_auditor_excludes_tombstoned_vectors():
    """Deleted vectors are not audited for NPA and are not live owners."""
    maint, X, _ = _make_maintainer()
    _prepare_probe(maint, X)
    seg = maint._segment

    victim = VID_DISTRACTORS[0]
    maint.delete(victim)
    assert victim in seg.tombstones

    # Re-attach the deleted vector to the far posting.  If tombstoned vectors
    # were audited this would be a gross NPA violation.
    _move_ownership(seg, victim, src_pid=1, dst_pid=PID_B)

    expected_live = _live_vids(include_trigger=False) - {victim}
    audit = audit_maintainer(maint, expected_live_vids=expected_live)

    assert audit.passed is True
    assert audit.live_vectors_checked == len(expected_live)

    # Sanity: the very same ownership state is reported once it is live again.
    seg.tombstones.discard(victim)
    reported = audit_maintainer(maint, expected_live_vids=_live_vids(include_trigger=False))
    assert reported.violation_count == 1
    assert reported.examples[0]["vid"] == victim


# ---------------------------------------------------------------------------
# Split regression: neighbouring posting NPA violation
# ---------------------------------------------------------------------------


def test_split_reassigns_neighbouring_vector():
    """A split of A must hand the probe to its new nearest posting."""
    maint, X, counters = _make_maintainer(split_threshold=3, n_reassign_neighbours=8)
    seg = maint._segment
    _prepare_probe(maint, X)

    # ---- pre-split: the index is valid and the probe sits in B -------------
    pre = audit_maintainer(maint, expected_live_vids=_live_vids(include_trigger=False))
    assert pre.passed is True, pre.npa_violations
    assert _owner_pids(seg, VID_PROBE) == [PID_B]

    probe = X[VID_PROBE]
    pre_nearest, pre_best_sim, _ = _nearest_pid(seg, VID_PROBE)
    assert pre_nearest == PID_B
    assert _angle_between(probe, seg.postings[PID_A].centroid) > _angle_between(
        probe, seg.postings[PID_B].centroid
    )

    # ---- production split of A (4th owned vector crosses split_threshold) --
    splits_before = counters.splits
    maint.insert(VID_TRIGGER, X[VID_TRIGGER])

    assert counters.splits == splits_before + 1
    assert len(seg.postings) == NEW_PID_AFTER_SPLIT + 1
    assert seg.levels[-1].centroids.shape[0] == len(seg.postings)

    split_pids = {PID_A, NEW_PID_AFTER_SPLIT}
    split_angles = sorted(
        round(_angle_between(seg.postings[pid].centroid, _unit(A_OLD_ANGLE)), 1)
        for pid in split_pids
    )
    # The split produced the 4 deg / 45 deg pair (which posting holds which is
    # the k-means label order, which the test does not pin).
    assert split_angles == [4.0, 45.0]

    # ---- the new centroid is strictly nearer to the probe than B's --------
    nearest_pid, best_sim, runner_up_sim = _nearest_pid(seg, VID_PROBE)
    assert nearest_pid in split_pids
    assert _angle_between(probe, seg.levels[-1].centroids[nearest_pid]) < 25.0
    assert best_sim - runner_up_sim > 1e-2, "expected a unique nearest posting"
    assert best_sim > pre_best_sim  # the split really did move a centroid closer

    # ---- the maintenance path must have handed the probe over -------------
    owners = _owner_pids(seg, VID_PROBE)
    assert owners == [nearest_pid], (
        f"probe left owned by {owners} but its nearest posting is {nearest_pid}"
    )

    # ---- global invariants after the split -------------------------------
    post = audit_maintainer(maint, expected_live_vids=_live_vids(include_trigger=True))
    assert post.violation_count == 0, post.npa_violations
    assert post.passed is True, f"{post.summary()}; examples={post.examples}"
    assert post.postings_checked == len(seg.postings)

    # exactly one canonical owner per live vector, owned ⊆ members
    owners_count: dict[int, int] = {}
    for p in seg.postings:
        assert set(int(v) for v in p.owned) <= set(int(v) for v in p.members)
        for v in p.owned:
            owners_count[int(v)] = owners_count.get(int(v), 0) + 1
    assert set(owners_count) == _live_vids(include_trigger=True)
    assert all(c == 1 for c in owners_count.values())


def test_subsequent_inserts_and_splits_keep_npa():
    """Later churn in the split region must not reintroduce the violation."""
    maint, X, counters = _make_maintainer(split_threshold=3, n_reassign_neighbours=8)
    _prepare_probe(maint, X)
    maint.insert(VID_TRIGGER, X[VID_TRIGGER])
    assert counters.splits >= 1

    live = _live_vids(include_trigger=True)
    for offset, (vid, angle) in enumerate(zip(VID_EXTRA, EXTRA_ANGLES)):
        maint.insert(vid, X[vid])
        live.add(vid)
        audit = audit_maintainer(maint, expected_live_vids=live)
        assert audit.violation_count == 0, (
            f"insert of {vid} ({angle} deg) broke NPA: {audit.examples}"
        )

    final = audit_maintainer(maint, expected_live_vids=live)
    assert final.passed is True
    assert final.live_vectors_checked == len(live)

    # Tombstoning a vector must remove it from the live NPA check.
    maint.delete(VID_TRIGGER)
    live_after_delete = live - {VID_TRIGGER}
    after_delete = audit_maintainer(maint, expected_live_vids=live_after_delete)
    assert after_delete.passed is True
    assert after_delete.live_vectors_checked == len(live_after_delete)


def test_pruning_bound_includes_exact_ties():
    """The pruning bound must admit the tie boundary, not just strict wins.

    Canonical ownership uses the stable ``np.argmax`` tie-break: on an exact
    tie the lower posting id owns the vector.  A reachable tie is therefore a
    reachable ownership change, so the bound may prune only when its maximum is
    strictly below the competitor.

    These arguments are exact in IEEE double, so the boundary is tested without
    relying on rounding.
    """
    # max over the ball is exactly 0 -> a tie is reachable -> must not prune
    assert _ball_may_switch(0.0, 0.5, 0.5, 1.0) is True
    assert _ball_may_switch(0.0, 0.0, 0.0, 1.0) is True
    # coincident centroids: every vector ties, so ownership may still move
    assert _ball_may_switch(1.0, 1.0, 0.0, 0.0) is True
    # strictly below the boundary -> provably safe to prune
    assert _ball_may_switch(0.0, 0.5, 0.49, 1.0) is False
    assert _ball_may_switch(0.25, 0.5, 0.2, 1.0) is False
    # strictly above the boundary -> must not prune
    assert _ball_may_switch(0.0, 0.5, 0.51, 1.0) is True


def test_exact_tie_resolves_to_lowest_posting_id():
    """An exact tie between the owner and a new split centroid moves to the lower id.

    Geometry (2-D unit vectors; the probe is exactly (1, 0), so inner products
    with it are just the x components and ties are exact in float32):

        probe v      = (1, 0)
        A's member u = (0.5, +sqrt(3)/2)   -> v·u = 0.5
        B centroid   = (0.5, -sqrt(3)/2)   -> v·B = 0.5   (exact tie)
        A's centroid = (0, 1)
        A's other member w = 150 deg
        ten distractors at 225..297 deg

    B (id 11) owns v before the split.  Inserting the second A member splits A
    into u and w; ``balanced_kmeans`` returns the two singletons as centroids
    exactly, so u becomes posting A's centroid (id 0).  Under the documented
    tie-break the probe must move from B to posting 0.
    """
    x, y = np.float32(0.5), np.float32(np.sqrt(3.0) / 2.0)
    probe = np.array([1.0, 0.0], dtype=np.float32)
    u = np.array([x, y], dtype=np.float32)
    w = np.array([-y, x], dtype=np.float32)
    a_old = np.array([0.0, 1.0], dtype=np.float32)
    b_centroid = np.array([x, -y], dtype=np.float32)
    distractor_angles = (225.0, 233.0, 241.0, 249.0, 257.0,
                         265.0, 273.0, 281.0, 289.0, 297.0)

    vids: list[tuple[int, np.ndarray]] = [(0, u), (1, w)]
    for offset, angle in enumerate(distractor_angles):
        vids.append((2 + offset, _unit(angle)))
    vid_probe = 2 + len(distractor_angles)
    vids.append((vid_probe, probe))
    X = np.stack([v for _, v in vids]).astype(np.float32)

    pid_b = 1 + len(distractor_angles)
    postings = [Posting(0, a_old, np.array([0], np.int64), np.array([0], np.int64))]
    for offset in range(len(distractor_angles)):
        vid = 2 + offset
        postings.append(Posting(1 + offset, X[vid], np.array([vid], np.int64),
                                np.array([vid], np.int64)))
    postings.append(Posting(pid_b, b_centroid, np.array([], np.int64),
                            np.array([], np.int64)))
    leaf = HierarchyLevel(
        centroids=np.stack([p.centroid for p in postings]).astype(np.float32),
        children=[],
    )
    seg = Segment(levels=[leaf], postings=postings, vectors=X)

    cfg_index = IndexConfig(dim=2, leaf_size=100, branching_factor=4, seed=0, depth=1)
    cfg_maintain = MaintainConfig(strategy="inplace", split_threshold=1)
    maint = InPlaceMaintainer(X, cfg_index, cfg_maintain, Counters(), initial_segment=seg)
    seg = maint._segment  # __init__ clones the supplied segment

    # probe joins B; the index is still valid
    maint.insert(vid_probe, X[vid_probe])
    assert _owner_pids(seg, vid_probe) == [pid_b]
    pre = audit_maintainer(maint, expected_live_vids={0, *range(2, len(X))})
    assert pre.passed is True, pre.npa_violations

    # second A member splits A into the two singletons
    maint.insert(1, X[1])

    sims = seg.levels[-1].centroids @ probe
    best = float(sims.max())
    tied = sorted(int(p) for p in np.flatnonzero(sims == sims.max()))
    assert best == 0.5, f"fixture is not an exact tie: best={best}"
    assert tied == [0, pid_b], f"expected an exact tie between A and B, got {tied}"

    # stable tie-break: the lower posting id is the canonical owner
    assert _owner_pids(seg, vid_probe) == [min(tied)] == [0]

    audit = audit_maintainer(maint, expected_live_vids=set(range(len(X))))
    assert audit.violation_count == 0
    assert audit.passed is True, audit.examples


def test_random_workload_with_splits_stays_npa_clean():
    """A plain insertion workload with 30+ splits must stay NPA-clean.

    This is not a hand-built geometry: it is the ordinary "insert everything"
    path on synthetic data.  The old centroid-neighbourhood reassignment left
    three violated owners after 700 inserts on this seed (vectors 66, 462 and
    513, each owned by a posting that a later split's new centroid beat).
    """
    X = make_synthetic(n=1200, dim=8, n_topics=8, seed=3)
    n_insert = 700
    counters = Counters()
    maint = InPlaceMaintainer(
        X,
        IndexConfig(dim=8, leaf_size=25, branching_factor=4, seed=3),
        MaintainConfig(strategy="inplace", split_threshold=40),
        counters,
    )
    for vid in range(n_insert):
        maint.insert(vid, X[vid])

    audit = audit_maintainer(maint, expected_live_vids=set(range(n_insert)))

    assert counters.splits > 20
    assert audit.passed is True, f"{audit.summary()}; examples={audit.examples}"
    assert audit.live_vectors_checked == n_insert
    assert audit.postings_checked == len(maint._segment.postings)


# ---------------------------------------------------------------------------
# R1 Task 3 — pruning / tie-boundary regression
#
# Ownership is ``np.argmax`` over the leaf centroids, so an exact score tie
# resolves to the lowest posting id.  The split candidate filter prunes a
# posting only through ``_ball_may_switch``; these tests pin the four boundary
# cases of that predicate end-to-end (the owner the production path must
# produce), plus the float round-off slack that separates a real proof from a
# rounded bound.
# ---------------------------------------------------------------------------


def _two_posting_maintainer(
    first_vec: np.ndarray,
    second_vec: np.ndarray,
    probe: np.ndarray,
    a_old: np.ndarray,
    b_centroid: np.ndarray,
    *,
    delta: float | None = None,
    split_threshold: int = 1,
) -> tuple[InPlaceMaintainer, np.ndarray]:
    """A owning {vid0}, B owning {vid2}; inserting vid1 splits A into singletons.

    ``balanced_kmeans`` with two points and two clusters is the identity, so
    ``first_vec`` becomes posting 0's centroid (the reused, lower id) and
    ``second_vec`` becomes the appended posting's centroid [2].  B keeps id 1.
    """
    X = np.stack([first_vec, second_vec, probe]).astype(np.float32)
    postings = [
        Posting(0, np.asarray(a_old, dtype=np.float32),
                np.array([0], dtype=np.int64), np.array([0], dtype=np.int64)),
        Posting(1, np.asarray(b_centroid, dtype=np.float32),
                np.array([2], dtype=np.int64), np.array([2], dtype=np.int64)),
    ]
    leaf = HierarchyLevel(
        centroids=np.stack([p.centroid for p in postings]).astype(np.float32),
        children=[],
    )
    seg = Segment(levels=[leaf], postings=postings, vectors=X)
    cfg_index = IndexConfig(
        dim=X.shape[1], leaf_size=100, branching_factor=4, seed=0, depth=1,
        delta=delta,
    )
    cfg_maint = MaintainConfig(strategy="inplace", split_threshold=split_threshold)
    maint = InPlaceMaintainer(X, cfg_index, cfg_maint, Counters(), initial_segment=seg)
    return maint, X


def _strict_owner(seg: Segment, vid: int) -> int:
    sims = seg.levels[-1].centroids @ seg.vectors[vid].astype(np.float32)
    return int(np.argmax(sims))


def test_bound_prunes_strictly_worse_candidate():
    """A posting whose ball bound is strictly below the competitor is skipped.

    Predicate form: the exact float arguments put the maximum clearly below
    zero.  End-to-end form: B's ball degenerates to the probe itself and the
    split's new centroids are far worse, so B is a non-source and the probe
    must keep B as its owner.
    """
    assert _ball_may_switch(0.0, 0.5, 0.49, 1.0) is False
    assert _ball_may_switch(0.25, 0.5, 0.2, 1.0) is False

    probe = _unit(0.0)
    maint, X = _two_posting_maintainer(
        _unit(120.0), _unit(130.0), probe, _unit(125.0), probe
    )
    maint.insert(1, X[1])
    seg = maint._segment
    assert len(seg.postings) == 3
    # probe . first = cos 120 = -0.5, probe . B = 1 -> bound = -1.5, pruned.
    assert _owner_pids(seg, 2) == [1]
    assert _strict_owner(seg, 2) == 1
    audit = audit_maintainer(maint, expected_live_vids={0, 1, 2})
    assert audit.passed is True, audit.examples


def test_tie_with_lower_id_target_moves_owner():
    """A reachable exact tie is resolved to the lower posting id.

    B (id 1) owns the probe.  The split reuses id 0 with a centroid that
    ``probe`` scores exactly equally against, so the probe must move to 0.  This
    is the case a ``> 0`` (strict) prune would have missed.
    """
    probe = _unit(0.0)
    maint, X = _two_posting_maintainer(
        _unit(60.0), _unit(65.0), probe, _unit(70.0), _unit(-60.0)
    )
    maint.insert(1, X[1])
    seg = maint._segment
    assert len(seg.postings) == 3

    sims = seg.levels[-1].centroids @ probe
    assert sims[0] == sims[1], "fixture must produce an exact tie between 0 and 1"
    assert sims[0] == np.float32(0.5)

    assert _owner_pids(seg, 2) == [0]
    assert _strict_owner(seg, 2) == 0
    audit = audit_maintainer(maint, expected_live_vids={0, 1, 2})
    assert audit.passed is True, audit.examples


def test_tie_with_lower_id_owner_keeps_owner():
    """An exact tie against a higher-id target leaves the lower-id owner in place.

    B (id 1) owns the probe; the split's appended centroid (id 2) scores exactly
    equally against it, while the reused centroid (id 0) is strictly worse.  The
    tie is B's, so the probe must stay with B.
    """
    probe = _unit(0.0)
    maint, X = _two_posting_maintainer(
        _unit(150.0), _unit(60.0), probe, _unit(70.0), _unit(-60.0)
    )
    maint.insert(1, X[1])
    seg = maint._segment
    assert len(seg.postings) == 3

    sims = seg.levels[-1].centroids @ probe
    assert sims[1] == sims[2], "fixture must tie pid 1 (owner) with pid 2"
    assert sims[0] < sims[1]

    assert _owner_pids(seg, 2) == [1]
    assert _strict_owner(seg, 2) == 1
    audit = audit_maintainer(maint, expected_live_vids={0, 1, 2})
    assert audit.passed is True, audit.examples


def test_bound_roundoff_near_tie_is_not_pruned():
    """A bound a few ULP below zero is not a proof, so it must not prune.

    The bound is assembled from float32 similarities, a float32 radius and a
    float32 squared distance; near zero it is only trustworthy up to a small
    slack.  A reachable tie whose computed bound lands at ``-1e-9`` is admitted
    (over-inclusion is the safe direction); only a clearly negative bound
    prunes.
    """
    # exact boundary (zero) is admitted -- present regression, kept
    assert _ball_may_switch(0.0, 0.5, 0.5, 1.0) is True
    # round-off below zero: still admitted
    assert _ball_may_switch(0.0, 0.5, 0.5 - 1e-9, 1.0) is True
    assert _ball_may_switch(0.0, 0.5, 0.5 - 1e-12, 1.0) is True
    # strictly above zero: admitted
    assert _ball_may_switch(0.0, 0.5, 0.5 + 1e-9, 1.0) is True
    # clearly negative: pruned
    assert _ball_may_switch(0.0, 0.5, 0.5 - 1e-2, 1.0) is False
    # coincident centroids: admitted (reachable tie, lower id wins)
    assert _ball_may_switch(1.0, 1.0, 0.0, 0.0) is True


def test_near_tie_resolves_to_strict_argmax_owner():
    """A one-ULP score margin still resolves to the strict ``argmax`` owner.

    The split's reused centroid scores ``probe`` one float32 ULP above B's, so
    the production metric ranks it strictly first.  The maintenance path must
    produce that same owner, not fall back to the incumbent on a "near tie".
    """
    one_ulp_up = np.nextafter(np.float32(0.5), np.float32(1.0))
    assert one_ulp_up > np.float32(0.5)
    y = np.float32(np.sqrt(max(0.0, 1.0 - float(one_ulp_up) ** 2)))
    target = np.array([one_ulp_up, y], dtype=np.float32)

    probe = _unit(0.0)
    maint, X = _two_posting_maintainer(
        target, _unit(65.0), probe, _unit(70.0), _unit(-60.0)
    )
    maint.insert(1, X[1])
    seg = maint._segment

    sims = seg.levels[-1].centroids @ probe
    assert sims[0] > sims[1], "fixture must give pid 0 a strict one-ULP edge"
    assert sims[0] - sims[1] < np.float32(1e-6)

    assert _owner_pids(seg, 2) == [0]
    assert _owner_pids(seg, 2) == [_strict_owner(seg, 2)]
    audit = audit_maintainer(maint, expected_live_vids={0, 1, 2})
    assert audit.passed is True, audit.examples


def test_split_moves_owner_outside_old_neighbourhood():
    """The repaired split moves a vector the centroid-neighbour list never saw.

    Rebuilds the Task-1 geometry (a probe owned by B, far from the split's
    centroid neighbourhood) and additionally reconstructs the *old* heuristic's
    candidate set to prove it excludes B, i.e. the move is only possible because
    the bounding-radius filter is sound.
    """
    maint, X, _ = _make_maintainer(split_threshold=3, n_reassign_neighbours=8)
    seg = maint._segment
    _prepare_probe(maint, X)
    orig_centroid = seg.postings[PID_A].centroid.copy()

    maint.insert(VID_TRIGGER, X[VID_TRIGGER])
    assert len(seg.postings) == NEW_PID_AFTER_SPLIT + 1
    split_pids = [p.id for p in seg.postings
                  if int(p.id) in (PID_A, NEW_PID_AFTER_SPLIT)]
    assert set(split_pids) == {PID_A, NEW_PID_AFTER_SPLIT}

    leaf = seg.levels[-1].centroids
    K = len(leaf)
    n_search = min(8 + 2, K)
    sims_orig = leaf @ orig_centroid
    sims_c0 = leaf @ leaf[PID_A]
    sims_c1 = leaf @ leaf[NEW_PID_AFTER_SPLIT]
    old_candidates: set[int] = set()
    for s in (sims_orig, sims_c0, sims_c1):
        old_candidates.update(int(p) for p in _stable_top_k(-s, n_search))

    # The old heuristic re-scores only the postings it selected.  B -- the
    # probe's owner -- is not among them, so the probe was never re-examined.
    assert PID_B not in old_candidates, (
        "fixture no longer exercises the stale-centroid case"
    )
    # The repaired filter did examine B (its bounding ball reaches a new
    # centroid) and moved the probe to its true nearest posting.
    new_owner = _strict_owner(seg, VID_PROBE)
    assert new_owner in (PID_A, NEW_PID_AFTER_SPLIT)
    assert _owner_pids(seg, VID_PROBE) == [new_owner]
    assert _owner_pids(seg, VID_PROBE) != [PID_B]
    audit = audit_maintainer(maint, expected_live_vids=_live_vids(include_trigger=True))
    assert audit.passed is True, audit.examples
