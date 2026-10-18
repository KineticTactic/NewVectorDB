"""maintain/inplace.py — in-place maintenance strategy.

DESIGN §6.6: one segment mutated in place.
Insert → nearest leaf → append → split if over threshold.
After split: neighbourhood reassignment (not local-only — §2 critical rule).
Promote centroids upward; recurse on overflow; add level if root splits.
Records: splits, vectors_reassigned, npa_violations_detected, depth.
"""

from __future__ import annotations

import math

import numpy as np

from ..core.types import VectorId, Posting
from ..core.config import IndexConfig, MaintainConfig
from ..core.metrics import Counters
from ..core.distance import sim_one_to_many, sim_many_to_many
from ..core.clustering import balanced_kmeans
from ..index.hierarchy import build_hierarchy, _stable_top_k
from ..index.postings import posting_add_vector
from ..index.segment import Segment
from .base import IndexMaintainer


# ---------------------------------------------------------------------------
# Geometry helpers for sound candidate selection after a split
# ---------------------------------------------------------------------------

def _owned_spread(vectors: np.ndarray, centroid: np.ndarray,
                  owned: np.ndarray) -> float:
    """Largest Euclidean distance from *centroid* to any vector it owns.

    This is a maintenance-side bounding radius, not a counted query distance:
    it never touches the distance counter buckets.  It is what makes the
    neighbourhood candidate set below sound — a posting whose centroid is far
    from a new centroid can still hold a vector that the new centroid beats.
    """
    if len(owned) == 0:
        return 0.0
    X = vectors[owned].astype(np.float32)
    return float(np.max(np.linalg.norm(X - centroid.astype(np.float32), axis=1)))


def _ball_may_switch(
    ref_dot_target: float,
    ref_dot_competitor: float,
    radius: float,
    target_minus_competitor_sq: float,
) -> bool:
    """Could some ``x`` with ``||x - ref|| <= radius`` satisfy
    ``x·target >= x·competitor``?

    The comparison is inclusive because canonical ownership uses the same stable
    tie-break as ``np.argmax`` over the leaf centroids: when two postings score
    a vector equally, the **lower posting id** owns it.  A reachable tie is
    therefore a reachable ownership change, so pruning is only allowed when the
    bound is *strictly* below the competitor.

    ``ref_dot_target`` / ``ref_dot_competitor`` are inner products of the ball
    centre with the two centroids; ``target_minus_competitor_sq`` is
    ``||target - competitor||²``.  Over the ball the inner-product difference
    ``x·(target - competitor)`` is maximised at
    ``ref·(target - competitor) + radius·||target - competitor||``; a strictly
    negative maximum is a proof that no owned vector can tie or switch.
    """
    if target_minus_competitor_sq <= 0.0:
        # Coincident centroids: every vector in the ball ties, and the lower
        # posting id wins the tie, so ownership may change.
        return True
    return (
        ref_dot_target - ref_dot_competitor
        + radius * math.sqrt(target_minus_competitor_sq)
    ) >= 0.0


# ---------------------------------------------------------------------------
# Fixed-Δ replica membership
# ---------------------------------------------------------------------------

# Rows scored per chunk in a full membership pass; 32K x K float32 keeps peak
# memory around 160 MB at K = 1250, matching index.postings.assign.
MEMBERSHIP_CHUNK = 32_768


def _clamp_distance(sim: float) -> float:
    """Production distance ``1 - similarity``, floored at zero.

    Similarity is an inner product and centroids are means of unit vectors, so
    ``1 - sim`` can dip a hair below zero on float noise.  Clamping keeps the
    relative-Δ ratio well defined without changing the metric.
    """
    return max(1.0 - float(sim), 0.0)


def _overlap_eligible(sim_owner: float, sim_other: float,
                      delta: float | None) -> bool:
    """Does a vector qualify for one extra membership under fixed Δ?

    Policy (DESIGN §6.4, mirrored by ``index.postings.assign``): with
    ``d = 1 - similarity`` the boundary statistic is ``r(x) = d2/d1 - 1`` and a
    vector qualifies iff ``r(x) <= delta``.  That is algebraically identical to
    the build path's ``d2 <= d1 * (1 + delta)``, so Δ keeps exactly one meaning.

    Degenerate cases:
      * ``delta is None`` — overlap disabled, never eligible.
      * ``d1 == 0`` — exactly on the owner centroid, the ratio is undefined;
        eligible only when ``d2 == 0`` too, which is what the product form
        yields at ``d1 = 0``.
      * negative distances from float noise are clamped to 0 first.
    """
    if delta is None:
        return False
    d1 = _clamp_distance(sim_owner)
    d2 = _clamp_distance(sim_other)
    if d1 <= 0.0:
        return d2 <= 0.0
    return d2 <= d1 * (1.0 + delta)


class InPlaceMaintainer(IndexMaintainer):

    def __init__(
        self,
        vectors: np.ndarray,
        cfg_index: IndexConfig,
        cfg_maintain: MaintainConfig,
        counters: Counters,
        initial_vids: np.ndarray | None = None,
        initial_segment: Segment | None = None,
    ) -> None:
        self._vectors = vectors
        self._cfg_index = cfg_index
        self._cfg_maintain = cfg_maintain
        self._counters = counters
        self._tombstones: set[VectorId] = set()

        if initial_segment is not None:
            self._segment = initial_segment.clone()
            # An externally supplied segment may already carry tombstones; adopt
            # them so liveness filtering (membership passes) and delete
            # idempotency see the same set as the segment.
            self._tombstones |= set(initial_segment.tombstones)
        elif initial_vids is not None and len(initial_vids) > 0:
            self._segment = Segment.build(
                initial_vids, vectors, cfg_index, counters, tombstones=set()
            )
        else:
            from ..core.types import HierarchyLevel
            centroid = np.zeros(cfg_index.dim, dtype=np.float32)
            empty_posting = Posting(
                id=0, centroid=centroid,
                members=np.array([], dtype=np.int64),
                owned=np.array([], dtype=np.int64),
            )
            leaf_level = HierarchyLevel(centroids=centroid[np.newaxis], children=[])
            self._segment = Segment(
                levels=[leaf_level], postings=[empty_posting],
                vectors=vectors, tombstones=set()
            )

        # A supplied segment may arrive with tombstones already applied to its
        # tombstone set but not yet to its postings.  Purge them now, so no dead
        # id is ever a member (or an owner) of a live index.
        self._drop_dead_memberships()

        # Per-posting bounding radius, aligned with self._segment.postings.
        # Kept in lockstep with owned-set changes (insert, split, reassignment,
        # tombstone rebuild) so split candidate selection stays sound.
        self._spread: list[float] = [
            _owned_spread(vectors, p.centroid, p.owned)
            for p in self._segment.postings
        ]

    def _drop_dead_memberships(self) -> None:
        """Remove already-tombstoned ids from every ``owned``/``members`` array.

        Used once at construction when the caller supplies a segment whose
        tombstone set is ahead of its arrays (e.g. restored state).  Through the
        API this is `delete`'s job, and doing it here keeps the invariant
        "``members`` holds live vectors only" true from the first operation.
        """
        dead = np.array(
            sorted(self._tombstones | set(self._segment.tombstones)),
            dtype=np.int64,
        )
        if len(dead) == 0:
            return
        for pid, p in enumerate(self._segment.postings):
            if not (np.any(np.isin(p.owned, dead)) or np.any(np.isin(p.members, dead))):
                continue
            self._segment.postings[pid] = Posting(
                p.id, p.centroid,
                p.members[~np.isin(p.members, dead)].astype(np.int64),
                p.owned[~np.isin(p.owned, dead)].astype(np.int64),
            )

    # ------------------------------------------------------------------
    def _replica_target(self, sims: np.ndarray, owner_pid: int) -> int | None:
        """Second-nearest posting for a vector, if fixed Δ requires a replica.

        The owner is chosen by the same stable rule as ``np.argmax`` over the
        leaf centroids (ties -> lowest posting id), so the replica target is
        the best *other* posting under the same tie-break.

        ``IndexConfig.rng_rule`` is deliberately not consulted: SPANN's
        relative-neighbourhood filter is a build-path option and is out of
        scope for in-place maintenance (see DESIGN §6.4).  A config that turns
        it on therefore keeps Δ-only membership after any maintenance, which
        is the cheaper (more replication) of the two rules.
        """
        delta = self._cfg_index.delta
        if delta is None or len(sims) < 2:
            return None
        masked = sims.copy()
        masked[owner_pid] = -np.inf
        other = int(np.argmax(masked))  # ties -> lowest posting id
        if not _overlap_eligible(float(sims[owner_pid]), float(masked[other]), delta):
            return None
        return other

    def insert(self, vid: VectorId, vec: np.ndarray) -> None:
        leaf_level = self._segment.levels[-1]
        sims = sim_one_to_many(vec, leaf_level.centroids, self._counters, "routing")
        pid = int(_stable_top_k(-sims, 1)[0])
        replica_pid = self._replica_target(sims, pid)

        p = posting_add_vector(self._segment.postings[pid], vid, owned=True)
        self._segment.postings[pid] = p
        if replica_pid is not None:
            rp = posting_add_vector(self._segment.postings[replica_pid], vid, owned=False)
            self._segment.postings[replica_pid] = rp
            self._counters.replicas_added += 1

        self._spread[pid] = max(
            self._spread[pid],
            float(np.linalg.norm(vec.astype(np.float32) - p.centroid.astype(np.float32))),
        )
        if len(p.owned) > self._cfg_maintain.split_threshold:
            self._split(pid)

    def delete(self, vid: VectorId) -> None:
        """Tombstone *vid* and drop every membership it holds.

        Deleting is idempotent.  Membership is removed eagerly from every
        posting (owner and replicas alike) so that no dead id survives in any
        ``members``/``owned`` array; the tombstone set is kept for query-time
        filtering, ``is_alive`` and repeated-delete semantics.  This replaces
        the old "keep tombstones until the rebuild fraction fires" heuristic in
        the in-place arm: a stale entry is exactly what the membership contract
        forbids, so the rebuild now happens on the delete itself.
        """
        if vid in self._tombstones or vid in self._segment.tombstones:
            return
        self._tombstones.add(vid)
        self._segment.mark_deleted(vid)
        for pid, p in enumerate(self._segment.postings):
            owned_here = bool(np.any(p.owned == vid))
            member_here = bool(np.any(p.members == vid))
            if not (owned_here or member_here):
                continue
            # A membership the deleted vector did not own is a replica.
            if member_here and not owned_here:
                self._counters.replicas_removed += 1
            self._rebuild_posting(pid)

    def maintain(self) -> None:
        pass  # maintenance is eager

    def searchable_units(self) -> list[Segment]:
        return [self._segment]

    def stats(self) -> dict:
        return {
            "strategy": "inplace",
            "depth": len(self._segment.levels),
            "n_postings": len(self._segment.postings),
            "splits": self._counters.splits,
            "vectors_reassigned": self._counters.vectors_reassigned,
            "replicas_added": self._counters.replicas_added,
            "replicas_removed": self._counters.replicas_removed,
            "membership_vectors_recomputed": self._counters.membership_vectors_recomputed,
            "npa_violations_detected": self._counters.npa_violations_detected,
            "hierarchy_rebuilds": self._counters.hierarchy_rebuilds,
            "tombstones": len(self._tombstones),
        }

    # ------------------------------------------------------------------
    # Split
    # ------------------------------------------------------------------

    def _split(self, pid: int) -> None:
        p = self._segment.postings[pid]
        owned_vids = p.owned
        if len(owned_vids) < 2:
            return
        orig_centroid = p.centroid.copy()
        orig_spread = self._spread[pid]
        X_owned = self._vectors[owned_vids].astype(np.float32)
        try:
            centroids2, labels2, _ = balanced_kmeans(
                X_owned, 2,
                seed=self._cfg_index.seed + self._counters.splits,
                counters=self._counters
            )
        except Exception:
            return

        new_pid = len(self._segment.postings)
        own0 = owned_vids[labels2 == 0]
        own1 = owned_vids[labels2 == 1]

        # Replica memberships are not redistributed here: the split changes the
        # centroid layout, so the whole membership set is re-derived from the
        # new centroids by `_recompute_membership` after ownership is repaired.
        p0 = Posting(pid, centroids2[0].astype(np.float32), own0, own0)
        p1 = Posting(new_pid, centroids2[1].astype(np.float32), own1, own1)
        self._segment.postings[pid] = p0
        self._segment.postings.append(p1)
        self._counters.splits += 1

        leaf_level = self._segment.levels[-1]
        new_row = centroids2[1].astype(np.float32)[np.newaxis]
        leaf_level.centroids = np.vstack(
            [leaf_level.centroids, new_row]
        ).astype(np.float32)
        leaf_level.centroids[pid] = centroids2[0].astype(np.float32)

        # Bounding radii follow the new owned sets (p1 is appended, so the
        # spread list stays aligned with self._segment.postings).
        self._spread[pid] = _owned_spread(self._vectors, p0.centroid, p0.owned)
        self._spread.append(_owned_spread(self._vectors, p1.centroid, p1.owned))

        self._reassign_neighbourhood(orig_centroid, orig_spread, pid, new_pid)
        # Replica eligibility can change for vectors whose owner did not change
        # (a new centroid may become the second-nearest), so membership is
        # rebuilt for every live vector, not just the reassigned ones.
        self._recompute_membership()
        self._promote_split(pid, new_pid)

    def _reassign_neighbourhood(self, orig_centroid: np.ndarray, orig_spread: float,
                                pid: int, new_pid: int) -> None:
        """Move every vector whose canonical owner the split just invalidated.

        The split changed exactly two centroids: ``pid`` now holds ``c0`` and
        ``new_pid`` holds ``c1``.  Every other posting centroid is unchanged,
        and the index satisfied NPA before the split.  Therefore a vector ``x``
        owned by an unchanged posting ``b`` can only lose ownership to ``c0`` or
        ``c1``; it cannot newly lose to another unchanged posting.

        Candidate postings are selected with the bounding radius of their owned
        set (``self._spread``), not by centroid proximity:

          (1) posting ``b`` may need its own vectors moved if ``c0`` or ``c1``
              can beat ``b`` for some vector in ``ball(b.centroid, r_b)``;
          (2) posting ``b`` may be the new owner of a vector of the split
              posting if ``b`` can beat ``c0``/``c1`` for some vector in
              ``ball(orig_centroid, orig_spread)``.

        Both are decided by an upper bound on the achievable inner-product
        difference over the ball, so a posting is skipped only when it is
        provably unaffected.  The previous centroid-proximity heuristic missed
        exactly the case where a vector sits far from its own posting centroid
        (stale centroid) while a new centroid lands near it.

        Ties follow the same rule as the auditor (``np.argmax`` over the leaf
        centroids): an exact tie is resolved to the lowest posting id.  That is
        why the bound admits the closed boundary (``_ball_may_switch``) and why
        the per-vector screen below uses ``>=``: a vector that only ties with a
        new centroid still changes owner when that centroid has a lower id.
        """
        leaf_centroids = self._segment.levels[-1].centroids
        K = len(leaf_centroids)
        c0 = leaf_centroids[pid].astype(np.float32)
        c1 = leaf_centroids[new_pid].astype(np.float32)

        sims_orig = sim_one_to_many(orig_centroid, leaf_centroids, self._counters, "build")
        sims_c0 = sim_one_to_many(c0, leaf_centroids, self._counters, "build")
        sims_c1 = sim_one_to_many(c1, leaf_centroids, self._counters, "build")
        norms_sq = np.einsum("ij,ij->i", leaf_centroids, leaf_centroids)
        c0_norm_sq = float(norms_sq[pid])
        c1_norm_sq = float(norms_sq[new_pid])

        # Possible destinations (new owners) and the postings whose own
        # vectors may need to move.
        destinations: set[int] = {pid, new_pid}
        sources: set[int] = {pid, new_pid}
        for b in range(K):
            if b in destinations:
                continue
            r_b = self._spread[b]
            d0_sq = max(0.0, float(norms_sq[b] + c0_norm_sq - 2.0 * sims_c0[b]))
            d1_sq = max(0.0, float(norms_sq[b] + c1_norm_sq - 2.0 * sims_c1[b]))
            steals_from_b = (
                _ball_may_switch(float(sims_c0[b]), float(norms_sq[b]), r_b, d0_sq)
                or _ball_may_switch(float(sims_c1[b]), float(norms_sq[b]), r_b, d1_sq)
            )
            beats_split_posting = (
                _ball_may_switch(float(sims_orig[b]), float(sims_orig[pid]), orig_spread, d0_sq)
                or _ball_may_switch(float(sims_orig[b]), float(sims_orig[new_pid]), orig_spread, d1_sq)
            )
            if steals_from_b:
                sources.add(b)
            if steals_from_b or beats_split_posting:
                destinations.add(b)

        # Ascending order keeps np.argmax ties resolving to the lowest posting
        # id, matching the auditor's global tie-break.
        dest_pids = np.array(sorted(destinations), dtype=np.int64)
        dest_centroids = leaf_centroids[dest_pids]

        new_owned: dict[int, list[int]] = {ap: [] for ap in dest_pids.tolist()}
        # A destination that is not a source keeps every vector it already owns;
        # it may only gain vectors that no longer belong to a source posting.
        for ap in sorted(destinations - sources):
            new_owned[ap] = self._segment.postings[ap].owned.tolist()
        old_owner: dict[int, int] = {}
        n_reassigned = 0

        def _place(vids: np.ndarray, chosen: np.ndarray) -> None:
            nonlocal n_reassigned
            for vid, new_pid_int in zip(vids.tolist(), chosen.tolist()):
                if old_owner.get(vid, -1) != new_pid_int:
                    n_reassigned += 1
                new_owned[new_pid_int].append(vid)

        # Vectors of the split posting: every destination is a legal winner
        # because the centroid they were matched against no longer exists.
        split_vids = np.unique(np.concatenate([
            self._segment.postings[pid].owned,
            self._segment.postings[new_pid].owned,
        ])).astype(np.int64)
        for ap in (pid, new_pid):
            for v in self._segment.postings[ap].owned.tolist():
                old_owner[int(v)] = ap
        if len(split_vids) > 0:
            X_split = self._vectors[split_vids].astype(np.float32)
            sims = sim_many_to_many(X_split, dest_centroids, self._counters, "build")
            _place(split_vids, dest_pids[np.argmax(sims, axis=1)])

        # Vectors of unchanged postings: only c0 and c1 moved, so three
        # similarities decide them exactly; a full destination scan is only
        # needed for the (rare) vectors the new centroids contest.
        for b in sorted(sources - {pid, new_pid}):
            vids_b = self._segment.postings[b].owned
            for v in vids_b.tolist():
                old_owner[int(v)] = b
            if len(vids_b) == 0:
                continue
            X_b = self._vectors[vids_b].astype(np.float32)
            own_sim = sim_one_to_many(leaf_centroids[b], X_b, self._counters, "build")
            s0 = sim_one_to_many(c0, X_b, self._counters, "build")
            s1 = sim_one_to_many(c1, X_b, self._counters, "build")
            contested = np.maximum(s0, s1) >= own_sim
            if contested.any():
                X_c = X_b[contested]
                sims = sim_many_to_many(X_c, dest_centroids, self._counters, "build")
                _place(vids_b[contested], dest_pids[np.argmax(sims, axis=1)])
            kept = vids_b[~contested]
            if len(kept) > 0:
                new_owned[b].extend(kept.tolist())

        self._counters.vectors_reassigned += n_reassigned

        # Only postings that actually lost or gained ownership need rewriting;
        # destinations that merely stay possible homes keep their owned set.
        touched = {pid, new_pid} | {
            ap for ap in dest_pids.tolist()
            if set(new_owned[ap]) != set(self._segment.postings[ap].owned.tolist())
        }
        for ap in sorted(touched):
            p = self._segment.postings[ap]
            new_own_arr = np.array(new_owned.get(ap, []), dtype=np.int64)
            # `members` is re-derived from the *new* centroid layout by
            # `_recompute_membership`; patching it here from the previous
            # membership would let a stale replica survive, and a vector that
            # only became eligible now would never be added.  What is kept is
            # only the part that ownership cannot invalidate: memberships that
            # were already replicas (never owned here) plus the new owned set.
            # A member that loses ownership is dropped, so `owned ⊆ members`
            # holds immediately and stale ownership is never miscounted as a
            # removed replica.
            replicas = p.members[~np.isin(p.members, p.owned)]
            self._segment.postings[ap] = Posting(
                p.id, p.centroid,
                np.union1d(new_own_arr, replicas).astype(np.int64),
                new_own_arr,
            )
            self._spread[ap] = _owned_spread(self._vectors, p.centroid, new_own_arr)

    def _recompute_membership(self) -> None:
        """Re-derive every posting's ``members`` from the current centroids.

        This is the correctness-first baseline for a structural change: a split
        moves/creates centroids, so *both* canonical ownership and replica
        eligibility change.  Ownership has already been repaired by
        `_reassign_neighbourhood`; here ``owned`` is the input and
        ``owned ∪ {replica}`` the output, where the replica is the fixed-Δ
        top-2 rule's single extra posting (or none).

        A full pass is used on purpose.  A local candidate set can only be
        sound if it bounds every posting a vector could become eligible for,
        and the previous attempt (neighbourhood of the changed centroids) had
        no such bound — it missed vectors that became newly eligible for a
        posting they were never near before.  Cost is one ``N_live × K``
        similarity block per call, charged to the ``maintenance`` distance
        bucket so it is never confused with query routing.

        Idempotent: calling it on an already-correct index rewrites nothing.
        """
        postings = self._segment.postings
        if not postings:
            return
        delta = self._cfg_index.delta
        if delta is None:
            # Overlap disabled: `members == owned` is maintained directly by
            # insert/split/delete, so there is no membership work to do (and no
            # distance pass to charge for it).
            return
        leaf_centroids = self._segment.levels[-1].centroids
        n_leaves = len(leaf_centroids)

        # `owned` is the coverage source (that is the NPA auditor's question);
        # tombstones are removed here so a dead id can never be re-added.
        owned_all = np.concatenate([p.owned for p in postings]).astype(np.int64)
        if len(owned_all) == 0:
            live = np.array([], dtype=np.int64)
        else:
            dead = self._tombstones | set(self._segment.tombstones)
            live = np.setdiff1d(
                owned_all, np.array(sorted(dead), dtype=np.int64)
            )

        # owner_of[vid] = posting that owns vid in `owned` (-1 when absent).
        owner_of = np.full(len(self._vectors), -1, dtype=np.int64)
        for p in postings:
            if len(p.owned) > 0:
                owner_of[p.owned] = p.id
        if len(live) > 0:
            live = live[owner_of[live] >= 0]

        member_lists: list[list[int]] = [[] for _ in postings]
        for start in range(0, len(live), MEMBERSHIP_CHUNK):
            chunk = live[start:start + MEMBERSHIP_CHUNK]
            sims = sim_many_to_many(
                self._vectors[chunk].astype(np.float32),
                leaf_centroids,
                self._counters,
                "maintenance",
            )
            rows = np.arange(len(chunk))
            owner = owner_of[chunk]

            other: np.ndarray | None = None
            eligible = np.zeros(len(chunk), dtype=bool)
            if n_leaves >= 2:
                masked = sims.copy()
                masked[rows, owner] = -np.inf
                # np.argmax over ascending posting ids: an exact tie resolves
                # to the lowest posting id, matching `insert` and the auditor.
                other = np.argmax(masked, axis=1)
                own_sim = sims[rows, owner].astype(np.float64)
                other_sim = masked[rows, other].astype(np.float64)
                d1 = np.maximum(1.0 - own_sim, 0.0)
                d2 = np.maximum(1.0 - other_sim, 0.0)
                # Same rule as `_overlap_eligible`, vectorised:
                #   d1 > 0  ->  d2 <= d1 * (1 + delta)      (r(x) <= delta)
                #   d1 == 0 ->  d2 <= 0                     (ratio undefined)
                eligible = np.where(d1 > 0.0, d2 <= d1 * (1.0 + delta), d2 <= 0.0)

            for i, vid in enumerate(chunk.tolist()):
                member_lists[int(owner[i])].append(vid)
            if other is not None:
                for i in np.flatnonzero(eligible).tolist():
                    member_lists[int(other[i])].append(int(chunk[i]))

        n_added = 0
        n_removed = 0
        for pid, p in enumerate(postings):
            own_arr = np.unique(p.owned)
            # union1d: sorted and de-duplicated, so a posting can never hold the
            # same vector twice and member order stays deterministic.
            new_members = np.union1d(
                own_arr, np.array(member_lists[pid], dtype=np.int64)
            )
            prev_members = np.unique(p.members)
            prev_rep = np.setdiff1d(prev_members, own_arr)
            new_rep = np.setdiff1d(new_members, own_arr)
            n_added += int(np.setdiff1d(new_rep, prev_rep).size)
            n_removed += int(np.setdiff1d(prev_rep, new_rep).size)
            if (len(new_members) != len(p.members)
                    or not np.array_equal(new_members, p.members)):
                postings[pid] = Posting(p.id, p.centroid, new_members, p.owned)

        self._counters.replicas_added += n_added
        self._counters.replicas_removed += n_removed
        self._counters.membership_vectors_recomputed += int(len(live))

    def rebuild_membership(self) -> None:
        """Public entry point: re-derive fixed-Δ membership for the whole index.

        Needed when a segment is supplied through ``initial_segment=`` built by
        a different path (or from persisted state), where the maintainer cannot
        assume ``members`` already follows this policy.
        """
        self._recompute_membership()

    def _promote_split(self, pid: int, new_pid: int) -> None:
        if len(self._segment.levels) == 1:
            from ..core.types import HierarchyLevel
            lc = self._segment.levels[-1].centroids
            root_c = lc.mean(axis=0, keepdims=True).astype(np.float32)
            root_level = HierarchyLevel(
                centroids=root_c,
                children=[np.arange(len(lc), dtype=np.int64)]
            )
            self._segment.levels.insert(0, root_level)
            return

        routing = self._segment.levels[:-1]
        bottom = routing[-1]
        for node_idx in range(len(bottom.children)):
            if np.any(bottom.children[node_idx] == pid):
                bottom.children[node_idx] = np.append(
                    bottom.children[node_idx], new_pid
                ).astype(np.int64)
                if len(bottom.children[node_idx]) > self._cfg_index.branching_factor * 2:
                    self._rebuild_hierarchy()
                break

    def _rebuild_hierarchy(self) -> None:
        leaf_centroids = self._segment.levels[-1].centroids
        new_levels = build_hierarchy(
            leaf_centroids,
            branching_factor=self._cfg_index.branching_factor,
            seed=self._cfg_index.seed,
            depth=self._cfg_index.depth,
            counters=self._counters,
        )
        self._counters.hierarchy_rebuilds += 1
        self._segment.levels = new_levels

    def _rebuild_posting(self, pid: int) -> None:
        p = self._segment.postings[pid]
        tomb = list(self._tombstones)
        live_owned = p.owned[~np.isin(p.owned, tomb)]
        live_members = p.members[~np.isin(p.members, tomb)]
        new_p = Posting(p.id, p.centroid, live_members, live_owned)
        self._segment.postings[pid] = new_p
        self._spread[pid] = _owned_spread(self._vectors, new_p.centroid, live_owned)
