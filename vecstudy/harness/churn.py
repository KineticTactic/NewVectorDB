"""harness/churn.py — Deterministic churn workload generator, dynamic ground truth, and NPA auditor."""

from __future__ import annotations

import hashlib
from dataclasses import dataclass, field
from pathlib import Path
from typing import Literal

import numpy as np

from ..core.metrics import Counters
from ..core.types import VectorId
from ..maintain.base import IndexMaintainer
from ..maintain.inplace import InPlaceMaintainer
from ..maintain.outofplace import OutOfPlaceMaintainer
from ..index.segment import Segment
from .datasets import DATA_ROOT, dataset_checksum, _sha


@dataclass(frozen=True)
class ChurnTrace:
    """Deterministic trace of insert and delete operations."""
    seed: int
    workload: str
    initial_n: int
    reserve_n: int
    operations: list[tuple[str, int]]  # list of ("insert", vid) or ("delete", vid)
    trace_hash: str

    def to_dict(self) -> dict:
        return {
            "seed": self.seed,
            "workload": self.workload,
            "initial_n": self.initial_n,
            "reserve_n": self.reserve_n,
            "n_operations": len(self.operations),
            "trace_hash": self.trace_hash,
        }


def generate_churn_trace(
    initial_n: int,
    reserve_n: int,
    n_operations: int,
    insert_prob: float = 0.5,
    seed: int = 42,
    workload: str = "mixed",
) -> ChurnTrace:
    """Generate a deterministic sequence of insert and delete operations.

    Rules:
    - Inserts draw from the uninserted reserve pool [initial_n, initial_n + reserve_n).
    - Deletes only select currently live vector IDs.
    - No live ID is deleted more than once without reinsertion.
    - Operation ordering is completely deterministic based on seed.
    """
    rng = np.random.RandomState(seed)
    current_live: set[int] = set(range(initial_n))
    next_insert_vid = initial_n
    max_insert_vid = initial_n + reserve_n

    operations: list[tuple[str, int]] = []

    for _ in range(n_operations):
        can_insert = next_insert_vid < max_insert_vid
        can_delete = len(current_live) > 0

        if not can_insert and not can_delete:
            break

        do_insert = False
        if not can_delete:
            do_insert = True
        elif not can_insert:
            do_insert = False
        else:
            if workload == "mixed":
                do_insert = bool(rng.rand() < insert_prob)
            elif workload == "insert_only":
                do_insert = True
            elif workload == "delete_only":
                do_insert = False
            else:
                do_insert = bool(rng.rand() < insert_prob)

        if do_insert:
            vid = next_insert_vid
            next_insert_vid += 1
            current_live.add(vid)
            operations.append(("insert", vid))
        else:
            # Deterministic selection from sorted live IDs for reproducibility
            sorted_live = sorted(current_live)
            victim_idx = int(rng.randint(0, len(sorted_live)))
            victim = sorted_live[victim_idx]
            current_live.remove(victim)
            operations.append(("delete", victim))

    # Stable trace representation and hash
    trace_repr = f"seed={seed};init={initial_n};res={reserve_n};wl={workload};" + ";".join(
        f"{op}:{vid}" for op, vid in operations
    )
    trace_hash = hashlib.sha256(trace_repr.encode("utf-8")).hexdigest()[:16]

    return ChurnTrace(
        seed=seed,
        workload=workload,
        initial_n=initial_n,
        reserve_n=reserve_n,
        operations=operations,
        trace_hash=trace_hash,
    )


def compute_dynamic_ground_truth(
    queries: np.ndarray,
    vectors: np.ndarray,
    live_vids: np.ndarray,
    k: int = 10,
    cache_dir: Path | None = None,
) -> np.ndarray:
    """Compute exact ground truth for queries against ONLY current live vectors.

    Important: Maps row offsets in vectors[live_vids] back to stable global VectorIds.
    Cache key includes query checksum, live IDs checksum, and vector content checksum.
    """
    if len(live_vids) == 0:
        return np.empty((len(queries), 0), dtype=np.int32)

    live_vids_arr = np.asarray(live_vids, dtype=np.int64)
    live_base = vectors[live_vids_arr].astype(np.float32)

    cache_dir = cache_dir or (DATA_ROOT / "gt_cache")
    cache_dir.mkdir(parents=True, exist_ok=True)

    vids_hash = hashlib.sha256(live_vids_arr.tobytes()).hexdigest()[:16]
    data_hash = hashlib.sha256(live_base.tobytes()).hexdigest()[:16]
    query_hash = _sha(queries)
    cache_key = f"dynamic_gt_{query_hash}_{vids_hash}_{data_hash}_k{k}"
    cache_file = cache_dir / f"{cache_key}.npy"

    if cache_file.exists():
        return np.load(str(cache_file))

    # Brute-force search returns local offsets in live_base
    nq = len(queries)
    eff_k = min(k, len(live_vids_arr))
    gt_global = np.empty((nq, eff_k), dtype=np.int64)

    for i, q in enumerate(queries):
        sims = live_base @ q.astype(np.float32)
        top_local = np.argpartition(-sims, kth=min(eff_k, len(sims) - 1))[:eff_k]
        top_local = top_local[np.argsort(-sims[top_local])]
        # Map local offset -> global VectorId
        gt_global[i] = live_vids_arr[top_local[:eff_k]]

    np.save(str(cache_file), gt_global)
    return gt_global


@dataclass
class AuditResult:
    """Result of structural and NPA verification.

    ``npa_violations`` holds every violation found; ``examples`` is a bounded
    prefix of that list for logging and assertion messages.  Auditing only
    reports: it never repairs ownership or mutates the index.
    """
    passed: bool
    npa_violations: list[dict]
    ownership_errors: list[str]
    coverage_errors: list[str]
    live_vectors_checked: int = 0
    postings_checked: int = 0
    examples: list[dict] = field(default_factory=list)
    examples_truncated: bool = False

    @property
    def violation_count(self) -> int:
        return len(self.npa_violations)

    def summary(self) -> str:
        return (
            f"AuditResult(passed={self.passed}, "
            f"npa_violations={len(self.npa_violations)}, "
            f"ownership_errors={len(self.ownership_errors)}, "
            f"coverage_errors={len(self.coverage_errors)}, "
            f"live_vectors_checked={self.live_vectors_checked}, "
            f"postings_checked={self.postings_checked})"
        )

    def to_dict(self) -> dict:
        return {
            "passed": self.passed,
            "violation_count": self.violation_count,
            "live_vectors_checked": self.live_vectors_checked,
            "postings_checked": self.postings_checked,
            "ownership_errors": list(self.ownership_errors),
            "coverage_errors": list(self.coverage_errors),
            "examples": list(self.examples),
            "examples_truncated": self.examples_truncated,
        }


def _audit_segment_npa(
    seg: Segment,
    tombstones: set[int],
    tolerance: float,
    segment_index: int | None = None,
) -> tuple[list[dict], list[str], set[int], int]:
    """Check canonical ownership against every live leaf centroid of *seg*.

    NPA is evaluated with the production metric: similarity is the inner
    product of the stored vector with the posting centroid, distance is
    ``1 - similarity`` and the nearest posting is the ``np.argmax`` of the
    similarities, i.e. exact ties resolve to the lowest posting id (the same
    stable tie-break used by ``index.query.descend``).

    Only canonical ownership is checked.  A vector that merely appears in a
    posting's ``members`` (a Δ-overlap replica) is not its owner and is not
    audited here.

    Out of scope: Δ-overlap membership semantics.  ``members`` is required only
    to be a superset of ``owned``; the replica rule never enters this check.
    Use :func:`audit_membership` for that half — it independently re-derives the
    expected ``members`` set for every live vector under the configured Δ.
    A passing NPA audit therefore proves coverage, unique canonical ownership
    and NPA, and says nothing about whether replicas are correct or complete.

    Returns ``(violations, ownership_errors, live_owned_vids, postings_checked)``.
    """
    lc = seg.levels[-1].centroids
    violations: list[dict] = []
    ownership_errors: list[str] = []
    owner_count: dict[int, int] = {}

    for p in seg.postings:
        p_members_set = set(int(v) for v in p.members)
        for vid in p.owned:
            vi = int(vid)
            if vi not in p_members_set:
                prefix = f"Sealed {segment_index} " if segment_index is not None else ""
                ownership_errors.append(
                    f"{prefix}Posting {p.id}: owned vid {vi} not in members"
                )
            if vi in tombstones:
                continue

            owner_count[vi] = owner_count.get(vi, 0) + 1

            # Posting ids are positional (postings[pid].id == pid), so p.id is
            # both the centroid row and the ownership identity.
            sims = lc @ seg.vectors[vi].astype(np.float32)
            nearest_pid = int(np.argmax(sims))
            own_sim = float(sims[p.id])
            best_sim = float(sims[nearest_pid])

            if nearest_pid != p.id and (best_sim - own_sim) > tolerance:
                record = {
                    "vid": vi,
                    "owned_pid": p.id,
                    "nearest_pid": nearest_pid,
                    "own_sim": own_sim,
                    "best_sim": best_sim,
                    "own_distance": 1.0 - own_sim,
                    "nearest_distance": 1.0 - best_sim,
                    "margin": best_sim - own_sim,
                }
                if segment_index is not None:
                    record["segment"] = segment_index
                violations.append(record)

    for vi, cnt in owner_count.items():
        if cnt > 1:
            ownership_errors.append(f"vid {vi} has {cnt} owners")

    return violations, ownership_errors, set(owner_count), len(seg.postings)


def audit_maintainer(
    maint: IndexMaintainer,
    expected_live_vids: set[int] | None = None,
    tolerance: float = 1e-5,
    max_examples: int = 10,
) -> AuditResult:
    """Validate structural invariants and Nearest Partition Assumption (NPA).

    Invariants checked:
    1. Coverage: all live vectors are present in searchable units.
    2. Ownership: owned is a subset of members; every live vector has exactly
       one canonical owner.
    3. NPA: for every live vector, its owning posting centroid is nearest
       among all live leaf centroids of its segment.
    4. Exclusion: tombstoned vectors are skipped by the NPA check and are not
       counted as live owners.

    Not checked: Δ-overlap membership semantics.  ``owned ⊆ members`` is
    verified, but whether the ``members`` sets satisfy the relative-Δ replica
    rule is a separate property checked by :func:`audit_membership` (whose
    expectation is derived independently of the maintainer).  A ``passed``
    result here must not be read as an overlap-correctness claim on its own.

    The audit is read-only: it reports violations, never repairs them.
    ``max_examples`` bounds ``AuditResult.examples`` only; the full violation
    list is always returned in ``npa_violations``.
    """
    npa_violations: list[dict] = []
    ownership_errors: list[str] = []
    coverage_errors: list[str] = []
    live_vectors_checked = 0
    postings_checked = 0

    if isinstance(maint, InPlaceMaintainer):
        seg = maint._segment
        v, o, all_owned, n_postings = _audit_segment_npa(
            seg, seg.tombstones, tolerance
        )
        npa_violations.extend(v)
        ownership_errors.extend(o)
        live_vectors_checked = len(all_owned)
        postings_checked = n_postings

        if expected_live_vids is not None:
            missing = expected_live_vids - all_owned
            extra = all_owned - expected_live_vids
            if missing:
                coverage_errors.append(f"{len(missing)} live vectors missing from index (e.g. {sorted(missing)[:5]})")
            if extra:
                coverage_errors.append(f"{len(extra)} dead/unknown vectors in index (e.g. {sorted(extra)[:5]})")

    elif isinstance(maint, OutOfPlaceMaintainer):
        # Sealed segments: NPA is checked within each sealed segment.
        global_owner_count: dict[int, list[str]] = {}
        for s_idx, seg in enumerate(maint._sealed):
            v, o, live_owned, n_postings = _audit_segment_npa(
                seg, seg.tombstones | maint._tombstones, tolerance, s_idx
            )
            npa_violations.extend(v)
            ownership_errors.extend(o)
            for vi in live_owned:
                global_owner_count.setdefault(vi, []).append(f"sealed[{s_idx}]")
            postings_checked += n_postings

        # MemTable entries
        memtable_postings = 0
        memtable_counts: dict[int, int] = {}
        for vid in maint._memtable:
            vi = int(vid)
            if vi not in maint._tombstones:
                memtable_counts[vi] = memtable_counts.get(vi, 0) + 1
                global_owner_count.setdefault(vi, []).append("memtable")
                memtable_postings = 1

        for vi, count in memtable_counts.items():
            if count > 1:
                ownership_errors.append(f"vid {vi} duplicated {count} times in memtable")

        for vi, locations in global_owner_count.items():
            if len(locations) > 1:
                ownership_errors.append(f"vid {vi} has multiple owners across units: {locations}")

        all_live_found = set(global_owner_count)
        postings_checked += memtable_postings
        live_vectors_checked = len(all_live_found)

        if expected_live_vids is not None:
            missing = expected_live_vids - all_live_found
            extra = all_live_found - expected_live_vids
            if missing:
                coverage_errors.append(f"{len(missing)} live vectors missing from searchable units")
            if extra:
                coverage_errors.append(f"{len(extra)} unexpected vectors live across units")

    passed = len(npa_violations) == 0 and len(ownership_errors) == 0 and len(coverage_errors) == 0
    return AuditResult(
        passed=passed,
        npa_violations=npa_violations,
        ownership_errors=ownership_errors,
        coverage_errors=coverage_errors,
        live_vectors_checked=live_vectors_checked,
        postings_checked=postings_checked,
        examples=npa_violations[:max_examples],
        examples_truncated=len(npa_violations) > max_examples,
    )


# ---------------------------------------------------------------------------
# Fixed-Δ replica-membership audit
# ---------------------------------------------------------------------------
#
# The NPA audit above answers "does every live vector have the right canonical
# owner?".  It deliberately says nothing about `members`, because a replica is
# a membership without ownership.  This second audit answers the other half:
# "does every live posting's `members` array equal the set the fixed-Δ top-2
# policy requires?"
#
# It is written from the R2 boundary statistic, not from the maintainer's
# helper functions: the comparison is the ratio form r(x) = d2/d1 - 1 <= Δ
# (the product form d2 <= d1(1+Δ) is the same rule, reached by a different
# arithmetic path), and the "best other posting" is chosen with a stable
# descending sort.  Nothing here calls `_replica_target`, `_overlap_eligible`
# or `_recompute_membership`, so a defect in the production selection logic
# cannot make this audit agree with it.

_UNSET = object()


@dataclass
class MembershipAuditResult:
    """Problems found by the fixed-Δ membership audit (empty lists == pass)."""
    missing: list[dict]
    extra: list[dict]
    duplicates: list[dict]
    missing_from_owner: list[dict]
    dead_members: list[dict]
    ownership_conflicts: list[dict]
    live_vectors_checked: int
    postings_checked: int

    @property
    def passed(self) -> bool:
        return not (self.missing or self.extra or self.duplicates
                    or self.missing_from_owner or self.dead_members
                    or self.ownership_conflicts)

    @property
    def violation_count(self) -> int:
        return (len(self.missing) + len(self.extra) + len(self.duplicates)
                + len(self.missing_from_owner) + len(self.dead_members)
                + len(self.ownership_conflicts))

    def summary(self) -> str:
        return (
            f"MembershipAuditResult(passed={self.passed}, "
            f"missing={len(self.missing)}, extra={len(self.extra)}, "
            f"duplicates={len(self.duplicates)}, "
            f"missing_from_owner={len(self.missing_from_owner)}, "
            f"dead_members={len(self.dead_members)}, "
            f"ownership_conflicts={len(self.ownership_conflicts)}, "
            f"live_vectors_checked={self.live_vectors_checked}, "
            f"postings_checked={self.postings_checked})"
        )

    def to_dict(self) -> dict:
        return {
            "passed": self.passed,
            "violation_count": self.violation_count,
            "n_missing": len(self.missing),
            "n_extra": len(self.extra),
            "n_duplicates": len(self.duplicates),
            "n_missing_from_owner": len(self.missing_from_owner),
            "n_dead_members": len(self.dead_members),
            "n_ownership_conflicts": len(self.ownership_conflicts),
            "live_vectors_checked": self.live_vectors_checked,
            "postings_checked": self.postings_checked,
            "missing": list(self.missing),
            "extra": list(self.extra),
            "duplicates": list(self.duplicates),
            "missing_from_owner": list(self.missing_from_owner),
            "dead_members": list(self.dead_members),
            "ownership_conflicts": list(self.ownership_conflicts),
        }


def _expected_membership(
    sims: np.ndarray, owner_pid: int, delta: float | None
) -> tuple[int, ...]:
    """Posting ids a vector must appear in, derived from its own coordinates.

    Rule (DESIGN §6.4, R2 fixed-Δ experiments): with ``d = 1 - sim``,
    ``d1 = d(owner)`` and ``d2 = d(best posting other than the owner)``, the
    boundary statistic is ``r = d2/d1 - 1`` and the vector receives one extra
    membership iff ``r <= delta``.  The canonical owner always keeps its
    membership, and no vector can hold more than two (top-2 cap).

    Ties are resolved by a stable descending sort, i.e. the lowest posting id
    wins — the same tie-break as ``np.argmax`` over the leaf centroids.

    Degenerate cases: ``d1 == 0`` (the vector sits exactly on its owner
    centroid) makes the ratio undefined, so eligibility requires ``d2 == 0``
    as well; negative distances from float round-off are clamped to zero.
    """
    owner_pid = int(owner_pid)
    if delta is None or len(sims) < 2:
        return (owner_pid,)

    order = np.argsort(-sims, kind="stable").tolist()
    other = next((int(pid) for pid in order if int(pid) != owner_pid), None)
    if other is None:
        return (owner_pid,)

    d1 = max(1.0 - float(sims[owner_pid]), 0.0)
    d2 = max(1.0 - float(sims[other]), 0.0)
    if d1 <= 0.0:
        qualifies = d2 <= 0.0
    else:
        qualifies = (d2 / d1) - 1.0 <= delta
    return (owner_pid, other) if qualifies else (owner_pid,)


def _audit_segment_membership(
    seg: Segment,
    tombstones: set[int],
    delta: float | None,
    segment_prefix: str = "",
) -> tuple[list[dict], list[dict], list[dict], list[dict], list[dict], list[dict], int, int]:
    centroids = seg.levels[-1].centroids
    owners: dict[int, list[int]] = {}
    members_of: dict[int, list[int]] = {}
    duplicates: list[dict] = []
    dead_members: list[dict] = []

    for p in seg.postings:
        counts: dict[int, int] = {}
        for vid in p.members.tolist():
            v = int(vid)
            counts[v] = counts.get(v, 0) + 1
            members_of.setdefault(v, []).append(int(p.id))
            if v in tombstones:
                rec = {"vid": v, "pid": int(p.id)}
                if segment_prefix:
                    rec["segment"] = segment_prefix
                dead_members.append(rec)
        for v, c in sorted(counts.items()):
            if c > 1:
                rec = {"vid": v, "pid": int(p.id), "count": c}
                if segment_prefix:
                    rec["segment"] = segment_prefix
                duplicates.append(rec)
        for vid in p.owned.tolist():
            v = int(vid)
            if v in tombstones:
                continue
            owners.setdefault(v, []).append(int(p.id))

    missing: list[dict] = []
    extra: list[dict] = []
    missing_from_owner: list[dict] = []
    ownership_conflicts: list[dict] = []

    for vid in sorted(owners):
        owner_pids = owners[vid]
        if len(owner_pids) != 1:
            rec = {"vid": vid, "owner_pids": owner_pids}
            if segment_prefix:
                rec["segment"] = segment_prefix
            ownership_conflicts.append(rec)
            continue
        owner_pid = owner_pids[0]

        # float64 accumulation: the ratio below is d2/d1 - 1 with d = 1 - sim
        sims = centroids.astype(np.float64) @ seg.vectors[vid].astype(np.float64)
        expected = _expected_membership(sims, owner_pid, delta)
        actual = sorted(set(members_of.get(vid, [])))

        missing_pids = [pid for pid in expected if pid not in actual]
        extra_pids = [pid for pid in actual if pid not in expected]
        if missing_pids:
            rec = {
                "vid": vid, "owner_pid": owner_pid,
                "expected_pids": list(expected), "actual_pids": actual,
                "missing_pids": missing_pids,
            }
            if segment_prefix:
                rec["segment"] = segment_prefix
            missing.append(rec)
        if extra_pids:
            rec = {
                "vid": vid, "owner_pid": owner_pid,
                "expected_pids": list(expected), "actual_pids": actual,
                "extra_pids": extra_pids,
            }
            if segment_prefix:
                rec["segment"] = segment_prefix
            extra.append(rec)
        if owner_pid not in members_of.get(vid, []):
            rec = {"vid": vid, "owner_pid": owner_pid}
            if segment_prefix:
                rec["segment"] = segment_prefix
            missing_from_owner.append(rec)

    return (
        missing, extra, duplicates, missing_from_owner, dead_members,
        ownership_conflicts, len(owners), len(seg.postings),
    )


def audit_membership(
    maint: IndexMaintainer,
    delta_override: object = _UNSET,
    allow_outofplace: bool = False,
) -> MembershipAuditResult:
    """Independently verify fixed-Δ replica membership for in-place and out-of-place indexes.

    Expected membership is derived only from the current live leaf centroids,
    the current canonical owner (read from ``owned``) and the configured Δ —
    never from ``members`` itself and never from a production membership
    helper.  Detected violations:

      * ``missing``     — a required replica membership is absent.
      * ``extra``       — a membership the policy does not justify (including
        a third membership, i.e. a broken top-2 cap).
      * ``duplicates``  — the same vector twice in one posting.
      * ``missing_from_owner`` — a live vector absent from its owner's members.
      * ``dead_members`` — a tombstoned/deleted vector still in ``members``.
      * ``ownership_conflicts`` — a vector with zero or several owners.

    Read-only: it reports, it never repairs.  ``delta_override`` exists for
    tests that need to audit against a deliberately different threshold; by
    default the maintainer's configured Δ is used, which is the contract the
    index is supposed to satisfy.
    """
    if isinstance(maint, OutOfPlaceMaintainer):
        if not allow_outofplace:
            raise TypeError(
                "fixed-Δ membership is maintained by InPlaceMaintainer only; "
                f"got {type(maint).__name__}"
            )
    elif not isinstance(maint, InPlaceMaintainer):
        raise TypeError(
            "fixed-Δ membership is maintained by InPlaceMaintainer only; "
            f"got {type(maint).__name__}"
        )
    delta = (
        maint._cfg_index.delta
        if delta_override is _UNSET
        else delta_override
    )

    if isinstance(maint, InPlaceMaintainer):
        seg = maint._segment
        tombstones = set(seg.tombstones) | set(getattr(maint, "_tombstones", set()))
        res = _audit_segment_membership(seg, tombstones, delta)
        return MembershipAuditResult(
            missing=res[0],
            extra=res[1],
            duplicates=res[2],
            missing_from_owner=res[3],
            dead_members=res[4],
            ownership_conflicts=res[5],
            live_vectors_checked=res[6],
            postings_checked=res[7],
        )

    # OutOfPlaceMaintainer
    all_missing: list[dict] = []
    all_extra: list[dict] = []
    all_duplicates: list[dict] = []
    all_missing_from_owner: list[dict] = []
    all_dead_members: list[dict] = []
    all_ownership_conflicts: list[dict] = []
    total_live = 0
    total_postings = 0

    global_tombstones = set(maint._tombstones)
    for s_idx, seg in enumerate(maint._sealed):
        tombstones = set(seg.tombstones) | global_tombstones
        res = _audit_segment_membership(seg, tombstones, delta, segment_prefix=f"sealed[{s_idx}]")
        all_missing.extend(res[0])
        all_extra.extend(res[1])
        all_duplicates.extend(res[2])
        all_missing_from_owner.extend(res[3])
        all_dead_members.extend(res[4])
        all_ownership_conflicts.extend(res[5])
        total_live += res[6]
        total_postings += res[7]

    # MemTable check: flat buffer has no replicas
    if maint._memtable:
        mem_counts: dict[int, int] = {}
        for vid in maint._memtable:
            vi = int(vid)
            if vi in global_tombstones:
                all_dead_members.append({"vid": vi, "unit": "memtable"})
            else:
                mem_counts[vi] = mem_counts.get(vi, 0) + 1
        for vi, c in sorted(mem_counts.items()):
            if c > 1:
                all_duplicates.append({"vid": vi, "unit": "memtable", "count": c})
        total_live += len(mem_counts)
        total_postings += 1

    return MembershipAuditResult(
        missing=all_missing,
        extra=all_extra,
        duplicates=all_duplicates,
        missing_from_owner=all_missing_from_owner,
        dead_members=all_dead_members,
        ownership_conflicts=all_ownership_conflicts,
        live_vectors_checked=total_live,
        postings_checked=total_postings,
    )


def audit_outofplace_membership(
    maint: OutOfPlaceMaintainer,
    delta_override: object = _UNSET,
) -> MembershipAuditResult:
    """Audit fixed-Δ replica membership for OutOfPlaceMaintainer across sealed segments."""
    return audit_membership(maint, delta_override=delta_override, allow_outofplace=True)

