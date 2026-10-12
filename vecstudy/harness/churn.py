"""harness/churn.py — Deterministic churn workload generator, dynamic ground truth, and NPA auditor."""

from __future__ import annotations

import hashlib
from dataclasses import dataclass
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
    """Result of structural and NPA verification."""
    passed: bool
    npa_violations: list[dict]
    ownership_errors: list[str]
    coverage_errors: list[str]

    def summary(self) -> str:
        return (
            f"AuditResult(passed={self.passed}, "
            f"npa_violations={len(self.npa_violations)}, "
            f"ownership_errors={len(self.ownership_errors)}, "
            f"coverage_errors={len(self.coverage_errors)})"
        )


def audit_maintainer(
    maint: IndexMaintainer,
    expected_live_vids: set[int] | None = None,
    tolerance: float = 1e-5,
) -> AuditResult:
    """Validate structural invariants and Nearest Partition Assumption (NPA).

    Invariants checked:
    1. Coverage: all live vectors are present in searchable units.
    2. Ownership: owned is a subset of members; every live vector has valid canonical ownership.
    3. NPA: for every live vector, its owning posting centroid is nearest among all leaf centroids.
    4. Exclusion: deleted vectors are tombstoned and not in live ownership.
    """
    npa_violations: list[dict] = []
    ownership_errors: list[str] = []
    coverage_errors: list[str] = []

    if isinstance(maint, InPlaceMaintainer):
        seg = maint._segment
        lc = seg.levels[-1].centroids
        tombstones = seg.tombstones

        # Track ownership counts
        owner_count: dict[int, int] = {}
        all_owned: set[int] = set()

        for p in seg.postings:
            # Check owned is subset of members
            p_members_set = set(int(v) for v in p.members)
            for vid in p.owned:
                vi = int(vid)
                if vi not in p_members_set:
                    ownership_errors.append(f"Posting {p.id}: owned vid {vi} not in members")

                if vi in tombstones:
                    continue

                owner_count[vi] = owner_count.get(vi, 0) + 1
                all_owned.add(vi)

                # Check NPA against ALL leaf centroids
                sims = lc @ seg.vectors[vi].astype(np.float32)
                best_pid = int(np.argmax(sims))
                own_sim = float(sims[p.id])
                best_sim = float(sims[best_pid])

                if best_pid != p.id and (best_sim - own_sim) > tolerance:
                    npa_violations.append({
                        "vid": vi,
                        "owned_pid": p.id,
                        "nearest_pid": best_pid,
                        "own_sim": own_sim,
                        "best_sim": best_sim,
                        "margin": best_sim - own_sim,
                    })

        # Check unique ownership
        for vi, cnt in owner_count.items():
            if cnt > 1:
                ownership_errors.append(f"vid {vi} has {cnt} owners")

        # Compare with expected live IDs
        if expected_live_vids is not None:
            missing = expected_live_vids - all_owned
            extra = all_owned - expected_live_vids
            if missing:
                coverage_errors.append(f"{len(missing)} live vectors missing from index (e.g. {sorted(missing)[:5]})")
            if extra:
                coverage_errors.append(f"{len(extra)} dead/unknown vectors in index (e.g. {sorted(extra)[:5]})")

    elif isinstance(maint, OutOfPlaceMaintainer):
        # Sealed segments
        all_live_found: set[int] = set()
        for s_idx, seg in enumerate(maint._sealed):
            lc = seg.levels[-1].centroids
            for p in seg.postings:
                p_members_set = set(int(v) for v in p.members)
                for vid in p.owned:
                    vi = int(vid)
                    if vi not in p_members_set:
                        ownership_errors.append(f"Sealed {s_idx} posting {p.id}: owned vid {vi} not in members")
                    if seg.is_alive(vi) and vi not in maint._tombstones:
                        all_live_found.add(vi)
                        # Check NPA within this sealed segment
                        sims = lc @ seg.vectors[vi].astype(np.float32)
                        best_pid = int(np.argmax(sims))
                        own_sim = float(sims[p.id])
                        best_sim = float(sims[best_pid])
                        if best_pid != p.id and (best_sim - own_sim) > tolerance:
                            npa_violations.append({
                                "vid": vi,
                                "segment": s_idx,
                                "owned_pid": p.id,
                                "nearest_pid": best_pid,
                                "own_sim": own_sim,
                                "best_sim": best_sim,
                                "margin": best_sim - own_sim,
                            })

        # MemTable entries
        for vid in maint._memtable:
            vi = int(vid)
            if vi not in maint._tombstones:
                all_live_found.add(vi)

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
    )

