"""index/postings.py — leaf postings, Δ-overlap assignment, tombstones.

DESIGN §6.4 rules:
  - Δ is RELATIVE: replica iff d2 <= d1 * (1 + delta).
  - Top-2 rule: each vector in at most 2 postings.
  - SPANN RNG rule: skip replica when centroid-centroid dist < d2.
  - Simulated I/O only; no real file I/O.
  - Tombstones filtered AFTER scoring, never before.
"""

from __future__ import annotations

import numpy as np

from ..core.types import Posting, PostingId, VectorId
from ..core.metrics import Counters
from ..core.distance import sim_many_to_many

POSTING_HEADER_BYTES = 64


def assign(
    X: np.ndarray,
    centroids: np.ndarray,
    *,
    delta: float | None,
    rng_rule: bool,
    counters: Counters,
) -> tuple[list[Posting], float]:
    """Assign N vectors to K postings with optional Δ-overlap.

    Returns (postings, bloat) where bloat = total_members/N - 1.
    """
    X = np.asarray(X, dtype=np.float32)
    centroids = np.asarray(centroids, dtype=np.float32)
    N, D = X.shape
    K = centroids.shape[0]

    # Compute top-2 nearest centroids in chunks to bound peak memory.
    # A full (N, K) float32 matrix is N*K*4 bytes — 1M*1250*4 = 4.7 GB.
    # Processing CHUNK_SIZE rows at a time keeps peak memory under ~128 MB.
    CHUNK_SIZE = 32_768  # rows per chunk; 32K * 1250 * 4 B ≈ 160 MB

    nearest = np.empty(N, dtype=np.int64)
    second  = np.empty(N, dtype=np.int64)
    sim1    = np.empty(N, dtype=np.float32)
    sim2    = np.empty(N, dtype=np.float32)

    for start in range(0, N, CHUNK_SIZE):
        end = min(start + CHUNK_SIZE, N)
        X_chunk = X[start:end]
        # sim_many_to_many counts distances correctly per chunk
        sims_chunk = sim_many_to_many(X_chunk, centroids, counters, "build")  # (chunk, K)

        if K >= 2:
            top2 = np.argpartition(-sims_chunk, kth=1, axis=1)[:, :2]
            rows_local = np.arange(end - start)
            s0 = sims_chunk[rows_local, top2[:, 0]]
            s1 = sims_chunk[rows_local, top2[:, 1]]
            swap = s0 < s1
            tmp = top2[swap, 0].copy()
            top2[swap, 0] = top2[swap, 1]
            top2[swap, 1] = tmp
            nearest[start:end] = top2[:, 0].astype(np.int64)
            second[start:end]  = top2[:, 1].astype(np.int64)
            sim1[start:end] = sims_chunk[rows_local, top2[:, 0]]
            sim2[start:end] = sims_chunk[rows_local, top2[:, 1]]
        else:
            nearest[start:end] = 0
            second[start:end]  = 0
            sim1[start:end] = sims_chunk[:, 0]
            sim2[start:end] = sims_chunk[:, 0]

    d1 = (1.0 - sim1).astype(np.float64)
    d2 = (1.0 - sim2).astype(np.float64)

    owned_lists:   list[list[int]] = [[] for _ in range(K)]
    members_lists: list[list[int]] = [[] for _ in range(K)]

    for vid in range(N):
        c1 = int(nearest[vid])
        c2 = int(second[vid])
        owned_lists[c1].append(vid)
        members_lists[c1].append(vid)

        if delta is not None and K >= 2 and c1 != c2:
            if d2[vid] <= d1[vid] * (1.0 + delta):
                add_replica = True
                if rng_rule:
                    cc_dist = float(1.0 - np.dot(centroids[c1], centroids[c2]))
                    if cc_dist < d2[vid]:
                        add_replica = False
                if add_replica:
                    members_lists[c2].append(vid)

    postings: list[Posting] = []
    total_members = 0
    for pid in range(K):
        oa = np.array(owned_lists[pid],   dtype=np.int64)
        ma = np.array(members_lists[pid], dtype=np.int64)
        total_members += len(ma)
        postings.append(Posting(id=pid, centroid=centroids[pid].copy(),
                                members=ma, owned=oa))

    bloat = total_members / N - 1.0 if N > 0 else 0.0
    return postings, bloat


# ---------------------------------------------------------------------------
# Simulated I/O
# ---------------------------------------------------------------------------

def fetch_posting(posting: Posting, D: int, counters: Counters) -> Posting:
    """Simulate reading a posting from disk (no real I/O)."""
    counters.postings_read += 1
    counters.bytes_read += POSTING_HEADER_BYTES + len(posting.members) * D * 4
    return posting


# ---------------------------------------------------------------------------
# Posting mutation helpers (used by inplace maintainer)
# ---------------------------------------------------------------------------

def posting_add_vector(p: Posting, vid: VectorId, owned: bool) -> Posting:
    """Return new Posting with vid added to members (and owned if owned=True)."""
    new_members = np.append(p.members, vid).astype(np.int64)
    new_owned   = np.append(p.owned, vid).astype(np.int64) if owned else p.owned.copy()
    return Posting(id=p.id, centroid=p.centroid, members=new_members, owned=new_owned)


def posting_remove_vids(p: Posting, vids: np.ndarray) -> Posting:
    """Return new Posting with vids removed from both owned and members."""
    new_owned   = p.owned[~np.isin(p.owned, vids)].astype(np.int64)
    new_members = p.members[~np.isin(p.members, vids)].astype(np.int64)
    return Posting(id=p.id, centroid=p.centroid, members=new_members, owned=new_owned)


def posting_set_centroid(p: Posting, centroid: np.ndarray) -> Posting:
    return Posting(id=p.id, centroid=centroid.astype(np.float32),
                   members=p.members.copy(), owned=p.owned.copy())
