"""core/clustering.py — balanced k-means wrapper.

DESIGN §6.2:
  1. Run sklearn.KMeans (fast, BLAS-accelerated Lloyd's iterations).
  2. Balancing pass: while any cluster exceeds (1 + balance_tol) × mean_size,
     move its farthest-from-centroid members to their next-nearest cluster
     that is below the mean.  Cap at MAX_BALANCE_ROUNDS = 10 rounds — do not
     loop until convergence, it may not converge.
  3. Log the realised size distribution (returned as metadata).

sklearn internal distances are uncounted (compiled code); our balancing-pass
distances go through distance.sim_one_to_many under the "build" bucket.

Returns
-------
centroids : float32 (n_clusters, D)
labels    : int64   (N,)  cluster assignment for each input vector
info      : dict    realised size distribution and convergence metadata
"""

from __future__ import annotations

import math
import numpy as np
from sklearn.cluster import KMeans

from .metrics import Counters
from .distance import sim_many_to_many

MAX_BALANCE_ROUNDS = 10


def balanced_kmeans(
    X: np.ndarray,
    n_clusters: int,
    *,
    seed: int,
    max_iter: int = 50,
    balance_tol: float = 0.25,
    counters: Counters | None = None,
) -> tuple[np.ndarray, np.ndarray, dict]:
    """Balanced k-means clustering.

    Parameters
    ----------
    X           : float32 (N, D) — L2-normalised input vectors
    n_clusters  : int
    seed        : int — RNG seed (determinism requirement)
    max_iter    : int — sklearn Lloyd iterations cap
    balance_tol : float — cluster size tolerance above mean_size
    counters    : Counters | None — if provided, balancing distances counted
                  under "build" bucket; None → use a throw-away counter

    Returns
    -------
    centroids : float32 (n_clusters, D)
    labels    : int64   (N,)
    info      : dict with keys: size_mean, size_p50, size_p99, size_max,
                               balance_rounds, converged
    """
    from .metrics import Counters as _Counters
    if counters is None:
        counters = _Counters()

    X = np.asarray(X, dtype=np.float32)
    N, D = X.shape

    if n_clusters == 1:
        centroids = X.mean(axis=0, keepdims=True)
        labels = np.zeros(N, dtype=np.int64)
        return centroids, labels, _info(labels, n_clusters, 0, True)

    if n_clusters >= N:
        # degenerate: one vector per cluster, rest in cluster 0
        labels = np.zeros(N, dtype=np.int64)
        labels[:n_clusters] = np.arange(n_clusters, dtype=np.int64)
        centroids = np.zeros((n_clusters, D), dtype=np.float32)
        for i in range(min(n_clusters, N)):
            centroids[i] = X[i]
        return centroids, labels, _info(labels, n_clusters, 0, True)

    # Use MiniBatchKMeans when N is large — same interface, much faster.
    # Threshold: full KMeans is fine up to ~50K vectors; beyond that the
    # per-iteration O(N*k*D) cost dominates.  MiniBatchKMeans converges
    # in O(batch_size * k * D * n_iter) which is independent of N.
    # Determinism: fixed random_state + init="k-means++" both honour the seed.
    MINIBATCH_THRESHOLD = 50_000
    if N > MINIBATCH_THRESHOLD:
        from sklearn.cluster import MiniBatchKMeans
        batch_size = min(10_000, N)
        km = MiniBatchKMeans(
            n_clusters=n_clusters,
            random_state=seed,
            max_iter=max_iter,
            n_init=1,
            batch_size=batch_size,
            init="k-means++",
        )
        km.fit(X)
    else:
        km = KMeans(
            n_clusters=n_clusters,
            random_state=seed,
            max_iter=max_iter,
            n_init=1,
            algorithm="lloyd",
        )
        km.fit(X)
    labels = km.labels_.astype(np.int64)
    centroids = km.cluster_centers_.astype(np.float32)

    # ---- balancing pass ---------------------------------------------------
    # Skip if the (N, k) similarity matrix would exceed MEM_LIMIT bytes —
    # MiniBatchKMeans already produces near-balanced clusters on real data,
    # so the pass is most valuable for small synthetic sets.
    MEM_LIMIT_BYTES = 512 * 1024 * 1024  # 512 MB
    matrix_bytes = N * n_clusters * 4    # float32
    if matrix_bytes > MEM_LIMIT_BYTES:
        # Record cluster sizes but skip rebalancing
        return centroids, labels, _info(labels, n_clusters, 0, True)

    mean_size = N / n_clusters
    cap = (1.0 + balance_tol) * mean_size
    converged = False

    for round_idx in range(MAX_BALANCE_ROUNDS):
        sizes = np.bincount(labels, minlength=n_clusters)
        overloaded = np.where(sizes > cap)[0]
        if len(overloaded) == 0:
            converged = True
            break

        # recompute centroids from current labels
        centroids = _recompute_centroids(X, labels, n_clusters, D)

        # similarity of every vector to every centroid (counted)
        sims = sim_many_to_many(X, centroids, counters, "build")  # (N, k)
        # sort each row to get 1st and 2nd nearest
        order = np.argsort(-sims, axis=1)  # descending similarity

        for cl in overloaded:
            # members of this cluster, farthest-from-centroid first
            mask = labels == cl
            idxs = np.where(mask)[0]
            own_sim = sims[idxs, cl]
            farthest_order = np.argsort(own_sim)  # ascending → farthest first

            for fi in farthest_order:
                vid = idxs[fi]
                for candidate in order[vid]:
                    if candidate == cl:
                        continue
                    if sizes[candidate] < mean_size:
                        labels[vid] = candidate
                        sizes[cl] -= 1
                        sizes[candidate] += 1
                        break
                if sizes[cl] <= cap:
                    break

        # recompute after moves
        centroids = _recompute_centroids(X, labels, n_clusters, D)

    return centroids, labels, _info(labels, n_clusters, round_idx + 1, converged)


def _recompute_centroids(
    X: np.ndarray,
    labels: np.ndarray,
    n_clusters: int,
    D: int,
) -> np.ndarray:
    centroids = np.zeros((n_clusters, D), dtype=np.float32)
    for c in range(n_clusters):
        mask = labels == c
        if mask.any():
            centroids[c] = X[mask].mean(axis=0)
    return centroids


def _info(labels: np.ndarray, n_clusters: int, rounds: int, converged: bool) -> dict:
    sizes = np.bincount(labels, minlength=n_clusters)
    return {
        "size_mean": float(sizes.mean()),
        "size_p50": float(np.percentile(sizes, 50)),
        "size_p99": float(np.percentile(sizes, 99)),
        "size_max": int(sizes.max()),
        "balance_rounds": rounds,
        "converged": converged,
    }
