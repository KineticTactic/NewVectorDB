"""core/distance.py — counted distance kernels.

Vectors are L2-normalised at load time (see harness/datasets.py).
Similarity  = inner product.
Distance    = 1 - similarity.

DESIGN §6.1 rule: there must be NO path in the codebase that computes a
distance without going through one of the two public functions below.
Every call increments counters.distance_computations[bucket] by the number
of pairs scored.

Buckets used across the codebase:
  "routing"  — beam descent through the routing tree
  "scan"     — scoring vectors inside fetched postings
  "build"    — index construction (overlap assignment, balancing,
               reassignment, NPA checks, ground-truth computation)

sklearn.KMeans internal distances are compiled code and cannot be counted;
they are documented as offline build cost and do NOT appear in any bucket.
This is safe because every §2 finding depends only on routing vs scan
separability at query time.
"""

from __future__ import annotations

import numpy as np
from .metrics import Counters


def sim_one_to_many(
    q: np.ndarray,
    X: np.ndarray,
    counters: Counters,
    bucket: str,
) -> np.ndarray:
    """Inner-product similarity of query vector *q* against every row of *X*.

    Parameters
    ----------
    q : shape (D,) float32
    X : shape (N, D) float32
    counters : Counters — distance_computations[bucket] incremented by N
    bucket : str — counter bucket name

    Returns
    -------
    sims : shape (N,) float32   sims[i] = dot(q, X[i])
    """
    q = np.asarray(q, dtype=np.float32)
    X = np.asarray(X, dtype=np.float32)
    n = X.shape[0]
    if n == 0:
        return np.empty(0, dtype=np.float32)
    counters.add_distance(bucket, n)
    return X @ q   # (N, D) @ (D,) → (N,)


def sim_many_to_many(
    A: np.ndarray,
    B: np.ndarray,
    counters: Counters,
    bucket: str,
) -> np.ndarray:
    """Inner-product similarity of every row of *A* against every row of *B*.

    Parameters
    ----------
    A : shape (M, D) float32
    B : shape (N, D) float32
    counters : Counters — distance_computations[bucket] incremented by M*N
    bucket : str — counter bucket name

    Returns
    -------
    sims : shape (M, N) float32   sims[i, j] = dot(A[i], B[j])
    """
    A = np.asarray(A, dtype=np.float32)
    B = np.asarray(B, dtype=np.float32)
    m, n = A.shape[0], B.shape[0]
    if m == 0 or n == 0:
        return np.empty((m, n), dtype=np.float32)
    counters.add_distance(bucket, m * n)
    return A @ B.T   # (M, D) @ (D, N) → (M, N)


def dist_one_to_many(
    q: np.ndarray,
    X: np.ndarray,
    counters: Counters,
    bucket: str,
) -> np.ndarray:
    """Convenience wrapper: distance = 1 - similarity."""
    return 1.0 - sim_one_to_many(q, X, counters, bucket)


def dist_many_to_many(
    A: np.ndarray,
    B: np.ndarray,
    counters: Counters,
    bucket: str,
) -> np.ndarray:
    """Convenience wrapper: distance = 1 - similarity."""
    return 1.0 - sim_many_to_many(A, B, counters, bucket)
