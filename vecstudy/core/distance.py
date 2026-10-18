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


# ---------------------------------------------------------------------------
# float64 kernels for the fixed-Δ relative-distance statistic
# ---------------------------------------------------------------------------
#
# When a vector sits very close to a centroid, similarity is close to 1 and the
# distance ``d = 1 - sim`` loses the leading digits of ``sim`` to cancellation:
# at ``sim ~ 1 - 1e-4`` a float32 input carries a relative error of ~1e-3 in
# ``d``.  The fixed-Δ rule compares the *ratio* ``d2/d1 - 1`` against Δ, so that
# cancellation can flip the decision for a vector genuinely near the boundary.
#
# These variants promote the operands to float64 *before* the dot product, so
# the accumulated similarity keeps float64 precision and ``1 - sim`` is accurate
# to ~1e-12 relative.  They are used only where the Δ membership decision is
# taken; the query kernels above stay float32 so routing/scan behaviour and cost
# are unchanged.

def sim_one_to_many_f64(
    q: np.ndarray,
    X: np.ndarray,
    counters: Counters,
    bucket: str,
) -> np.ndarray:
    """float64 variant of :func:`sim_one_to_many` (operands promoted pre-dot).

    Inputs are the same stored float32 data; only the accumulation is float64.
    Distinct posting ids still resolve ties by the caller's stable argmax.
    """
    q = np.asarray(q, dtype=np.float64)
    X = np.asarray(X, dtype=np.float64)
    n = X.shape[0]
    if n == 0:
        return np.empty(0, dtype=np.float64)
    counters.add_distance(bucket, n)
    return X @ q


def sim_many_to_many_f64(
    A: np.ndarray,
    B: np.ndarray,
    counters: Counters,
    bucket: str,
) -> np.ndarray:
    """float64 variant of :func:`sim_many_to_many` (operands promoted pre-dot)."""
    A = np.asarray(A, dtype=np.float64)
    B = np.asarray(B, dtype=np.float64)
    m, n = A.shape[0], B.shape[0]
    if m == 0 or n == 0:
        return np.empty((m, n), dtype=np.float64)
    counters.add_distance(bucket, m * n)
    return A @ B.T


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
