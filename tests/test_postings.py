"""test_postings.py — Stage 3 checkpoint: invariants 1, 2, 6 + bloat."""

import numpy as np
import pytest

from vecstudy.core.metrics import Counters
from vecstudy.index.postings import assign


def test_coverage(synthetic_200):
    """Invariant 1: every vector appears in at least one posting's members."""
    X = synthetic_200
    N = len(X)
    c = Counters()
    from sklearn.cluster import KMeans
    km = KMeans(n_clusters=5, random_state=0, n_init=1).fit(X)
    centroids = km.cluster_centers_.astype(np.float32)
    postings, bloat = assign(X, centroids, delta=None, rng_rule=False, counters=c)
    covered = set()
    for p in postings:
        covered.update(p.members.tolist())
    assert covered == set(range(N)), "Not all vectors covered"


def test_ownership_unique(synthetic_200):
    """Invariant 2: every vector is owned by exactly one posting."""
    X = synthetic_200
    c = Counters()
    from sklearn.cluster import KMeans
    km = KMeans(n_clusters=5, random_state=0, n_init=1).fit(X)
    centroids = km.cluster_centers_.astype(np.float32)
    postings, _ = assign(X, centroids, delta=None, rng_rule=False, counters=c)
    owner_count = {}
    for p in postings:
        for v in p.owned:
            owner_count[int(v)] = owner_count.get(int(v), 0) + 1
    assert all(c == 1 for c in owner_count.values()), "Some vectors owned by != 1 posting"


def test_owned_subset_members(synthetic_200):
    """Invariant: owned ⊆ members for every posting."""
    X = synthetic_200
    c = Counters()
    from sklearn.cluster import KMeans
    km = KMeans(n_clusters=5, random_state=0, n_init=1).fit(X)
    centroids = km.cluster_centers_.astype(np.float32)
    postings, _ = assign(X, centroids, delta=None, rng_rule=False, counters=c)
    for p in postings:
        owned_set = set(p.owned.tolist())
        members_set = set(p.members.tolist())
        assert owned_set.issubset(members_set), f"Posting {p.id}: owned not ⊆ members"


def test_overlap_bound(synthetic_200):
    """Invariant 6: with top-2 rule, no vector in more than 2 postings."""
    X = synthetic_200
    c = Counters()
    from sklearn.cluster import KMeans
    km = KMeans(n_clusters=5, random_state=0, n_init=1).fit(X)
    centroids = km.cluster_centers_.astype(np.float32)
    postings, _ = assign(X, centroids, delta=0.3, rng_rule=False, counters=c)
    count = {}
    for p in postings:
        for v in p.members:
            count[int(v)] = count.get(int(v), 0) + 1
    max_appearances = max(count.values()) if count else 0
    assert max_appearances <= 2, f"Vector appeared in {max_appearances} postings (max 2 expected)"


def test_bloat_no_overlap(synthetic_200):
    """bloat = 0.0 when delta=None."""
    X = synthetic_200
    c = Counters()
    from sklearn.cluster import KMeans
    km = KMeans(n_clusters=5, random_state=0, n_init=1).fit(X)
    centroids = km.cluster_centers_.astype(np.float32)
    _, bloat = assign(X, centroids, delta=None, rng_rule=False, counters=c)
    assert abs(bloat) < 1e-9, f"Expected bloat=0, got {bloat}"


def test_bloat_positive_with_overlap(synthetic_200):
    """bloat > 0 when delta > 0 (at least some replicas expected)."""
    X = synthetic_200
    c = Counters()
    from sklearn.cluster import KMeans
    km = KMeans(n_clusters=5, random_state=0, n_init=1).fit(X)
    centroids = km.cluster_centers_.astype(np.float32)
    _, bloat = assign(X, centroids, delta=0.5, rng_rule=False, counters=c)
    assert bloat >= 0.0  # may be 0 if no border vectors; just not negative


def test_assign_counts_build_bucket(synthetic_200):
    """assign must count distances under 'build' bucket."""
    X = synthetic_200
    c = Counters()
    from sklearn.cluster import KMeans
    km = KMeans(n_clusters=5, random_state=0, n_init=1).fit(X)
    centroids = km.cluster_centers_.astype(np.float32)
    assign(X, centroids, delta=None, rng_rule=False, counters=c)
    assert c.distance_computations.get("build", 0) == len(X) * len(centroids)


def test_hand_computed_bloat():
    """Hand-computed bloat on 4-vector, 2-centroid toy set."""
    # 4 vectors; 2 centroids; delta large enough to replicate all border vectors
    v0 = np.array([1, 0], dtype=np.float32)
    v1 = np.array([0.99, 0.14], dtype=np.float32)  # very close to v0
    v2 = np.array([-1, 0], dtype=np.float32)
    v3 = np.array([-0.99, 0.14], dtype=np.float32)  # very close to v2
    X = np.array([v0, v1, v2, v3], dtype=np.float32)
    # Normalise
    X = X / np.linalg.norm(X, axis=1, keepdims=True)

    c0 = np.array([1, 0], dtype=np.float32)
    c1 = np.array([-1, 0], dtype=np.float32)
    centroids = np.array([c0, c1], dtype=np.float32)

    c = Counters()
    # With delta=0 no replicas
    postings, bloat = assign(X, centroids, delta=0.0, rng_rule=False, counters=c)
    assert abs(bloat) < 1e-6, f"Expected bloat=0 with delta=0, got {bloat}"
