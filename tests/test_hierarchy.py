"""test_hierarchy.py — Stage 2 checkpoint: invariant 9 + beam monotonicity."""

import numpy as np
import pytest

from vecstudy.core.metrics import Counters
from vecstudy.core.distance import sim_one_to_many
from vecstudy.index.hierarchy import build_hierarchy, descend


def _brute_nearest(q, centroids):
    sims = centroids @ q
    return int(np.argmax(sims))


def test_depth1_equivalence(synthetic_200):
    """Invariant 9: depth-1 with beam=n_leaves returns same as brute-force nearest."""
    X = synthetic_200
    n_leaves = 10
    # Build flat leaf centroids via simple k-means labels
    from sklearn.cluster import KMeans
    km = KMeans(n_clusters=n_leaves, random_state=0, n_init=1).fit(X)
    leaf_centroids = km.cluster_centers_.astype(np.float32)

    levels = build_hierarchy(leaf_centroids, branching_factor=n_leaves, seed=0, depth=1)
    assert len(levels) == 1  # depth 1: only the leaf level

    c = Counters()
    q = X[5]
    # beam_width = n_leaves => exhaustive scan of leaf level
    result = descend(q, levels, beam_widths=[n_leaves], n_postings=1, counters=c)
    brute = _brute_nearest(q, leaf_centroids)
    assert result == [brute], f"depth-1 descent {result} != brute {brute}"


def test_beam_monotonicity(synthetic_200):
    """Invariant 4: recall@1 is non-decreasing in beam width."""
    X = synthetic_200
    n_leaves = 20
    from sklearn.cluster import KMeans
    km = KMeans(n_clusters=n_leaves, random_state=0, n_init=1).fit(X)
    leaf_centroids = km.cluster_centers_.astype(np.float32)
    levels = build_hierarchy(leaf_centroids, branching_factor=4, seed=0)

    q = X[0]
    brute = _brute_nearest(q, leaf_centroids)

    prev_found = False
    for bw in [1, 2, 4, 8, n_leaves]:
        c = Counters()
        result = descend(q, levels, beam_widths=[bw], n_postings=1, counters=c)
        found = brute in result
        if prev_found:
            assert found, f"beam {bw}: recall dropped after it was found"
        prev_found = prev_found or found


def test_levels_top_first(synthetic_200):
    """Levels are top-first; last level is the leaf level."""
    X = synthetic_200
    from sklearn.cluster import KMeans
    n_leaves = 15
    km = KMeans(n_clusters=n_leaves, random_state=0, n_init=1).fit(X)
    leaf_centroids = km.cluster_centers_.astype(np.float32)
    levels = build_hierarchy(leaf_centroids, branching_factor=4, seed=0)
    # Last level must have exactly n_leaves centroids
    assert levels[-1].centroids.shape[0] == n_leaves


def test_routing_counts_only_routing_bucket(synthetic_200):
    """descend must only increment 'routing' bucket."""
    X = synthetic_200
    from sklearn.cluster import KMeans
    km = KMeans(n_clusters=8, random_state=0, n_init=1).fit(X)
    leaf_centroids = km.cluster_centers_.astype(np.float32)
    levels = build_hierarchy(leaf_centroids, branching_factor=4, seed=0)
    c = Counters()
    descend(X[0], levels, beam_widths=[4], n_postings=1, counters=c)
    assert "scan" not in c.distance_computations
    assert "build" not in c.distance_computations
    assert c.distance_computations.get("routing", 0) > 0


def test_explicit_depth(synthetic_200):
    """explicit depth=2 produces exactly 2 levels."""
    X = synthetic_200
    from sklearn.cluster import KMeans
    n_leaves = 16
    km = KMeans(n_clusters=n_leaves, random_state=0, n_init=1).fit(X)
    leaf_centroids = km.cluster_centers_.astype(np.float32)
    levels = build_hierarchy(leaf_centroids, branching_factor=4, seed=0, depth=2)
    assert len(levels) == 2
