"""test_distance.py — Stage 1 checkpoint: distance kernels + counter correctness."""

import numpy as np
import pytest

from vecstudy.core.metrics import Counters
from vecstudy.core.distance import (
    sim_one_to_many, sim_many_to_many,
    dist_one_to_many, dist_many_to_many,
)


def _unit(v):
    v = np.array(v, dtype=np.float32)
    return v / np.linalg.norm(v)


# ---------------------------------------------------------------------------
# Basic correctness
# ---------------------------------------------------------------------------

def test_sim_one_to_many_identity():
    """A unit vector has similarity 1.0 with itself."""
    c = Counters()
    v = _unit([1, 0, 0, 0])
    X = np.array([v, _unit([0, 1, 0, 0])], dtype=np.float32)
    sims = sim_one_to_many(v, X, c, "routing")
    assert abs(sims[0] - 1.0) < 1e-6
    assert abs(sims[1]) < 1e-6


def test_sim_one_to_many_counts(toy_vectors):
    c = Counters()
    X = toy_vectors
    _ = sim_one_to_many(X[0], X, c, "routing")
    assert c.distance_computations["routing"] == len(X)


def test_sim_many_to_many_counts(toy_vectors):
    c = Counters()
    X = toy_vectors
    _ = sim_many_to_many(X[:3], X[:5], c, "scan")
    assert c.distance_computations["scan"] == 15  # 3 * 5


def test_sim_many_to_many_shape(toy_vectors):
    c = Counters()
    X = toy_vectors
    out = sim_many_to_many(X[:4], X[:6], c, "build")
    assert out.shape == (4, 6)


def test_dist_is_one_minus_sim(toy_vectors):
    c1, c2 = Counters(), Counters()
    q = toy_vectors[0]
    X = toy_vectors
    sims = sim_one_to_many(q, X, c1, "scan")
    dists = dist_one_to_many(q, X, c2, "scan")
    np.testing.assert_allclose(dists, 1.0 - sims, atol=1e-6)


# ---------------------------------------------------------------------------
# Counter separability
# ---------------------------------------------------------------------------

def test_bucket_separation(toy_vectors):
    c = Counters()
    X = toy_vectors
    sim_one_to_many(X[0], X, c, "routing")
    sim_one_to_many(X[0], X[:3], c, "scan")
    assert c.distance_computations["routing"] == len(X)
    assert c.distance_computations["scan"] == 3


def test_counter_reset(toy_vectors):
    c = Counters()
    X = toy_vectors
    sim_one_to_many(X[0], X, c, "routing")
    c.reset_search()
    assert c.distance_computations == {}
    assert c.postings_read == 0


def test_counter_add(toy_vectors):
    c1, c2 = Counters(), Counters()
    X = toy_vectors
    sim_one_to_many(X[0], X, c1, "routing")
    sim_one_to_many(X[0], X[:3], c2, "scan")
    total = c1 + c2
    assert total.distance_computations["routing"] == len(X)
    assert total.distance_computations["scan"] == 3


# ---------------------------------------------------------------------------
# Hand-computed values on a 10-vector toy set
# ---------------------------------------------------------------------------

def test_hand_computed_toy(toy_vectors):
    """Counter must equal n for sim_one_to_many on 10-vector set."""
    c = Counters()
    X = toy_vectors  # fixture: 10 vectors
    sim_one_to_many(X[0], X, c, "routing")
    assert c.distance_computations["routing"] == 10, \
        "Exactly 10 distance computations expected for 10-vector set"


def test_empty_X():
    c = Counters()
    q = np.array([1.0, 0.0], dtype=np.float32)
    X = np.empty((0, 2), dtype=np.float32)
    sims = sim_one_to_many(q, X, c, "routing")
    assert len(sims) == 0
    assert c.distance_computations.get("routing", 0) == 0
