"""Shared fixtures for vecstudy tests."""

import os
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"
os.environ["OPENBLAS_NUM_THREADS"] = "1"

import numpy as np
import pytest

from vecstudy.harness.datasets import make_synthetic, l2_normalize


@pytest.fixture(scope="session")
def toy_vectors():
    """10 L2-normalised 8-D vectors, fixed seed."""
    rng = np.random.RandomState(42)
    X = rng.randn(10, 8).astype(np.float32)
    norms = np.linalg.norm(X, axis=1, keepdims=True)
    return (X / norms).astype(np.float32)


@pytest.fixture(scope="session")
def synthetic_200():
    """200 L2-normalised 32-D vectors, fixed seed."""
    return make_synthetic(n=200, dim=32, n_topics=5, seed=7)


@pytest.fixture(scope="session")
def synthetic_1000():
    """1000 L2-normalised 32-D vectors, fixed seed."""
    return make_synthetic(n=1000, dim=32, n_topics=10, seed=13)
