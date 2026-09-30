"""core/types.py — fundamental type aliases and dataclasses.

All types here are pure data; no logic lives here.
"""

from __future__ import annotations

from dataclasses import dataclass, field
import numpy as np

# ---------------------------------------------------------------------------
# Simple scalar aliases
# ---------------------------------------------------------------------------
VectorId = int      # index into the global vector array (0-based)
PostingId = int     # unique id of a leaf posting
NodeId = int        # node index within a hierarchy level


# ---------------------------------------------------------------------------
# Posting
# ---------------------------------------------------------------------------
@dataclass(frozen=True)
class Posting:
    """A leaf posting in the index.

    Invariants (enforced by tests/test_invariants.py):
      - owned ⊆ members
      - every live VectorId is owned by exactly one posting
      - for every vid in owned, the centroid of this posting is the nearest
        posting centroid (NPA invariant)
    """
    id: PostingId
    centroid: np.ndarray        # float32, shape (D,)
    members: np.ndarray         # int64 VectorIds — used for search (includes replicas)
    owned: np.ndarray           # int64 VectorIds — used for maintenance decisions


# ---------------------------------------------------------------------------
# HierarchyLevel
# ---------------------------------------------------------------------------
@dataclass
class HierarchyLevel:
    """One level in the routing tree.

    Levels are stored top-first: index 0 is the root level, the last entry
    is the leaf-centroid level.

    children[i] is a 1-D int64 array of indices into the *next* level down.
    At the leaf level children is an empty list.
    """
    centroids: np.ndarray           # float32, (n_nodes, D)
    children: list[np.ndarray]      # children[i] → child indices in level below
