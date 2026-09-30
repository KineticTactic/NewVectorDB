"""maintain/base.py — IndexMaintainer abstract base class."""

from __future__ import annotations

from abc import ABC, abstractmethod
import numpy as np

from ..core.types import VectorId
from ..index.segment import Segment


class IndexMaintainer(ABC):
    """Abstract base for index maintenance strategies.

    Both inplace and outofplace implement this interface unchanged.
    The query path (index/query.py) calls searchable_units() and never
    branches on strategy type.
    """

    @abstractmethod
    def insert(self, vid: VectorId, vec: np.ndarray) -> None:
        """Insert a single vector into the index."""
        ...

    @abstractmethod
    def delete(self, vid: VectorId) -> None:
        """Mark a vector as deleted (tombstone)."""
        ...

    @abstractmethod
    def maintain(self) -> None:
        """Perform pending maintenance (called after each insert batch)."""
        ...

    @abstractmethod
    def searchable_units(self) -> list[Segment]:
        """Return all segments visible to the query path."""
        ...

    @abstractmethod
    def stats(self) -> dict:
        """Return a dict of current maintenance statistics."""
        ...
