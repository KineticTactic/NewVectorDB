"""core/metrics.py — counter collection.

Two distinct counter lifetimes:
  - Search counters  (distance_computations, postings_read, bytes_read,
                      vectors_scanned) reset per query via reset_search().
  - Maintenance counters (splits, vectors_reassigned, compaction_events,
                         vectors_reclustered, npa_violations_detected)
    accumulate across the entire run.

Never mix the two lifetimes in the same snapshot.
"""

from __future__ import annotations

from dataclasses import dataclass, field
import copy


@dataclass
class Counters:
    # ---- search counters (reset per query) --------------------------------
    # Buckets: at minimum "routing" and "scan"; "build" for index construction.
    distance_computations: dict[str, int] = field(default_factory=dict)
    postings_read: int = 0
    bytes_read: int = 0
    vectors_scanned: int = 0

    # ---- maintenance counters (accumulate for whole run) ------------------
    splits: int = 0
    vectors_reassigned: int = 0
    compaction_events: int = 0
    vectors_reclustered: int = 0
    npa_violations_detected: int = 0
    flushes: int = 0
    forced_checkpoint_flushes: int = 0
    hierarchy_rebuilds: int = 0

    # -----------------------------------------------------------------------

    def reset_search(self) -> None:
        """Zero only the per-query search counters."""
        self.distance_computations = {}
        self.postings_read = 0
        self.bytes_read = 0
        self.vectors_scanned = 0

    def add_distance(self, bucket: str, n: int) -> None:
        """Increment the distance counter for *bucket* by *n* pairs."""
        self.distance_computations[bucket] = (
            self.distance_computations.get(bucket, 0) + n
        )

    def snapshot(self) -> "Counters":
        """Return a deep copy (safe to store as a result row)."""
        return copy.deepcopy(self)

    def search_dict(self) -> dict:
        """Flat dict of search counters only — for result rows."""
        out: dict = {}
        for bucket, v in self.distance_computations.items():
            out[f"dist_{bucket}"] = v
        out["postings_read"] = self.postings_read
        out["bytes_read"] = self.bytes_read
        out["vectors_scanned"] = self.vectors_scanned
        return out

    def maintain_dict(self) -> dict:
        """Flat dict of maintenance counters only — for result rows."""
        return {
            "splits": self.splits,
            "vectors_reassigned": self.vectors_reassigned,
            "compaction_events": self.compaction_events,
            "vectors_reclustered": self.vectors_reclustered,
            "npa_violations_detected": self.npa_violations_detected,
            "flushes": self.flushes,
            "forced_checkpoint_flushes": self.forced_checkpoint_flushes,
            "hierarchy_rebuilds": self.hierarchy_rebuilds,
        }

    def to_dict(self) -> dict:
        """Combined flat dict (search + maintenance)."""
        return {**self.search_dict(), **self.maintain_dict()}

    def __add__(self, other: "Counters") -> "Counters":
        """Element-wise sum of two Counters (for aggregating across queries)."""
        c = self.snapshot()
        # search
        for bucket, v in other.distance_computations.items():
            c.distance_computations[bucket] = (
                c.distance_computations.get(bucket, 0) + v
            )
        c.postings_read += other.postings_read
        c.bytes_read += other.bytes_read
        c.vectors_scanned += other.vectors_scanned
        # maintenance
        c.splits += other.splits
        c.vectors_reassigned += other.vectors_reassigned
        c.compaction_events += other.compaction_events
        c.vectors_reclustered += other.vectors_reclustered
        c.npa_violations_detected += other.npa_violations_detected
        c.flushes += other.flushes
        c.forced_checkpoint_flushes += other.forced_checkpoint_flushes
        c.hierarchy_rebuilds += other.hierarchy_rebuilds
        return c
