# R1 Out-Of-Place Maintenance — Task 1: Compaction, Tombstone GC, and Lifecycle Hardening

**Scope:** `vecstudy/maintain/outofplace.py`, deletion semantics, tombstone garbage collection on compaction, deletion-driven compaction threshold, and a new focused test suite `tests/test_outofplace_compaction.py`.

**Not in scope, and not implemented:** in-place maintainer modifications, dynamic-Δ calibration, SPANN RNG filtering, graph navigation over centroids, query-engine changes.

**Status:** complete. `198 passed` (was 188: +10 out-of-place lifecycle and compaction tests).

---

## 1. Problem and Root Cause

Under DESIGN §6.6 and research track R1, out-of-place maintenance buffers writes in a MemTable, flushes to immutable sealed segments at `memtable_limit`, and compacts sealed segments at `compaction_threshold`. However, the initial implementation had several critical gaps:

1. **Tombstone Leakage:** When `_compact()` re-clustered live vectors, `self._tombstones` was never pruned. As deletes accumulated over time, `stats()["tombstones"]` and internal tombstone sets grew unboundedly, even though the compacted segment had physically discarded those deleted vectors.
2. **Missing Deletion-Driven Compaction:** Under delete-heavy workloads where few or no inserts occur, `len(self._sealed)` never crosses `compaction_threshold`. Without deletion-driven compaction, tombstoned vectors remain inside sealed segments, degrading scan distances, I/O bytes read, and query recall.
3. **Lack of Compaction Atomicity:** Compaction was non-atomic. An exception during `Segment.build()` (e.g., clustering or memory error) would leave the maintainer in an inconsistent intermediate state.
4. **Missing Key Statistics:** `OutOfPlaceMaintainer.stats()` lacked `depth`, `n_postings`, `unit_count`, and `hierarchy_rebuilds`.

---

## 2. The Policy and Implementation Decisions

### 2.1 Tombstone Garbage Collection
When `_compact()` merges sealed segments into a single re-clustered segment:
* The live vector set is derived strictly from `seg.is_alive(vid) and vid not in self._tombstones`.
* `new_seg = Segment.build(...)` builds a fresh, clean index containing exclusively live vectors.
* Purged vectors from the old sealed segments are permanently eliminated.
* `self._tombstones` is pruned: only tombstones corresponding to vectors still pending in `self._memtable` (if any) are retained (`self._tombstones &= memtable_set`).

### 2.2 Deletion-Driven Compaction Trigger
In addition to segment-count thresholding (`len(self._sealed) >= compaction_threshold`), `maintain()` now checks the global sealed tombstone ratio:
$$\text{tombstone\_frac} = \frac{\sum_{s \in \text{sealed}} |s.\text{tombstones}|}{\sum_{s \in \text{sealed}, p \in s.\text{postings}} |p.\text{owned}|}$$
If $\text{tombstone\_frac} \ge \text{cfg\_maintain.tombstone\_rebuild\_frac}$ (default 0.20), compaction triggers immediately to purge tombstones, reclaim posting storage, and restore search scan locality (reflecting LSM-VEC and DESIGN §6.6).

### 2.3 Compaction Atomicity and Rollback
Compaction takes snapshots of `_sealed`, `_tombstones`, and `_counters` before running `Segment.build()`. If an exception occurs, all states roll back bit-for-bit to their pre-compaction values.

### 2.4 Idempotent Deletion and Revival
* Repeated `delete(vid)` calls are no-ops.
* Calling `insert(vid, vec)` for a previously tombstoned ID revives it and discards the tombstone.

---

## 3. Files Changed

| File | Purpose |
|---|---|
| `vecstudy/maintain/outofplace.py` | Atomic `_compact`, tombstone garbage collection, deletion-driven compaction, idempotent delete, reinsert revival, comprehensive `stats()` |
| `tests/test_outofplace_compaction.py` **(new)** | 10 focused tests covering MemTable buffering, flushing, compaction, tombstone GC, deletion-driven compaction, atomicity rollback, and empty index edge cases |

---

## 4. Verification and Reproduction

Run focused tests:
```powershell
python -m pytest tests/test_outofplace_compaction.py -v
```
Result: `10 passed in 1.86s`.

Run full test suite:
```powershell
python -m pytest -q
```
Result: `198 passed`.
