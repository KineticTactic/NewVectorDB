# R1 Out-Of-Place Maintenance — Task 2: Cross-Segment Canonical Ownership and Fixed-Δ Membership

**Scope:** `vecstudy/harness/churn.py` (auditors `audit_maintainer` and `audit_membership`), `vecstudy/maintain/outofplace.py` (replica metric tracking), and a new suite `tests/test_outofplace_membership.py`.

**Not in scope, and not implemented:** dynamic-Δ calibration, SPANN RNG filtering, query engine modifications, in-place maintainer modifications.

**Status:** complete. `204 passed` (was 198: +6 out-of-place membership and cross-segment ownership tests). Full suite passes cleanly.

---

## 1. Problem and Root Cause

Task 2 of the in-place arm (`R1_T2_Report.md`) established independent fixed-Δ membership verification for single-segment indexes, but explicitly left out-of-place maintenance unchecked (`TypeError: fixed-Δ membership is maintained by InPlaceMaintainer only`). 

Furthermore, `audit_maintainer` suffered from an omission in the out-of-place branch:
* Within each sealed segment, NPA was checked independently.
* However, `all_live_found.update(live_owned)` used a Python `set.update()`, silently collapsing duplicate ownership across multiple sealed segments or between a sealed segment and MemTable.
* Consequently, if a flush or compaction error leaked an owned vector into two segments, or if MemTable held a vector already present in a sealed segment, canonical ownership was violated without raising an error.

---

## 2. Policy and Implementation

### 2.1 Cross-Segment Canonical Ownership Invariant
In an LSM vector database (LSM-VEC §3.2, DESIGN §6.6), every live vector must have **exactly one canonical owner** across the entire database:
* `audit_maintainer` now tracks `global_owner_count: dict[int, list[str]]`.
* If a vector appears as owned in more than one unit (e.g. `sealed[0]` and `sealed[1]`, or `sealed[0]` and `memtable`), it is reported as a fatal ownership conflict:
  `vid {vi} has multiple owners across units: {locations}`.
* MemTable duplicate insertions without prior deletion are also flagged.

### 2.2 Fixed-Δ Replica Membership across Multiple Units
* Each sealed segment $S$ is an independent IVF partition over its live vectors.
* For every non-tombstoned vector owned by posting $p \in S$, its expected memberships are derived strictly from $S$'s leaf centroids and $p$'s centroid using the exact ratio form:
  $$r(x) = \frac{d_2}{d_1} - 1 \le \Delta$$
* MemTable is a flat buffer; all unsealed entries have $members = owned$ (top-2 cap is respected; no spurious replicas).
* Tombstoned vectors in any posting's `members` array are flagged as `dead_members`.
* Duplicate entries in any posting's `members` are flagged as `duplicates`.
* A dedicated helper `audit_outofplace_membership(maint)` provides ergonomic verification while preserving backwards-compatibility with legacy tests.

### 2.3 Replica Metric Accounting in OutOfPlaceMaintainer
* `_flush()` records `replicas_added` for all new replicas assigned during segment build.
* `_compact()` records `membership_vectors_recomputed` for all live vectors re-clustered.
* `stats()` exposes `replicas_added`, `replicas_removed`, and `membership_vectors_recomputed`.

---

## 3. Files Changed

| File | Purpose |
|---|---|
| `vecstudy/harness/churn.py` | Cross-segment duplicate ownership check in `audit_maintainer`; general `_audit_segment_membership`; multi-unit out-of-place auditing in `audit_membership`; added `audit_outofplace_membership` |
| `vecstudy/maintain/outofplace.py` | Replica tracking counters on `_flush` and `_compact`; exposed in `stats()` |
| `tests/test_outofplace_membership.py` **(new)** | 6 focused tests covering clean multi-segment membership, MemTable membership, cross-segment duplicate detection, oracle fault injection, and compaction preservation |

---

## 4. Verification and Reproduction

Run focused tests:
```powershell
python -m pytest tests/test_outofplace_membership.py -v
```
Result: `6 passed in 1.90s`.

Run full test suite:
```powershell
python -m pytest -q
```
Result: `204 passed in 18.36s`.
