# R1 Out-Of-Place Maintenance — Task 3: Integration & Stability Matrix across Depths 1–4

**Scope:** `tests/test_outofplace_integration.py`, cross-checking the OutOfPlace maintainer against both independent validators (`audit_maintainer`, `audit_outofplace_membership`) and query invariants across a fixed-seed matrix.

**Not in scope, and not implemented:** in-place maintainer modifications, dynamic-Δ calibration, SPANN RNG filtering, query engine modifications.

**Status:** complete. `225 passed` (was 204: +21 parameterized integration tests).

---

## 1. Objectives

To match the rigorous validation established for the in-place arm in Task 5 (`R1_T5_Report.md`), this task provides a deterministic integration test suite for the out-of-place arm across all architectural dimensions:
1. All four hierarchy depths (depth 1 flat routing, depths 2, 3, 4 hierarchical routing).
2. Overlap configurations ($\Delta = \text{None}$ and $\Delta = 0.30$).
3. Write workloads (`mixed` balanced churn, `insert_only`, and `delete_only`).
4. Buffer and compaction thresholds (`memtable_limit` $\in \{15, 25\}$, `compaction_threshold` $\in \{3, 4\}$).

---

## 2. Invariants Audited per Operation

Across every single step in every workload, five independent invariants are validated:
1. **NPA & Coverage (`audit_maintainer`):** Every non-tombstoned vector has exactly one canonical owner across all searchable units. For each sealed segment, every owned vector is nearest to its owning posting centroid.
2. **Fixed-Δ Membership (`audit_outofplace_membership`):** Replicas in sealed segments strictly satisfy the ratio condition $r(x) = d_2/d_1 - 1 \le \Delta$. Unsealed MemTable entries have $members = owned$.
3. **Immutable Tombstone Semantics:** In an LSM-style vector index (LSM-VEC §3.2), sealed segments are write-once disk blocks. Deletions do not rewrite on-disk postings; rather, they mark segment tombstones. Compaction re-clusters live vectors and purges dead members.
4. **Hierarchy Structural Validity:** Positional posting IDs match leaf-centroid rows; higher routing levels maintain complete, disjoint partitions of child nodes.
5. **Query Soundness:** Exhaustive search across all searchable units returns bit-for-bit identical similarity scores and top-k rankings as brute force over currently live vectors, with zero tombstoned IDs and zero duplicates.

---

## 3. Results Matrix

| Depth | $\Delta$ | MemTable Limit | Compaction Threshold | Workload | Operations | Status |
|---|---|---|---|---|---|---|
| 1 | None | 15 | 3 | mixed | 80 | PASSED |
| 1 | None | 25 | 4 | mixed | 80 | PASSED |
| 1 | 0.3 | 15 | 3 | mixed | 80 | PASSED |
| 1 | 0.3 | 25 | 4 | mixed | 80 | PASSED |
| 2 | None | 15 | 3 | mixed | 80 | PASSED |
| 2 | None | 25 | 4 | mixed | 80 | PASSED |
| 2 | 0.3 | 15 | 3 | mixed | 80 | PASSED |
| 2 | 0.3 | 25 | 4 | mixed | 80 | PASSED |
| 3 | None | 15 | 3 | mixed | 80 | PASSED |
| 3 | None | 25 | 4 | mixed | 80 | PASSED |
| 3 | 0.3 | 15 | 3 | mixed | 80 | PASSED |
| 3 | 0.3 | 25 | 4 | mixed | 80 | PASSED |
| 4 | None | 15 | 3 | mixed | 80 | PASSED |
| 4 | None | 25 | 4 | mixed | 80 | PASSED |
| 4 | 0.3 | 15 | 3 | mixed | 80 | PASSED |
| 4 | 0.3 | 25 | 4 | mixed | 80 | PASSED |
| 1 | 0.3 | 15 | 3 | insert_only | 80 | PASSED |
| 1 | 0.3 | 15 | 3 | delete_only | 80 | PASSED |
| 3 | None | 20 | 3 | insert_only | 80 | PASSED |
| 3 | None | 20 | 3 | delete_only | 80 | PASSED |
| 4 | 0.3 | 15 | 3 | delete_only | 80 | PASSED |

Total: 21 / 21 passed.

---

## 4. Verification and Reproduction

Run integration tests:
```powershell
python -m pytest tests/test_outofplace_integration.py -v
```
Result: `21 passed in 3.92s`.

Run entire test suite:
```powershell
python -m pytest -q
```
Result: `225 passed in 15.05s`.
