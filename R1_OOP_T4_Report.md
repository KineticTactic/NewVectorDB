# R1 Out-Of-Place Maintenance — Task 4: Continuous Live-Churn Monitor Support

**Scope:** `vecstudy/harness/live.py`, `scripts/live_churn.py`, and `tests/test_live_churn.py`. Adapting the continuous live-churn monitor to support both in-place (`SPFresh`) and out-of-place (`LSM-Vec`) maintenance arms with zero regressions and symmetric audit coverage.

**Not in scope, and not implemented:** in-place maintainer modifications, sweep runner execution (§11 depth study, covered in Task 5).

**Status:** complete. `232 passed` (was 225: +7 focused live-churn out-of-place tests).

---

## 1. Objectives

To achieve parity with Task 7 of the in-place arm (`R1_T7_Report.md`), the live-churn monitoring harness (`vecstudy.harness.live`) and entry point script (`scripts/live_churn.py`) were extended to support the out-of-place strategy:
1. **Strategy Selection:** Add `--strategy {inplace,outofplace}` CLI option (defaulting to `inplace`).
2. **Buffer & Compaction Parameters:** Expose `--memtable-limit` and `--compaction-threshold` for out-of-place tuning.
3. **Multi-Segment Backing Store Re-binding:** Ensure `GrowingVectorStore.bind()` atomically updates the vector matrix reference for all sealed segments in `maint._sealed` as well as dynamic MemTable units.
4. **Structural Event Observation:** Define `ObservedOutOfPlaceMaintainer` to observe and record real `FLUSH` and `COMPACT` events without altering underlying maintenance logic.
5. **Dual Auditor Integration:** Route membership validation through `audit_outofplace_membership()` when running out-of-place while maintaining strict NPA auditing across all units.
6. **Per-Operation Formatting:** Extend console output and summaries to report sealed segment counts, pending MemTable entries, flushes, and compaction events.

---

## 2. Design & Implementation

### 2.1 Backing Store Re-binding (`GrowingVectorStore.bind`)
In out-of-place maintenance, multiple immutable sealed segments coexist alongside an in-memory MemTable. When `GrowingVectorStore` grows its float32 matrix, it now re-points:
- `maint._vectors = self._data`
- `seg.vectors = self._data` for every segment in `maint._sealed`
- `unit.vectors = self._data` for every unit returned by `maint.searchable_units()`

This prevents any sealed segment or newly initialized unit from referencing stale or truncated vector buffers.

### 2.2 Structural Event Observation (`ObservedOutOfPlaceMaintainer`)
Subclasses `OutOfPlaceMaintainer` and overrides `_flush` and `_compact`:
- `FLUSH`: Records `segment_count` after the flush and `vectors_affected` (number of live vectors flushed from the MemTable).
- `COMPACT`: Records `segment_count`, `vectors_affected` (vectors re-clustered into the consolidated segment), and `hierarchy_rebuilds`.
- Both events are formatted clearly in the per-operation output stream:
  ```text
  [00:00:00.124] #0004 INSERT vid=24 live=24 leaves=2 sealed=2 flushes=1 compactions=0 recall@10=1.0000 NPA=PASS MEMBERSHIP=PASS
    FLUSH sealed_segments=2 vectors=4
  ```

### 2.3 Configuration & Auditor Routing
`LiveChurnConfig` and `LiveChurnMonitor` were updated to:
- Accept `strategy="outofplace"` alongside `--memtable-limit` and `--compaction-threshold`.
- Call `self._maint.maintain()` following each mutation, driving compaction triggers when sealed segment counts or tombstone ratios reach thresholds.
- Audit out-of-place membership using `audit_outofplace_membership()`, ensuring MemTable entries are flat ($members = owned$) and sealed segments satisfy fixed-$\Delta$ replica bounds.

---

## 3. Test Coverage (`tests/test_live_churn.py`)

Seven new automated tests were added:
1. `test_outofplace_cli_parse_and_resolve`: Validates CLI argument parsing and config resolution for out-of-place flags.
2. `test_outofplace_observed_maintainer_lifecycle`: Executes 20 churn operations, verifying `FLUSH` and `COMPACT` events are logged, and `NPA=PASS` / `MEMBERSHIP=PASS` on every operation.
3. `test_outofplace_store_growth_rebinds_sealed_and_memtable`: Triggers multiple dynamic matrix resizes and verifies all sealed segments and search units re-bind correctly.
4. `test_outofplace_exhaustive_search_recall_is_one`: Confirms recall@k is exactly 1.0 under exhaustive search across sealed segments and MemTable.
5. `test_outofplace_deleted_ids_are_not_returned`: Verifies that deleted vectors are never returned in search results across multi-segment units.
6. `test_outofplace_audit_failure_is_visible_and_stops`: Verifies that invariant violations immediately halt the monitor and format error diagnostics.
7. `test_outofplace_main_runs_finite_and_exits_zero`: Exercises the CLI entry point end-to-end with `--strategy outofplace`.

---

## 4. Verification

Run live churn monitor tests:
```powershell
python -m pytest tests/test_live_churn.py -v
```
Result: `32 passed in 2.18s`.

Run complete repository test suite:
```powershell
python -m pytest -q
```
Result: `232 passed in 15.18s`.
