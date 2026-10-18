# R1 In-Place Maintenance — Task 7: Continuous Live Churn and Recall Monitor

Deliverable: `scripts/live_churn.py` (entry point) backed by
`vecstudy/harness/live.py` (all logic, so it is unit-testable) plus
`tests/test_live_churn.py`.

The monitor runs an **open-ended** balanced churn cycle against the real
`InPlaceMaintainer`, recomputes recall@k against the currently live vectors after
**every** operation, prints one line per operation (with real split/merge
events), and keeps running until Ctrl+C. No maintenance or query algorithm was
reimplemented or redesigned; `OutOfPlaceMaintainer` was not touched.

---

## 1. Launch commands

From the repository root:

```bash
# default configuration, runs until Ctrl+C
.venv/bin/python scripts/live_churn.py

# the task's configurable example (verified to run)
.venv/bin/python scripts/live_churn.py \
  --initial-size 300 \
  --k 10 \
  --query-count 32 \
  --depth 2 \
  --split-threshold 40 \
  --delta 0.3 \
  --interval 0.2 \
  --seed 7
```

The script pins single-thread BLAS before importing numpy, makes the checkout
importable, and forwards to `vecstudy.harness.live.main`. `--max-operations N`
exists only as a demo/testing aid; the default is **unbounded** (run until
Ctrl+C).

All CLI flags: `--k`, `--query-count`, `--seed`, `--initial-size`, `--dataset`,
`--dim`, `--depth`, `--leaf-size`, `--branching-factor`, `--split-threshold`,
`--merge-threshold`, `--delta` (number or `none`/`off`), `--interval`,
`--audit-every`, `--summary-every`, `--n-postings`, `--beam`,
`--continue-on-audit-failure`, `--max-operations`. Values are validated and the
resolved configuration is printed before the loop starts.

---

## 2. Sample real output

A short run (`timeout -s INT`, small leaves so split/merge churn is visible):

```text
$ .venv/bin/python scripts/live_churn.py --initial-size 60 --dim 32 \
    --leaf-size 6 --split-threshold 6 --branching-factor 8 --k 5 \
    --query-count 8 --depth 2 --delta 0.3 --interval 0.1 --seed 7 --summary-every 10

Live churn monitor — in-place maintenance
  dataset=synthetic dim=32 seed=7 initial_size=60 live=60 postings=10 depth=2
  leaf_size=6 branching_factor=8 split_threshold=6 merge_threshold=3 delta=0.30
  k=5 query_count=8 interval=0.10s audit_every=1
  (unique ids & tombstones grow without bound; matrix doubles on demand)
  initial recall@5=0.6000
[00:00:00.002] #0001 INSERT vid=60 live=61 leaves=11 splits=1 merges=0 recall@5=0.5750 NPA=PASS MEMBERSHIP=PASS
  SPLIT pid=1 new_pid=10 vectors_reassigned=2 hierarchy_rebuilds=0
  delta: splits=+1 replicas_added=+2 replicas_removed=+2 vectors_reassigned=+2 membership_vectors_recomputed=+61 maint_similarity_evaluations=+671 prunes=not_instrumented
[00:00:00.116] #0002 DELETE vid=4 live=60 leaves=14 splits=5 merges=1 recall@5=0.5750 NPA=PASS MEMBERSHIP=PASS
  MERGE survivor=0 removed=1 vectors_reassigned=2 hierarchy_rebuilds=1
  SPLIT pid=3 new_pid=10 vectors_reassigned=3 hierarchy_rebuilds=0
  SPLIT pid=7 new_pid=11 vectors_reassigned=1 hierarchy_rebuilds=0
  SPLIT pid=9 new_pid=12 vectors_reassigned=2 hierarchy_rebuilds=0
  SPLIT pid=10 new_pid=13 vectors_reassigned=0 hierarchy_rebuilds=0
  delta: splits=+4 merges=+1 hierarchy_rebuilds=+1 replicas_added=+24 replicas_removed=+4 vectors_reassigned=+6 merge_vectors_reassigned=+2 merge_candidates_examined=+2 membership_vectors_recomputed=+300 maint_similarity_evaluations=+3840 prunes=not_instrumented
[00:00:00.218] #0003 INSERT vid=61 live=61 leaves=14 splits=5 merges=1 recall@5=0.5750 NPA=PASS MEMBERSHIP=PASS
[00:00:00.324] #0004 DELETE vid=55 live=60 leaves=13 splits=5 merges=2 recall@5=0.6250 NPA=PASS MEMBERSHIP=PASS
  MERGE survivor=3 removed=6 vectors_reassigned=3 hierarchy_rebuilds=1
...
--- summary @ op 10: inserts=5 deletes=5 splits=5 merges=2 live=60 postings=13 recall mean=0.6075 best=0.6500 worst=0.5750 avg_op=1.36ms
...
[00:00:01.813] #0018 DELETE vid=9 live=60 leaves=13 splits=8 merges=5 recall@5=0.5500 NPA=PASS MEMBERSHIP=PASS
  MERGE survivor=0 removed=6 vectors_reassigned=5 hierarchy_rebuilds=1
  MERGE survivor=0 removed=7 vectors_reassigned=3 hierarchy_rebuilds=1
  MERGE survivor=0 removed=4 vectors_reassigned=4 hierarchy_rebuilds=1
  SPLIT pid=7 new_pid=11 vectors_reassigned=1 hierarchy_rebuilds=0
  SPLIT pid=8 new_pid=12 vectors_reassigned=1 hierarchy_rebuilds=0
  delta: splits=+2 merges=+3 hierarchy_rebuilds=+3 replicas_added=+14 replicas_removed=+2 vectors_reassigned=+2 merge_vectors_reassigned=+12 merge_candidates_examined=+16 membership_vectors_recomputed=+180 maint_similarity_evaluations=+2280 prunes=not_instrumented

Interrupted (Ctrl+C) — shutting down.
====================================================
Final summary
  operations=19 inserts=10 deletes=9
  splits=9 merges=5 hierarchy_rebuilds=5
  live=61 postings=14 tombstones=9
  latest recall@5=0.5500 best=0.6500 worst=0.5500
  auditor: NPA=PASS MEMBERSHIP=PASS
  stopped: Ctrl+C
====================================================
```

Every line carries elapsed time, op number, op type, vector id, live count,
leaf count, cumulative splits/merges, recall@k, and both auditor statuses.
Operations that cause several structural events print **every** event (op #0002
and #0018 above), so structural churn is never hidden. Splits/merges print their
real posting ids; prunes have no counter in the maintainer and are reported as
`prunes=not_instrumented` rather than inferred.

---

## 3. How each requirement is met

### 3.1 Data, ids, storage
* Vector ids are **rows** of one float32 matrix shared by the maintainer and its
  single segment. New ids are assigned monotonically and never reused.
* Initial build uses `InPlaceMaintainer(..., initial_vids=...)` (the real
  `Segment.build` path). New vectors are appended by
  `GrowingVectorStore`, which reallocates the matrix and then re-binds **both**
  `maint._vectors` and every `unit.vectors` (`GrowingVectorStore.bind`) — the
  array is never replaced in one holder while leaving a stale reference in
  another. Verified by `test_growth_rebinds_maintainer_and_segment`.
* New vectors are copies of a uniformly random *initial* base row (row in
  `[0, initial_size)`), chosen by the seeded churn RNG, each with a distinct id.
  Copying an existing value is documented and keeps the live distribution
  stationary. Verified: `test_fresh_insert_is_retrievable_by_production_search`.

### 3.2 Balanced, bounded churn
* Each cycle is one INSERT (fresh id) then one DELETE of a random *live* victim,
  excluding the id the cycle just inserted so the fresh vector participates.
  Live count therefore stays within ±1 of `initial_size`.
* Victim choice is reproducible (`numpy.random.RandomState(seed)`); the loop does
  not stop when any pool is exhausted (there is no finite reserve) — verified by
  `test_loop_runs_past_initial_pool` (40 ops from `initial_size=10`).

### 3.3 Recall after every operation
* Fixed query set (`make_synthetic(..., seed=seed+1)`), fixed for the whole run.
* After each mutation: exact brute-force top-k over **only the current live
  vectors** (tombstones excluded), mapped back to global ids, compared with the
  production `index.query.search` via `recall_at_k`. No caching; ground truth is
  recomputed every time (`test_recall_uses_only_current_live_vectors`).
* Exhaustive-search baseline: with `--n-postings` large enough to fetch every
  posting, recall is exactly 1.0 (`test_exhaustive_search_recall_is_one`).
* If `live < k`, `recall_at_k` uses the live-neighbour denominator; if no live
  vectors remain, recall is `N/A` (`test_recall_is_na_when_no_live_vectors`).
* Recall and the audits are independent: recall can be low with audits PASSING
  (visible in §2), and a corrupt index is flagged by the audits, not by recall.

### 3.4 Structural / correctness events
* Splits, merges, hierarchy rebuilds, ownership reassignment, replica
  add/remove, membership recomputations and maintenance similarity evaluations
  are read as **counter deltas** per operation. The real `_split`/`_merge_pair`
  calls are additionally observed by `ObservedInPlaceMaintainer`, which records
  ids from the actual maintenance call (it only observes — it calls `super()` and
  never alters maintenance).
* `test_split_and_merge_events_are_reported` and
  `test_structural_event_ids_are_valid` assert real events are surfaced.

### 3.5 Auditors
* `audit_maintainer` (coverage + unique ownership + NPA) and
  `audit_membership` (independent fixed-Δ oracle) run after every mutation by
  default (`--audit-every 1`; `0` disables; `N` throttles).
* On `FAIL`, the summary is printed with `!!` detail (violation counts + a
  sample) and the run **stops by default** so the failure is reproducible;
  `--continue-on-audit-failure` opts out. Verified by
  `test_audit_failure_is_visible_and_stops` and
  `test_continue_on_audit_failure_keeps_running`. The auditors are read-only; the
  monitor never repairs from them.

### 3.6 Shutdown
* `KeyboardInterrupt` is caught: the loop stops generating operations, prints a
  final summary (ops, inserts, deletes, splits, merges, live, postings,
  tombstones, latest/best/worst recall, last auditor status, stop reason) and
  exits **successfully** (the process returns 0; `timeout` reports 124 only
  because it is `timeout` that fired the signal). Verified by
  `test_keyboard_interrupt_prints_summary_and_keeps_index_valid`, which also
  asserts both audits still pass afterwards.
* Unexpected exceptions are not swallowed; outside `KeyboardInterrupt` they
  propagate with a traceback.

---

## 4. Test results

Focused:

```
.venv/bin/python -m pytest tests/test_live_churn.py -q   ->  25 passed in 1.45s
```

Full suite (unchanged behaviour elsewhere):

```
.venv/bin/python -m pytest -q                            -> 188 passed in 6.40s
```

Covered by `tests/test_live_churn.py`: recall over live only and recomputed;
fresh-id insert extends storage and is retrievable; ids monotonic / never
reused; per-op recall line; real split/merge events; loop runs past the initial
pool; `stop()` and Ctrl+C both yield a final summary with a valid index; audit
failure visible + safe stop (and opt-out); fixed seed reproduces operation
choices (and a different seed changes them); CLI parsing/validation; `main()`
exits 0.

---

## 5. State of the preceding numerical task (Task 6)

The Task 5 report flagged a **zero-tolerance auditor** brittleness at the Δ
boundary (float32 `1 - sim` cancellation). That was fixed in commit `9f86d6f`
(`fix(membership): compute the fixed-Δ distance statistic in float64`): the Δ
decision points and the oracle now use float64 similarity kernels. The
`R1_T5_Report.md` §7 "auditor Δ-boundary brittleness" note is therefore
**superseded** — the oracle is now numerically reliable and no blocker remains.
This monitor consumes the auditors and Δ semantics unchanged.

---

## 6. Known limitations

* **Dataset support.** Only `--dataset synthetic` is implemented (the task's
  "where supported"). Supporting SIFT would require fetching a ~168 MB corpus
  and defining a new-vector source; not done.
* **Unbounded memory by design.** The live count is constant, but unique ids,
  the tombstone set and the backing matrix all grow by one row per cycle and are
  never freed. A long run eventually consumes memory; this is documented in the
  header line, the module docstring and here. This is inherent to "never reuse
  ids" plus "never stop".
* **New vectors are copies** of existing base rows, so the churn does not
  introduce genuinely novel vectors; the distribution is stationary.
* **No prune counter.** The maintainer counts splits and merges but not
  empty-leaf prunes; prunes are shown as `not_instrumented` rather than inferred
  from leaf-count change (which the task forbids for splits/merges).
* **Merges are threshold-dependent.** With the default
  `merge_threshold = max(1, split_threshold//2)`, a posting must fall below that
  live-owned count to merge; large `--split-threshold` values (e.g. the default
  example's 40) may run for a while before any merge appears. Small leaves show
  merges immediately (see §2). This is the same split/merge regime audited in
  Task 5.
* **Recall depends on the query config.** Defaults (`--n-postings 1`,
  `--beam 8`) are the project defaults and give moderate recall; raise
  `--n-postings` to approach the exhaustive 1.0 baseline. Recall is a quality
  signal, not a correctness audit.
* **Ctrl+C mid-operation.** Between operations shutdown is fully clean. If the
  interrupt lands inside a single in-flight mutation, that mutation may be
  incomplete; the process still prints a final summary and exits normally. The
  merge path is atomic for ordinary exceptions (snapshot/rollback) but
  `KeyboardInterrupt` is a `BaseException` and bypasses that rollback — the index
  is discarded on exit, so there is no observable half-state.
* **Single segment assumption.** The monitor grows and re-binds the in-place
  arm's single segment; it is not written for the out-of-place multi-segment
  layout.

---

## 7. Files

* `scripts/live_churn.py` — thin entry point (thread pinning, path, `main`).
* `vecstudy/harness/live.py` — `GrowingVectorStore`,
  `ObservedInPlaceMaintainer`, `LiveChurnConfig`, `LiveChurnMonitor`, CLI.
* `tests/test_live_churn.py` — 25 tests.
