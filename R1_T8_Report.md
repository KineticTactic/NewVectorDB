# R1 In-Place Maintenance — Task 8: In-Place vs Out-of-Place Paired Benchmark

Deliverable: `scripts/compare_maintenance.py` (thin entry point) backed by
`vecstudy/harness/compare.py` (all logic, unit-testable) plus
`tests/test_compare_maintenance.py`.

The harness runs **one** initial index and **one** deterministic churn workload
against **both** the real `InPlaceMaintainer` and the real
`OutOfPlaceMaintainer`, and reports search quality, query latency, update
latency, maintenance cost, storage overhead and correctness side by side. No
maintenance or query algorithm is reimplemented, and neither strategy is
modified to look more comparable.

---

## 1. Launch commands

From the repository root (single-thread BLAS for reproducibility):

```bash
# smallest end-to-end run (2 depths x 1 split x 1 delta = 2 cells = 4 runs)
.venv/bin/python scripts/compare_maintenance.py --profile smoke

# default sweep: 4 depths x 2 splits x 2 deltas = 16 cells = 32 runs (dim=128)
.venv/bin/python scripts/compare_maintenance.py --profile quick --blas-threads 1

# same sweep but small MemTable so the out-of-place flush/compaction path fires
.venv/bin/python scripts/compare_maintenance.py --profile smoke \
    --memtable-limit 10 --compaction-threshold 2 --deltas none,0.3

# multi-seed research profile (seed 0,1,2) and explicit axis control
.venv/bin/python scripts/compare_maintenance.py --profile research \
    --depths 1,2,3 --split-thresholds 24,48 --deltas none,0.3 \
    --seeds 0,1,2 --n-postings 4 --beam-widths 8 --k 10 --recall-ks 1,10,100
```

Every flag: `--profile {smoke,quick,research}`, `--dataset`, `--initial-size`,
`--operations`, `--workload {mixed,insert_only,delete_only}`, `--insert-prob`,
`--dim`, `--leaf-size`, `--branching-factor`, `--depths`,
`--split-thresholds`, `--merge-thresholds`, `--deltas`, `--n-postings`,
`--beam-widths`, `--seeds`, `--k`, `--recall-ks`, `--query-count`,
`--checkpoint-interval`, `--memtable-limit`, `--compaction-threshold`, `--out`,
`--overwrite`, `--blas-threads`, `--table-a-rows {final,all,none}`,
`--flush-final`, `--continue-on-audit-failure`. `--profile quick` is the
default. Without `--overwrite` each invocation writes into a unique timestamped
sub-directory of `--out`.

---

## 2. Sample real output

### 2.1 Files produced (quick profile)

```text
Output directory: /tmp/cmp_quick2
Files: results_checkpoints.csv (288 rows), results_raw.csv (288 rows),
       results_paired.csv (144 rows), results_summary.csv (64 rows),
       run_metadata.json (1 rows),
       results_checkpoints.parquet (288 rows), results_paired.parquet (144 rows)
```

`results_raw.csv` and `results_checkpoints.csv` both hold every checkpoint row
(one row per strategy x cell x checkpoint); `results_paired.csv` holds one row
per matched (in-place, out-of-place) checkpoint; `results_summary.csv` holds
`row_kind=run` rows (per run) and `row_kind=aggregate` rows (across seeds).
Parquet copies are written when pyarrow is importable.

### 2.2 Table A — per-run results (`--table-a-rows final`; depth-1 excerpt)

```text
 run   strategy depth split delta seed  cp  ops  live  recall  q_p50ms  q_p95ms  ins_p50  del_p50    mut_ms   scan/q  bloat% split# merge# flush/cp  npa  mem  status
   0    inplace     1    24   n/a    0   8  400   392  0.7875     0.03     0.03     0.01     0.42     613.9     69.2     0.0    210    195        0    0 PASS  PASSED
   0 outofplace     1    24   n/a    0   8  400   392  0.9094     0.09     0.10     0.00     0.00       0.9    336.8     0.0      0      0        0    0  n/a  PASSED
   1    inplace     1    24   0.3    0   8  400   392  0.8094     0.03     0.03     0.01     0.43     889.6     74.4     7.9    210    195        0    0 PASS  PASSED
   1 outofplace     1    24   0.3    0   8  400   392  0.9187     0.12     0.12     0.00     0.00       0.9    461.2    32.4      0      0        0    0  n/a  PASSED
   2    inplace     1    48   n/a    0   8  400   392  0.8437     0.05     0.05     0.01     0.19      47.7    147.3     0.0      3      2        0    0 PASS  PASSED
   2 outofplace     1    48   n/a    0   8  400   392  0.9094     0.09     0.10     0.00     0.00       0.9    336.8     0.0      0      0        0    0  n/a  PASSED
```

`mut_ms` is the cumulative mutation wall time (updates + attributed
maintenance); `scan/q` is scan-bucket distance computations per query; `bloat%`
is membership bloat (replica references over a unique live vector); `split#`/
`merge#` are cumulative in-place structural events; `flush/cp` is the flush
delta in the shown (final) checkpoint; `npa`/`mem` are the two auditors
(`mem = n/a` for out-of-place by construction, see §5). The 210 splits / 195
merges at `split=24` (vs 3 / 2 at `split=48`) reproduce the Task-5 small-leaf
split/merge churn.

### 2.3 Table B — paired comparison (depth-1 excerpt)

`diff = in-place minus out-of-place`; `ratio = in-place / out-of-place`
(> 1 means in-place is slower or larger); `dRec_pp` is recall difference in
percentage points (positive favours in-place).

```text
seed depth split delta  cp ip_recall oop_recall  dRec_pp q_p50_ratio ins_ratio del_ratio  mnt_ratio scan_ratio dBloat_pp   dBytes  ip_corr  oop_corr
   0     1    24   0.3   0    0.8625    0.8625     0.00       1.00       n/a       n/a        n/a       1.00      0.00        0   PASSED    PASSED
   0     1    24   0.3   8    0.8094    0.9187   -10.94       0.26     70.61   2054.92    1134.96       0.16    -24.49     5616   PASSED    PASSED
   0     1    24   n/a   8    0.7875    0.9094   -12.19       0.32     39.16   1995.71     750.83       0.21      0.00     6896   PASSED    PASSED
   0     1    48   0.3   8    0.8844    0.9187    -3.44       0.49     60.70    776.67      27.20       0.39     -5.10    -5728   PASSED    PASSED
   0     1    48   n/a   8    0.8437    0.9094    -6.56       0.51     34.65    901.22       7.43       0.44      0.00    -5056   PASSED    PASSED
```

At checkpoint 0 the arms are bit-identical (same initial state): identical
recall, `scan_ratio = 1.00`, zero bloat/byte difference. As churn proceeds the
gaps open exactly where the design predicts: in-place has far larger
maintenance wall time (`mnt_ratio` up to ~1135x) and more `scan/q` work per
single unit, while out-of-place inserts are near-free (`ins_ratio` ~35-70x)
because they only append to the MemTable.

### 2.4 Cross-seed aggregate (successful runs only; mean ± std)

```text
  strategy depth split delta  nq nok nfail recall_mean recall_std  q_p50_mean ins_p50_mean del_p50_mean mnt_ms_mean bloat_mean  scan_mean
   inplace     1    24   n/a   1   1     0      0.7875     0.0000        0.03         0.01         0.42      574.82       0.00       69.2
outofplace     1    24   n/a   1   1     0      0.9094     0.0000        0.09         0.00         0.00        0.77       0.00      336.8
outofplace     1    24   0.3   1   1     0      0.9187     0.0000        0.12         0.00         0.00        0.75      32.40      461.2
   inplace     1    24   0.3   1   1     0      0.8094     0.0000        0.03         0.01         0.43      855.48       7.91       74.4
```

With `--profile research` (`--seeds 0,1,2`) each configuration has `n_reps = 3`;
`n_ok` counts replicas that passed the correctness gate and `n_failed` counts
those that did not, so failures are visible and never silently dropped
(aggregate means are computed over `n_ok` only).

### 2.5 Out-of-place flush + compaction actually fire

```text
# scripts/compare_maintenance.py --profile smoke --memtable-limit 10 --compaction-threshold 2 ...
 outofplace delta n/a | flush 2 comp 2 segs 2 mem 9 mnt_ms 1.606
outofplace delta 0.3 | flush 2 comp 2 segs 2 mem 9 mnt_ms 1.507
```

---

## 3. How each requirement is met

### 3.1 Identical initial state, identical workload
* Both arms are built from **one** `Segment.build(...).clone()`
  (`_DatasetCache.initial_segment`); `clone()` deep-copies centroids, hierarchy,
  postings, memberships and owned sets, verified mutably independent by
  `test_same_initial_state`.
* Both arms receive the **same** `ChurnTrace` (identical op order and ids) and
  the **same** fixed query set. `test_same_trace_applied_to_both` asserts the
  per-checkpoint cumulative insert/delete counts and live counts are equal, and
  that the recorded `trace_hash` matches.
* The backing store is pre-allocated to `initial_size + reserve_size` rows and
  shared by both arms; an insert selects a reserve row by id, so **no array is
  reallocated or re-bound mid-run** (no stale matrix references).

### 3.2 Paired execution
* Paired runs are **sequential**; the arm that runs first alternates with
  `run_index % 2` (`test_run_sweep_alternates_order`) so order effects cancel
  across the sweep.
* `build_paired` only pairs rows whose full key (dataset, seed, depth, leaf,
  split, merge, delta, MemTable, compaction, k, beam, n_postings, checkpoint)
  matches, and asserts both arms are present (`test_paired_rows_match_only_...`).
* Checkpoint 0 of every pair is the identical initial state, so Table B shows
  zero diff/ratio 1.0 there — an internal sanity check.

### 3.3 Search quality
* Ground truth is recomputed **dynamically** against the currently live set
  after every mutation (`dynamic_ground_truth`); a deleted or non-live vector can
  never appear in ground truth (`test_dynamic_ground_truth_tracks_mutations`,
  `test_ground_truth_excludes_non_live_vectors`).
* Recall@k compares the production `index.query.search` result set with that
  live-only ground truth, and any returned non-live id is counted
  (`deleted_id_retrievals`), which must be 0 (`test_recall_uses_only_live...`).

### 3.4 Query, update and maintenance cost
* Each checkpoint performs a full query pass over the **current** searchable
  units of the arm, with the production query path and a **separate**
  `Counters` object, so query work is never conflated with maintenance counters
  (`test_query_work_independent_of_maintenance_counters`).
* For out-of-place the units include the live MemTable unit plus the current
  sealed segments, so a fresh insert is searchable and `units_searched` reflects
  the real engine state (`test_oop_units_reflect_memtable_and_sealed`,
  `test_oop_flush_and_compaction_counters_advance`).
* Insert/delete/query latencies are logged per operation and reported as
  p50/p95; attributed maintenance wall time and maintenance similarity
  evaluations are reported separately (with the caveat in §5).

### 3.5 Storage overhead
* Per-unit live membership references (`M`) are counted against unique live
  vectors (`N = canonical_owned_live_count`), giving
  `replica_count = M - N` and `membership_bloat_pct = 100 (M - N) / N`
  (`test_bloat_arithmetic_with_replicas`). Index bytes are estimated from array
  shapes/dtypes.
* `scan/q` and `vectors_scanned_per_query` expose the work-normalized view; the
  harness measures actual work rather than assuming a global budget that the
  engine does not provide.

### 3.6 Correctness gate
* Every checkpoint runs on both arms: `audit_maintainer` (coverage, ownership,
  NPA) plus `audit_membership` for in-place. A checkpoint fails when any audit
  fails, a non-live id leaks, membership fails, or canonical ownership does not
  cover the live set.
* A deliberately corrupted index is detected and marked failed with a reason
  (`test_corrupted_index_fails_the_gate`); failed runs are kept in
  `results_summary.csv` and counted in `n_failed`
  (`test_summary_reports_failed_runs_not_dropped`), and aggregate means use only
  successful repetitions (`test_aggregate_means_exclude_failed_runs`).

### 3.7 Reproducibility
* One seeded RNG drives workload, dataset and query set;
  `generate_churn_trace(seed)` reproduces the op list, and re-running a cell
  reproduces recall and structural counters (`test_same_seed_reproduces...`).
* The resolved configuration, environment (Python/numpy/platform/BLAS
  threadpool), git SHA, flush policy, per-checkpoint counts and the explicit
  measurement-limitation list are written to `run_metadata.json`.

---

## 4. Test results

```text
$ .venv/bin/python -m pytest tests/test_compare_maintenance.py -q
21 passed

$ .venv/bin/python -m pytest -q
209 passed
```

`tests/test_compare_maintenance.py` (21 tests) covers all fifteen required
checks plus CLI plumbing: identical initial state, same trace, dynamic ground
truth, live-only recall, out-of-place current units, independent search vs
maintenance work, bloat arithmetic, counter accumulate/reset semantics, seed
reproducibility, pairing, correctness gating, aggregate-over-successful-reps,
CSV/Parquet schema and row counts, the `smoke` command end to end, `resolve_sweep`
axis expansion, and arm-order alternation. The full suite (188 prior + 21) is
green.

---

## 5. Known measurement limitations

Recorded in the module docstring and in `run_metadata.json`
(`measurement_limitations`), and honoured rather than papered over:

1. **Per-unit posting budget.** `search` descends each searchable unit with
   `n_postings` independently, so out-of-place search over `S` units may examine
   up to `S` times the postings of the single in-place unit at the same nominal
   configuration. The engine has no global budget and was not modified; actual
   work is measured and a work-normalized view (`scan/q`,
   `vectors_scanned_per_query`) is provided instead of assuming equality.
2. **Maintenance distance work is a lower bound.** `balanced_kmeans` delegates
   Lloyd iterations to scikit-learn, whose distances are uncounted by design, so
   counters only see NumPy kernels. The undercount is symmetric across arms, but
   absolute maintenance-evaluation counts understate work.
3. **Empty-leaf pruning has no counter** in the in-place arm; it is reported
   `n/a` and never inferred from a leaf-count delta.
4. **Fixed-Δ membership audit is in-place only.** `audit_membership` rejects the
   out-of-place maintainer, so its membership audit is `n/a`; out-of-place NPA is
   checked inside each sealed segment.
5. **RSS** requires `psutil` (not installed here); macOS `ru_maxrss` is *peak*,
   labelled `peak`, and index bytes are an **estimate** from array shapes.
6. **Maintenance time cannot be cleanly separated** from update time: in-place
   maintenance is eager (inside insert/delete) and out-of-place flushes happen
   inside insert. Update latency therefore includes triggered maintenance; the
   attributed maintenance wall time is reported separately with that caveat.
7. **Latency is wall-clock** and outside the algorithmic study (DESIGN §1); it is
   reported with a warm-up pass and pinned BLAS threads.

---

## 6. Files

* `vecstudy/harness/compare.py` — paired runner, metrics, tables, CSV/Parquet/
  JSON writers, CLI (`CompareConfig`, `SweepSpec`, `run_strategy`, `run_sweep`,
  `build_paired`, `build_summary`, `render_table`, `write_outputs`, `main`).
* `scripts/compare_maintenance.py` — thin entry point (BLAS pinning, import path,
  forwards to `compare.main`).
* `tests/test_compare_maintenance.py` — 21 tests.
* Runtime outputs (per invocation): `results_raw.csv`,
  `results_checkpoints.csv`, `results_paired.csv`, `results_summary.csv`,
  `run_metadata.json`, plus `.parquet` copies.
