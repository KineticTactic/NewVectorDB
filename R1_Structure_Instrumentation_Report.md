# R1 — Structural Maintenance Instrumentation Report

Additive, observational instrumentation of the *index shape* produced by
maintenance, emitted once per `(strategy, base config, seed, checkpoint)`.
No indexing, maintenance or query semantics were changed, and the large
split-threshold sweep was **not** run (it is a separate experiment that will
consume this instrumentation).

## What changed

| File | Change |
| --- | --- |
| `vecstudy/harness/structure.py` (new) | Pure summarizers, event/timing blocks, row builder, instrumented maintainer subclasses, metadata. |
| `vecstudy/harness/structure_report.py` (new) | Single structural diagnostics figure + `structure_threshold_series.csv`. |
| `scripts/plot_structure.py` (new) | CLI wrapper for the figure. |
| `vecstudy/harness/search_budget.py` (modified) | Uses the instrumented maintainers; captures a structure row per checkpoint; writes `results_structure_checkpoints.csv`/`.parquet`; records `structure_metrics` + `n_structure_rows` in `run_metadata.json`. |
| `tests/test_structure.py` (new) | 19 focused tests. |

Instrumented maintainers only **wrap** existing hooks:
`InstrumentedInPlaceMaintainer` times `_split`, `_merge_after_delete` and counts
`_prune_empty_leaves`; `InstrumentedOutOfPlaceMaintainer` times `_flush` and
`_compact`. A reentrancy depth guard means nested hooks (e.g.
`_merge_after_delete → _split_oversize → _split`) are counted once.

## Output path and schema

`<out_dir>/results_structure_checkpoints.csv` (+ `.parquet`), one row per
`(strategy, base config, seed, checkpoint)` — preset-independent (values repeat
if copied into per-preset rows).

* Identity/join columns match the raw schema: `experiment_id`, `run_index`,
  `strategy`, `dataset`, `dim`, `initial_size`, `reserve_size`, `workload`,
  `insert_prob`, `seed`, `trace_seed`, `trace_hash`, `depth_config`,
  `leaf_size`, `branching_factor`, `split_threshold`, `merge_threshold`,
  `delta`, `k`, `memtable_limit`, `compaction_threshold`, `checkpoint`,
  `operations_processed`, `live_vector_count`, `correctness_status`,
  `checkpoint_correct`.
* Structure: `live_posting_count`, `posting_size_{median,mean,p90,p95,p99,max}`,
  `fraction_postings_over_split_threshold`,
  `fraction_vectors_in_overthreshold_postings`, `member_list_size_{count,median,p95,max}`,
  `searchable_unit_count`, `sealed_segment_count`, `memtable_live_vector_count`.
* Events (interval + cumulative): `splits_*`, `merges_*`, `pruned_postings_*`,
  `flushes_*`, `compactions_*`, `compaction_vectors_reclustered_*`.
* Timing (ms): `maintenance_wall_ms_{interval,cumulative}`,
  `update_wall_ms_{interval,cumulative}`.

Conventions (also in `run_metadata.json → structure_metrics`):

* Sizes count **live, canonical-owned** vectors only; replicas are reported
  separately via `member_list_size_*` and never fold into canonical sizes.
* Out-of-place stats cover **sealed** segment leaf postings; the MemTable is
  reported separately and excluded from the posting-size distribution.
* Percentiles: `numpy.percentile`, default `linear` interpolation.
* Fractions are `[0,1]`, not percent. Over-threshold is strict `size > split_threshold`.
* Interval = per-checkpoint delta of the existing cumulative counters;
  production counters are never reset. Checkpoint-0 intervals and wall-time
  intervals are `0` (initial build excluded); cumulative wall time is monotone.
* Empty structure → size stats `nan`, counts `0`, fractions `0.0`.
* **N/A vs zero:** `None`/empty (null in Parquet) = not applicable to the
  strategy; `0` = applicable but did not happen.
* Timers use `perf_counter_ns` around maintenance mutations only; search and
  ground-truth work are excluded. `update_wall_ms` (whole insert/delete op) is
  reported separately.

## Unsupported / N-A metrics

* `pruned_postings` — in-place only (empty-leaf pruning); N/A for out-of-place.
* `sealed_segment_count`, `memtable_live_vector_count` — out-of-place only.
* Posting-size percentiles for out-of-place are over sealed segment leaf
  postings only (all hierarchies at their leaf level).

## Test results

* `tests/test_structure.py`: **19 passed**.
* Full suite: **329 passed** (was 310; +19).

Coverage includes: percentile/fraction arithmetic; strict `>`; empty-structure
safety; live-only sizes; replica counted once and reported separately;
out-of-place aggregation + separate MemTable; interval deltas (`2→5→5` ⇒
`2→3→0`) without resetting source counters; cumulative monotonicity;
checkpoint-0 zero convention; N/A distinct from zero (CSV empty vs `"0"`;
Parquet null vs `3`); nested-hook single-counting; timer isolation from
query work; snapshot immutability; **behaviour preservation** (instrumented vs
vanilla maintainers give identical postings, stats and counters); CSV
schema/uniqueness; CLI smoke producing the structure CSV + metadata; the
threshold-series/figure pipeline.

## Smoke run (validation, not the deferred sweep)

```
.venv/bin/python scripts/compare_search_budget.py --phase calibration \
  --initial-size 120 --operations 60 --checkpoint-interval 20 \
  --depths 1,2 --split-thresholds 24,48 --deltas none \
  --beam-width-values 2 --n-postings-values 1 --calibration-seeds 0 \
  --blas-threads 1 --out results/search_budget/structure_smoke --overwrite
.venv/bin/python scripts/plot_structure.py --input results/search_budget/structure_smoke
```

Outputs: `results/search_budget/structure_smoke/results_structure_checkpoints.{csv,parquet}`
and `.../structure/fig_structure_posting_sizes.{png,pdf,svg}` +
`structure_threshold_series.csv`.

Sample interpretation (in-place, `split_threshold=24`, `leaf_size=40`):

* Checkpoint 0: 3 leaves, median 41, max 55, 2/3 postings over 24 — the initial
  build ignores `split_threshold` (it is a trigger, not a cap).
* Checkpoint 1: 4 splits fired (`splits_interval=4`), median drops to 20.5,
  max 23, fraction-over-threshold `0.0`; cumulative maintenance `4.45 ms`.
* In-place splits then stay near or below the threshold; out-of-place shows the
  same initial over-threshold shape but no splits (its reductions come from
  flush/compaction instead), and `memtable_live_vector_count` tracks the
  buffered tail.

This is the expected contrast the deferred sweep will quantify: in-place
`split_threshold` directly reshapes leaves, while out-of-place reshaping is
governed by flush/compaction.
