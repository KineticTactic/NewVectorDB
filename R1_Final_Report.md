# R1 Final Comparison — Per-Stratum Recall-vs-Work Frontiers

Deliverable: `scripts/compare_frontier.py` (thin entry point) backed by
`vecstudy/harness/frontier.py` (all logic, unit-testable) plus
`tests/test_frontier.py`, and the figures/report generator
`vecstudy/harness/plot_search_budget.py` (`scripts/plot_search_budget.py`,
`tests/test_plot_search_budget.py`).

This is the **final** in-place vs out-of-place comparison. It reads the raw
per-preset table from an **already-completed** Task-9 evaluation
(`scripts/compare_search_budget.py`) and, within each experimental stratum,
builds an independent Recall@10-vs-work frontier per strategy and compares them
at common, *observed* scan budgets. The benchmark is **not** rerun and the source
files are never modified.

---

## 1. Why a second analysis (and what changed since Task 9)

Task 9 (`R1_T9_Report.md`) already replaced the nominal-configuration Task-8
comparison with an observed-work comparison, but its headline came from a
**pooled** frontier and from **matched-preset pairing**. Two problems motivated
this final analysis:

1. **The pooled frontier mixes heterogeneous configurations** (different depths,
   splits, deltas, seeds), so "in-place ≥ out-of-place at every budget" was a
   best-achievable aggregate, not a within-configuration statement.
2. **Matched-preset pairing conflates operating point with strategy**: when the
   (beam × n_postings) grid is coarse, two presets at equal *measured* scan can
   still have very different recall (Task 9 §5.2 showed a 24–27 pp artifact at
   checkpoint 0 where the two arms are the *identical* index).

The final analysis fixes both by (a) grouping every result into a **stratum =
base config × seed × checkpoint** and comparing strategies *inside* the
stratum, and (b) treating the measured scan count as the budget, so each
strategy is scored by its **best preset under the same cap** rather than by a
single matched pair. It also re-ran the Task-9 evaluation once with a **denser
56-preset** grid to shrink the discrete gap between the two strategies' work
levels.

---

## 2. Launch commands

Repository root, single-thread BLAS. The frontier analysis is **post-processing
of an existing run** — it takes a Task-9 output directory and never replays the
workload:

```bash
# (a) Denser-grid held-out evaluation: same protocol as the Task-9 evaluation,
#     but a 56-preset grid (7 beams x 8 n_postings), reused calibration pairs.
.venv/bin/python scripts/compare_search_budget.py --phase evaluate \
    --calibration-results results/search_budget/calibration \
    --initial-size 2000 --operations 1000 --query-count 64 \
    --checkpoint-interval 250 \
    --depths 1,2,3,4 --split-thresholds 48,192 --deltas none,0.3 \
    --evaluation-seeds 3,4,5 \
    --beam-width-values 1,2,4,8,16,32,64 \
    --n-postings-values 1,2,3,4,6,8,12,16 \
    --blas-threads 1 --out results/search_budget/evaluation_dense

# (b) Per-stratum frontier comparison on the dense run.
.venv/bin/python scripts/compare_frontier.py \
    --raw results/search_budget/evaluation_dense/<stamp> \
    --out results/search_budget/frontier_dense

# (c) Publication figures + research_figures_report.md (post-processing only).
.venv/bin/python scripts/plot_search_budget.py \
    --input results/search_budget/evaluation_dense/<stamp>
```

Frontier flags: `--raw` (run dir or a `results_budget_raw.{csv,parquet}`),
`--out`, `--overwrite`, `--match-within` (default 5.0), `--match-approx`
(default 10.0), `--bootstrap` (default 2000), `--bootstrap-seed`
(default 20261019).

---

## 3. Sample real output

### 3.1 Files produced

```text
Evaluation (dense): results/search_budget/evaluation_dense/sb-20261019-021313-13046
  results_budget_raw.csv: 26880 rows (56 presets x 2 strategies x 240 strata)
  workload: synthetic dim=128, initial_size=2000, operations=1000,
            query_count=64, checkpoint_interval=250 (5 checkpoints),
            depths 1-4, splits 48/192, deltas none/0.3, seeds 3/4/5

Frontier: results/search_budget/frontier_dense/fr-20261019-030115-17066
  results_budget_frontier_by_stratum.csv (5397 rows)
  results_budget_work_overlap.csv        (240 rows)
  results_budget_frontier_paired.csv     (720 rows = 240 strata x 3 budgets)
  results_budget_frontier_summary.csv    (60 rows = 48 cells + 12 depth macros)
  run_metadata.json

Figures: results/search_budget/evaluation_dense/sb-20261019-021313-13046/figures/
  fig01_recall_work_frontier.{png,pdf,svg}
  fig02_recall_gain_by_work_budget.{png,pdf,svg}
  figure_data/fig01_recall_work_frontier.csv
  figure_data/fig02_recall_gain_by_work_budget.csv
  research_figures_report.md
analysis_summary.csv (at the evaluation-dir root)
```

The coarse Task-9 comparison (`results/search_budget/frontier/fr-...`) is kept
for the density contrast in §7; the **final** numbers are the dense run.

### 3.2 Validation and coverage

```text
validation: ok=True  n_rows=26880  n_strata=240  n_unpaired_strata=0
            n_basis_mismatches=0  has_dedicated_query_counters=True
```

Both arms of every stratum share one `dataset_checksum`, `trace_hash`,
`operations_processed` and `live_vector_count`, so the frontier difference is
not a "different data" artifact. **All 240 strata** have a non-empty common
observed work interval (median width ≈ 687 scans/query), so no budget is
fabricated and `no_common_work_range` never fires here.

### 3.3 Matching, dense grid

```text
per-stratum (config x seed x checkpoint) pair status over 720 rows:
  matched (<=5% scan gap): 352   approximate (<=10%): 125   unmatched: 243
mean relative scan gap on matched rows: 1.54%
```

For contrast the **coarse** 20-preset Task-9 grid gave `matched=211`,
`approximate=64`, `unmatched=445` with a ≈15.7 % median gap. The denser grid
nearly doubles the matched fraction (29 % → 49 %) and cuts the residual gap by
~10x on matched rows — the discretisation was indeed the dominant limitation.

### 3.4 Primary result — per-stratum frontier at common budgets

Budgets are data-derived **within each stratum** at 25 % (low) / 50 % (middle) /
75 % (high) through that stratum's common observed scan interval. Per strategy
the score is `max(recall over presets with measured work <= B)`. `Δrecall_pp`
(macro mean, positive = in-place higher) with a 95 % bootstrap interval that
resamples **seeds** (3 held-out seeds), pooled over split/delta:

```text
depth budget   Δrecall_pp  95% CI          matched/approx/unmatched (of 60)
    1   low       +4.65   [+2.71, +8.38]    30 /  9 / 21
    1 middle      -0.27   [-1.63, +1.70]    34 / 12 / 14
    1   high      -1.07   [-2.19, +0.23]    27 / 10 / 23
    2   low       +4.65   [+2.71, +8.38]    30 /  9 / 21
    2 middle      -0.27   [-1.63, +1.70]    34 / 12 / 14
    2   high      -1.07   [-2.19, +0.23]    27 / 10 / 23
    3   low       +5.06   [+2.43, +8.47]    28 / 10 / 22
    3 middle      +0.16   [-1.36, +2.39]    27 / 14 / 19
    3   high      -0.60   [-1.29, +0.63]    28 / 12 / 20
    4   low       +4.79   [+2.77, +7.54]    26 / 10 / 24
    4 middle      -0.36   [-1.31, +1.46]    29 / 11 / 20
    4   high      -0.79   [-1.66, +0.49]    32 /  6 / 22
```

**Honest reading.** At the **low** budget in-place is clearly ahead
(≈ +4.6 to +5.1 pp; CI excludes zero at every depth). At the **middle** and
**high** budgets the difference is not distinguishable from zero (CI spans zero)
and, if anything, tilts slightly to out-of-place. Of the 48 (split × depth ×
delta × budget) cells, 45 have a bootstrap CI excluding zero, split **24
in-place / 21 out-of-place** — i.e. **there is no uniform winner.** The pooled
"in-place wins everywhere" of Task 9 does **not** survive a within-stratum,
equal-work comparison on the denser grid.

### 3.5 The split-threshold interaction (the key qualitative finding)

Averaging the cell means by split threshold shows the sign of the effect is set
almost entirely by the split threshold, i.e. by the index's unit structure:

```text
split   low      middle    high       (Δrecall_pp, positive = in-place)
 48   +13.77    +6.93    +4.27        in-place higher at every budget
192    -4.20    -7.30    -6.03        out-of-place higher at every budget
```

* At `split=48` in-place is consistently ahead — the same direction, though
  smaller, as the Task-9 pooled result.
* At `split=192` out-of-place is consistently ahead. This is the opposite sign
  and it is **not** a matched-preset artifact here (the dense grid lets both
  arms land within a few percent of the same scan budget). It is the expected
  consequence of the structural asymmetry Task 9 §5.1 documented: out-of-place
  routes across two units and its per-unit posting budget interacts differently
  with unit fan-out, which is itself a function of the split threshold.

The interaction, not a main effect, is the robust result. Figure 2 makes it
visible directly: the panels are faceted by (split, delta) and the paired
difference flips sign between the `split=48` and `split=192` columns while the
per-depth lines stay close to their cell's mean.

### 3.6 Matched-row recall deltas remain descriptive

On the 352 matched rows the same-preset subset (34 rows) has mean Δrecall
**0.009 pp** (the arms are the same operating point), while the different-preset
subset (318 rows) averages **2.34 pp**. Consistent with Task 9 §5.2, matched-row
recall deltas are reported for completeness but are *not* used for the strategy
claim; the claim rests on the budget-based frontier of §3.4–§3.5.

---

## 4. How each requirement is met

* **Per-stratum frontiers.** `group_strata` keys on the full
  `STRATUM_FIELDS` tuple (dataset, dim, initial_size, workload, insert_prob,
  leaf_size, branching_factor, depth, split, merge, delta, memtable_limit,
  compaction_threshold, seed, checkpoint); each strategy's nondominated
  `(work, recall)` envelope is derived independently
  (`build_frontier_by_stratum`, `frontier_points`). Dominated points are removed
  from the *frontier only*; the raw table is untouched
  (`test_dominated_removed_from_frontier_only`, `test_analysis_does_not_mutate_raw`).
* **Common observed work interval.** `work_overlap` intersects the two arms'
  observed scan ranges; empty intersections are labelled
  `no_common_work_range` with no fabricated budget
  (`test_no_common_work_range_is_explicit`).
* **Data-derived budgets.** `budget_schedule` places low/middle/high at
  25/50/75 % of the common interval (`test_budget_schedule_positions_within_interval`).
* **Attainable-performance labelling.** Outputs are tagged
  `empirical_frontier / best_observed_at_budget`; the metadata records that this
  is attainable performance over the tested grid, not a frozen production preset
  and not a causal isolate of the maintenance algorithm.
* **Seed-level CIs.** `build_frontier_summary` resamples **seeds**, not
  checkpoints or rows (`test_summary_ci_resamples_seeds_not_rows`,
  `test_single_seed_ci_is_degenerate`); `resample_unit="seed/run"` is written to
  metadata.
* **Both arms on the same basis.** `validate_raw` checks that each stratum's two
  arms agree on `dataset_checksum`/`trace_hash`/`operations_processed`/
  `live_vector_count` (`test_validation_flags_basis_mismatch`,
  `test_validation_flags_unpaired_stratum`).
* **Measured work, not nominal presets.** The budget picker uses
  `vectors_scanned_per_query` (`test_picker_never_exceeds_budget`,
  `test_matching_uses_measured_work_not_nominal`), with a deterministic
  tie-break (higher recall → lower work → lower p50 → label)
  (`test_picker_prefers_higher_recall_then_lower_work`,
  `test_tie_break_is_deterministic`).

---

## 5. Claim scope and limitations

* **Empirical, in-sample envelope.** At budget `B` each strategy shows its best
  preset with measured work ≤ `B`, chosen *on the same measured checkpoint*.
  This is an attainable-performance bound over the tested grid, not a fixed
  deployment preset.
* **Not a causal maintenance effect.** The difference combines index layout,
  unit fan-out and routing; the frontier does not isolate the maintenance
  algorithm from those.
* **Discrete grid.** Recall is bounded by the tested beam × n_postings grid; a
  denser grid can only raise a frontier. The dense grid still leaves a median
  common-interval width ≈ 687 scans, so budgets are not exact scan equalities —
  both the selected scan counts and the relative gap are reported.
* **Few held-out seeds (3).** Intervals are descriptive bootstrap intervals over
  seeds (clustered; checkpoints within a seed are not independent), not precise
  inferential statements.
* **Synthetic, in-memory.** Clustered synthetic vectors; nothing about
  disk/NVMe behaviour.
* **No maintenance wall time.** The evaluation raw schema carries per-unit
  maintenance *counters* (`delta_*`, `maintenance_similarity_evaluations`) but no
  cumulative maintenance **wall-time** column, so Figure 3 is skipped rather
  than fabricated; maintenance *timing* is the Task-8 deliverable
  (`scripts/compare_maintenance.py`, `R1_T8_Report.md`).
* **Diagnostic tables excluded.** `results_budget_matched.csv` (Task-9
  selected pairing) and `results_budget_oracle.csv` (per-checkpoint oracle) are
  never mixed into headline numbers
  (`test_oracle_and_matched_tables_excluded_from_headline`).

---

## 6. Figures

Generated by `scripts/plot_search_budget.py` from the dense run (see
`figures/research_figures_report.md` for the full data table and captions):

* **Figure 1** — Recall@10 vs measured work, one nondominated mean frontier per
  (depth, delta, split, strategy); small multiples facet depth (rows) and delta
  (columns); split is line style, strategy is colour; faint markers are
  per-seed frontiers.
* **Figure 2** — Paired Recall@10 difference (pp, >0 favours in-place) at a
  common work budget, faceted by (split, delta), a line per depth; this is the
  figure that exposes the §3.5 sign flip.
* **Figure 3** — *not produced* (no maintenance wall-time column; see §5).

---

## 7. Density contrast (coarse vs dense grid)

| | coarse (20 presets) | dense (56 presets) |
|---|---|---|
| raw rows | 9,600 | 26,880 |
| matched / approx / unmatched (of 720) | 211 / 64 / 445 | 352 / 125 / 243 |
| matched fraction | 29.3 % | 48.9 % |
| mean scan gap (matched) | ~15.7 % (median) | 1.54 % |
| depth-macro low Δrecall | +8.9 pp | +4.6 pp |
| depth-macro high Δrecall | +1.8 pp | −1.0 pp |

The coarse grid's large, uniformly in-place-positive effect was substantially a
**discretisation artifact**: with few presets the two arms could not be placed at
comparable scan budgets, so the comparison degenerated toward pairing a
high-recall in-place preset against a low-recall out-of-place one. Once the
grid is dense enough to compare at equal work, the effect shrinks to a
low-budget in-place edge plus a strong split-threshold interaction (§3.5).

---

## 8. Test results

```text
$ .venv/bin/python -m pytest tests/test_frontier.py -q
18 passed

$ .venv/bin/python -m pytest tests/test_plot_search_budget.py -q
19 passed

$ .venv/bin/python -m pytest -q
310 passed
```

`tests/test_frontier.py` (18) covers separate strata grouping, budget picker
bounds/ties, explicit `no_common_work_range`, measured-work (not nominal)
matching, dominated-point removal on the frontier only, deterministic
tie-breaks, basis-mismatch and unpaired-stratum validation, unmatched counts in
summaries, seed-level (not row-level) CIs, the budget schedule and a
file-writing end-to-end run. `tests/test_plot_search_budget.py` (19) covers
Pareto dominance, final-checkpoint-only headline data, best-recall-under-budget
(ties + no eligible preset), seed/config separation in pairwise grouping,
visible ineligible rows, exclusion of the diagnostic matched/oracle tables,
seed-level bootstrap, the skipped maintenance figure, and the CLI writing the
report + figures plus failing on a malformed schema.

---

## 9. Files

* `vecstudy/harness/frontier.py` — `group_strata`, `frontier_points`,
  `work_overlap`, `budget_schedule`, `pick_at_budget`, `build_frontier_by_stratum`,
  `build_work_overlap`, `build_frontier_paired`, `build_frontier_summary`,
  `validate_raw`, `run_analysis`, CLI (`build_parser`, `main`).
* `scripts/compare_frontier.py` — thin entry point (BLAS pinning, import path).
* `tests/test_frontier.py` — 18 tests.
* `vecstudy/harness/plot_search_budget.py` — figures/report/`analysis_summary.csv`
  generator (matplotlib, Agg); `scripts/plot_search_budget.py` — thin entry
  point; `tests/test_plot_search_budget.py` — 19 tests.
* Runtime outputs (per invocation): `results_budget_frontier_by_stratum.csv`,
  `results_budget_work_overlap.csv`, `results_budget_frontier_paired.csv`,
  `results_budget_frontier_summary.csv`, `run_metadata.json`, plus `.parquet`
  copies; figures under `<eval-run>/figures/`.
* Related: `R1_T9_Report.md` (observed-work harness and pooled frontier),
  `R1_T8_Report.md` (paired maintenance benchmark), `DESIGN.md`.
