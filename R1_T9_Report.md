# R1 In-Place Maintenance — Task 9: Equal-Search-Work Comparison Harness

Deliverable: `scripts/compare_search_budget.py` (thin entry point) backed by
`vecstudy/harness/search_budget.py` (all logic, unit-testable) plus
`tests/test_search_budget.py`.

The harness replaces the nominal-configuration comparison of Task 8 with an
**observed-work** comparison. For each base configuration it replays the churn
workload **once per strategy** with the production maintainers (no algorithm is
reimplemented or modified), clones the engine state at every checkpoint, and then
evaluates the **whole search-preset grid** (`beam × n_postings`) on those
checkpoints. Search work is measured as `vectors_scanned_per_query`, and
in-place / out-of-place presets are paired by that measured work.

Two phases with a strict separation of data:

* **calibration** — derives, per depth, the preset pair that minimises the median
  relative scan gap on calibration seeds only;
* **evaluate** — applies those **fixed**, previously-selected pairs to held-out
  seeds and reports the residual scan gap, recall difference and
  correctness/atomicity gate.

---

## 1. Launch commands

From the repository root (single-thread BLAS for reproducibility):

```bash
# Phase 1 - calibration (seeds 0,1,2): 3 depths x 2 splits x 2 deltas = 12 cells
.venv/bin/python scripts/compare_search_budget.py --phase calibration \
    --initial-size 400 --operations 400 --query-count 32 --checkpoint-interval 50 \
    --depths 1,2,3 --split-thresholds 24,48 --deltas none,0.3 \
    --calibration-seeds 0,1,2 \
    --beam-width-values 2,4,8,16,32 --n-postings-values 1,2,4,8 \
    --blas-threads 1 --out results/search_budget/calibration

# Phase 2 - held-out evaluation (seeds 3,4,5): larger workload, fixed pairs
.venv/bin/python scripts/compare_search_budget.py --phase evaluate \
    --calibration-results results/search_budget/calibration \
    --initial-size 2000 --operations 1000 --query-count 64 --checkpoint-interval 250 \
    --depths 1,2,3,4 --split-thresholds 48,192 --deltas none,0.3 \
    --evaluation-seeds 3,4,5 \
    --beam-width-values 2,4,8,16,32 --n-postings-values 1,2,4,8 \
    --blas-threads 1 --out results/search_budget/evaluation
```

`--phase evaluate` requires `--calibration-results`; if it points at a parent
directory the newest run sub-directory containing `results_budget_selection.csv`
is used, so the exact calibration output path need not be copied. Every flag:
`--phase {calibration,evaluate}`, `--dataset`, `--dim`, `--initial-size`,
`--operations`, `--workload`, `--insert-prob`, `--leaf-size`,
`--branching-factor`, `--depths`, `--split-thresholds`, `--merge-thresholds`,
`--deltas`, `--calibration-seeds`, `--evaluation-seeds`, `--beam-width-values`,
`--n-postings-values`, `--k`, `--recall-ks`, `--query-count`,
`--checkpoint-interval`, `--memtable-limit`, `--compaction-threshold`,
`--match-within` (default 5.0), `--match-approx` (default 10.0),
`--calibration-results`, `--out`, `--overwrite`, `--blas-threads`. Without
`--overwrite` each invocation writes into a unique timestamped sub-directory of
`--out`.

---

## 2. Sample real output

### 2.1 Files produced

```text
Calibration: results/search_budget/calibration/sb-20261019-011157-4226
Files: results_budget_raw.csv (12960 rows), results_budget_matched.csv (324 rows),
       results_budget_summary.csv (18 rows), results_budget_pareto.csv (160 rows),
       results_budget_selection.csv (4 rows), results_budget_oracle.csv (0 rows),
       results_budget_raw.parquet (12960 rows), results_budget_matched.parquet (324 rows)

Evaluation: results/search_budget/evaluation/sb-20261019-015145-9486
Files: results_budget_raw.csv (9600 rows), results_budget_matched.csv (240 rows),
       results_budget_summary.csv (30 rows), results_budget_pareto.csv (200 rows),
       results_budget_selection.csv (4 rows), results_budget_oracle.csv (240 rows),
       results_budget_raw.parquet (9600 rows), results_budget_matched.parquet (240 rows)
```

`results_budget_raw.csv` holds every (checkpoint × preset × strategy) row;
`results_budget_matched.csv` holds one row per checkpoint with the two presets
selected and the resulting gap/recall/status; `results_budget_summary.csv` groups
matched rows by (depth, split, delta, work band). Parquet copies are written when
pyarrow is importable. `run_metadata.json` records `immutability_violations`,
`matching_rule`, `search_semantics`, the resolved grid/seeds and the output
schema.

### 2.2 Calibration selection (fixed pairs; calibration seeds only)

```text
      bucket             kind  ip_preset oop_preset  medGap%     p25     p75   f<=5%   f<=10%   ip_q50  oop_q50  ngrp
           1            depth      b16p8      b16p8    39.38   19.50   97.44    0.11     0.11 0.090289 0.139205   108
           2            depth      b32p8       b4p8    39.38   19.50   97.44    0.11     0.11 0.093750 0.144992   108
           3            depth      b16p8       b2p8    19.23   10.95   79.14    0.11     0.24 0.095037 0.119644   108
  __global__  global_fallback      b16p8       b2p8    33.68   15.57   92.03    0.11     0.15 0.093049 0.135202   324
```

Selection is **per depth** with a `__global__` fallback; the objective is the
median relative scan gap over calibration checkpoints, tie-broken by mean query
p50 (both arms), then combined nominal budget, then label. Depth 4 (unseen at
calibration) falls back to `__global__`; the fallback pair is visibly
indistinguishable from the depth-3 pair.

### 2.3 Held-out evaluation — matched rows (fixed pairs applied)

`relative_scan_gap_pct = 100·|ip−oop| / ((ip+oop)/2)`; `recall_diff_pp` is in
recall points, positive favouring in-place. Checkpoint-0 rows prove the two arms
share one initial state (equal scan at equal preset).

```text
depth split delta seed  cp ip_preset oop_preset  ip_scan/q  oop_scan/q  gap%   status  ip_rec@10 oop_rec@10  dRec_pp
    1   192   0.3    3   0     b16p8      b16p8    512.59      512.59   0.00  matched   0.9484    0.9484     0.00
    1    48   n/a    3   0     b16p8      b16p8    336.70      336.70   0.00  matched   0.9297    0.9297     0.00
    2   192   0.3    3   0     b32p8       b4p8    512.59      512.59   0.00  matched   0.9484    0.9484     0.00
    3   192   0.3    3   0     b16p8       b2p8    512.59      497.80   2.93  matched   0.9484    0.7031    24.53
    3   192   n/a    3   0     b16p8       b2p8    336.70      324.53   3.68  matched   0.9297    0.6547    27.50
```

Held-out status counts: `matched=41`, `approximate=12`, `unmatched=187` (240
checkpoint rows). The matched subset is small and its recall deltas are
**not** a strategy effect — see §5.2: at depth ≥ 3 the matcher pairs *different*
presets, and at checkpoint 0 the arms are the identical index, so the 24–27 pp
gaps there are an operating-point artifact of pairing `b16p8` against `b2p8`.

### 2.4 Held-out frontier — best recall at a measured work budget

The honest strategy statement comes from the pooled recall-vs-work frontier
(each strategy's best preset at or below a scan budget, all held-out rows):

```text
budget | depth | ip_best | oop_best | dRec_pp | ip_preset | oop_preset
    64 |     1 |   0.506 |    0.436 |    7.03 |      b2p1 |       b2p1
   128 |     1 |   0.664 |    0.644 |    2.03 |      b2p2 |       b2p2
   256 |     1 |   0.837 |    0.806 |    3.12 |      b2p4 |       b2p4
   512 |     1 |   0.948 |    0.934 |    1.41 |      b2p8 |       b2p8
    64 |     3 |   0.592 |    0.436 |   15.63 |      b8p2 |       b8p1
   128 |     3 |   0.687 |    0.644 |    4.37 |      b8p2 |       b8p2
   512 |     3 |   0.948 |    0.934 |    1.41 |      b8p8 |       b8p8
    64 |     4 |   0.509 |    0.436 |    7.34 |      b8p1 |       b8p1
   512 |     4 |   0.939 |    0.934 |    0.47 |     b16p8 |      b16p8
```

At **every** budget the in-place frontier is ≥ the out-of-place frontier
(+0.5 to +15.6 pp), which is the expected consequence of the structural
asymmetry in §5.1 (out-of-place routes across multiple units).

### 2.5 Oracle diagnostic (descriptive only)

`results_budget_oracle.csv` (labelled `diagnostic="oracle_per_checkpoint"`) picks
the best pair **per checkpoint** purely as an upper bound on what the knob grid
can match. Even oracle-chosen pairs leave a mean recall difference of ≈5.6 pp
across the 166/240 checkpoints that can be matched within 5 %; oracle output is
never used for calibration and is reported with zeros excluded from any
strategy claim.

---

## 3. How each requirement is met

### 3.1 One churn replay per (base config, strategy), whole grid on clones
* `run_base_config` replays the churn trace **once** per strategy, capturing a
  deep clone of the searchable units at every checkpoint
  (`test_same_checkpoint_state_for_all_presets` asserts every preset sees the
  same snapshot). The preset grid is then evaluated on the frozen clones, so
  search tuning cannot perturb maintenance.
* `QueryConfig` is derived from the preset and is fully separate from
  `IndexConfig` / `MaintainConfig`; a runtime assertion (`test_preset_structural_
  immutability`) confirms that evaluating a preset advances **no** structural or
  maintenance counter (`_snapshot_advanced`).

### 3.2 Beam translation and the depth-1 fast path
* A scalar beam `B` maps to `beam_widths=(B,)`; `hierarchy.descend` indexes
  routing levels and clamps to the last element. At `depth == 1` the flat fast
  path is taken, so **beam width is unused and only `n_postings` changes work**
  (`test_beam_translation`, `test_depth1_beam_irrelevant`). This prediction is
  visible in §2.2/§2.4: depth-1 rows are constant across beam.

### 3.3 Work measurement and matching
* Work is the observed `vectors_scanned_per_query` summed over a query pass with
  a **separate** `Counters` object (query work never mixes with maintenance).
* `relative_scan_gap` and `match_status` implement the 5 % / 10 % rule exactly
  (`test_relative_scan_gap_formula`, `test_match_thresholds`); selection
  minimises the median gap over calibration groups only
  (`test_selection_uses_calibration_rows_only`).

### 3.4 Calibration / evaluation separation (no leakage)
* Calibration and evaluation seeds are disjoint by default and asserted disjoint
  in tests (`test_seed_separation`, `test_custom_seed_split`). The selection table
  is a pure function of the calibration matched rows and is written once; the
  evaluation reads it back and applies it unchanged.
* `--phase evaluate` refuses to run without `--calibration-results`
  (`test_evaluate_requires_calibration`).

### 3.5 Preset clone independence and immutability of the engine
* Clones are produced with `segment.clone` / unit cloning; the immutability check
  over all base-config runs is aggregated into
  `run_metadata["immutability_violations"]` and is **0** for both phases here.
  The check compares structural counters before/after the preset sweep
  (`_snapshot_advanced`), fixing an earlier false positive from a truthy-dict
  comparison.

### 3.6 Correctness / atomicity gate
* Both arms receive the same audit stack: `audit_maintainer` (coverage,
  ownership, NPA) plus a fixed-Δ membership audit. Out-of-place is now audited by
  `churn.audit_outofplace_membership`, so unlike Task 8 both arms get a real
  membership audit (`audit_asymmetry` recorded in metadata). Failed runs are
  labelled `run_failed` and surfaced, never dropped
  (`test_missing_pair_marking`, `test_unmatched_stays_visible`).

---

## 4. Test results

```text
$ .venv/bin/python -m pytest tests/test_search_budget.py -q
20 passed

$ .venv/bin/python -m pytest -q
273 passed
```

`tests/test_search_budget.py` (20 tests) covers: scalar-beam translation, the
depth-1 beam-is-irrelevant fast path, the scan-gap formula and match thresholds,
identical checkpoint state across presets, preset structural immutability,
live-only ground truth, calibration/evaluation seed separation, evaluate-
requires-calibration, custom seed split, selection-as-a-function-of-calibration-
only, unmatched rows staying visible, missing-pair marking, summary band counts,
Pareto dominance, the default 20-preset grid, argument validation, and a CLI
smoke run of calibration → evaluate.

---

## 5. Findings and limitations

### 5.1 Matching is structurally coarse (a result, not a bug)
With the (beam, n_postings) grid the two strategies are only **coarsely**
matchable: on calibration, only ~11 % of checkpoint groups fall within 5 %
median gap, and the chosen pairs sit at 19–39 % median gap. The cause is visible
in the raw scans: out-of-place searches **two** units at this size (one sealed
segment plus the flat MemTable unit), which imposes a work floor (≈81 scans/q at
`p1`) and a unit-count granularity that a single in-place unit cannot cancel.
This is exactly the per-unit posting-budget limitation Task 8 documented, now
quantified by observed work.

### 5.2 A matched scan gap is not sufficient to attribute recall
Because search work and recall are both functions of the preset, pairing
different presets at equal scan conflates the **operating point** with the
strategy. Evidence: at checkpoint 0 the two arms are the *identical* index, yet
the depth-3 matcher pairs `b16p8` (ip) with `b2p8` (oop) and reports a 24–27 pp
recall gap — purely a preset artifact. Of the 41 matched held-out rows, the 12
that use the **same** preset both arms have a 0.00 pp difference; the 29 that use
different presets average 17.1 pp, but only the frontier comparison (§2.4) is
attributable to the strategy. The report therefore treats matched-row recall
deltas as descriptive and bases the strategy claim on the frontier, which shows
in-place ≥ out-of-place at every budget.

### 5.3 Deployment deviations (recorded; results are deterministic)
* The held-out workload was reduced for runtime: `--operations 1000` (5
  checkpoints) instead of 2000, `--split-thresholds 48,192` and `--deltas
  none,0.3` (dropping split 96 and delta 0.15, whose in-place split/merge churn
  costs 50–60 s per cell), and `--evaluation-seeds 3,4,5` (3 held-out seeds)
  instead of 3–9. Calibration is unaffected. All runs were deterministic and
  `run_errors` is empty.
* Runtime is dominated by the single churn replay, **not** the grid: a 20-preset
  sweep adds ~2.7 s to a ~29 s in-place cell, confirming the design's
  efficiency premise (presets are nearly free next to the maintenance replay).

### 5.4 Measurement limitations (recorded in `run_metadata.json`)
1. **Per-unit posting budget** (shared with Task 8): out-of-place descends each
   unit with `n_postings` independently, so actual work is measured rather than
   assumed equal.
2. **Fixed-Δ audit asymmetry closed**: out-of-place membership is audited via
   `audit_outofplace_membership`; both arms now get a real audit.
3. **`psutil` / `matplotlib` are not installed here**; RSS falls back to macOS
   `ru_maxrss` (peak) and no plots are generated (CSV/Parquet only).
4. **Pooling caveat**: the frontier in §2.4 pools heterogeneous configurations;
   it is a *best-achievable* view, not a per-configuration paired test.
5. **Latency** is wall-clock and outside the algorithmic study; reported with a
   warm-up pass and pinned BLAS threads.
6. **Oracle rows are descriptive** and are excluded from calibration/selection.

---

## 6. Files

* `vecstudy/harness/search_budget.py` — preset grid (`SearchPreset`,
  `BudgetSpec`, `build_presets`), checkpoint capture/clone, per-preset
  evaluation, `run_base_config` (immutability check), `relative_scan_gap`,
  `match_status`, `select_pairs`, selection I/O, `build_matched`, `build_oracle`,
  `build_summary`, `build_pareto`, CLI (`build_parser`, `resolve_spec`, `main`,
  `run_experiment`).
* `scripts/compare_search_budget.py` — thin entry point (BLAS pinning, import
  path, forwards to `search_budget.main`).
* `tests/test_search_budget.py` — 20 tests.
* Runtime outputs (per invocation): `results_budget_raw.csv`,
  `results_budget_matched.csv`, `results_budget_summary.csv`,
  `results_budget_pareto.csv`, `results_budget_selection.csv`,
  `results_budget_oracle.csv`, `run_metadata.json`, plus `.parquet` copies.
