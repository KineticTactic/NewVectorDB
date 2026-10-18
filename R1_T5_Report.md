# R1 In-Place Maintenance — Task 5: Integration and Structural-Stability Audit

Audit-only task. The integration was reproduced, independently validated, and the
high structural-event count was traced to a root cause. Two documentation errors
in `R1_T4_Report.md` were corrected (they are the only "fix" — no production
behaviour was changed, per the task constraints). One real numerical limitation
was found in the *auditor*, not the maintainer.

---

## 1. Reproduction

Commands and exact results (working tree clean at the merge commit `ddb92e7`):

```
.venv/bin/python -m pytest tests/test_inplace_npa.py -q    ->  16 passed in 1.28s
.venv/bin/python -m pytest tests/test_membership.py  -q    ->  28 passed in 0.82s
.venv/bin/python -m pytest tests/test_inplace_merge.py -q  ->  33 passed in 0.88s
.venv/bin/python -m pytest -q                              -> 140 passed in 3.73s
```

All four match the merge agent's report (140 passed). The reviewed diff is the
Task 4 commit itself:

```
$ git show --stat ddb92e7
 tests/test_inplace_merge.py   |  33 tests
 vecstudy/core/config.py       |  +6   (MaintainConfig.merge_threshold)
 vecstudy/core/metrics.py      | +21   (5 merge counters + wiring)
 vecstudy/maintain/inplace.py  | +386  (merge path + _split_oversize fix)
 tests/test_inplace_npa.py     |  ±6
 tests/test_membership.py      | ±19
```

The NPA / membership / merge reports were read but treated as secondary to the
implementation and the test results, as instructed.

---

## 2. Structural-event investigation

### 2.1 The reported workload, exactly

The merge report's §4 prose described the workload as *"1 200 vectors, `initial_n=600`,
800 operations"*. That is wrong; it reproduces only as:

> 600 2-D clustered unit vectors (10 clusters, angular spread 0.05),
> `branching_factor=8`, **`leaf_size = split_threshold`**, `initial_n=300`,
> **400 operations**, `insert_prob=0.35`, trace seed 7.

All eight rows of the report's cost table reproduce **bit-for-bit** under these
parameters (`R1_T4_Report.md` §4 has been corrected; `K` is the **final** leaf
count — initial `K` is 15 at `split_threshold=20` and 8 at `split_threshold=40`).

### 2.2 Accounting per operation

Instrumented `_split` / `_merge_pair` / `_prune_empty_leaves`, counting leaf-count
changes for every API operation:

| workload | thr | K₀ | K_final | inserts | deletes | no-op del | splits (ins / del) | merges | prune-removals | identity K₀+split−merge−prune | max events / op | ops with merge **and** split |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| mixed | 20 | 15 | 10 | 121 | 279 | 0 | 679 (14 / 665) | 611 | 73 | 10 ✓ | 19 | **175 / 400** |
| mixed | 40 | 8 | 6 | 121 | 279 | 0 | 49 (4 / 45) | 51 | 0 | 6 ✓ | 5 | 28 / 400 |
| delete-only | 20 | 15 | 1 | 50 | 350 | 0 | 404 (0 / 404) | 415 | 3 | 1 ✓ | 15 | 140 / 400 |
| delete-only | 40 | 8 | 1 | 50 | 350 | 0 | 190 (0 / 190) | 191 | 6 | 1 ✓ | 6 | 138 / 400 |

The trace deletes only live ids (no-op deletes = 0); `delete-only` still contains
50 inserts because the generator re-inserts from the reserve once the live set
empties, then deletes again.

The accounting closes exactly: `K₀ + splits − merges − empty-leaf prunes = K_final`.
A split adds one leaf, a merge (via `_remove_leaf`) removes one, a prune removes
one empty leaf; every event is a real structural change and none is double
counted. **The counters are not inflated.**

### 2.3 `_merge_after_delete` / `_split_oversize` / `_prune_empty_leaves` undo each other

Within **one** delete the events are not interleaved, but the merge block is
systematically undone by the split block. Representative catastrophic op
(`delete vid=261`, `split_threshold=20`), leaf sizes sampled after each event:

```
before delete  K=21  sizes [20,18,18,18,16,16,16,15,15,15,14,14,13,12,11,11,11,10,9,6,5]
 merge(2,4)    K=20  [23,21,20,...]          merge(0,6)  K=15  [33,32,31,...]
 merge(2,10)   K=19  [33,27,...]            merge(0,2)  K=14  [...]
 merge(0,2)    K=18                          merge(8,12) K=13
 merge(14,16)  K=17                          merge(0,8)  K=12  [33,32,31,29,27,25,24,23,17,15,14,12]
 split 1..19   K=12 -> 21  (9 splits)        prune       K=21 -> 20
after  delete  K=20  sizes [23,20,19,18,18,16,15,14,14,14,14,13,13,12,12,11,11,10,10,5]
```

Nine merges (K 21→12) are followed by nine splits (K 12→21) and one prune, leaving
the leaf layout almost identical to the start. Net: one leaf fewer. 175 of the 400
mixed deletes contain both a merge and a split for exactly this reason.

### 2.4 Root cause (evidence-based)

**Every chosen merge pair joins fragments of *different* true clusters.** For the
`delete vid=261` state, the dominant true cluster of each merged pair:

```
merge(2,4) domA=4 domB=9   merge(14,16) domA=1 domB=5   merge(8,12) domA=8 domB=7
merge(2,10) domA=9 domB=6  merge(6,14) domA=2 domB=1    merge(0,8) domA=3 domB=-
```

Cross-cluster pairing is forced by the eligibility cap. Brute-force over that
state:

```
cheapest OVERALL Ward pairs (all same-cluster, cost ~0.006–0.03):   comb. owned 33,29,34,25,26...  -> INELIGIBLE (> split_threshold)
cheapest ELIGIBLE pairs: (14,18) cost 0.411 (clusters 8 / 7)       <- the only options are cross-cluster
```

An underfilled posting (`< merge_threshold = 10`) whose same-cluster partner sits
at/above the split cap makes `live_owned(a)+live_owned(b) > split_threshold`, so
the *only* eligible partners are far-away fragments. Merging two fragments from
different clusters produces a compromise centroid; reassignment then hands the
union's vectors to whichever neighbour is closer. Measured per merge (union size →
survivor-owned after reassignment):

```
(9+11=20 -> 5)  (5+15=20 -> 0)  (9+6=15 -> 1)  (11+5=16 -> 0)  ...
```

Those neighbours cross `split_threshold`, so `_split_oversize` splits them —
generally recreating the pre-merge structure. That is the full mechanism: **the
hysteresis cap that is meant to stop re-splits is what forces the cross-cluster
merges that cause them.**

### 2.5 Correctness verdict

No correctness defect: both independent auditors pass on every operation of every
workload, ownership matches the global argmax, membership matches the Δ oracle,
and the leaf-count identity closes. This is a **structural-stability / quality
limitation** of the documented merge policy at small `split_threshold`, not a
violation of ownership/NPA/membership. Per constraint 6 it is documented, not
"fixed" by changing the merge policy (which the task forbids).

---

## 3. Integration matrix

A committed, fixed-seed matrix: `tests/test_inplace_integration.py` (23 tests).
Every operation is checked with **both** auditors plus structural and query
invariants. Axes:

* **depths 1, 2, 3, 4**
* **Δ disabled** (`None`) **and two on values** (0.3, plus 0.3 across the wider
  exploratory matrix); the committed set also exercises `None` on depths 1/3/4
* **low / high split threshold** (24 / 48 at `leaf_size=16`)
* **insert-only, delete-only, mixed**

```
.venv/bin/python -m pytest tests/test_inplace_integration.py -q   -> 23 passed
```

A wider exploratory run (72 configs: depths × {None,0.15,0.35} × {24,48} ×
{insert_only,delete_only,mixed}) also passed both auditors on **every** operation.

Per-operation checks (`_check_*` in the test):

1. `audit_maintainer` — coverage, unique canonical ownership, NPA.
2. `audit_membership` — fixed-Δ replica oracle (independent ratio-form derivation).
3. posting ids positional and equal to the leaf-centroid row;
4. routing levels have valid, complete, disjoint parent→child references;
5. stored spread radii equal independently recomputed radii;
6. exhaustive query returns exactly the brute-force live top-k and never a
   tombstoned id.

No assertion weakens the oracles and the maintainer shares no oracle code.

---

## 4. Query path and invariants

After repeated splits and merges the test asserts, per operation:

* **ids** contiguous, `postings[i].id == i`, `postings[i].centroid == leaf row i`;
* **hierarchy** valid at the configured depth (every child index in range; each
  level's children partition the level below);
* **ownership** exactly one canonical owner per live vector;
* **ownership == global inner-product argmax**, ties → lowest posting id (auditor
  + `test_inplace_merge.py::test_exact_tie_follows_lowest_posting_id`);
* **membership == oracle**;
* **tombstones** absent from live results (query never returns a deleted id;
  `test_inplace_stress.py` additionally asserts `members ∩ tombstones = ∅`);
* **spreads** match the independent radius;
* **query** exact after repeated structural churn;
* **empty index** and **single posting** are query-safe (dedicated tests).

---

## 5. Honest performance split

Query work and maintenance work are counted separately (query buckets reset by
`Counters.reset_search()`; maintenance counters accumulate). Measured on the
§2 workload, one maintenance pass of 400 ops then 100 exhaustive queries:

| thr / leaf | maintenance wall | splits | merges | merge reassign | merge sim evals | hierarchy rebuilds | 100 queries | routing dist | scan dist |
|---|---|---|---|---|---|---|---|---|---|
| 20 / 20 | ~1183 ms | 679 | 611 | 5 684 | 163 417 | 611 | ~4.6 ms | ~10 / q | ~142 / q |
| 40 / 40 | ~115 ms | 49 | 51 | 952 | 11 007 | 51 | ~4.2 ms | ~7 / q | ~142 / q |

The 10× maintenance cost at `split_threshold=20` is the direct price of ~13× more
structural events (and their repair similarity evaluations); **query cost is
unchanged** (~4 ms per 100 exhaustive 2-D queries, independent of the churn
count). No performance-improvement claim is made.

---

## 6. Fixes made

1. **`R1_T4_Report.md` corrected** (documentation defects, reproducible):
   * §1.2 — the size cap bounds the *survivor*, not the whole pass; a merge can
     still inflate a neighbour that `_split_oversize` then splits. Removed the
     false "so a merge cannot immediate re-split".
   * §4 — corrected the workload description (600 vectors / `initial_n=300` /
     400 ops / `leaf_size = split_threshold`) and defined **K = final leaf count**.
   * §4 observations — merge≈split is mutual undo, not a healthy steady state.
2. **`tests/test_inplace_integration.py` added** — the reproducible matrix above.

**No production code was changed.** The merge policy is implemented as documented
(candidate rule, min-Ward partner, lower-id survivor, live-mean centroid); the
high event count is a consequence of that policy, and the task forbids
redesigning it. Changing the cap, or rejecting merges that inflate a neighbour,
would be a policy change and is out of scope.

---

## 7. Remaining limitations

* **Merge/split mutual undo (§2).** At small `split_threshold` (≈ `leaf_size`), a
  delete can trigger O(K) merges and O(K) re-splits that nearly cancel. Correct,
  but structural work is high. A future policy change (e.g. merge only
  geometrically-adjacent fragments, or reject a merge that oversizes a
  neighbour) would remove it; not done here.
* **Auditor Δ-boundary brittleness (found, not a maintainer defect).** With very
  tight clusters (`d = 1 − sim ≈ 1e-4`), `1 − sim` suffers float32 cancellation and
  the boundary statistic `d2/d1 − 1` carries ~2e-3 relative noise. At `Δ = 0.15` an
  exploratory config produced an `audit_membership` `extra`; exact float64
  arithmetic was **eligible** (ratio−1 = 0.14942 ≤ 0.15) and the maintainer's
  batched matmul agreed (0.14955), while the auditor's per-vector matvec
  disagreed (0.15035). The maintainer is correct; the zero-tolerance oracle is the
  outlier. The committed matrix uses a less degenerate spread (and the standard
  `make_synthetic`), where the oracle is numerically reliable. This is a
  test-harness precision limit worth a documented tolerance.
* **Depth-1 grows a routing root.** At `depth=1`, the first insert-split inserts
  a single-node root level (`_promote_split`), so `len(levels)` becomes 2; a later
  merge `_rebuild_hierarchy` restores the configured depth. Hierarchy references
  stay valid and audits pass; noted for completeness.
* **`split_threshold` is a soft cap after merges** (carried over from Task 4): a
  posting can end a pass slightly over threshold when `_split_oversize`'s
  single-attempt guard cannot reduce it. Not an ownership/NPA/membership issue.
