# R1 Task 3 — Finalize NPA correctness and validate invariants (in-place arm)

**Scope:** the split candidate filter in `vecstudy/maintain/inplace.py`; the NPA
regression suite `tests/test_inplace_npa.py`; a new deterministic stress suite
`tests/test_inplace_stress.py`. The out-of-place maintainer is untouched.

**Not in scope, and not implemented:** posting merges on underflow, centroid
refresh, dynamic-Δ, budget→Δ inversion, split-policy changes, SPANN RNG
filtering, query-engine changes, out-of-place maintenance.

**Status:** complete. `107 passed` (was 94: +6 tie/pruning regression tests,
+7 stress cases). On every measured workload the maintenance decision is
unchanged; the only behavioural difference is that a bound a few float32 ULP
*below* zero no longer prunes a candidate — a candidate that was already
admitted at exactly zero.

---

## 1. The change

### 1.1 A round-off gap in `_ball_may_switch`

`_ball_may_switch` answers "can some `x` with `‖x − ref‖ ≤ radius` satisfy
`x·target ≥ x·competitor`?" by maximising the inner-product difference over the
ball and comparing the maximum against `0`:

```python
# before
sphere_term = radius * math.sqrt(target_minus_competitor_sq)
return (ref_dot_target - ref_dot_competitor + sphere_term) >= 0.0
```

The inputs are float32 similarities, a float32 radius and a float32 squared
distance. Near zero the assembled bound is only trustworthy up to a small slack,
so a computed value a few ULP below zero is **not** a proof that the true maximum
is negative. The predicate is now tolerant:

```python
# after
sphere_term = radius * math.sqrt(target_minus_competitor_sq)
bound = ref_dot_target - ref_dot_competitor + sphere_term
slack = _BOUND_REL_TOL * (abs(ref_dot_target) + abs(ref_dot_competitor) + abs(sphere_term))
return bound >= -max(slack, _BOUND_ABS_TOL)   # _BOUND_REL_TOL = 1e-5, _BOUND_ABS_TOL = 1e-9
```

`1e-5` is not arbitrary: it is the acceptance tolerance the NPA auditor already
uses (`vecstudy/harness/churn.py:284`, a violation is only reported when
`best_sim − own_sim > tolerance`). The filter and the auditor now agree on what
"a tie" means.

### 1.2 The unsafe direction is *prune*

Pruning is the only irreversible step in the filter: a candidate that is not
admitted is never re-examined by the split. Over-inclusion is safe because the
filter is only a candidate screen — every admitted posting is subsequently
checked per-vector, and canonical ownership is then the exact `np.argmax` over
the leaf centroids. Admitting one extra posting costs work; pruning one
reachable posting loses correctness. The comparison is therefore made
*inclusive* and slack-tolerant, never the reverse.

The same reasoning is why the pre-existing exact `>= 0.0` was already the right
shape: a reachable **tie** is a reachable ownership change, because the tie-break
gives the vector to the lower posting id.

### 1.3 What was already correct

* An exact tie (`bound == 0.0`) is admitted —
  `test_pruning_bound_includes_exact_ties`.
* The tie-break is `np.argmax` over the leaf centroids, i.e. the lowest posting
  id — `test_exact_tie_resolves_to_lowest_posting_id`, and the split reuses
  `pid` for the lower centroid so the ordering is preserved.
* The per-vector screen and `np.argmax` still produce the production owner even
  when the filter over-includes, so widening the screen cannot change ownership.

---

## 2. Was there a real bug?

Three independent lines of evidence, all in this task:

1. **Random differential search.** Synthetic workloads (seeds 3/5/11/42, dims
   4/8/32, Δ ∈ {None, 0.5}, leaf sizes 20–80, 2 000–5 000 vectors) were driven
   through `insert`/`delete` and compared against a strict-`argmax` recomputation
   after every split. **No strict-argmax mismatch** was found. Scripts:
   `/tmp/explore.py`, `/tmp/probe_bound.py`, `/tmp/probe_bound2.py`.
2. **Analysis.** With real centroids (means of unit vectors, so `‖c‖ ≤ 1`) the
   exact bound for an *owned* vector can reach `0` only when the ball radius is
   `0`, i.e. the owned vector equals its centroid. Coincident-centroid cases are
   handled separately. The realistic hazard is therefore not an exact zero but a
   **computed** bound that lands within round-off of zero in near-degenerate
   geometry.
3. **Sensitivity check.** The exact predicate returns `False` for the
   `radius = 0.5 − 1e-9` and `0.5 − 1e-12` cases that the new test admits
   (`/tmp/check_old.py`), so `test_bound_roundoff_near_tie_is_not_pruned` fails
   against the old code — the test is not vacuous.

**Conclusion:** this was a *latent round-off hazard*, not a reproducible failing
state. The hardening closes it without enlarging the candidate set on any
workload where the bound is not within round-off of zero.

---

## 3. Regression tests added (`tests/test_inplace_npa.py`)

| test | what it pins |
|---|---|
| `test_bound_prunes_strictly_worse_candidate` | a bound clearly below zero still prunes (the fix must not stop pruning) |
| `test_tie_with_lower_id_target_moves_owner` | a reachable tie to a *lower-id* centroid moves ownership |
| `test_tie_with_lower_id_owner_keeps_owner` | a reachable tie to a *higher-id* centroid keeps the incumbent owner |
| `test_bound_roundoff_near_tie_is_not_pruned` | exact zero, `−1e-9`, `−1e-12` admitted; `−1e-2` pruned; coincident admitted |
| `test_near_tie_resolves_to_strict_argmax_owner` | a one-ULP score margin resolves to the strict `argmax` owner, not the incumbent |
| `test_split_moves_owner_outside_old_neighbourhood` | a split moves a vector that was not in the split posting's old neighbourhood (filter must not rely on proximity to the old posting) |

The two `_ball_may_switch` tests are pure-unit and exercise the predicate
directly; the other four drive `insert` and assert both the produced owner and a
clean `audit_maintainer`.

---

## 4. Stress matrix (`tests/test_inplace_stress.py`)

Six deterministic `(depth, Δ, leaf_size, split_threshold, seed)` configs
(both hierarchy depths, Δ ∈ {None, 0.2, 0.3, 0.5, 0.6}, leaf 20–50), each run
through a **mixed** insert/delete trace (`insert_prob = 0.7`, 400 operations,
800 vectors) with a checkpoint every 50 operations. At every checkpoint and at
the end:

* `audit_maintainer(maint, expected_live_vids=live)` — coverage, exactly one
  canonical owner per live vector, and NPA (`owner == argmax` over live leaf
  centroids, tombstones excluded);
* `audit_membership(maint)` — the independent fixed-Δ replica oracle;
* a direct assertion that no tombstoned id appears in any `members`.

Each config asserts `splits > 0`, so the split path is exercised rather than
theoretically covered. Plus `test_a_far_posting_is_untouched_by_a_split`: a
hand-built 2-D unit-vector fixture where a posting at 180° owns a vector equal
to its own centroid, and a split at 0°/4° must leave it bit-for-bit unchanged —
guarding the over-correction direction.

Both auditors are read-only and derive their expectation from the leaf centroids
and Δ, never from a production helper (`test_auditor_does_not_mutate_index`,
`test_oracle_does_not_mutate_the_index`).

---

## 5. Invariants verified

Across the tie/pruning tests, the six stress configs (every 50 ops) and the
existing NPA/membership suites:

* every live vector has **exactly one** canonical owner, and it is the strict
  `argmax` over live leaf centroids under the lowest-posting-id tie-break;
* `owned ⊆ members`, and `members` holds live vectors only (tombstones purged);
* the expected replica set matches `r(x) = d2/d1 − 1 ≤ Δ`, with the top-2 cap;
* replica membership changes are detected even when the canonical owner does not
  (`test_split_grants_replica_to_unchanged_owner`);
* a split can change ownership, membership, both, or neither — cases 1–3 are in
  the suites, the "neither" case is `test_a_far_posting_is_untouched_by_a_split`;
* the tolerance change does not alter the produced owner partition: 25 seeds
  (2 000 vectors, 1 700 inserts each) give **identical** `owned` partitions and
  split counts with the exact predicate vs the tolerant one (`/tmp/diff_tol.py`).

---

## 6. Cost

Measured by wrapping the in-place module's distance functions from an external
script and attributing each call to a phase (split filter, split repair,
membership pass, insert routing). The production hot path is untouched.

### 6.1 The filter is exactly `3·K` per split — old and new

Same config (n = 1 800, dim = 32, leaf = 50, threshold = 100, seed = 11, 500
initial then 1 300 inserts; and n = 5 000 insert-only):

| config | code | splits | `Σ K` at split | filter evals | filter / (`3·ΣK`) |
|---|---|---|---|---|---|
| 1 800 | old `838fd39` | 12 | 198 | 594 | **1.0** |
| 1 800 | new `HEAD` | 12 | 198 | 594 | **1.0** |
| 5 000 | old `838fd39` | 69 | 2 484 | 7 452 | **1.0** |
| 5 000 | new `HEAD` | 72 | 2 700 | 8 100 | **1.0** |

The filter issues exactly **three full-centroid inner-product scans per split**
in both versions. This confirms the claim in `R1_Report.md` §8 (`3·K` per split,
`8 775 → 9 006`) is a split-count/`K` artefact: the tolerance edit adds no
distance evaluations whatsoever. The small split-count difference (69 → 72) is
the split *sequence*, not the filter.

### 6.2 The filter is Δ-independent

New code, 1 800-vector config, same split sequence (12 splits, `ΣK = 198`):

| Δ | filter evals | split repair | membership pass |
|---|---|---|---|
| `None` | 594 | 56 104 | 0 |
| `0.2` | 594 | 56 104 | 229 219 |
| `0.5` | 594 | 56 104 | 229 219 |

Both the filter and the split-time ownership repair are bit-identical across Δ —
they only touch centroids and `owned`. Δ selects *which* vectors get a replica,
never how much similarity work is done.

### 6.3 The delta-membership pass (added by Task 2)

| config | Δ | splits | build bucket | maintenance bucket | `membership_vectors_recomputed` | replicas +/− |
|---|---|---|---|---|---|---|
| 1 800 | `None` | 12 | 67 394 | 0 | 0 | 0 / 0 |
| 1 800 | `0.2` | 12 | 67 394 | 229 219 | 13 138 | 162 / 129 |
| 1 800 | `0.5` | 12 | 67 394 | 229 219 | 13 138 | 620 / 566 |
| old 1 800 | `0.5` | 12 | 122 290 | 0 | – (no counter) | – |
| 5 000 | `0.5` | 72 | 650 039 | 8 802 721 | 179 065 | 11 317 / 3 778 |
| old 5 000 | `0.5` | 69 | 636 640 | 0 | – (no counter) | – |

The maintenance bucket is charged only to the membership pass and equals it
exactly; it is `O(N_live × K)` per split and, as §6.2 shows, **independent of
Δ** — Δ changes the decision, not the work. This is the cost of the correctness
Task 2 bought: the old code did not maintain Δ-membership at all, so its
"build" bucket had no such term.

### 6.4 Query cost of replicas

50 full-probe queries (beam width = number of postings), so a vector is scanned
once per posting it is a member of:

| Δ | query `scan` evals |
|---|---|
| `None` | 90 000 |
| `0.2` | 93 300 |
| `0.5` | 98 250 |
| old `0.5` | 99 900 |

Replicas trade query scan work for recall, as designed; with Δ = `None` the scan
count is exactly `50 × N_live`.

---

## 7. Test results

```
.venv/bin/python -m pytest tests/test_inplace_npa.py -q     # 16 passed
.venv/bin/python -m pytest tests/test_inplace_stress.py -q  # 7 passed
.venv/bin/python -m pytest -q                               # 107 passed
```

No existing test was weakened, relaxed, or removed.

---

## 8. Remaining limitations

* The tolerance is a **slack, not a proof**: a bound within `1e-5` of zero is
  admitted even if the true maximum is marginally negative. This can only
  over-admit candidates (never under-admit), and the per-vector screen plus
  `argmax` still decide the owner, so it trades a little split-time work for
  soundness — the direction the task asked for.
* The round-off hazard was **not reproduced as a failing state**; the search
  found no mismatch. The fix is justified by analysis and by keeping the filter
  at least as permissive as the auditor, not by a captured failing case.
* Still no posting merges, no centroid refresh, no dynamic-Δ, no split-policy
  change — canonical ownership is sound and validated, but this remains an
  approximation of SPFresh/LIRE, not a faithful implementation.
* One large-workload run (`n = 5 000`, Δ = `0.5`) shows the membership pass at
  ~8.8 M distance evaluations, ~93 % of all counted distances. It is
  `O(N_live × K)` per split and Δ-insensitive; a future incremental membership
  update would remove this term, but that is out of scope here.
