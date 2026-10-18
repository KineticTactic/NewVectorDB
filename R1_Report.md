# R1 — Deterministic NPA regression and in-place split repair

**Scope:** `vecstudy/maintain/inplace.py`, the NPA auditor in `vecstudy/harness/churn.py`,
and a new regression suite `tests/test_inplace_npa.py`.

**Status:** complete. Full suite `66 passed` (was 56) at the end of this task. The old
maintenance path is shown to fail the new regressions; the repaired path passes them with
zero global NPA violations. (R1 Task 2 then added fixed-Δ membership maintenance and
`tests/test_membership.py`, taking the suite to `94 passed` — see `R1_T2_Report.md`.)

---

## 1. What was wrong

DESIGN §2/§6.6 require in-place maintenance to reassign vectors **across neighbouring
postings** after every split. The implementation did reassign across neighbours, but it chose
the neighbours by **centroid proximity**: take the `n_reassign_neighbours + 2` postings
nearest to the split posting's old centroid, to `c0`, and to `c1`, union them, and re-score
that union.

That selection is *unsound*. `InPlaceMaintainer.insert()` never updates a posting centroid, so
a posting centroid can be stale and a vector can sit arbitrarily far from the centroid of the
posting that owns it. A split can then place a new centroid right next to that vector while
its owner is nowhere near the split on the centroid-distance ranking. The vector is never
examined, keeps a non-nearest owner, and the Nearest Partition Assumption is violated with no
error, no counter, and no failing test.

This is the silent failure mode DESIGN §2 warns about, reached through an entry point the
original neighbourhood heuristic did not cover.

### 1.1 Auditor status

`audit_maintainer()` already did the right check for the in-place arm: for every
non-tombstoned vector it scored the canonical vector against **all** leaf centroids with the
production inner-product metric, reported `(vid, owned_pid, nearest_pid, scores)` when the
owner was not the argmax, and never mutated the index. It was kept, not duplicated. What it
lacked was structure: no live-vector/postings counts, no bounded example list, no
machine-readable form. It also could not be pointed at a segment without a maintainer.

---

## 2. Files changed

| File | Purpose |
|---|---|
| `tests/test_inplace_npa.py` **(new)** | 5 auditor tests + 5 maintenance tests (split regression, churn, random workload, tie boundary predicate, end-to-end tie) |
| `vecstudy/maintain/inplace.py` | Sound split reassignment: per-posting bounding radius and ball-bound candidate selection; `_reassign_neighbourhood` rewritten |
| `vecstudy/harness/churn.py` | Auditor diagnostics (`live_vectors_checked`, `postings_checked`, `violation_count`, bounded `examples`, `to_dict()`), shared per-segment helper, `max_examples` parameter; docstrings state the Δ-overlap scope |
| `vecstudy/harness/runner.py` | Comment on `correctness_status`: coverage/ownership/NPA only, not overlap correctness |
| `vecstudy/core/config.py` | `n_reassign_neighbours` documented as legacy/compatibility-only |
| `DESIGN.md` | §6.6 reassignment description corrected |
| `REPORT.md` | Implementation notes and decision 4.4 corrected |
| `README.md` | Config comment corrected |

Unchanged on purpose: `Posting`, `Segment`, `core/distance.py`, `index/query.py`,
`maintain/outofplace.py`, the `IndexMaintainer` interface, the production metric, the
distance counter buckets, and the canonical-ownership vs. membership distinction.

---

## 3. The deterministic reproducer

`tests/test_inplace_npa.py::test_split_reassigns_neighbouring_vector`.

Two-dimensional unit vectors. Similarity is the inner product and distance is
`1 - similarity`, so angular order equals similarity order; every angle below is exact
geometry, not clustered data.

| angle | role |
|---|---|
| 0° | centroid of posting **A** before the split (A owns 3°, 5°, 43°) |
| 3°, 5° | A members, group 0 |
| 43° | A member, group 1 |
| 47° | pending A member; inserting it crosses `split_threshold` and splits A |
| −1° … −14.5° | ten distractor postings, one vector each, on the far side of A |
| 120° | centroid of posting **B** (B owns nothing yet) |
| 65° | probe vector **v**, inserted with `InPlaceMaintainer.insert` |

Before the split (degrees):

```
v(65) -> B(120) = 55   <- canonical owner, globally nearest
v(65) -> A(0)   = 65
v(65) -> d_i    >= 66
```

The probe is inserted through the production path and lands in B; the pre-split audit passes.

Inserting the 47° vector through the production path pushes A over the threshold and runs the
real `_split`. k-means over `{3, 5}` and `{43, 47}` produces `c0 = 4°` and `c1 = 45°`
(which posting holds which is the k-means label order; the test does not pin it).

After the split:

```
v(65) -> c1(45) = 20   sim 0.9391   <- strictly nearest, unique
v(65) -> B(120) = 55   sim 0.5736   <- current owner: NPA violated
v(65) -> d_i    >= 66
```

The margin is `0.3655`, so the expected owner is unambiguous.

**Why the old heuristic misses it.** B is not among the 10 postings nearest to `A_old`,
`c0` or `c1`: A itself plus the ten distractors all rank ahead of B for each of those three
queries. The neighbourhood set is therefore `{A, distractors}` and B's vector is never
re-scored.

The fixture starts from an explicitly constructed `Segment` (explicit vectors, centroids,
postings) and then drives the real `insert` → `_split` → `_reassign_neighbourhood` path. It
does not corrupt ownership after the split.

---

## 4. Evidence: failed before the fix

Old maintenance from `HEAD`, new auditor and new tests, run from a clean
`git worktree`:

```
$ python -m pytest tests/test_inplace_npa.py -q
3 failed, 5 passed

test_split_reassigns_neighbouring_vector
  AssertionError: probe left owned by [11] but its nearest posting is 0
test_subsequent_inserts_and_splits_keep_npa
test_random_workload_with_splits_stays_npa_clean
```

All five auditor tests pass in that worktree, so auditor correctness and maintenance
correctness are independently testable.

### 4.1 The bug is not confined to the hand-built fixture

`test_random_workload_with_splits_stays_npa_clean` is the ordinary "insert everything" path on
`make_synthetic(n=1200, dim=8, n_topics=8, seed=3)`, `leaf_size=25`, `split_threshold=40`,
first 700 vectors inserted:

| | splits | vectors_reassigned | audit |
|---|---|---|---|
| old | 32 | 282 | **3 violations** — vids 462, 513, 66 (margins 0.026, 0.011, 0.014) |
| new | 34 | 502 | 0 violations |

A wider grid (6 seeds × 4 configs — `(dim, leaf_size, split_threshold, delta)` =
`(8,25,40,None)`, `(32,40,70,None)`, `(16,30,50,0.3)`, `(32,20,30,0.6)` — 1200 inserts
each, audited every 100 operations and after deleting 20 % of the vectors):

| | configs with ≥1 NPA violation |
|---|---|
| old | **8 / 24** |
| new | **0 / 24** |

The two `delta > 0` configurations are not overlap evidence. This in-place index is built by
`insert` alone, and `insert` never creates a Δ replica, so those runs contain **zero** replica
members (`members == owned` in every posting, verified). They exercise NPA over `owned` with
`delta` merely configured. Even with replicas present the auditor would not check the Δ rule
(§6.1), so these numbers are not an overlap-correctness claim. (At the time of this task
in-place maintenance did not maintain replicas at all; R1 Task 2 added that, with a separate
membership audit — see `R1_T2_Report.md`.)

---

## 5. The repair

### 5.1 Bounding radius per posting

`InPlaceMaintainer` now keeps `self._spread[pid]`, the maximum Euclidean distance from a
posting's centroid to a vector it owns (`_owned_spread`). It is initialised from the segment
and updated on every ownership change: `insert`, `_split`, `_reassign_neighbourhood`,
`_rebuild_posting`. It follows the metric ranking because for unit vectors a smaller chord is
a larger inner product.

This is a maintenance-side geometric bound, not a counted query distance; it never touches
`Counters.distance_computations`.

### 5.2 Sound candidate selection

After a split, exactly two centroids changed: `pid` now holds `c0` and `new_pid` holds `c1`.
Every other centroid is untouched, and the index satisfied NPA before the split. Therefore a
vector `x` owned by an unchanged posting `b` can only lose ownership to `c0` or `c1` — never
to another unchanged posting.

Two questions are answered with the ball bound

```
max over {x : ||x - ref|| <= r}  x·(target - competitor)
    = ref·(target - competitor) + r·||target - competitor||
```

(`_ball_can_beat`). A posting is skipped only when the maximum is non-positive, i.e. when no
vector it owns can switch:

1. **Does `b` need its own vectors moved?** `ref = centroid_b`, `r = r_b`; `c0` or `c1` as
   target. True ⇒ `b` is a *source*.
2. **Could `b` be the new owner of a vector of the split posting?**
   `ref = A_old`, `r = r_A_old`, target `= centroid_b`. True ⇒ `b` is a *destination*.

Sources ∪ destinations is the affected set. The bound is a proof, not a heuristic: a posting is
excluded only when every vector in its radius ball is provably unaffected.

### 5.3 Exact reassignment

- **Split posting's vectors** (`pid`, `new_pid`): the centroid they were matched against no
  longer exists, so they are scored against every destination and moved to the argmax.
- **Unchanged postings' vectors:** only `c0` and `c1` moved, so three similarities decide
  them exactly — owner, `c0`, `c1`. A full destination scan is done only for the contested
  vectors, using `max(s0, s1) >= own_sim` so ties are resolved by the same rule as a global
  argmax instead of being silently kept.

Destinations are sorted ascending and ties resolve to the lowest posting id, matching the
auditor's `np.argmax` over all leaf centroids.

Ownership is written back to `owned`, replicas (`members \ owned`) are preserved, and radii
are refreshed for every posting that gained or lost a vector. Only postings whose owned set
actually changed are rewritten.

`n_reassign_neighbours` no longer gates correctness. The field stays in `MaintainConfig` so
existing sweep configs still load; it is documented as legacy.

### 5.4 Tie semantics at the pruning boundary

The documented tie-break is the one the auditor uses: nearest is `np.argmax` over the leaf
centroids, so an **exact tie resolves to the lowest posting id**. Two parts of the
reassignment have to agree with that, and they now do:

- the per-vector screen moves a vector when `max(s0, s1) >= own_sim`, i.e. a tie is enough;
- the pruning bound `_ball_may_switch` returns "may switch" when the maximum over the ball is
  `>= 0`, so the closed boundary is admitted. It also returns `True` for coincident centroids
  (`||target - competitor|| = 0`), where every vector in the ball ties and the lower id wins.

An earlier version pruned with a strict `> 0`, which disagreed with the `>=` screen at the
boundary: a posting could be dropped even though a reachable exact tie would have moved a
vector to a lower-id posting. That is the inconsistency fixed here.

One honest limitation: with unit-norm vectors, any *real* tie sits strictly inside the
posting's bounding ball (only a vector equal to its own centroid reaches the tangent point),
so an end-to-end tie produces a strictly positive bound and was already admitted by the old
strict comparison. The boundary itself is therefore tested directly on the predicate
(`test_pruning_bound_includes_exact_ties`, exact IEEE-double arguments), and the ownership
consequence is tested end-to-end (`test_exact_tie_resolves_to_lowest_posting_id`, exact
float32 tie between the owner and a new split centroid).

Both tests are mutation-checked: flipping the bound's `>=` back to `>` fails the predicate
test, and flipping the screen's `>=` back to `>` fails the end-to-end tie test.

---

## 6. Auditor changes

`audit_maintainer(maint, expected_live_vids=None, tolerance=1e-5, max_examples=10)` keeps its
signature and semantics and returns `AuditResult` with:

| field | meaning |
|---|---|
| `passed` | validity: no NPA violations, no ownership errors, no coverage errors |
| `npa_violations` | complete list of violation records |
| `violation_count` | `len(npa_violations)` |
| `live_vectors_checked` | non-tombstoned vectors with a canonical owner |
| `postings_checked` | live leaf postings examined |
| `examples` | bounded prefix of `npa_violations` |
| `examples_truncated` | whether the prefix omitted records |
| `ownership_errors`, `coverage_errors` | unchanged |
| `to_dict()` | machine-readable summary |

Each violation record carries `vid`, `owned_pid`, `nearest_pid`, `own_sim`, `best_sim`,
`own_distance`, `nearest_distance`, `margin` (and `segment` for out-of-place).

The audit remains read-only: it never repairs ownership and never mutates postings, centroids,
levels or tombstones. `test_auditor_does_not_mutate_index` snapshots the segment before and
after a dirty audit to prove it.

The check itself is unchanged in substance and now shared by both maintenance arms
(`_audit_segment_npa`): canonical ownership against **all** live leaf centroids with the
production metric, tombstones excluded, membership explicitly not treated as ownership.

### 6.1 What a passing audit does and does not prove

Checked:

- coverage of the live vector set;
- `owned ⊆ members` for every posting;
- exactly one canonical owner per live vector;
- NPA over canonical ownership: the owner is a nearest leaf centroid (ties allowed within
  `tolerance`, resolved in reporting by lowest posting id);
- tombstoned vectors excluded from all of the above.

**Not checked: Δ-overlap membership semantics.** The audit never evaluates
`d2 <= d1 * (1 + delta)`, the top-2 cap, or the SPANN RNG rule, and `members` is only
required to be a superset of `owned`. At the time of this task the in-place maintainer did not
maintain replicas either (inserts never added one; reassignment dropped moved vectors rather
than re-deriving the replica set); R1 Task 2 fixed that and added `audit_membership` as the
separate check. This audit itself still ignores the replica rule, so a `passed=True` result
remains an ownership/NPA statement, **not** an overlap-correctness statement.

---

## 7. Tests added

| test | pins |
|---|---|
| `test_auditor_accepts_valid_index` | clean index audits with zero violations and full counts |
| `test_auditor_detects_deliberately_invalid_owner` | hand-injected wrong owner is reported with correct `owned_pid`, `nearest_pid`, scores and margin |
| `test_auditor_does_not_mutate_index` | dirty audit leaves the segment byte-identical |
| `test_auditor_uses_ownership_not_membership` | a Δ-overlap replica member owned elsewhere is not a violation |
| `test_pruning_bound_includes_exact_ties` | the pruning predicate admits the exact tie boundary (max = 0), coincident centroids, and prunes only strictly below it |
| `test_exact_tie_resolves_to_lowest_posting_id` | an exact float32 tie between the owner and a new split centroid moves to the lower posting id |
| `test_auditor_excludes_tombstoned_vectors` | deleted vectors are skipped; the same state is reported once live again |
| `test_split_reassigns_neighbouring_vector` | the focused geometry above: pre-split audit clean, split shifts ownership, post-split audit clean, one owner per vector, `owned ⊆ members` |
| `test_subsequent_inserts_and_splits_keep_npa` | seven further inserts (more splits) audited after every insert, then a tombstone |
| `test_random_workload_with_splits_stays_npa_clean` | 700-insert synthetic workload with 20+ splits stays clean |

Verification commands:

```bash
cd ~/dev/cc/NewVectorDB
.venv/bin/python -m pytest tests/test_inplace_npa.py -q     # 10 passed
.venv/bin/python -m pytest -q                              # 66 passed
```

---

## 8. Cost

5 000 inserts, `dim=32`, `leaf_size=50`, `split_threshold=100`, `seed=11`:

| | splits | vectors_reassigned | build distance computations | wall |
|---|---|---|---|---|
| old | 75 | 1 873 | 611 190 | 0.12 s |
| new | 76 | 1 867 | 639 273 | 0.13 s |

That headline is not granular enough to check the claim "the filter costs the same 3·K", so the
counter was decomposed per split by logging every `add_distance(..., "build", n)` while a
split was in progress (helper call shapes identify the phase; the sum reproduces the counter
exactly):

| phase | old | new |
|---|---|---|
| centroid filter (`3·K` per split) | 8 775 | 9 006 |
| reassignment | 598 452 (one `\|V_aff\|·\|aff\|` scan) | 626 373 (= 340 017 per-source + 286 356 destination) |
| k-means balancing / hierarchy rebuild | 3 963 | 3 894 |
| **total build counter** | **611 190** | **639 273** |

Per split: filter 117.0 → 118.5, reassignment 7 979 → 8 242.

So the `3·K` statement is **true at the code level but not exactly true in the totals**: both
versions issue exactly three full-centroid scans per split, and the measured filter difference
(+231, +2.6 %) is entirely the split sequence itself — 75 splits with mean `K` = 39.0 versus 76
splits with mean `K` = 39.5, because earlier reassignments change where later inserts land.
The real added cost is in reassignment (+27 921, +4.7 %), split as 340 017 for three
similarities per owned vector of a provably affected posting and 286 356 for destination scans
(the split posting's vectors and contested vectors). `vectors_reassigned` can also rise where
the old code was silently missing victims (282 → 502 on the random workload above); that is
the reassignment the split actually required.

Worst case: when posting radii are large relative to centroid spacing, the bound admits many
destinations and a split degrades toward `O(N·K)`, i.e. the global pass. This is the cost of
soundness; it is bounded and measurable, and normal workloads sit at the figures above. The
auditor itself is `O(live vectors × postings)` per call and lives in tests/harness, not on the
query path.

---

## 9. Remaining gap to SPFresh / LIRE

The invariant now holds under splits, but this is not a faithful SPFresh/LIRE implementation:

- No posting merges on underflow; deletes only tombstone and optionally rebuild one posting.
- Split decision is size-only (`owned > split_threshold`) and always `k = 2` k-means; LIRE's
  split boundary / local-radius criterion is not implemented.
- Centroid statistics are never refreshed after inserts, which is precisely why vectors can
  drift far from their posting centroid. Radii make the reassignment sound despite this; they
  do not remove the staleness.
- Δ-overlap replicas are not maintained incrementally: inserts never add replicas,
  reassignment drops moved vectors from `members` instead of re-deriving the replica set.
  The audit does not check the replication rule at all, so a passing NPA audit must not be
  cited as overlap correctness (see §6.1).
- Hierarchy promotion appends to a child list and rebuilds on overflow; there is no
  propagate-and-resplit.
- Reassignment is triggered by splits only, never by deletes or tombstone rebuilds.
- `Counters.npa_violations_detected` is still never incremented by maintenance; auditing is an
  external, read-only check.

---

## 10. Reproduction of the before/after comparison

```bash
# old maintenance, new auditor + tests
git worktree add /tmp/old HEAD
cp tests/test_inplace_npa.py /tmp/old/tests/
cp vecstudy/harness/churn.py /tmp/old/vecstudy/harness/
cd /tmp/old && PYTHONPATH=/tmp/old ~/dev/cc/NewVectorDB/.venv/bin/python \
    -m pytest tests/test_inplace_npa.py -q     # 3 failed, 5 passed
```

```
E  AssertionError: probe left owned by [11] but its nearest posting is 0
```

Remove the worktree with `git worktree remove /tmp/old --force` afterwards.
