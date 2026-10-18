# R1 Task 4 — Deletion-driven posting merges (in-place arm)

**Scope:** the in-place maintainer `vecstudy/maintain/inplace.py` (a merge path
driven from `delete`), the new config knob `MaintainConfig.merge_threshold`, the
new merge counters in `vecstudy/core/metrics.py`, and a new focused suite
`tests/test_inplace_merge.py`. The out-of-place maintainer is untouched.

**Not in scope, and not implemented:** centroid refresh, dynamic-Δ, budget→Δ
inversion, split-policy changes, SPANN RNG filtering, query-engine changes,
out-of-place maintenance, and merge formulation for anything other than the
same-parent, combined-size-bounded pair rule described below.

**Status:** complete. `140 passed` (was 107): +33 merge tests, and two legacy
tests adjusted for the new eager merge-on-delete behaviour (no assertions were
weakened — see §5). On every measured workload both auditors pass, and merge is
strictly the structural inverse of split.

---

## 1. The policy

A merge is driven **only** by `delete` (`InPlaceMaintainer.delete` calls
`_merge_after_delete` at `vecstudy/maintain/inplace.py:294`). Insert never
merges, so the split path (grow on insert) and the merge path (shrink on delete)
stay separable and each is exercised by its own tests.

### 1.1 Trigger (underflow)

A leaf posting is **underfilled** when its *live owned* count is strictly below
the effective merge threshold:

```
effective = merge_threshold if merge_threshold is not None
            else max(1, split_threshold // 2)          # inplace.py:714
underfilled[i] = len(live_owned(postings[i])) < effective
```

Replica membership (`members`) is deliberately **not** counted: occupancy is the
auditor's question (`owned`), so a posting that is only being *borrowed* by other
vectors is still allowed to merge. Strict `<` means a posting sitting exactly on
the threshold is not underfilled (pinned by
`test_no_merge_when_owning_count_stays_at_threshold`).

### 1.2 Candidate rule

A pair `(a, b)` is eligible iff:

1. at least one of `a`, `b` is underfilled;
2. both leaves share the **same immediate parent** in the current hierarchy
   (`_leaf_parent_of`, `inplace.py:734`). At depth 1 there is no routing level,
   the mapping is empty, and every leaf is a sibling;
3. `live_owned(a) + live_owned(b) <= split_threshold` — the same hysteresis the
   split path uses. This bounds the *survivor's* occupancy: its live owned set is
   a subset of the union, so the merged posting itself is never over threshold.
   It does **not** stop a merge from pushing a *neighbour* over threshold —
   reassignment can scatter union vectors into other postings, which
   `_split_oversize` then splits. The measured consequence (merge/split mutual
   undo at small thresholds) is in `R1_T5_Report.md` §3.

### 1.3 Partner selection (deterministic)

Among eligible pairs, the pair with the **smallest Ward cost** wins

```
C(a, b) = n_a·n_b/(n_a+n_b) · ‖μ_a − μ_b‖²          # inplace.py:792
```

where `n_i` is the live-owned count and `μ_i` the mean of the live-owned vectors
(the k-means objective on the union, *not* the stored centroid). An empty posting
has no mean and is given cost `0`, so an emptied leaf is removed without a
division by zero. Ties are broken by the key `(cost, a, b)`, i.e. the lower
posting id (`test_equal_costs_resolve_deterministically_to_lower_id`).

(The three eligibility predicates — underflow, same parent, combined size — are
each pinned by their own test: `..._triggers_merge_...`,
`test_merge_only_within_same_parent`, `test_no_merge_when_combined_exceeds_split_threshold`.)

### 1.4 Survivor and centroid

The survivor is the **lower posting id**. The merged centroid is the mean of the
**union of the live owned vectors** — the same un-normalised, float32 k=1
k-means centroid the build and split paths use (`inplace.py:890`;
`test_delete_below_threshold_triggers_merge_and_centroid_is_live_mean`). If the
union is empty (both leaves fully tombstoned) the survivor keeps its old centroid
so the row stays well defined; the empty leaf is then pruned.

### 1.5 Structure

`_remove_leaf` (`inplace.py:801`) drops the removed posting and its leaf-centroid
row, compacting in order and renumbering `Posting.id` so the positional invariant
(`postings[i].id == i`, leaf row `i` ↔ posting `i`) survives. The upper routing
levels are stale at that point, so `_rebuild_hierarchy` rebuilds them from the new
leaf centroids at the configured depth. The final layout is checked for
`ids == range(K)`, leaf-centroid sync, and full parent-child coverage at every
depth (`test_higher_id_removed_and_ids_stay_contiguous`,
`test_hierarchy_is_valid_after_merge`).

### 1.6 Ownership repair

`_reassign_after_merge` (`inplace.py:909`) mirrors the split repair and relies on
the same pre-condition: the merge changes **exactly one** leaf centroid (the
survivor's) and removes one, so for every unchanged posting the merged centroid
is the only new competitor. It:

* screens unchanged postings with the sound ball bound `_ball_may_switch`
  (ref = the unchanged posting's centroid, target = merged centroid, competitor =
  same, radius = that posting's spread);
* decides contested vectors with a full `np.argmax` over the resulting leaf
  centroids (ties → lowest posting id);
* gives every vector formerly owned by either merged posting a full global scan
  (its old centroid no longer exists).

After the repair, ownership is asserted to equal a brute-force `argmax` for
**every** live vector, not just the ones expected to move
(`test_ownership_equals_global_argmax_after_merge`). The three move classes are
covered explicitly: a vector from the removed posting changing owner
(`test_vector_from_removed_posting_changes_owner`), an unchanged posting keeping
its vectors (`test_untouched_posting_keeps_vectors_and_spread`), and an exact tie
resolving to the lower id (`test_exact_tie_follows_lowest_posting_id`).

### 1.7 Membership, oversize, pruning, atomicity

* **Membership** is re-derived from the final centroids by the production
  fixed-Δ pass `_recompute_membership` (`inplace.py:867`), so replicas that
  became/stayed/no-longer eligible are handled by the same code as a split
  (`test_merge_membership_matches_oracle_with_overlap`,
  `test_unchanged_owner_replica_is_recomputed`). A tombstoned vector can never be
  re-added (`test_deleting_multi_replica_vector_clears_stale_references`).
* **Spreads** are recomputed from the final live owned sets
  (`_recompute_all_spreads`), verified against an independent recomputation.
* **Oversize**: if reassignment pushes a posting above `split_threshold`, the
  existing `_split` is reused (`_split_oversize`).
* **Empty leaves** are pruned, keeping at least one posting, so an all-deleted
  index stays query-safe (`test_no_live_vectors_is_query_safe`,
  `test_single_remaining_posting_stays_valid`).
* **Atomicity**: `_merge_after_delete` snapshots the segment, tombstones, spreads
  and counters, and rolls all of them back on any exception; the exception is
  re-raised so a half-applied merge is never observable
  (`test_exception_during_merge_rolls_back`). The `delete`'s own tombstoning is
  *not* part of the transaction — only the merge is atomic.

### 1.8 Config and telemetry

* `MaintainConfig.merge_threshold: int | None = None` — default
  `max(1, split_threshold // 2)`. Setting `0` disables merges entirely (used by
  the split-isolation legacy test).
* New counters (`metrics.py`): `merges`, `merge_candidates_examined`,
  `merge_vectors_reassigned`, `merge_similarity_evaluations`,
  `merge_hierarchy_rebuilds`. `stats()` exposes `merge_threshold` plus all five,
  so the harness Parquet rows carry them automatically.

---

## 2. A real bug found and fixed while testing

`_split_oversize` originally looped, splitting the smallest over-threshold
posting until the index "stabilised", and raised `RuntimeError` after a cap. On a
posting whose live owned set does **not** partition cleanly (interleaved vectors
produce two nearly-coincident balanced-k-means centroids, after which the
`argmax` ownership repair sends every vector back to one of them), the split
appended an *empty* leaf and removed no vector, so the loop appended empty leaves
forever until it raised — even though the index was otherwise correct.

Fix: `_split_oversize` (`inplace.py:1014`) is now **best-effort and
terminating** — each over-threshold posting is split **at most once**. A posting
a split cannot divide is left over-threshold (still correct), and the empty leaf
the attempt created is removed by `_prune_empty_leaves`. The reproduced case now
finishes with `merges=1, splits=1` and both auditors passing.

---

## 3. Regression suite (`tests/test_inplace_merge.py`, 33 tests)

| area | tests |
|---|---|
| trigger boundary & centroid | `test_no_merge_when_owning_count_stays_at_threshold`, `test_delete_below_threshold_triggers_merge_and_centroid_is_live_mean` |
| no compatible / size-blocked sibling | `test_no_merge_without_compatible_sibling`, `test_no_merge_when_combined_exceeds_split_threshold` |
| partner selection | `test_lowest_ward_cost_partner_is_selected`, `test_equal_costs_resolve_deterministically_to_lower_id` |
| hierarchy depth / parent | `test_depth1_has_implicit_root_and_merges_siblings`, `test_merge_only_within_same_parent[3,4]`, `test_depth1_and_2_treat_all_leaves_as_siblings[1,2]`, `test_hierarchy_is_valid_after_merge[1,2,3,4]` |
| structure / ids | `test_higher_id_removed_and_ids_stay_contiguous` |
| ownership repair | `test_vector_from_removed_posting_changes_owner`, `test_ownership_equals_global_argmax_after_merge`, `test_untouched_posting_keeps_vectors_and_spread`, `test_exact_tie_follows_lowest_posting_id`, `test_merge_leaves_no_zero_or_multiple_owners` |
| deletions / replicas / membership | `test_deleted_vector_is_absent_from_all_live_arrays_after_merge`, `test_deleting_multi_replica_vector_clears_stale_references`, `test_merge_membership_matches_oracle_with_overlap`, `test_unchanged_owner_replica_is_recomputed` |
| spreads | `test_spread_radii_match_independent_recomputation` |
| repeated ops | `test_split_then_delete_merge_is_consistent`, `test_repeated_delete_merge_accumulates_no_stale_or_duplicates`, `test_mixed_churn_keeps_both_audits`, `test_single_remaining_posting_stays_valid`, `test_no_live_vectors_is_query_safe` |
| determinism / atomicity | `test_failed_merge_attempt_leaves_state_unchanged`, `test_exception_during_merge_rolls_back` |

Every test that mutates the index runs **both** independent validators:
`audit_maintainer` (coverage, exactly-one-owner, NPA) and `audit_membership` (the
fixed-Δ replica oracle). The mixed-churn test asserts `merges > 0 and splits > 0`
across a 300-operation trace, so both structural paths are exercised rather than
theoretically covered.

---

## 4. Cost

Distance work is attributed by the production counters (the merge path charges
the `build` bucket, the same bucket the split repair uses). Deterministic churn on
600 2-D clustered unit vectors (10 clusters, angular spread 0.05),
`branching_factor=8`, `leaf_size = split_threshold`, `initial_n=300`, 400
operations, `insert_prob=0.35`, trace seed 7; `merge_threshold` defaulted to
`split_threshold // 2`. **`K` is the final leaf count** (initial `K` is 15 at
`split_threshold=20` and 8 at `split_threshold=40`).

| workload | split_thr | depth | Δ | K | merges | splits | cand. examined | vectors reassigned | sim evals | audits |
|---|---|---|---|---|---|---|---|---|---|---|
| mixed | 20 | 1 | None | 10 | 611 | 679 | 8 199 | 5 684 | 163 417 | pass |
| mixed | 20 | 2 | 0.4 | 10 | 611 | 679 | 8 199 | 5 684 | 163 417 | pass |
| mixed | 40 | 1 | None | 6 | 51 | 49 | 126 | 952 | 11 007 | pass |
| mixed | 40 | 2 | 0.4 | 6 | 51 | 49 | 126 | 952 | 11 007 | pass |
| delete-only | 20 | 1 | None | 1 | 415 | 404 | 3 557 | 4 071 | 94 503 | pass |
| delete-only | 20 | 2 | 0.4 | 1 | 415 | 404 | 3 557 | 4 071 | 94 503 | pass |
| delete-only | 40 | 1 | None | 1 | 191 | 190 | 490 | 3 294 | 31 664 | pass |
| delete-only | 40 | 2 | 0.4 | 1 | 191 | 190 | 490 | 3 294 | 31 664 | pass |

Observations:

* Merge is roughly as frequent as split. This is **not** a healthy inverse-operation
  count: at small thresholds the two paths largely undo each other within a single
  delete (a merge scatters its union into neighbours and `_split_oversize`
  re-splits them). The identity
  `K_initial + splits − merges − empty-leaf prunes = K_final` still holds and both
  audits pass, but the structural work is high. Evidence and cause in
  `R1_T5_Report.md` §3.
* Candidate examination is cheap: `2.5` eligible pairs examined per merge at
  `split_threshold=40` (`13` at the tiny threshold 20, where many small leaves
  are simultaneously underfilled).
* Similarity work per merge is `O((|selected| + contested) × K)`: it averaged
  `~166–228` evaluations per merge here, the same order as a split repair. The
  `build` bucket is the only place merge cost appears; routing and scan are
  untouched.
* Δ does not change the merge work or the produced structure on these workloads
  (identical numbers for `None` vs `0.4`): Δ selects replicas, never ownership or
  merge decisions.

---

## 5. Legacy tests adjusted

The eager merge-on-delete can move a vector / collapse postings in fixtures whose
premise was "delete only tombstones". Two tests were adjusted to keep their
original *intent*; neither assertion was weakened:

* `tests/test_inplace_npa.py::test_auditor_excludes_tombstoned_vectors` — the test
  is about the auditor ignoring tombstones, not about merges. It now tombstones
  the victim directly (`maint._tombstones.add(...)`, `seg.mark_deleted(...)`)
  instead of calling `delete`, so the merge path cannot collapse the posting it
  later re-attaches ownership to.
* `tests/test_membership.py::test_split_after_delete_does_not_resurrect_the_vector`
  — this test isolates split semantics. It constructs the maintainer with
  `merge_threshold=0` (merges disabled) so the emptied posting stays in place and
  the later split is the only structural change; the "tombstoned id never comes
  back" and audit assertions are unchanged.

---

## 6. Invariants verified

Across the merge suite, the churn measurements above, and the existing NPA /
membership / stress suites:

* every live vector has exactly one canonical owner, equal to the strict
  `argmax` over live leaf centroids (lowest-id tie-break);
* `owned ⊆ members`, `members` holds live vectors only (no tombstone survives a
  merge);
* `postings[i].id == i` and leaf-centroid row `i` matches posting `i` after every
  merge;
* the routing hierarchy is rebuilt with full parent-child coverage and valid
  indices at depths 1–4;
* spreads equal the independently recomputed live-owned radii;
* a failed merge leaves the segment bit-for-bit unchanged and restores the
  affected counters.

---

## 7. Test results

```
.venv/bin/python -m pytest tests/test_inplace_merge.py -q   # 33 passed
.venv/bin/python -m pytest -q                               # 140 passed
```

No existing test was deleted; the two edits in §5 preserve each test's original
intent.

---

## 8. Remaining limitations

* **Not SPFresh/LIRE equivalence.** The merge rule is a deliberate simplification:
  same-parent + size-bounded + Ward-cost pair selection with a k=1 centroid. It
  is documented and tested, not claimed to match any published merge policy.
* **Parent isolation is only as strong as the hierarchy.** After a merge shrinks
  `K`, `build_hierarchy` may place the surviving leaves under one routing node, at
  which point they become siblings and can merge — the depth ≥ 3 cross-parent
  guarantee is a property of the *current* tree, not a permanent partition.
* **Over-threshold leaves can persist.** Because a split is attempted at most once
  (`_split_oversize`), a posting whose owned set does not partition cleanly can
  stay above `split_threshold`. This is correct but means `split_threshold` is a
  soft cap after merges, not an invariant.
* **Membership re-derivation is a full pass.** `_merge_after_delete` ends with the
  whole-index `_recompute_membership`, i.e. `O(N_live × K)` when Δ ≠ `None`; an
  incremental update would remove that term but is out of scope.
* **No centroid refresh.** Merged centroids are plain means of live-owned
  vectors; there is still no periodic re-centring, so long churn traces can drift
  from the construction centroids (unchanged from Task 3).
