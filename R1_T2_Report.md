# R1 Task 2 — Correct fixed-Δ replica membership in the in-place maintainer

**Scope:** `vecstudy/maintain/inplace.py`, the in-place arm only; a new independent
membership audit in `vecstudy/harness/churn.py`; a new suite `tests/test_membership.py`.

**Not in scope, and not implemented:** dynamic-Δ calibration, budget→Δ inversion, automatic
knee detection, SPANN RNG filtering, posting merges, new split criteria, centroid refresh,
out-of-place maintenance, query-engine changes.

**Status:** complete. `94 passed` (was 66). Δ stayed fixed throughout; the metric, the
comparison and the tie-break were taken from the existing build path, not redefined.

---

## 1. The failure and its root cause

NPA repair (Task 1) made canonical ownership correct. Membership is a **different**
obligation — "which postings can find this vector" — and it was never derived; it was
carried forward. Four distinct defects:

| # | where | old behaviour | consequence |
|---|---|---|---|
| 1 | `insert` | appended the vector only to the nearest posting | a vector inserted near a boundary **never got its replica**, and no later operation could give it one without a split |
| 2 | `_split` | took the split posting's *existing* replica list and re-homed each entry to whichever new centroid it scored better against (`s0 >= s1`) | preserved stale memberships, invented new ones by coin-flip of the centroid split, and could not see a vector outside the split posting that had just become eligible |
| 3 | `_reassign_neighbourhood` | rewrote `members` as `new_owned ∪ (old_members \ old_owned)` | any replica the split invalidated survived ownership repair, and any replica the split newly required was never added |
| 4 | `delete` | tombstoned the id and left it in `members`/`owned` until the `tombstone_rebuild_frac` trigger fired | dead ids remained live memberships (returned by nothing, but present in `members`, in bloat, and in any later `members` copy) |

So "membership" was whatever history had left behind. Nothing computed it, nothing checked
it: the NPA auditor deliberately ignores `members` beyond `owned ⊆ members`, so every one of
these defects was silent. Root cause: **membership was treated as a by-product of ownership
maintenance instead of a property of the current centroid layout.**

## 2. The exact rule and metric (unchanged from the build path)

* Similarity = inner product; distance = `1 - similarity` (unit-normalised vectors).
* `d1` = distance to the **canonical owner** (not "nearest", so the rule is well defined even
  while an NPA violation is being repaired); `d2` = distance to the best posting **other than
  the owner**.
* Boundary statistic `r(x) = d2/d1 - 1`; a vector gets exactly one extra membership iff
  `r(x) <= Δ`, i.e. `d2 <= d1*(1+Δ)`. Top-2 cap: no vector holds more than two memberships.
* Tie-break: `np.argmax` over the leaf centroids, i.e. an exact tie goes to the **lowest
  posting id** — the same rule the auditor and `descend` use. The replica target is the
  masked `argmax`, so it inherits that tie-break too.
* Degenerate cases: `Δ = None` ⇒ no membership pass at all (`members == owned`);
  `d1 = 0` ⇒ ratio undefined, eligible only if `d2 = 0`; coincident centroids ⇒ every vector
  ties, lower id wins, replica required; negative distances from float noise are clamped to 0.
* `Δ` has one meaning: `r(x) <= Δ` and `d2 <= d1*(1+Δ)` are the same inequality. The
  maintainer uses the product form, the oracle uses the ratio form (different arithmetic
  path on purpose).

**Discrepancy found and left as a documented scope boundary:** `IndexConfig.rng_rule` is a
build-path option (`assign` skips a replica when the two centroids are closer than `d2`).
The in-place path does **not** apply it, which is the task's explicit exclusion of SPANN RNG
filtering. A config with `rng_rule=True` therefore holds more replicas after maintenance than
a fresh build would; `test_rng_rule_is_a_build_path_option_only` pins the divergence on a
geometry where the two rules disagree, and `_replica_target` documents it.

## 3. Files and implementation decisions

| file | change |
|---|---|
| `vecstudy/maintain/inplace.py` | `_overlap_eligible` / `_clamp_distance` / `_replica_target`; insert-time replica; `_recompute_membership` (full pass) + public `rebuild_membership`; split re-derives membership; ownership-only rewrite in `_reassign_neighbourhood`; eager delete; `maintenance` distance bucket; `_spread` upkeep |
| `vecstudy/core/metrics.py` | `replicas_added`, `replicas_removed`, `membership_vectors_recomputed` (cumulative, kept apart from `vectors_reassigned`), exposed in `stats()`/`maintain_dict()` |
| `vecstudy/harness/churn.py` | `MembershipAuditResult` + `audit_membership` (independent oracle, read-only); NPA-auditor docstrings now point at it instead of claiming membership is unchecked |
| `tests/test_membership.py` | 28 deterministic tests (see §5) |
| `DESIGN.md` §6.4 | ownership-vs-membership table: what each event does to which, and why a split re-derives membership |
| `README.md` | correctness section: `correctness_status` is NPA-only; membership has its own audit and its own counters |

Decisions worth naming:

1. **Insert computes the replica from the same similarity row that chose the owner** — one
   routing-distance vector, no extra scan; a second posting is touched only when Δ requires
   it.
2. **A split runs a full membership pass over all live vectors** (`O(N_live × K)`), because
   eligibility depends on the centroid layout. Candidate sets built from "postings near the
   change" are exactly the unsound heuristic that Task 1 removed for ownership; no bound
   exists that would justify one here.
3. **`_reassign_neighbourhood` no longer writes membership at all.** It writes `owned`, and
   keeps only memberships that ownership cannot invalidate (pre-existing replicas) so that
   `owned ⊆ members` holds in the interim; the pass then derives the real set. Writing
   `members` there is what made defect 3 possible.
4. **Delete is eager and counts replica removals**; `members`/`owned` are cleaned in the same
   call. Repeated delete is a no-op. Deletion cannot make any *other* vector eligible — the
   rule reads only that vector's coordinates — so no pass is needed on delete.
5. **Overlap disabled costs nothing**: `delta is None` returns before any similarity block,
   and `members == owned` is maintained by the other paths.
6. **Construction normalises tombstoned input.** A segment supplied through
   `initial_segment=` may carry tombstones that its arrays have not absorbed yet; those ids are
   purged from `owned`/`members` once at construction, so "`members` holds live vectors only"
   is true before the first operation instead of after the first rebuild.

## 4. Evidence that the oracle can fail

`audit_membership` derives the expected set from the leaf centroids, the owner and Δ, with a
stable descending sort and the ratio form — it never calls `_replica_target`,
`_overlap_eligible` or `_recompute_membership`. Each defect class is injected by hand into
an otherwise clean fixture (`test_oracle_detects_*`):

| injected fault | reported |
|---|---|
| replica deleted from the second posting | `missing = [{vid: 0, owner_pid: 0, expected_pids: [0,1], actual_pids: [0], missing_pids: [1]}]` |
| ineligible vector added to a posting | `extra = [{vid: 1, expected_pids: [0], extra_pids: [1]}]` |
| same vector twice in one posting | `duplicates = [{vid: 0, pid: 0, count: 2}]` |
| tombstoned id left in `members` | `dead_members = [{vid: 2, pid: 1}]` |
| owner's `members` missing an owned id | `missing_from_owner = [{vid: 1, owner_pid: 0}]` |
| vector owned by two postings | `ownership_conflicts = [{vid: 1, owner_pids: [0,1]}]` |
| (control) clean fixture | `passed`, `live_vectors_checked = 3` |
| (control) audit twice | index snapshot identical — the audit does not mutate |

Mutation checks against the **final** code (each mutation reverted afterwards; both files
restored byte-identical):

| mutation | result |
|---|---|
| insert adds no replica | 6 tests fail |
| no membership recompute after a split | 6 tests fail |
| eligibility `<=` → `<` | 1 test fails (the exact-tie fixture) |
| oracle's rule → "never replicate" | 9 tests fail (including both oracle-detection tests) |

## 5. Test results per operation

`tests/test_membership.py` — 28 tests, all passing, no existing test weakened:

* **Initial membership** (6): `delta=None` keeps single membership with zero counter work;
  insert creates the required replica; insert that does not qualify stays single; exact tie
  resolves to the lowest posting id (bitwise-equal similarities, `Δ = 0`); `d1 = 0` grants no
  replica; coincident centroids tie and replicate.
* **Oracle** (9): the eight cases in §4 plus "rejects a non-in-place maintainer".
* **Splits** (6): a real split (2-point posting → two singletons) removes a replica when the
  owner's centroid tightens — owner pid unchanged; the same split moves a vector to the new
  posting; a 4-point posting split grants a replica to a vector whose owner never changed;
  the pass visits every live vector; the pass is idempotent (second call adds/removes
  nothing); distances land in `maintenance`, not `routing`/`scan`, with
  `postings_read == bytes_read == 0`.
* **Deletions / rebuild** (5): deleting a 2-membership vector clears both, counts one replica
  removal, and keeps audits clean; repeated delete is idempotent; the deleted id is absent
  from search results at three query points; a later split does not resurrect it (checked in
  every posting, in the coverage audit and in search); a segment handed over with a
  pre-existing tombstone keeps that id out of `owned` and `members` across a split.
* **Policy agreement** (2): `Segment.build`'s Δ assignment satisfies the same oracle on 300
  synthetic vectors with bloat > 0; `rng_rule` divergence pinned (see §2).

Every test that exercises the maintainer runs **both** audits: `audit_maintainer`
(coverage, unique ownership, NPA) and `audit_membership`.

**Reassignment-only case:** with singleton split pieces, an owner-change vector has `d1 = 0`
after the split, so `r → ∞` and it can never be Δ-eligible at the same time. "Owner changed
*and* replica set recalculated" is therefore exercised on the same split by different
vectors (ownership move in the 2-point fixture, replica gain/loss in the 4-point fixture),
not by a single vector — a geometric property of unit vectors, not a test shortcut.

## 6. Both audits pass at scale

Churn stress grid: `Δ ∈ {0.1, 0.5, 1.0} × leaf_size ∈ {20, 50} × split_threshold ∈ {40, 100}`
= 12 configs, 1000 initial vectors, 800 mixed insert/delete operations each, both audits every
50 operations:

```
audit checkpoints=192 failures=0 max_bloat=1.000
```

(`max_bloat` is at Δ = 1.0, where `r <= 1` admits nearly every vector; Δ = 0.1 gives 0.21–0.35,
Δ = 0.5 gives 0.74–0.95. Bloat is a property of the fixed policy on this layout, not a
membership error — the build path applies the same inequality.)

## 7. Measured cost of the full pass

`make_synthetic(4000, 32, seed=5)`, `Δ = 0.5`, wall clock for the insert loop only, counters
from the same run. "recompute OFF" is the same code with the pass monkey-patched to a no-op —
a timing baseline only.

| workload | splits | K | `maintenance` pairs | share of counted distances | insert s (ON / OFF) |
|---|---|---|---|---|---|
| insertion-heavy (2000 → 4000, leaf 50, thr 100) | 35 | 75 | 6 159 908 | 91.1 % | 0.218 / 0.059 |
| split-heavy (1500 → 4000, leaf 10, thr 20) | 259 | 409 | 217 269 959 | 99.1 % | 4.227 / 0.396 |
| few splits (2000 → 4000, leaf 200, thr 400) | 4 | 14 | 191 273 | 65.4 % | 0.031 / 0.020 |
| overlap off (`delta=None`) | 35 | 75 | 0 | 0 % | 0.054 |

Distance counts are deterministic (identical across runs); wall clock is not — repeated runs
of the same "recompute OFF" row gave 0.059–0.098 s, so treat the seconds as one measurement
and the pair counts as the stable quantity.

Per split the pass costs `N_live × K` similarities: 175 997 pairs in run 1, 838 880 in run 2,
47 818 in run 3 (live ≈ 3–4 k, K growing as postings split). It dominates exactly when splits
are frequent relative to inserts and K is large — 259 splits over 409 postings take ~11× the
wall time of the same workload without the pass, and 99 % of all counted distances.

**This is the correctness-first baseline, not a performance claim.** The in-place arm is
`O(splits × N_live × K)` in maintenance distances; incremental local recomputation is future
work and must be validated against this pass (which is now also the oracle's contract).

State at rest is what the oracle checks, and it is correct; the counters confirm the work is
real: run 1 reports `membership_vectors_recomputed = 102 341` (= 35 passes × ~2.9 k live) and
`replicas +7937 / -2764`.

## 8. Remaining limitations

1. **Stale centroids.** Centroids are never refreshed (out of scope). Membership is defined
   against the *stored* centroids, so a posting whose centroid no longer represents its
   vectors can hold memberships that a refreshed index would not. Both auditors use the
   stored centroids and are therefore blind to this by construction.
2. **No posting merges.** Deletion-heavy postings shrink and stay; an empty posting keeps its
   centroid and can still win a replica. Correctness of membership is unaffected; density and
   recall are.
3. **Full pass per split.** No incremental candidate set is used, and none is claimed to be
   safe. Any future optimisation must be bounded (like Task 1's ball test for ownership) and
   checked against `audit_membership`.
4. **`rng_rule` is not applied in place** (§2). Configs that enable it will hold more
   replicas after maintenance than a fresh build. Documented, tested, not silently "fixed".
5. **Out-of-place arm unchanged.** `audit_membership` raises `TypeError` for it rather than
   pretending to check it; seams/memtable membership semantics remain as before.
   `runner.correctness_status` also stays ownership/NPA-only — the membership audit is run by
   the test suite (and available to the harness), not folded into that field, so experiment
   result rows keep their existing meaning.
6. **Counters are cumulative, not current-state**, and `replicas_removed` counts every
   membership the pass drops (including those dropped because a split removed the eligibility),
   so it is an event count, not "replicas the index lacks right now".
7. **This is not SPFresh/LIRE.** Fixed-Δ membership is now maintained correctly on insert,
   split and delete; the surrounding protocol (merges, centroid refresh, incremental replica
   maintenance, RNG filtering, dynamic Δ) is not implemented, and no compatibility claim is
   made.
