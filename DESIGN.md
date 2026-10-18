# DESIGN.md — Hierarchical Clustering Vector Index (Research Reference Implementation)

**Audience:** an AI coding agent implementing this from scratch.
**Language:** Python 3.11+ with NumPy and scikit-learn. **Not C++.** See §12.
**Purpose:** a controlled research instrument for measuring how index-maintenance strategy interacts with hierarchy depth. It is **not** a production vector database.

Read §1 and §2 before writing any code. They contain the constraints that make the rest of the document make sense, and several of them are counterintuitive.

---

## 1. What this is for, and why that changes the design

This codebase exists to answer one question:

> How does the choice between **in-place** and **out-of-place** index maintenance interact with **hierarchy depth** in a clustering-based vector index?

Everything follows from that. Three consequences that will feel wrong if you're used to building production systems:

**1. The primary outputs are counters, not timings.** Recall, distance-computation counts, simulated page accesses, and reassignment counts are *algorithmic properties* — identical whether the code is fast or slow. Wall-clock time is not, and is explicitly out of scope. **Never optimise at the cost of clarity, and never add a fast path that changes counted behaviour.**

**2. The two maintenance strategies must share everything except maintenance.** One query path, one distance kernel, one clustering routine, one file format. If you find yourself writing `if strategy == "inplace"` anywhere outside `maintain/`, you have introduced a confound and the experiment is invalid. This is the single most important rule in the document.

**3. Determinism is a correctness requirement, not a nicety.** Same seed plus same config must produce byte-identical counters. A later C++ port will be validated by exact counter equality against this implementation, so any nondeterminism here destroys that.

---

## 2. Findings this design encodes

These come from prior experiments. They are **requirements**, not suggestions. Do not "simplify" them away.

| Finding | Requirement |
|---|---|
| Single-path descent through a hierarchy loses **24 pp** of recall@10 at depth 3–4 | Beam width must be configurable per level and **default to 8**. Width 1 must be reachable only by explicit configuration. |
| Beam width fully recovers depth-induced recall loss; at beam 8, depth 4 reaches within 0.4 pp of exhaustive flat routing at **58% of the routing cost** | Depth must be a free parameter. Do not hard-code 2 levels. |
| Pruning *inside* a leaf (a leaf-level mini-IVF choosing one sub-cluster) costs ~**39 pp** of recall to save ~11% of query latency | **Never prune within a posting.** Always exact-score every live vector in every fetched posting. Local centroids may be used to *order* a scan for early termination, never to skip. |
| Splitting a partition violates the Nearest Partition Assumption for vectors in *neighbouring* partitions, not just the split one | In-place maintenance must reassign across nearby partitions after every split. Local-only reassignment is a silent correctness bug. |
| Δ-overlap raises the single-posting recall ceiling; beam raises recall *toward* that ceiling. Different mechanisms. | Both must be independently configurable and independently measurable. |
| Δ as an absolute distance is unusable across clusters of differing radius; on isotropic data every Δ from 0.02–0.40 duplicates 100% of vectors | Δ is **relative**: a vector is a border vector iff `d2 <= d1 * (1 + delta)`. Always log the realised bloat. |

---

## 3. Non-goals — do not implement these

An agent's instinct is to build a complete system. Resist it. Each of these is explicitly out of scope, and adding any of them makes the codebase worse for its purpose:

- Product quantization, scalar quantization, or any compression
- Graph indexes (HNSW, NSG, Vamana)
- Concurrency, threading, async, multiprocessing — **single-threaded throughout**
- Real file I/O for postings (page accesses are *simulated counters*, see §6.4)
- A network layer, server, or REST API
- A CLI beyond what §10 specifies
- Caching, memoisation, or any optimisation that changes counted work
- Metadata filtering (a later extension; do not stub it)
- Any C++ (§12)
- Progress bars, logging frameworks, colourised output

If a requirement seems to need one of these, **stop and flag it** rather than implementing it.

---

## 4. Module layout

Create exactly this structure. Do not add top-level modules without flagging.

```
vecstudy/
  __init__.py
  core/
    types.py          # dataclasses and type aliases
    config.py         # IndexConfig, ExperimentConfig
    metrics.py        # Counters, the metric collection context
    distance.py       # distance kernels, counted
    clustering.py     # balanced k-means wrapper
  index/
    hierarchy.py      # routing tree: build + beam descent
    postings.py       # leaf postings, Δ-overlap assignment, tombstones
    segment.py        # Segment = one hierarchy + its postings (a searchable unit)
    query.py          # THE query path (shared by both arms)
  maintain/
    base.py           # IndexMaintainer ABC
    outofplace.py     # buffer → flush → sealed segment → compaction
    inplace.py        # split → promote → LIRE reassignment
  harness/
    datasets.py       # loaders, ground-truth computation
    runner.py         # sweep execution
    results.py        # result storage (Parquet or SQLite)
tests/
  test_distance.py
  test_hierarchy.py
  test_postings.py
  test_query.py
  test_maintain.py
  test_invariants.py
```

---

## 5. Core types

```python
# core/types.py
VectorId = int          # index into the global vector array
PostingId = int
NodeId = int

@dataclass(frozen=True)
class Posting:
    id: PostingId
    centroid: np.ndarray            # float32, shape (D,)
    members: np.ndarray             # int64 VectorIds, may include replicas
    owned: np.ndarray               # int64, members for which this is the NEAREST posting
    # invariant: owned ⊆ members

@dataclass
class HierarchyLevel:
    centroids: np.ndarray           # float32, (n_nodes, D)
    children: list[np.ndarray]      # children[i] = child indices in the level below
```

`owned` vs `members` matters: `owned` drives maintenance decisions (splits, NPA checks); `members` drives search. Conflating them is a common and subtle bug.

---

## 6. Components

### 6.1 Distance (`core/distance.py`)

Vectors are **L2-normalised at load**; similarity is inner product, distance is `1 - similarity`. Two functions:

```python
def sim_one_to_many(q: np.ndarray, X: np.ndarray, counters: Counters, bucket: str) -> np.ndarray
def sim_many_to_many(A: np.ndarray, B: np.ndarray, counters: Counters, bucket: str) -> np.ndarray
```

Every call increments `counters.distance_computations[bucket]` by the number of pairs scored. Buckets are at minimum `"routing"` and `"scan"` — the distinction is load-bearing, since the finding in §2 depends on routing cost and scan cost being separable.

There must be **no** path that computes a distance without going through these functions.

### 6.2 Clustering (`core/clustering.py`)

```python
def balanced_kmeans(X, n_clusters, *, seed, max_iter=50, balance_tol=0.25) -> (centroids, labels)
```

Plain `sklearn.KMeans` first, then a balancing pass: while any cluster exceeds `(1 + balance_tol) × mean_size`, move its farthest-from-centroid members to their next-nearest cluster that is under the mean. Cap at 10 balancing rounds and log the realised size distribution — do not loop until convergence, it may not converge.

Balance matters because unbalanced postings make "one posting read" mean wildly different amounts of work per query, which contaminates the counters.

### 6.3 Hierarchy (`index/hierarchy.py`)

**Build** is bottom-up. Given leaf centroids and a branching factor `B`:

```
level = leaf_centroids
while len(level) > B:
    parent_centroids, labels = balanced_kmeans(level, ceil(len(level)/B), seed=...)
    record children mapping
    level = parent_centroids
```

Levels are stored **top-first**. Depth 1 means no routing levels above the leaves — a flat scan of all leaf centroids. This is the control arm and must work.

**Descent** is a restricted beam:

```python
def descend(q, hierarchy, beam_widths: list[int], n_postings: int,
            counters) -> list[PostingId]
```

1. Score all nodes at the top level. Keep the best `beam_widths[0]`.
2. For each level down: gather the union of children of survivors, score all of them, keep the best `beam_widths[level]` **overall** (not per-parent).
3. At the leaf level keep the best `n_postings`.

`beam_widths` is a survivor count per level, not a branching factor — cost stays bounded rather than growing as bᴸ. If `beam_widths` is shorter than the depth, repeat its last element.

Also implement `descend_global`, identical but maintaining a single frontier of the best `w` candidates seen across all levels, allowing lateral movement between subtrees. Default off; it is an experimental arm.

### 6.4 Postings (`index/postings.py`)

**Overlap assignment:**

```python
def assign(X, centroids, *, delta: float | None, rng_rule: bool, counters) -> (postings, bloat)
```

For each vector, find the two nearest centroids. Always assign to the nearest (this becomes `owned`). If `delta is not None` and `d2 <= d1 * (1 + delta)`, also add it to the second's `members`.

When `rng_rule` is on, skip the replica if the two centroids are closer to each other than the vector is to the second — this is SPANN's relative-neighbourhood rule and it prevents wasted replication between near-duplicate cells.

Return the realised `bloat = total_members / N - 1`. **Always record it**; it's half of every cost comparison involving overlap.

**Membership ≠ ownership.** `owned` is the canonical owner (exactly one posting per live vector, NPA-checked); `members` is the searchable set (owner plus the Δ replicas). Δ is **fixed**: no calibration, no budget inversion, no RNG filtering in the in-place arm. The in-place maintainer keeps the two consistent by construction:

| event | ownership | membership |
|---|---|---|
| insert | nearest leaf centroid (ties → lowest posting id) | owner + the best *other* posting iff `d2 <= d1*(1+Δ)` (one extra at most) |
| split | full reassignment of every posting the new centroids can contest | full re-derivation for **all** live vectors from the new centroids |
| delete | tombstone, dropped from `owned` | dropped from every `members` array immediately; never re-added (tombstones are excluded from live sets) |
| overlap disabled (`delta is None`) | as above | `members == owned`, no membership pass runs |

A split re-derives membership because eligibility is a property of the *centroid layout*: a vector whose owner does not change can still gain or lose a replica, and the set of vectors whose owner changes is not the set whose replicas change. The candidate set for the membership pass is "all live vectors" on purpose — any neighbourhood-restricted set needs a bound proving no vector outside it can become eligible, and no such bound exists for a stale-centroid index. Cost is one `N_live × K` similarity block per split, charged to the `maintenance` distance bucket, never to `routing`/`scan`.

Membership is audited separately from NPA by `harness.churn.audit_membership`, which re-derives the expected `members` set per live vector from the centroids, the canonical owner and Δ using the `r(x) = d2/d1 - 1 <= Δ` form (independently of the maintainer's helpers).

**Simulated I/O.** Postings are in memory. Reading one increments:

```python
counters.postings_read += 1
counters.bytes_read += posting_header_bytes + len(members) * D * 4
```

`posting_header_bytes` defaults to 64. This is the implementation-independent stand-in for disk cost — no real file I/O.

**Tombstones.** Each segment holds a `set[VectorId]` of deleted ids. Deleted vectors are filtered *after* scoring, never before — that's what real systems do and it's why over-fetch is needed.

### 6.5 Query (`index/query.py`)

**This is the only search function in the codebase.** Both maintenance arms call it unchanged.

```python
def search(q: np.ndarray,
           units: list[Segment],
           cfg: QueryConfig,
           counters: Counters) -> list[tuple[VectorId, float]]
```

For each unit: descend to `cfg.n_postings` postings, fetch them, **exact-score every live member of every fetched posting** (see §2 — no pruning within a posting), collect candidates. Then merge across units, filter tombstones, and return the top `k`.

Over-fetch: internally retrieve `k' = ceil(k * (1 + cfg.overfetch))`, default `overfetch=0.2`, so tombstoned results don't shrink the answer below `k`.

A unit contributing zero live candidates is normal, not an error.

### 6.6 Maintainers (`maintain/`)

```python
class IndexMaintainer(ABC):
    @abstractmethod
    def insert(self, vid: VectorId, vec: np.ndarray) -> None: ...
    @abstractmethod
    def delete(self, vid: VectorId) -> None: ...
    @abstractmethod
    def maintain(self) -> None: ...          # called after each batch
    @abstractmethod
    def searchable_units(self) -> list[Segment]: ...
    @abstractmethod
    def stats(self) -> dict: ...
```

**Out-of-place (`outofplace.py`)** — LSM-Vec-style log-structured maintenance.

- **Zero-distance insert path:** Buffer incoming vectors into an in-memory MemTable (flat vector list, scored exhaustively at query time as a flat searchable unit). No hierarchy traversal is performed on insert ($O(1)$ write cost, 0 distance evaluations).
- **Batch flushing:** When the MemTable reaches `memtable_limit`, live vectors are clustered into a new sealed immutable `Segment` with complete hierarchy levels and fixed-Δ replicas. Any deleted vectors that were buffered in the MemTable are cleared from tombstones upon sealing.
- **Dual compaction triggers:** Compaction is triggered when the sealed segment count reaches `compaction_threshold` (read amplification mitigation) OR when the global tombstone ratio across sealed segments exceeds `tombstone_rebuild_frac` (deletion-driven storage reclamation).
- **Atomic compaction with rollback:** Unions live vectors across all sealed segments (`is_alive(vid) and vid not in _tombstones`), re-clusters from scratch, builds a consolidated segment, and replaces the sealed segment list. If an error occurs, the previous segment list and tombstone state are restored.
- **Tombstone garbage collection:** Following compaction, tombstones for vectors belonging to the purged sealed segments are permanently discarded from `_tombstones`, retaining only any pending MemTable tombstones.
- **Immutable sealed segments:** In accordance with LSM-VEC (§3.2), sealed segments are write-once disk blocks. Deletions never rewrite on-disk postings; they are recorded in segment tombstone sets and filtered during query scoring until compaction reclaims physical space.
- **Fixed-Δ replica membership:** Each sealed segment enforces relative border membership ($r(x) = d_2/d_1 - 1 \le \Delta$). The MemTable maintains strictly flat membership ($members = owned$).
- **Cross-segment canonical ownership:** Every live vector is owned by exactly one unit (either a specific leaf posting of one sealed segment, or the MemTable), audited by `audit_maintainer`.
- Record `flushes`, `compaction_events`, `vectors_reclustered`, `hierarchy_rebuilds`, and unit counts over time.

**In-place (`inplace.py`)** — one segment, mutated.

- Insert: descend to the nearest leaf posting, append to `owned` and `members`.
- When a posting's `owned` count exceeds `split_threshold`, split it in two via `balanced_kmeans(k=2)`.
- **Then reassign, and this is the part that is easy to get wrong.** After a split produces centroids `c1, c2`, only two centroids have changed: the two halves of the split posting. Every other posting centroid is untouched, and the index satisfied NPA before the split, so a vector owned by another posting `b` can only lose ownership to `c1` or `c2`. Select candidate postings with a bound that is *provably sufficient*, not by centroid proximity: keep a per-posting bounding radius `r_b` (the maximum distance from `b`'s centroid to a vector it owns) up to date on every ownership change, and include `b` only when the ball `(centroid_b, r_b)` can contain a vector that `c1`/`c2` can beat — or, for the split posting's own vectors, when `b` can beat `c1`/`c2` inside the split posting's ball. Recompute the owner of each affected vector over the affected centroids and move it if the owner changed. Ranking postings by centroid distance ("the k nearest to the split") is **unsound**: a vector can sit far from its own stale centroid while a new centroid lands on top of it, so the split invalidates a vector in a posting that the k-nearest set never contains. `n_reassign_neighbours` is retained in `MaintainConfig` for config compatibility only; it no longer gates correctness.
- Promote both new centroids into the parent level. If the parent exceeds `branching_factor`, split it too and recurse upward. If the root splits, add a level.
- Record `splits`, `vectors_reassigned`, `npa_violations_detected`, and depth over time.

Deletes in both arms: add to the tombstone set. Out-of-place clears them at compaction; in-place rebuilds a posting when its tombstone fraction exceeds `tombstone_rebuild_frac` (default 0.2).

---

## 7. Configuration

```python
@dataclass(frozen=True)
class IndexConfig:
    dim: int
    leaf_size: int = 800              # target vectors per posting
    branching_factor: int = 16
    depth: int | None = None          # None = derive from branching_factor
    delta: float | None = None        # None = no overlap
    rng_rule: bool = False
    seed: int = 0

@dataclass(frozen=True)
class QueryConfig:
    k: int = 10
    beam_widths: tuple[int, ...] = (8,)   # per level; last value repeats
    n_postings: int = 1                   # postings fetched per unit
    overfetch: float = 0.2
    global_beam: bool = False

@dataclass(frozen=True)
class MaintainConfig:
    strategy: Literal["inplace", "outofplace"]
    memtable_limit: int = 10_000
    compaction_threshold: int = 4         # m
    split_threshold: int = 1600           # 2 × leaf_size
    n_reassign_neighbours: int = 8
    tombstone_rebuild_frac: float = 0.2
```

Every config is frozen and serialised into each result row. A result that can't be traced to its exact config is not a result.

---

## 8. Metrics

```python
@dataclass
class Counters:
    distance_computations: dict[str, int]   # "routing", "scan"
    postings_read: int
    bytes_read: int
    vectors_scanned: int
    # maintenance
    splits: int
    vectors_reassigned: int
    compaction_events: int
    vectors_reclustered: int
    npa_violations_detected: int
```

Plus, computed per experiment: `recall@k`, realised `bloat`, posting size distribution (mean, p50, p99, max), current depth, unit count.

Counters reset per query for search metrics and accumulate across the run for maintenance metrics. Keep those two lifetimes clearly separate.

---

## 9. Tests

Tests assert **invariants**, not magic numbers. Every test uses a fixed seed.

`test_invariants.py` is the important one:

1. **Coverage** — every non-deleted vector appears in at least one posting's `members`.
2. **Ownership** — `owned ⊆ members` for every posting; every vector is `owned` by exactly one.
3. **NPA** — for every vector, the centroid of its owning posting is its nearest among all posting centroids. Must hold after every maintenance operation. Exact ties resolve to the lowest posting id (the same stable tie-break as `descend`) and are accepted as valid within the auditor's tolerance. *This is the test that catches the local-only-reassignment bug.*
4. **Beam monotonicity** — recall@10 is non-decreasing in beam width, all else fixed.
5. **Probe monotonicity** — recall@10 is non-decreasing in `n_postings`.
6. **Overlap bound** — with the top-2 rule, no vector appears in more than 2 postings.
7. **Determinism** — same seed plus same config produces identical counters across two runs in the same process *and* across two processes.
8. **Tombstones** — deleted vectors never appear in results.
9. **Depth-1 equivalence** — at depth 1 with `beam_widths=(n_leaves,)`, descent returns the same posting as brute-force nearest-centroid.
10. **Exhaustive-search sanity** — `n_postings = n_leaves` with full beam gives recall@10 = 1.0.

Invariants 3 and 10 are the ones that catch real bugs. Write them first.

---

## 10. CLI

Exactly two entry points:

```bash
python -m vecstudy.harness.runner sweep --config sweeps/depth_study.yaml --out results/
python -m vecstudy.harness.runner single --dataset sift1m --strategy inplace \
    --depth 3 --beam 8 --delta 0.1 --out results/
```

Each result row carries: full config, all counters, recall, the git commit SHA, timestamp, and the dataset checksum. Write Parquet, or SQLite if Parquet is awkward.

## 11. Build order

Work in this order, with a checkpoint at each stage. Do not start a stage until the previous one's tests pass.

| Stage | Deliverable | Checkpoint |
|---|---|---|
| 1 | `core/` — types, config, counters, distance, clustering | `test_distance.py` passes; counters match hand-computed values on a 10-vector toy set |
| 2 | `index/hierarchy.py` | Invariant 9; beam monotonicity on synthetic data |
| 3 | `index/postings.py` | Invariants 1, 2, 6; bloat matches a hand-computed value on a toy set |
| 4 | `index/query.py` + `index/segment.py` | Invariant 10; recall@10 = 1.0 under exhaustive search |
| 5 | `harness/` — datasets, ground truth, runner, results | End-to-end run on synthetic data producing one result row |
| 6 | `maintain/outofplace.py` | Static recall matches a directly-built index; compaction preserves recall |
| 7 | `maintain/inplace.py` | **Invariant 3 after 10,000 inserts with splits** |
| 8 | Sweep configs and reproduction script | Full depth × strategy grid runs to completion |

Stage 7 is where the difficulty is concentrated. Budget accordingly, and do not begin it before stage 6 is green.

**Datasets:** support SIFT1M and GIST1M in `.fvecs`/`.bvecs`, plus a synthetic generator (anisotropic mixture: topic centres, decaying spectrum, L2-normalised) for tests. Ground truth by exact brute force, cached to disk keyed by dataset checksum.

---

## 12. Why Python, and what comes later

This implementation exists to produce results, not throughput. NumPy's inner products call into BLAS and are already near-optimal for the scan; a first-pass C++ implementation would likely be *slower*. More importantly, the primary metrics are counters, which are identical in any language, so C++ buys nothing on the result and costs weeks of debugging.

A C++ port is planned separately and will be validated by **exact counter equality** against this implementation on fixed seeds. That is the main reason determinism is a hard requirement here: this codebase is the oracle.

---

## 13. When to stop and ask

Flag rather than guessing if:

- A requirement in §2 appears to conflict with something else in this document
- The NPA invariant cannot be maintained under some maintenance path
- A dataset loader needs network access (the sandbox may block it)
- Balanced k-means fails to converge within its round cap on real data
- You believe something in §3's non-goals is genuinely required

Do not resolve any of these by relaxing an invariant or by adding a strategy-specific branch in shared code. Both destroy the experiment, and neither failure is visible in the output.
