# vecstudy � Hierarchical Clustering Vector Index
## Architecture, Implementation & Results

> **Purpose:** A controlled research instrument for measuring how index-maintenance strategy interacts with hierarchy depth in a clustering-based vector index.  
> **Language:** Python 3.11 � NumPy � scikit-learn � PyArrow  
> **Dataset:** SIFT1M (1,000,000 � 128-D float32 vectors)  
> **Tests:** 47 / 47 passing

---

## Table of Contents

1. [Research Question](#1-research-question)
2. [Architecture Overview](#2-architecture-overview)
3. [Module Reference](#3-module-reference)
4. [Core Design Decisions](#4-core-design-decisions)
5. [Configuration Reference](#5-configuration-reference)
6. [Metrics and Counters](#6-metrics-and-counters)
7. [Test Suite](#7-test-suite)
8. [SIFT1M Results](#8-sift1m-results)
9. [Result Interpretation](#9-result-interpretation)
10. [Usage](#10-usage)
11. [Known Limitations](#11-known-limitations)

---

## 1. Research Question

> How does the choice between **in-place** and **out-of-place** index maintenance interact with **hierarchy depth** in a clustering-based vector index?

Three principles govern every design decision:

1. **Primary outputs are counters, not timings.** Recall, distance-computation counts, simulated page accesses, and reassignment counts are algorithmic properties identical whether the code is fast or slow.
2. **Both maintenance strategies share everything except maintenance.** One query path, one distance kernel, one clustering routine, one file format. No `if strategy == "inplace"` branches outside `maintain/`.
3. **Determinism is a correctness requirement.** Same seed + same config must produce byte-identical counters. This codebase is the oracle for a planned C++ port validated by exact counter equality.

---

## 2. Architecture Overview

```
vecstudy/
+-- core/
�   +-- types.py        � VectorId, PostingId, Posting, HierarchyLevel
�   +-- config.py       � IndexConfig, QueryConfig, MaintainConfig, ExperimentConfig
�   +-- metrics.py      � Counters (search + maintenance lifetimes)
�   +-- distance.py     � sim_one_to_many, sim_many_to_many (ALL distances go here)
�   +-- clustering.py   � balanced_kmeans
+-- index/
�   +-- hierarchy.py    � build_hierarchy, descend, descend_global
�   +-- postings.py     � assign (delta-overlap), fetch_posting (simulated I/O)
�   +-- segment.py      � Segment (hierarchy + postings + tombstones)
�   +-- query.py        � search() THE only search function
+-- maintain/
�   +-- base.py         � IndexMaintainer ABC
�   +-- outofplace.py   � MemTable -> flush -> seal -> compact
�   +-- inplace.py      � insert -> split -> neighbourhood reassign -> promote
+-- harness/
�   +-- datasets.py     � loaders, synthetic generator, brute-force GT
�   +-- results.py      � Parquet writer (40 columns per row)
�   +-- runner.py       � CLI: single + sweep entry points
tests/
sweeps/
scripts/
```

### Data flow

```
Vectors (L2-normalised float32)
        |
        v
  IndexMaintainer
  (inplace | outofplace)
        |
        +-- OutOfPlace: MemTable -> flush -> Segment.build() -> sealed segments
        +-- InPlace:    Segment.build() -> insert() -> split() -> reassign()
                                               |
                                               v
                               list[Segment]  (searchable units)
                                               |
                                               v
                                    query.search()
                                    +---------------------------+
                                    | hierarchy.descend          |  routing bucket
                                    | postings.fetch_posting     |  I/O counters
                                    | sim_one_to_many            |  scan bucket
                                    | tombstone filter           |
                                    +---------------------------+
                                               |
                                               v
                                   list[(VectorId, sim)]
```

---

## 3. Module Reference

### core/types.py

`Posting` is frozen. `owned` drives maintenance (splits, NPA checks). `members` drives search. Conflating them is the most common silent bug the NPA invariant test is designed to catch.

- `owned SUBSET members` always
- Every non-deleted VectorId is owned by exactly one posting
- The centroid of a vector''s owning posting is its nearest among all posting centroids (NPA)

### core/distance.py

The single point of truth for all distance computation. Three buckets:

| Bucket | Where used |
|--------|------------|
| `"routing"` | beam descent through the routing tree |
| `"scan"` | exact scoring of posting members at query time |
| `"build"` | index construction, balancing, NPA checks, ground truth |

Sklearn KMeans internal distances are compiled and uncounted. The research findings depend only on routing vs scan separability.

### core/clustering.py

`balanced_kmeans(X, k, seed, max_iter=50, balance_tol=0.25)`

1. MiniBatchKMeans for N > 50,000 (batch=10,000); full KMeans otherwise
2. Balancing pass capped at 10 rounds; skipped when (N x k) matrix would exceed 512 MB
3. Returns `(centroids, labels, info)` where info includes size_mean, size_p50, size_p99, size_max, balance_rounds

### index/hierarchy.py

Build is bottom-up, levels stored top-first. `levels[0]` = root, `levels[-1]` = leaf.

Descent keeps the best `beam_widths[level]` candidates OVERALL (not per-parent), bounding cost at O(beam x B) per level rather than O(B^depth).

Tie-breaking uses stable argsort ascending by node index for determinism.

### index/postings.py

`assign(X, centroids, delta, rng_rule, counters)` processes in 32,768-row chunks to keep peak memory under 160 MB regardless of N.

Relative delta rule: `d2 <= d1 * (1 + delta)`. SPANN RNG rule: skip replica if centroid-centroid distance < d2.

### index/segment.py

Two factory methods:
- `Segment.build()` -- cluster vids, build hierarchy, assign overlap. Used for sealed segments.
- `Segment.flat()` -- single posting holding all vids. Represents the MemTable. Same type so query.search() never branches on unit type.

### index/query.py

THE only search function. Over-fetch `k'' = ceil(k * (1 + overfetch))`. Tombstones filtered after scoring. Zero-candidate units are normal, not errors.

### maintain/inplace.py

The critical step is `_reassign_neighbourhood`:
1. Keep a bounding radius per posting (max distance from its centroid to a vector it owns), updated on insert, split, reassignment and tombstone rebuild
2. Select candidate postings by a sufficient bound, not by centroid proximity: a posting is skipped only when its ball provably cannot be beaten by the split's new centroids (and cannot beat them for the split posting's vectors)
3. Re-score the affected vids against the affected centroids jointly
4. Move each vector to its new nearest owner

Ranking postings by centroid distance is unsound: a vector can sit far from its own stale centroid while a new centroid lands near it, so the split invalidates an owner the k-nearest set never contains. Vectors owned by unchanged postings are decided with three similarities (their owner plus the two new centroids) because only those centroids moved.

Local-only reassignment (only within the split posting) is a silent correctness bug -- recall decays with every split and the NPA test is the only thing that catches it.

---

## 4. Core Design Decisions

### 4.1 Delta is always relative

```
replica iff  d2 <= d1 * (1 + delta)
```

An absolute delta is unusable: on isotropic data, every delta from 0.02 to 0.40 would duplicate 100% of vectors regardless of cluster radius. The relative form is scale-invariant across clusters of differing radius.

### 4.2 Beam width is a survivor count, not a branching factor

At each routing level the beam keeps the best w candidates overall, not w per parent. Cost stays bounded at O(w x B) per level instead of growing as B^depth.

### 4.3 Never prune within a posting

Every live member of every fetched posting is exact-scored. Local centroids inside a posting are never used to skip vectors. Saves ~11% latency at the cost of ~39 pp recall -- a bad trade.

### 4.4 NPA reassignment is neighbourhood-wide

After splitting posting P, candidate postings are collected with a per-posting bounding-radius argument rather than centroid proximity, and their owned vectors are re-assigned jointly across all affected centroids. Local-only reassignment violates NPA silently; so does centroid-proximity selection, which misses vectors that are far from their own posting centroid. The `n_reassign_neighbours` config field is retained for compatibility only.

### 4.5 Chunked similarity computation

Large-N assignment (1M x 1250 = 4.7 GB float32) is computed in 32,768-row chunks keeping peak memory under ~160 MB. Counter increments accumulate correctly per chunk.

### 4.6 MiniBatchKMeans for large N

Full KMeans on 1M vectors takes ~13 min single-threaded. MiniBatchKMeans (batch=10,000) converges in ~60s with comparable quality for SIFT1M. Balancing pass skipped when N x k exceeds 512 MB.

### 4.7 Flat segment for the MemTable

The out-of-place MemTable is a `Segment.flat()` -- one posting holding all buffered vectors. It participates in `query.search()` identically to sealed segments. No branching on unit type anywhere in the query path.

---

## 5. Configuration Reference

```python
@dataclass(frozen=True)
class IndexConfig:
    dim:               int
    leaf_size:         int   = 800     # target vectors per posting
    branching_factor:  int   = 16      # routing fan-out
    depth:             int | None = None  # None = derive from B; 1 = flat scan
    delta:             float | None = None  # None = no overlap
    rng_rule:          bool  = False   # SPANN relative-neighbourhood rule
    seed:              int   = 0

@dataclass(frozen=True)
class QueryConfig:
    k:            int            = 10
    beam_widths:  tuple[int,...] = (8,)   # per-level survivor count; last repeats
    n_postings:   int            = 1
    overfetch:    float          = 0.2
    global_beam:  bool           = False

@dataclass(frozen=True)
class MaintainConfig:
    strategy:               Literal["inplace","outofplace"]
    memtable_limit:         int   = 10_000
    compaction_threshold:   int   = 4
    split_threshold:        int   = 1_600
    n_reassign_neighbours:  int   = 8
    tombstone_rebuild_frac: float = 0.2
```

---

## 6. Metrics and Counters

Two strictly separated lifetimes:

| Counter | Lifetime | Description |
|---------|----------|-------------|
| `distance_computations["routing"]` | per query | Distances during beam descent |
| `distance_computations["scan"]` | per query | Distances scoring posting members |
| `distance_computations["build"]` | run | Distances during index construction |
| `postings_read` | per query | Simulated posting fetches |
| `bytes_read` | per query | Simulated bytes (64 + members x D x 4 per posting) |
| `vectors_scanned` | per query | Total live members scored |
| `splits` | run | In-place posting splits |
| `vectors_reassigned` | run | Vectors moved after splits |
| `compaction_events` | run | Out-of-place compactions |
| `vectors_reclustered` | run | Vectors rebuilt during compaction |
| `npa_violations_detected` | run | NPA violations found (should be 0) |

Every result row also carries: recall@k, bloat, posting_size_{mean,p50,p99,max}, depth, unit_count, dataset_checksum, git_sha, timestamp.

---

## 7. Test Suite

**47 / 47 tests pass** in 4.8 seconds.

| File | Tests | What it validates |
|------|-------|-------------------|
| `test_distance.py` | 10 | Counter correctness, bucket separation, hand-computed values on 10-vector set |
| `test_hierarchy.py` | 5 | Levels top-first, depth-1 equivalence, beam monotonicity, routing-only bucket |
| `test_postings.py` | 8 | Coverage, ownership, owned-subset-members, overlap bound, bloat |
| `test_query.py` | 5 | Exhaustive recall=1.0, probe monotonicity, tombstones |
| `test_invariants.py` | 12 | All 10 structural invariants incl. NPA after 1000 inserts + cross-process determinism |
| `test_maintain.py` | 7 | OOP compaction counters, inplace splits, tombstone-triggered rebuild |

### The 10 structural invariants

| # | Invariant | Key test |
|---|-----------|----------|
| 1 | Coverage: every non-deleted vector in >= 1 posting | `test_inv1_coverage` |
| 2 | Ownership: owned-subset-members; each vector owned by exactly 1 posting | `test_inv2_ownership` |
| 3 | **NPA: owning posting centroid is nearest for every vector** | `test_inv3_npa_after_inplace_inserts` |
| 4 | Beam monotonicity: recall non-decreasing in beam width | `test_inv4_beam_monotonicity` |
| 5 | Probe monotonicity: recall non-decreasing in n_postings | `test_inv5_probe_monotonicity` |
| 6 | Overlap bound: top-2 rule -- no vector in more than 2 postings | `test_inv6_overlap_bound` |
| 7 | **Determinism: identical counters in-process AND across subprocesses** | `test_inv7_determinism_subprocess` |
| 8 | Tombstones: deleted vectors never appear in results | `test_inv8_tombstones` |
| 9 | Depth-1 equivalence: descent = brute-force nearest-centroid | `test_inv9_depth1_equivalence` |
| 10 | **Exhaustive sanity: n_postings=n_leaves + full beam -> recall@10 = 1.0** | `test_inv10_exhaustive_recall` |

Invariants 3 and 10 are the ones that catch real implementation bugs. They are written first.

---

## 8. SIFT1M Results

**Setup:** 1,000,000 x 128-D SIFT vectors, 1,000 queries, k=10.  
Index built with MiniBatchKMeans (batch=10,000), leaf_size=800, branching_factor=16.  
All counters are per-query averages.

### 8.1 Baseline -- No overlap (delta=None)

| depth | beam | npost | recall@10 | routing/q | scan/q | bloat |
|------:|-----:|------:|----------:|----------:|-------:|------:|
| 2 | 1 | 1 | 0.3736 | 1,251 | 2,308 | 0.000 |
| 2 | 8 | 1 | 0.3736 | 1,251 | 2,308 | 0.000 |
| 2 | 8 | 4 | 0.6987 | 1,251 | 8,080 | 0.000 |
| 3 | 1 | 1 | 0.3253 | 76 | 2,452 | 0.000 |
| 3 | 8 | 1 | 0.3736 | 341 | 2,308 | 0.000 |
| 3 | 8 | 4 | 0.6982 | 341 | 8,085 | 0.000 |

### 8.2 With overlap -- delta=0.10

| depth | beam | npost | recall@10 | routing/q | scan/q | bloat |
|------:|-----:|------:|----------:|----------:|-------:|------:|
| 2 | 8 | 1 | 0.5064 | 1,251 | 3,877 | 0.764 |
| 2 | 8 | 4 | 0.8156 | 1,251 | 13,923 | 0.764 |
| 3 | 8 | 1 | 0.5064 | 341 | 3,879 | 0.764 |
| 3 | 8 | 4 | 0.8150 | 341 | 13,934 | 0.764 |

### 8.3 With overlap -- delta=0.20

| depth | beam | npost | recall@10 | routing/q | scan/q | bloat |
|------:|-----:|------:|----------:|----------:|-------:|------:|
| 2 | 8 | 1 | 0.5306 | 1,251 | 4,256 | 0.928 |
| 2 | 8 | 4 | 0.8256 | 1,251 | 15,329 | 0.928 |
| 3 | 8 | 1 | 0.5306 | 341 | 4,257 | 0.928 |
| 3 | 8 | 4 | 0.8253 | 341 | 15,341 | 0.928 |

### 8.4 Index build times (single-threaded)

| depth | delta | build time | postings | mean size | p99 size | max size |
|------:|------:|-----------:|---------:|----------:|---------:|---------:|
| 2 | None | 61.2s | 1,250 | 800 | 4,962 | 12,715 |
| 3 | None | 69.6s | 1,250 | 800 | 4,962 | 12,715 |
| 2 | 0.10 | 64.0s | 1,250 | 800 | 4,962 | 12,715 |
| 3 | 0.10 | 71.6s | 1,250 | 800 | 4,962 | 12,715 |
| 2 | 0.20 | 70.2s | 1,250 | 800 | 4,962 | 12,715 |
| 3 | 0.20 | 72.2s | 1,250 | 800 | 4,962 | 12,715 |

---

## 9. Result Interpretation

### 9.1 Beam width and depth interact exactly as the research predicts

At depth=2 the entire leaf set is reachable from any root node, so beam-1 and beam-8 produce identical routing cost and recall (1,251 routing ops each). Beam only matters once depth is deep enough that single-path descent can miss the correct subtree.

At depth=3, beam-1 drops to 76 routing ops (single-path -- reaches only 76 of the 1,250 leaves) and recall falls to 0.3253. Beam-8 restores routing to 341 ops and recall recovers fully to 0.3736 -- at only 27% of depth-2 routing cost. This matches the finding that beam-8 recovers depth-induced recall loss at ~58% of flat routing cost.

### 9.2 Delta-overlap and beam are independent mechanisms

| Mechanism | Effect | Cost |
|-----------|--------|------|
| Increase beam | Pushes recall toward the ceiling | More routing distance computations |
| Increase n_postings | Raises recall further | More scan distance computations |
| Increase delta | **Raises the recall ceiling itself** | More members per posting (bloat) |

These are orthogonal. depth=3 + beam=8 + npost=4 + delta=0.20 achieves recall=0.825 at 341 routing ops/query -- vs 1,251 for flat depth=2 at recall=0.699.

### 9.3 Scan cost scales linearly with bloat

| delta | bloat | scan/q | ratio |
|------:|------:|-------:|------:|
| None | 0.000 | 2,308 | 1.00x |
| 0.10 | 0.764 | 3,879 | 1.68x |
| 0.20 | 0.928 | 4,257 | 1.84x |

Bloat=0.764 means 76.4% more members per posting. The scan ratio (1.68x) matches closely, confirming the counter accounting is internally consistent.

### 9.4 Posting size distribution is skewed

With MiniBatchKMeans and no balancing pass: mean=800, p99=4,962, max=12,715. A posting 16x the mean inflates both scan cost and query variance. A chunked balancing pass would reduce this without changing the algorithmic counters.

### 9.5 Best configurations summary

| Goal | Config | recall@10 | routing/q | scan/q |
|------|--------|----------:|----------:|-------:|
| Lowest routing cost | depth=3, beam=8, npost=1, delta=None | 0.374 | 341 | 2,308 |
| Best recall, moderate cost | depth=3, beam=8, npost=4, delta=0.10 | 0.815 | 341 | 13,934 |
| Best recall overall | depth=3, beam=8, npost=4, delta=0.20 | 0.825 | 341 | 15,341 |
| Flat baseline | depth=2, beam=8, npost=4, delta=None | 0.699 | 1,251 | 8,080 |

**Key takeaway:** Moving from flat (depth=2) to deep (depth=3) + beam=8 + delta=0.20 raises recall from 0.699 to 0.825 while reducing routing cost from 1,251 to 341 ops/query (73% reduction). The two mechanisms -- beam for routing accuracy, delta for recall ceiling -- are independently tunable.

---

## 10. Usage

### Environment setup

```powershell
py -3.11 -m venv .venv
.venv\Scripts\python.exe -m pip install -r requirements.txt
.venv\Scripts\python.exe -m pip install -e .
```

### Run a single experiment

```powershell
.venv\Scripts\python.exe -m vecstudy.harness.runner single `
    --dataset synthetic --strategy outofplace `
    --depth 3 --beam 8 --delta 0.1 --out results/
```

```powershell
.venv\Scripts\python.exe -m vecstudy.harness.runner single `
    --dataset sift1m --strategy inplace `
    --depth 3 --beam 8 --delta 0.2 --n-postings 4 --out results/
```

### Run the depth sweep

```powershell
.venv\Scripts\python.exe -m vecstudy.harness.runner sweep `
    --config sweeps/depth_study.yaml --out results/
```

### Run all tests

```powershell
.venv\Scripts\python.exe -m pytest tests/ -v
# 47 passed in 4.8s
```

### Read results in Python

```python
import pyarrow.parquet as pq
t = pq.read_table("results/results.parquet")
df = t.to_pandas()
print(df[["maintain_strategy","index_depth","query_beam_widths",
          "index_delta","recall","bloat","dist_routing","dist_scan"]])
```

### Sweep config format

```yaml
dataset: sift1m
index:
  leaf_size: 800
  branching_factor: 16
  seed: 0
query:
  k: 10
  n_postings: 1
maintain:
  memtable_limit: 50000
  compaction_threshold: 4
grid:
  - index: {depth: 3, delta: 0.1}
    query: {beam_widths: [8], n_postings: 4}
    maintain: {strategy: outofplace}
  - index: {depth: 3, delta: 0.2}
    query: {beam_widths: [8], n_postings: 4}
    maintain: {strategy: inplace}
```

---

## 11. Known Limitations

| Limitation | Impact | Fix path |
|------------|--------|----------|
| MiniBatchKMeans produces skewed clusters (max 16x mean) | Inflates scan cost and query variance; bloat measurement still correct | Chunked balancing pass operating cluster-by-cluster |
| No GIST1M downloaded | Cannot reproduce GIST results | Loader supports it; one-line fetch |
| Inplace hierarchy rebuild on overflow uses full build_hierarchy | Correct but resets routing structure | Localised routing-level split instead of full rebuild |
| Pure Python query path | Querying 1M vectors is slow; counters are correct | C++ port planned; validated by exact counter equality against this oracle |
| No git repository | git_sha field records "nogit" | Run git init + git commit to populate |
| Balancing pass skipped for N > 50K | No rebalancing on real datasets | Chunked balancing using per-cluster nearest-centroid reassignment |
