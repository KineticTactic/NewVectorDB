# VectorDB Research

A Python research implementation for studying how hierarchy depth interacts with in-place and out-of-place maintenance in a clustering-based vector index. It is an experimental reference implementation, not a production vector database.

## Requirements

- Python 3.11 or newer
- Windows, macOS, or Linux
- Internet access only when running experiments with the SIFT1M dataset for the first time

## Setup

From the repository root, create and activate a virtual environment, then install the dependencies and package:

```powershell
py -3.11 -m venv .venv
.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
python -m pip install -e .
```

If PowerShell blocks activation, use the environment's Python executable directly instead, for example `.venv\Scripts\python.exe -m pytest tests/ -v`.

## Run Tests

Run all automated unit and invariant tests:

```powershell
python -m pytest tests/ -v
```

## Run Experiments

### 1. Static Benchmarks

Run a single static experiment with the built-in synthetic dataset:

```powershell
python -m vecstudy.harness.runner single `
    --dataset synthetic --strategy outofplace `
    --depth 3 --beam 8 --delta 0.1 --out results/
```

Run the static depth sweep:

```powershell
python -m vecstudy.harness.runner sweep `
    --config sweeps/depth_study.yaml --out results/
```

### 2. Fair Churn Benchmark (Recall under Churn)

Run the eight-way depth study under dynamic churn:

```powershell
python -m vecstudy.harness.runner churn `
    --config sweeps/churn_depth_study.yaml `
    --out results/churn/
```

This compares `inplace` and `outofplace` strategies across initial hierarchy depths 1, 2, 3, and 4 under the **exact same initial corpus, index configuration, and deterministic ordered churn workload**.

#### Churn Configuration Schema (`sweeps/churn_depth_study.yaml`)

```yaml
dataset: synthetic           # dataset name ("synthetic", "sift1m", "siftsmall")
initial_size: 5000          # initial live corpus size (IDs 0..initial_size-1)
reserve_size: 5000          # reserve pool for future inserts (IDs initial_size..initial_size+reserve_size-1)

index:
  leaf_size: 200            # target vectors per leaf posting
  branching_factor: 4       # routing tree branching factor
  seed: 0                   # index construction seed

query:
  k: 10                     # top-k neighbors
  beam_widths: [8]          # beam width per hierarchy level
  n_postings: 1             # leaf postings fetched per searchable unit

maintain:
  memtable_limit: 500       # out-of-place: MemTable flush threshold
  compaction_threshold: 4   # out-of-place: sealed segment compaction threshold
  split_threshold: 400      # in-place: posting split threshold (2 * leaf_size)
  n_reassign_neighbours: 8  # in-place: legacy hint, kept for config compatibility
                            # (split reassignment selects candidates by a sound radius bound)
  tombstone_rebuild_frac: 0.2 # in-place: tombstone rebuild fraction

churn:
  operations: 2000          # total insert/delete operations to execute
  checkpoint_interval: 500  # interval between evaluation checkpoints (e.g. 0, 500, 1000, 1500, 2000)
  insert_prob: 0.6          # probability of insert vs delete operation
  seed: 42                  # seed for deterministic workload trace generation
  workload: "mixed"         # workload label ("mixed", "insert_only", etc.)
  flush_memtable_at_checkpoint: false # whether to force-flush MemTable at checkpoint boundaries

grid:
  - index: {depth: 1}
    maintain: {strategy: outofplace}
  - index: {depth: 2}
    maintain: {strategy: outofplace}
  - index: {depth: 3}
    maintain: {strategy: outofplace}
  - index: {depth: 4}
    maintain: {strategy: outofplace}
  - index: {depth: 1}
    maintain: {strategy: inplace}
  - index: {depth: 2}
    maintain: {strategy: inplace}
  - index: {depth: 3}
    maintain: {strategy: inplace}
  - index: {depth: 4}
    maintain: {strategy: inplace}
```

## Churn Experiment Design & Semantics

### Fair Initialization Contract
- Both `inplace` and `outofplace` arms start from the **exact same initial live vector IDs** (`0 .. initial_size-1`), vector embeddings, queries, seeds, and index parameters.
- The initial index is built once via `Segment.build()` and cloned independently (`Segment.clone()`) into each maintainer arm before churn begins.
- Neither arm benchmarks cold ingestion. Checkpoint zero records identical baseline metrics (recall, live vector count, segment count, actual depth, posting-size distribution, and counters) across both strategies.

### Deterministic Workload Trace
- The trace generator produces an explicit sequence of `("insert", vid)` and `("delete", vid)` operations from a fixed seed.
- Insert operations draw strictly from uninserted reserve IDs; delete operations target currently live IDs only; no live ID is deleted twice.
- Both arms replay the exact same operations in the exact same order.
- A deterministic `trace_hash` and `trace_seed` are recorded in every result row.

### Checkpoint, MemTable, Flush, and Compaction Semantics
- At every checkpoint, **all live vectors must be searchable**.
- For `outofplace`, `searchable_units()` exposes the pending MemTable as a flat single-posting segment (`Segment.flat`) so newly inserted vectors are searchable immediately without requiring forced flushes.
- If `flush_memtable_at_checkpoint: true` is configured, pending MemTable entries are sealed via the public `maint.flush()` method, recorded under `forced_checkpoint_flushes`.
- Compactions fire strictly when the number of sealed segments reaches `compaction_threshold`. Compactions are never forced merely to simulate in-place structure.

### Dynamic Ground Truth
- Query vectors remain fixed across all checkpoints.
- At each checkpoint, brute-force ground truth is computed dynamically against **only the currently live vectors**. Deleted vectors are excluded; newly inserted vectors are eligible.
- Row offsets returned by brute-force search are mapped back to stable global `VectorId`s.
- Ground truth is cached using a compound key of query hash, live vector IDs hash, live data checksum, and `k`.

### Correctness and Global NPA Auditing
- At each checkpoint, a global auditor validates:
  1. **Coverage:** All live vectors are present in searchable units; deleted vectors are excluded.
  2. **Ownership:** `owned ⊆ members`; every live vector has a unique canonical owning posting.
  3. **Nearest Partition Assumption (NPA):** For every live vector, the centroid of its owning posting is nearest among all leaf centroids.
- If any invariant fails, `correctness_status` is marked `FAILED` and recorded in the result row.
- `correctness_status` covers coverage, unique canonical ownership and NPA only. It does **not**
  validate Δ-overlap membership semantics (the `members` replica rule); that is a separate check,
  `churn.audit_membership`, which re-derives the expected `members` set for every live vector from
  the current leaf centroids, the canonical owner and the configured Δ (independently of the
  maintainer's helpers) and reports missing/extra/duplicate/dead memberships.
- In-place membership is correct at rest after every operation: insert adds the required replica
  immediately, every split re-derives membership for **all** live vectors (Δ fixed; no dynamic
  calibration, no RNG filtering), and delete removes the vector from every `members` array at once.
  Replica work is counted by `replicas_added`, `replicas_removed` and
  `membership_vectors_recomputed`, separate from `vectors_reassigned` and from the query counters;
  its distances land in the `maintenance` bucket.

## Results and Data

Experiment output is appended to `results/results.parquet` (or `results/churn/results.parquet`). Each result row captures 60 metrics including strategy, depths, checkpoint index, cumulative operations, current live vector count, recall@k, search distance computations, cumulative maintenance counters, checkpoint maintenance deltas, and correctness status.
