import os
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"
os.environ["OPENBLAS_NUM_THREADS"] = "1"
import sys, time
import numpy as np
sys.path.insert(0, r"D:\projects\VectorDBResearch")
from vecstudy.harness.datasets import read_fvecs, read_ivecs, l2_normalize, recall_at_k, fetch_sift
from vecstudy.core.config import IndexConfig, QueryConfig
from vecstudy.core.metrics import Counters
from vecstudy.index.segment import Segment
from vecstudy.index.query import search

N_QUERIES = 1000
K = 10
LEAF_SIZE = 800
BRANCH = 16
SEED = 0

# Grid: (depth, beam, n_postings, delta)
# delta=None  -> no overlap (baseline)
# delta=0.10  -> 10% relative threshold
# delta=0.20  -> 20% relative threshold (usually sweet spot)
GRID = [
    # --- baseline: no overlap ---
    (2, 1, 1, None),
    (2, 8, 1, None),
    (2, 8, 4, None),
    (3, 1, 1, None),
    (3, 8, 1, None),
    (3, 8, 4, None),
    # --- overlap delta=0.10 ---
    (2, 8, 1, 0.10),
    (2, 8, 4, 0.10),
    (3, 8, 1, 0.10),
    (3, 8, 4, 0.10),
    # --- overlap delta=0.20 ---
    (2, 8, 1, 0.20),
    (2, 8, 4, 0.20),
    (3, 8, 1, 0.20),
    (3, 8, 4, 0.20),
]

print("Loading SIFT1M ...", flush=True)
sift_dir = fetch_sift()
base    = l2_normalize(read_fvecs(sift_dir / "sift_base.fvecs"))
queries = l2_normalize(read_fvecs(sift_dir / "sift_query.fvecs"))[:N_QUERIES]
gt      = read_ivecs(sift_dir / "sift_groundtruth.ivecs")[:N_QUERIES]
print(f"base {base.shape}  queries {queries.shape}  gt {gt.shape}", flush=True)

_seg_cache = {}

def get_segment(depth, delta):
    key = (depth, delta)
    if key in _seg_cache:
        return _seg_cache[key]
    cfg = IndexConfig(dim=128, leaf_size=LEAF_SIZE, branching_factor=BRANCH,
                      depth=depth, delta=delta, seed=SEED)
    delta_str = f"{delta:.2f}" if delta is not None else "None"
    print(f"\nBuilding segment depth={depth} delta={delta_str} ...", flush=True)
    t0 = time.time()
    c = Counters()
    vids = np.arange(len(base), dtype=np.int64)
    seg = Segment.build(vids, base, cfg, c)
    elapsed = time.time() - t0
    sizes = np.array([len(p.owned) for p in seg.postings])
    total_members = sum(len(p.members) for p in seg.postings)
    bloat = total_members / len(base) - 1.0
    print(f"  depth={depth} delta={delta_str}  postings={len(seg.postings)}"
          f"  size mean={sizes.mean():.0f} p99={np.percentile(sizes,99):.0f} max={sizes.max()}"
          f"  bloat={bloat:.3f}  built in {elapsed:.1f}s", flush=True)
    _seg_cache[key] = seg
    return seg

def run_queries(seg, beam, n_postings):
    cfg_q = QueryConfig(k=K, beam_widths=(beam,), n_postings=n_postings)
    results = []
    tot_routing = tot_scan = 0
    for q in queries:
        c = Counters()
        res = search(q, [seg], cfg_q, c)
        results.append([v for v, _ in res])
        tot_routing += c.distance_computations.get("routing", 0)
        tot_scan    += c.distance_computations.get("scan", 0)
    rc = recall_at_k(results, gt, K)
    return rc, tot_routing // N_QUERIES, tot_scan // N_QUERIES

print()
print("="*80)
print(f"{'depth':>5} {'beam':>5} {'npost':>5} {'delta':>6}  {'recall@10':>10}  {'routing/q':>10}  {'scan/q':>10}")
print("="*80, flush=True)

for depth, beam, n_post, delta in GRID:
    seg = get_segment(depth, delta)
    rc, r_dist, s_dist = run_queries(seg, beam, n_post)
    delta_str = f"{delta:.2f}" if delta is not None else "  None"
    print(f"{depth:>5} {beam:>5} {n_post:>5} {delta_str:>6}  {rc:>10.4f}  {r_dist:>10}  {s_dist:>10}", flush=True)

print("="*80)
print("Done.", flush=True)
