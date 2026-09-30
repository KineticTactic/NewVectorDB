"""test_invariants.py — all 10 structural invariants (DESIGN §9)."""

from __future__ import annotations
import subprocess, sys, textwrap, numpy as np, pytest

from vecstudy.core.config import IndexConfig, QueryConfig, MaintainConfig
from vecstudy.core.metrics import Counters
from vecstudy.harness.datasets import compute_ground_truth, recall_at_k
from vecstudy.index.segment import Segment
from vecstudy.index.query import search
from vecstudy.index.postings import assign
from vecstudy.maintain.inplace import InPlaceMaintainer


def _build(X, leaf_size=40, seed=0, delta=None, depth=None):
    N, D = X.shape
    cfg = IndexConfig(dim=D, leaf_size=leaf_size, branching_factor=4,
                      seed=seed, delta=delta, depth=depth)
    c = Counters()
    seg = Segment.build(np.arange(N, dtype=np.int64), X, cfg, c)
    return seg, cfg, c


def _check_npa(seg):
    lc = seg.levels[-1].centroids
    violations = []
    for p in seg.postings:
        for vid in p.owned:
            vi = int(vid)
            if vi in seg.tombstones:
                continue
            sims = lc @ seg.vectors[vi].astype(np.float32)
            best = int(np.argmax(sims))
            if best != p.id:
                violations.append(f"vid={vi} owned={p.id} nearest={best}")
    return violations


def test_inv1_coverage(synthetic_200):
    seg, _, _ = _build(synthetic_200)
    covered = {int(v) for p in seg.postings for v in p.members}
    assert covered == set(range(len(synthetic_200)))


def test_inv2_ownership(synthetic_200):
    seg, _, _ = _build(synthetic_200)
    owner: dict[int,int] = {}
    for p in seg.postings:
        assert set(int(v) for v in p.owned) <= set(int(v) for v in p.members)
        for v in p.owned:
            vid = int(v); owner[vid] = owner.get(vid, 0) + 1
    assert all(c == 1 for c in owner.values())


def test_inv3_npa_after_build(synthetic_200):
    seg, _, _ = _build(synthetic_200)
    assert not _check_npa(seg)


def test_inv3_npa_after_inplace_inserts(synthetic_1000):
    X = synthetic_1000; D = X.shape[1]
    cfg_idx = IndexConfig(dim=D, leaf_size=50, branching_factor=4, seed=0)
    cfg_m = MaintainConfig(strategy="inplace", split_threshold=100, n_reassign_neighbours=8)
    maint = InPlaceMaintainer(X, cfg_idx, cfg_m, Counters())
    for vid in range(len(X)):
        maint.insert(vid, X[vid])
    violations = _check_npa(maint.searchable_units()[0])
    assert not violations, f"NPA violations: {violations[:5]}"


def test_inv4_beam_monotonicity(synthetic_200):
    X = synthetic_200; seg, _, _ = _build(X, leaf_size=20)
    n_leaves = len(seg.postings)
    queries = X[:10]
    gt = compute_ground_truth(queries, X, k=10, dataset_key="inv4")
    prev = 0.0
    for bw in [1, 2, 4, 8, n_leaves]:
        cfg_q = QueryConfig(k=10, beam_widths=(bw,), n_postings=1)
        res = [[v for v,_ in search(q,[seg],cfg_q,Counters())] for q in queries]
        rc = recall_at_k(res, gt, 10)
        assert rc >= prev - 1e-9; prev = rc


def test_inv5_probe_monotonicity(synthetic_200):
    X = synthetic_200; seg, _, _ = _build(X, leaf_size=20)
    n_leaves = len(seg.postings)
    queries = X[:10]
    gt = compute_ground_truth(queries, X, k=10, dataset_key="inv5")
    prev = 0.0
    for np_ in [1, 2, 4, n_leaves]:
        cfg_q = QueryConfig(k=10, beam_widths=(n_leaves,), n_postings=np_)
        res = [[v for v,_ in search(q,[seg],cfg_q,Counters())] for q in queries]
        rc = recall_at_k(res, gt, 10)
        assert rc >= prev - 1e-9; prev = rc


def test_inv6_overlap_bound(synthetic_200):
    X = synthetic_200
    from sklearn.cluster import KMeans
    km = KMeans(n_clusters=5, random_state=0, n_init=1).fit(X)
    centroids = km.cluster_centers_.astype(np.float32)
    postings, _ = assign(X, centroids, delta=0.5, rng_rule=False, counters=Counters())
    count: dict[int,int] = {}
    for p in postings:
        for v in p.members: count[int(v)] = count.get(int(v),0)+1
    assert max(count.values()) <= 2


def test_inv7_determinism_inprocess(synthetic_200):
    X = synthetic_200; q = X[0]
    def run():
        seg, _, _ = _build(X, leaf_size=40, seed=0)
        cfg_q = QueryConfig(k=10, beam_widths=(4,), n_postings=2)
        c = Counters(); search(q,[seg],cfg_q,c)
        return c.distance_computations.copy()
    assert run() == run()


def test_inv7_determinism_subprocess():
    script = textwrap.dedent(r"""
        import os, sys, json
        os.environ["OMP_NUM_THREADS"]="1"; os.environ["MKL_NUM_THREADS"]="1"
        sys.path.insert(0, r'D:\projects\VectorDBResearch')
        import numpy as np
        from vecstudy.harness.datasets import make_synthetic
        from vecstudy.core.config import IndexConfig, QueryConfig
        from vecstudy.core.metrics import Counters
        from vecstudy.index.segment import Segment
        from vecstudy.index.query import search
        X = make_synthetic(n=200, dim=32, n_topics=5, seed=7)
        cfg = IndexConfig(dim=32, leaf_size=40, branching_factor=4, seed=0)
        seg = Segment.build(np.arange(len(X),dtype=np.int64), X, cfg, Counters())
        cfg_q = QueryConfig(k=10, beam_widths=(4,), n_postings=2)
        c2 = Counters(); search(X[0],[seg],cfg_q,c2)
        print(json.dumps(c2.distance_computations))
    """)
    venv_py = sys.executable
    results = []
    for _ in range(2):
        r = subprocess.run([venv_py,"-c",script], capture_output=True, text=True, timeout=60)
        assert r.returncode == 0, r.stderr
        import json; results.append(json.loads(r.stdout.strip()))
    assert results[0] == results[1]


def test_inv8_tombstones(synthetic_200):
    X = synthetic_200; seg, _, _ = _build(X, leaf_size=40)
    n_leaves = len(seg.postings); deleted = {0, 3, 7}
    for d in deleted: seg.mark_deleted(d)
    cfg_q = QueryConfig(k=10, beam_widths=(n_leaves,), n_postings=n_leaves)
    for q in X[20:30]:
        res = search(q,[seg],cfg_q,Counters())
        assert {v for v,_ in res}.isdisjoint(deleted)


def test_inv9_depth1_equivalence(synthetic_200):
    X = synthetic_200; seg, _, _ = _build(X, leaf_size=40, depth=1)
    assert len(seg.levels) == 1
    n_leaves = len(seg.postings)
    cfg_q = QueryConfig(k=1, beam_widths=(n_leaves,), n_postings=1)
    for q in X[:5]:
        res = search(q,[seg],cfg_q,Counters())
        assert len(res) > 0


def test_inv10_exhaustive_recall(synthetic_200):
    X = synthetic_200; seg, _, _ = _build(X, leaf_size=20)
    n_leaves = len(seg.postings)
    queries = X[:20]
    gt = compute_ground_truth(queries, X, k=10, dataset_key="inv10")
    cfg_q = QueryConfig(k=10, beam_widths=(n_leaves,), n_postings=n_leaves)
    results = [[v for v,_ in search(q,[seg],cfg_q,Counters())] for q in queries]
    rc = recall_at_k(results, gt, 10)
    assert rc == 1.0, f"Exhaustive recall = {rc:.4f}"
