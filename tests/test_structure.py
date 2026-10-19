"""test_structure.py — structural maintenance instrumentation tests.

Covers the pure summarizers (live-only posting sizes, replica separation, strict
over-threshold fraction, empty-structure safety), out-of-place aggregation with
a separate MemTable, interval/cumulative event accounting, checkpoint-0 and N/A
conventions, the reentrancy-guarded maintenance timer, snapshot immutability,
behaviour preservation versus the vanilla maintainers, output schema/uniqueness,
and an end-to-end structure-CSV + metadata smoke.
"""

from __future__ import annotations

import csv
import json
import time

import numpy as np
import pytest

from vecstudy.core.metrics import Counters
from vecstudy.core.types import Posting
from vecstudy.harness.churn import generate_churn_trace
from vecstudy.harness.compare import (
    CompareConfig,
    _DatasetCache,
    _index_config,
    _maintain_config,
    write_csv,
    write_parquet,
)
from vecstudy.harness.search_budget import build_parser, resolve_spec, run_experiment
from vecstudy.harness.structure import (
    _MaintenanceTimerMixin,
    build_structure_row,
    event_fields,
    inplace_structure,
    make_instrumented_maintainer,
    member_size_stats,
    outofplace_structure,
    posting_size_stats,
    structure_snapshot,
    timing_fields,
)
from vecstudy.maintain.inplace import InPlaceMaintainer
from vecstudy.maintain.outofplace import OutOfPlaceMaintainer


# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------

def _posting(pid, owned, members, dim=2):
    return Posting(
        id=pid,
        centroid=np.zeros(dim, dtype=np.float32),
        members=np.asarray(members, dtype=np.int64),
        owned=np.asarray(owned, dtype=np.int64),
    )


class _FakeSeg:
    def __init__(self, postings):
        self.postings = postings


class _FakeInPlace:
    def __init__(self, postings):
        self._segment = _FakeSeg(postings)

    def searchable_units(self):
        return [self._segment]


class _FakeOOP:
    def __init__(self, sealed, memtable=()):
        self._sealed = sealed
        self._memtable = [int(v) for v in memtable]

    def searchable_units(self):
        return list(self._sealed)


def _tiny_cfg(**over) -> CompareConfig:
    base = dict(
        dataset="synthetic", dim=8, initial_size=80, reserve_size=40,
        operations=40, workload="mixed", insert_prob=0.5, seed=0, depth=2,
        leaf_size=20, branching_factor=4, split_threshold=24,
        merge_threshold=None, delta=None, memtable_limit=30,
        compaction_threshold=3, k=5, recall_ks=(1, 5), query_count=6,
        n_postings=1, beam_widths=(1,),
    )
    base.update(over)
    return CompareConfig(**base)


def _replay(cfg, strategy, *, instrumented):
    cache = _DatasetCache()
    vectors = cache.data(cfg)
    seg = cache.initial_segment(cfg)
    idx = _index_config(cfg)
    mcfg = _maintain_config(cfg, strategy)
    counters = Counters()
    if instrumented:
        maint = make_instrumented_maintainer(strategy, vectors, idx, mcfg,
                                             counters, seg)
    else:
        cls = InPlaceMaintainer if strategy == "inplace" else OutOfPlaceMaintainer
        maint = cls(vectors, idx, mcfg, counters, initial_segment=seg.clone())
    trace = generate_churn_trace(
        initial_n=cfg.initial_size, reserve_n=cfg.reserve_size,
        n_operations=cfg.operations, insert_prob=cfg.insert_prob,
        seed=cfg.seed, workload=cfg.workload,
    )
    for op_type, vid in trace.operations:
        if op_type == "insert":
            maint.insert(int(vid), vectors[int(vid)])
        else:
            maint.delete(int(vid))
        maint.maintain()
    return maint, counters


# ---------------------------------------------------------------------------
# 1. posting-size summary + strict threshold
# ---------------------------------------------------------------------------

def test_posting_size_stats_distribution_and_fractions():
    stats = posting_size_stats([10, 11, 3, 12], split_threshold=10)
    assert stats["live_posting_count"] == 4
    assert stats["posting_size_median"] == 10.5
    assert stats["posting_size_max"] == 12.0
    # numpy linear interpolation
    assert abs(stats["posting_size_p90"]
               - float(np.percentile([10, 11, 3, 12], 90))) < 1e-12
    # strict > : 11 and 12 are over, 10 is not
    assert stats["fraction_postings_over_split_threshold"] == 0.5
    assert abs(stats["fraction_vectors_in_overthreshold_postings"]
               - (23.0 / 36.0)) < 1e-12


def test_strict_over_threshold_comparison():
    assert posting_size_stats([10, 10], 10)["fraction_postings_over_split_threshold"] == 0.0
    assert posting_size_stats([11], 10)["fraction_postings_over_split_threshold"] == 1.0


def test_empty_structure_has_nan_sizes_and_zero_counts():
    stats = posting_size_stats([], 10)
    assert stats["live_posting_count"] == 0
    assert stats["fraction_postings_over_split_threshold"] == 0.0
    assert stats["fraction_vectors_in_overthreshold_postings"] == 0.0
    for k in ("posting_size_median", "posting_size_p90", "posting_size_p99",
              "posting_size_max"):
        assert np.isnan(stats[k])
    ms = member_size_stats([])
    assert ms["member_list_size_count"] == 0
    assert np.isnan(ms["member_list_size_median"])


# ---------------------------------------------------------------------------
# 2. live-only canonical sizes; replicas counted once, reported separately
# ---------------------------------------------------------------------------

def test_live_only_canonical_sizes_and_replicas():
    # vector 3 is owned by p1 and replicated into p0's member list.
    postings = [
        _posting(0, owned=[0, 1, 2], members=[0, 1, 2, 3]),
        _posting(1, owned=[3], members=[3]),
    ]
    maint = _FakeInPlace(postings)
    s = inplace_structure(maint, np.ones(4, dtype=bool), split_threshold=2)
    # canonical sizes: [3, 1]; the replica in p0 does NOT inflate posting sizes
    assert s["live_posting_count"] == 2
    assert s["posting_size_max"] == 3.0
    # member lists include the replica in p0 -> max 4
    assert s["member_list_size_max"] == 4.0


def test_tombstoned_vectors_excluded_and_empty_posting_dropped():
    postings = [
        _posting(0, owned=[0, 1, 2], members=[0, 1, 2, 3]),
        _posting(1, owned=[3], members=[3]),
    ]
    maint = _FakeInPlace(postings)
    live = np.array([True, True, True, False])   # vector 3 deleted
    s = inplace_structure(maint, live, split_threshold=2)
    assert s["live_posting_count"] == 1          # p1 is now empty, excluded
    assert s["posting_size_max"] == 3.0
    assert s["member_list_size_max"] == 3.0      # replica of 3 gone too


# ---------------------------------------------------------------------------
# 3. out-of-place aggregation + separate MemTable
# ---------------------------------------------------------------------------

def test_outofplace_sealed_aggregation_and_memtable_separate():
    sealed = [
        _FakeSeg([_posting(0, [0, 1], [0, 1]), _posting(1, [2], [2])]),
        _FakeSeg([_posting(0, [3, 4, 5], [3, 4, 5])]),
    ]
    maint = _FakeOOP(sealed, memtable=[6, 7])
    live = np.ones(8, dtype=bool)
    s = outofplace_structure(maint, live, split_threshold=2)
    # sizes across BOTH sealed segments: [2,1,3]; memtable not a posting
    assert s["live_posting_count"] == 3
    assert s["posting_size_max"] == 3.0
    assert s["sealed_segment_count"] == 2
    assert s["memtable_live_vector_count"] == 2


def test_outofplace_memtable_tombstones_excluded():
    maint = _FakeOOP([], memtable=[0, 1, 2])
    live = np.array([True, False, True])
    s = outofplace_structure(maint, live, split_threshold=2)
    assert s["live_posting_count"] == 0
    assert s["memtable_live_vector_count"] == 2


# ---------------------------------------------------------------------------
# 4. interval vs cumulative event counters
# ---------------------------------------------------------------------------

def _cum(splits=0, merges=0, flushes=0, compactions=0, reclustered=0):
    return {"splits": splits, "merges": merges, "flushes": flushes,
            "compaction_events": compactions, "vectors_reclustered": reclustered}


def test_interval_counters_are_deltas_and_never_reset():
    c1 = _cum(splits=2)
    c2 = _cum(splits=5)
    c3 = _cum(splits=5)
    e1 = event_fields(strategy="inplace", cumulative=c1,
                      interval={"splits": 2}, pruned_cumulative=0,
                      pruned_interval=0)
    e2 = event_fields(strategy="inplace", cumulative=c2,
                      interval=_delta(c1, c2), pruned_cumulative=0,
                      pruned_interval=0)
    e3 = event_fields(strategy="inplace", cumulative=c3,
                      interval=_delta(c2, c3), pruned_cumulative=0,
                      pruned_interval=0)
    assert (e1["splits_interval"], e1["splits_cumulative"]) == (2, 2)
    assert (e2["splits_interval"], e2["splits_cumulative"]) == (3, 5)
    assert (e3["splits_interval"], e3["splits_cumulative"]) == (0, 5)
    # cumulative is non-decreasing and the source dicts are untouched
    assert c1 == _cum(splits=2) and c2 == _cum(splits=5)


def _delta(before, after):
    return {k: after[k] - before[k] for k in after}


def test_cumulative_monotonicity():
    cums = [0, 1, 1, 4, 4]
    vals = []
    prev = None
    for c in cums:
        e = event_fields(strategy="inplace", cumulative=_cum(splits=c),
                         interval={"splits": c}, pruned_cumulative=0,
                         pruned_interval=0)
        vals.append(e["splits_cumulative"])
        if prev is not None:
            assert e["splits_cumulative"] >= prev
        prev = e["splits_cumulative"]
    assert vals == cums


# ---------------------------------------------------------------------------
# 5. checkpoint-0 + N/A conventions
# ---------------------------------------------------------------------------

def test_checkpoint_zero_interval_and_timing_are_zero():
    e = event_fields(strategy="inplace", cumulative=_cum(splits=3),
                     interval={"splits": 0}, pruned_cumulative=0, pruned_interval=0)
    assert e["splits_interval"] == 0
    t = timing_fields(maintenance_ns=12345, prev_maintenance_ns=12345,
                      update_ns=99, prev_update_ns=99)
    assert t["maintenance_wall_ms_interval"] == 0.0
    assert t["update_wall_ms_interval"] == 0.0
    assert t["maintenance_wall_ms_cumulative"] > 0.0


def test_na_is_none_and_zero_is_zero():
    e = event_fields(strategy="outofplace", cumulative=_cum(flushes=1),
                     interval={"flushes": 1}, pruned_cumulative=None,
                     pruned_interval=None)
    # in-place-only events are N/A for out-of-place ...
    assert e["splits_interval"] is None
    assert e["merges_cumulative"] is None
    assert e["pruned_postings_interval"] is None
    # ... while applicable-but-zero stays 0
    assert e["compactions_interval"] == 0
    assert e["flushes_interval"] == 1


# ---------------------------------------------------------------------------
# 6. timer: no double counting, isolation from query work
# ---------------------------------------------------------------------------

class _NestedTimed(_MaintenanceTimerMixin):
    def outer(self):
        self._timer_enter()
        try:
            self.inner()
        finally:
            self._timer_exit()

    def inner(self):
        self._timer_enter()
        try:
            time.sleep(0.02)
        finally:
            self._timer_exit()


def test_nested_hooks_not_double_counted():
    t = _NestedTimed()
    t._timer_init()
    t.outer()
    assert t._maint_depth == 0
    # one 20 ms sleep, not 40 ms
    assert 0.015e9 <= t.maintenance_ns <= 0.05e9


class _SingleTimed(_MaintenanceTimerMixin):
    def work(self):
        self._timer_enter()
        try:
            time.sleep(0.02)
        finally:
            self._timer_exit()


def test_timer_excludes_non_maintenance_work():
    t = _SingleTimed()
    t._timer_init()
    time.sleep(0.03)     # e.g. query / ground-truth work outside the timer
    t.work()
    assert 0.015e9 <= t.maintenance_ns <= 0.05e9   # only the maintenance sleep


# ---------------------------------------------------------------------------
# 7. observational snapshots do not mutate state
# ---------------------------------------------------------------------------

def test_structure_snapshot_does_not_mutate_postings():
    p0 = _posting(0, [0, 1, 2], [0, 1, 2, 3])
    p1 = _posting(1, [3], [3])
    maint = _FakeInPlace([p0, p1])
    before = (p0.owned.copy(), p0.members.copy(), p1.owned.copy(), p1.members.copy())
    s1 = structure_snapshot(strategy="inplace", maint=maint, live=range(4),
                            n_total=4, split_threshold=2)
    s2 = structure_snapshot(strategy="inplace", maint=maint, live=range(4),
                            n_total=4, split_threshold=2)
    assert s1 == s2
    assert (p0.owned == before[0]).all() and (p0.members == before[1]).all()
    assert (p1.owned == before[2]).all() and (p1.members == before[3]).all()


def test_instrumentation_preserves_state_and_counters():
    cfg = _tiny_cfg(depth=2, split_threshold=20)
    for strategy in ("inplace", "outofplace"):
        vanilla, cv = _replay(cfg, strategy, instrumented=False)
        instr, ci = _replay(cfg, strategy, instrumented=True)
        assert vanilla.stats()["n_postings"] == instr.stats()["n_postings"]
        assert cv.maintain_dict() == ci.maintain_dict()
        for a, b in zip(vanilla.searchable_units(), instr.searchable_units()):
            for pa, pb in zip(a.postings, b.postings):
                assert np.array_equal(pa.owned, pb.owned)
                assert np.array_equal(pa.members, pb.members)
    # instrumented objects expose the maintenance timer / prune counter
    instr, _ = _replay(cfg, "inplace", instrumented=True)
    assert instr.maintenance_ns > 0
    assert instr.pruned_postings >= 0


# ---------------------------------------------------------------------------
# 8. output schema, uniqueness, N/A-vs-zero in files
# ---------------------------------------------------------------------------

def _row(strategy, checkpoint, splits_iv, splits_cum, flushed_iv, flushed_cum,
         pruned_iv, pruned_cum, run_index=0):
    events = event_fields(
        strategy=strategy,
        cumulative=_cum(splits=splits_cum, flushes=flushed_cum),
        interval={"splits": splits_iv, "flushes": flushed_iv},
        pruned_cumulative=pruned_cum, pruned_interval=pruned_iv,
    )
    return build_structure_row(
        identity={"dataset": "synthetic", "dim": 8, "seed": 0,
                  "depth_config": 2, "split_threshold": 24, "delta": -1},
        experiment_id="t", run_index=run_index, strategy=strategy,
        checkpoint=checkpoint, operations_processed=checkpoint * 20,
        live_vector_count=100, structure=posting_size_stats([5, 6], 24),
        events=events,
        timing=timing_fields(maintenance_ns=checkpoint * 1000,
                             prev_maintenance_ns=(checkpoint - 1) * 1000,
                             update_ns=checkpoint * 2000,
                             prev_update_ns=(checkpoint - 1) * 2000),
        correctness_status="PASSED", checkpoint_correct=True,
    )


def test_csv_schema_uniqueness_and_na_vs_zero(tmp_path):
    rows = [
        _row("inplace", 0, 0, 0, 0, 0, 0, 0),        # cp0 intervals all zero
        _row("inplace", 1, 2, 2, None, 0, 0, 0),     # oop-only events N/A
        _row("outofplace", 1, None, 0, 3, 3, None, 0),
    ]
    path = tmp_path / "results_structure_checkpoints.csv"
    write_csv(path, rows)
    back = list(csv.DictReader(open(path)))
    assert len(back) == 3
    assert "splits_interval" in back[0] and "maintenance_wall_ms_cumulative" in back[0]
    # unique join key
    keys = [(r["strategy"], r["depth_config"], r["split_threshold"], r["seed"],
             r["checkpoint"], r["run_index"]) for r in back]
    assert len(set(keys)) == len(keys)
    # N/A -> empty string; applicable-but-zero -> "0"; applicable non-zero -> "3"
    assert back[2]["splits_interval"] == ""
    assert back[1]["flushes_interval"] == ""
    assert back[0]["splits_interval"] == "0"
    assert back[2]["flushes_interval"] == "3"


def test_parquet_written_with_null_na(tmp_path):
    rows = [
        _row("inplace", 1, 2, 2, None, 0, 0, 0),
        _row("outofplace", 1, None, 0, 3, 3, None, 0),
    ]
    path = tmp_path / "results_structure_checkpoints.parquet"
    if not write_parquet(path, rows):
        pytest.skip("pyarrow unavailable")
    import pyarrow.parquet as pq
    table = pq.read_table(path)
    d = table.to_pydict()
    # out-of-place row is index 1: in-place-only splits column is null there
    assert d["splits_interval"][1] is None
    assert d["flushes_interval"][1] == 3


# ---------------------------------------------------------------------------
# 9. end-to-end CLI smoke: structure CSV + metadata, correctness preserved
# ---------------------------------------------------------------------------

def _cli(*args):
    return build_parser().parse_args(list(args))


def test_cli_smoke_writes_structure_table_and_metadata(tmp_path):
    out = tmp_path / "cal"
    args = [
        "--phase", "calibration", "--initial-size", "120", "--operations", "60",
        "--query-count", "8", "--checkpoint-interval", "20",
        "--depths", "1,2", "--split-thresholds", "24", "--deltas", "none",
        "--beam-width-values", "2", "--n-postings-values", "1",
        "--calibration-seeds", "0", "--blas-threads", "1",
        "--out", str(out), "--overwrite",
    ]
    written = run_experiment(resolve_spec(_cli(*args)))
    assert "results_structure_checkpoints.csv" in written
    csv_path = out / "results_structure_checkpoints.csv"
    assert csv_path.exists()
    rows = list(csv.DictReader(open(csv_path)))
    # one row per (strategy, base config, seed, checkpoint)
    assert rows
    keys = [(r["strategy"], r["depth_config"], r["split_threshold"], r["delta"],
             r["seed"], r["checkpoint"], r["run_index"]) for r in rows]
    assert len(set(keys)) == len(keys) == len(rows)
    assert {r["strategy"] for r in rows} == {"inplace", "outofplace"}
    # checkpoint 0 intervals are zero; both strategies present and correct
    cp0 = [r for r in rows if r["checkpoint"] == "0"]
    assert cp0 and all(r["splits_interval"] == "0" for r in cp0 if r["strategy"] == "inplace")
    assert all(r["checkpoint_correct"] in ("True", "False") for r in rows)

    meta = json.loads((out / "run_metadata.json").read_text())
    assert meta["n_structure_rows"] == len(rows)
    assert meta["structure_metrics"]["percentile_method"].startswith("numpy.percentile")
    assert meta["structure_metrics"]["na_convention"]
    assert "results_structure_checkpoints.csv" in meta["output_schema"]
    assert meta["immutability_violations"] == 0


# ---------------------------------------------------------------------------
# 10. structural diagnostics figure pipeline
# ---------------------------------------------------------------------------

def test_threshold_series_and_figure(tmp_path):
    from vecstudy.harness.structure_report import (
        load_structure_rows,
        plot_structure_diagnostics,
        threshold_series,
    )

    def obs(strategy, split, p50, p95, pmax, live):
        return {"strategy": strategy, "split_threshold": str(split),
                "posting_size_median": str(p50), "posting_size_p90": str(p95),
                "posting_size_p95": str(p95), "posting_size_max": str(pmax),
                "live_posting_count": str(live)}

    rows = [
        obs("inplace", 24, 5, 10, 12, 4),
        obs("inplace", 24, 7, 12, 14, 6),          # median of the two = 6/11/13
        obs("inplace", 48, 20, 30, 40, 3),
        obs("outofplace", 24, 9, 15, 18, 5),
    ]
    series = threshold_series(rows)
    by = {(s["strategy"], s["split_threshold"]): s for s in series}
    assert len(by) == 3
    assert by[("inplace", 24.0)]["posting_size_p50"] == 6.0
    assert by[("inplace", 24.0)]["posting_size_p95"] == 11.0
    assert by[("inplace", 24.0)]["posting_size_max"] == 13.0
    assert by[("inplace", 24.0)]["n_observations"] == 2
    assert by[("inplace", 48.0)]["posting_size_p50"] == 20.0
    assert by[("outofplace", 24.0)]["posting_size_p50"] == 9.0

    src = tmp_path / "results_structure_checkpoints.csv"
    write_csv(src, rows)
    assert load_structure_rows(src) == [
        {k: v for k, v in r.items()} for r in rows]  # round-trips
    res = plot_structure_diagnostics(src, tmp_path / "structure", make_figures=True)
    for name in ("structure_threshold_series.csv", "fig_structure_posting_sizes.png"):
        assert (tmp_path / "structure" / name).exists(), name
    assert res["series"] == series
