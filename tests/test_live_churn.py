"""test_live_churn.py — continuous live-churn monitor (``harness.live``).

These tests never need a human watching a terminal: they drive the monitor's
public helpers directly, capture output into a ``StringIO``, and use small
synthetic vectors.  They cover the requirements that matter for correctness of
the deliverable:

* recall is computed over the *current live* vectors only, and recomputed;
* a fresh-id insert extends the backing store and is retrievable;
* ids are monotonic and never reused;
* every operation yields one recall line;
* splits/merges are reported from real maintenance activity;
* the loop runs past any initial pool and stops only when told to;
* Ctrl+C (KeyboardInterrupt) prints a final summary and leaves the index valid;
* audit failures are visible and stop the run by default;
* a fixed seed reproduces the operation choices.
"""

from __future__ import annotations

import io

import numpy as np
import pytest

from vecstudy.core.metrics import Counters
from vecstudy.harness import live as livemod
from vecstudy.harness.churn import (
    AuditResult,
    audit_maintainer,
    audit_membership,
)
from vecstudy.harness.live import (
    GrowingVectorStore,
    LiveChurnConfig,
    LiveChurnMonitor,
    OperationRecord,
    ObservedInPlaceMaintainer,
)
from vecstudy.index.query import search


# ---------------------------------------------------------------------------
# Fixtures / helpers
# ---------------------------------------------------------------------------

def _cfg(**over) -> LiveChurnConfig:
    base = dict(
        initial_size=24, k=3, query_count=5, dim=8, depth=1,
        leaf_size=4, split_threshold=4, merge_threshold=None, delta=None,
        interval=0.0, seed=0, audit_every=1, summary_every=1000,
    )
    base.update(over)
    return LiveChurnConfig(**base)


def _monitor(**over) -> LiveChurnMonitor:
    return LiveChurnMonitor(_cfg(**over), out=io.StringIO())


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

def test_config_validation_rejects_bad_values():
    for bad in (
        dict(initial_size=0),
        dict(k=0),
        dict(query_count=0),
        dict(dim=1),
        dict(leaf_size=0),
        dict(split_threshold=0),
        dict(interval=-0.1),
        dict(audit_every=-1),
        dict(dataset="sift1m"),
    ):
        with pytest.raises(ValueError):
            LiveChurnConfig(**{**_cfg().to_dict(), **bad})


def test_cli_parse_and_resolve():
    args = livemod.build_parser().parse_args(
        ["--initial-size", "100", "--delta", "off", "--split-threshold", "25",
         "--leaf-size", "10", "--k", "7", "--seed", "3"]
    )
    cfg = livemod.resolve_config(args)
    assert cfg.initial_size == 100
    assert cfg.delta is None
    assert cfg.split_threshold == 25
    assert cfg.leaf_size == 10
    assert cfg.k == 7 and cfg.seed == 3


def test_cli_defaults_split_threshold_follows_leaf_size():
    cfg = livemod.resolve_config(livemod.build_parser().parse_args([]))
    assert cfg.delta is None
    assert cfg.split_threshold == cfg.leaf_size
    cfg2 = livemod.resolve_config(
        livemod.build_parser().parse_args(["--delta", "0.3"])
    )
    assert cfg2.delta == pytest.approx(0.3)


# ---------------------------------------------------------------------------
# Growing backing store
# ---------------------------------------------------------------------------

def test_store_growth_preserves_data():
    store = GrowingVectorStore(np.eye(2, dtype=np.float32))
    assert store.size == 2 and store.capacity == 2
    grows_before = store.n_grows
    for i in range(12):
        assert store.append(np.full(2, float(i), dtype=np.float32)) == 2 + i
    assert store.size == 14
    assert store.n_grows > grows_before
    assert np.allclose(store.data[0], [1.0, 0.0])
    assert np.allclose(store.data[13], [11.0, 11.0])


def test_growth_rebinds_maintainer_and_segment():
    m = _monitor(initial_size=6, leaf_size=2, split_threshold=2)
    for _ in range(20):
        m.run_one_operation()
    assert m._store.n_grows >= 1
    assert m._maint._vectors is m._store.data
    for unit in m._maint.searchable_units():
        assert unit.vectors is m._store.data


def test_fresh_insert_is_retrievable_by_production_search():
    m = _monitor()
    q = m._queries[0].copy()
    vid = m._store.size
    m._append_vector(q, vid)                 # vector identical to the query
    m._live.append(vid)
    m._live_set.add(vid)
    m._next_id += 1
    res = search(q, m._maint.searchable_units(), m._cfg_query, Counters())
    assert res
    assert res[0][0] == vid
    assert res[0][1] == pytest.approx(1.0, abs=1e-5)


# ---------------------------------------------------------------------------
# Recall correctness
# ---------------------------------------------------------------------------

def test_recall_uses_only_current_live_vectors():
    m = _monitor()
    gt_before = m._live_ground_truth()
    victim = int(gt_before[0, 0])            # a top-1 neighbour of query 0
    assert victim in m._live_set
    m._live.remove(victim)
    m._live_set.discard(victim)
    gt_after = m._live_ground_truth()
    assert gt_after.shape[1] == gt_before.shape[1]
    assert victim not in gt_after[0]         # recomputed, not cached
    assert not np.array_equal(gt_before[0], gt_after[0])


def test_exhaustive_search_recall_is_one():
    # Fetch every posting → production search is exact → recall must be 1.0.
    m = _monitor(n_postings=1000, beam=1000, depth=1)
    for _ in range(15):
        m.run_one_operation()
    assert m.evaluate_recall() == pytest.approx(1.0)


def test_deleted_ids_are_not_returned_by_query():
    m = _monitor(n_postings=1000, beam=1000, depth=1)
    deleted: set[int] = set()
    for _ in range(8):
        r = m.run_one_operation()
        if r.op == "DELETE":
            deleted.add(r.vid)
    assert deleted
    assert m._live_set.isdisjoint(deleted)
    for q in m._queries:
        for vid, _ in search(q, m._maint.searchable_units(), m._cfg_query,
                             Counters()):
            assert vid not in deleted


def test_recall_is_na_when_no_live_vectors():
    m = _monitor()
    m._live = []
    m._live_set = set()
    assert m.evaluate_recall() is None
    rec = OperationRecord(
        index=1, op="DELETE", vid=1, live=0, leaves=1, splits_total=0,
        merges_total=0, recall=None, elapsed_total=0.0, elapsed_op=0.0,
    )
    assert "recall@3=N/A" in m.format_record(rec)[0]


# ---------------------------------------------------------------------------
# Ids
# ---------------------------------------------------------------------------

def test_ids_are_monotonic_and_never_reused():
    m = _monitor()
    inserted: list[int] = []
    deleted: set[int] = set()
    for _ in range(30):
        r = m.run_one_operation()
        if r.op == "INSERT":
            inserted.append(r.vid)
        else:
            assert r.vid not in deleted     # a live id is deleted at most once
            deleted.add(r.vid)
    assert inserted == sorted(inserted)
    assert len(set(inserted)) == len(inserted)
    assert all(v >= m.cfg.initial_size for v in inserted)
    # A deleted id must never come back as a fresh insert: monotonic ids make
    # reuse impossible, so every delete victim is either an initial vector or an
    # id we already saw inserted.
    for victim in deleted:
        assert victim < m.cfg.initial_size or victim in set(inserted)
    assert m._live_set.isdisjoint(deleted)


def test_fixed_seed_reproduces_operation_choices():
    a = _monitor(seed=11)
    b = _monitor(seed=11)
    seq_a = [a.run_one_operation() for _ in range(20)]
    seq_b = [b.run_one_operation() for _ in range(20)]
    assert [(r.op, r.vid, r.source_row) for r in seq_a] == \
           [(r.op, r.vid, r.source_row) for r in seq_b]


def test_different_seed_changes_operation_choices():
    a = _monitor(seed=1)
    b = _monitor(seed=2)
    seq_a = [a.run_one_operation() for _ in range(20)]
    seq_b = [b.run_one_operation() for _ in range(20)]
    assert [(r.op, r.vid, r.source_row) for r in seq_a] != \
           [(r.op, r.vid, r.source_row) for r in seq_b]


# ---------------------------------------------------------------------------
# Per-operation output
# ---------------------------------------------------------------------------

def test_every_operation_prints_a_recall_line():
    m = _monitor()
    m.run(max_operations=6)
    main = [l for l in m._out.getvalue().splitlines()
            if " NPA=" in l and "vid=" in l]
    assert len(main) == 6
    assert m._op_count == 6
    for line in main:
        assert "recall@3=" in line
        assert "NPA=PASS" in line and "MEMBERSHIP=PASS" in line


def test_summary_is_printed_periodically():
    m = _monitor(summary_every=3)
    m.run(max_operations=7)
    assert m._out.getvalue().count("summary @ op") >= 2


# ---------------------------------------------------------------------------
# Structural events
# ---------------------------------------------------------------------------

def test_split_and_merge_events_are_reported():
    m = _monitor(initial_size=40, leaf_size=4, split_threshold=4)
    m.run(max_operations=60)
    stats = m._maint.stats()
    out = m._out.getvalue()
    assert stats["splits"] > 0 and stats["merges"] > 0
    assert "SPLIT pid=" in out
    assert "MERGE survivor=" in out


def test_structural_event_ids_are_valid():
    m = _monitor(initial_size=40, leaf_size=4, split_threshold=4)
    seen_split = seen_merge = False
    for _ in range(60):
        r = m.run_one_operation()
        assert isinstance(r.events, list)
        for ev in r.events:
            if ev.kind == "SPLIT":
                seen_split = True
                assert ev.pid is not None and ev.new_pid is not None
                assert ev.new_pid > ev.pid
            elif ev.kind == "MERGE":
                seen_merge = True
                assert ev.survivor is not None and ev.removed is not None
                assert ev.survivor == min(ev.survivor, ev.removed)
    assert seen_split and seen_merge


def test_observed_maintainer_is_a_real_maintainer():
    m = _monitor()
    assert isinstance(m._maint, ObservedInPlaceMaintainer)
    assert m._maint.drain_events() == []
    m.run_one_operation()
    # events list is drained per operation; the maintainer stays usable
    assert isinstance(m._maint.drain_events(), list)


# ---------------------------------------------------------------------------
# Continuous loop / shutdown
# ---------------------------------------------------------------------------

def test_loop_runs_past_initial_pool():
    m = _monitor(initial_size=10, leaf_size=4, split_threshold=4)
    m.run(max_operations=40)
    assert m._op_count == 40
    assert m._inserts >= 15
    assert m._next_id > m.cfg.initial_size + 10
    assert not m._failed


def test_stop_method_halts_loop():
    m = _monitor()

    def hook(rec):
        if rec.index >= 4:
            m.stop()
            return False

    m.run(on_operation=hook)
    assert m._op_count == 4
    assert "Final summary" in m._out.getvalue()


def test_keyboard_interrupt_prints_summary_and_keeps_index_valid():
    m = _monitor()

    def hook(rec):
        if rec.index >= 5:
            raise KeyboardInterrupt

    m.run(on_operation=hook)
    out = m._out.getvalue()
    assert "Interrupted (Ctrl+C)" in out
    assert "Final summary" in out
    assert "stopped: Ctrl+C" in out
    assert m._op_count == 5
    # Index left in a valid state by the graceful shutdown.
    assert audit_maintainer(m._maint, expected_live_vids=m._live_set).passed
    assert audit_membership(m._maint).passed


# ---------------------------------------------------------------------------
# Audits
# ---------------------------------------------------------------------------

def _failing_audit(maint, expected_live_vids=None, tolerance=1e-5,
                   max_examples=10):
    return AuditResult(
        passed=False,
        npa_violations=[{"vid": 0, "owned_pid": 1, "nearest_pid": 2}],
        ownership_errors=[], coverage_errors=[],
        live_vectors_checked=1, postings_checked=1,
        examples=[{"vid": 0}],
    )


def test_audit_failure_is_visible_and_stops(monkeypatch):
    m = _monitor()
    monkeypatch.setattr(livemod, "audit_maintainer", _failing_audit)
    m.run(max_operations=10)
    out = m._out.getvalue()
    assert "NPA=FAIL" in out
    assert "!!" in out
    assert "stopped: audit failure" in out
    assert m._op_count == 1                 # stopped on the very first op
    assert m._failed


def test_continue_on_audit_failure_keeps_running(monkeypatch):
    m = _monitor(continue_on_audit_failure=True)
    monkeypatch.setattr(livemod, "audit_maintainer", _failing_audit)
    m.run(max_operations=4)
    assert m._op_count == 4
    assert m._failed                         # flagged, but not fatal


def test_audit_every_zero_skips(monkeypatch):
    m = _monitor(audit_every=0)

    def _boom(*a, **k):
        raise AssertionError("auditors must not run when audit_every == 0")

    monkeypatch.setattr(livemod, "audit_maintainer", _boom)
    monkeypatch.setattr(livemod, "audit_membership", _boom)
    m.run(max_operations=3)
    assert "NPA=SKIP" in m._out.getvalue()
    assert m._op_count == 3


# ---------------------------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------------------------

def test_main_runs_finite_and_exits_zero(capsys):
    rc = livemod.main([
        "--initial-size", "20", "--dim", "8", "--leaf-size", "4",
        "--split-threshold", "4", "--k", "3", "--query-count", "4",
        "--interval", "0", "--max-operations", "3", "--summary-every", "0",
        "--depth", "1",
    ])
    assert rc == 0
    out = capsys.readouterr().out
    assert "Live churn monitor" in out
    assert "initial recall@3=" in out
    assert "Final summary" in out
    assert out.count("vid=") == 3
