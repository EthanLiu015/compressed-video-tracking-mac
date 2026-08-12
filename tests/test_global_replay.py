"""Correctness checks for mvtrack.sched.global_replay -- guards against the
two real bugs found while building Phase C of the global scheduler plan:
(1) a too-loose budget_per_tick default making the arbiter a silent no-op,
(2) the naive baseline's flat urgency=0.0 collapsing to deterministic
stream-id priority under BudgetArbiter's stable sort instead of genuine
uncoordinated sharing. Both were caught by real MOT17 replay results being
suspiciously identical between conditions that should have differed --
these synthetic tests would have caught them immediately instead."""

from mvtrack.sched.global_replay import (
    TraceEntry, simulate_global_allocation, simulate_independent_allocation,
    simulate_naive_shared_allocation,
)


def _flat_trace(n, urgency, max_interval_placeholder_ok=True):
    return [TraceEntry(frame_index=i, urgency=urgency, intra_frac=0.1,
                        pict_type="I" if i == 0 else "P") for i in range(n)]


def test_global_favors_busier_stream_over_naive():
    # Regression test for both real bugs: budget_per_tick=1 forces genuine
    # per-tick contention (bug #1), and naive must NOT deterministically
    # starve the same stream every run (bug #2).
    busy = _flat_trace(200, urgency=5.0)
    quiet = _flat_trace(200, urgency=2.0)
    traces = {0: busy, 1: quiet}

    glob = simulate_global_allocation(traces, total_budget=150, budget_per_tick=1,
                                       min_interval=1, max_interval=100000)
    naive = simulate_naive_shared_allocation(traces, total_budget=150, budget_per_tick=1,
                                              min_interval=1, max_interval=100000)

    # global must not distribute identically to naive (that's bug #1's signature)
    assert {k: len(v) for k, v in glob.items()} != {k: len(v) for k, v in naive.items()}
    # global should heavily favor the busier stream given a real, sustained urgency gap
    assert len(glob[0]) > len(glob[1])
    # naive must give the "quiet" stream a real, non-trivial share (bug #2's
    # signature was 0% -- a flat urgency tie collapsing to strict stream-id priority)
    assert len(naive[1]) > 0.2 * (len(naive[0]) + len(naive[1]))


def test_naive_baseline_not_deterministically_biased_by_stream_id():
    # Three EQUALLY urgent streams -- naive sharing should split roughly
    # evenly across all three, not hand everything to stream 0.
    traces = {i: _flat_trace(300, urgency=3.0) for i in range(3)}
    naive = simulate_naive_shared_allocation(traces, total_budget=150, budget_per_tick=1,
                                              min_interval=1, max_interval=100000)
    counts = [len(v) for v in naive.values()]
    assert all(c > 0 for c in counts), naive
    assert max(counts) - min(counts) < 0.5 * (sum(counts) / len(counts)), counts


def test_total_budget_respected_by_both_simulators():
    traces = {0: _flat_trace(200, urgency=5.0), 1: _flat_trace(200, urgency=2.0)}
    for sim in (simulate_global_allocation, simulate_naive_shared_allocation):
        result = sim(traces, total_budget=40, budget_per_tick=1, min_interval=1, max_interval=100000)
        assert sum(len(v) for v in result.values()) <= 40


def test_independent_allocation_respects_max_interval_safety_net():
    # Even with urgency always below spike_factor, max_interval must still
    # force periodic anchors -- the safety net findings.md #15 depends on.
    trace = _flat_trace(50, urgency=0.1)
    anchors = simulate_independent_allocation(trace, min_interval=2, max_interval=8, spike_factor=1.4)
    sorted_anchors = sorted(anchors)
    gaps = [b - a for a, b in zip(sorted_anchors, sorted_anchors[1:])]
    assert all(g <= 8 for g in gaps), gaps


if __name__ == "__main__":
    test_global_favors_busier_stream_over_naive()
    test_naive_baseline_not_deterministically_biased_by_stream_id()
    test_total_budget_respected_by_both_simulators()
    test_independent_allocation_respects_max_interval_safety_net()
    print("all global_replay checks passed")
