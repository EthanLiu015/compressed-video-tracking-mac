"""Unit tests for the pure BudgetArbiter policy (src/mvtrack/sched/
global_budget.py) -- synthetic urgency traces, no real video/processes.
Covers the behaviors the global-scheduler plan explicitly calls out:
reallocation toward a spiking stream, forced-request preemption,
total-budget accounting never overspends, and starvation-floor honoring."""

from mvtrack.sched import BudgetArbiter, Request


def test_reallocation_toward_spiking_stream():
    # 3 streams, 1 slot per tick, one clearly more urgent than its peers --
    # the arbiter should consistently pick it, not split evenly/randomly.
    arb = BudgetArbiter(budget_per_tick=1)
    pending = [
        Request(stream_id=0, frame_index=10, urgency=1.0),
        Request(stream_id=1, frame_index=10, urgency=5.0),  # the spike
        Request(stream_id=2, frame_index=10, urgency=1.2),
    ]
    decisions = arb.decide(pending)
    granted = [d for d in decisions if d.granted]
    assert len(granted) == 1
    assert granted[0].stream_id == 1


def test_forced_request_preempts_higher_urgency_optional():
    # A forced (starvation-floor) request must win a slot even against a
    # numerically more "urgent" optional request from a different stream --
    # the whole point of `forced` is "this isn't optional, it's a real
    # invariant (findings.md #15's anti-ghost-track max_age coupling), not
    # just a preference."
    arb = BudgetArbiter(budget_per_tick=1)
    pending = [
        Request(stream_id=0, frame_index=5, urgency=0.1, forced=True),
        Request(stream_id=1, frame_index=5, urgency=9.0, forced=False),
    ]
    decisions = arb.decide(pending)
    granted = {d.stream_id for d in decisions if d.granted}
    assert granted == {0}


def test_total_budget_never_overspent():
    arb = BudgetArbiter(budget_per_tick=10, total_budget=3)
    total_granted = 0
    for tick in range(5):
        pending = [Request(stream_id=i, frame_index=tick, urgency=float(i)) for i in range(4)]
        decisions = arb.decide(pending)
        total_granted += sum(1 for d in decisions if d.granted)
    assert total_granted == 3
    assert arb.remaining == 0


def test_total_budget_exhausted_denies_even_forced_requests():
    # Documented, deliberate tradeoff (see BudgetArbiter's own docstring):
    # once a finite total_budget is spent, even a forced request is denied
    # -- there's no way to honor "N total calls, held equal" otherwise.
    arb = BudgetArbiter(budget_per_tick=10, total_budget=0)
    pending = [Request(stream_id=0, frame_index=0, urgency=1.0, forced=True)]
    decisions = arb.decide(pending)
    assert decisions[0].granted is False


def test_rate_mode_has_no_total_ceiling():
    # total_budget=None (the live Phase D case): the same stream can keep
    # winning tick after tick, unbounded -- only the per-tick rate caps it.
    arb = BudgetArbiter(budget_per_tick=1)
    for tick in range(50):
        pending = [
            Request(stream_id=0, frame_index=tick, urgency=9.0),
            Request(stream_id=1, frame_index=tick, urgency=0.1),
        ]
        decisions = arb.decide(pending)
        granted = {d.stream_id for d in decisions if d.granted}
        assert granted == {0}
    assert arb.remaining is None


def test_tick_cap_shared_between_forced_and_optional():
    # 1 forced + 2 optional requests, budget_per_tick=2 -- forced takes one
    # slot, the single remaining slot goes to the higher-urgency optional
    # request, not both optional requests.
    arb = BudgetArbiter(budget_per_tick=2)
    pending = [
        Request(stream_id=0, frame_index=0, urgency=0.5, forced=True),
        Request(stream_id=1, frame_index=0, urgency=8.0, forced=False),
        Request(stream_id=2, frame_index=0, urgency=2.0, forced=False),
    ]
    decisions = arb.decide(pending)
    granted = {d.stream_id for d in decisions if d.granted}
    assert granted == {0, 1}


if __name__ == "__main__":
    test_reallocation_toward_spiking_stream()
    test_forced_request_preempts_higher_urgency_optional()
    test_total_budget_never_overspent()
    test_total_budget_exhausted_denies_even_forced_requests()
    test_rate_mode_has_no_total_ceiling()
    test_tick_cap_shared_between_forced_and_optional()
    print("all budget arbiter checks passed")
