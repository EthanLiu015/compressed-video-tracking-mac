"""Pure budget-arbitration policy for the global cross-stream scheduler.

Today, `run_multistream` (mvtrack/bench/harness.py) runs N concurrent
streams, each with its own `Adaptive` scheduler deciding anchor timing in
total isolation -- confirmed by direct code read, zero shared state. This
module is the policy that replaces "N independent thresholds" with "N
streams reporting a comparable urgency score (`Adaptive.urgency`) to a
shared arbiter that spends a bounded detector-call budget on whoever needs
it most."

Deliberately contains ZERO process/IPC logic -- it's a pure function of its
inputs, reused unchanged by both the offline trace-replay simulator
(global_replay.py, Phase C) and the live multiprocess arbiter
(arbiter_process.py, Phase D). That matters: it keeps the "does this help
accuracy" experiment (Phase C, no real processes) and the "does this work
live" demo (Phase D, real processes) honest against each other -- one
policy, two harnesses, not two policies that could quietly drift apart.
"""

from dataclasses import dataclass


@dataclass(frozen=True)
class Request:
    stream_id: int
    frame_index: int
    urgency: float
    # A stream's own local safety net (its `since_anchor >= max_interval`
    # bound, mirroring Adaptive's existing max_interval safety net) --
    # not a claim about global importance, just "this track hasn't been
    # checked in too long and risks MVTracker pruning it as a false ghost
    # (findings.md #15) if it goes much longer." Honored ahead of
    # urgency-ranked optional requests, but still drawn from a finite
    # total_budget if one is set -- see BudgetArbiter.decide.
    forced: bool = False


@dataclass(frozen=True)
class Decision:
    stream_id: int
    frame_index: int
    granted: bool


class BudgetArbiter:
    """`budget_per_tick` is a real throughput ceiling (however many
    detector calls this tick's realistic inference budget allows) and
    applies in both modes. `total_budget`, if set, is an ADDITIONAL
    shrinking cap across every call to `decide()` combined -- used by
    Phase C to hold two allocation strategies to the exact same total
    detector-call count for a fair comparison. Leave it `None` for the
    live Phase D case, where the only real constraint is the per-tick rate.

    Every `decide()` call: forced requests get priority slots (ranked by
    urgency among themselves, for determinism), then remaining slots go to
    optional requests ranked by descending urgency. If `total_budget` is
    exhausted, `tick_cap` collapses to 0 and nothing is granted that tick --
    including forced requests. This is a real, documented tradeoff: under a
    tight enough total_budget, even a starved track's forced request can be
    denied. That's an honest cost of the total-budget experiment, not a
    bug to hide -- flagged in the plan's own risk section.
    """

    def __init__(self, budget_per_tick: int, total_budget: int | None = None):
        self.budget_per_tick = budget_per_tick
        self.total_budget = total_budget
        self.remaining = total_budget  # None == unlimited

    def decide(self, pending: list[Request]) -> list[Decision]:
        tick_cap = self.budget_per_tick
        if self.remaining is not None:
            tick_cap = min(tick_cap, max(0, self.remaining))

        forced = sorted((r for r in pending if r.forced), key=lambda r: r.urgency, reverse=True)
        optional = sorted((r for r in pending if not r.forced), key=lambda r: r.urgency, reverse=True)

        granted: list[Request] = []
        for r in forced:
            if len(granted) >= tick_cap:
                break
            granted.append(r)
        for r in optional:
            if len(granted) >= tick_cap:
                break
            granted.append(r)

        granted_keys = {(r.stream_id, r.frame_index) for r in granted}
        decisions = [
            Decision(r.stream_id, r.frame_index, (r.stream_id, r.frame_index) in granted_keys)
            for r in pending
        ]

        if self.remaining is not None:
            self.remaining -= len(granted)

        return decisions
