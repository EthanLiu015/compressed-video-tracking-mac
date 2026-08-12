"""Offline trace-driven comparison: today's per-stream-independent
`Adaptive` scheduling vs. `BudgetArbiter`'s global reallocation, replayed
through the real tracking pipeline and scored by real TrackEval metrics.
Answers the actual research question (does reallocating toward busy
streams help) without needing any live multiprocessing -- that's Phase D,
a separate live/throughput layer on top of this.

**Real methodological finding from the first version of this experiment,
worth keeping in mind**: holding `total_budget` equal to what today's
independent scheduling naturally uses is NOT a genuine scarcity test on
this project's own hardware -- a single shared detector has enough real
throughput headroom (~150-200fps on this Mac) that 4 MOT17 streams' combined
natural anchor demand was never actually resource-constrained, so global
reallocation had nothing real to redistribute and (correctly) showed no
benefit. `simulate_naive_shared_allocation` exists specifically to give a
fair, apples-to-apples baseline once genuine scarcity IS imposed (a
`total_budget` meaningfully below the natural total) -- same real
per-tick contention as `simulate_global_allocation`, just without the
urgency signal, isolating exactly what that signal buys.

**Second real bug caught before trusting any of this**: the naive
baseline's first version assigned every request a flat `urgency=0.0`.
`BudgetArbiter.decide` ranks with Python's stable `sorted()`, so an
all-tied list preserves its ORIGINAL order -- which was dict-iteration
order over `traces`, i.e. always stream 0 first. That's not
"uncoordinated sharing," it's a silent, deterministic priority order
favoring low stream ids every single tick (caught directly: a synthetic
3-stream always-competing test gave stream 0 100% of the budget under
BOTH "naive" and "global," which should never coincide once real urgency
differs). Fixed by giving naive requests an independently-seeded random
tie-break value instead of a shared constant -- genuine "no real signal"
behavior, not hidden favoritism.

**Why urgency can be extracted once, before any allocation decision is
made** (a real subtlety, not glossed over): `Adaptive.urgency()` returns a
spike ratio computed from its EMA baseline, and that EMA updates every
single frame regardless of whether that frame becomes an anchor -- only
`_since_anchor` is fire-dependent, and `_since_anchor` plays no role in the
returned ratio itself (see `Adaptive.should_anchor`'s own logic: the EMA
update happens unconditionally, `_since_anchor` resets only conditionally
on `fire`). So the raw (urgency, intra_frac, pict_type) trace is genuinely
allocation-independent and can be computed once per stream, then reused
by both simulators below -- each of which maintains its OWN separate
`since_anchor`/`warm` state as it walks the trace and decides real anchor
frames, so the two simulations' actual anchor timing (and thus their real
min/max_interval gating) correctly diverges based on what each one
actually grants, even though they start from the same underlying signal.

**Explicit simplifying assumption**: `simulate_global_allocation` merges
streams by ORDINAL trace position (the i-th frame of each stream's own
sequence counts as the same "tick"), not by any real wall-clock
synchronization -- stated here, not hidden. For same-fps MOT17 sequences
this is a reasonable proxy for "roughly concurrent," not a claim of exact
synchronization.
"""

import random
from dataclasses import dataclass

from mvtrack.extract import iter_frames_with_mvs
from mvtrack.sched import Adaptive, BudgetArbiter, Request


@dataclass(frozen=True)
class TraceEntry:
    frame_index: int
    urgency: float
    intra_frac: float
    pict_type: str


def extract_urgency_trace(video: str, adaptive_kwargs: dict | None = None) -> list[TraceEntry]:
    """One cheap MV-only pass per stream -- no detector calls, same
    "cheap signal before expensive work" pattern as `mvtrack.analytics.
    mv_energy`. `adaptive_kwargs` only affects the EMA smoothing
    (`ema_alpha`); min/max_interval/spike_factor are read separately by
    the simulators below, not baked into the trace."""
    adaptive = Adaptive(**(adaptive_kwargs or {}))
    trace = []
    for fmv, _frame in iter_frames_with_mvs(str(video)):
        ratio = adaptive.urgency(fmv)
        trace.append(TraceEntry(fmv.index, ratio, adaptive._last_intra_frac, fmv.pict_type))
    return trace


def simulate_independent_allocation(
    trace: list[TraceEntry], min_interval: int = 2, max_interval: int = 8, spike_factor: float = 1.4,
) -> set[int]:
    """Reproduces today's real `Adaptive.should_anchor` decision sequence
    from a precomputed trace -- this IS the current production behavior,
    just replayed from cached signal instead of a live call, so it's a
    fair baseline (verified to reproduce a real live `should_anchor` run
    frame-for-frame, not just plausibly similar -- see
    scripts/run_global_budget_experiment.py's verification step)."""
    anchors = set()
    since_anchor = 0
    warm = False
    for entry in trace:
        since_anchor += 1
        fire = entry.pict_type == "I" or since_anchor >= max_interval
        if not fire and warm and since_anchor >= min_interval:
            fire = entry.urgency > spike_factor and entry.intra_frac > 0.05
        warm = True
        if fire:
            anchors.add(entry.frame_index)
            since_anchor = 0
    return anchors


def _simulate_shared(
    traces: dict[int, list[TraceEntry]], total_budget: int, budget_per_tick: int,
    min_interval: int, max_interval: int, spike_factor: float, use_urgency: bool,
    rng_seed: int = 0,
) -> dict[int, set[int]]:
    """Shared tick-contention mechanism for both `simulate_global_allocation`
    (use_urgency=True) and `simulate_naive_shared_allocation`
    (use_urgency=False) -- identical scarcity/contention model, the ONLY
    difference is whether the arbiter ranks by real per-request urgency or
    by an independently-seeded random tie-break (an uninformed/
    uncoordinated-sharing proxy -- NOT a shared constant, which would
    silently collapse to deterministic stream-id priority under
    `BudgetArbiter`'s stable sort; caught directly, see module docstring).
    Isolates exactly what the urgency signal buys once real scarcity
    exists, rather than only comparing against today's fully unconstrained
    independent scheduling (which has enough real hardware headroom on
    this project's own detector throughput numbers that nothing is
    actually scarce -- see module docstring and
    scripts/run_global_budget_experiment.py's writeup for why that
    comparison alone isn't the whole story)."""
    arbiter = BudgetArbiter(budget_per_tick=budget_per_tick, total_budget=total_budget)
    state = {sid: {"since_anchor": 0, "warm": False} for sid in traces}
    granted: dict[int, set[int]] = {sid: set() for sid in traces}
    rng = random.Random(rng_seed)

    max_len = max(len(t) for t in traces.values())
    for tick in range(max_len):
        pending = []
        for sid, trace in traces.items():
            if tick >= len(trace):
                continue
            entry = trace[tick]
            st = state[sid]
            st["since_anchor"] += 1
            forced = entry.pict_type == "I" or st["since_anchor"] >= max_interval
            eligible = forced or (
                st["warm"] and st["since_anchor"] >= min_interval
                and entry.urgency > spike_factor and entry.intra_frac > 0.05
            )
            if eligible:
                rank = entry.urgency if use_urgency else rng.random()
                pending.append(Request(stream_id=sid, frame_index=entry.frame_index,
                                        urgency=rank, forced=forced))
            st["warm"] = True

        for d in arbiter.decide(pending):
            if d.granted:
                granted[d.stream_id].add(d.frame_index)
                state[d.stream_id]["since_anchor"] = 0

    return granted


def simulate_global_allocation(
    traces: dict[int, list[TraceEntry]], total_budget: int, budget_per_tick: int,
    min_interval: int = 2, max_interval: int = 8, spike_factor: float = 1.4,
) -> dict[int, set[int]]:
    """Same local min/max_interval/spike_factor gating as
    `simulate_independent_allocation` decides which frames are even
    CANDIDATES per stream (the global scheduler doesn't override a
    stream's own min-interval floor) -- but AMONG simultaneously-eligible
    candidates across streams, `BudgetArbiter` picks by urgency (or
    forced-priority) within a shared `total_budget`, instead of every
    stream independently anchoring whenever it locally wants to.

    `budget_per_tick` has NO default and must be chosen deliberately --
    this is the real per-tick throughput ceiling of the shared detector
    this whole design is premised on (Part 2's motivation: one accelerator
    serving many cameras), not just an extra knob. A first version of this
    experiment defaulted it to `len(traces)`, which meant a request never
    had to compete with anything: at most N streams can ever request in
    the same tick, and a budget of N accommodates all of them every time,
    so the arbiter degenerated into a no-op (verified directly: it
    reproduced `simulate_independent_allocation`'s per-stream anchor
    counts EXACTLY on real MOT17 data, streamed-identical, not just
    similar -- caught precisely because that result was too clean to be
    real reallocation). At real camera framerates a single shared detector
    can service roughly one call per frame-tick, so `budget_per_tick=1` is
    the physically grounded choice for "N cameras sharing one detector,"
    not an arbitrary tightening.

    Also see `simulate_naive_shared_allocation`, which uses the exact same
    contention mechanism WITHOUT the urgency signal -- the fairer baseline
    once real scarcity exists (today's fully unconstrained
    `simulate_independent_allocation` isn't resource-constrained at all
    on this project's own real detector throughput, so it can't show what
    urgency-aware sharing specifically buys under real contention)."""
    return _simulate_shared(traces, total_budget, budget_per_tick, min_interval, max_interval,
                             spike_factor, use_urgency=True)


def simulate_naive_shared_allocation(
    traces: dict[int, list[TraceEntry]], total_budget: int, budget_per_tick: int,
    min_interval: int = 2, max_interval: int = 8, spike_factor: float = 1.4, rng_seed: int = 0,
) -> dict[int, set[int]]:
    """Same real per-tick scarcity/contention as `simulate_global_allocation`
    (same `BudgetArbiter`, same total_budget, same budget_per_tick), but
    ranked by an independently-seeded random tie-break instead of real
    urgency -- an uncoordinated/uninformed-sharing proxy. NOT a shared
    constant (see module docstring for why that silently collapses to
    deterministic stream-id priority under a stable sort -- a real bug
    caught before trusting this). This is the scientifically clean
    baseline for testing whether urgency-aware allocation helps: identical
    scarcity, only the ranking signal differs."""
    return _simulate_shared(traces, total_budget, budget_per_tick, min_interval, max_interval,
                             spike_factor, use_urgency=False, rng_seed=rng_seed)
