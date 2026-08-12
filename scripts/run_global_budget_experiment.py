"""Phase C of the global cross-stream scheduler
(~/.claude/plans/sparkling-sauteeing-boole.md): does reallocating a
GENUINELY SCARCE total detector-call budget by real per-stream urgency beat
naive, uncoordinated sharing of that same scarce budget?

**Redesigned after a real methodological finding** (see
mvtrack/sched/global_replay.py's module docstring for the full writeup):
the first version of this experiment held `total_budget` equal to what
today's independent scheduling already naturally uses -- not a genuine
constraint on this project's own hardware (a single shared detector has
~150-200fps of real headroom, more than enough for 4 MOT17 streams'
combined natural demand), so there was nothing real to reallocate and
global correctly showed no benefit. Fixed by imposing REAL scarcity
(`total_budget` well below the natural total) and comparing THREE
conditions at that same real constraint:

  - "independent": today's real, fully unconstrained per-stream scheduling
    (reference point only -- NOT resource-constrained, included so the
    real cost of imposing scarcity at all is visible, not to be compared
    apples-to-apples against the other two).
  - "naive-shared": the SAME real per-tick contention/scarcity as
    "global" (same BudgetArbiter, same total_budget, same budget_per_tick)
    but with no urgency signal -- an uncoordinated-sharing proxy. This is
    the fair baseline for the actual hypothesis.
  - "global": same real scarcity, WITH urgency-aware reallocation.

Two real scenarios, both using MOT17 train FRCNN sequences already on
disk (no new downloads), real GT for TrackEval scoring -- per-sequence
average simultaneous pedestrians pulled directly from each sequence's own
gt.txt, not assumed:

  MOT17-02=50.0  MOT17-04=102.9  MOT17-05=9.6  MOT17-09=19.8
  MOT17-10=26.7  MOT17-11=11.8   MOT17-13=26.9

- "uneven": MOT17-04+02 (dense) concurrent with MOT17-05+11 (sparse) -- a
  genuine ~5-10x contrast. Pre-registered hypothesis: under real scarcity,
  "global" beats "naive-shared" because it can tell which streams actually
  need the budget more; the gap should be larger here than in "even".
- "even": MOT17-09+10+11+13 (11.8-26.9, ~2.3x spread -- the closest to
  "similarly dense" the remaining 7 sequences allow). Pre-registered
  expectation: "global" vs "naive-shared" gap should be smaller here, since
  there's less real difference in need to exploit.

`total_budget` is set to 60% of "independent"'s own natural total per
scenario -- a real, meaningful cut, not a token one.
"""

import contextlib
import csv
import io
import pathlib
import re
import sys

ROOT = pathlib.Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "eval"))
import run as eval_run  # noqa: E402
from mvtrack.sched.global_replay import (  # noqa: E402
    extract_urgency_trace, simulate_global_allocation, simulate_independent_allocation,
    simulate_naive_shared_allocation,
)

VIDEO_ROOT = ROOT / "data" / "MOT17" / "videos"
RESULTS_DIR = ROOT / "results"
SCARCITY_FRACTION = 0.6
BUDGET_PER_TICK = 1  # one shared detector -- see global_replay.py's docstring


def _parse_combined_metrics(evaluate_output: str) -> dict:
    """Pulls the aggregate HOTA and MOTA off evaluate()'s printed table --
    it doesn't return structured data, and reparsing its own stdout is
    less invasive than changing its signature for every other caller."""
    hota = mota = None
    for line in evaluate_output.splitlines():
        if line.startswith("COMBINED") and hota is None:
            nums = re.findall(r"[-+]?\d*\.?\d+", line)
            if nums:
                hota = float(nums[0])
        elif line.startswith("COMBINED") and mota is None:
            nums = re.findall(r"[-+]?\d*\.?\d+", line)
            if nums:
                mota = float(nums[0])
    return {"HOTA": hota, "MOTA": mota}


def run_scenario(name: str, seq_names: list, csv_out: pathlib.Path):
    videos = {seq: VIDEO_ROOT / f"{seq}.mp4" for seq in seq_names}
    print(f"\n=== [{name}] extracting urgency traces (MV-only, no detector) ===")
    traces = {seq: extract_urgency_trace(str(videos[seq])) for seq in seq_names}

    independent_anchors = {seq: simulate_independent_allocation(traces[seq]) for seq in seq_names}
    natural_total = sum(len(a) for a in independent_anchors.values())
    total_budget = round(natural_total * SCARCITY_FRACTION)
    print(f"[{name}] independent's natural total: {natural_total} -- "
          f"real scarcity budget imposed for the other two conditions: {total_budget} "
          f"({SCARCITY_FRACTION*100:.0f}%)")

    stream_ids = dict(enumerate(seq_names))
    traces_by_id = {i: traces[seq] for i, seq in stream_ids.items()}

    naive_by_id = simulate_naive_shared_allocation(traces_by_id, total_budget=total_budget,
                                                     budget_per_tick=BUDGET_PER_TICK)
    naive_anchors = {stream_ids[i]: a for i, a in naive_by_id.items()}
    naive_total = sum(len(a) for a in naive_anchors.values())

    global_by_id = simulate_global_allocation(traces_by_id, total_budget=total_budget,
                                               budget_per_tick=BUDGET_PER_TICK)
    global_anchors = {stream_ids[i]: a for i, a in global_by_id.items()}
    global_total = sum(len(a) for a in global_anchors.values())

    print(f"[{name}] naive-shared total: {naive_total} (<= {total_budget})   "
          f"global total: {global_total} (<= {total_budget})")
    for seq in seq_names:
        print(f"    {seq}: independent={len(independent_anchors[seq])} "
              f"naive-shared={len(naive_anchors[seq])} global={len(global_anchors[seq])}")

    rows = []
    conditions = [
        ("independent", independent_anchors, natural_total),
        ("naive-shared", naive_anchors, total_budget),
        ("global", global_anchors, total_budget),
    ]
    for condition, anchors_map, budget_ceiling in conditions:
        pipeline_name = f"mv-replay-{name}-{condition}"
        out_dir = eval_run.RESULTS / pipeline_name / "data"
        for seq in seq_names:
            out_txt = out_dir / f"{seq}.txt"
            frames, dt = eval_run.run_mv_replay(videos[seq], out_txt, anchor_frames=anchors_map[seq])
            print(f"    [{condition}] {seq}: {frames} frames in {dt:.1f}s")

        buf = io.StringIO()
        with contextlib.redirect_stdout(buf):
            eval_run.evaluate(pipeline_name, seq_names)
        output = buf.getvalue()
        print(output)
        agg = _parse_combined_metrics(output)
        actual_total = sum(len(a) for a in anchors_map.values())
        rows.append({
            "scenario": name, "condition": condition,
            "budget_ceiling": budget_ceiling, "total_anchor_calls": actual_total,
            "HOTA": agg["HOTA"], "MOTA": agg["MOTA"],
        })

    csv_out.parent.mkdir(parents=True, exist_ok=True)
    with open(csv_out, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=[
            "scenario", "condition", "budget_ceiling", "total_anchor_calls", "HOTA", "MOTA",
        ])
        writer.writeheader()
        writer.writerows(rows)
    print(f"wrote {csv_out}")
    return rows


if __name__ == "__main__":
    run_scenario(
        "uneven", ["MOT17-04-FRCNN", "MOT17-02-FRCNN", "MOT17-05-FRCNN", "MOT17-11-FRCNN"],
        RESULTS_DIR / "global_budget_uneven.csv",
    )
    run_scenario(
        "even", ["MOT17-09-FRCNN", "MOT17-10-FRCNN", "MOT17-11-FRCNN", "MOT17-13-FRCNN"],
        RESULTS_DIR / "global_budget_even_control.csv",
    )
