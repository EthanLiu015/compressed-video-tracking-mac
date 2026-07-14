"""Grid-search MVTracker.max_age against real MOT17 accuracy.

Diagnostic (scratchpad greedy-recall probe, not committed) showed
mv-fixed's FN count roughly matches baseline's (~8600 vs ~8630 on
MOT17-09+10) -- recall isn't the gap. FP is: mv-fixed's CLR_FP is ~3.3x
baseline's on the same subset (4120 vs 1257). Root cause: a track missed
at an anchor frame isn't pruned until `since_detection > max_age` (default
30), so it keeps getting propagated *and reported* as a live box for up
to 30 more frames / ~6 anchor cycles -- a ghost track. Sweeps max_age on
the same two fast tuning sequences used by tune_scheduler.py.

max_age has to be tuned per-pipeline, not shared: mv-fixed's optimum
(5, == its anchor_interval) *breaks* mv-adaptive (HOTA 31.7->9.5) because
mv-adaptive's scheduler can leave gaps up to max_interval=8 -- pruning at
5 kills legitimate tracks before their real next anchor ever arrives.
Hence --pipeline instead of a single shared grid.

Usage: python scripts/tune_max_age.py --pipeline mv-fixed
       python scripts/tune_max_age.py --pipeline mv-adaptive
Writes results/max_age_tune_<pipeline>.csv
"""

import argparse
import contextlib
import io
import pathlib
import sys

ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "eval"))

import run as eval_run  # eval/run.py

TUNE_SEQS = ["MOT17-09-FRCNN", "MOT17-10-FRCNN"]
GRIDS = {
    "mv-fixed": [1, 2, 3, 5, 8, 15, 30],
    "mv-adaptive": [6, 8, 10, 12, 15, 20, 30],
}
RUNNERS = {"mv-fixed": eval_run.run_mv_fixed, "mv-adaptive": eval_run.run_mv_adaptive}


def _extract_mota(text: str) -> float:
    lines = text.splitlines()
    for i, line in enumerate(lines):
        if line.startswith("CLEAR:"):
            for j in range(i + 1, len(lines)):
                if lines[j].startswith("COMBINED"):
                    return float(lines[j].split()[1])
    raise RuntimeError(f"MOTA not found in TrackEval output:\n{text}")


def run_combo(pipeline: str, pipeline_label: str, max_age: int) -> tuple[float, float]:
    out_dir = eval_run.RESULTS / pipeline_label / "data"
    fps_all = []
    for seq in TUNE_SEQS:
        video = ROOT / "data" / "MOT17" / "videos" / f"{seq}.mp4"
        res_txt = out_dir / f"{seq}.txt"
        frames, dt = RUNNERS[pipeline](
            video, res_txt, weights="yolov8s.pt", tracker_kwargs={"max_age": max_age}
        )
        fps_all.append(frames / dt)

    buf = io.StringIO()
    with contextlib.redirect_stdout(buf):
        eval_run.evaluate(pipeline_label, TUNE_SEQS)
    mota = _extract_mota(buf.getvalue())
    return mota, sum(fps_all) / len(fps_all)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--pipeline", choices=GRIDS, default="mv-fixed")
    args = ap.parse_args()

    grid = GRIDS[args.pipeline]
    pipeline_label = f"tune-max-age-{args.pipeline}"
    print(f"max_age grid: {grid} on {TUNE_SEQS} ({args.pipeline})")
    rows = []
    for max_age in grid:
        mota, fps = run_combo(args.pipeline, pipeline_label, max_age)
        print(f"  max_age={max_age} -> MOTA={mota:.2f} fps={fps:.1f}")
        rows.append({"max_age": max_age, "mota": mota, "mean_fps": fps})

    rows.sort(key=lambda r: -r["mota"])
    best = rows[0]
    print(f"\nbest: {best}")

    out_csv = ROOT / "results" / f"max_age_tune_{args.pipeline}.csv"
    with open(out_csv, "w") as f:
        f.write("max_age,mota,mean_fps\n")
        for r in rows:
            f.write(f"{r['max_age']},{r['mota']:.3f},{r['mean_fps']:.1f}\n")
    print(f"wrote {out_csv}")
