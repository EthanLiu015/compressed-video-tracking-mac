"""Grid-search Adaptive scheduler params against real MOT17 accuracy.

The four Adaptive parameters (min_interval, max_interval, spike_factor,
ema_alpha) were pure dataclass defaults, never tuned against ground truth
-- confirmed via a repo-wide grep before this script existed. Sweeps a
grid on two fast tuning sequences (not the full 7 used for final
reporting, to avoid picking a combination that just happens to fit that
exact mix), scores MOTA via the same eval/run.py + TrackEval path used
everywhere else, and reports the winner. Caller should then validate the
chosen combination on the full 7-sequence set the normal way
(`eval/run.py --pipeline mv-adaptive`) before adopting it as the new
`Adaptive` default.

Usage: python scripts/tune_scheduler.py
Writes results/scheduler_tune.csv
"""

import contextlib
import io
import itertools
import pathlib
import sys

ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "eval"))

import run as eval_run  # eval/run.py

TUNE_SEQS = ["MOT17-09-FRCNN", "MOT17-10-FRCNN"]
PIPELINE_LABEL = "tune-scheduler"

GRID = {
    "min_interval": [2],
    "max_interval": [8, 10, 15, 20],
    "spike_factor": [1.2, 1.4, 1.6, 2.0],
    "ema_alpha": [0.2],
}


def _extract_mota(text: str) -> float:
    lines = text.splitlines()
    for i, line in enumerate(lines):
        if line.startswith("CLEAR:"):
            for j in range(i + 1, len(lines)):
                if lines[j].startswith("COMBINED"):
                    return float(lines[j].split()[1])
    raise RuntimeError(f"MOTA not found in TrackEval output:\n{text}")


def run_combo(params: dict) -> tuple[float, float]:
    """Returns (mota, mean_fps) on the tuning sequences for this combo."""
    out_dir = eval_run.RESULTS / PIPELINE_LABEL / "data"
    fps_all = []
    for seq in TUNE_SEQS:
        video = ROOT / "data" / "MOT17" / "videos" / f"{seq}.mp4"
        res_txt = out_dir / f"{seq}.txt"
        frames, dt = eval_run.run_mv_adaptive(
            video, res_txt, weights="yolov8s.pt", scheduler_kwargs=params
        )
        fps_all.append(frames / dt)

    buf = io.StringIO()
    with contextlib.redirect_stdout(buf):
        eval_run.evaluate(PIPELINE_LABEL, TUNE_SEQS)
    mota = _extract_mota(buf.getvalue())
    return mota, sum(fps_all) / len(fps_all)


if __name__ == "__main__":
    keys = list(GRID.keys())
    combos = list(itertools.product(*GRID.values()))
    print(f"grid: {len(combos)} combinations on {TUNE_SEQS}")

    rows = []
    for combo in combos:
        params = dict(zip(keys, combo))
        mota, fps = run_combo(params)
        print(f"  {params} -> MOTA={mota:.2f} fps={fps:.1f}")
        rows.append({**params, "mota": mota, "mean_fps": fps})

    rows.sort(key=lambda r: -r["mota"])
    best = rows[0]
    print(f"\nbest: {best}")

    out_csv = ROOT / "results" / "scheduler_tune.csv"
    with open(out_csv, "w") as f:
        f.write(",".join(keys) + ",mota,mean_fps\n")
        for r in rows:
            f.write(",".join(str(r[k]) for k in keys) + f",{r['mota']:.3f},{r['mean_fps']:.1f}\n")
    print(f"wrote {out_csv}")
