"""Multi-stream throughput sweep (weeks 11-12): how many concurrent mvtrack
pipeline instances can the M4 sustain before per-stream fps drops below a
target? Also samples power via `powermetrics` if passwordless sudo is set
up; degrades gracefully (just omits the power column) if not, rather than
blocking on a password prompt.

Usage:
    python scripts/bench_multistream.py --pipeline mv-adaptive \
        --video data/people_baseline.mp4 --streams 1 2 3 4 6 8
Writes results/bench_multistream_<pipeline>.csv (committed, not gitignored,
so the throughput/energy Pareto plots stay reproducible without rerunning).
"""

import argparse
import csv
import pathlib
import subprocess
import sys
import threading

ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from mvtrack.bench import run_multistream

TARGET_FPS = 25.0


def sample_power(duration_s: float) -> float | None:
    """Average combined CPU+GPU power (mW) via powermetrics over
    `duration_s` seconds, or None if unavailable. powermetrics needs root;
    this uses `sudo -n` (non-interactive) so a missing password just fails
    fast instead of hanging the benchmark on a prompt."""
    try:
        proc = subprocess.run(
            ["sudo", "-n", "powermetrics", "--samplers", "cpu_power,gpu_power",
             "-i", "1000", "-n", str(max(1, int(duration_s)))],
            capture_output=True, text=True, timeout=duration_s + 15,
        )
    except Exception:
        return None
    if proc.returncode != 0:
        return None
    watts = []
    for line in proc.stdout.splitlines():
        if "Combined Power" in line and "mW" in line:
            try:
                watts.append(float(line.split(":")[1].strip().split()[0]))
            except (IndexError, ValueError):
                continue
    return sum(watts) / len(watts) if watts else None


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--pipeline", required=True)
    ap.add_argument("--video", required=True)
    ap.add_argument("--streams", type=int, nargs="+", default=[1, 2, 3, 4, 6, 8])
    ap.add_argument("--anchor-interval", type=int, default=5)
    ap.add_argument("--power", action="store_true", help="sample powermetrics (needs passwordless sudo)")
    args = ap.parse_args()

    out_dir = ROOT / "results"
    out_dir.mkdir(exist_ok=True)
    out_csv = out_dir / f"bench_multistream_{args.pipeline}.csv"

    if args.power and sample_power(1.0) is None:
        print("--power requested but powermetrics unavailable/needs sudo password; skipping energy column")
        args.power = False

    rows = []
    for n in args.streams:
        power_holder: dict[str, float] = {}
        power_thread = None
        if args.power:
            def _sample():
                mw = sample_power(6.0)
                if mw is not None:
                    power_holder["mw"] = mw
            power_thread = threading.Thread(target=_sample)
            power_thread.start()

        agg_fps, per_fps, wall = run_multistream(
            args.pipeline, pathlib.Path(args.video), n, args.anchor_interval
        )
        if power_thread:
            power_thread.join()
        mw = power_holder.get("mw")
        print(
            f"n={n}: aggregate={agg_fps:.1f}fps per-stream={per_fps:.1f}fps"
            + (f" power={mw:.0f}mW" if mw else "")
        )
        rows.append(
            {
                "n_streams": n,
                "aggregate_fps": round(agg_fps, 2),
                "per_stream_fps": round(per_fps, 2),
                "wall_s": round(wall, 2),
                "power_mw": round(mw, 1) if mw else "",
            }
        )
        if per_fps < TARGET_FPS:
            print(f"per-stream fps dropped below {TARGET_FPS} at n={n}; stopping sweep")
            break

    with open(out_csv, "w", newline="") as f:
        writer = csv.DictWriter(
            f, fieldnames=["n_streams", "aggregate_fps", "per_stream_fps", "wall_s", "power_mw"]
        )
        writer.writeheader()
        writer.writerows(rows)
    print(f"wrote {out_csv}")


if __name__ == "__main__":
    main()
