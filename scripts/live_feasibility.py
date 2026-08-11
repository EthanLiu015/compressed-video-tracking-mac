"""Does live, real-time tracking on this MacBook actually need mv-tracking's
throughput win, or would the baseline pipeline be fine?

Reuses `eval/run.py`'s own `run_baseline`/`run_mv_fixed` (frames, seconds)
timers rather than writing a one-off benchmark loop -- same convention as
every other pipeline comparison in this project. No MOT-format ground truth
exists for this tennis clip, so TrackEval scoring is skipped; only the
timing half of eval/run.py is needed here.

The test: process the full clip exactly like a live camera feed would
arrive (source is 30fps), and check whether wall-clock processing time
stays under the video's own real-time duration. If processing takes longer
than the clip's real duration, a live feed would fall permanently behind --
not a batch-speed nuisance, a hard feasibility failure for real-time use
(e.g. live courtside coaching feedback, discussed in the prior session).
"""

import pathlib
import sys

REPO_ROOT = pathlib.Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT / "eval"))
import run as eval_run  # noqa: E402

VIDEO = REPO_ROOT / "results" / "tennis_video.mp4"
OUT_DIR = REPO_ROOT / "outputs" / "live_feasibility"
SOURCE_FPS = 30

# A real match runs far longer than this 60s clip; project the same
# per-frame processing rate out to a realistic single-set duration to show
# whether lag is a rounding error or something that compounds badly.
PROJECT_SECONDS = 45 * 60  # ~45 min, one set including changeovers


def verdict(pipeline_name, frames, proc_seconds, source_seconds):
    achieved_fps = frames / proc_seconds
    lag = proc_seconds - source_seconds
    keeps_up = proc_seconds <= source_seconds
    projected_lag = (proc_seconds / source_seconds - 1) * PROJECT_SECONDS
    print(f"\n{pipeline_name}:")
    print(f"  {frames} frames in {proc_seconds:.1f}s wall-clock "
          f"({achieved_fps:.1f} fps, source is {SOURCE_FPS} fps)")
    print(f"  clip real-time duration: {source_seconds:.1f}s")
    if keeps_up:
        print(f"  KEEPS UP with live feed -- {-lag:.1f}s of slack on this clip")
    else:
        print(f"  FALLS BEHIND live feed by {lag:.1f}s on this clip alone")
    sign = "behind" if projected_lag > 0 else "ahead"
    print(f"  projected over a {PROJECT_SECONDS/60:.0f}-min set: "
          f"{abs(projected_lag):.0f}s {sign} real time")
    return keeps_up


def main():
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    print("running baseline (full decode + detect every frame)...")
    b_frames, b_seconds = eval_run.run_baseline(VIDEO, OUT_DIR / "baseline.txt")
    source_seconds = b_frames / SOURCE_FPS

    print("running mv-fixed (anchor every 5th frame)...")
    m_frames, m_seconds = eval_run.run_mv_fixed(VIDEO, OUT_DIR / "mv_fixed.txt")

    verdict("baseline", b_frames, b_seconds, source_seconds)
    verdict("mv-fixed", m_frames, m_seconds, source_seconds)


if __name__ == "__main__":
    main()
