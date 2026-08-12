"""PULSE on CAVIAR's Meet_WalkTogether1 (INRIA lobby, single wide-angle
camera) -- validates the group/companion-detection metric
(mvtrack.analytics.group_dwell) against a real scenario CAVIAR's own
description names directly: two people meet and walk together.

Meet_Crowd.mpg (a second real scenario, same server) 403'd consistently on
every retry this session while Meet_WalkTogether1.mpg succeeded immediately
with an identical request -- not chased further; one real validated
companion-pair example is enough to confirm the mechanism works, and this
project doesn't block on a flaky secondary download when the primary
succeeded cleanly.

Calibration: CAVIAR's own published pixel(col,row)->world(X,Y cm)
correspondence table for the INRIA 1st-set camera, 384x288-native scaling
(the non-commented point block on the CAVIAR TestScenarios page -- a
second, commented-out block with larger pixel values exists for a different
image scaling and was NOT used). Same cv2.getPerspectiveTransform exact-fit
approach already validated for CAVIAR shop's cor/front views.

Note: "walk together" does not imply either person ever stops -- unlike
every other PULSE scenario so far, this clip may produce zero dwellers by
design. The real thing being validated here is the underlying companion-
pair proximity signal (mvtrack.analytics.group_dwell.find_companion_pairs),
not the dwell-gated classify_group_dwell wrapper, which only matters once
some zone-stopping behavior also exists.
"""

import pathlib
import sys

import cv2
import numpy as np

REPO_ROOT = pathlib.Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT / "eval"))
sys.path.insert(0, str(REPO_ROOT / "scripts"))
import run as eval_run  # noqa: E402
from epfl_rlc_fusion import load_mot_tracks  # noqa: E402 -- model-agnostic, reused as-is
from mvtrack.analytics import DwellParams, find_companion_pairs  # noqa: E402
from mvtrack.analytics import track_and_classify_dwells as track_and_classify_dwells_shared  # noqa: E402

DATA_ROOT = REPO_ROOT / "data" / "caviar_meet"
VIDEO_ROOT = DATA_ROOT / "videos"
OUT_DIR = REPO_ROOT / "outputs" / "caviar_meet_fusion"

FRAME_DT = 1.0 / 25
MIN_DWELL_SECONDS = 3.0  # short clip (708 frames / 28.3s) -- and dwelling isn't even the expected behavior here
MAX_DWELL_RADIUS = 60.0  # cm
MIN_DWELL_RADIUS = 5.0  # cm

# Real published correspondence points, INRIA 1st-set camera, 384x288 native.
_CALIB_PX = np.array([(64, 88), (211, 40), (349, 184), (39, 187)], dtype=np.float32)
_CALIB_WORLD = np.array([(0, 671.5), (1116, 670), (1545, 190), (0, 0)], dtype=np.float32)


def load_calibration() -> np.ndarray:
    return cv2.getPerspectiveTransform(_CALIB_PX, _CALIB_WORLD)


def pixel_to_ground(px: np.ndarray, H: np.ndarray) -> np.ndarray:
    homog = np.concatenate([px.astype(np.float64), np.ones((len(px), 1))], axis=1)
    world = homog @ H.T
    return world[:, :2] / world[:, 2:3]


def foot_points_by_frame(tracks_by_frame, H) -> dict:
    per_frame_world = {}
    for frame, boxes in tracks_by_frame.items():
        boxes = np.array(boxes)
        foot = np.stack([(boxes[:, 0] + boxes[:, 2]) / 2.0, boxes[:, 3]], axis=1)
        per_frame_world[frame] = pixel_to_ground(foot, H)
    return per_frame_world


def run_mvtrack(anchor_interval=5):
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    video = VIDEO_ROOT / "meet.mp4"
    out = OUT_DIR / "mv_fixed_meet.txt"
    print("running mv-fixed on meet.mp4...")
    frames, seconds = eval_run.run_mv_fixed(video, out, anchor_interval=anchor_interval)
    print(f"  meet: {frames} frames in {seconds:.1f}s ({frames/seconds:.1f} fps)")
    return load_mot_tracks(str(out))


def main():
    H = load_calibration()
    tracks_by_frame = run_mvtrack()
    per_frame_world = foot_points_by_frame(tracks_by_frame, H)

    params = DwellParams(
        frame_dt=FRAME_DT, min_dwell_seconds=MIN_DWELL_SECONDS,
        min_dwell_radius=MIN_DWELL_RADIUS, max_dwell_radius=MAX_DWELL_RADIUS,
        max_step=150.0, max_age=10,
    )
    tracks, dwellers = track_and_classify_dwells_shared(per_frame_world, params)
    print(f"\n{len(tracks)} total tracks, {len(dwellers)} classify as dwelling "
          f"(>= {MIN_DWELL_SECONDS}s within {MAX_DWELL_RADIUS}cm)")

    companions = find_companion_pairs(tracks, proximity_cm=150.0, min_fraction_together=0.6)
    print(f"\n{len(companions)} companion pair(s) (>= 60% of shared time within 150cm):")
    for tid_a, tid_b, frac in sorted(companions, key=lambda c: -c[2]):
        hist_a, hist_b = dict(tracks[tid_a]), dict(tracks[tid_b])
        lo = max(min(f for f, _ in tracks[tid_a]), min(f for f, _ in tracks[tid_b]))
        hi = min(max(f for f, _ in tracks[tid_a]), max(f for f, _ in tracks[tid_b]))
        print(f"    track {tid_a} + track {tid_b}: together {frac*100:.0f}% of overlap, "
              f"frames {lo}-{hi}")


if __name__ == "__main__":
    main()
