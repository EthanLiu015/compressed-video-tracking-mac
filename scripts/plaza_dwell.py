"""Stage 1 of PULSE: single-camera dwell/lingering detection.

Whyte's core finding in "The Social Life of Small Urban Spaces" wasn't
where people walk, it's where they *stop* -- what makes someone linger
instead of just passing through (seating, sun, street performers, other
people). This is the single-camera slice of that: per track, is this
person passing through or dwelling in one spot?

Per track: record foot-point (box bottom-center) position every frame it's
visible. A track "dwells" if it survives long enough (MIN_DWELL_FRAMES)
and its whole position history stays within a small radius (MAX_DWELL_RADIUS)
-- i.e. real duration with low displacement, not just a long track that
happens to be walking slowly across the whole frame.

Tracking via ultralytics' own ByteTrack (same as eval/run.py's run_baseline
and scripts/checkpoint_reid.py) -- this stage doesn't need mv-tracking's
box-propagation, just clean per-clip identities. The multi-camera ground-
plane fusion (stage 2) builds on top of this per-camera track history.
"""

import pathlib

import cv2
import numpy as np
from ultralytics import YOLO

from mvtrack.detect import pick_device

PERSON_CLS = 0
MIN_DWELL_FRAMES = 90  # >=3s @ 30fps of sustained presence
MAX_DWELL_RADIUS = 40  # px -- whole track history must stay within this of its centroid
# A live person sways/shifts even standing "still" -- exact-zero jitter across
# 90+ frames is physically implausible for a real human and is instead the
# signature of a static object flickering in and out of detection at low
# conf (found directly: ~15 overlapping near-identical-coordinate "dwellers"
# piled at one spot, radius 0-2px, on real Times Square footage). Real
# dwellers in the same run showed 15-39px of natural jitter. Mirrors the
# continuity/plausibility gates already used in scripts/court_positioning.py
# and checkpoint_reid.py.
MIN_DWELL_RADIUS = 5  # px

REPO_ROOT = pathlib.Path(__file__).resolve().parent.parent
VIDEO = REPO_ROOT / "data" / "plaza_ts_north.mp4"
OUT_DIR = REPO_ROOT / "outputs" / "plaza_dwell"
# Times Square's elevated cams put people at ~15-30px native -- default
# imgsz=640 downsamples that below the detector's floor entirely (measured:
# 0 detections on a visibly crowded frame). imgsz=1920 (native res, no
# downsample) + conf=0.1 (vs. the project default 0.25) recovered 36-63 real
# detections on the same kind of frame; verified visually, boxes land on
# individual people at crosswalks/sidewalks. The extreme-density core crowd
# (e.g. right at the TKTS steps) still won't resolve into individual boxes --
# fine here, dwelling behavior lives at the edges/seating, not mid-crosswalk.
DETECT_KWARGS = dict(conf=0.1, imgsz=1920)


def track_positions(video_path, weights="yolov8s.pt"):
    model = YOLO(weights)
    device = pick_device()
    positions = {}  # tid -> list of (frame_idx, x, y)
    last_frame = None
    for i, res in enumerate(model.track(
        source=str(video_path), device=device, tracker="bytetrack.yaml",
        stream=True, verbose=False, classes=[PERSON_CLS], persist=True, **DETECT_KWARGS,
    )):
        last_frame = res.orig_img
        if res.boxes.id is None:
            continue
        ids = res.boxes.id.int().tolist()
        xyxy = res.boxes.xyxy.cpu().numpy()
        for tid, (x0, y0, x1, y1) in zip(ids, xyxy):
            positions.setdefault(tid, []).append((i, (x0 + x1) / 2.0, y1))
    return positions, last_frame


def classify_dwells(positions):
    dwellers = {}
    for tid, pts in positions.items():
        if len(pts) < MIN_DWELL_FRAMES:
            continue
        xy = np.array([(x, y) for _, x, y in pts])
        centroid = xy.mean(axis=0)
        radius = np.linalg.norm(xy - centroid, axis=1).max()
        if MIN_DWELL_RADIUS <= radius <= MAX_DWELL_RADIUS:
            dwellers[tid] = (centroid, len(pts), radius)
    return dwellers


def main():
    print(f"tracking {VIDEO.name}...")
    positions, ref_frame = track_positions(VIDEO)
    print(f"  {len(positions)} total tracks")

    dwellers = classify_dwells(positions)
    print(f"  {len(dwellers)} tracks classify as dwelling "
          f"(>= {MIN_DWELL_FRAMES / 30:.0f}s within {MAX_DWELL_RADIUS}px)")

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    overlay = ref_frame.copy()
    for tid, (centroid, n_frames, radius) in sorted(dwellers.items(), key=lambda t: -t[1][1]):
        cx, cy = centroid.astype(int)
        r = max(int(radius), 10)
        cv2.circle(overlay, (cx, cy), r, (0, 0, 255), 2)
        cv2.putText(overlay, f"{n_frames / 30:.0f}s", (cx - 10, cy - r - 5),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)
        print(f"    track {tid}: {n_frames / 30:.1f}s at ({cx},{cy}), radius {radius:.0f}px")

    out_path = OUT_DIR / "dwell_overlay.png"
    cv2.imwrite(str(out_path), overlay)
    print(f"\nsaved dwell overlay -> {out_path}")


if __name__ == "__main__":
    main()
