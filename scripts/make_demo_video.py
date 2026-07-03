"""Side-by-side demo: full-decode-every-frame tracking vs. mv-adaptive,
with anchor firings visualized. Both sides use the same detector and the
same MVTracker/IoU-Hungarian association logic -- the only difference is
whether the "baseline" side always calls step_anchor (full detect every
frame) or the "mv-adaptive" side mixes step_anchor with step_propagate on
a residual-energy-scheduled subset of frames. That keeps the comparison
apples-to-apples: it isolates the effect of MV propagation + scheduling,
not incidental differences between two unrelated tracker implementations.

Usage: python scripts/make_demo_video.py [--video data/people_baseline.mp4]
Writes outputs/demo_side_by_side.mp4
"""

import argparse
import pathlib
import sys

import cv2
import numpy as np

ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from mvtrack.detect import Detector
from mvtrack.extract import iter_frames_with_mvs
from mvtrack.sched import Adaptive
from mvtrack.track import MVTracker

PERSON_CLS = 0


def color_for_id(track_id: int) -> tuple[int, int, int]:
    rng = np.random.default_rng(track_id * 2654435761 % (2**32))
    return tuple(int(c) for c in rng.integers(60, 255, size=3))


def draw_tracks(img: np.ndarray, tracks, label: str, anchor: bool) -> np.ndarray:
    out = img.copy()
    for tr in tracks:
        x0, y0, x1, y1 = tr.box.astype(int)
        color = color_for_id(tr.id)
        cv2.rectangle(out, (x0, y0), (x1, y1), color, 2)
        cv2.putText(out, str(tr.id), (x0, max(0, y0 - 5)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
    border = (0, 215, 255) if anchor else (60, 60, 60)
    cv2.rectangle(out, (0, 0), (out.shape[1] - 1, out.shape[0] - 1), border, 6)
    cv2.putText(out, label, (10, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
    if anchor:
        cv2.putText(out, "ANCHOR", (10, out.shape[0] - 15),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 215, 255), 2)
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--video", default=str(ROOT / "data" / "people_baseline.mp4"))
    ap.add_argument("--out", default=str(ROOT / "outputs" / "demo_side_by_side.mp4"))
    ap.add_argument("--max-frames", type=int, default=300)
    args = ap.parse_args()

    detector = Detector("yolov8n.pt")
    baseline_tracker = MVTracker()
    adaptive_tracker = MVTracker()
    scheduler = Adaptive()

    writer = None
    n = 0
    for fmv, frame in iter_frames_with_mvs(args.video):
        n += 1
        if n > args.max_frames:
            break
        img = frame.to_ndarray(format="bgr24")

        boxes, scores, cls_ids = detector(img)
        keep = cls_ids == PERSON_CLS
        base_tracks = baseline_tracker.step_anchor(boxes[keep], scores[keep])

        is_anchor = scheduler.should_anchor(fmv)
        if is_anchor:
            adapt_tracks = adaptive_tracker.step_anchor(boxes[keep], scores[keep])
        else:
            adapt_tracks = adaptive_tracker.step_propagate(fmv)

        left = draw_tracks(img, base_tracks, "BASELINE (every frame)", anchor=True)
        right = draw_tracks(img, adapt_tracks, "MV-ADAPTIVE", anchor=is_anchor)
        combined = np.hstack([left, right])

        if writer is None:
            h, w = combined.shape[:2]
            writer = cv2.VideoWriter(
                args.out, cv2.VideoWriter_fourcc(*"mp4v"), 30.0, (w, h)
            )
        writer.write(combined)

    if writer is not None:
        writer.release()
    print(f"wrote {n} frames -> {args.out}")


if __name__ == "__main__":
    main()
