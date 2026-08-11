"""Opponent-relative positioning stat for tennis broadcast footage.

Standard recovery-speed stats (Hawk-Eye, broadcast graphics) measure return
to a *fixed* geometric court center. This measures something the fixed-center
version can't: whether a player recovers toward the *dynamically correct*
spot given where the opponent actually is right now -- the angle-bisector
target real coaches teach, not the static center mark.

Requires a wide-angle, locked-camera frame range (see `mvtrack.court`
docstring -- the homography is meaningless on close-up/replay frames).
`find_wide_angle_range` gates for that with a cheap background-similarity
check against the calibration frame, no ML needed: the wide shot's
background (court, chair umpire, ad boards) is static, so frames from a
different camera setup look nothing like it even downsampled.

Two detections per gated frame (top-2 person confs, split near/far by box
bottom y) rather than the full MOT-style tracker: with exactly 2 players and
a clear near/far separation by court position, id-association across a short
continuous camera-locked segment is trivial and the tracker's re-association
machinery (built for MOT17-style crowds) is unneeded complexity here.
"""

import pathlib

import av
import cv2
import numpy as np

from mvtrack.court.homography import (
    HALF_LENGTH,
    HALF_WIDTH_SINGLES,
    fit_homography,
    pixel_to_court,
)
from mvtrack.detect import Detector, pick_device

PERSON_CLS = 0
CALIBRATION_FRAME = 30
FPS = 30
MAX_SPRINT_MPS = 9.0  # elite sprint speed, generous margin over real tennis movement
REPO_ROOT = pathlib.Path(__file__).resolve().parent.parent
VIDEO = REPO_ROOT / "results" / "tennis_video.mp4"


def find_wide_angle_range(video_path, calibration_frame=CALIBRATION_FRAME, corr_threshold=0.9):
    """Return (start, end) exclusive frame indices of the contiguous wide-angle
    run containing `calibration_frame`, via downsampled-grayscale correlation
    against that frame."""
    container = av.open(str(video_path))
    stream = container.streams.video[0]
    frames = list(container.decode(stream))
    container.close()

    ref = cv2.cvtColor(frames[calibration_frame].to_ndarray(format="bgr24"), cv2.COLOR_BGR2GRAY)
    ref_small = cv2.resize(ref, (64, 36)).astype(np.float32).flatten()

    def is_wide(i):
        img = cv2.cvtColor(frames[i].to_ndarray(format="bgr24"), cv2.COLOR_BGR2GRAY)
        small = cv2.resize(img, (64, 36)).astype(np.float32).flatten()
        return np.corrcoef(ref_small, small)[0, 1] > corr_threshold

    start = calibration_frame
    while start > 0 and is_wide(start - 1):
        start -= 1
    end = calibration_frame
    while end < len(frames) - 1 and is_wide(end + 1):
        end += 1
    return start, end + 1, frames


def split_players(boxes, scores, top_n=2):
    """Top-N person detections by confidence, sorted near-to-far by box-bottom y
    (bigger y = lower on screen = closer to camera in this locked angle)."""
    if len(boxes) < top_n:
        return None
    order = np.argsort(-scores)[:top_n]
    picked = boxes[order]
    picked = picked[np.argsort(-picked[:, 3])]  # sort by y1 (box bottom), descending
    return picked  # [near_player, far_player]


def foot_point(box_xyxy):
    x0, y0, x1, y1 = box_xyxy
    return ((x0 + x1) / 2.0, y1)  # bottom-center = ground contact point


def optimal_recovery_x(opponent_xy, player_baseline_y):
    """Angle-bisector recovery target: the x-position on `player_baseline_y`
    that bisects the shot angle opponent has into the two extreme corners of
    the singles court at that baseline. This is the geometric definition
    coaches call 'the bisector' -- not the fixed court-center approximation."""
    ox, oy = opponent_xy
    corner_l = np.array([-HALF_WIDTH_SINGLES, player_baseline_y]) - [ox, oy]
    corner_r = np.array([HALF_WIDTH_SINGLES, player_baseline_y]) - [ox, oy]
    corner_l = corner_l / np.linalg.norm(corner_l)
    corner_r = corner_r / np.linalg.norm(corner_r)
    bisector = corner_l + corner_r
    if abs(bisector[1]) < 1e-9:
        return 0.0  # opponent dead-center on the net line; bisector is along it
    t = (player_baseline_y - oy) / bisector[1]
    return ox + t * bisector[0]


def main():
    start, end, frames = find_wide_angle_range(VIDEO)
    print(f"wide-angle segment: frames [{start}, {end}) ({end - start} frames, "
          f"{(end - start) / 30:.1f}s @ 30fps)")

    H = fit_homography()
    detector = Detector("yolov8s.pt", device=pick_device(), conf=0.3)

    rows = []
    last_near, last_near_frame = None, None
    last_far, last_far_frame = None, None
    dropped_speed = 0
    for i in range(start, end):
        img = frames[i].to_ndarray(format="bgr24")
        boxes, scores, cls_ids = detector(img)
        keep = cls_ids == PERSON_CLS
        players = split_players(boxes[keep], scores[keep])
        if players is None:
            continue
        near_box, far_box = players
        near_xy = pixel_to_court(H, [foot_point(near_box)])[0]
        far_xy = pixel_to_court(H, [foot_point(far_box)])[0]

        # Sanity gate: a real near/far split keeps each player on their own
        # side of the net. Occasionally "top-2 by confidence" grabs a ball
        # kid or line judge instead of the actual player (observed on 2/113
        # frames here) -- that detection lands on the wrong side of the net
        # entirely, a physically impossible track position, not just noise.
        if near_xy[1] >= 0 or far_xy[1] <= 0:
            continue

        # Continuity gate: same failure mode (top-2-by-confidence grabbing a
        # ball kid/line judge instead of the real player) can also land on
        # the *correct* side of the net, so the sign check above misses it --
        # but it still shows up as an implausible frame-to-frame jump (a real
        # player can't cover >MAX_SPRINT_MPS worth of court in the elapsed
        # time). Observed directly: far player's x bounced ~1.5m<->6.4m
        # between single frames pre-filter, a ~4m/frame jump at 30fps.
        if last_near is not None:
            dt = (i - last_near_frame) / FPS
            if np.linalg.norm(near_xy - last_near) > MAX_SPRINT_MPS * dt:
                dropped_speed += 1
                continue
        if last_far is not None:
            dt = (i - last_far_frame) / FPS
            if np.linalg.norm(far_xy - last_far) > MAX_SPRINT_MPS * dt:
                dropped_speed += 1
                continue
        last_near, last_near_frame = near_xy, i
        last_far, last_far_frame = far_xy, i

        near_target_x = optimal_recovery_x(far_xy, -HALF_LENGTH)
        far_target_x = optimal_recovery_x(near_xy, HALF_LENGTH)
        near_dev = abs(near_xy[0] - near_target_x)
        far_dev = abs(far_xy[0] - far_target_x)

        rows.append((i, *near_xy, near_target_x, near_dev, *far_xy, far_target_x, far_dev))

    if not rows:
        print("no frames with 2 detected players in the wide-angle segment")
        return

    print(f"dropped {dropped_speed} frames on implausible-jump continuity gate")
    arr = np.array(rows)
    out = REPO_ROOT / "results" / "court_positioning.csv"
    header = ("frame,near_x,near_y,near_target_x,near_dev_m,"
              "far_x,far_y,far_target_x,far_dev_m")
    np.savetxt(out, arr, delimiter=",", header=header, comments="", fmt="%.4f")
    print(f"wrote {out} ({len(rows)} frames)")
    print(f"near player mean deviation from bisector target: {arr[:,4].mean():.2f}m "
          f"(fixed-center baseline would compare against x=0.0)")
    print(f"far  player mean deviation from bisector target: {arr[:,8].mean():.2f}m")


if __name__ == "__main__":
    main()
