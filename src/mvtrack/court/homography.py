"""Pixel -> real-world-court-meters homography for a fixed, locked broadcast
camera angle.

Valid only for frames matching the calibrated camera setup (see
CALIBRATION_SOURCE) -- a broadcast cuts to close-ups, replays, and graphics
between points, and this homography is meaningless on those frames. Filtering
to only the locked wide-angle shot is a separate, unbuilt problem; this
module assumes that's already been handled (or is applied to a manually
restricted frame range) and just does the pixel -> meters mapping once
you're on the right frame.

Court coordinate system (world, meters): origin at net center. +x toward the
right doubles sideline as seen in the calibration frame, +y toward the far
baseline (away from camera). Standard ITF singles/doubles dimensions.
"""

import numpy as np
import cv2

# Standard ITF court dimensions, meters (net at the origin/half-court line).
HALF_LENGTH = 11.885  # net -> baseline
HALF_WIDTH_DOUBLES = 5.485  # net -> doubles sideline
HALF_WIDTH_SINGLES = 4.115  # net -> singles sideline
SERVICE_LINE_DIST = 6.40  # net -> service line

CALIBRATION_SOURCE = "results/tennis_video.mp4, frame 30 (locked wide broadcast angle)"

# (pixel_x, pixel_y) -> (court_x, court_y) meters. Picked by eye against a
# 40px reference grid overlaid on the calibration frame -- expect a few
# pixels of manual-pick error, not sub-pixel precise. 8 correspondences (4
# baseline corners + 4 service-line/singles-sideline corners), overdetermined
# vs. the 4 a homography strictly needs, so the fit reports real residual
# error instead of being trivially exact.
REFERENCE_POINTS = [
    ((413, 168), (-HALF_WIDTH_DOUBLES, HALF_LENGTH)),  # far doubles corner, left
    ((855, 168), (HALF_WIDTH_DOUBLES, HALF_LENGTH)),  # far doubles corner, right
    ((222, 570), (-HALF_WIDTH_DOUBLES, -HALF_LENGTH)),  # near doubles corner, left
    ((1048, 570), (HALF_WIDTH_DOUBLES, -HALF_LENGTH)),  # near doubles corner, right
    ((463, 213), (-HALF_WIDTH_SINGLES, SERVICE_LINE_DIST)),  # far service line x singles, left
    ((805, 213), (HALF_WIDTH_SINGLES, SERVICE_LINE_DIST)),  # far service line x singles, right
    ((383, 423), (-HALF_WIDTH_SINGLES, -SERVICE_LINE_DIST)),  # near service line x singles, left
    ((883, 423), (HALF_WIDTH_SINGLES, -SERVICE_LINE_DIST)),  # near service line x singles, right
]


def fit_homography(reference_points=REFERENCE_POINTS) -> np.ndarray:
    """Least-squares pixel -> court-meters homography from >=4 correspondences."""
    px = np.array([p for p, _ in reference_points], dtype=np.float64)
    world = np.array([w for _, w in reference_points], dtype=np.float64)
    H, _ = cv2.findHomography(px, world, method=0)
    if H is None:
        raise ValueError("homography fit failed -- degenerate reference points")
    return H


def pixel_to_court(H: np.ndarray, points_px) -> np.ndarray:
    """Map Nx2 pixel points to Nx2 court coordinates (meters)."""
    pts = np.asarray(points_px, dtype=np.float64).reshape(-1, 1, 2)
    world = cv2.perspectiveTransform(pts, H)
    return world.reshape(-1, 2)


def court_to_pixel(H: np.ndarray, points_world) -> np.ndarray:
    """Map Nx2 court coordinates (meters) back to Nx2 pixel points."""
    H_inv = np.linalg.inv(H)
    pts = np.asarray(points_world, dtype=np.float64).reshape(-1, 1, 2)
    px = cv2.perspectiveTransform(pts, H_inv)
    return px.reshape(-1, 2)


def reprojection_error(H: np.ndarray, reference_points=REFERENCE_POINTS) -> np.ndarray:
    """Per-point distance (meters) between fitted and true world coordinates --
    the check that REFERENCE_POINTS were picked consistently, since 8 points
    into a homography (4 DOF minimum) is overdetermined, not a trivial fit."""
    px = [p for p, _ in reference_points]
    true_world = np.array([w for _, w in reference_points], dtype=np.float64)
    fitted = pixel_to_court(H, px)
    return np.linalg.norm(fitted - true_world, axis=1)


if __name__ == "__main__":
    H = fit_homography()
    err = reprojection_error(H)
    for (px, world), e in zip(REFERENCE_POINTS, err):
        fitted = pixel_to_court(H, [px])[0]
        print(
            f"{px} -> fit {tuple(np.round(fitted, 3))}  true {world}  "
            f"err={e * 100:.1f}cm"
        )
    print(f"mean reprojection error: {err.mean() * 100:.1f}cm, max: {err.max() * 100:.1f}cm")
