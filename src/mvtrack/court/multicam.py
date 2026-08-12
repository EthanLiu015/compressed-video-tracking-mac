"""General N-camera ground-plane registration.

Built after the EPFL-RLC fusion work turned out to be camera-count-specific
in exactly the way that blocks reuse: `register_to_cam0()` there is a
hardcoded `if cam == 2 / elif cam == 0 / else raise`, and the registration
transform itself (a rotation + translation aligning cam2 onto cam0) was
computed once, interactively, and typed into the script as constants.
Neither generalizes to a museum's N cameras, or any dataset where the
camera count isn't 2 and known in advance.

What *does* generalize, and is what's actually implemented here: given any
number of cameras, each with its own working pixel-to-ground-plane
function (regardless of camera model -- Tsai, OpenCV pinhole, or a
hand-fit homography, see `CameraModel` below), and one shared timestamp
with an image from each camera, automatically find real point
correspondences (ORB + RANSAC-homography, same technique validated on
EPFL-RLC), fit a rigid registration transform per camera (Kabsch/
Procrustes), and *keep or drop each camera based on its own fit quality* --
turning "cam1 failed so I excluded it" from a one-off human judgment call
into a repeatable, automatic decision that runs the same way regardless of
how many cameras are given.
"""

from dataclasses import dataclass
from typing import Callable, Protocol

import cv2
import numpy as np


class CameraModel(Protocol):
    """Anything that can turn Nx2 pixel points into Nx2 ground-plane world
    points (mm or whatever consistent unit) implements this. Concrete
    examples in this codebase: `mvtrack.court.homography` (tennis, hand-fit
    from court-line correspondences), the OpenCV-pinhole projector in
    `scripts/wildtrack_fusion.py`, the Tsai-model projector in
    `scripts/epfl_rlc_fusion.py`. None of those need to change -- this
    module only needs each one exposed as a plain callable."""

    def __call__(self, pixel_points: np.ndarray) -> np.ndarray: ...


@dataclass
class CameraRegistration:
    included: bool
    rotation: np.ndarray  # 2x2
    translation_ref: np.ndarray  # 2, centroid in reference camera's frame
    translation_cam: np.ndarray  # 2, centroid in this camera's own frame
    residual_m: float  # mean post-alignment residual against the reference, in meters
    n_correspondences: int

    def apply(self, world_pts: np.ndarray) -> np.ndarray:
        """World points (this camera's own frame, mm) -> reference frame (mm)."""
        if not self.included:
            raise ValueError("this camera failed registration -- check .residual_m before calling apply()")
        return (world_pts - self.translation_cam) @ self.rotation.T + self.translation_ref


def _orb_correspondences(img_a: np.ndarray, img_b: np.ndarray, ransac_thresh_px=5.0):
    """Real matched pixel pairs between two images of the same static scene,
    filtered to geometrically-consistent inliers -- not manual point-picking,
    which was error-prone enough on EPFL-RLC's angle-differences to produce
    one bad correspondence set (57-90m residual) before this automated
    approach replaced it."""
    orb = cv2.ORB_create(nfeatures=2000)
    kp_a, des_a = orb.detectAndCompute(img_a, None)
    kp_b, des_b = orb.detectAndCompute(img_b, None)
    if des_a is None or des_b is None:
        return np.zeros((0, 2)), np.zeros((0, 2))
    bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
    matches = bf.match(des_a, des_b)
    if len(matches) < 4:
        return np.zeros((0, 2)), np.zeros((0, 2))
    pts_a = np.array([kp_a[m.queryIdx].pt for m in matches])
    pts_b = np.array([kp_b[m.trainIdx].pt for m in matches])
    H, mask = cv2.findHomography(pts_a, pts_b, cv2.RANSAC, ransac_thresh_px)
    if H is None:
        return np.zeros((0, 2)), np.zeros((0, 2))
    inliers = mask.ravel().astype(bool)
    return pts_a[inliers], pts_b[inliers]


def _fit_rigid(world_ref: np.ndarray, world_cam: np.ndarray):
    """Kabsch/Procrustes: rotation + translation aligning world_cam onto
    world_ref. Returns (R, c_ref, c_cam, mean_residual)."""
    c_ref = world_ref.mean(axis=0)
    c_cam = world_cam.mean(axis=0)
    H = (world_cam - c_cam).T @ (world_ref - c_ref)
    U, _, Vt = np.linalg.svd(H)
    R = Vt.T @ U.T
    if np.linalg.det(R) < 0:
        Vt[-1, :] *= -1
        R = Vt.T @ U.T
    aligned = (world_cam - c_cam) @ R.T + c_ref
    residual = np.linalg.norm(aligned - world_ref, axis=1).mean()
    return R, c_ref, c_cam, residual


def register_cameras(
    camera_models: dict[int, CameraModel],
    reference_cam: int,
    shared_images: dict[int, np.ndarray],
    max_residual_m: float = 3.0,
    min_correspondences: int = 6,
) -> dict[int, CameraRegistration]:
    """For every camera != reference_cam: find real ORB correspondences
    against the reference's shared-timestamp image, project both sides
    through each camera's own (already-working) ground-plane function, fit
    a rigid transform, and include the camera only if enough correspondences
    survived RANSAC *and* the fit residual is small enough to trust -- the
    two failure modes actually seen (EPFL-RLC cam1): too few real matches,
    or a fit that "succeeds" numerically but leaves meters of residual
    error because the matches weren't real correspondences to begin with.
    """
    results = {
        reference_cam: CameraRegistration(
            included=True, rotation=np.eye(2), translation_ref=np.zeros(2),
            translation_cam=np.zeros(2), residual_m=0.0, n_correspondences=0,
        )
    }
    ref_img = shared_images[reference_cam]
    ref_model = camera_models[reference_cam]

    for cam, model in camera_models.items():
        if cam == reference_cam:
            continue
        px_ref, px_cam = _orb_correspondences(ref_img, shared_images[cam])
        n = len(px_ref)
        if n < min_correspondences:
            results[cam] = CameraRegistration(
                included=False, rotation=np.eye(2), translation_ref=np.zeros(2),
                translation_cam=np.zeros(2), residual_m=float("inf"), n_correspondences=n,
            )
            continue
        world_ref = ref_model(px_ref)
        world_cam = model(px_cam)
        # A CameraModel may internally reject unstable points (e.g. the
        # grazing-ray filter in scripts/epfl_rlc_fusion.py) -- it must
        # return one row per input point, in order, using NaN for a
        # rejected point, so the ref/cam correspondence pairing survives.
        # Drop any pair where either side is NaN before fitting.
        valid = ~(np.isnan(world_ref).any(axis=1) | np.isnan(world_cam).any(axis=1))
        world_ref, world_cam = world_ref[valid], world_cam[valid]
        if len(world_ref) < min_correspondences:
            results[cam] = CameraRegistration(
                included=False, rotation=np.eye(2), translation_ref=np.zeros(2),
                translation_cam=np.zeros(2), residual_m=float("inf"), n_correspondences=len(world_ref),
            )
            continue
        R, c_ref, c_cam, residual_mm = _fit_rigid(world_ref, world_cam)
        residual_m = residual_mm / 1000.0
        results[cam] = CameraRegistration(
            included=residual_m <= max_residual_m,
            rotation=R, translation_ref=c_ref, translation_cam=c_cam,
            residual_m=residual_m, n_correspondences=n,
        )
    return results


def fuse_multicam_points(
    per_cam_points: dict[int, np.ndarray],
    registrations: dict[int, CameraRegistration],
    fuse_radius: float,
) -> np.ndarray:
    """Register every included camera's world points onto the reference
    frame and cluster points within `fuse_radius` as one real person --
    the N-camera generalization of the 2-camera fuse_points() in
    epfl_rlc_fusion.py (that logic itself already generalizes fine; only
    the "which cameras and what transform" part didn't)."""
    all_pts = []
    for cam, pts in per_cam_points.items():
        reg = registrations.get(cam)
        if reg is None or not reg.included or len(pts) == 0:
            continue
        all_pts.append(reg.apply(pts))
    points = np.concatenate(all_pts, axis=0) if all_pts else np.zeros((0, 2))

    if len(points) == 0:
        return points
    used = np.zeros(len(points), dtype=bool)
    fused = []
    for i in range(len(points)):
        if used[i]:
            continue
        dists = np.linalg.norm(points - points[i], axis=1)
        group = (dists <= fuse_radius) & ~used
        used |= group
        fused.append(points[group].mean(axis=0))
    return np.array(fused)
