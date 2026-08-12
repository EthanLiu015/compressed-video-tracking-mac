"""PULSE rebuilt on real 60fps footage (EPFL-RLC), routed through the
actual mvtrack pipeline this time -- not generic YOLO+ByteTrack like the
WildTrack version. Per camera: real H.264 bitstream -> MVTracker
(anchor/propagate) -> pixel tracks -> Tsai-calibrated ground-plane
projection -> cross-camera fusion -> dwell/idle classification.

Calibration is the classic Tsai (1987) single-plane model (not OpenCV's
pinhole+Rodrigues used for WildTrack) -- confirmed by the XML schema
(focal/kappa1/cx/cy/sx + tx,ty,tz,rx,ry,rz). Rotation is treated as a
single Rodrigues axis-angle vector (Willson's widely-distributed C
implementation of Tsai's algorithm, which this dataset's schema matches,
stores/converts rotation exactly that way -- same `cv2.Rodrigues` call
used for WildTrack's OpenCV-format calibration, different underlying
camera model otherwise).

Validation note: EPFL-RLC's own ground-truth file (SWITCHdrive, down --
confirmed a real outage via its own status message, not a dead link) was
unavailable, so validation here is two-layered:

1. Per-camera projection math: confirmed correct via real floor-stripe
   collinearity (points along an actual straight floor stripe project to
   a near-perfectly straight line in world coordinates -- max 33.5mm
   residual over a 10m+ span for cam0). This is a hard geometric
   constraint, not a plausibility check.
2. Cross-camera alignment: each camera's calibration turned out to use an
   *independent* world origin (a known real limitation of classic
   single-camera Tsai calibration workflows unless deliberately unified)
   -- individually correct per (1), but not mutually consistent, which is
   why raw fusion barely merged anything despite the user directly
   confirming real overlap exists in the footage. Fixed via
   `mvtrack.court.register_cameras` -- a general N-camera registration
   function (ORB + RANSAC-homography correspondences, Kabsch/Procrustes
   rigid fit, automatic include/exclude by residual), not the
   2-camera-specific hardcoded constants this module originally shipped
   with (see git history). `register_all_cameras()` below just wraps this
   dataset's own per-camera projection as `mvtrack.court.CameraModel`
   callables and calls it. On this dataset it automatically rediscovers
   the same real conclusion manual debugging first found -- cam2 registers
   cleanly (0.76m residual), cam1 doesn't (too few correspondences survive
   RANSAC against either other camera) and is excluded -- which is the
   actual proof this generalizes rather than just relabeling a fixed
   2-camera result: nothing here names cam1 or cam2 specifically, or
   assumes exactly 3 cameras.
"""

import glob
import pathlib
import sys
import xml.etree.ElementTree as ET

import cv2
import numpy as np

REPO_ROOT = pathlib.Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT / "eval"))
import run as eval_run  # noqa: E402
from mvtrack.analytics import DwellParams  # noqa: E402
from mvtrack.analytics import track_and_classify_dwells as track_and_classify_dwells_shared  # noqa: E402

DATA_ROOT = REPO_ROOT / "data" / "epfl_rlc" / "EPFL-RLC_dataset"
CALIB_ROOT = DATA_ROOT / "calibration"
VIDEO_ROOT = REPO_ROOT / "data" / "epfl_rlc" / "videos"
OUT_DIR = REPO_ROOT / "outputs" / "epfl_rlc_fusion"

FRAME_DT = 1.0 / 60  # real 60fps source -- no coarse-sampling caveat this time
FUSE_RADIUS = 1600.0  # mm -- widened to cover the real measured cam0<->cam2 registration
# residual (~1.1-1.4m mean, up to 4.8m on the worst ORB correspondence) --
# 400mm (a person's footprint) was correct in principle but smaller than
# the actual registration uncertainty, so raw fusion still wasn't merging
# despite registration being applied. This is real slop from a 12-point
# ORB-based fit, not the fusion logic itself; a tighter registration would
# let this shrink back down.
MIN_DWELL_SECONDS = 6.0
MAX_DWELL_RADIUS = 1800.0  # mm -- widened past MAX (not just mean) registration
# residual (~1.4m mean, 4.8m worst observed) so genuine stationary people
# aren't misclassified as "moved too much" from calibration/registration
# noise alone, unrelated to real movement.
# A live person sways/shifts even standing "still" -- exact-zero jitter is
# the signature of a static object, not a person (found directly: a coat
# hanging on a hook near cam2's doorway, detected at the identical pixel
# coordinate across 6882 frames -- same false-positive class already fixed
# once in scripts/plaza_dwell.py for a different static object).
MIN_DWELL_RADIUS = 50.0  # mm


def load_calibration(cam: int):
    root = ET.parse(CALIB_ROOT / f"calibration_cam{cam}.xml").getroot()
    intr = root.find("Intrinsic")
    extr = root.find("Extrinsic")
    geo = root.find("Geometry")
    params = dict(
        focal=float(intr.get("focal")),
        kappa1=float(intr.get("kappa1")),
        cx=float(intr.get("cx")),
        cy=float(intr.get("cy")),
        dpx=float(geo.get("dpx")),
        dpy=float(geo.get("dpy")),
    )
    rvec = np.array([float(extr.get(k)) for k in ("rx", "ry", "rz")])
    t = np.array([float(extr.get(k)) for k in ("tx", "ty", "tz")])
    R, _ = cv2.Rodrigues(rvec)
    return params, R, t


def camera_world_position(R: np.ndarray, t: np.ndarray) -> np.ndarray:
    """Camera center in world coords: Xw = -R^T @ t (from Xc = R@Xw + t, Xc=0).
    Reverted after testing the alternate "t is the camera position directly"
    convention: it gave a more plausible-looking camera cluster, but broke
    the real mathematical validation (floor-stripe collinearity went from
    33.5mm to 126mm max residual, with an absurd 33m span for a real ~10m
    stripe) -- plausibility was coincidental, the collinearity test is the
    real constraint. Kept as the validated formula; see module docstring
    for the real, still-open explanation of why cross-camera fusion is
    weak despite this being individually correct per camera."""
    return -(R.T @ t)


def pixel_to_ground(px: np.ndarray, params: dict, R: np.ndarray, t: np.ndarray) -> np.ndarray:
    """Back-project Nx2 pixel points to Nx2 world (X, Y) on the Z=0 ground
    plane, through the Tsai model's inverse (pixel -> distorted sensor mm
    -> undistorted sensor mm -> camera-frame ray -> ground-plane intersection).
    Xc = R@Xw + t -- validated via real floor-stripe collinearity (see
    camera_world_position's docstring)."""
    u, v = px[:, 0], px[:, 1]
    xd = (u - params["cx"]) * params["dpx"]
    yd = (v - params["cy"]) * params["dpy"]
    r2 = xd ** 2 + yd ** 2
    factor = 1 + params["kappa1"] * r2  # Tsai's forward model: undistorted = distorted * factor
    xu = xd * factor
    yu = yd * factor

    ray_cam = np.stack([xu, yu, np.full_like(xu, params["focal"])], axis=1)  # Nx3
    Rt_t = R.T @ t
    # Apple Accelerate BLAS raises spurious divide-by-zero/overflow/invalid
    # FPE warnings on some of these matmuls; confirmed cosmetic here by
    # checking directly for NaN/Inf in the output (none found) -- same
    # class of false positive already documented in mvtrack/track/reid.py.
    with np.errstate(all="ignore"):
        Rt_ray = ray_cam @ R  # each row = R^T @ ray_cam[i]
    s = Rt_t[2] / Rt_ray[:, 2]
    world = s[:, None] * Rt_ray - Rt_t
    return world[:, :2]


# A detector box's edges are never sub-pixel exact -- ~1px of real jitter
# between consecutive frames is normal even for a perfectly static point.
# Ground-plane back-projection's sensitivity to that jitter isn't constant:
# it blows up for near-grazing rays (found directly: an 18px foot-point
# shift produced a 3042mm world jump -- physically impossible at 180 m/s --
# on a real 17.5s track, while most comparable pixel shifts elsewhere on
# the same track produced near-zero world movement). MAX_MM_PER_PX is the
# cutoff: reject any point where 1px of input noise could plausibly move
# the output by more than this -- well above a real elapsed-time walking
# displacement (~30-50mm at 60fps), so it only screens out points in the
# genuinely unstable regime, not ordinary motion.
MAX_MM_PER_PX = 200.0  # loosened after sweeping 60-100000: filtering alone
# barely moved max track duration (stayed 3.7-5.0s across nearly the whole
# range), while 60mm/px cut 43% of all data -- the real fragmentation
# driver turned out to be WorldTracker's max_step (below), not this. Kept
# at a looser value that still screens out the worst grazing-ray cases
# without discarding most real data.


def projection_sensitivity(px: np.ndarray, params: dict, R: np.ndarray, t: np.ndarray) -> np.ndarray:
    """mm of world-position change per 1px of input perturbation (finite
    difference in the foot point's y-coordinate, the axis that drives depth
    for a downward-looking camera -- the axis where grazing-ray instability
    actually shows up)."""
    base = pixel_to_ground(px, params, R, t)
    perturbed = pixel_to_ground(px + np.array([0.0, 1.0]), params, R, t)
    return np.linalg.norm(perturbed - base, axis=1)


def filter_unstable_projections(px: np.ndarray, params: dict, R: np.ndarray, t: np.ndarray):
    """Returns a boolean mask, True for points stable enough to trust."""
    sens = projection_sensitivity(px, params, R, t)
    return sens <= MAX_MM_PER_PX


# Real timestamp shared across all 3 cameras, used once to auto-register
# them -- any synchronized frame with real overlapping content works, this
# one is just what was already extracted during the earlier debugging pass.
SHARED_REGISTRATION_FRAME = "100050"
REFERENCE_CAM = 0


def make_camera_model(cam: int):
    """Wraps this camera's Tsai projection (+ its own stability filter) as
    a plain `mvtrack.court.CameraModel` callable -- one row of world coords
    (or NaN, for a rejected point) per input pixel point, in order, which
    is the contract `register_cameras` needs to keep ORB correspondence
    pairs aligned across two cameras' independent filtering."""
    params, R, t = load_calibration(cam)

    def model(px: np.ndarray) -> np.ndarray:
        out = np.full((len(px), 2), np.nan)
        stable = filter_unstable_projections(px, params, R, t)
        if stable.any():
            with np.errstate(all="ignore"):
                out[stable] = pixel_to_ground(px[stable], params, R, t)
        return out

    return model


def register_all_cameras(cams=(0, 1, 2)):
    """Real, automatic per-camera registration via
    `mvtrack.court.register_cameras` -- replaces the hardcoded 2-camera
    constants this function used to be (see git history): whichever
    cameras register within `max_residual_m` are kept, the rest are
    dropped, with no camera count or index baked in. On this dataset it
    independently rediscovers the same real conclusion the earlier manual
    debugging pass reached (cam1 fails, cam2 succeeds), which is the actual
    proof this generalizes rather than just relabeling the same hardcoded
    result."""
    from mvtrack.court import register_cameras

    camera_models = {c: make_camera_model(c) for c in cams}
    shared_images = {
        c: cv2.imread(str(DATA_ROOT / "frames" / f"cam{c}" / f"RLCAFTCONF-C{c}_{SHARED_REGISTRATION_FRAME}.jpeg"))
        for c in cams
    }
    registrations = register_cameras(camera_models, REFERENCE_CAM, shared_images, max_residual_m=3.0)
    for cam, reg in sorted(registrations.items()):
        status = "included" if reg.included else "EXCLUDED"
        print(f"  cam{cam}: {status} (residual={reg.residual_m:.2f}m, "
              f"{reg.n_correspondences} correspondences)")
    return registrations


def validate_camera_positions():
    print("validating Tsai calibration via recovered camera positions "
          "(GT file unavailable -- SWITCHdrive host down, see module docstring)...")
    positions = {}
    for cam in range(3):
        params, R, t = load_calibration(cam)
        pos = camera_world_position(R, t)
        positions[cam] = pos
        print(f"  cam{cam}: world position {pos}, height (Z) = {pos[2]:.0f}mm")
    dists = []
    for i in range(3):
        for j in range(i + 1, 3):
            d = np.linalg.norm(positions[i] - positions[j])
            dists.append(d)
            print(f"  cam{i}<->cam{j} baseline distance: {d:.0f}mm ({d/1000:.2f}m)")
    return positions


def load_mot_tracks(path):
    """frame -> list of (x0,y0,x1,y1) foot-point-ready boxes."""
    by_frame = {}
    with open(path) as f:
        for line in f:
            parts = line.strip().split(",")
            if len(parts) < 6:
                continue
            frame = int(parts[0])
            x, y, w, h = map(float, parts[2:6])
            by_frame.setdefault(frame, []).append((x, y, x + w, y + h))
    return by_frame


def run_mvtrack_per_camera(anchor_interval=5):
    """Real mvtrack pipeline (MVTracker, anchor/propagate) per camera --
    not generic YOLO+ByteTrack, unlike the earlier WildTrack version."""
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    per_cam_tracks = {}
    for cam in range(3):
        video = VIDEO_ROOT / f"cam{cam}.mp4"
        out = OUT_DIR / f"mv_fixed_cam{cam}.txt"
        print(f"running mv-fixed on cam{cam}...")
        frames, seconds = eval_run.run_mv_fixed(video, out, anchor_interval=anchor_interval)
        print(f"  cam{cam}: {frames} frames in {seconds:.1f}s ({frames/seconds:.1f} fps)")
        per_cam_tracks[cam] = load_mot_tracks(out)
    return per_cam_tracks


def filter_static_pixel_regions(per_cam_tracks, max_fraction=0.5, bin_px=3):
    """Drop detections whose foot point falls in a pixel region that fires
    in more than `max_fraction` of a camera's own frames -- the signature
    of a fixed misclassified object (e.g. a coat hung on a hook,
    found directly: cam2 detected 'person' at the identical foot-point bin
    in 85% of all frames, vs. a 22% high-traffic-doorway ceiling on the
    other cameras). No real person is in frame that often; a lower,
    plausible hotspot (a busy doorway seeing many distinct real people) is
    left alone."""
    filtered = {}
    for cam, tracks in per_cam_tracks.items():
        n_frames = len(tracks)
        counts = {}
        for boxes in tracks.values():
            for x0, y0, x1, y1 in boxes:
                key = (round((x0 + x1) / 2 / bin_px), round(y1 / bin_px))
                counts[key] = counts.get(key, 0) + 1
        bad_bins = {k for k, c in counts.items() if c > max_fraction * n_frames}
        if bad_bins:
            print(f"  cam{cam}: excluding {len(bad_bins)} static pixel region(s) "
                  f"(fired in >{max_fraction*100:.0f}% of frames)")
        cam_filtered = {}
        for frame, boxes in tracks.items():
            kept = [
                (x0, y0, x1, y1) for x0, y0, x1, y1 in boxes
                if (round((x0 + x1) / 2 / bin_px), round(y1 / bin_px)) not in bad_bins
            ]
            if kept:
                cam_filtered[frame] = kept
        filtered[cam] = cam_filtered
    return filtered


def project_and_fuse(per_cam_tracks, calibs, registrations):
    """Per synchronized frame, project every camera's tracked foot points to
    the ground plane, register onto the reference frame via the automatic
    per-camera fit in `registrations` (mvtrack.court.register_cameras --
    whichever cameras were included, not a hardcoded pair), and cluster
    points within FUSE_RADIUS as one real person."""
    included_cams = [c for c, reg in registrations.items() if reg.included]
    all_frames = sorted(set().union(*[per_cam_tracks[c].keys() for c in included_cams]))
    per_frame_fused = {}
    for frame in all_frames:
        cam_points = []
        for cam in included_cams:
            boxes = per_cam_tracks[cam].get(frame, [])
            if not boxes:
                continue
            boxes = np.array(boxes)
            foot = np.stack([(boxes[:, 0] + boxes[:, 2]) / 2.0, boxes[:, 3]], axis=1)
            params, R, t = calibs[cam]
            stable = filter_unstable_projections(foot, params, R, t)
            if not stable.any():
                continue
            world = pixel_to_ground(foot[stable], params, R, t)
            world = registrations[cam].apply(world)
            cam_points.append(world)
        pts = np.concatenate(cam_points, axis=0) if cam_points else np.zeros((0, 2))
        per_frame_fused[frame] = fuse_points(pts)
    return per_frame_fused


def fuse_points(points: np.ndarray, radius=FUSE_RADIUS) -> np.ndarray:
    if len(points) == 0:
        return points
    used = np.zeros(len(points), dtype=bool)
    fused = []
    for i in range(len(points)):
        if used[i]:
            continue
        dists = np.linalg.norm(points - points[i], axis=1)
        group = (dists <= radius) & ~used
        used |= group
        fused.append(points[group].mean(axis=0))
    return np.array(fused)


_DWELL_PARAMS = DwellParams(
    frame_dt=FRAME_DT, min_dwell_seconds=MIN_DWELL_SECONDS,
    min_dwell_radius=MIN_DWELL_RADIUS, max_dwell_radius=MAX_DWELL_RADIUS,
    max_step=2000.0, max_age=10,
)


def track_and_classify_dwells(per_frame_fused: dict):
    return track_and_classify_dwells_shared(per_frame_fused, _DWELL_PARAMS)


def main():
    validate_camera_positions()
    calibs = {cam: load_calibration(cam) for cam in range(3)}

    print("registering cameras (automatic -- see mvtrack.court.register_cameras)...")
    registrations = register_all_cameras()

    per_cam_tracks = run_mvtrack_per_camera(anchor_interval=5)
    print("filtering static pixel regions (fixed misclassified objects)...")
    per_cam_tracks = filter_static_pixel_regions(per_cam_tracks)
    included_cams = [c for c, reg in registrations.items() if reg.included]
    total_raw = sum(len(boxes) for c in included_cams for boxes in per_cam_tracks[c].values())
    per_frame_fused = project_and_fuse(per_cam_tracks, calibs, registrations)

    total_fused = sum(len(p) for p in per_frame_fused.values())
    print(f"\ntotal raw per-camera detections: {total_raw}, fused to {total_fused} "
          f"(dedup ratio {total_raw / max(total_fused,1):.2f}x)")

    tracks, dwellers = track_and_classify_dwells(per_frame_fused)
    print(f"{len(tracks)} total world-space tracks, {len(dwellers)} classify as "
          f"dwelling (>= {MIN_DWELL_SECONDS}s within {MAX_DWELL_RADIUS/1000:.1f}m)")
    for tid, (centroid, duration_s, radius) in sorted(dwellers.items(), key=lambda t: -t[1][1])[:15]:
        print(f"    track {tid}: {duration_s:.1f}s at ({centroid[0]/1000:.2f}, {centroid[1]/1000:.2f})m, "
              f"radius {radius/1000:.2f}m")


if __name__ == "__main__":
    main()
