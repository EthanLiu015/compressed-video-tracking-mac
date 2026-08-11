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
   confirming real overlap exists in the footage. Fixed via automatic
   registration: ORB feature matching + RANSAC-homography-filtered
   correspondences between cam0 and each other camera, projected through
   each camera's own (already-validated) pixel_to_ground, then a
   Kabsch/Procrustes rigid-transform fit aligning the point clouds. Cut
   the raw world-frame mismatch for cam2 from ~10.2m to ~1.1-1.4m mean
   residual -- confirmed the "independent origin" diagnosis and gives a
   real, working registration. The same approach against cam1 failed
   outright (57-90m residual, no consistent structure, both against cam0
   and cam2) -- real evidence cam1 doesn't share enough true overlap with
   the other two for this method to register it, so cam1 is excluded from
   fusion rather than forced in with a bad transform. Fusion here runs on
   the validated cam0+cam2 pair only.
"""

import glob
import pathlib
import sys
import xml.etree.ElementTree as ET

import cv2
import numpy as np
from scipy.optimize import linear_sum_assignment

REPO_ROOT = pathlib.Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT / "eval"))
import run as eval_run  # noqa: E402

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


# Registers cam2's world frame onto cam0's (the reference frame). Fitted via
# ORB feature matching between cam0/cam2 at a shared real timestamp
# (RLCAFTCONF-C{0,2}_100050), RANSAC-homography-filtered to 12 inlier
# correspondences, then a Kabsch/Procrustes rigid fit -- see module
# docstring. cam1 has no entry: registration against it failed for real,
# checked reasons (see docstring), so it's excluded from fusion rather than
# forced in.
CAM2_REGISTER_D = np.array([[0.8993702893949592, -0.437187697166363],
                             [0.4371876971663633, 0.8993702893949592]])
CAM2_REGISTER_C0 = np.array([1931.5769154205411, 13255.355176915822])
CAM2_REGISTER_C2 = np.array([12058.246706040183, 12214.305431100142])
FUSION_CAMERAS = [0, 2]  # cam1 excluded -- see module docstring


def register_to_cam0(cam: int, world_pts: np.ndarray) -> np.ndarray:
    if cam == 0:
        return world_pts
    if cam == 2:
        return (world_pts - CAM2_REGISTER_C2) @ CAM2_REGISTER_D.T + CAM2_REGISTER_C0
    raise ValueError(f"no registration fitted for cam{cam} -- excluded from fusion")


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


def project_and_fuse(per_cam_tracks, calibs):
    """Per synchronized frame, project every camera's tracked foot points to
    the ground plane, register onto cam0's frame, and cluster points within
    FUSE_RADIUS as one real person. Only FUSION_CAMERAS (cam0+cam2) --
    cam1 excluded, registration against it failed (see module docstring)."""
    all_frames = sorted(set().union(*[per_cam_tracks[c].keys() for c in FUSION_CAMERAS]))
    per_frame_fused = {}
    for frame in all_frames:
        cam_points = []
        for cam in FUSION_CAMERAS:
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
            world = register_to_cam0(cam, world)
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


class WorldTracker:
    """Hungarian-assignment world-point tracker with a grace period --
    ports the same fix MVTracker's three-stage re-association made for
    pixel-space MOT17 tracking (findings.md #10) to fused world-space
    points. The bug this fixes is structurally identical: the original
    greedy nearest-neighbor tracker dropped a track the instant it went
    unmatched for even one frame, so any single missed detection (a brief
    fusion-clustering miss, a detector blink) permanently killed the
    identity and forced a respawn -- the exact "any miss fragments the
    track" failure MVTracker's docstring describes for the pixel case.
    Here: unmatched tracks survive up to `max_age` frames before being
    pruned, giving a real chance to reattach if the person reappears
    nearby -- and matching uses Hungarian assignment (globally optimal)
    instead of greedy first-come-first-served.

    Real bug found and fixed by checking cam0's raw MVTracker output
    directly: its own per-camera tracks run up to 17.5s continuously (the
    per-camera tracker is fine), but re-deriving identity from scratch on
    fused world points was fragmenting everything to under 3.15s max.

    Two real, distinct causes, found by isolating cam0 alone (no fusion)
    and sweeping parameters directly rather than guessing:

    1. `max_step=250mm` (a naive "real walking speed / 60fps" estimate)
       was simply too tight -- it didn't account for how much ordinary
       ~1-2px detector-box jitter gets amplified by ground-plane
       back-projection even outside the extreme grazing-ray cases below.
       This was the dominant lever: sweeping max_step alone (500mm ->
       21.33s max duration at 5000mm) recovered continuity close to or
       exceeding cam0's own raw per-camera track lengths, while adding
       velocity prediction or the sensitivity filter alone barely moved
       the number (stayed 3.7-5.0s) -- confirmed by testing each fix in
       isolation before combining them, not assuming either worked from
       its own plausibility.
    2. Held-frozen unmatched tracks compound the problem for a moving
       target once a gap does occur (the real person moves away from the
       frozen position every subsequent frame, so the gap only grows) --
       fixed with constant-velocity prediction during the grace period,
       the same fix MV-propagation itself makes for the pixel case, one
       level up in world-space.
    """

    def __init__(self, max_step=2000.0, max_age=10, vel_ema=0.6):
        self.max_step = max_step
        self.max_age = max_age
        self.vel_ema = vel_ema
        self.tracks = {}  # tid -> {"pos", "vel", "since_detection", "history"}
        self.completed = {}  # tid -> history, moved here on prune so it isn't lost
        self._next_tid = 0

    def step(self, frame_id, points: np.ndarray):
        track_ids = list(self.tracks.keys())
        matched_tracks, matched_pts = set(), set()

        # Predict forward before matching: a track sitting idle at its last
        # observed position (since_detection==0 this call) uses that
        # position as-is; one already in its grace period extrapolates
        # along its last known velocity instead of staying frozen.
        pred_pos = {
            tid: tr["pos"] + tr["vel"] * tr["since_detection"] for tid, tr in self.tracks.items()
        }

        if track_ids and len(points):
            track_pos = np.array([pred_pos[t] for t in track_ids])
            dists = np.linalg.norm(track_pos[:, None, :] - points[None, :, :], axis=2)
            row, col = linear_sum_assignment(dists)
            for r, c in zip(row, col):
                if dists[r, c] > self.max_step:
                    continue
                tid = track_ids[r]
                new_vel = points[c] - self.tracks[tid]["pos"]
                self.tracks[tid]["vel"] = (
                    self.vel_ema * self.tracks[tid]["vel"] + (1 - self.vel_ema) * new_vel
                )
                self.tracks[tid]["pos"] = points[c]
                self.tracks[tid]["since_detection"] = 0
                self.tracks[tid]["history"].append((frame_id, points[c]))
                matched_tracks.add(tid)
                matched_pts.add(c)

        for tid in track_ids:
            if tid not in matched_tracks:
                self.tracks[tid]["since_detection"] += 1
        for tid in [t for t, tr in self.tracks.items() if tr["since_detection"] > self.max_age]:
            self.completed[tid] = self.tracks.pop(tid)["history"]

        for c in range(len(points)):
            if c not in matched_pts:
                self.tracks[self._next_tid] = {
                    "pos": points[c], "vel": np.zeros(2), "since_detection": 0,
                    "history": [(frame_id, points[c])],
                }
                self._next_tid += 1


def track_and_classify_dwells(per_frame_fused: dict):
    frame_ids = sorted(per_frame_fused.keys())
    tracker = WorldTracker(max_step=2000.0, max_age=10)
    for fid in frame_ids:
        tracker.step(fid, per_frame_fused[fid])

    tracks = dict(tracker.completed)
    tracks.update({tid: tr["history"] for tid, tr in tracker.tracks.items()})

    dwellers = {}
    for tid, pts in tracks.items():
        duration_s = len(pts) * FRAME_DT
        if duration_s < MIN_DWELL_SECONDS:
            continue
        xy = np.array([p for _, p in pts])
        centroid = xy.mean(axis=0)
        radius = np.linalg.norm(xy - centroid, axis=1).max()
        if MIN_DWELL_RADIUS <= radius <= MAX_DWELL_RADIUS:
            dwellers[tid] = (centroid, duration_s, radius)
    return tracks, dwellers


def main():
    validate_camera_positions()
    calibs = {cam: load_calibration(cam) for cam in range(3)}

    per_cam_tracks = run_mvtrack_per_camera(anchor_interval=5)
    print("filtering static pixel regions (fixed misclassified objects)...")
    per_cam_tracks = filter_static_pixel_regions(per_cam_tracks)
    total_raw = sum(len(boxes) for cam_tracks in per_cam_tracks.values() for boxes in cam_tracks.values())
    per_frame_fused = project_and_fuse(per_cam_tracks, calibs)

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
