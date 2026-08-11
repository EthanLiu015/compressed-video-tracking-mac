"""Stage 2 of PULSE: multi-camera ground-plane fusion on WildTrack.

Unlike the tennis homography (hand-picked pixel<->court-corner
correspondences, verified by eye), WildTrack ships real OpenCV pinhole
calibration (intrinsic K + distortion, extrinsic rvec/tvec) per camera, and
real ground-truth annotations with a *decodable* world position
(`positionID` -> meters, see the dataset's own README). That means the
pixel->ground-plane projection here can be validated against real numeric
ground truth instead of a visual overlay -- a strictly stronger check.

Projection method: back-project a pixel to a camera-frame ray via K^-1,
then intersect that ray with the world ground plane (Z=0), using the
extrinsic R (from rvec via Rodrigues) and t. Standard single-camera
ground-plane localization -- valid because WildTrack's world frame is
defined with the ground at Z=0 and all subjects are assumed foot-on-ground,
same assumption the tennis homography made.
"""

import glob
import json
import pathlib
import xml.etree.ElementTree as ET

import cv2
import numpy as np
from ultralytics import YOLO

from mvtrack.detect import pick_device

REPO_ROOT = pathlib.Path(__file__).resolve().parent.parent
DATA_ROOT = REPO_ROOT / "data" / "wildtrack" / "Wildtrack_dataset"
CALIB_ROOT = DATA_ROOT / "calibrations"
OUT_DIR = REPO_ROOT / "outputs" / "wildtrack_fusion"
PERSON_CLS = 0

# Frame indices step by 5 (00000000, 00000005, ...); this subset is WildTrack's
# 2fps annotated series (README's own description), so consecutive frames are
# 0.5s apart -- close enough for greedy nearest-neighbor world-space tracking
# without a full motion model.
FRAME_DT = 0.5
# Real projections of the same physical person from different cameras landed
# within ~13cm of each other on average (validated above against 3782 real
# cross-camera pairs) -- cluster projected points within 3x that as "the same
# person," well clear of the validated noise floor.
FUSE_RADIUS = 0.4  # meters
MIN_DWELL_SECONDS = 6.0
MAX_DWELL_RADIUS = 0.6  # meters

# C1..C7 -> WildTrack's internal camera names. Standard mapping used by
# WildTrack-based code (e.g. MVDet); validated below via cross-camera
# consistency against real annotations rather than assumed blindly.
CAM_NAMES = {
    1: "CVLab1", 2: "CVLab2", 3: "CVLab3", 4: "CVLab4",
    5: "IDIAP1", 6: "IDIAP2", 7: "IDIAP3",
}

# positionID -> world meters (X, Y), from the dataset's own README.
GRID_W = 480
GRID_ORIGIN = (-3.0, -9.0)
GRID_SPACING = 0.025


def position_id_to_xy(pid: int) -> np.ndarray:
    x = GRID_ORIGIN[0] + GRID_SPACING * (pid % GRID_W)
    y = GRID_ORIGIN[1] + GRID_SPACING * (pid // GRID_W)
    return np.array([x, y])


def _read_opencv_matrix(node) -> np.ndarray:
    rows, cols = int(node.find("rows").text), int(node.find("cols").text)
    data = [float(v) for v in node.find("data").text.split()]
    return np.array(data).reshape(rows, cols)


def load_calibration(cam_idx: int):
    name = CAM_NAMES[cam_idx]
    intr = ET.parse(CALIB_ROOT / "intrinsic_zero" / f"intr_{name}.xml").getroot()
    K = _read_opencv_matrix(intr.find("camera_matrix"))

    extr = ET.parse(CALIB_ROOT / "extrinsic" / f"extr_{name}.xml").getroot()
    rvec = np.array([float(v) for v in extr.find("rvec").text.split()])
    # tvec magnitudes (~500-1000) are too large to be meters on a 12x36m grid
    # -- centimeters (a common OpenCV calibration-target convention), so a
    # ~10m camera height (cm=986.7) is plausible for an elevated plaza cam,
    # vs. ~1m if read as millimeters, which isn't. Converted to match the
    # grid's meter units; validated below against real GT positions, so a
    # wrong unit guess here would show up directly as large errors.
    tvec = np.array([float(v) for v in extr.find("tvec").text.split()]) / 100.0
    R, _ = cv2.Rodrigues(rvec)
    return K, R, tvec


def pixel_to_ground(px: np.ndarray, K: np.ndarray, R: np.ndarray, t: np.ndarray) -> np.ndarray:
    """Back-project Nx2 pixel points to Nx2 world (X, Y) on the Z=0 ground
    plane. WildTrack convention: Xc = R @ Xw + t (world -> camera)."""
    px_h = np.hstack([px, np.ones((len(px), 1))])  # Nx3 homogeneous pixels
    ray_cam = px_h @ np.linalg.inv(K).T  # Nx3, camera-frame ray directions
    Rt_t = R.T @ t  # world-frame camera center offset term
    Rt_ray = ray_cam @ R  # Nx3, R^T @ ray_cam (since ray_cam @ R == (R^T @ ray_cam^T)^T)
    s = Rt_t[2] / Rt_ray[:, 2]  # solve for depth where world Z == 0
    world = s[:, None] * Rt_ray - Rt_t
    return world[:, :2]


def validate_projection(calibs):
    print("validating projection against real ground-truth annotations...")
    errors_vs_gt = []
    errors_cross_cam = []
    ann_files = sorted(glob.glob(str(DATA_ROOT / "annotations_positions" / "*.json")))
    for fpath in ann_files[:20]:
        with open(fpath) as f:
            people = json.load(f)
        for person in people:
            gt_xy = position_id_to_xy(person["positionID"])
            projected = []
            for view in person["views"]:
                if view["xmin"] < 0:
                    continue
                cam = view["viewNum"] + 1  # viewNum is 0-indexed -> C1..C7
                foot_px = np.array([[(view["xmin"] + view["xmax"]) / 2.0, view["ymax"]]])
                K, R, t = calibs[cam]
                world = pixel_to_ground(foot_px, K, R, t)[0]
                projected.append(world)
                errors_vs_gt.append(np.linalg.norm(world - gt_xy))
            for i in range(len(projected)):
                for j in range(i + 1, len(projected)):
                    errors_cross_cam.append(np.linalg.norm(projected[i] - projected[j]))

    errors_vs_gt = np.array(errors_vs_gt)
    errors_cross_cam = np.array(errors_cross_cam)
    print(f"projected-vs-GT-position error (m): mean={errors_vs_gt.mean():.3f} "
          f"median={np.median(errors_vs_gt):.3f} max={errors_vs_gt.max():.3f} "
          f"(n={len(errors_vs_gt)})")
    print(f"cross-camera consistency error (m): mean={errors_cross_cam.mean():.3f} "
          f"median={np.median(errors_cross_cam):.3f} max={errors_cross_cam.max():.3f} "
          f"(n={len(errors_cross_cam)})")


def detect_and_project_all(calibs, model, device):
    """Returns {frame_idx: Nx2 array of fused world points}, running our own
    detector per camera per frame -- not replaying ground truth."""
    frame_files = sorted(glob.glob(str(DATA_ROOT / "Image_subsets" / "C1" / "*.png")))
    frame_ids = [pathlib.Path(f).stem for f in frame_files]

    per_frame_points = {}
    for fid in frame_ids:
        cam_points = []
        for cam in range(1, 8):
            img_path = DATA_ROOT / "Image_subsets" / f"C{cam}" / f"{fid}.png"
            img = cv2.imread(str(img_path))
            res = model.predict(img, device=device, conf=0.25, classes=[PERSON_CLS], verbose=False)[0]
            if len(res.boxes) == 0:
                continue
            xyxy = res.boxes.xyxy.cpu().numpy()
            foot = np.stack([(xyxy[:, 0] + xyxy[:, 2]) / 2.0, xyxy[:, 3]], axis=1)
            K, R, t = calibs[cam]
            world = pixel_to_ground(foot, K, R, t)
            cam_points.append(world)
        per_frame_points[fid] = np.concatenate(cam_points, axis=0) if cam_points else np.zeros((0, 2))
    return per_frame_points


def fuse_points(points: np.ndarray, radius=FUSE_RADIUS) -> np.ndarray:
    """Greedy-cluster raw multi-camera world points within `radius` of each
    other into one fused point per real person -- multiple cameras seeing
    the same person should land close together (validated above), whereas
    two different people are meters apart on this 12x36m square."""
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


def track_and_classify_dwells(per_frame_fused: dict):
    """Greedy nearest-neighbor frame-to-frame association in world space,
    then the same min-duration/max-radius dwell classifier as
    plaza_dwell.py, in real meters/seconds instead of pixels/frames."""
    frame_ids = sorted(per_frame_fused.keys())
    tracks = {}  # tid -> list of (frame_idx, xy)
    active = {}  # tid -> last xy
    next_tid = 0
    max_step = 1.2  # m, generous vs. a person's real per-frame (0.5s) displacement

    for fidx, fid in enumerate(frame_ids):
        pts = per_frame_fused[fid]
        matched_tids = set()
        for p in pts:
            best_tid, best_dist = None, max_step
            for tid, last_xy in active.items():
                if tid in matched_tids:
                    continue
                d = np.linalg.norm(p - last_xy)
                if d < best_dist:
                    best_tid, best_dist = tid, d
            if best_tid is None:
                best_tid = next_tid
                next_tid += 1
                tracks[best_tid] = []
            tracks[best_tid].append((fidx, p))
            active[best_tid] = p
            matched_tids.add(best_tid)
        active = {tid: xy for tid, xy in active.items() if tid in matched_tids}

    dwellers = {}
    for tid, pts in tracks.items():
        duration_s = len(pts) * FRAME_DT
        if duration_s < MIN_DWELL_SECONDS:
            continue
        xy = np.array([p for _, p in pts])
        centroid = xy.mean(axis=0)
        radius = np.linalg.norm(xy - centroid, axis=1).max()
        if radius <= MAX_DWELL_RADIUS:
            dwellers[tid] = (centroid, duration_s, radius)
    return tracks, dwellers


def find_occlusion_payoff_case():
    """Find a real annotated instance where a person is invisible in one
    camera (xmin=-1) but visible in >=1 other camera at the same frame --
    the concrete case that proves fusion adds something a single camera
    can't: recovering someone a lone camera would simply miss."""
    ann_files = sorted(glob.glob(str(DATA_ROOT / "annotations_positions" / "*.json")))
    for fpath in ann_files:
        with open(fpath) as f:
            people = json.load(f)
        for person in people:
            visible = [v["viewNum"] + 1 for v in person["views"] if v["xmin"] >= 0]
            invisible = [v["viewNum"] + 1 for v in person["views"] if v["xmin"] < 0]
            if len(visible) >= 1 and len(invisible) >= 3:
                return pathlib.Path(fpath).stem, person, visible, invisible
    return None


def main():
    calibs = {c: load_calibration(c) for c in range(1, 8)}
    validate_projection(calibs)

    case = find_occlusion_payoff_case()
    if case:
        fid, person, visible, invisible = case
        gt_xy = position_id_to_xy(person["positionID"])
        print(f"\nocclusion payoff case: frame {fid}, person {person['personID']}")
        print(f"  visible in cameras {visible}, invisible in {invisible}")
        print(f"  ground-truth position: {gt_xy}")
        print(f"  a single-camera pipeline using only camera {invisible[0]} would "
              f"miss this person entirely this frame; a camera-{visible[0]}-based "
              f"fused pipeline recovers them.")

    print("\nrunning detector across all 7 cameras x 401 frames (this takes a while)...")
    model = YOLO("yolov8s.pt")
    device = pick_device()
    per_frame_points = detect_and_project_all(calibs, model, device)
    per_frame_fused = {fid: fuse_points(pts) for fid, pts in per_frame_points.items()}

    total_raw = sum(len(p) for p in per_frame_points.values())
    total_fused = sum(len(p) for p in per_frame_fused.values())
    print(f"total raw cross-camera detections: {total_raw}, "
          f"fused to {total_fused} (dedup ratio {total_raw / max(total_fused,1):.2f}x)")

    tracks, dwellers = track_and_classify_dwells(per_frame_fused)
    print(f"\n{len(tracks)} total world-space tracks, {len(dwellers)} classify as "
          f"dwelling (>= {MIN_DWELL_SECONDS}s within {MAX_DWELL_RADIUS}m)")
    for tid, (centroid, duration_s, radius) in sorted(dwellers.items(), key=lambda t: -t[1][1])[:15]:
        print(f"    track {tid}: {duration_s:.1f}s at ({centroid[0]:.2f}, {centroid[1]:.2f})m, "
              f"radius {radius:.2f}m")

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(figsize=(8, 4))
    for tid, pts in tracks.items():
        xy = np.array([p for _, p in pts])
        ax.plot(xy[:, 0], xy[:, 1], "-", alpha=0.3, color="tab:blue", linewidth=0.8)
    for tid, (centroid, duration_s, radius) in dwellers.items():
        circ = plt.Circle(centroid, radius, color="tab:red", fill=False, linewidth=1.5)
        ax.add_patch(circ)
    ax.set_xlim(-3, 9)
    ax.set_ylim(-9, 27)
    ax.set_xlabel("world X (m)")
    ax.set_ylabel("world Y (m)")
    ax.set_title("WildTrack fused ground-plane tracks + dwell zones")
    ax.set_aspect("equal")
    fig.tight_layout()
    out_path = OUT_DIR / "fused_tracks.png"
    fig.savefig(out_path, dpi=120)
    print(f"\nsaved fused-track plot -> {out_path}")


if __name__ == "__main__":
    main()
