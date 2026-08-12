"""PULSE on the EPFL CVLAB 6-person Laboratory sequence, the fallback
dataset after two real searches (WebSearch + direct extraction) confirmed
no publicly-available multi-camera museum dataset ships real calibration --
MuseumVisitors (3 cams, MICC/Bargello) has none, and no other candidate
turned up one either. This is the closest real analog: real indoor room,
4 fixed calibrated cameras, real people, real 25fps video -- run through
the same real mvtrack pipeline (MVTracker, not generic YOLO+ByteTrack) and
the same general N-camera registration (`mvtrack.court.register_cameras`)
built for the museum generalization request. "Exhibit zones" below are a
hand-picked stand-in (the rug vs. the open floor) for "which object in the
room draws attention", not real museum exhibit labels -- there are none in
this dataset either.

Calibration is a single 3x3 planar ground-homography per camera (not Tsai,
not OpenCV pinhole) -- confirmed by the file's own section header ("Ground
plane homography") and by testing which of {H, H^-1, H^T, H^-T} applied to
a real cross-camera point correspondence gives a physically consistent,
non-degenerate world position. Two independent real correspondences (a
static cable-junction box on the floor, and one person's foot point) were
hand-picked in both cam0 and cam1's frame 1500 and projected: `H @
[px,py,1]` (forward, NOT inverted) agrees to 7-14 units across cameras on
a ~150-260 unit scale (~5-10% relative) -- consistent with real hand-
picking precision on a small object, not a systematic direction/transpose
bug (the wrong conventions tested gave either a >160-unit mismatch or
degenerate near-zero output). Units are unlabeled in the source file;
treated as cm given the resulting scale matches a real ~5m room.

Because all 4 cameras' ground homographies are already defined against one
shared coordinate frame (by construction, unlike EPFL-RLC's independent
per-camera Tsai origins), `register_cameras()` is expected to find each
camera already near-aligned (small residual, not the "some cameras
excluded" result EPFL-RLC produced) -- a different, complementary
validation of the same general module: it should recognize "already
correct" instead of just detecting and rejecting real misalignment.
"""

import pathlib
import re
import sys

import cv2
import numpy as np

REPO_ROOT = pathlib.Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT / "eval"))
sys.path.insert(0, str(REPO_ROOT / "scripts"))
import run as eval_run  # noqa: E402
from epfl_rlc_fusion import load_mot_tracks  # noqa: E402 -- model-agnostic, reused as-is
from mvtrack.analytics import DwellParams, Zone  # noqa: E402
from mvtrack.analytics import point_in_polygon as _point_in_polygon  # noqa: E402
from mvtrack.analytics import track_and_classify_dwells as track_and_classify_dwells_shared  # noqa: E402

DATA_ROOT = REPO_ROOT / "data" / "epfl_lab"
VIDEO_ROOT = DATA_ROOT / "videos"
OUT_DIR = REPO_ROOT / "outputs" / "epfl_lab_fusion"
VIZ_DIR = REPO_ROOT / "outputs" / "epfl_lab_viz"

FRAME_DT = 1.0 / 25  # real 25fps source
FUSE_RADIUS = 60.0  # cm -- ~person-footprint scale; cameras already share one frame so no registration slop to absorb, unlike EPFL-RLC's 1600mm
MIN_DWELL_SECONDS = 4.0  # sequence is only 118s total -- EPFL-RLC's 6.0s threshold would leave almost no room to observe a dwell at all
MAX_DWELL_RADIUS = 60.0  # cm
MIN_DWELL_RADIUS = 5.0  # cm -- reject exact-static-object false positives, same class fixed in plaza_dwell.py/epfl_rlc_fusion.py

# Hand-picked rug polygon corners in cam0 pixel space (360x288 native res),
# read off outputs/epfl_lab_frames/cam0_f1500_big.png by eye -- illustrative
# zone boundary, not survey-grade. Projected to world once at import time.
_RUG_CORNERS_PX_CAM0 = np.array([
    (100.0, 97.5),   # top-left
    (180.0, 87.5),   # top-right
    (165.0, 135.0),  # bottom-right
    (5.0, 130.0),    # bottom-left
])


def load_calibration(cam: int) -> np.ndarray:
    """Parse this camera's 3x3 ground-plane homography from the shared
    calibration-6p.txt file. Direction confirmed empirically (see module
    docstring): world = normalize(H @ [px, py, 1]), not the inverse."""
    text = (DATA_ROOT / "calibration-6p.txt").read_text()
    blocks = re.split(r"#+\n# Camera (\d+)\n#+\n", text)[1:]
    for i in range(0, len(blocks), 2):
        if int(blocks[i]) != cam:
            continue
        body = blocks[i + 1]
        m = re.search(r"# Ground plane homography\n(.*?)\n\n", body, re.S)
        nums = [float(x) for x in m.group(1).split()]
        return np.array(nums).reshape(3, 3)
    raise ValueError(f"no calibration found for cam{cam}")


def pixel_to_ground(px: np.ndarray, H: np.ndarray) -> np.ndarray:
    """Nx2 pixel points -> Nx2 world (cm) points on the shared ground plane."""
    homog = np.concatenate([px, np.ones((len(px), 1))], axis=1)
    world = homog @ H.T
    return world[:, :2] / world[:, 2:3]


def make_camera_model(cam: int):
    H = load_calibration(cam)

    def model(px: np.ndarray) -> np.ndarray:
        return pixel_to_ground(px, H)

    return model


SHARED_REGISTRATION_FRAME = 1500
REFERENCE_CAM = 0


def register_all_cameras(cams=(0, 1, 2, 3)):
    """Same general registration used for the museum-generalization request
    (`mvtrack.court.register_cameras`) -- run here as a validation of the
    *other* real case: cameras that are already correctly co-registered
    should come back with small residuals and every camera included,
    rather than needing a real fix like EPFL-RLC's cam1 exclusion."""
    from mvtrack.court import register_cameras

    camera_models = {c: make_camera_model(c) for c in cams}
    shared_images = {}
    for c in cams:
        path = VIZ_DIR.parent / "epfl_lab_frames" / f"cam{c}_f1500.png"
        img = cv2.imread(str(path))
        if img is None:
            cap = cv2.VideoCapture(str(VIDEO_ROOT / f"cam{c}.mp4"))
            cap.set(cv2.CAP_PROP_POS_FRAMES, SHARED_REGISTRATION_FRAME)
            ok, img = cap.read()
            cap.release()
            if not ok:
                raise RuntimeError(f"could not read shared frame for cam{c}")
        shared_images[c] = img

    registrations = register_cameras(camera_models, REFERENCE_CAM, shared_images, max_residual_m=3.0)
    for cam, reg in sorted(registrations.items()):
        status = "included" if reg.included else "EXCLUDED"
        # residual_m is really "residual, homography units / 100" given the
        # cm-scale assumption -- printed as-is, not relabeled to a unit we
        # haven't independently confirmed.
        print(f"  cam{cam}: {status} (residual={reg.residual_m * 100:.1f}cm, "
              f"{reg.n_correspondences} correspondences)")
    return registrations


def run_mvtrack_per_camera(anchor_interval=5):
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    per_cam_tracks = {}
    for cam in range(4):
        video = VIDEO_ROOT / f"cam{cam}.mp4"
        out = OUT_DIR / f"mv_fixed_cam{cam}.txt"
        print(f"running mv-fixed on cam{cam}...")
        frames, seconds = eval_run.run_mv_fixed(video, out, anchor_interval=anchor_interval)
        print(f"  cam{cam}: {frames} frames in {seconds:.1f}s ({frames/seconds:.1f} fps)")
        per_cam_tracks[cam] = load_mot_tracks(str(out))
    return per_cam_tracks


def project_and_fuse(per_cam_tracks, homs, registrations):
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
            world = pixel_to_ground(foot, homs[cam])
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
    max_step=80.0, max_age=10,  # cm-scale room -- EPFL-RLC's 2000mm step would let any two people in the room swap identities
)


def track_and_classify_dwells(per_frame_fused: dict):
    return track_and_classify_dwells_shared(per_frame_fused, _DWELL_PARAMS)


def rug_zone() -> Zone:
    H0 = load_calibration(0)
    return Zone(name="rug", polygon=pixel_to_ground(_RUG_CORNERS_PX_CAM0, H0))


def classify_zone_attention(dwellers, tracks):
    """Real dwell duration and idle-visit count per hand-picked zone
    ('rug' vs 'open floor') -- the stand-in for 'which exhibit gets more
    attention' on a dataset with no real exhibit labels."""
    zone = rug_zone()
    zone_time = {"rug": 0.0, "open_floor": 0.0}
    zone_visits = {"rug": 0, "open_floor": 0}
    for tid, (centroid, duration_s, radius) in dwellers.items():
        name = "rug" if _point_in_polygon(centroid, zone.polygon) else "open_floor"
        zone_time[name] += duration_s
        zone_visits[name] += 1
    return zone_time, zone_visits


def main():
    homs = {cam: load_calibration(cam) for cam in range(4)}

    print("registering cameras (automatic -- mvtrack.court.register_cameras)...")
    registrations = register_all_cameras()

    per_cam_tracks = run_mvtrack_per_camera(anchor_interval=5)
    included_cams = [c for c, reg in registrations.items() if reg.included]
    total_raw = sum(len(boxes) for c in included_cams for boxes in per_cam_tracks[c].values())
    per_frame_fused = project_and_fuse(per_cam_tracks, homs, registrations)

    total_fused = sum(len(p) for p in per_frame_fused.values())
    print(f"\ntotal raw per-camera detections: {total_raw}, fused to {total_fused} "
          f"(dedup ratio {total_raw / max(total_fused,1):.2f}x)")

    tracks, dwellers = track_and_classify_dwells(per_frame_fused)
    print(f"{len(tracks)} total world-space tracks, {len(dwellers)} classify as "
          f"dwelling (>= {MIN_DWELL_SECONDS}s within {MAX_DWELL_RADIUS}cm)")
    for tid, (centroid, duration_s, radius) in sorted(dwellers.items(), key=lambda t: -t[1][1])[:15]:
        print(f"    track {tid}: {duration_s:.1f}s at ({centroid[0]:.0f}, {centroid[1]:.0f})cm, "
              f"radius {radius:.1f}cm")

    zone_time, zone_visits = classify_zone_attention(dwellers, tracks)
    print("\nattention by zone (rug vs. open floor, stand-in for exhibit attention):")
    for zone in ("rug", "open_floor"):
        print(f"    {zone}: {zone_visits[zone]} dwell visit(s), {zone_time[zone]:.1f}s total")


if __name__ == "__main__":
    main()
