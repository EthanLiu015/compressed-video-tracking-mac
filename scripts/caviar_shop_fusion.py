"""PULSE on the real CAVIAR "OneShopOneWait1" shopping-mall clip -- the
dataset picked after the EPFL CVLAB Laboratory sequence (used for the
museum-generalization validation) turned out to have NO genuine idle
behavior: it's a staged pedestrian-detection benchmark where subjects
walk continuously for the whole clip, and both "6.0s dwell" candidates
found there were confirmed false (frame-by-frame visual check showed
different people crossing the same spot mid-stride, chained into one
track by the world tracker's grace period -- see scripts/epfl_lab_fusion.py).

CAVIAR's own scenario description for this clip: "Couple walking on the
corridor, one goes inside a store, the other waits outside, later they
rejoin and leave together." That's real, described-in-advance idle
behavior -- a person waiting -- the closest available real analog to a
museum visitor pausing at an exhibit (the storefront stands in for the
exhibit). Confirmed directly before building anything further: frame 500
of the front view already shows a person standing motionless outside the
Promod storefront (outputs/caviar_frames/front_f500.png).

Two synchronized camera views: "cor" (corridor, looking down the hallway)
and "front" (an elevated view looking across the balcony into the shop
front) -- genuinely different vantage points of the same real mall, unlike
the EPFL Lab sequence's already-literally-shared calibration frame.

Calibration here is NOT reverse-engineered like the Lab sequence's -- CAVIAR
publishes real pixel<->world (cm) correspondence points directly (see
http://homepages.inf.ed.ac.uk/rbf/CAVIAR/, "TestScenarios_files/shoppingplane.png"
and the accompanying point table), 4 points per view, both already defined
against one shared (X,Y) world frame (confirmed by the figure: cor's axes
and front's axes are drawn as the same +X/+Y directions from one origin).
Exactly 4 points per view -> cv2.getPerspectiveTransform gives an exact fit,
no least-squares residual to report.

cor and front are NOT frame-index-synchronized -- confirmed directly from
each video's own burned-in timecode overlay (frame 500 of cor reads
10:32:52.237; frame 500 of front reads 10:32:49.117, a real ~3.1s / ~78-frame
offset at 25fps). The dataset's own text warns of exactly this ("each video
set may start at a slightly different time, and the time-code ... gives the
necessary information"). Given that uncertainty, this module tracks and
classifies dwells PER CAMERA independently in shared world coordinates
first (each camera's own detections are self-consistent regardless of the
other camera's frame offset) rather than assuming naive frame-index fusion
-- cross-camera fusion via `mvtrack.court.register_cameras` is attempted as
a secondary, clearly-labeled step.
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
from mvtrack.analytics import DwellParams, Zone, capture_rate, classify_zone_traffic, find_companion_pairs  # noqa: E402
from mvtrack.analytics import track_and_classify_dwells as track_and_classify_dwells_shared  # noqa: E402

DATA_ROOT = REPO_ROOT / "data" / "caviar_shop"
VIDEO_ROOT = DATA_ROOT / "videos"
OUT_DIR = REPO_ROOT / "outputs" / "caviar_shop_fusion"

FRAME_DT = 1.0 / 25  # real 25fps source (avg_frame_rate confirmed via ffprobe)
MIN_DWELL_SECONDS = 4.0  # clip is ~43-55s -- EPFL-RLC's 6.0s would leave little margin
MAX_DWELL_RADIUS = 60.0  # cm
MIN_DWELL_RADIUS = 5.0  # cm -- reject exact-static-object false positives

# Real published pixel(col,row) -> world(X,Y cm) correspondences, both views
# already in one shared frame (see module docstring). Source: CAVIAR's own
# TestScenarios page, the table accompanying shoppingplane.png.
_CALIB_POINTS = {
    "cor": {
        "px": np.array([(91, 163), (241, 163), (98, 266), (322, 265)], dtype=np.float32),
        "world": np.array([(0, 975), (290, 975), (0, -110), (290, -110)], dtype=np.float32),
    },
    "front": {
        "px": np.array([(60, 153), (359, 153), (50, 201), (367, 200)], dtype=np.float32),
        "world": np.array([(0, 0), (0, 975), (382, 98), (382, 878)], dtype=np.float32),
    },
}


# Hand-picked storefront-walkway zone in front-view pixel space (native
# 384x288), read off outputs/caviar_frames/front_f500.png by eye -- brackets
# the walkway strip right at the Promod entrance, where the confirmed real
# dweller (track 12) and real passersby (id15) both actually walked. Same
# "illustrative, not survey-grade" caveat as every other hand-picked zone in
# this project (see scripts/epfl_lab_fusion.py's rug polygon).
_STOREFRONT_ZONE_PX = np.array([(40, 130), (330, 130), (330, 180), (40, 180)], dtype=np.float32)


def storefront_zone(proximity_cm=50.0) -> Zone:
    H = load_calibration("front")
    world = pixel_to_ground(_STOREFRONT_ZONE_PX, H)
    return Zone(name="promod_storefront", polygon=world, proximity_cm=proximity_cm)


def load_calibration(view: str) -> np.ndarray:
    """3x3 homography, pixel -> world (cm), fit exactly from the 4 real
    published correspondence points for this view."""
    pts = _CALIB_POINTS[view]
    return cv2.getPerspectiveTransform(pts["px"], pts["world"])


def pixel_to_ground(px: np.ndarray, H: np.ndarray) -> np.ndarray:
    homog = np.concatenate([px.astype(np.float64), np.ones((len(px), 1))], axis=1)
    world = homog @ H.T
    return world[:, :2] / world[:, 2:3]


def make_camera_model(view: str):
    H = load_calibration(view)

    def model(px: np.ndarray) -> np.ndarray:
        return pixel_to_ground(px, H)

    return model


def run_mvtrack(view: str, anchor_interval=5):
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    video = VIDEO_ROOT / f"{view}.mp4"
    out = OUT_DIR / f"mv_fixed_{view}.txt"
    print(f"running mv-fixed on {view}...")
    frames, seconds = eval_run.run_mv_fixed(video, out, anchor_interval=anchor_interval)
    print(f"  {view}: {frames} frames in {seconds:.1f}s ({frames/seconds:.1f} fps)")
    return load_mot_tracks(str(out))


def foot_points_by_frame(tracks_by_frame, H) -> dict:
    """Raw per-frame pixel boxes -> per-frame world-space foot points."""
    per_frame_world = {}
    for frame, boxes in tracks_by_frame.items():
        boxes = np.array(boxes)
        foot = np.stack([(boxes[:, 0] + boxes[:, 2]) / 2.0, boxes[:, 3]], axis=1)
        per_frame_world[frame] = pixel_to_ground(foot, H)
    return per_frame_world


def track_and_classify_dwells(tracks_by_frame, H, max_step=80.0):
    """Per-camera: project foot points to world, track, classify dwells."""
    params = DwellParams(
        frame_dt=FRAME_DT, min_dwell_seconds=MIN_DWELL_SECONDS,
        min_dwell_radius=MIN_DWELL_RADIUS, max_dwell_radius=MAX_DWELL_RADIUS,
        max_step=max_step, max_age=10,
    )
    return track_and_classify_dwells_shared(foot_points_by_frame(tracks_by_frame, H), params)


def try_register_cameras():
    """Secondary, clearly-labeled attempt at the general N-camera
    registration module used for the museum-generalization request. cor and
    front look at genuinely disjoint parts of the corridor (down-the-hall
    vs. into-the-shop) from very different heights/angles -- unlike the
    EPFL Lab sequence's already-literally-shared frame, real ORB
    correspondence may or may not survive here. Reported as-is, not forced."""
    from mvtrack.court import register_cameras

    cor_img = cv2.imread(str(REPO_ROOT / "outputs" / "caviar_frames" / "cor_f500.png"))
    front_img = cv2.imread(str(REPO_ROOT / "outputs" / "caviar_frames" / "front_f500.png"))
    camera_models = {0: make_camera_model("cor"), 1: make_camera_model("front")}
    shared_images = {0: cor_img, 1: front_img}
    registrations = register_cameras(camera_models, 0, shared_images, max_residual_m=3.0)
    for cam, reg in sorted(registrations.items()):
        status = "included" if reg.included else "EXCLUDED"
        print(f"  {['cor','front'][cam]}: {status} (residual={reg.residual_m*100:.1f}cm, "
              f"{reg.n_correspondences} correspondences)")
    return registrations


def main():
    for view in ("cor", "front"):
        H = load_calibration(view)
        tracks_by_frame = run_mvtrack(view)
        tracks, dwellers = track_and_classify_dwells(tracks_by_frame, H)
        print(f"\n{view}: {len(tracks)} total tracks, {len(dwellers)} classify as "
              f"dwelling (>= {MIN_DWELL_SECONDS}s within {MAX_DWELL_RADIUS}cm)")
        for tid, (centroid, duration_s, radius) in sorted(dwellers.items(), key=lambda t: -t[1][1])[:10]:
            first_frame = tracks[tid][0][0]
            last_frame = tracks[tid][-1][0]
            print(f"    track {tid}: {duration_s:.1f}s at ({centroid[0]:.0f}, {centroid[1]:.0f})cm, "
                  f"radius {radius:.1f}cm, frames {first_frame}-{last_frame}")

        # Real companion pairs -- CAVIAR's own scenario description ("couple
        # walking on the corridor... later they rejoin") predicts real
        # people moving together in this footage; visually confirmed on cor
        # (track 2+1, two men walking side by side, frames 96-240).
        companions = find_companion_pairs(tracks, proximity_cm=150.0, min_fraction_together=0.6)
        companions = [c for c in companions
                      if min(tracks[c[0]][-1][0], tracks[c[1]][-1][0])
                      - max(tracks[c[0]][0][0], tracks[c[1]][0][0]) >= 20]
        print(f"    {len(companions)} companion pair(s) (>=60% of shared time within 150cm, >=20 frames):")
        for tid_a, tid_b, frac in sorted(companions, key=lambda c: -c[2])[:5]:
            lo = max(tracks[tid_a][0][0], tracks[tid_b][0][0])
            hi = min(tracks[tid_a][-1][0], tracks[tid_b][-1][0])
            print(f"        track {tid_a}+{tid_b}: {frac*100:.0f}% together, frames {lo}-{hi}")

        if view == "front":
            zone = storefront_zone()
            traffic = classify_zone_traffic(tracks, dwellers, zone)
            counts = {"stopped": 0, "passed": 0, "unrelated": 0}
            for v in traffic.values():
                counts[v] += 1
            rate = capture_rate(traffic)
            print(f"    storefront capture rate: {counts['stopped']} stopped / "
                  f"{counts['stopped'] + counts['passed']} passersby "
                  f"({counts['unrelated']} unrelated tracks excluded) = {rate:.2f}")

    print("\nattempting cross-camera registration (mvtrack.court.register_cameras)...")
    try_register_cameras()


if __name__ == "__main__":
    main()
