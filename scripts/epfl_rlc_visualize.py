"""Two visualizations of the PULSE/EPFL-RLC pipeline, now that the real
fragmentation bug is fixed:

1. A top-down "meshed homography" map: cam0's and cam2's fields of view,
   projected onto the shared ground plane (post-registration), so you can
   see how their coverage actually overlaps in the unified world frame --
   not just trust that it does.
2. An annotated video: cam0's real footage with live track ID + running
   dwell-candidate status drawn on top, so tracking/dwell behavior can be
   watched directly instead of read off a table of numbers.
"""

import pathlib
import sys

import cv2
import matplotlib
import numpy as np

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.patches import Polygon  # noqa: E402

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))
from epfl_rlc_fusion import (  # noqa: E402
    DATA_ROOT, FRAME_DT, MAX_DWELL_RADIUS, MIN_DWELL_SECONDS, OUT_DIR,
    VIDEO_ROOT, filter_static_pixel_regions, filter_unstable_projections,
    load_calibration, load_mot_tracks, pixel_to_ground, project_and_fuse,
    projection_sensitivity, register_all_cameras,
)
from mvtrack.track import WorldTracker  # noqa: E402

VIZ_DIR = OUT_DIR.parent / "epfl_rlc_viz"
_CAM_COLOR_CYCLE = ["tab:blue", "tab:orange", "tab:green", "tab:red", "tab:purple", "tab:brown"]


def cam_color(cam: int) -> str:
    return _CAM_COLOR_CYCLE[cam % len(_CAM_COLOR_CYCLE)]


def camera_fov_footprint(cam: int, calib, registration, n=60):
    """Project this camera's image bottom rows onto the ground plane -- a
    rough visual guide to where each camera looks, not a precision data
    source. Sampling several rows near the bottom (not just the very last
    pixel row, which is exactly where grazing-ray instability is worst)
    and using a looser stability cutoff than real tracking data gets a
    usable footprint without the near-empty result the strict filter gave
    or the >1000m blowup no filter gave."""
    params, R, t = calib
    w, h = 480, 270
    pts = np.array([(x, y) for y in [h - 15, h - 8, h - 1] for x in np.linspace(0, w - 1, n)])
    with np.errstate(all="ignore"):
        sens = projection_sensitivity(pts, params, R, t)
    stable = sens <= 800.0  # looser than MAX_MM_PER_PX -- a visual guide, not tracked data
    with np.errstate(all="ignore"):
        world = pixel_to_ground(pts[stable], params, R, t)
    return registration.apply(world)


def render_mesh_map():
    print("rendering top-down meshed-homography map...")
    calibs = {c: load_calibration(c) for c in range(3)}
    registrations = register_all_cameras()
    included_cams = [c for c, reg in registrations.items() if reg.included]

    per_cam_tracks = {c: load_mot_tracks(str(OUT_DIR / f"mv_fixed_cam{c}.txt")) for c in range(3)}
    per_cam_tracks = filter_static_pixel_regions(per_cam_tracks)
    per_frame_fused = project_and_fuse(per_cam_tracks, calibs, registrations)

    frame_ids = sorted(per_frame_fused.keys())
    tracker = WorldTracker(max_step=2000.0, max_age=10)
    for fid in frame_ids:
        tracker.step(fid, per_frame_fused[fid])
    tracks = dict(tracker.completed)
    tracks.update({tid: tr["history"] for tid, tr in tracker.tracks.items()})

    fig, ax = plt.subplots(figsize=(10, 8))
    for cam in included_cams:
        footprint = camera_fov_footprint(cam, calibs[cam], registrations[cam]) / 1000.0
        order = np.argsort(np.arctan2(footprint[:, 1] - footprint[:, 1].mean(),
                                       footprint[:, 0] - footprint[:, 0].mean()))
        poly = Polygon(footprint[order], closed=True, alpha=0.15, color=cam_color(cam),
                        label=f"cam{cam} ground-plane footprint")
        ax.add_patch(poly)
        ax.plot(footprint[:, 0], footprint[:, 1], "o", color=cam_color(cam), markersize=2)

    for tid, pts in tracks.items():
        xy = np.array([p for _, p in pts]) / 1000.0
        if len(xy) < 10:
            continue
        ax.plot(xy[:, 0], xy[:, 1], "-", color="gray", alpha=0.3, linewidth=0.6)

    for tid, pts in tracks.items():
        duration_s = len(pts) * FRAME_DT
        if duration_s < MIN_DWELL_SECONDS:
            continue
        xy = np.array([p for _, p in pts]) / 1000.0
        centroid = xy.mean(axis=0)
        radius = np.linalg.norm(xy - centroid, axis=1).max()
        if radius > MAX_DWELL_RADIUS / 1000.0:
            continue  # long-duration but not a real dweller -- just a long walk, don't mislabel it
        circ = plt.Circle(centroid, radius, color="red", fill=False, linewidth=2, zorder=5)
        ax.add_patch(circ)
        ax.annotate(f"{duration_s:.1f}s", centroid, color="red", fontsize=9, zorder=6)

    # Clip view to the real track data's own range (+ margin) -- a safety
    # net so any remaining footprint outlier (this projection genuinely has
    # a long tail even after filtering, see module/pixel_to_ground docs)
    # can't silently rescale the whole plot and hide the actual data again.
    all_track_pts = np.concatenate([np.array([p for _, p in pts]) for pts in tracks.values()]) / 1000.0
    margin = 3.0
    ax.set_xlim(all_track_pts[:, 0].min() - margin, all_track_pts[:, 0].max() + margin)
    ax.set_ylim(all_track_pts[:, 1].min() - margin, all_track_pts[:, 1].max() + margin)

    ax.set_xlabel("world X (m)")
    ax.set_ylabel("world Y (m)")
    ax.set_title("EPFL-RLC: meshed camera coverage (post-registration) + tracks + dwell candidates")
    ax.legend(loc="upper right", fontsize=8)
    ax.set_aspect("equal")
    fig.tight_layout()
    VIZ_DIR.mkdir(parents=True, exist_ok=True)
    out_path = VIZ_DIR / "mesh_map.png"
    fig.savefig(out_path, dpi=130)
    print(f"saved -> {out_path}")


def render_annotated_video(start_frame=5900, end_frame=6700):
    """Cam0's real footage, live tracking + dwell-candidate status drawn on
    top, for the window containing the one dwell candidate found so far."""
    print(f"rendering annotated video for frames {start_frame}-{end_frame}...")
    params, R, t = load_calibration(0)
    tracks_raw = load_mot_tracks(str(OUT_DIR / "mv_fixed_cam0.txt"))
    tracks_raw = filter_static_pixel_regions({0: tracks_raw})[0]

    tracker = WorldTracker(max_step=2000.0, max_age=10)
    # running dwell state per world-track id, updated as we go so the video
    # shows dwell status becoming true live, not computed with hindsight
    frame_to_tid_pos = {}
    for frame in sorted(tracks_raw.keys()):
        boxes = np.array(tracks_raw[frame])
        foot = np.stack([(boxes[:, 0] + boxes[:, 2]) / 2.0, boxes[:, 3]], axis=1)
        stable = filter_unstable_projections(foot, params, R, t)
        with np.errstate(all="ignore"):
            world = pixel_to_ground(foot[stable], params, R, t)
        tracker.step(frame, world)
        # remember which raw box each active world-track currently maps to,
        # by nearest-position match (good enough for visualization)
        active_pos = {tid: tr["pos"] for tid, tr in tracker.tracks.items()}
        frame_to_tid_pos[frame] = (boxes[stable], active_pos)

    VIZ_DIR.mkdir(parents=True, exist_ok=True)
    out_path = VIZ_DIR / "cam0_tracking_dwell.mp4"
    writer = cv2.VideoWriter(str(out_path), cv2.VideoWriter_fourcc(*"mp4v"), 60, (480 * 2, 270 * 2))

    frame_dir = DATA_ROOT / "frames" / "cam0"
    track_start_frame = {}
    for frame in range(start_frame, end_frame):
        img_path = frame_dir / f"RLCAFTCONF-C0_{100000 + frame - 1}.jpeg"
        img = cv2.imread(str(img_path))
        if img is None or frame not in frame_to_tid_pos:
            continue
        img = cv2.resize(img, (480 * 2, 270 * 2))
        boxes, active_pos = frame_to_tid_pos[frame]
        for i, (x0, y0, x1, y1) in enumerate(boxes):
            best_tid, best_d = None, 1e9
            fx, fy = (x0 + x1) / 2.0, y1
            with np.errstate(all="ignore"):
                world_pt = pixel_to_ground(np.array([[fx, fy]]), params, R, t)[0]
            for tid, pos in active_pos.items():
                d = np.linalg.norm(pos - world_pt)
                if d < best_d:
                    best_tid, best_d = tid, d
            if best_tid is None:
                continue
            track_start_frame.setdefault(best_tid, frame)
            dwell_time = (frame - track_start_frame[best_tid]) * FRAME_DT
            is_dwelling = dwell_time >= MIN_DWELL_SECONDS
            color = (0, 0, 255) if is_dwelling else (0, 255, 0)
            cv2.rectangle(img, (int(x0 * 2), int(y0 * 2)), (int(x1 * 2), int(y1 * 2)), color, 2)
            label = f"id{best_tid} {dwell_time:.1f}s" + (" DWELL" if is_dwelling else "")
            cv2.putText(img, label, (int(x0 * 2), int(y0 * 2) - 6),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)
        writer.write(img)
    writer.release()
    print(f"saved -> {out_path}")


if __name__ == "__main__":
    render_mesh_map()
    render_annotated_video()
