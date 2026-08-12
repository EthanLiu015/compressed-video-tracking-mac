"""Shared dwell classification core -- was copy-pasted with minor variations
across scripts/epfl_rlc_fusion.py, scripts/epfl_lab_fusion.py, and
scripts/caviar_shop_fusion.py; consolidated here so new PULSE scripts don't
repeat it a fourth time. Each calling script stays responsible for its own
pixel-to-world projection (Tsai, planar homography, whatever) -- this module
only sees already-projected per-frame world points.
"""

from dataclasses import dataclass

import numpy as np

from mvtrack.track import WorldTracker


@dataclass
class DwellParams:
    frame_dt: float
    min_dwell_seconds: float
    min_dwell_radius: float
    max_dwell_radius: float
    max_step: float
    max_age: int = 10


def track_and_classify_dwells(per_frame_points: dict, params: DwellParams):
    """`per_frame_points`: frame_id -> Nx2 world-space points for that frame.
    Returns (all_tracks, dwellers) where `dwellers` maps track id ->
    (centroid, duration_s, radius) for tracks meeting both the duration and
    radius bounds -- `min_dwell_radius` rejects exact-static-object false
    positives (findings.md's coat-on-a-hook class of bug), `max_dwell_radius`
    rejects a long walk that happens to stay in frame that long."""
    tracker = WorldTracker(max_step=params.max_step, max_age=params.max_age)
    for fid in sorted(per_frame_points.keys()):
        tracker.step(fid, per_frame_points[fid])
    tracks = tracker.all_tracks()

    dwellers = {}
    for tid, pts in tracks.items():
        duration_s = len(pts) * params.frame_dt
        if duration_s < params.min_dwell_seconds:
            continue
        xy = np.array([p for _, p in pts])
        centroid = xy.mean(axis=0)
        radius = np.linalg.norm(xy - centroid, axis=1).max()
        if params.min_dwell_radius <= radius <= params.max_dwell_radius:
            dwellers[tid] = (centroid, duration_s, radius)
    return tracks, dwellers
