"""Zone "activity/bustle" straight from the compressed bitstream -- no
detector, no inference at all. This is the one PULSE metric that's actually
on-brand for a project whose whole premise is that useful signal lives in
the codec: `Adaptive` (src/mvtrack/sched/scheduler.py) already reads
per-block intra/inter occupancy as an internal scheduling signal; this
module repurposes the same cheap per-frame pass, plus the actual motion
vector magnitudes it never uses, as an output metric in its own right.

Combines two real signals per zone, per frame:
  - motion energy: mean |mv_grid| over inter-coded (real motion-matched)
    cells inside the zone -- literal crowd-movement magnitude.
  - intra fraction: 1 - occupancy.mean() restricted to the zone -- the same
    proxy `Adaptive` uses for "the encoder couldn't motion-match here",
    which fires at scene cuts/occlusion/new content, not just movement.
Neither alone is a full activity signal (a stationary crowd has near-zero
motion energy but might still show intra spikes from people shuffling in
frame; a single fast walker can spike motion energy without much intra
change) -- combined here as each series' own z-score, summed, so no
cross-series unit/weighting assumption is baked in.
"""

import numpy as np

from mvtrack.extract import BLOCK, iter_frames_with_mvs

from .zones import point_in_polygon


def zone_cell_mask(width: int, height: int, zone_polygon_px) -> np.ndarray:
    """(H/BLOCK, W/BLOCK) bool grid, True where that block's center pixel
    falls inside `zone_polygon_px` (pixel-space polygon, same convention as
    every other hand-picked zone in this project)."""
    h_cells = (height + BLOCK - 1) // BLOCK
    w_cells = (width + BLOCK - 1) // BLOCK
    mask = np.zeros((h_cells, w_cells), dtype=bool)
    for cy in range(h_cells):
        for cx in range(w_cells):
            px = cx * BLOCK + BLOCK / 2.0
            py = cy * BLOCK + BLOCK / 2.0
            mask[cy, cx] = point_in_polygon((px, py), zone_polygon_px)
    return mask


def _zscore(x: np.ndarray) -> np.ndarray:
    s = x.std()
    return (x - x.mean()) / s if s > 1e-9 else np.zeros_like(x)


def zone_energy_trace(video: str, zone_polygon_px, max_frames: int | None = None):
    """Returns (frame_indices, motion_energy, intra_fraction, combined) --
    four equal-length arrays, one entry per decoded frame. `combined` is the
    sum of each series' own z-score (see module docstring for why not a
    fixed weighted sum)."""
    mask = None
    frame_indices, motion, intra = [], [], []
    for fmv, _frame in iter_frames_with_mvs(video):
        if mask is None:
            mask = zone_cell_mask(fmv.width, fmv.height, zone_polygon_px)
            if not mask.any():
                raise ValueError("zone_polygon_px selects zero grid cells -- check pixel coordinates")
        mag = np.linalg.norm(fmv.mv_grid, axis=2)
        inter_in_zone = fmv.occupancy.astype(bool) & mask
        motion.append(float(mag[inter_in_zone].mean()) if inter_in_zone.any() else 0.0)
        intra.append(float(1.0 - fmv.occupancy[mask].mean()))
        frame_indices.append(fmv.index)
        if max_frames is not None and len(frame_indices) >= max_frames:
            break

    motion = np.array(motion)
    intra = np.array(intra)
    combined = _zscore(motion) + _zscore(intra)
    return np.array(frame_indices), motion, intra, combined
