"""Approach dynamics: not just "did they stop," but "did they slow down and
look without ever fully stopping" -- retail's own term for this is "window
shopping," a real, distinct interest signal from a full dwell. `WorldTracker`
already computes an EMA-smoothed velocity per track for matching purposes
but never surfaces it (src/mvtrack/track/world_tracker.py); this module
doesn't touch that internal state at all -- it recomputes a speed profile
directly from the already-stored position history, which is simpler and
keeps this analysis fully decoupled from the tracker's own matching logic.
"""

import numpy as np

from .zones import Zone, distance_to_polygon


def speed_profile(track_history, frame_dt: float) -> np.ndarray:
    """World-units-per-second speed for each consecutive step in a track's
    history. Length is len(track_history) - 1 (undefined for a single-point
    track)."""
    xy = np.array([p for _, p in track_history])
    if len(xy) < 2:
        return np.array([])
    step_dist = np.linalg.norm(np.diff(xy, axis=0), axis=1)
    return step_dist / frame_dt


def classify_approach(
    track_history, zone: Zone, frame_dt: float, is_dweller: bool, decel_fraction: float = 0.4
) -> str | None:
    """'stopped' (already a confirmed dweller -- reuse that classification,
    don't duplicate it), 'decelerated' (came near the zone and visibly
    slowed relative to its own speed elsewhere on the same track, but never
    met the dwell criteria), 'passthrough' (near the zone at constant
    speed), or None (this track's trajectory never came near the zone at
    all -- not a zone-relevant observation)."""
    if is_dweller:
        return "stopped"

    xy = np.array([p for _, p in track_history])
    dist_to_zone = np.array([distance_to_polygon(p, zone.polygon) for p in xy])
    if not (dist_to_zone <= zone.proximity_cm).any():
        return None

    speeds = speed_profile(track_history, frame_dt)
    if len(speeds) < 2:
        return "passthrough"  # too short a track to say anything about dynamics

    # speeds[i] is the step from xy[i] to xy[i+1]; treat it as "near the
    # zone" if either endpoint of that step was within the proximity buffer.
    near_step = (dist_to_zone[:-1] <= zone.proximity_cm) | (dist_to_zone[1:] <= zone.proximity_cm)
    if not near_step.any():
        return "passthrough"

    near_speed = speeds[near_step].mean()
    far_speed = speeds[~near_step].mean() if (~near_step).any() else speeds.mean()
    if far_speed <= 1e-6:
        return "passthrough"

    if (far_speed - near_speed) / far_speed >= decel_fraction:
        return "decelerated"
    return "passthrough"
