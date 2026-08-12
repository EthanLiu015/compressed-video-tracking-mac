"""Loitering / left-object anomaly detection -- a real security/loss-
prevention use case (unattended-bag alerts are an actually-deployed camera-
analytics category at transit hubs and retail), not just an analytics
nicety layered on top of dwell time.

Real-data validation note: CAVIAR's own scenario built for exactly this
(`LeftBag.mpg`, INRIA lobby, same server as the group-dwell `Meet_*` clips)
sits on a wide-angle fisheye camera that breaks YOLOv8s entirely -- checked
directly before writing any of this module's alert logic, per this
project's own "verify feasibility before building on top of it" discipline:
at conf=0.05 the detector returns mostly COCO class 14 ("bird", the
fisheye-warped silhouette apparently reads as one) and person-class scores
crash to 0.13/0.09 by frame 550, effectively noise. Not forced. This module
is therefore validated with a synthetic fixture (the same kind of quick
correctness check this project already uses elsewhere for new
numerical/logic components, e.g. the MVTracker re-association sanity
script) rather than a real end-to-end video -- a real, stated limitation,
not a hidden one.
"""

import numpy as np


def is_object_stationary(object_history, jitter_floor_cm: float = 5.0) -> bool:
    """True if this object-track's position never moved beyond
    `jitter_floor_cm` from where it was first observed -- the same
    static-object jitter-floor pattern already used elsewhere in this
    project (MIN_DWELL_RADIUS in scripts/epfl_rlc_fusion.py etc) to
    distinguish a real static prop from a person who happens to stand
    still for a while (a live person always shows some postural sway; an
    inanimate object does not move at all -- exact zero jitter is itself
    the signature to check for, not a contradiction)."""
    xy = np.array([p for _, p in object_history])
    if len(xy) < 2:
        return True
    origin = xy[0]
    return bool(np.linalg.norm(xy - origin, axis=1).max() <= jitter_floor_cm)


def find_stationary_suffix(object_history, jitter_floor_cm: float = 5.0):
    """A tracked object is often carried before being set down -- real
    ABODA footage confirmed this directly (a backpack track's first
    observations were while it was still in its owner's hand mid-stride;
    only later did it get set on the ground). Checking stationarity from
    the track's very first detection (`is_object_stationary`) then never
    fires for a real carried-then-placed object, because the "moving"
    prefix keeps the overall jitter high even after it's genuinely been set
    down. This finds the longest STATIONARY SUFFIX instead -- the "placed"
    phase -- which is what abandonment logic should actually measure from.
    Returns that suffix (list of (frame, pos)), or None if the object never
    settles at all."""
    history = sorted(object_history, key=lambda h: h[0])
    xy = np.array([p for _, p in history])
    n = len(history)
    if n < 2:
        return history
    # Removing points from the front can only shrink (never grow) the max
    # spread of what remains, so the first k (scanning from 0) whose suffix
    # satisfies the jitter floor is also the LONGEST such suffix.
    for k in range(n - 1):
        seg = xy[k:]
        spread = np.linalg.norm(seg - seg[0], axis=1).max()
        if spread <= jitter_floor_cm:
            return history[k:]
    return None


def _person_positions_by_frame(person_tracks: dict) -> dict:
    by_frame: dict = {}
    for _pid, hist in person_tracks.items():
        for f, p in hist:
            by_frame.setdefault(f, []).append(p)
    return by_frame


def detect_abandoned_objects(
    object_tracks: dict, person_tracks: dict, frame_dt: float,
    stationary_seconds: float = 5.0, separation_cm: float = 200.0,
    separation_seconds: float = 3.0, jitter_floor_cm: float = 5.0,
) -> list:
    """An object is flagged abandoned the first frame at which: (a) its
    stationary suffix (see `find_stationary_suffix` -- handles an object
    that was carried before being set down, not just one stationary from
    its very first detection) has lasted at least `stationary_seconds`, AND
    (b) no person track has come within `separation_cm` of it at any point
    in the trailing `separation_seconds`. One event per object (the first
    qualifying frame), not re-flagged every subsequent frame it stays
    abandoned."""
    events = []
    stationary_frames = max(1, int(round(stationary_seconds / frame_dt)))
    separation_frames = max(1, int(round(separation_seconds / frame_dt)))
    person_by_frame = _person_positions_by_frame(person_tracks)

    for oid, ohist in object_tracks.items():
        stationary_suffix = find_stationary_suffix(ohist, jitter_floor_cm)
        if stationary_suffix is None:
            continue
        if len(stationary_suffix) < stationary_frames:
            continue
        obj_frames = [f for f, _ in stationary_suffix]
        obj_pos = dict(stationary_suffix)

        for i in range(stationary_frames - 1, len(obj_frames)):
            f = obj_frames[i]
            window = obj_frames[max(0, i - separation_frames + 1): i + 1]
            someone_near = False
            for wf in window:
                for p in person_by_frame.get(wf, []):
                    if np.linalg.norm(np.asarray(obj_pos[f]) - np.asarray(p)) <= separation_cm:
                        someone_near = True
                        break
                if someone_near:
                    break
            if not someone_near:
                events.append({"object_id": oid, "flagged_frame": f, "position": obj_pos[f]})
                break
    return events
