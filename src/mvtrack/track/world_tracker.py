"""World-space (fused, ground-plane) point tracker.

Moved here from scripts/epfl_rlc_fusion.py, where it was originally built and
validated, because it's core tracking infrastructure reused by every PULSE
fusion script (EPFL-RLC, EPFL Lab, CAVIAR shop/meet/leftbag) rather than
something specific to any one dataset's fusion pipeline.
"""

import numpy as np
from scipy.optimize import linear_sum_assignment


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

    def all_tracks(self) -> dict:
        """Completed + still-active tracks, merged -- the pattern every
        caller used to repeat manually (`dict(tracker.completed);
        tracks.update(...)`)."""
        tracks = dict(self.completed)
        tracks.update({tid: tr["history"] for tid, tr in self.tracks.items()})
        return tracks
