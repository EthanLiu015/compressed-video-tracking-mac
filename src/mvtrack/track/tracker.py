"""Minimal IoU-matching track manager.

Two update modes per frame:
- `step_anchor`: full detector output re-associates with existing tracks via
  IoU (Hungarian assignment). This is the only place identities are created,
  corrected, or dropped.
- `step_propagate`: no detector; every live track's box is shifted using the
  frame's motion-vector grid (see `propagate.py`). Track identity is never
  re-decided here — MV propagation stands in for a Kalman-filter predict
  step, since the encoder already computed the motion.

This plays the role ByteTrack's Kalman predict + IoU-match loop plays in the
full-decode baseline, but the "predict" comes from the codec instead of a
motion model — that swap is the whole point of the project.
"""

from dataclasses import dataclass, field

import numpy as np
from scipy.optimize import linear_sum_assignment

from mvtrack.extract import FrameMV
from mvtrack.track.propagate import propagate_boxes


def _iou_matrix(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    if len(a) == 0 or len(b) == 0:
        return np.zeros((len(a), len(b)), np.float32)
    ax0, ay0, ax1, ay1 = a[:, 0:1], a[:, 1:2], a[:, 2:3], a[:, 3:4]
    bx0, by0, bx1, by1 = b[:, 0], b[:, 1], b[:, 2], b[:, 3]
    ix0, iy0 = np.maximum(ax0, bx0), np.maximum(ay0, by0)
    ix1, iy1 = np.minimum(ax1, bx1), np.minimum(ay1, by1)
    inter = np.clip(ix1 - ix0, 0, None) * np.clip(iy1 - iy0, 0, None)
    area_a = ((ax1 - ax0) * (ay1 - ay0))
    area_b = (bx1 - bx0) * (by1 - by0)
    union = area_a + area_b - inter
    return np.where(union > 0, inter / union, 0.0)


@dataclass
class Track:
    id: int
    box: np.ndarray  # xyxy
    score: float
    since_detection: int = 0  # frames since last detector match; pruned past max_age
    hits: int = 1


@dataclass
class MVTracker:
    iou_thresh: float = 0.3
    max_age: int = 30
    high_conf_thresh: float = 0.5  # splits detections into ByteTrack-style high/low buckets
    iou_thresh_low: float = 0.2  # stage 2: low-conf dets recovering a track, looser gate
    iou_thresh_grace: float = 0.15  # stage 3: last chance before spawning a new ID
    _next_id: int = field(default=1, init=False)
    _tracks: dict[int, Track] = field(default_factory=dict, init=False)

    def _spawn(self, box: np.ndarray, score: float) -> None:
        self._tracks[self._next_id] = Track(id=self._next_id, box=box, score=score)
        self._next_id += 1

    def step_anchor(self, boxes: np.ndarray, scores: np.ndarray) -> list[Track]:
        """Three-stage ByteTrack-style association, replacing a single flat
        IoU match. A single pass caused real ID churn: any anchor frame
        where the detector missed a track (common with YOLOv8's imperfect
        recall) immediately spawned a brand-new identity, and the
        anchor-interval ablation showed this made *more* anchors hurt MOTA.

        Stage 1: high-confidence detections vs. all tracks (the original
          logic, at the original threshold).
        Stage 2: low-confidence detections (which never spawn new tracks
          on their own) get a chance to recover tracks stage 1 missed, at
          a looser IoU gate -- exactly ByteTrack's core idea.
        Stage 3 (grace period): detections still unmatched after stage 1
          get one more loose-IoU chance to reattach to a still-unmatched
          track before falling through to spawning a new ID.
        Only high-confidence detections ever spawn a new track; unmatched
        low-confidence ones are discarded as too unreliable to seed an ID.
        """
        ids = list(self._tracks.keys())
        high_idx = np.where(scores >= self.high_conf_thresh)[0]
        low_idx = np.where(scores < self.high_conf_thresh)[0]

        matched_tracks: set[int] = set()
        matched_high: set[int] = set()

        def _apply(track_ids, det_idx, thresh):
            m_tracks, m_dets = set(), set()
            if not track_ids or len(det_idx) == 0:
                return m_tracks, m_dets
            track_boxes = np.array([self._tracks[t].box for t in track_ids])
            iou = _iou_matrix(track_boxes, boxes[det_idx])
            row, col = linear_sum_assignment(-iou)
            for r, c in zip(row, col):
                if iou[r, c] >= thresh:
                    tid = track_ids[r]
                    di = int(det_idx[c])
                    self._tracks[tid].box = boxes[di]
                    self._tracks[tid].score = scores[di]
                    self._tracks[tid].since_detection = 0
                    self._tracks[tid].hits += 1
                    m_tracks.add(tid)
                    m_dets.add(di)
            return m_tracks, m_dets

        # Stage 1: high-confidence detections vs. all tracks.
        m1_tracks, m1_dets = _apply(ids, high_idx, self.iou_thresh)
        matched_tracks |= m1_tracks
        matched_high |= m1_dets

        # Stage 2: low-confidence detections recover tracks stage 1 missed.
        remaining = [t for t in ids if t not in matched_tracks]
        _m2_tracks, _m2_dets = _apply(remaining, low_idx, self.iou_thresh_low)
        matched_tracks |= _m2_tracks
        # low-conf detections never spawn, so their indices aren't tracked further

        # Stage 3: grace period for still-unmatched high-conf detections.
        still_unmatched = [t for t in ids if t not in matched_tracks]
        unmatched_high = np.array([di for di in high_idx if di not in matched_high])
        m3_tracks, m3_dets = _apply(still_unmatched, unmatched_high, self.iou_thresh_grace)
        matched_tracks |= m3_tracks
        matched_high |= m3_dets

        for tid in ids:
            if tid not in matched_tracks:
                self._tracks[tid].since_detection += 1
        for tid in [t for t, tr in self._tracks.items() if tr.since_detection > self.max_age]:
            del self._tracks[tid]

        for di in high_idx:
            if int(di) not in matched_high:
                self._spawn(boxes[di], scores[di])

        return self.active_tracks()

    def step_propagate(self, fmv: FrameMV) -> list[Track]:
        if self._tracks:
            ids = list(self._tracks.keys())
            boxes = np.array([self._tracks[i].box for i in ids])
            new_boxes = propagate_boxes(boxes, fmv)
            for tid, box in zip(ids, new_boxes):
                self._tracks[tid].box = box
                self._tracks[tid].since_detection += 1
        for tid in [t for t, tr in self._tracks.items() if tr.since_detection > self.max_age]:
            del self._tracks[tid]
        return self.active_tracks()

    def correct_boxes(self, new_boxes: dict[int, np.ndarray]) -> list[Track]:
        """Overwrite live tracks' boxes (e.g. with a learned correction
        applied on top of MV propagation). Does not touch since_detection."""
        for tid, box in new_boxes.items():
            if tid in self._tracks:
                self._tracks[tid].box = box
        return self.active_tracks()

    def active_tracks(self) -> list[Track]:
        return list(self._tracks.values())
