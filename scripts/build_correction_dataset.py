"""Build a (feature, residual) training set for CorrectionNet from MOT17 GT.

For each pair of consecutive frames a track appears in, propagate its GT
box from frame f to f+1 with the existing MV-median method, then record the
leftover error against the real GT box at f+1. This isolates propagation
error from detector error entirely -- no detector runs here.

Usage: python scripts/build_correction_dataset.py
Writes outputs/correction_dataset.npz (features, labels, seq_id per row).
"""

import pathlib
import sys

import numpy as np

ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from mvtrack.extract import iter_frames_with_mvs
from mvtrack.mot_gt import load_gt
from mvtrack.track.correct import extract_features
from mvtrack.track.propagate import propagate_boxes

MOT = ROOT / "data" / "MOT17"


def build_sequence(seq_name: str) -> tuple[np.ndarray, np.ndarray]:
    seq_dir = MOT / "train" / seq_name
    video = MOT / "videos" / f"{seq_name}.mp4"
    gt = load_gt(seq_dir)

    feats, labels = [], []
    prev_boxes: dict[int, np.ndarray] | None = None
    for idx, (fmv, _frame) in enumerate(iter_frames_with_mvs(str(video))):
        gt_frame = idx + 1  # gt.txt is 1-indexed
        cur_boxes = gt.get(gt_frame, {})
        if prev_boxes:
            common = set(prev_boxes) & set(cur_boxes)
            if common:
                ids = list(common)
                prev = np.stack([prev_boxes[i] for i in ids])
                prop = propagate_boxes(prev, fmv)
                gt_next = np.stack([cur_boxes[i] for i in ids])
                w = np.clip(prop[:, 2] - prop[:, 0], 1.0, None)
                h = np.clip(prop[:, 3] - prop[:, 1], 1.0, None)
                scale = np.stack([w, h, w, h], axis=1)
                resid = (gt_next - prop) / scale
                for p, r in zip(prop, resid):
                    feats.append(extract_features(p, fmv))
                    labels.append(r)
        prev_boxes = cur_boxes
    return np.array(feats, np.float32), np.array(labels, np.float32)


if __name__ == "__main__":
    seqs = sorted(p.name for p in (MOT / "train").glob("MOT17-*-FRCNN"))
    all_feats, all_labels, seq_ids = [], [], []
    for i, seq in enumerate(seqs):
        f, l = build_sequence(seq)
        print(f"{seq}: {len(f)} training pairs")
        all_feats.append(f)
        all_labels.append(l)
        seq_ids.append(np.full(len(f), i))

    feats = np.concatenate(all_feats)
    labels = np.concatenate(all_labels)
    seq_ids = np.concatenate(seq_ids)
    out = ROOT / "outputs" / "correction_dataset.npz"
    out.parent.mkdir(parents=True, exist_ok=True)
    np.savez(out, feats=feats, labels=labels, seq_ids=seq_ids, seq_names=np.array(seqs))
    print(f"total: {len(feats)} pairs -> {out}")
