"""Build an on-policy rollout training set for CorrectionNet (DAgger-style
fix for the train/inference distribution mismatch found in weeks 8-10).

build_correction_dataset.py only ever supervises a single propagation step
starting from a perfect GT box. At inference (eval.run.run_mv_learned),
correction is applied every non-anchor frame across an anchor window,
chaining the net's own (imperfect) output back into the next propagation
step -- a distribution the single-step net never saw during training.

This script re-walks each anchor window (same anchor_interval as eval)
using a trained checkpoint to produce the actual rollout trajectory, and
records the target the net *should* have produced at each step along that
trajectory -- i.e. on-policy data, same idea as DAgger.

Usage: python scripts/build_rollout_dataset.py --checkpoint outputs/correction_net.pt
Writes outputs/correction_dataset_rollout.npz (same schema as the single-step set).
"""

import argparse
import pathlib
import sys

import numpy as np
import torch

ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from mvtrack.detect import pick_device
from mvtrack.extract import iter_frames_with_mvs
from mvtrack.mot_gt import load_gt
from mvtrack.track.correct import CorrectionNet, apply_correction, extract_features
from mvtrack.track.propagate import propagate_boxes

MOT = ROOT / "data" / "MOT17"


def build_sequence(
    seq_name: str, net: CorrectionNet, device: str, anchor_interval: int
) -> tuple[np.ndarray, np.ndarray]:
    seq_dir = MOT / "train" / seq_name
    video = MOT / "videos" / f"{seq_name}.mp4"
    gt = load_gt(seq_dir)

    feats, labels = [], []
    live: dict[int, np.ndarray] = {}  # track_id -> current rolled-out box
    for idx, (fmv, _frame) in enumerate(iter_frames_with_mvs(str(video))):
        frame_no = idx + 1  # gt.txt is 1-indexed
        is_anchor = (frame_no - 1) % anchor_interval == 0
        cur_gt = gt.get(frame_no, {})

        if is_anchor:
            live = dict(cur_gt)  # "detector" = perfect GT, same isolation as before
            continue
        if not live:
            continue

        ids = list(live.keys())
        prop = propagate_boxes(np.stack([live[i] for i in ids]), fmv)

        keep_ids, keep_prop = [], []
        for i, tid in enumerate(ids):
            if tid in cur_gt:
                w = max(prop[i, 2] - prop[i, 0], 1.0)
                h = max(prop[i, 3] - prop[i, 1], 1.0)
                label = (cur_gt[tid] - prop[i]) / np.array([w, h, w, h], np.float32)
                feats.append(extract_features(prop[i], fmv))
                labels.append(label)
                keep_ids.append(tid)
                keep_prop.append(prop[i])

        if not keep_ids:
            live = {}
            continue
        corrected = apply_correction(net, np.stack(keep_prop), fmv, device=device)
        live = dict(zip(keep_ids, corrected))  # chain the net's OWN output forward

    return np.array(feats, np.float32), np.array(labels, np.float32)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--checkpoint", default=str(ROOT / "outputs" / "correction_net.pt"))
    ap.add_argument("--anchor-interval", type=int, default=5)
    args = ap.parse_args()

    device = pick_device()
    net = CorrectionNet().to(device)
    net.load_state_dict(torch.load(args.checkpoint, map_location=device))
    net.eval()

    seqs = sorted(p.name for p in (MOT / "train").glob("MOT17-*-FRCNN"))
    all_feats, all_labels, seq_ids = [], [], []
    for i, seq in enumerate(seqs):
        f, l = build_sequence(seq, net, device, args.anchor_interval)
        print(f"{seq}: {len(f)} rollout pairs")
        all_feats.append(f)
        all_labels.append(l)
        seq_ids.append(np.full(len(f), i))

    feats = np.concatenate(all_feats)
    labels = np.concatenate(all_labels)
    seq_ids = np.concatenate(seq_ids)
    out = ROOT / "outputs" / "correction_dataset_rollout.npz"
    np.savez(out, feats=feats, labels=labels, seq_ids=seq_ids, seq_names=np.array(seqs))
    print(f"total: {len(feats)} pairs -> {out}")
