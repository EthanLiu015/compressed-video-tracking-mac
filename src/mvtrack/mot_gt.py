"""Shared MOT17 ground-truth loading for dataset-building scripts."""

import pathlib

import numpy as np
import pandas as pd

GT_COLS = ["frame", "id", "l", "t", "w", "h", "conf", "cls", "vis"]


def load_gt(seq_dir: pathlib.Path) -> dict[int, dict[int, np.ndarray]]:
    """frame -> {track_id: box_xyxy}, pedestrian class + evaluated boxes only."""
    df = pd.read_csv(seq_dir / "gt" / "gt.txt", header=None, names=GT_COLS)
    df = df[(df["cls"] == 1) & (df["conf"] == 1)]
    out: dict[int, dict[int, np.ndarray]] = {}
    for row in df.itertuples():
        box = np.array([row.l, row.t, row.l + row.w, row.t + row.h], np.float32)
        out.setdefault(row.frame, {})[row.id] = box
    return out
