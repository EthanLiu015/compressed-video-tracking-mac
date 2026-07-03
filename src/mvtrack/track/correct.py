"""Learned MV-domain box correction (weeks 8-10, stretch goal in the plan).

`propagate_boxes` is a zero-order model: shift by the median MV under the
box. It can't correct for partial-occlusion drift, box-scale drift (subject
moving toward/away from camera), or systematic per-part motion within a box.
`CorrectionNet` is a tiny MLP over a pooled MV/occupancy patch that predicts
a residual on top of the already-propagated box. Trained on real MOT17
propagation targets: GT box at frame t -> GT box at frame t+1, which
isolates pure propagation error from detector error (see
scripts/build_correction_dataset.py).
"""

from dataclasses import dataclass

import cv2
import numpy as np
import torch
from torch import nn

from mvtrack.extract import BLOCK, FrameMV

PATCH = 4  # pooled patch side length (cells)
FEAT_DIM = PATCH * PATCH * 3 + 2  # (dx, dy, occupancy) grid + log(w), log(h)


def extract_features(box_xyxy: np.ndarray, fmv: FrameMV) -> np.ndarray:
    """Pool the MV/occupancy patch under `box_xyxy` to a fixed-size vector.

    `box_xyxy` is the *already-propagated* box in fmv's frame -- features
    and correction both key off the box's current (possibly drifted)
    position, not its pre-propagation one.
    """
    x0, y0, x1, y1 = box_xyxy
    h_cells, w_cells = fmv.occupancy.shape
    cx0 = int(np.clip(x0 // BLOCK, 0, w_cells - 1))
    cx1 = int(np.clip((x1 - 1) // BLOCK + 1, 1, w_cells))
    cy0 = int(np.clip(y0 // BLOCK, 0, h_cells - 1))
    cy1 = int(np.clip((y1 - 1) // BLOCK + 1, 1, h_cells))
    cx1, cy1 = max(cx1, cx0 + 1), max(cy1, cy0 + 1)

    patch = np.dstack(
        [fmv.mv_grid[cy0:cy1, cx0:cx1], fmv.occupancy[cy0:cy1, cx0:cx1, None]]
    ).astype(np.float32)
    patch = cv2.resize(patch, (PATCH, PATCH), interpolation=cv2.INTER_AREA)

    w, h = max(x1 - x0, 1.0), max(y1 - y0, 1.0)
    size_feat = np.array([np.log(w), np.log(h)], np.float32)
    return np.concatenate([patch.reshape(-1), size_feat])


class CorrectionNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(FEAT_DIM, 64), nn.ReLU(),
            nn.Linear(64, 32), nn.ReLU(),
            nn.Linear(32, 4),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


def apply_correction(
    model: CorrectionNet, boxes_xyxy: np.ndarray, fmv: FrameMV, device: str = "cpu"
) -> np.ndarray:
    """Apply the learned residual to already-propagated boxes. Residual is
    predicted in units of the box's own (w, h) -- scale-invariant, matches
    how targets are built in build_correction_dataset.py."""
    if len(boxes_xyxy) == 0:
        return boxes_xyxy
    feats = np.stack([extract_features(b, fmv) for b in boxes_xyxy]).astype(np.float32)
    with torch.no_grad():
        resid = model(torch.from_numpy(feats).to(device)).cpu().numpy()
    w = np.clip(boxes_xyxy[:, 2] - boxes_xyxy[:, 0], 1.0, None)[:, None]
    h = np.clip(boxes_xyxy[:, 3] - boxes_xyxy[:, 1], 1.0, None)[:, None]
    scale = np.concatenate([w, h, w, h], axis=1)
    out = boxes_xyxy + resid * scale
    out[:, [0, 2]] = np.clip(out[:, [0, 2]], 0, fmv.width)
    out[:, [1, 3]] = np.clip(out[:, [1, 3]], 0, fmv.height)
    return out
