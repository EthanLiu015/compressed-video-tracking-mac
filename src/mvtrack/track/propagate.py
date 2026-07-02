"""Propagate track boxes across frames using motion-vector grids.

Core compressed-domain primitive: instead of re-detecting, shift each box by
the median motion of the MV-carrying blocks it covers.
"""

import numpy as np

from mvtrack.extract import BLOCK, FrameMV


def propagate_boxes(boxes_xyxy: np.ndarray, fmv: FrameMV) -> np.ndarray:
    """Shift boxes [N,4] xyxy by the median MV inside each box.

    Boxes covering no MV-carrying blocks (e.g. on I-frames or static
    regions) are returned unchanged.
    """
    if len(boxes_xyxy) == 0:
        return boxes_xyxy
    out = boxes_xyxy.astype(np.float32).copy()
    h_cells, w_cells = fmv.occupancy.shape
    for n, (x0, y0, x1, y1) in enumerate(boxes_xyxy):
        cx0 = int(np.clip(x0 // BLOCK, 0, w_cells - 1))
        cx1 = int(np.clip((x1 - 1) // BLOCK + 1, 1, w_cells))
        cy0 = int(np.clip(y0 // BLOCK, 0, h_cells - 1))
        cy1 = int(np.clip((y1 - 1) // BLOCK + 1, 1, h_cells))
        occ = fmv.occupancy[cy0:cy1, cx0:cx1].astype(bool)
        if not occ.any():
            continue
        mv = fmv.mv_grid[cy0:cy1, cx0:cx1][occ]  # (K, 2)
        dx, dy = np.median(mv[:, 0]), np.median(mv[:, 1])
        out[n, [0, 2]] += dx
        out[n, [1, 3]] += dy
    out[:, [0, 2]] = np.clip(out[:, [0, 2]], 0, fmv.width)
    out[:, [1, 3]] = np.clip(out[:, [1, 3]], 0, fmv.height)
    return out
