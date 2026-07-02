"""Motion-vector extraction demo: overlay MV arrows on decoded frames.

Writes outputs/mv_overlay.mp4 and prints per-frame-type stats. Acceptance:
P-frames show a dense, motion-coherent arrow field; I-frames show none.
"""

import pathlib
import sys
from collections import Counter

import cv2
import numpy as np

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1] / "src"))
from mvtrack.extract import BLOCK, iter_frames_with_mvs  # noqa: E402

ROOT = pathlib.Path(__file__).resolve().parents[1]
CLIP = ROOT / "data" / "people_baseline.mp4"
OUT = ROOT / "outputs" / "mv_overlay.mp4"


def main() -> None:
    OUT.parent.mkdir(exist_ok=True)
    writer = None
    types = Counter()
    occ_by_type = Counter()
    frames = 0

    for fmv, frame in iter_frames_with_mvs(str(CLIP)):
        img = frame.to_ndarray(format="bgr24")
        if writer is None:
            writer = cv2.VideoWriter(
                str(OUT), cv2.VideoWriter_fourcc(*"mp4v"), 30, (fmv.width, fmv.height)
            )
        ys, xs = np.nonzero(fmv.occupancy)
        for y, x in zip(ys, xs):
            cx, cy = x * BLOCK + BLOCK // 2, y * BLOCK + BLOCK // 2
            dx, dy = fmv.mv_grid[y, x]
            cv2.arrowedLine(
                img,
                (int(cx - dx), int(cy - dy)),
                (cx, cy),
                (0, 255, 0),
                1,
                tipLength=0.3,
            )
        cv2.putText(
            img, fmv.pict_type, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2
        )
        writer.write(img)
        types[fmv.pict_type] += 1
        occ_by_type[fmv.pict_type] += float(fmv.occupancy.mean())
        frames += 1

    writer.release()
    print(f"frames: {frames}, pict types: {dict(types)}")
    for t in types:
        print(f"  {t}-frames mean MV block occupancy: {occ_by_type[t] / types[t]:.1%}")
    print(f"wrote {OUT}")


if __name__ == "__main__":
    main()
