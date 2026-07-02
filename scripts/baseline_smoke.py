"""Full-decode baseline: YOLOv8n + ByteTrack (ultralytics built-in) on the
test clip. This is the accuracy ceiling / throughput floor for everything.

Prints fps and unique track count; writes outputs/baseline_tracked.mp4.
"""

import pathlib
import sys
import time

import cv2

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1] / "src"))
from mvtrack.detect import pick_device  # noqa: E402
from ultralytics import YOLO  # noqa: E402

ROOT = pathlib.Path(__file__).resolve().parents[1]
CLIP = ROOT / "data" / "people_baseline.mp4"
OUT = ROOT / "outputs" / "baseline_tracked.mp4"


def main() -> None:
    OUT.parent.mkdir(exist_ok=True)
    device = pick_device()
    model = YOLO("yolov8n.pt")
    writer = None
    track_ids = set()
    frames = 0
    t0 = time.perf_counter()

    for res in model.track(
        source=str(CLIP), device=device, tracker="bytetrack.yaml",
        stream=True, verbose=False, conf=0.25,
    ):
        img = res.plot()
        if writer is None:
            h, w = img.shape[:2]
            writer = cv2.VideoWriter(
                str(OUT), cv2.VideoWriter_fourcc(*"mp4v"), 30, (w, h)
            )
        writer.write(img)
        if res.boxes.id is not None:
            track_ids.update(res.boxes.id.int().tolist())
        frames += 1

    dt = time.perf_counter() - t0
    writer.release()
    print(f"device: {device}")
    print(f"frames: {frames}, unique tracks: {len(track_ids)}")
    print(f"wall: {dt:.1f}s -> {frames / dt:.1f} fps (decode+detect+track)")
    print(f"wrote {OUT}")


if __name__ == "__main__":
    main()
