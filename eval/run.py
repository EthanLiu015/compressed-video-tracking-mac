"""MOT17 evaluation harness.

Runs a pipeline over re-encoded MOT17 train videos, writes MOTChallenge
result files, and scores MOTA/IDF1 with motmetrics. (HOTA via TrackEval is
planned; TrackEval must be installed from GitHub.)

Usage:
    python eval/run.py --pipeline baseline [--seqs MOT17-02 ...]
"""

import argparse
import pathlib
import sys
import time

import motmetrics as mm
import numpy as np

ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

MOT = ROOT / "data" / "MOT17"
RESULTS = ROOT / "outputs" / "results"

PERSON_CLS = 0  # COCO 'person' in YOLO


def run_baseline(video: pathlib.Path, out_txt: pathlib.Path) -> tuple[int, float]:
    """Full decode + YOLOv8 + ByteTrack every frame. Returns (frames, seconds)."""
    from ultralytics import YOLO

    from mvtrack.detect import pick_device

    model = YOLO("yolov8n.pt")
    rows = []
    frames = 0
    t0 = time.perf_counter()
    for res in model.track(
        source=str(video), device=pick_device(), tracker="bytetrack.yaml",
        stream=True, verbose=False, conf=0.25, classes=[PERSON_CLS],
    ):
        frames += 1
        if res.boxes.id is None:
            continue
        ids = res.boxes.id.int().tolist()
        xyxy = res.boxes.xyxy.cpu().numpy()
        confs = res.boxes.conf.cpu().numpy()
        for tid, (x0, y0, x1, y1), c in zip(ids, xyxy, confs):
            rows.append(
                f"{frames},{tid},{x0:.2f},{y0:.2f},{x1 - x0:.2f},{y1 - y0:.2f},{c:.3f},-1,-1,-1"
            )
    dt = time.perf_counter() - t0
    out_txt.write_text("\n".join(rows) + "\n")
    return frames, dt


PIPELINES = {"baseline": run_baseline}


def score(seq_name: str, res_txt: pathlib.Path) -> mm.MOTAccumulator:
    gt_path = MOT / "train" / f"{seq_name}-FRCNN" / "gt" / "gt.txt"
    gt = mm.io.loadtxt(gt_path, fmt="mot16")  # filters to valid pedestrians
    ts = mm.io.loadtxt(res_txt, fmt="mot15-2D")
    return mm.utils.compare_to_groundtruth(gt, ts, "iou", distth=0.5)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--pipeline", choices=PIPELINES, default="baseline")
    ap.add_argument("--seqs", nargs="*", help="e.g. MOT17-02 (default: all)")
    args = ap.parse_args()

    videos = sorted((MOT / "videos").glob("MOT17-*.mp4"))
    if args.seqs:
        videos = [v for v in videos if any(v.stem.startswith(s) for s in args.seqs)]
    if not videos:
        sys.exit("no sequence videos found — run scripts/prep_mot17.py first")

    out_dir = RESULTS / args.pipeline
    out_dir.mkdir(parents=True, exist_ok=True)

    accs, names, fps_all = [], [], []
    for video in videos:
        seq = video.stem.replace("-FRCNN", "")
        res_txt = out_dir / f"{seq}.txt"
        frames, dt = PIPELINES[args.pipeline](video, res_txt)
        fps = frames / dt
        fps_all.append(fps)
        accs.append(score(seq, res_txt))
        names.append(seq)
        print(f"{seq}: {frames} frames, {fps:.1f} fps")

    mh = mm.metrics.create()
    summary = mh.compute_many(
        accs, names=names, generate_overall=True,
        metrics=["mota", "idf1", "num_switches", "num_frames"],
    )
    print(mm.io.render_summary(summary, formatters=mh.formatters,
                               namemap=mm.io.motchallenge_metric_names))
    print(f"mean fps: {np.mean(fps_all):.1f} ({args.pipeline})")


if __name__ == "__main__":
    main()
