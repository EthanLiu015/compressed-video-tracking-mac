"""MOT17 evaluation harness.

Runs a pipeline over re-encoded MOT17 train videos, writes MOTChallenge
result files, and scores HOTA/MOTA/IDF1 with TrackEval (the official
MOTChallenge metrics implementation).

Usage:
    python eval/run.py --pipeline baseline [--seqs MOT17-02 ...]

Requires data/MOT17/train/<seq>-FRCNN/{gt,seqinfo.ini} (scripts/prep_mot17.py)
and data/MOT17/videos/<seq>-FRCNN.mp4 (also from prep_mot17.py).
"""

import argparse
import pathlib
import sys
import time

import numpy as np

# TrackEval predates numpy 2.0's removal of these deprecated aliases.
for _name, _builtin in (("float", float), ("int", int), ("bool", bool)):
    if not hasattr(np, _name):
        setattr(np, _name, _builtin)

import trackeval

ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

MOT = ROOT / "data" / "MOT17"
RESULTS = ROOT / "outputs" / "results"

PERSON_CLS = 0  # COCO 'person' in YOLO


def run_baseline(
    video: pathlib.Path, out_txt: pathlib.Path, weights: str = "yolov8s.pt", **_kwargs
) -> tuple[int, float]:
    """Full decode + YOLOv8 + ByteTrack every frame. Returns (frames, seconds)."""
    from ultralytics import YOLO

    from mvtrack.detect import pick_device

    model = YOLO(weights)
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
    out_txt.parent.mkdir(parents=True, exist_ok=True)
    out_txt.write_text("\n".join(rows) + "\n")
    return frames, dt


def run_mv_fixed(
    video: pathlib.Path,
    out_txt: pathlib.Path,
    anchor_interval: int = 5,
    weights: str = "yolov8s.pt",
    use_reid: bool = False,
    tracker_kwargs: dict | None = None,
    **_kwargs,
) -> tuple[int, float]:
    """Detector fires every `anchor_interval` frames; MV propagation fills the
    rest. Frame 1 is always an anchor. Returns (frames, seconds)."""
    from mvtrack.detect import Detector, pick_device
    from mvtrack.extract import iter_frames_with_mvs
    from mvtrack.track import MVTracker

    detector = Detector(weights)
    # max_age must cover the longest possible gap between anchors (here,
    # anchor_interval itself) or a track that simply hasn't hit its next
    # anchor yet gets pruned as a false "ghost" -- see findings.md #15.
    kwargs = {"max_age": anchor_interval, **(tracker_kwargs or {})}
    tracker = MVTracker(use_appearance=use_reid, **kwargs)
    reid = None
    if use_reid:
        from mvtrack.track.reid import ReIDEmbedder
        reid = ReIDEmbedder(device=pick_device())
    rows = []
    frames = 0
    t0 = time.perf_counter()
    for fmv, frame in iter_frames_with_mvs(str(video)):
        frames += 1
        if frames % anchor_interval == 1:
            img = frame.to_ndarray(format="bgr24")
            boxes, scores, cls_ids = detector(img)
            keep = cls_ids == PERSON_CLS
            embs = reid(img, boxes[keep]) if reid is not None else None
            tracks = tracker.step_anchor(boxes[keep], scores[keep], embeddings=embs)
        else:
            tracks = tracker.step_propagate(fmv)
        for tr in tracks:
            x0, y0, x1, y1 = tr.box
            rows.append(
                f"{frames},{tr.id},{x0:.2f},{y0:.2f},{x1 - x0:.2f},{y1 - y0:.2f},{tr.score:.3f},-1,-1,-1"
            )
    dt = time.perf_counter() - t0
    out_txt.parent.mkdir(parents=True, exist_ok=True)
    out_txt.write_text("\n".join(rows) + "\n")
    return frames, dt


def run_mv_adaptive(
    video: pathlib.Path,
    out_txt: pathlib.Path,
    weights: str = "yolov8s.pt",
    scheduler_kwargs: dict | None = None,
    use_reid: bool = False,
    tracker_kwargs: dict | None = None,
    **_kwargs,
) -> tuple[int, float]:
    """Same as run_mv_fixed but anchor timing comes from Adaptive (residual-
    energy-proxy) instead of a fixed interval. Returns (frames, seconds)."""
    from mvtrack.detect import Detector, pick_device
    from mvtrack.extract import iter_frames_with_mvs
    from mvtrack.sched import Adaptive
    from mvtrack.track import MVTracker

    detector = Detector(weights)
    scheduler = Adaptive(**(scheduler_kwargs or {}))
    # max_age must cover the scheduler's longest possible anchor gap
    # (max_interval) or a track gets pruned as a false "ghost" before its
    # next anchor ever arrives -- see findings.md #15. Using max_age=5
    # (tuned for mv-fixed's fixed 5-frame gap) here instead collapsed
    # mv-adaptive's HOTA from 31.7 to 9.5 by killing legitimate tracks early.
    kwargs = {"max_age": scheduler.max_interval, **(tracker_kwargs or {})}
    tracker = MVTracker(use_appearance=use_reid, **kwargs)
    reid = None
    if use_reid:
        from mvtrack.track.reid import ReIDEmbedder
        reid = ReIDEmbedder(device=pick_device())
    rows = []
    frames = 0
    anchors = 0
    t0 = time.perf_counter()
    for fmv, frame in iter_frames_with_mvs(str(video)):
        frames += 1
        if scheduler.should_anchor(fmv):
            anchors += 1
            img = frame.to_ndarray(format="bgr24")
            boxes, scores, cls_ids = detector(img)
            keep = cls_ids == PERSON_CLS
            embs = reid(img, boxes[keep]) if reid is not None else None
            tracks = tracker.step_anchor(boxes[keep], scores[keep], embeddings=embs)
        else:
            tracks = tracker.step_propagate(fmv)
        for tr in tracks:
            x0, y0, x1, y1 = tr.box
            rows.append(
                f"{frames},{tr.id},{x0:.2f},{y0:.2f},{x1 - x0:.2f},{y1 - y0:.2f},{tr.score:.3f},-1,-1,-1"
            )
    dt = time.perf_counter() - t0
    out_txt.parent.mkdir(parents=True, exist_ok=True)
    out_txt.write_text("\n".join(rows) + "\n")
    print(f"  anchor rate: {anchors}/{frames} ({100 * anchors / frames:.1f}%)")
    return frames, dt


def run_mv_learned(
    video: pathlib.Path,
    out_txt: pathlib.Path,
    anchor_interval: int = 5,
    correction_checkpoint: str = "correction_net.pt",
    weights: str = "yolov8s.pt",
    **_kwargs,
) -> tuple[int, float]:
    """Same as run_mv_fixed but CorrectionNet adjusts each propagated box.
    Requires outputs/<correction_checkpoint> (scripts/train_correction.py).
    Returns (frames, seconds)."""
    import torch

    from mvtrack.detect import Detector, pick_device
    from mvtrack.extract import iter_frames_with_mvs
    from mvtrack.track import MVTracker
    from mvtrack.track.correct import CorrectionNet, apply_correction

    device = pick_device()
    net = CorrectionNet().to(device)
    net.load_state_dict(
        torch.load(ROOT / "outputs" / correction_checkpoint, map_location=device)
    )
    net.eval()

    detector = Detector(weights)
    tracker = MVTracker()
    rows = []
    frames = 0
    t0 = time.perf_counter()
    for fmv, frame in iter_frames_with_mvs(str(video)):
        frames += 1
        if frames % anchor_interval == 1:
            img = frame.to_ndarray(format="bgr24")
            boxes, scores, cls_ids = detector(img)
            keep = cls_ids == PERSON_CLS
            tracks = tracker.step_anchor(boxes[keep], scores[keep])
        else:
            tracks = tracker.step_propagate(fmv)
            if tracks:
                ids = [t.id for t in tracks]
                boxes = np.stack([t.box for t in tracks])
                corrected = apply_correction(net, boxes, fmv, device=device)
                tracks = tracker.correct_boxes(dict(zip(ids, corrected)))
        for tr in tracks:
            x0, y0, x1, y1 = tr.box
            rows.append(
                f"{frames},{tr.id},{x0:.2f},{y0:.2f},{x1 - x0:.2f},{y1 - y0:.2f},{tr.score:.3f},-1,-1,-1"
            )
    dt = time.perf_counter() - t0
    out_txt.parent.mkdir(parents=True, exist_ok=True)
    out_txt.write_text("\n".join(rows) + "\n")
    return frames, dt


def run_mv_replay(
    video: pathlib.Path,
    out_txt: pathlib.Path,
    anchor_frames: set,
    weights: str = "yolov8s.pt",
    use_reid: bool = False,
    tracker_kwargs: dict | None = None,
    **_kwargs,
) -> tuple[int, float]:
    """Anchor timing comes from a precomputed frame-index set (built by
    `mvtrack.sched.global_replay`'s independent/global allocation
    simulators) instead of a live interval/scheduler decision -- lets the
    global-scheduler experiment's two allocation strategies be scored
    through this exact same TrackEval-backed path with no new scoring
    plumbing. `anchor_frames` uses `FrameMV.index` (0-based), matching how
    `extract_urgency_trace` builds its trace.

    `anchor_frames` has no default and isn't driven by `main()`'s generic
    `--pipeline` CLI loop (it's a per-video precomputed set, not a CLI
    scalar) -- call this directly from a driver script, e.g.
    `scripts/run_global_budget_experiment.py`. Registered in `PIPELINES`
    anyway so `evaluate()`'s TrackEval scoring path treats it like any
    other pipeline.
    """
    from mvtrack.detect import Detector, pick_device
    from mvtrack.extract import iter_frames_with_mvs
    from mvtrack.track import MVTracker

    detector = Detector(weights)
    # max_age must cover the largest real gap this SPECIFIC anchor_frames
    # set actually has, or a track that hasn't hit its next anchor yet gets
    # pruned as a false ghost -- findings.md #15, same invariant
    # run_mv_fixed/run_mv_adaptive rely on, derived here from the concrete
    # anchor set rather than a scheduler parameter.
    sorted_anchors = sorted(anchor_frames)
    gaps = [b - a for a, b in zip(sorted_anchors, sorted_anchors[1:])]
    default_max_age = max(gaps) if gaps else 1
    kwargs = {"max_age": default_max_age, **(tracker_kwargs or {})}
    tracker = MVTracker(use_appearance=use_reid, **kwargs)
    reid = None
    if use_reid:
        from mvtrack.track.reid import ReIDEmbedder
        reid = ReIDEmbedder(device=pick_device())
    rows = []
    frames = 0
    t0 = time.perf_counter()
    for fmv, frame in iter_frames_with_mvs(str(video)):
        frames += 1
        if fmv.index in anchor_frames:
            img = frame.to_ndarray(format="bgr24")
            boxes, scores, cls_ids = detector(img)
            keep = cls_ids == PERSON_CLS
            embs = reid(img, boxes[keep]) if reid is not None else None
            tracks = tracker.step_anchor(boxes[keep], scores[keep], embeddings=embs)
        else:
            tracks = tracker.step_propagate(fmv)
        for tr in tracks:
            x0, y0, x1, y1 = tr.box
            rows.append(
                f"{frames},{tr.id},{x0:.2f},{y0:.2f},{x1 - x0:.2f},{y1 - y0:.2f},{tr.score:.3f},-1,-1,-1"
            )
    dt = time.perf_counter() - t0
    out_txt.parent.mkdir(parents=True, exist_ok=True)
    out_txt.write_text("\n".join(rows) + "\n")
    return frames, dt


PIPELINES = {
    "baseline": run_baseline,
    "mv-fixed": run_mv_fixed,
    "mv-adaptive": run_mv_adaptive,
    "mv-learned": run_mv_learned,
    "mv-global-replay": run_mv_replay,
}


def write_seqmap(seq_names: list[str], path: pathlib.Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("name\n" + "\n".join(seq_names) + "\n")


def evaluate(pipeline: str, seq_names: list[str]) -> None:
    """Score result files against GT via TrackEval; prints HOTA/CLEAR/Identity."""
    seqmap_path = RESULTS / "seqmap.txt"
    write_seqmap(seq_names, seqmap_path)

    dataset_config = {
        "GT_FOLDER": str(MOT / "train"),
        "TRACKERS_FOLDER": str(RESULTS),
        "SEQMAP_FILE": str(seqmap_path),
        "SKIP_SPLIT_FOL": True,  # our layout has no BENCHMARK-SPLIT subfolder
        "TRACKER_SUB_FOLDER": "data",
        "TRACKERS_TO_EVAL": [pipeline],
        "CLASSES_TO_EVAL": ["pedestrian"],
        "PRINT_CONFIG": False,
    }
    eval_config = trackeval.Evaluator.get_default_eval_config()
    eval_config.update({"PRINT_CONFIG": False, "TIME_PROGRESS": False})

    evaluator = trackeval.Evaluator(eval_config)
    dataset = trackeval.datasets.MotChallenge2DBox(dataset_config)
    metrics = [
        trackeval.metrics.HOTA(),
        trackeval.metrics.CLEAR({"PRINT_CONFIG": False}),
        trackeval.metrics.Identity({"PRINT_CONFIG": False}),
    ]
    evaluator.evaluate([dataset], metrics)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--pipeline", choices=PIPELINES, default="baseline")
    ap.add_argument("--seqs", nargs="*", help="e.g. MOT17-02 (default: all)")
    ap.add_argument("--anchor-interval", type=int, default=5, help="mv-fixed only")
    ap.add_argument(
        "--correction-checkpoint", default="correction_net.pt", help="mv-learned only"
    )
    ap.add_argument("--weights", default="yolov8s.pt", help="YOLO weights, all pipelines")
    ap.add_argument(
        "--use-reid", action="store_true",
        help="blend a learned appearance embedding into association (mv-fixed/mv-adaptive)",
    )
    ap.add_argument(
        "--max-age", type=int, default=None,
        help="MVTracker max_age override (frames unmatched before a track is pruned)",
    )
    args = ap.parse_args()
    tracker_kwargs = {"max_age": args.max_age} if args.max_age is not None else None

    videos = sorted((MOT / "videos").glob("MOT17-*-FRCNN.mp4"))
    if args.seqs:
        videos = [v for v in videos if any(v.stem.startswith(s) for s in args.seqs)]
    if not videos:
        sys.exit("no sequence videos found — run scripts/prep_mot17.py first")

    out_dir = RESULTS / args.pipeline / "data"
    seq_names, fps_all = [], []
    for video in videos:
        seq = video.stem  # e.g. MOT17-02-FRCNN — must match GT folder name
        res_txt = out_dir / f"{seq}.txt"
        frames, dt = PIPELINES[args.pipeline](
            video,
            res_txt,
            anchor_interval=args.anchor_interval,
            correction_checkpoint=args.correction_checkpoint,
            weights=args.weights,
            use_reid=args.use_reid,
            tracker_kwargs=tracker_kwargs,
        )
        fps = frames / dt
        fps_all.append(fps)
        seq_names.append(seq)
        print(f"{seq}: {frames} frames, {fps:.1f} fps")

    evaluate(args.pipeline, seq_names)
    print(f"mean fps: {np.mean(fps_all):.1f} ({args.pipeline})")


if __name__ == "__main__":
    main()
