"""Run N concurrent instances of an eval.run pipeline and measure aggregate
throughput. Streams run as separate processes, not threads -- PyTorch/YOLO
inference holds the GIL enough of the time that threads wouldn't show real
concurrency gains, and MPS device contexts are safer one-per-process
(spawn, not fork).
"""

import multiprocessing as mp
import pathlib
import sys
import time

ROOT = pathlib.Path(__file__).resolve().parents[3]


def _run_one(args: tuple[str, str, int, int]) -> tuple[int, float]:
    pipeline_name, video_path, anchor_interval, worker_idx = args
    sys.path.insert(0, str(ROOT / "src"))
    sys.path.insert(0, str(ROOT / "eval"))
    import run as eval_run  # eval/run.py

    out = ROOT / "outputs" / "bench_tmp" / f"{pipeline_name}_{worker_idx}.txt"
    return eval_run.PIPELINES[pipeline_name](
        pathlib.Path(video_path),
        out,
        anchor_interval=anchor_interval,
        correction_checkpoint="correction_net.pt",
    )


def run_multistream(
    pipeline_name: str, video: pathlib.Path, n_streams: int, anchor_interval: int = 5
) -> tuple[float, float, float]:
    """Returns (aggregate_fps, per_stream_fps, wall_seconds)."""
    args = [(pipeline_name, str(video), anchor_interval, i) for i in range(n_streams)]
    t0 = time.perf_counter()
    with mp.get_context("spawn").Pool(n_streams) as pool:
        results = pool.map(_run_one, args)
    wall = time.perf_counter() - t0
    total_frames = sum(f for f, _ in results)
    aggregate_fps = total_frames / wall
    return aggregate_fps, aggregate_fps / n_streams, wall
