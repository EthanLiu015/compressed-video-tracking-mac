# mvtrack — compressed-domain video analytics on Apple Silicon

Multi-object detection + tracking that avoids fully decoding most video frames.
Codec side-information (H.264 motion vectors, block partitions) propagates
track boxes between sparse **anchor frames**, which are the only frames that
get full decode + neural-network inference. An adaptive scheduler decides
when to fire an anchor based on a residual-energy proxy and scene cuts.

Headline result: tracking accuracy (HOTA/MOTA/IDF1 on MOT17, via the official
TrackEval implementation) vs. throughput (fps, concurrent streams per chip)
on Apple Silicon (M4, PyTorch MPS, no CUDA/NVDEC).

## Results

All numbers are on real MOT17 train ground truth (7 FRCNN sequences),
YOLOv8s person-class detector (the project default — a ceiling check
confirmed YOLOv8n was leaving real accuracy on the table across every
pipeline) plus a fixed three-stage ByteTrack-style re-association in
`MVTracker` (see below). Full detail and honest negative results in
`findings.md`; per-session narrative in `.claude/CLAUDE.md`.

| Pipeline | HOTA | MOTA | IDF1 | mean fps |
|---|---|---|---|---|
| baseline (full decode + detect every frame) | 40.5 | 38.9 | 48.6 | 20.8 |
| mv-fixed (anchor every 5th frame) | 35.7 | 26.4 | 42.5 | 38.5 |
| mv-adaptive (residual-energy-proxy scheduler, ~8% anchor rate) | 32.5 | 21.8 | 39.1 | 62.9 |
| mv-learned (mv-fixed + learned box correction) | 34.6 | 25.5 | 41.7 | 33.7 |

![HOTA/MOTA vs throughput](results/plots/pipeline_pareto.png)

MV propagation still trades real accuracy for real throughput — not a
free win — but about half of that gap turned out to be a fixable bug, not
an inherent cost. The original single-pass IoU re-association in
`MVTracker.step_anchor` spawned a brand-new track ID for any detection it
couldn't match, and an anchor-interval ablation showed this made *more*
anchors actively *hurt* MOTA (findings.md #7) — the opposite of intuition.
Rewriting it as a three-stage match (high-confidence detections vs. all
tracks, then low-confidence detections recovering tracks stage 1 missed,
then a loose-IoU "grace period" before spawning a new ID) eliminated the
inversion and roughly halved mv-fixed's MOTA gap to baseline (findings.md
#10). mv-adaptive still roughly doubles baseline's max concurrent-stream
capacity on the same chip at a fixed 25fps/stream quality bar (multi-stream
and ablation numbers below were measured on YOLOv8n with the pre-fix
tracker — not rerun since a slower detector only lowers these ceilings,
it doesn't change the baseline-vs-mv-* comparison):

| Pipeline | max concurrent streams @ >=25fps/stream |
|---|---|
| baseline | 4 |
| mv-fixed | 8 |
| mv-adaptive | 8 |

![Multi-stream scaling](results/plots/multistream_scaling.png)

The anchor-interval ablation below (mv-fixed at intervals 2/3/5/8/10) is
the **post-fix** curve: MOTA is now roughly flat across the whole range
instead of rising monotonically, and HOTA/IDF1 decrease with larger
intervals as intuition would predict — see `findings.md` #7 and #10 for
the full before/after story.

![Anchor interval ablation](results/plots/ablation_anchor_interval.png)

## Layout

```
src/mvtrack/
  extract/   # bitstream -> motion-vector grids (PyAV)
  detect/    # YOLO wrapper (PyTorch MPS)
  track/     # MVTracker (IoU/Hungarian + MV propagation), CorrectionNet
  sched/     # FixedInterval and Adaptive anchor schedulers
  bench/     # multi-stream throughput harness
  mot_gt.py  # shared MOT17 ground-truth loader
eval/run.py  # MOT17 scoring harness (HOTA/CLEAR/Identity via TrackEval)
scripts/     # data prep, dataset builders, training, benchmarking, plots
results/     # committed CSVs + plots (reproducible, unlike gitignored outputs/)
```

## Setup

```bash
brew install ffmpeg
python3 -m venv .venv && source .venv/bin/activate
pip install -e .
```

MOT17 requires a Kaggle account + API token (`~/.kaggle/kaggle.json` —
Settings -> API -> Create New Token on kaggle.com), since the official
`motchallenge.net` host has been unreachable across many sessions:

```bash
kaggle datasets download -d wenhoujinjust/mot-17 -p data/
mv data/mot-17.zip data/MOT17.zip
python scripts/prep_mot17.py              # unzip + re-encode to H.264
```

## Quick start

```bash
python scripts/get_test_clip.py           # fetch + re-encode a small sample clip
python scripts/mv_demo.py                 # motion-vector extraction demo
python scripts/baseline_smoke.py          # full-decode detect+track baseline
```

## Evaluation

```bash
python eval/run.py --pipeline baseline                 # --weights yolov8s.pt is the default
python eval/run.py --pipeline mv-fixed --anchor-interval 5
python eval/run.py --pipeline mv-adaptive
python eval/run.py --pipeline mv-learned --correction-checkpoint correction_net_v2.pt
python eval/run.py --pipeline baseline --weights yolov8n.pt   # compare against the old default
```

## Learned box correction (optional, stretch goal)

```bash
python scripts/build_correction_dataset.py     # single-step GT propagation-error pairs
python scripts/train_correction.py             # trains outputs/correction_net.pt
python scripts/build_rollout_dataset.py        # on-policy DAgger targets from that checkpoint
python scripts/train_correction.py --datasets outputs/correction_dataset.npz \
    outputs/correction_dataset_rollout.npz --out outputs/correction_net_v2.pt
```

## Benchmarking and plots

```bash
python scripts/bench_multistream.py --pipeline mv-adaptive \
    --video data/people_baseline.mp4 --streams 1 2 3 4 6 8
python scripts/make_plots.py              # regenerates results/plots/*.png from results/*.csv
```
