# mvtrack — compressed-domain video analytics on Apple Silicon

Multi-object detection + tracking that avoids fully decoding most video frames.
Codec side-information (H.264/HEVC motion vectors, block types, residual energy)
propagates track boxes between sparse **anchor frames**, which are the only frames
that get full decode + neural-network inference. An adaptive scheduler decides
when to fire an anchor based on residual energy and track drift.

Headline metric: tracking accuracy (HOTA/MOTA/IDF1 on MOT17, UA-DETRAC) vs.
throughput (streams per chip) vs. energy (watts) on Apple Silicon.

## Layout

```
src/mvtrack/
  extract/   # bitstream -> motion-vector / residual tensors (PyAV)
  detect/    # YOLO wrappers (PyTorch MPS / Core ML)
  track/     # tracker + MV box propagation
  sched/     # adaptive anchor scheduler
  bench/     # throughput, energy, multi-stream harness
eval/        # MOT17/UA-DETRAC adapters, HOTA via TrackEval
scripts/     # demos, data prep
```

## Setup

```bash
brew install ffmpeg
python3 -m venv .venv && source .venv/bin/activate
pip install -e .
```

## Quick start

```bash
python scripts/get_test_clip.py           # fetch + re-encode sample clip
python scripts/mv_demo.py                 # motion-vector extraction demo
python scripts/baseline_smoke.py          # full-decode detect+track baseline
```
