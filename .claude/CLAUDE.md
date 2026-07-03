# mvtrack

Compressed-domain video analytics: multi-object tracking that propagates
boxes via codec motion vectors instead of decoding + detecting every frame.
Full plan: `~/.claude/plans/breezy-knitting-dragon.md`.

## Environment

- Mac only (Apple M4), no CUDA/NVDEC. Decode via FFmpeg/PyAV, inference via
  PyTorch MPS (Core ML export is a possible later optimization, not done).
- `python3 -m venv .venv && .venv/bin/pip install -e .` — deps in `pyproject.toml`.
- `brew install ffmpeg` required (PyAV bundles its own libav, but the CLI is
  used directly in `scripts/prep_mot17.py`).

## Layout

```
src/mvtrack/
  extract/   # bitstream -> motion-vector grids (PyAV, see mv_extract.py)
  detect/    # YOLO wrapper (PyTorch MPS)
  track/     # MV-based box propagation
  sched/     # adaptive anchor scheduler (not yet implemented)
  bench/     # throughput/energy harness (not yet implemented)
eval/run.py  # MOT17 scoring via TrackEval (HOTA/CLEAR/Identity)
scripts/     # data prep, demos, smoke tests
```

## Known gotchas

- **PyAV MV extraction**: container-level `flags2: +export_mvs` option is
  silently ignored. Must set `stream.codec_context.flags2 |=
  av.codec.context.Flags2.export_mvs` before the first `decode()` call. Side
  data comes back as a raw buffer, not `.to_ndarray()` — parse with the
  `MV_DTYPE` structured dtype in `mv_extract.py` (matches libavutil's
  `AVMotionVector` 40-byte struct).
- **TrackEval + numpy 2.x**: TrackEval uses the removed `np.float`/`np.int`
  aliases. `eval/run.py` shims them onto the `numpy` module before import —
  don't remove that shim without repinning numpy.
- **TrackEval directory contract**: rigid. GT at
  `data/MOT17/train/<seq>-FRCNN/gt/gt.txt` + `seqinfo.ini`; results at
  `outputs/results/<pipeline>/data/<seq>-FRCNN.txt`; a seqmap file listing
  sequence names. `eval/run.py` uses `SKIP_SPLIT_FOL=True` to match this
  layout directly (no `MOT17-train/` middle folder). When debugging the
  harness, a tiny synthetic gt+result pair scoring 100% is a fast way to
  isolate config bugs from data bugs (that's how the numpy shim above was
  found — no need to wait on a real dataset download to test the plumbing).
- **motchallenge.net**: unreachable (TCP connect timeout to 131.159.19.34,
  both IPv4/IPv6) across 4+ sessions — not a local network issue. Use the
  Kaggle mirror instead: `kaggle datasets download -d wenhoujinjust/mot-17`
  (needs `~/.kaggle/kaggle.json`, a free Kaggle account + API token — Settings
  → API → Create New Token). Matches the official directory layout exactly,
  so `scripts/prep_mot17.py` works unmodified against the unzipped result
  once renamed to `data/MOT17.zip`.
- **Test clips**: `scripts/get_test_clip.py` re-encodes to baseline-profile
  H.264 (`-bf 0`, no B-frames) deliberately — the MV propagation code
  doesn't yet handle backward-referencing (B-frame) vectors correctly for
  all cases. `MOT17` sequences should be encoded the same way
  (`scripts/prep_mot17.py` already does this).
- **fps run-to-run variance**: single-run fps numbers on MPS vary
  noticeably between runs (model load/warmup, thermal state). Don't trust
  a single measurement for the accuracy/throughput Pareto curves — average
  a few runs once that ablation is built.
- **MV-grid build cost scales with resolution**: `_frame_mv`'s original
  implementation looped in Python over every MV to paint grid cells — fine
  on a low-res smoke clip, but at MOT17's 1920x1080 (~9k MVs/frame) it cost
  22ms/frame, *more than decode itself* (7.7ms/frame), which is why an
  early mv-fixed/mv-adaptive throughput win on the smoke clip didn't
  reproduce on real MOT17 data. Fixed by vectorizing to a scatter
  assignment (H.264 partition shapes always tile within one 16px-aligned
  macroblock, so every MV maps to exactly one grid cell — no need to paint
  a range). Verified bit-identical output against the old loop on 200 real
  frames before trusting it. Now 0.63ms/frame; decode dominates.
- **No decoded residual coefficients via PyAV**: FFmpeg's side-data API
  exposes motion vectors but not the actual residual/DCT energy. The
  adaptive scheduler (`sched/scheduler.py`) proxies "residual energy" with
  `1 - occupancy.mean()` (intra-coded block fraction) plus I-frame/scene-cut
  detection — intra blocks are exactly the ones the encoder couldn't
  motion-match, which correlates with real residual energy. Revisit if a
  stronger signal is needed (would require a custom FFmpeg build or ctypes
  into libavcodec internals).

## Results (MOT17 train, all 7 FRCNN sequences, YOLOv8n person-class only)

| Pipeline | HOTA | MOTA | IDF1 | mean fps |
|---|---|---|---|---|
| baseline (full decode+detect every frame) | 36.0 | 33.3 | 42.1 | 29.8 |
| mv-fixed (anchor every 5th frame) | 32.0 | 11.3 | 36.0 | 40.2 |
| mv-adaptive (~8% anchor rate) | 29.3 | 13.3 | 33.7 | 69.8 |

MV propagation is a real accuracy/throughput tradeoff, not a free win —
MOTA drops hard (33→11-13%) because propagation drift between anchors hurts
more on MOT17's crowded/static-camera scenes than it did on the low-res
smoke-test clip. This is the honest weeks 5-7 result; report it as-is rather
than only the throughput number.

## Working conventions

- This is exploratory systems/research code, not a product — favor fast
  iteration and real measurements over up-front test suites. Do add a quick
  correctness check (a synthetic fixture, a known-answer case) whenever a
  new numerical/data-format component is built; that's what caught the
  numpy shim issue above before it silently produced wrong scores.
- Every pipeline variant must go through `eval/run.py` so accuracy and fps
  numbers are directly comparable across the baseline and later MV-based
  variants — don't build one-off benchmarking scripts that skip it.
