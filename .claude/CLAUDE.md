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

## Results (MOT17 train, all 7 FRCNN sequences, person-class only)

**Current default detector: YOLOv8s** (`weights="yolov8s.pt"`, default in
`Detector` and `eval/run.py --weights`). Confirmed via a direct ceiling
check that detector quality was a real, separate lever from the MV
propagation approach — swapping YOLOv8n -> YOLOv8s raised every pipeline's
accuracy with only a modest fps cost (much smaller for the mv-* pipelines
than for baseline, since they only run the detector on a fraction of
frames):

| Pipeline | HOTA | MOTA | IDF1 | mean fps |
|---|---|---|---|---|
| baseline (full decode+detect every frame) | 40.5 | 38.9 | 48.6 | 20.8 |
| mv-fixed (anchor every 5th frame) | 34.7 | 13.8 | 39.9 | 37.8 |
| mv-adaptive (~8% anchor rate) | 32.2 | 16.3 | 37.5 | 63.6 |
| mv-learned v2 (mv-fixed + CorrectionNet, DAgger rollout training) | 33.2 | 11.2 | 38.4 | 32.6 |

<details>
<summary>Superseded YOLOv8n numbers (kept for reference — see git history for the full swap commit)</summary>

| Pipeline | HOTA | MOTA | IDF1 | mean fps |
|---|---|---|---|---|
| baseline | 36.0 | 33.3 | 42.1 | 29.8 |
| mv-fixed | 32.0 | 11.3 | 36.0 | 40.2 |
| mv-adaptive | 29.3 | 13.3 | 33.7 | 69.8 |
| mv-learned v1 (single-step training) | 31.1 | 7.0 | 34.8 | 34.5 |
| mv-learned v2 (DAgger rollout training) | 30.5 | 9.1 | 35.0 | 40.0 |

</details>

MV propagation is a real accuracy/throughput tradeoff, not a free win —
MOTA drops hard (baseline ~39% vs mv-fixed/mv-adaptive ~14-16%) because
propagation drift between anchors hurts more on MOT17's crowded,
often-static-camera scenes than it did on the low-res smoke-test clip
used early on. This is the honest core result; report it as-is rather
than only the throughput number. Note baseline itself (HOTA 40.5) still
falls short of a well-tuned ByteTrack-on-MOT17 ballpark (~60) — YOLOv8s is
better than YOLOv8n but still a small/fast model; a larger detector
(yolov8m/l) or confidence/NMS tuning would likely raise the ceiling
further at additional throughput cost, not attempted here.

**CorrectionNet (weeks 8-10 stretch goal) made things worse, not better,
even after fixing the diagnosed train/inference mismatch.** v1 (single-step
training: GT box at t -> GT box at t+1) beat a zero-residual baseline by
~19% MSE in isolation but scored worse than plain mv-fixed on every
tracking metric. Diagnosis: training only saw one propagation step from a
perfect GT box, while inference chains the net's own corrections across
multiple already-drifted frames between anchors (exposure-bias-like
failure mode). Fix: `scripts/build_rollout_dataset.py` re-walks real anchor
windows using the trained checkpoint itself to generate on-policy targets
(DAgger-style — the net's own rollout trajectory, not GT-anchored steps),
then `train_correction.py --datasets <original> <rollout>` retrains from
scratch on the aggregate. Needed a robust loss (`SmoothL1Loss` + grad
clipping) since rollout targets are heavy-tailed — plain MSE training
destabilized (val loss spiking 10x between epochs) because a chained
correction occasionally sends a box far off in crowded/occluded cases.

Result of the fix: v2 val MSE improved (26% below zero-residual baseline,
vs. 19% for v1) and MOTA/IDF1 both improved over v1 in the full pipeline
(MOTA 7.0->9.1, IDF1 34.8->35.0) — the mismatch diagnosis was correct and
the fix measurably helped. But it didn't flip the verdict: v2 is still
worse than plain mv-fixed (no correction at all) on HOTA and MOTA. Current
read: a 50-param MLP over pooled MV/occupancy features (no appearance
signal) may just be under-powered for this correction task — its per-frame
noise costs more than its average drift-correction saves, so the tracker's
own periodic re-anchoring is currently a better use of that compute than
learned correction. Not pursuing further for now; mv-fixed/mv-adaptive
remain the pipelines to build on. Would revisit with appearance features
or a per-track confidence gate (only apply correction when the net is
confident) if returning to this.

## Multi-stream throughput (weeks 11-12, local smoke clip, max streams @ >=25fps/stream)

**Measured on YOLOv8n** (predates the detector swap above) — not rerun
with YOLOv8s since a bigger/slower detector would only lower these
ceilings, not invalidate the relative baseline-vs-mv-* comparison. Same
caveat applies to the anchor-interval ablation in `findings.md` #7.

| Pipeline | max streams @ >=25fps | aggregate fps ceiling |
|---|---|---|
| baseline | 4 (n=6 drops to 18.6) | ~111 fps (saturates n=3-4) |
| mv-fixed | 8 (n=10 drops to 21.5) | ~270 fps (peaks n=6) |
| mv-adaptive | 8 (n=12 drops to 17.1) | ~209 fps (saturates n=6) |

**Multi-stream sweeps can hang mid-run** (seen once: 6 spawned workers sat
at flat CPU time for 10+ minutes at the n=6 stage, no forward progress).
Cause not root-caused (suspect MPS/Metal context contention across
processes, not confirmed) — if a sweep looks stuck, check
`ps aux | grep bench_multistream` for CPU time not advancing across two
checks a minute apart, and just `kill -9` the tree and rerun rather than
waiting indefinitely. `bench_multistream.py` overwrites its output CSV
wholesale at the end of each run, so a killed run loses that run's data
entirely — rerun with the full `--streams` list rather than resuming from
where it died, or the CSV ends up with only the last completed subset.

`scripts/bench_multistream.py` runs N pipeline instances as separate
processes (spawn, not threads/fork — MPS contexts and GIL contention both
argue against threads) on the same clip, sweeping stream count until
per-stream fps drops below 25. Energy sampling via `powermetrics` needs
passwordless sudo (`sudo -n`); not set up in this environment, so the
power_mw column in `results/*.csv` is empty — set up a NOPASSWD sudoers
rule for powermetrics if energy numbers are wanted later (a system change,
do it yourself rather than having Claude edit sudoers). Results are
committed CSVs (`results/`, NOT gitignored like `outputs/`) so the
throughput/energy Pareto plots stay reproducible without rerunning.

## Working conventions

- This is exploratory systems/research code, not a product — favor fast
  iteration and real measurements over up-front test suites. Do add a quick
  correctness check (a synthetic fixture, a known-answer case) whenever a
  new numerical/data-format component is built; that's what caught the
  numpy shim issue above before it silently produced wrong scores.
- Every pipeline variant must go through `eval/run.py` so accuracy and fps
  numbers are directly comparable across the baseline and later MV-based
  variants — don't build one-off benchmarking scripts that skip it.
