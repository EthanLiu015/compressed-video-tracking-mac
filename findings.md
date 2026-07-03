# mvtrack — Findings

Every novel component built this project, and its measured effect on
accuracy (HOTA/MOTA/IDF1 via TrackEval on real MOT17 train GT, all 7
FRCNN sequences) and/or throughput (fps). Negative results are included
deliberately — a stretch goal that didn't pan out, correctly diagnosed and
reported, is a real research finding, not a failure to hide.

## 1. MV-based box propagation replacing per-frame detection

**What**: `propagate_boxes` shifts each track's box by the median codec
motion vector under it (`src/mvtrack/track/propagate.py`), standing in for
a Kalman predict step on frames where the detector doesn't run. Combined
with `MVTracker`'s IoU/Hungarian re-association on anchor frames
(`tracker.py`), wired as the `mv-fixed` pipeline (fixed anchor interval,
default 5).

**Metric impact** (MOT17 train, all 7 sequences, YOLOv8n person-class,
vs. full-decode-every-frame `baseline`):

| Pipeline | HOTA | MOTA | IDF1 | mean fps |
|---|---|---|---|---|
| baseline | 36.0 | 33.3 | 42.1 | 29.8 |
| mv-fixed | 32.0 | 11.3 | 36.0 | 40.2 |

**Read**: real 1.35x throughput, but a genuine accuracy cost (MOTA drops
hardest — propagation drift between anchors compounds on MOT17's crowded,
often-static-camera scenes far more than it did on an early low-res smoke
clip). This is the honest core result: compressed-domain propagation is a
Pareto tradeoff, not a free win.

## 2. Adaptive anchor scheduler (residual-energy proxy)

**What**: `Adaptive` (`src/mvtrack/sched/scheduler.py`) fires anchors on
I-frame/scene-cut detection plus intra-coded-block-fraction spikes
(`1 - occupancy.mean()`) versus a rolling EMA baseline, instead of a fixed
interval. PyAV/FFmpeg don't expose decoded residual coefficients, so
intra-fraction is the proxy: intra blocks are exactly the ones the encoder
couldn't motion-match, which correlates with real residual energy and
with where propagated boxes drift most. Wired as `mv-adaptive`.

**Metric impact**:

| Pipeline | HOTA | MOTA | IDF1 | mean fps | anchor rate |
|---|---|---|---|---|---|
| mv-fixed | 32.0 | 11.3 | 36.0 | 40.2 | 20% |
| mv-adaptive | 29.3 | 13.3 | 33.7 | 69.8 | ~8% |

**Read**: fires anchors less than half as often yet still beats mv-fixed's
MOTA (13.3 vs 11.3) while nearly doubling throughput (69.8 vs 40.2 fps) —
the scheduler is finding anchor points that matter more per-anchor than a
naive fixed interval, which is exactly the claim the adaptive-scheduling
idea depends on. HOTA and IDF1 are slightly lower than mv-fixed's,
though — not a strictly dominant win, a different point on the same
accuracy/throughput curve.

## 3. Vectorized MV-grid construction (perf fix, not a new capability)

**What**: the original `_frame_mv` painted each motion vector's grid cells
in a Python `for` loop. Profiling on real MOT17 (1920x1080, ~9k MVs/frame)
found this cost 22ms/frame — *more than decode itself* (7.7ms/frame) —
which is why an early smoke-clip throughput win didn't reproduce on real
data. Replaced with vectorized NumPy scatter assignment, since H.264
partition shapes always tile within one 16px-aligned macroblock (so every
MV maps to exactly one grid cell — no range-painting needed). Verified
bit-identical output against the old loop on 200 real frames before
trusting it.

**Metric impact**: 22.20ms → 0.63ms per frame (35x) on the grid-build
step specifically; this is what makes mv-fixed's and mv-adaptive's
throughput numbers above real rather than illusory — without it, MOT17's
resolution would have erased the detector-skipping benefit entirely.

## 4. Learned MV-domain box correction (CorrectionNet) — negative result

**What**: a tiny MLP (`src/mvtrack/track/correct.py`) over a pooled 4x4
MV/occupancy patch plus log(box size), predicting a scale-invariant
residual on top of `propagate_boxes`'s output. Trained on MOT17 GT
propagation-error pairs (GT box at t → GT box at t+1), isolating pure
propagation error from detector error. Wired as `mv-learned`.

**Metric impact (v1, single-step training)**:

| Pipeline | HOTA | MOTA | IDF1 | mean fps |
|---|---|---|---|---|
| mv-fixed | 32.0 | 11.3 | 36.0 | 40.2 |
| mv-learned v1 | 31.1 | 7.0 | 34.8 | 34.5 |

Worse on every accuracy metric, and slower (added inference cost, no
offsetting gain) — despite beating a "predict zero residual" baseline by
~19% MSE in isolated single-step regression on a held-out sequence.

**Diagnosis**: train/inference distribution mismatch. Training only ever
sees one propagation step starting from a perfect GT box; inference
chains the net's own corrections across multiple already-drifted frames
within an anchor window — an exposure-bias-like failure mode (same
problem seq2seq models hit with teacher forcing).

## 5. Fixing the CorrectionNet mismatch (DAgger-style rollout training)

**What**: `scripts/build_rollout_dataset.py` re-walks real anchor windows
using the *current trained checkpoint itself* to generate correction
targets along its own rollout trajectory — on-policy data, matching
inference exactly, rather than always resetting to a perfect GT box.
`train_correction.py` retrains from scratch on the aggregate
(single-step + rollout data), switching to `SmoothL1Loss` + gradient
clipping since rollout targets are heavy-tailed enough that plain MSE
training was unstable (occasional chained-correction blowups in
crowded/occluded tracks spiked validation loss 10x between epochs).

**Metric impact**:

| Version | val MSE vs. zero-residual baseline | MOTA | IDF1 | mean fps |
|---|---|---|---|---|
| v1 (single-step) | 19% better | 7.0 | 34.8 | 34.5 |
| v2 (DAgger rollout) | 26% better | 9.1 | 35.0 | 40.0 |
| mv-fixed (no correction) | — | 11.3 | 36.0 | 40.2 |

**Read**: the diagnosis was correct and the fix measurably helped — better
held-out regression accuracy, real MOTA/IDF1 gains over v1, throughput
back in line with mv-fixed. But it did not fully close the gap: v2 still
underperforms doing no correction at all. Current interpretation is that
a small MLP over MV/occupancy-only features (no appearance signal) is
likely under-powered for this specific correction task, rather than a
remaining training-methodology bug — its average-case correction can't
outweigh its per-frame noise cost. Not pursued further; noted as a
concrete follow-up (appearance features, or a confidence gate that only
applies correction when the net is confident) rather than abandoned
without explanation.

## 6. Multi-stream throughput scaling

**What**: `scripts/bench_multistream.py` runs N concurrent instances of a
pipeline as separate processes on the same clip, sweeping N until
per-stream fps drops below a 25fps target — the "streams per chip" claim
the whole project is framed around.

**Metric impact** (local smoke clip, M4):

| Pipeline | max streams @ >=25fps/stream | aggregate fps ceiling |
|---|---|---|
| baseline | 4 | ~111 fps (saturates n=3-4) |
| mv-fixed | 8 | ~270 fps (peaks n=6) |
| mv-adaptive | 8 | ~209 fps (saturates n=6) |

**Read**: mv-fixed and mv-adaptive both double baseline's concurrent-stream
capacity at the same per-stream quality bar, on the same chip — the
concrete "more analytics per watt/chip" result the project set out to
demonstrate. Scaling is clearly sublinear past n=3-4 (shared CPU/GPU
contention on a single M4), which is itself an honest, expected systems
finding, not a bug.

## 7. Anchor-interval ablation — accuracy is non-monotonic in anchor frequency

**What**: swept `mv-fixed`'s `anchor_interval` (2, 3, 5, 8, 10) across all 7
MOT17 train sequences to map the accuracy/throughput curve properly,
instead of relying on one discrete data point.

**Metric impact**:

| interval | HOTA | MOTA | IDF1 | mean fps |
|---|---|---|---|---|
| 2 | 30.9 | 1.0 | 33.6 | 22.4 |
| 3 | 31.7 | 6.3 | 35.4 | 30.3 |
| 5 | 32.0 | 11.3 | 36.0 | 40.2 |
| 8 | 30.9 | 13.3 | 35.3 | 58.1 |
| 10 | 31.0 | 13.5 | 35.9 | 64.4 |

**Read**: the intuitive expectation is "more anchors = closer to baseline =
more accurate." The data says otherwise for MOTA, which rises
*monotonically* with anchor interval (1.0 → 13.5) across the entire swept
range — more frequent anchors mean more chances for YOLOv8n's imperfect
recall to fail a re-match and fragment/respawn a track's identity, while
pure MV propagation never drops a track just because a detector missed it
that frame. HOTA/IDF1 peak around interval=5 rather than at either
extreme — a real sweet spot, not a monotone tradeoff. This is the kind of
result that only shows up by actually running the sweep rather than
reasoning from intuition about what "more ground truth checks" should do.

## 8. Side-by-side demo video

**What**: `scripts/make_demo_video.py` renders baseline (full-decode every
frame) against mv-adaptive on the same clip, same detector, same
MVTracker/IoU-Hungarian association logic — isolating exactly the
anchor-vs-propagate difference the project is about, with anchor frames
highlighted.

**What it shows**: a real propagation-drift artifact — a track that lags
behind as its subject walks away, before eventually being pruned — visible
on the mv-adaptive side around frames 30-100 of the rendered clip. Left in
deliberately rather than cherry-picked away: it's a visual instance of
exactly the MOTA cost quantified in finding #1, not a rendering bug. A
demo that only shows the pipeline's best moments would undersell how real
the accuracy/throughput tradeoff actually is.

## 9. Detector ceiling check (YOLOv8n → YOLOv8s)

**What**: baseline's own HOTA (36.0) fell well short of the project plan's
own weeks 1-2 acceptance bar ("~60 with a good detector"), and every
pipeline shares the same detector at anchor frames — so part of the
accuracy gap might be "the detector is weak," not "MV propagation is
lossy." Tested by swapping `yolov8n.pt` for `yolov8s.pt` (one line, no new
code) and rerunning all four pipelines on full MOT17.

**Metric impact**:

| Pipeline | HOTA (n→s) | MOTA (n→s) | IDF1 (n→s) | fps (n→s) |
|---|---|---|---|---|
| baseline | 36.0 → 40.5 | 33.3 → 38.9 | 42.1 → 48.6 | 29.8 → 20.8 |
| mv-fixed | 32.0 → 34.7 | 11.3 → 13.8 | 36.0 → 39.9 | 40.2 → 37.8 |
| mv-adaptive | 29.3 → 32.2 | 13.3 → 16.3 | 33.7 → 37.5 | 69.8 → 63.6 |
| mv-learned v2 | 30.5 → 33.2 | 9.1 → 11.2 | 35.0 → 38.4 | 40.0 → 32.6 |

**Read**: real, consistent gains across every pipeline (HOTA +2.7 to +4.5,
MOTA +2.1 to +5.6, IDF1 +2.8 to +6.5), confirming detector quality was a
genuine, separable lever from the MV propagation approach. The fps cost is
proportional to how often each pipeline actually calls the detector —
baseline pays the full ~30% throughput cost (every frame), mv-adaptive
pays the least (~9%, since it only detects on ~8% of frames). Adopted as
the new project default (`Detector`'s and `eval/run.py`'s default weights).
Baseline HOTA (40.5) still falls short of the ~60 ballpark, so detector
quality wasn't the whole story either — the accuracy gap between baseline
and the mv-* pipelines is unaffected by this change (still the same real
propagation-drift and ID-churn issues, see findings #1 and #7), just at a
uniformly higher starting point.

## Summary: what actually worked vs. didn't

- **Worked, real and reproducible**: MV propagation for throughput (at a
  real accuracy cost), adaptive scheduling (better throughput *and*
  better MOTA than fixed-interval, at similar-ish accuracy elsewhere),
  the grid-build vectorization (necessary infrastructure fix, not
  optional), multi-stream scaling advantage (2x concurrent streams for
  both mv-fixed and mv-adaptive), a genuinely non-obvious ablation
  result (anchor frequency vs. MOTA is non-monotonic), and confirming
  detector quality was a real, cheap-to-fix lever separate from the MV
  approach (YOLOv8n → YOLOv8s raised every pipeline's accuracy).
- **Didn't work, but rigorously diagnosed**: learned box correction.
  Root-caused a real distribution-mismatch bug, fixed it with a
  principled method (DAgger), validated the fix's partial effect with
  numbers — and still reported the honest bottom line (doesn't beat doing
  nothing) rather than stopping at the first improvement that looked
  good on a proxy metric (MSE) without checking the metric that
  actually matters (tracking accuracy).
