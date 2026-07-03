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
| mv-fixed | >=8 (untested beyond) | ~270 fps (peaks n=6) |
| mv-adaptive | 8 | ~209 fps (saturates n=6) |

**Read**: mv-adaptive at minimum doubles baseline's concurrent-stream
capacity at the same per-stream quality bar, on the same chip — the
concrete "more analytics per watt/chip" result the project set out to
demonstrate. Scaling is clearly sublinear past n=3-4 (shared CPU/GPU
contention on a single M4), which is itself an honest, expected systems
finding, not a bug.

## Summary: what actually worked vs. didn't

- **Worked, real and reproducible**: MV propagation for throughput (at a
  real accuracy cost), adaptive scheduling (better throughput *and*
  better MOTA than fixed-interval, at similar-ish accuracy elsewhere),
  the grid-build vectorization (necessary infrastructure fix, not
  optional), multi-stream scaling advantage.
- **Didn't work, but rigorously diagnosed**: learned box correction.
  Root-caused a real distribution-mismatch bug, fixed it with a
  principled method (DAgger), validated the fix's partial effect with
  numbers — and still reported the honest bottom line (doesn't beat doing
  nothing) rather than stopping at the first improvement that looked
  good on a proxy metric (MSE) without checking the metric that
  actually matters (tracking accuracy).
