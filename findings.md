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
| -------- | ---- | ---- | ---- | -------- |
| baseline | 36.0 | 33.3 | 42.1 | 29.8     |
| mv-fixed | 32.0 | 11.3 | 36.0 | 40.2     |

**Read**: real 1.35x throughput, but a genuine accuracy cost (MOTA drops
hardest — propagation drift between anchors compounds on MOT17's crowded,
often-static-camera scenes far more than it did on an early low-res smoke
clip). This is the honest core result: compressed-domain propagation is a
Pareto tradeoff, not a free win.

**Update**: about half of this MOTA gap turned out to be a fixable
re-association bug, not an inherent property of MV propagation — see #10.
Current (YOLOv8s + the fix) numbers: baseline 38.9 vs mv-fixed 26.4 MOTA,
a ~12.5-point gap instead of the ~22-point gap shown above.

## 2. Adaptive anchor scheduler (residual-energy proxy)

**What**: `Adaptive` (`src/mvtrack/sched/scheduler.py`) fires anchors on
I-frame/scene-cut detection plus intra-coded-block-fraction spikes
(`1 - occupancy.mean()`) versus a rolling EMA baseline, instead of a fixed
interval. PyAV/FFmpeg don't expose decoded residual coefficients, so
intra-fraction is the proxy: intra blocks are exactly the ones the encoder
couldn't motion-match, which correlates with real residual energy and
with where propagated boxes drift most. Wired as `mv-adaptive`.

**Metric impact**:

| Pipeline    | HOTA | MOTA | IDF1 | mean fps | anchor rate |
| ----------- | ---- | ---- | ---- | -------- | ----------- |
| mv-fixed    | 32.0 | 11.3 | 36.0 | 40.2     | 20%         |
| mv-adaptive | 29.3 | 13.3 | 33.7 | 69.8     | ~8%         |

**Read (original, pre-#10 fix)**: fires anchors less than half as often
yet still beats mv-fixed's MOTA (13.3 vs 11.3) while nearly doubling
throughput (69.8 vs 40.2 fps) — the scheduler is finding anchor points
that matter more per-anchor than a naive fixed interval, which is exactly
the claim the adaptive-scheduling idea depends on. HOTA and IDF1 are
slightly lower than mv-fixed's, though — not a strictly dominant win, a
different point on the same accuracy/throughput curve.

**Update (post-#10 re-association fix + YOLOv8s, before scheduler
tuning)**: the MOTA ranking flipped. mv-fixed 26.4 vs mv-adaptive 21.8 —
mv-fixed is now _better_ on MOTA (and HOTA/IDF1) despite anchoring 2.5x
more often, with mv-fixed's throughput advantage disappearing (mv-adaptive
is still 1.6x faster). Plausible explanation: the fixed pipeline's
re-association fix (stages 2-3) specifically helps _recover_ tracks
across anchors — a benefit that scales with how often anchors happen, so
the more-frequent fixed-interval schedule gets more chances to benefit
from the fix than the sparser adaptive schedule does. The `Adaptive`
scheduler's own thresholds had never been tuned against real accuracy at
all (arbitrary defaults) — confirmed as the actual explanation once tuned,
see #11.

## 3. Vectorized MV-grid construction (perf fix, not a new capability)

**What**: the original `_frame_mv` painted each motion vector's grid cells
in a Python `for` loop. Profiling on real MOT17 (1920x1080, ~9k MVs/frame)
found this cost 22ms/frame — _more than decode itself_ (7.7ms/frame) —
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

| Pipeline      | HOTA | MOTA | IDF1 | mean fps |
| ------------- | ---- | ---- | ---- | -------- |
| mv-fixed      | 32.0 | 11.3 | 36.0 | 40.2     |
| mv-learned v1 | 31.1 | 7.0  | 34.8 | 34.5     |

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
using the _current trained checkpoint itself_ to generate correction
targets along its own rollout trajectory — on-policy data, matching
inference exactly, rather than always resetting to a perfect GT box.
`train_correction.py` retrains from scratch on the aggregate
(single-step + rollout data), switching to `SmoothL1Loss` + gradient
clipping since rollout targets are heavy-tailed enough that plain MSE
training was unstable (occasional chained-correction blowups in
crowded/occluded tracks spiked validation loss 10x between epochs).

**Metric impact**:

| Version                  | val MSE vs. zero-residual baseline | MOTA | IDF1 | mean fps |
| ------------------------ | ---------------------------------- | ---- | ---- | -------- |
| v1 (single-step)         | 19% better                         | 7.0  | 34.8 | 34.5     |
| v2 (DAgger rollout)      | 26% better                         | 9.1  | 35.0 | 40.0     |
| mv-fixed (no correction) | —                                  | 11.3 | 36.0 | 40.2     |

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

| Pipeline    | max streams @ >=25fps/stream | aggregate fps ceiling      |
| ----------- | ---------------------------- | -------------------------- |
| baseline    | 4                            | ~111 fps (saturates n=3-4) |
| mv-fixed    | 8                            | ~270 fps (peaks n=6)       |
| mv-adaptive | 8                            | ~209 fps (saturates n=6)   |

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
| -------- | ---- | ---- | ---- | -------- |
| 2        | 30.9 | 1.0  | 33.6 | 22.4     |
| 3        | 31.7 | 6.3  | 35.4 | 30.3     |
| 5        | 32.0 | 11.3 | 36.0 | 40.2     |
| 8        | 30.9 | 13.3 | 35.3 | 58.1     |
| 10       | 31.0 | 13.5 | 35.9 | 64.4     |

**Read**: the intuitive expectation is "more anchors = closer to baseline =
more accurate." The data says otherwise for MOTA, which rises
_monotonically_ with anchor interval (1.0 → 13.5) across the entire swept
range — more frequent anchors mean more chances for YOLOv8n's imperfect
recall to fail a re-match and fragment/respawn a track's identity, while
pure MV propagation never drops a track just because a detector missed it
that frame. HOTA/IDF1 peak around interval=5 rather than at either
extreme — a real sweet spot, not a monotone tradeoff. This is the kind of
result that only shows up by actually running the sweep rather than
reasoning from intuition about what "more ground truth checks" should do.

**Update (see #10)**: this exact inversion is what motivated a fix to
`MVTracker.step_anchor`'s re-association logic. After the fix, the
inversion is gone — kept here as the finding that prompted the fix, not
as the current state of the pipeline.

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

| Pipeline      | HOTA (n→s)  | MOTA (n→s)  | IDF1 (n→s)  | fps (n→s)   |
| ------------- | ----------- | ----------- | ----------- | ----------- |
| baseline      | 36.0 → 40.5 | 33.3 → 38.9 | 42.1 → 48.6 | 29.8 → 20.8 |
| mv-fixed      | 32.0 → 34.7 | 11.3 → 13.8 | 36.0 → 39.9 | 40.2 → 37.8 |
| mv-adaptive   | 29.3 → 32.2 | 13.3 → 16.3 | 33.7 → 37.5 | 69.8 → 63.6 |
| mv-learned v2 | 30.5 → 33.2 | 9.1 → 11.2  | 35.0 → 38.4 | 40.0 → 32.6 |

**Read**: real, consistent gains across every pipeline (HOTA +2.7 to +4.5,
MOTA +2.1 to +5.6, IDF1 +2.8 to +6.5), confirming detector quality was a
genuine, separable lever from the MV propagation approach. The fps cost is
proportional to how often each pipeline actually calls the detector —
baseline pays the full ~30% throughput cost (every frame), mv-adaptive
pays the least (~9%, since it only detects on ~8% of frames). Adopted as
the new project default (`Detector`'s and `eval/run.py`'s default weights).
Baseline HOTA (40.5) still falls short of the ~60 ballpark, so detector
quality wasn't the whole story either — the accuracy gap between baseline
and the mv-\* pipelines is unaffected by this change (still the same real
propagation-drift and ID-churn issues, see findings #1 and #7), just at a
uniformly higher starting point.

## 10. Fixing anchor re-association resolves the MOTA inversion (biggest single accuracy win)

**What**: finding #7 showed more frequent anchors made MOTA _worse_, not
better. Root cause read from `MVTracker.step_anchor`: a single-pass
IoU-Hungarian match with a hard 0.3 threshold spawned a brand-new track ID
for any unmatched detection, no grace period, no confidence-tiered
association — so every anchor frame was an independent chance for
YOLO's imperfect recall to fragment an existing identity. Rewrote it as a
three-stage ByteTrack-style match (`src/mvtrack/track/tracker.py`):
(1) high-confidence detections vs. all tracks at the original threshold,
(2) low-confidence detections (score < 0.5, never spawn on their own) get
a looser-IoU chance to recover tracks stage 1 missed, (3) detections still
unmatched after stage 1 get one more loose-IoU "grace period" chance to
reattach to a still-unmatched track before spawning a new ID. Verified via
a 5-scenario synthetic sanity script (basic match, low-conf recovery,
grace reattachment, full-miss spawn, unmatched-low-conf discard) before
running on real data.

**Metric impact** — the anchor-interval ablation, re-run after the fix
(YOLOv8s, same 7 sequences):

| interval | HOTA (before→after) | MOTA (before→after) | IDF1 (before→after) | IDs produced (before→after, GT=546) |
| -------- | ------------------- | ------------------- | ------------------- | ----------------------------------- |
| 2        | 30.9→37.0           | 1.0→25.5            | 33.6→43.5           | 1458→584                            |
| 3        | 31.7→36.8           | 6.3→25.7            | 35.4→43.7           | 1326→580                            |
| 5        | 32.0→35.7           | 11.3→26.4           | 36.0→42.5           | 1070→545                            |
| 8        | 30.9→34.7           | 13.3→25.1           | 35.3→41.6           | 941→518                             |
| 10       | 31.0→34.3           | 13.5→24.6           | 35.9→41.4           | 1004→514                            |

(Before-numbers above are YOLOv8n/pre-fix from #7 for the original
inversion; the fix itself was tested on top of the YOLOv8s default from
#9, so some of the HOTA/IDF1 delta reflects both changes together — MOTA's
qualitative shape change, from monotonically rising to flat, is the signal
that isolates this fix's effect, since detector quality alone wouldn't
explain an inversion disappearing.)

**Read**: the inversion is gone. MOTA is now roughly flat (24.6-26.4%)
across the whole interval range instead of spanning a 25x range, and
HOTA/IDF1 now decrease with larger intervals as intuition would predict.
Track-ID counts dropped toward the real count (546) at every interval —
direct, mechanism-level evidence that ID fragmentation was the actual
problem, not a coincidental correlation. This was the single biggest
accuracy win in the whole project: on the full pipeline comparison,
mv-fixed's MOTA gap to baseline narrowed from ~22 points (11.3 vs 33.3
originally, both YOLOv8n) to ~12.5 points (26.4 vs 38.9, both YOLOv8s +
the fix), roughly halving the accuracy cost of the compressed-domain
approach without touching the core MV propagation idea at all — the win
came entirely from fixing how detections re-associate with existing
tracks, a pure software-engineering fix, not a new algorithm.

## 11. Tuning the Adaptive scheduler against real MOTA (closes the #2 regression)

**What**: `Adaptive`'s four parameters (`min_interval`, `max_interval`,
`spike_factor`, `ema_alpha`) were pure arbitrary dataclass defaults, never
tuned against ground truth (confirmed via repo-wide grep). After the
re-association fix (#10) flipped mv-adaptive from beating mv-fixed on
MOTA to trailing it, ran a 16-combination grid search
(`scripts/tune_scheduler.py`) on 2 fast tuning sequences (MOT17-09,
MOT17-10, held separate from the full-7 set used for final reporting),
holding `min_interval=2` and `ema_alpha=0.2` fixed and sweeping
`max_interval` in {8,10,15,20} x `spike_factor` in {1.2,1.4,1.6,2.0}.

**Metric impact** — grid search (tuning sequences only), showing MOTA vs.
`max_interval` (the dominant lever; `spike_factor` had a smaller effect
within each group, best value shifted slightly but stayed near 1.2-1.4):

| max_interval                 | 8    | 10   | 15   | 20   |
| ---------------------------- | ---- | ---- | ---- | ---- |
| best MOTA (any spike_factor) | 27.9 | 27.0 | 22.1 | 21.2 |

Then validated the winner (`max_interval=8, spike_factor=1.4`, down from
`15, 1.6`) on the full 7-sequence set:

| mv-adaptive                   | HOTA | MOTA | IDF1 | mean fps | anchor rate |
| ----------------------------- | ---- | ---- | ---- | -------- | ----------- |
| untuned (`max=15, spike=1.6`) | 32.5 | 21.8 | 39.1 | 62.9     | ~8%         |
| tuned (`max=8, spike=1.4`)    | 35.4 | 25.4 | 42.7 | 44.2     | ~15-17%     |

**Read**: `max_interval` dominated the sweep — MOTA fell monotonically as
`max_interval` grew from 8 to 20, meaning the scheduler's untuned default
(15) was simply anchoring too rarely once re-association could actually
use those anchors productively (#10's fix made anchors more valuable, so
the optimal anchor rate went up in response — the two fixes interact,
they aren't independent). The tuned scheduler restores mv-adaptive to
roughly matching mv-fixed on every accuracy metric (35.4/25.4/42.7 vs
35.7/26.4/42.5) while staying meaningfully faster (44.2 vs 38.5 fps) —
this closes the regression #2 found and gives mv-adaptive a real reason
to exist again: same accuracy as fixed-interval anchoring, less compute.
A grid search over dataclass defaults, no new capability, is the cheapest
possible lever in this entire project and had the second-largest impact
after #10 — a reminder that tuning already-correct code can matter as
much as fixing broken code.

## 12. Per-edge scale correction in propagate_boxes — negative result

**What**: after #10/#11 closed most of the ID-churn/scheduling gap, the
remaining MOTA gap to baseline turned out to be flat across anchor
frequency (mv-fixed at interval=2, 50% anchor rate, still only hit MOTA
25.5 — barely above interval=5's 26.4), proving the residual gap wasn't a
scheduling problem. `propagate_boxes` only ever did rigid translation
(one whole-box median MV shift, width/height never change) — a plausible
next lever, since MOT17 subjects walking toward/away from the camera
should drift in scale with no correction. Rewrote it to shift each edge
independently by the _local_ median MV near that edge (left/right bands
for x0/x1, top/bottom bands for y0/y1) instead of one global median, which
captures scale change for free without an explicit scale-factor estimate.
Verified via 5 synthetic scenarios first (uniform field -> pure
translation preserved exactly, diverging field -> box grows, converging
field -> box shrinks, narrow box doesn't invert, zero-coverage box
unchanged) before running on real data.

**Metric impact**:

| Pipeline              | HOTA (before→after) | MOTA (before→after) | IDF1 (before→after) |
| --------------------- | ------------------- | ------------------- | ------------------- |
| mv-fixed (interval=5) | 35.73→36.00         | 26.38→25.60         | 42.47→42.96         |
| mv-adaptive (tuned)   | 35.40→35.19         | 25.42→24.81         | 42.71→42.43         |

**Read**: flat-to-slightly-negative on the metric that matters most
(MOTA down in both pipelines), mixed/marginal on HOTA and IDF1. Reverted
rather than adopted. Likely explanation: MOT17 pedestrians mostly move
laterally across the frame rather than directly toward/away from the
camera, so scale drift may not be the dominant real error mode in this
dataset — while the per-edge bands (roughly 1-2 of the already-coarse
16px cells wide, since typical pedestrian boxes span only a few cells)
are small enough that their median estimates are noisier than the
whole-box median, adding box-size jitter that costs more localization
precision than the occasional real scale correction saves. Same shape of
result as CorrectionNet (#4): a mechanistically sound idea, carefully
implemented and verified, that simply didn't survive contact with real
data — reported here rather than either silently dropped or kept despite
a net-negative measurement. Would need either coarser/more robust
per-edge statistics or a way to detect when scale-change is actually
happening (e.g. from box-size trend over recent anchors) before applying
the correction selectively, rather than unconditionally on every box.

## 13. Appearance-based re-identification — generic backbone flat, real ReID model wins

**What**: motion vectors carry zero information about what something looks
like — a structural ceiling every fix so far had bumped into (findings.md
#9-12). Built `ReIDEmbedder` (`src/mvtrack/track/reid.py`): a generic
ImageNet-pretrained MobileNetV3-Small backbone (no time budget in this
pass to curate MOT17 identity triplets and train a dedicated person-ReID
model), global-average-pooled and L2-normalized, run on each anchor-frame
detection crop. `MVTracker` (`use_appearance` flag, opt-in, off by
default) blends cosine similarity into the IoU assignment cost across all
three association stages, with an EMA-smoothed per-track embedding and an
extra similarity floor gating stages 2/3 (the loosest, most
recovery-prone stages) against recovering the wrong person. Verified via
4 synthetic scenarios first: appearance breaks an IoU tie toward the
correct person, a wrong-appearance low-conf detection is correctly
rejected by the similarity floor despite good IoU, embedding EMA update
behaves as expected, and `use_appearance=False` is unaffected even if
embeddings are passed. Also hit and diagnosed a real but cosmetic bug:
Apple's Accelerate BLAS backend (numpy's default on Apple Silicon) raises
spurious divide-by-zero/overflow warnings on some embedding matmuls;
verified the actual output stays correct and finite via a manual
dot-product cross-check before suppressing the warning at the call site.

**Metric impact**:

| Pipeline | HOTA (off→on) | MOTA (off→on) | IDF1 (off→on) | mean fps (off→on) |
|---|---|---|---|---|
| mv-fixed (interval=5) | 35.73→35.82 | 26.38→26.49 | 42.47→42.55 | 38.5→36.1 |
| mv-adaptive (tuned) | 35.40→35.40 | 25.42→25.41 | 42.71→42.63 | 44.2→46.2 |

**Read**: essentially flat — every delta is within the fps/run-to-run
noise already documented elsewhere in this project, and mv-adaptive
shows literally zero net movement. Before concluding the idea itself is
dead, ran one more diagnostic: on MOT17-04 (the most crowded sequence),
appearance actually changes the Hungarian assignment vs. IoU-only on 16
of 61 sampled anchor-to-anchor comparisons (26%) — so the signal isn't
being gated into irrelevance by overly conservative thresholds, it's
genuinely influencing roughly a quarter of decisions. It just isn't
influencing them *correctly* enough, on net, to beat IoU alone. Most
likely explanation: a generic ImageNet classification backbone's features
emphasize "this is a person" (object category), not "this is *this*
person" (individual identity/clothing/texture) — exactly the gap the ReID
literature's dedicated metric-learning training (triplet/contrastive loss
on same-vs-different-person pairs) exists to close, and exactly what this
implementation doesn't have. Kept in the codebase as an opt-in
(`--use-reid`, off by default, zero cost when unused) rather than reverted
like #12, since it's a genuine working capability with a clearly
diagnosed limitation, not a net-negative default.

**Update: swapped in a real pretrained person-ReID model, and it works.**
The diagnosis pointed at the embedding source, not the association
machinery — so rather than train anything on MOT17 (small identity pool,
real overfitting risk), sourced OSNet (Zhou et al., ICCV'19), pretrained
by its original authors on MSMT17 (4101 identities, 15 cameras — an actual
ReID benchmark, not a classification dataset), hosted reliably on
Hugging Face Hub (`kaiyangzhou/osnet`, downloaded via `huggingface_hub`,
not the flaky Google Drive links torchreid's own model zoo uses). Zero
training on MOT17 at all, so zero overfitting risk to it. Verified the
checkpoint loads with 0 missing/0 unexpected keys against the `torchreid`
package's architecture definition (confirms it's the right checkpoint for
the right code, not a mismatched asset), and directly confirmed it's more
discriminative than the generic backbone on real MOT17 crops before
wiring it in: same-identity vs. different-identity cosine similarity gap
of 0.511 vs. 0.428 for the ImageNet backbone.

**Metric impact** (`ReIDEmbedder` swapped from ImageNet MobileNetV3 to
OSNet x0.25/MSMT17, same association code, both pipelines' `--use-reid`):

| Pipeline | HOTA (off→OSNet) | MOTA (off→OSNet) | IDF1 (off→OSNet) | mean fps (off→OSNet) |
|---|---|---|---|---|
| mv-fixed (interval=5) | 35.73→36.50 | 26.38→26.45 | 42.47→43.58 | 38.5→34.4 |
| mv-adaptive (tuned) | 35.40→35.59 | 25.42→25.56 | 42.71→43.06 | 44.2→38.9 |

**Read**: a real, non-noise-level win on mv-fixed's IDF1 (+2.6% relative)
and a smaller HOTA gain (+2.2% relative), but the honest framing needs the
cost held next to it: the fps drop is ~11% on mv-fixed and ~12% on
mv-adaptive — proportionally *larger* than the accuracy gain in relative
terms, and on mv-adaptive the gain is nearly nothing (+0.5% HOTA, +0.8%
IDF1) for that same ~12% throughput cost. So this is a genuine Pareto
tradeoff point, not a strict win: worth it if identity consistency
(IDF1) matters more than raw throughput, not worth it if the project's
actual headline claim (streams per chip) is what's being optimized —
the multi-stream sweep was never rerun with `--use-reid` on, so the
concrete impact on max-concurrent-streams is unmeasured but a single-run
fps hit this size would plausibly lower that ceiling too. Confirms the
original diagnosis was right (association mechanism was sound, embedding
quality was the bottleneck), and MOTA barely moving is consistent with
IDF1 being the metric this fix should affect most directly. Kept opt-in
(`--use-reid`) rather than default, both because of the cost and because
this project's own multi-stream throughput claim is the thing being
traded away.

## 14. Energy measurement (manual, one-off) — real signal after a noisy first attempt

**What**: `scripts/bench_multistream.py --power` needs passwordless sudo
for `powermetrics`, never configured. Rather than edit sudoers, tried a
one-off manual measurement: user runs `sudo powermetrics` themselves
(interactive password entry) while a pipeline runs, redirecting output to
a file read afterward. First attempt (15s sample overlapping a single
short sequence run) showed close to zero signal, and GPU power actually
*lower* during "load" than idle — a clear tell it was noise, most likely
from `powermetrics` measuring total system power (not per-process) plus
imprecise start-time overlap between two manually-launched commands. A
second attempt used a much longer, more sustained load: the full 7-sequence
MOT17 `mv-adaptive` eval (~150s wall time) against a matching 150-sample
power trace.

**Metric impact**:

| Condition | duration | CPU power | GPU power | Combined |
|---|---|---|---|---|
| Idle | 8s (8 samples) | 9494 mW | 137 mW | 9631 mW |
| mv-adaptive, full MOT17 | 150s (150 samples) | 10157 mW | 505 mW | 10663 mW |
| **Delta** | | **+7.0%** | **+2.7x** | **+10.7%** |

**Read**: a real, physically-sensible signal this time — GPU power nearly
tripling under sustained load is exactly what MPS inference actively
running should look like, unlike the backwards result from the short
sample. Caveats that matter: `powermetrics` reports total system power
(display, background processes, everything), not this process alone, so
the delta is a reasonable but not perfectly clean estimate of the
pipeline's own draw; the idle baseline (8s) is much shorter than the load
sample (150s) and ideally would be re-measured at matching duration for a
fairer comparison; and this covers `mv-adaptive` only — a full
"streams-per-watt" story (the original plan's stated goal) would need the
same treatment applied to `baseline` and `mv-fixed` too, not done here.
Summary in `results/energy_measurement.csv`.

## Summary: what actually worked vs. didn't

- **Worked, real and reproducible**: MV propagation for throughput (at a
  real accuracy cost, though substantially narrowed later), adaptive
  scheduling — after tuning (#11) restored it to matching mv-fixed's
  accuracy while staying faster, closing a regression #10's fix had
  introduced — the grid-build vectorization (necessary infrastructure
  fix, not optional), multi-stream scaling advantage (2x concurrent
  streams for both mv-fixed and mv-adaptive), confirming detector quality
  was a real, cheap-to-fix lever
  separate from the MV approach (#9), and — the single biggest accuracy
  win in the project — fixing anchor re-association to eliminate a real,
  measured ID-churn problem that had been making more-frequent anchors
  actively _hurt_ MOTA (#10), roughly halving mv-fixed's MOTA gap to
  baseline without changing the core MV propagation idea at all.
- **Didn't work, but rigorously diagnosed**: learned box correction.
  Root-caused a real distribution-mismatch bug, fixed it with a
  principled method (DAgger), validated the fix's partial effect with
  numbers — and still reported the honest bottom line (doesn't beat doing
  nothing) rather than stopping at the first improvement that looked
  good on a proxy metric (MSE) without checking the metric that
  actually matters (tracking accuracy).
- **Didn't work, reverted rather than kept**: per-edge scale correction in
  `propagate_boxes` (#12). A mechanistically reasonable idea (translation
  couldn't capture scale drift), verified correct on 5 synthetic scenarios,
  measured flat-to-slightly-negative on real MOT17 (MOTA down in both
  mv-fixed and mv-adaptive) — reverted to the original rather than kept
  for a mixed-at-best result, the same discipline applied to CorrectionNet.
- **Diagnosed flat, fixed, and a real (if costly) tradeoff**:
  appearance-based re-identification (#13). First pass (generic
  ImageNet-pretrained embedding) measured flat despite genuinely
  influencing ~26% of real assignment decisions — diagnosed cause: a
  classifier backbone encodes "this is a person," not "this is *this*
  person." Sourced OSNet, pretrained on MSMT17 (an actual ReID benchmark)
  from Hugging Face Hub instead of training anything on MOT17's small
  identity pool — zero overfitting risk. Result: a real HOTA/IDF1 gain on
  mv-fixed, next to nothing on mv-adaptive, at an ~11-12% fps cost on
  both — a genuine Pareto tradeoff, not a strict win, and one that likely
  cuts against this project's own headline streams-per-chip claim. Kept
  opt-in given the cost.
- **A real, if partial, energy result** (#14): manual `powermetrics`
  measurement (no automated sudo access configured) showed combined power
  up 10.7% and GPU power up 2.7x under sustained `mv-adaptive` load vs.
  idle — physically sensible (GPU tripling matches active MPS inference),
  unlike a first noisy short-sample attempt that showed backwards,
  not-credible numbers. Caveated honestly: total-system power, not
  per-process, and only one pipeline measured, not the full
  baseline-vs-mv-* comparison the original plan wanted.
