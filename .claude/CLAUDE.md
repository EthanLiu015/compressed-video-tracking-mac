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

**Current defaults: YOLOv8s detector + three-stage ByteTrack-style
re-association in `MVTracker.step_anchor` + tuned `Adaptive` scheduler
params (`max_interval=8, spike_factor=1.4`, was `15, 1.6`) + `max_age` tied
to each pipeline's max anchor gap (`anchor_interval` for mv-fixed,
`scheduler.max_interval` for mv-adaptive, was a flat `30` for both).** The
first three changes came from a follow-on accuracy-improvement pass (plan
in `~/.claude/plans/breezy-knitting-dragon.md`, "Follow-on" section) that
started from the observation that MOTA was hit far harder than HOTA/IDF1
across every mv-* pipeline; the fourth (`max_age`) came from a later pass
targeting the same MOTA gap directly (see item 4 and `findings.md` #15):

| Pipeline | HOTA | MOTA | IDF1 | mean fps |
|---|---|---|---|---|
| baseline (full decode+detect every frame) | 40.5 | 38.9 | 48.6 | 20.8 |
| mv-fixed (anchor every 5th frame) | 36.0 | 34.9 | 42.4 | 38.5* |
| mv-adaptive (~15-17% anchor rate, tuned) | 36.0 | 33.1 | 43.0 | 44.2* |

\* fps carried over from the pre-`max_age`-fix measurement (the fix only
changes when a track is pruned, not per-frame work, so it shouldn't move
throughput) — the in-session re-measurement after this fix landed 19.0/23.3,
but that run coincided with `Load Avg: 22` and 73% sys time on this
machine (confirmed via `top`, after ~45 min of continuous back-to-back
MPS jobs), a documented confound (see the "fps run-to-run variance"
gotcha below), not a real regression. Re-measure fps in a fresh/cool
session before trusting a revision to these numbers either way.

<details>
<summary>Superseded numbers (kept for reference — see git history for the swap/fix commits)</summary>

YOLOv8s + re-association fix + scheduler tuning, before the `max_age` fix
(the numbers reported as "current" prior to this pass):

| Pipeline | HOTA | MOTA | IDF1 | mean fps |
|---|---|---|---|---|
| mv-fixed | 35.7 | 26.4 | 42.5 | 38.5 |
| mv-adaptive | 35.4 | 25.4 | 42.7 | 44.2 |
| mv-learned v2 (mv-fixed + CorrectionNet, DAgger rollout training) | 34.6 | 25.5 | 41.7 | 33.7 |

YOLOv8s + re-association fix, before scheduler tuning (mv-adaptive still
on untuned `max_interval=15, spike_factor=1.6`):

| Pipeline | HOTA | MOTA | IDF1 | mean fps |
|---|---|---|---|---|
| mv-adaptive | 32.5 | 21.8 | 39.1 | 62.9 |

YOLOv8s, before the re-association fix:

| Pipeline | HOTA | MOTA | IDF1 | mean fps |
|---|---|---|---|---|
| baseline | 40.5 | 38.9 | 48.6 | 20.8 |
| mv-fixed | 34.7 | 13.8 | 39.9 | 37.8 |
| mv-adaptive | 32.2 | 16.3 | 37.5 | 63.6 |
| mv-learned v2 | 33.2 | 11.2 | 38.4 | 32.6 |

YOLOv8n, original numbers:

| Pipeline | HOTA | MOTA | IDF1 | mean fps |
|---|---|---|---|---|
| baseline | 36.0 | 33.3 | 42.1 | 29.8 |
| mv-fixed | 32.0 | 11.3 | 36.0 | 40.2 |
| mv-adaptive | 29.3 | 13.3 | 33.7 | 69.8 |
| mv-learned v1 (single-step training) | 31.1 | 7.0 | 34.8 | 34.5 |
| mv-learned v2 (DAgger rollout training) | 30.5 | 9.1 | 35.0 | 40.0 |

</details>

**What changed and why:** three fixes, in order of impact.

1. **Detector ceiling check** (YOLOv8n -> YOLOv8s): real gains across
   every pipeline, modest fps cost (see findings.md #9).
2. **Fixed `MVTracker.step_anchor`'s re-association** — this was the
   bigger lever. The original single-pass IoU-Hungarian match spawned a
   brand-new track ID for *any* unmatched detection, and the
   anchor-interval ablation had shown this actively caused ID churn: more
   anchors meant more chances for YOLO's imperfect recall to fragment an
   identity, so MOTA rose *monotonically with anchor interval* (1.0 at
   interval=2 up to 13.5 at interval=10) instead of falling — the
   opposite of intuition. Rewrote it as a three-stage ByteTrack-style
   match: (1) high-confidence detections vs. all tracks at the original
   IoU threshold, (2) low-confidence detections get a looser-threshold
   chance to recover tracks stage 1 missed (never spawn on their own),
   (3) detections still unmatched after stage 1 get one more loose-IoU
   "grace period" chance to reattach to a still-unmatched track before
   falling through to spawning a new ID. Verified via a synthetic
   sanity script (stage-1 basic match, stage-2 low-conf recovery,
   stage-3 grace reattachment, full-miss spawn, unmatched-low-conf
   discard) before trusting it on real MOT17.

   Effect on the anchor-interval ablation (mv-fixed, YOLOv8s): the
   inversion is **gone**. MOTA is now roughly flat (24.6-26.4%) across
   the whole interval range instead of rising 25x from one end to the
   other, and HOTA/IDF1 now decrease with larger intervals as intuition
   would suggest. Track-ID counts also dropped dramatically toward the
   real ground-truth count (e.g. interval=5: 545 IDs produced vs 546 real
   ones, down from 941-1458 IDs pre-fix depending on detector/interval) —
   direct evidence the fragmentation problem is what got fixed, not some
   unrelated confound. Full before/after ablation table in `findings.md`
   #7 and #10.

   One side effect worth flagging: this fix flipped the mv-fixed vs
   mv-adaptive MOTA ranking (mv-adaptive used to win, 13.3 vs 11.3;
   mv-fixed now wins pre-tuning, 26.4 vs 21.8) — plausible explanation is
   that re-association recovery benefits scale with how often anchors
   happen, so the denser fixed-interval schedule got more chances to
   benefit than the sparser untuned adaptive one. See item 3.

3. **Tuned the `Adaptive` scheduler's parameters against real MOTA**
   (`scripts/tune_scheduler.py`) — a 16-combination grid search on 2 fast
   tuning sequences (MOT17-09, MOT17-10), holding `min_interval=2` and
   `ema_alpha=0.2` fixed and sweeping `max_interval` in {8,10,15,20} x
   `spike_factor` in {1.2,1.4,1.6,2.0}. `max_interval` was the dominant
   lever by far — MOTA fell monotonically as `max_interval` grew (27.9 at
   8 down to ~20 at 20) — i.e. anchoring more often just helped a lot
   post-fix, unsurprising since re-association can now actually recover
   from a miss instead of always fragmenting. Winner: `max_interval=8,
   spike_factor=1.4` (from `15, 1.6`), validated on the full 7 sequences:
   HOTA 32.5->35.4, MOTA 21.8->25.4, IDF1 39.1->42.7, at a real fps cost
   (62.9->44.2, anchor rate ~8%->~15-17%) — this restores mv-adaptive to
   roughly matching mv-fixed (35.4/25.4/42.7 vs 35.7/26.4/42.5) while
   still being meaningfully faster (44.2 vs 38.5 fps).

4. **Tied `MVTracker.max_age` to each pipeline's actual max anchor gap**
   (was a flat `30` for every pipeline) — the single biggest MOTA win of
   any pass so far, bigger than items 1-3 combined. Motivated by a
   diagnostic showing item 1-3's residual gap wasn't a recall problem: a
   scratchpad probe (greedy IoU-recall vs ground truth, bucketed by
   frames-since-last-anchor on MOT17-04) found recall on a *fresh anchor
   frame itself* was only ~43.5%, nearly identical to frames 4 steps into
   the propagation window (~42.7%) — i.e. flat, and already at the
   detector's own ceiling on this scene, not degrading with drift as
   hypothesized. Confirmed via a matched-subset yolov8s-vs-yolov8m test:
   FN counts were nearly identical between baseline and mv-fixed on
   MOT17-09+10 (8633 vs 8648), but **CLR_FP was 3.3x higher for mv-fixed**
   (1257 vs 4120) — the gap was false positives, not missed detections.
   Root cause: a track missed at one anchor wasn't pruned until
   `since_detection > max_age`; with the default of 30 it kept being
   propagated *and reported* as a live box for up to 30 more frames (~6
   anchor cycles) — a ghost track. Fix: default `max_age` to
   `anchor_interval` for mv-fixed and to `scheduler.max_interval` for
   mv-adaptive in `eval/run.py` (the `MVTracker` dataclass keeps `30` only
   as a documented fallback for direct construction). Hit one real pitfall
   mid-fix: a `max_age=5` tuned against mv-fixed collapsed mv-adaptive's
   HOTA from 31.7 to 9.5 (IDSW 123->820), because mv-adaptive's scheduler
   can leave gaps up to `max_interval=8` — pruning at 5 killed legitimate
   tracks *before* their real next anchor ever arrived, forcing spurious
   respawns instead of just killing ghosts. Each pipeline needed its own
   value tuned/derived separately; see `findings.md` #15 for the full
   sweep (`scripts/tune_max_age.py`) and both failure/success numbers.

MV propagation is still a real accuracy/throughput tradeoff, not a free
win — MOTA still trails baseline (38.9 vs. 34.9/33.1 for mv-fixed/mv-adaptive)
even after all four fixes, because propagation drift between anchors is
still real. But the gap narrowed dramatically: from ~25-28 points before
this whole follow-on effort, to ~13-14 after items 1-3, to **~4-6 points
(10.1%/14.85% relative) after item 4** — the first time the relative MOTA
drop has come in under a 15% target. This is the honest core result;
report it as-is rather than only the throughput number. Note baseline
itself (HOTA 40.5) still falls short of a well-tuned ByteTrack-on-MOT17
ballpark (~60) — YOLOv8s is better than YOLOv8n but still a small/fast
model. A larger detector (yolov8m) was tried as a further lever on the
same 2 tuning sequences and **rejected**: it raised baseline's MOTA more
than mv-fixed's (since mv-fixed only exposes the detector to ~20% of
frames), which *widened* the relative gap (34.4%->37.8%) rather than
closing it — a clean negative result, not pursued further; see
`findings.md` #15. Tried per-edge scale correction in `propagate_boxes` as
a further lever (shift each box edge by local rather than whole-box median
MV, capturing scale drift with no explicit scale factor) — verified
correct via 5 synthetic scenarios, but measured flat-to-slightly-negative
on real MOT17 (MOTA down in both mv-fixed and mv-adaptive) and reverted;
see findings.md #12. mv-fixed at very short anchor intervals (2-3) also
confirmed the residual gap wasn't a scheduling-frequency problem on its
own — MOTA was flat ~25-26% at interval 2, 3, and 5 alike (pre-`max_age`-fix;
see item 4 for what the flatness actually traced back to).

Also built appearance-based re-identification (`src/mvtrack/track/reid.py`,
`MVTracker(use_appearance=True)`, CLI `--use-reid`) — blends a cosine
similarity embedding into the IoU association cost. First pass (generic
ImageNet-pretrained MobileNetV3-Small) measured flat on HOTA/MOTA/IDF1
despite genuinely influencing ~26% of real assignment decisions (verified
directly, not just inert behind conservative gating) — diagnosed cause:
a classifier backbone encodes "this is a person," not "this is *this*
person." Fixed by sourcing a real ReID-trained model instead of training
anything on MOT17 (small identity pool, real overfitting risk): **OSNet**
pretrained by its original authors on MSMT17 (4101 identities, 15
cameras — an actual ReID benchmark), downloaded from Hugging Face Hub
(`kaiyangzhou/osnet`, reliable, unlike torchreid's own Google Drive-hosted
model zoo). Verified the checkpoint loads with 0 missing/unexpected keys
and is measurably more discriminative on real MOT17 crops (same-vs-diff
identity cosine sim gap 0.511 vs. 0.428) before wiring it in. Real result
this time: HOTA +0.77/+0.19 and IDF1 +1.11/+0.35 on mv-fixed/mv-adaptive,
at an ~11-12% fps cost — kept opt-in (`--use-reid`) rather than default,
since it's a real, but not free, tradeoff, and this project's headline
throughput numbers shouldn't quietly get more expensive by default. See
findings.md #13. Requires `torchreid`/`huggingface_hub` (added to
`pyproject.toml`; `gdown`/`tensorboard` are unused transitive imports
inside torchreid's own package, not used directly). Also hit and resolved
a cosmetic gotcha: Apple's Accelerate BLAS (numpy's default backend on
Apple Silicon) raises spurious divide-by-zero/overflow warnings on some
embedding matmuls; verified output stays correct via a manual dot-product
cross-check, suppressed the warning at the call site (`np.errstate`)
rather than the underlying computation being wrong.

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

**Measured on YOLOv8n with the pre-fix single-pass tracker** (predates
both the detector swap and the re-association fix above) — not rerun
since a bigger/slower detector would only lower these ceilings, not
invalidate the relative baseline-vs-mv-* comparison, and the
re-association fix mainly affects accuracy, not fps. The anchor-interval
ablation in `findings.md` #7 (also YOLOv8n/pre-fix) has since been
superseded by a rerun in #10 that shows the fix resolves the "more
anchors hurt MOTA" inversion; #7 is kept as the finding that motivated
the fix.

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
passwordless sudo (`sudo -n`); the automated sweep's `power_mw` column is
still empty for that reason — a manual one-off measurement was taken
instead (see below), not integrated into the automated sweep since that
needs a NOPASSWD sudoers rule (a system change the user made a judgment
call to skip; the manual path below worked without one). Results are
committed CSVs (`results/`, NOT gitignored like `outputs/`) so the
throughput/energy Pareto plots stay reproducible without rerunning.

## Energy measurement (manual, one-off)

`sudo -n` (used by the automated `bench_multistream.py --power` sweep)
never got passwordless sudo configured, but a manual measurement was still
possible: macOS caches sudo credentials **per-terminal (tty), not
globally** by default, so the credential from a `sudo -v` run in the
user's own terminal never carries over to Claude's separate Bash tool
session — every attempt at scripting the sudo call from this side failed
silently (empty output file, no error) until the user just ran
`sudo powermetrics` directly themselves, redirecting to a file for
Claude to read afterward. A short (15s) sample overlapping a single-video
pipeline run showed close to zero signal (GPU power actually *lower*
during "load" than idle) — likely swamped by `powermetrics` measuring
total system power, not per-process, plus imprecise start-time overlap
between the manually-launched pipeline and the manually-launched sampler.
A long (150s) sample overlapping the full 7-sequence `mv-adaptive` MOT17
eval gave a real, physically-sensible signal:

| Condition | CPU power | GPU power | Combined |
|---|---|---|---|
| Idle (8 samples, 8s) | 9494 mW | 137 mW | 9631 mW |
| mv-adaptive, full MOT17 (150 samples, 150s) | 10157 mW | 505 mW | 10663 mW |
| **Delta** | **+7.0%** | **+2.7x** | **+10.7%** |

GPU power nearly tripling under load is the sanity-check signal that
makes this credible (MPS inference should show up there); the short
sample's backwards GPU reading was the tell that it was noise, not signal.
Caveats worth keeping in mind: `powermetrics` measures *total system*
power (display, background apps, everything), not this process alone, so
the delta is a reasonable but not perfectly clean estimate of the
pipeline's own draw; the idle baseline (8s) is much shorter than the load
sample (150s) and would ideally be re-measured at matching duration; and
this covers `mv-adaptive` only, not a baseline-vs-mv-* energy comparison
(would need the same treatment applied to `baseline` and `mv-fixed` for
the full "streams vs. watts" story the original plan called for). Summary
in `results/energy_measurement.csv`.

## Working conventions

- This is exploratory systems/research code, not a product — favor fast
  iteration and real measurements over up-front test suites. Do add a quick
  correctness check (a synthetic fixture, a known-answer case) whenever a
  new numerical/data-format component is built; that's what caught the
  numpy shim issue above before it silently produced wrong scores.
- Every pipeline variant must go through `eval/run.py` so accuracy and fps
  numbers are directly comparable across the baseline and later MV-based
  variants — don't build one-off benchmarking scripts that skip it.
