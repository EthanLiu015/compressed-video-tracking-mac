# mvtrack — Session Handoff

Compressed-domain video analytics: MOT tracking that propagates boxes via
codec motion vectors instead of full decode+detect on every frame. Full
plan: `~/.claude/plans/breezy-knitting-dragon.md` (14-week, Mac-only,
one-semester scope). Project conventions and gotchas: `.claude/CLAUDE.md`
(read that first — this file summarizes it, CLAUDE.md is the live source
of truth going forward).

## Accomplishments

**All 14 weeks of the plan are done.**

- **Weeks 1-2**: repo scaffold, PyAV motion-vector extraction (bitstream →
  MV grid, no full pixel reconstruction needed beyond what FFmpeg does for
  `export_mvs`), YOLOv8+ByteTrack full-decode baseline, TrackEval
  HOTA/CLEAR/Identity scoring harness (`eval/run.py`).
- **Weeks 3-4** (folded into the above): MV extraction verified against a
  real clip (95% P-frame occupancy), overlay visualizer
  (`scripts/mv_demo.py`).
- **Weeks 5-7**: `MVTracker` (IoU/Hungarian re-association on anchor
  frames + MV propagation between them), wired as the `mv-fixed` pipeline.
  Adaptive anchor scheduler (`sched/scheduler.py`, intra-block-fraction +
  scene-cut proxy for residual energy) pulled forward from weeks 11-12
  since it didn't depend on MOT17 access. **Got real MOT17 ground truth**
  (Kaggle mirror, after `motchallenge.net` stayed down across 4+ sessions)
  and ran the first genuine accuracy/throughput comparison — see Findings.
- **Weeks 8-10**: learned MV-domain correction net (`CorrectionNet`), a
  stretch goal. Built, trained, diagnosed a real train/inference
  mismatch, fixed it (DAgger-style rollout training), and validated the
  fix measurably helped without fully closing the gap. Honest negative
  result, documented rather than hidden.
- **Weeks 11-12**: multi-stream throughput harness
  (`scripts/bench_multistream.py`), sweeping concurrent pipeline instances
  on the M4 until per-stream fps drops below a 25fps target. Pinned all
  three pipelines' max-streams ceiling (baseline 4, mv-fixed 8, mv-adaptive
  8 — mv-fixed's was left unresolved earlier, now confirmed).
- **Weeks 13-14**: anchor-interval ablation sweep (mv-fixed at intervals
  2/3/5/8/10 — surfaced a non-obvious finding, see findings.md #7),
  Pareto/ablation/multistream plots (`scripts/make_plots.py`, in
  `results/plots/`), README rewritten with final results and full usage,
  and a side-by-side demo video (`scripts/make_demo_video.py`,
  `results/demo_side_by_side.mp4`) — baseline vs. mv-adaptive, same
  detector and tracker logic, anchor frames highlighted.

Everything is committed; 14 commits on `main`, working tree clean as of
this handoff.

## Current State

- **Real MOT17 data is in place**: `data/MOT17/` (train GT + re-encoded
  H.264 videos for all 7 FRCNN train sequences), via the Kaggle mirror
  (`~/.kaggle/kaggle.json` is set up on this machine).
- **Four eval pipelines** registered in `eval/run.py`: `baseline`,
  `mv-fixed`, `mv-adaptive`, `mv-learned`. All go through the same
  TrackEval-backed harness so numbers are directly comparable.
- **Two trained CorrectionNet checkpoints** in `outputs/` (gitignored):
  `correction_net.pt` (v1, single-step training) and
  `correction_net_v2.pt` (DAgger rollout training, the better one — not
  currently the default path baked into `eval/run.py`'s
  `--correction-checkpoint`, which still defaults to
  `correction_net.pt`/v1's filename; pass `--correction-checkpoint
  correction_net_v2.pt` explicitly, or rename v2 over the default if you
  want it to become the standard).
- **Multi-stream results** committed under `results/*.csv` (not
  gitignored, unlike `outputs/`) for the three fast-enough-to-benchmark
  pipelines.
- No test suite in the conventional sense — this is exploratory systems
  research code; correctness is checked via targeted synthetic
  fixtures/sanity scripts at each new numerical component (see
  CLAUDE.md's "Working conventions").

## Key Decisions

- **Mac-only, PyTorch MPS, no Core ML export yet** — deferred as a later
  optimization, not needed to get real numbers.
- **Kaggle over motchallenge.net** for MOT17 — the official host has been
  server-side unreachable across 4+ sessions; Kaggle mirror matches the
  official directory layout exactly, no code changes needed.
- **Isolate propagation error from detector error** when building
  CorrectionNet training data — training pairs come from GT box → GT box,
  never from actual detector output, so the correction net's failure mode
  is attributable to the MV signal, not detector noise.
- **DAgger-style on-policy retraining over single-step supervision** once
  the mismatch was diagnosed — re-walk real anchor windows with the
  current checkpoint to generate targets, rather than tuning
  architecture/hyperparameters on the same flawed data distribution.
- **Separate processes, not threads, for multi-stream benchmarking** —
  PyTorch/YOLO inference holds the GIL enough that threads wouldn't show
  real concurrency, and MPS device contexts are safer one-per-process.
- **Committed `results/` directory** distinct from gitignored `outputs/`
  — benchmark CSVs are meant to be reproducible artifacts per the plan's
  Verification section, not scratch output.
- **Security**: user's Kaggle API token was moved to the standard
  `~/.kaggle/kaggle.json` location (not left in-repo even gitignored) and
  `kaggle.json` added defensively to `.gitignore`.

## Touched Files (all committed)

```
.claude/CLAUDE.md                          # live project doc — READ THIS, most detail lives here
.gitignore                                 # + kaggle.json
pyproject.toml, README.md
eval/run.py                                # 4 pipelines: baseline, mv-fixed, mv-adaptive, mv-learned
src/mvtrack/
  extract/mv_extract.py                    # PyAV MV extraction, vectorized grid-build
  detect/yolo.py                           # YOLO/MPS wrapper
  track/
    propagate.py                           # MV-median box propagation
    tracker.py                             # MVTracker: IoU/Hungarian + propagate, correct_boxes()
    correct.py                             # CorrectionNet (learned box correction)
  sched/scheduler.py                       # FixedInterval, Adaptive schedulers
  bench/harness.py                         # multi-stream N-process runner
  mot_gt.py                                # shared MOT17 gt.txt loader
scripts/
  get_test_clip.py, mv_demo.py, baseline_smoke.py, prep_mot17.py
  build_correction_dataset.py              # single-step GT propagation-error pairs
  build_rollout_dataset.py                 # on-policy DAgger rollout pairs
  train_correction.py                      # trains CorrectionNet on 1+ datasets
  bench_multistream.py                     # multi-stream sweep + optional powermetrics
  make_plots.py                            # regenerates results/plots/*.png from results/*.csv
  make_demo_video.py                       # side-by-side baseline vs mv-adaptive demo
results/
  bench_multistream_{baseline,mv-fixed,mv-adaptive}.csv
  ablation_anchor_interval.csv, pipeline_comparison.csv
  plots/{pipeline_pareto,ablation_anchor_interval,multistream_scaling}.png
  demo_side_by_side.mp4
handoff.md, findings.md                    # this file and the metrics writeup
```

Not committed (gitignored, regeneratable): `data/`, `outputs/`
(checkpoints, per-run result txts, plots), `.venv/`, `*.egg-info/`.

## Blockers

- **None currently active.** The standing MOT17 blocker (`motchallenge.net`
  down) is resolved via the Kaggle mirror.
- `powermetrics` energy sampling needs passwordless `sudo` (`sudo -n`),
  not configured in this environment — this is a system/sudoers change
  the user should make deliberately if energy numbers are wanted, not
  something to script around.

## Next Steps

The 14-week plan is complete. What's left is optional polish, not core scope:

- **Optional**: revisit CorrectionNet with appearance features or a
  per-track confidence gate if the accuracy gap vs. mv-fixed is worth
  closing — current read is that MV/occupancy-only features are
  under-powered for this correction task (see findings.md #4-5).
- **Optional**: set up `powermetrics` passwordless sudo if real energy
  numbers (not just fps) are wanted for the multi-stream Pareto plot.
- **Optional**: extend the ablation to UA-DETRAC (vehicles, per the
  original plan's dataset list) if a second domain is wanted for
  robustness — not attempted, MOT17 alone was the whole plan's dataset in
  practice.
- **Known rough edge**: a hung multi-stream sweep process once occurred
  (see CLAUDE.md gotcha) — not root-caused, workaround (kill + rerun) is
  documented, low priority unless it recurs.
