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

- **Follow-on accuracy pass** (after the 14-week plan was "done," user
  pushed back that real accuracy was still bad): detector ceiling check
  (YOLOv8n → YOLOv8s, adopted), fixed `MVTracker.step_anchor`'s
  re-association (three-stage ByteTrack-style match — eliminated a real
  bug where more anchors were making MOTA *worse*), and tuned the
  never-tuned `Adaptive` scheduler against real MOTA (grid search, new
  defaults adopted). See findings.md #9-11 for full detail. Combined
  effect: mv-fixed's MOTA gap to baseline roughly halved (~22pts → ~12.5).

Everything through the follow-on accuracy pass is committed (18 commits on
`main`). Everything from the application-exploration pass onward
(tennis/marathon/plaza apps, PULSE, the multi-camera generalization, the
metric expansion, and the global scheduler's Phases A-B) is **not yet
committed** as of this handoff — see git status, working tree is not
clean.

- **Application-exploration pass** (after the follow-on accuracy pass):
  built real downstream applications on top of the finished tracker and
  reported honest results on each, not just the wins. Tennis positioning
  analytics (real court homography + opponent-relative "bisector"
  recovery stat, two real bugs caught and fixed by checking output).
  A real negative result on when throughput actually matters (single-court
  tennis already keeps up live on baseline — the multi-stream case is
  where this project's real advantage lives, not single-stream). Marathon
  cross-checkpoint re-identification on real same-event footage — a
  numerically-plausible result that visual inspection of the actual crop
  pairs revealed was a clothing-color false positive, not identity (see
  findings.md #16). A real detector-floor finding on extreme-elevation
  crowd cameras (findings.md #17), diagnosed and resolved by picking a
  better-suited site rather than continuing to tune around a mismatch.

- **PULSE multi-camera generalization pass**: the WildTrack download
  finished and got built into a real 7-camera fusion demo, but the user
  correctly flagged that everything up to that point (WildTrack, then
  EPFL-RLC) was hardcoded for one specific camera count/pair (`if cam == 2
  / elif cam == 0 / else raise` in the original `register_to_cam0()`).
  Replaced with `src/mvtrack/court/multicam.py`: `register_cameras()` (ORB
  + RANSAC-homography correspondences, Kabsch/Procrustes rigid fit,
  automatic include/exclude by residual) and `fuse_multicam_points()` — a
  real N-camera generalization, not a relabeled 2-camera result. Verified
  by running it against three structurally different real cases and
  checking it produced the right answer for each without any
  camera-count-specific code: EPFL-RLC (cam1 correctly excluded, cam2
  correctly included at 0.76m residual — reproduces the earlier manual
  finding automatically), EPFL CVLAB's 4-camera Laboratory sequence (all 4
  cameras already share one calibrated frame by construction — correctly
  recognized as already-aligned, 10-18.5cm residual, nothing excluded —
  the complementary case to EPFL-RLC), and CAVIAR's `OneShopOneWait1`
  cor/front pair (genuinely disjoint viewpoints, too little shared
  texture — correctly flagged as a bad fit, 126.9cm residual, a third
  distinct real outcome).

  Also searched hard for a real calibrated *museum* multi-camera dataset
  (the user's original ask) and found none exists publicly — checked
  directly, not assumed: MICC's MuseumVisitors (3 cams, Bargello) ships no
  calibration at all and is 97%-unlabeled for exhibit attribution; no
  other candidate turned up a calibrated museum alternative either. Used
  EPFL CVLAB's Laboratory sequence as the closest real analog for
  validating the generalization module instead, honestly framed as an
  analog, not a museum result.

- **PULSE metric expansion + honest limitation-fixing pass**: dwell-time
  detection alone read as generic CV analytics, not something that solves
  a real problem — the user pushed for (1) verified real data before any
  new build, and (2) at least one metric that's a genuine category shift
  from "person present here how long." Landed on five real metrics, all
  built on a new shared `src/mvtrack/analytics/` package (`WorldTracker`
  moved out of `epfl_rlc_fusion.py` into `src/mvtrack/track/world_tracker.py`
  where it belongs as core infra, since three-plus fusion scripts had
  started copy-pasting it):
  - **Capture rate** (`capture_rate.py`) — stopped/passersby, not just raw
    dwell count. Validated on CAVIAR `OneShopOneWait1`: 1 stop / 19
    passersby, spot-checked against real frames.
  - **MV-energy** (`mv_energy.py`) — zone "bustle" straight from raw
    `FrameMV.mv_grid`, no detector at all. The one on-brand metric (this
    project's whole premise is signal-in-the-codec). Combined z-score
    correctly ranked a real busy 3-person stretch (+1.52) above a real
    empty stretch (-0.55).
  - **Approach dynamics** (`approach_dynamics.py`) — deceleration/
    "window-shopping" classification from the position history
    `WorldTracker` already stores. Found and visually confirmed a real
    person on EPFL Lab footage who slowed to a near-standstill near the
    rug without ever meeting the dwell threshold — a case the binary
    dwell classifier would have missed entirely.
  - **Group/companion dwell** (`group_dwell.py`) — real detour: CAVIAR's
    own dedicated "meeting" scenario (`Meet_WalkTogether1`, INRIA lobby)
    sits on a wide-angle fisheye camera that breaks YOLOv8s completely
    (confirmed directly: mostly COCO "bird" hallucinations, person
    confidence noise-level). EPFL Lab (room too small — any two occupants
    register as "close") and EPFL-RLC (courtyard too sparse — only
    matched at absurd 20m thresholds) were both ambiguous too. Real fix:
    CAVIAR's own shop footage already has a real companion pair in it
    (its own scenario blurb says so) — reused the already-clean `cor` view
    and found it directly: two men walking side by side, 100% together
    for 5.8s, visually confirmed.
  - **Loitering / left-object** (`loitering.py`) — real security use case.
    Same INRIA fisheye problem killed CAVIAR's own `LeftBag` scenario
    (checked directly before writing any alert logic, per this project's
    own "verify feasibility first" discipline). Switched to ABODA
    (`github.com/kevinlin311tw/ABODA`, a real non-fisheye abandoned-object
    dataset) and found a real, useful bug along the way: the object was
    *carried* before being set down, so checking stationarity from its
    first detection never fired. Added `find_stationary_suffix()` — finds
    the "placed" phase of a track instead of assuming stillness from
    frame one — a genuine improvement to the module, not a hack. Also
    swapped `MVTracker` (built for moving targets) for a purpose-built
    `StaticObjectTracker` in `scripts/aboda_leftbag.py`, since sparse
    low-confidence bag detections were starving MVTracker's grace period
    faster than the real gaps warranted. Final result: correctly flagged
    the bag as abandoned at the real frame the owner had actually walked
    away, visually confirmed.

  Synthetic correctness checks for loitering logic in `tests/test_loitering.py`
  (real video validated feasibility separately, per above); `tests/`
  didn't exist before this pass.

- **Global cross-stream compute-budget scheduler — built, real
  infrastructure, negative result on the accuracy hypothesis (stopped
  after Phase C, by deliberate choice)**: the systems-flavored half of
  making this project read as more than a CV demo. Today's
  `run_multistream` runs N fully independent processes, each with its own
  `Adaptive` scheduler in total isolation — no shared awareness that one
  camera might be busy while another is empty. Built a shared arbiter that
  reallocates a fixed detector-call budget by real per-stream urgency
  instead. Full design (rejected alternatives, IPC math, phased plan) at
  `~/.claude/plans/sparkling-sauteeing-boole.md`; full real numbers and
  root-cause writeup in findings.md #18.
  - **Phase A (done)**: `Adaptive.urgency(fmv) -> float` exposes the spike
    ratio the scheduler already computed internally. Verified
    byte-identical behavior via a real before/after stash-and-rerun of
    `mv-adaptive` on MOT17-02 — anchor rate and every HOTA/CLEAR/Identity
    number matched exactly.
  - **Phase B (done)**: `src/mvtrack/sched/global_budget.py` —
    `BudgetArbiter`, a pure policy class (rate mode + total-budget mode),
    zero process/IPC logic so the same policy drives both the offline
    accuracy experiment and the live demo without risking drift between
    them. 6 unit tests in `tests/test_budget_arbiter.py`. Real (not
    estimated) `multiprocessing.Queue` throughput measurement: 79,756
    msgs/sec sustained, 32 bytes/message — 399x headroom over the actual
    target rate (8 streams × 25fps), confirming the score-only IPC design
    (never ships frame pixels) doesn't bottleneck.
  - **Phase C (done — real negative result)**:
    `src/mvtrack/sched/global_replay.py` + `scripts/
    run_global_budget_experiment.py`, offline trace-replay against real
    MOT17, scored through the existing TrackEval-backed `eval/run.py`. Two
    real bugs caught before trusting any result (a too-loose
    `budget_per_tick` default that made the arbiter a silent no-op; a flat
    `urgency=0.0` naive baseline that collapsed to deterministic
    stream-id favoritism under Python's stable sort — fixed with a
    seeded random tie-break). Once both were fixed and genuine scarcity
    was imposed (60% of independent's natural anchor total), urgency-aware
    reallocation didn't beat a fair naive-sharing baseline at either 4 or
    7 concurrent MOT17 streams (a wash at 4, slightly worse at 7 — see
    findings.md #18 for the full table). Root-caused via a direct
    tick-contention profile: genuine multi-stream contention (the only
    situation urgency ranking can matter in) occurs in just 9-15% of
    ticks. Considered and deliberately declined chasing a different
    (correlated-multi-camera) dataset next — EPFL-RLC/CAVIAR only offer
    2-3 real concurrent streams (fewer than the 7-stream test that already
    made things worse) and would run under `Adaptive` defaults tuned
    specifically for MOT17, adding a confound — two independent,
    mechanistically-explained replications was judged sufficient to stop
    and report, the same discipline already applied to CorrectionNet
    (#4-5) and appearance ReID on marathon footage (#16).
  - **Not built**: Phase D (live multiprocess arbiter) and Phase E (live
    throughput demo) — by explicit decision, since building a live demo of
    a mechanism the accuracy experiment couldn't validate would be
    building on top of a result that didn't hold up, not validating it
    further. If Part 2 is picked back up, the honest framing is a pure
    throughput/systems demo with no accuracy claim riding on it, not a
    rescue attempt for #18's result.
  - **New tests**: `tests/test_global_replay.py` (4 tests, regression
    guards for both real bugs above) alongside `tests/
    test_budget_arbiter.py`.

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
- `tests/` now exists (`test_loitering.py`, `test_budget_arbiter.py`) —
  synthetic correctness checks for new logic components, same spirit as
  the ad hoc sanity scripts elsewhere, just committed this time since the
  global-scheduler plan calls for them explicitly. `pytest` isn't
  installed in `.venv` yet; both files also run standalone via
  `python <file>.py` in the meantime.
- **Real datasets now on disk beyond MOT17/WildTrack/EPFL-RLC**:
  `data/epfl_lab/` (EPFL CVLAB 4-camera Laboratory sequence, homography
  calibration), `data/caviar_shop/` (CAVIAR `OneShopOneWait1` cor+front,
  real published pixel↔world correspondence points), `data/caviar_meet/`
  (`Meet_WalkTogether1`, INRIA fisheye camera — detector-broken, kept for
  the companion-pair mechanism even though the video itself isn't usable
  for detection), `data/caviar_leftbag/` (`LeftBag.mpg`, same fisheye
  problem, superseded by `data/aboda/` for the actual loitering
  validation). All re-encoded to baseline-profile H.264 under each
  dataset's own `videos/` subfolder, same convention as MOT17.
- **`src/mvtrack/analytics/`**: shared PULSE infrastructure (`Zone`,
  `DwellParams`/`track_and_classify_dwells`, `capture_rate`, `mv_energy`,
  `approach_dynamics`, `group_dwell`, `loitering`) — import from here
  rather than from another fusion script, that's the whole point of the
  consolidation.
- **`src/mvtrack/sched/global_budget.py`**: `BudgetArbiter`/`Request`/
  `Decision` — pure policy, no processes yet (Phase D).

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
- **YOLOv8s over YOLOv8n as the project-wide default** — a direct ceiling
  check showed YOLOv8n was leaving real accuracy on the table across every
  pipeline; adopted rather than just noted, with `--weights` kept as a CLI
  override for comparison.
- **Three-stage ByteTrack-style re-association over the original
  single-pass IoU match** in `MVTracker.step_anchor` — the single-pass
  version had a real, measured bug (more anchors making MOTA worse); fixed
  with high-conf/low-conf/grace-period stages rather than just tuning the
  existing IoU threshold, since the root cause was structural (no recovery
  path for a missed detection), not a threshold miscalibration.
- **Grid search over hand-picking scheduler defaults** — `Adaptive`'s four
  params had never been tuned at all; a small grid search against real
  MOTA (not a proxy) on 2 held-out-from-final-reporting sequences found
  the winner cheaply rather than guessing.
- **General N-camera registration (`multicam.py`) over more
  dataset-specific hardcoding** — the user's real objection was that
  every earlier fusion script assumed a fixed camera count/pair; fixed at
  the module level (ORB+RANSAC+Kabsch, automatic include/exclude) rather
  than adding a fourth special case.
- **No calibrated museum dataset exists — used the closest real analog
  instead of fabricating one.** Checked directly (MuseumVisitors has no
  calibration, 97% unlabeled) rather than assumed; reported the negative
  finding and picked EPFL CVLAB's Laboratory sequence for validating the
  generalization module on its own real merits (already-shared frame,
  25fps, real people), explicitly not framed as museum data.
  Note: MuseumVisitors's own download completed but was **not** kept —
  found genuinely unusable (no calibration, 745 corrupted frames in one
  session, 97% of annotations unlabeled) and deleted to reclaim disk
  space rather than left half-integrated.
- **Reused existing footage over chasing a broken camera further** (group
  dwell) — once CAVIAR's dedicated "meeting" scenario turned out to sit on
  a fisheye camera YOLOv8s can't handle, the fix was checking whether the
  ALREADY-working shop footage had the same real behavior in it (it did),
  not further fisheye-correction attempts on unreliable data.
  `find_stationary_suffix()` over assuming stationarity from frame one
  (loitering) — a real object-tracking bug found via actual ABODA
  footage (objects are often carried before being placed), fixed at the
  algorithm level since it's a genuinely common real-world case, not
  patched around with a one-off parameter tweak.
- **Purpose-built `StaticObjectTracker` over reusing `MVTracker` for
  bag-class objects** (`scripts/aboda_leftbag.py`) — `MVTracker`'s
  Hungarian+velocity design is for moving targets; sparse, low-confidence
  bag detections kept starving its grace period faster than the real
  detection gaps warranted. A simple same-approximate-location
  association is the right tool for something that, by definition, isn't
  moving.
- **Global scheduler arbitrates on small numeric scores, never ships
  frame pixels between processes** — real math, not just intuition: a
  1080p frame is ~5.9MB, and centralizing frame transport at realistic
  multi-stream anchor rates would run ~380MB/s through one
  `multiprocessing.Queue`, a real bottleneck that would undo the
  concurrency win this project already measured. Score-only messages
  measured at 79,756 msgs/sec / 32 bytes each — confirmed directly, not
  assumed, before committing to the design.
- **One shared `BudgetArbiter` policy class reused by both the offline
  replay (Phase C) and the live arbiter process (Phase D)**, rather than
  two separate implementations — keeps the "does this help accuracy"
  experiment and the "does this work live" demo honest against each
  other; they can't quietly diverge into different policies.

## Touched Files

Committed (18 commits, through the follow-on accuracy pass):

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
  tune_scheduler.py                        # grid-search Adaptive's params against real MOTA
results/
  bench_multistream_{baseline,mv-fixed,mv-adaptive}.csv
  ablation_anchor_interval.csv, pipeline_comparison.csv, scheduler_tune.csv
  plots/{pipeline_pareto,ablation_anchor_interval,multistream_scaling}.png
  demo_side_by_side.mp4
handoff.md, findings.md                    # this file and the metrics writeup
```

Application-exploration + PULSE + metric-expansion + global-scheduler
passes (not yet committed — see git status; full narrative in
`.claude/CLAUDE.md`):

```
src/mvtrack/
  court/
    homography.py                          # pixel <-> real-world court-meters, tennis
    multicam.py                            # general N-camera registration (ORB+RANSAC+Kabsch)
  track/world_tracker.py                   # WorldTracker, moved out of epfl_rlc_fusion.py
  analytics/                               # shared PULSE infra + the 5 new metrics
    zones.py, dwell.py, capture_rate.py, mv_energy.py,
    approach_dynamics.py, group_dwell.py, loitering.py
  sched/global_budget.py                   # BudgetArbiter, Request, Decision (Phase B)
scripts/
  court_positioning.py                     # opponent-relative "bisector" recovery stat
  live_feasibility.py                      # real-time keep-up test, single-stream negative result
  checkpoint_reid.py                       # cross-checkpoint appearance re-id (findings.md #16)
  plaza_dwell.py                           # single-camera dwell/lingering, "PULSE" (findings.md #17)
  epfl_rlc_fusion.py, epfl_rlc_visualize.py  # real 60fps 3-cam fusion (updated for multicam.py)
  epfl_lab_fusion.py                       # 4-cam Laboratory sequence, generalization validation
  caviar_shop_fusion.py                    # OneShopOneWait1 cor+front: dwell, capture rate, companions
  caviar_meet_fusion.py                    # Meet_WalkTogether1 (fisheye-limited, see Key Decisions)
  aboda_leftbag.py                         # real abandoned-object validation, StaticObjectTracker
tests/
  test_loitering.py, test_budget_arbiter.py
data/
  tennis_video.mp4, checkpoint_start_okc.mp4, checkpoint_finish_okc.mp4,
  checkpoint_start_pikespeak.mp4, plaza_ts_{crossroads,north}.mp4,
  plaza_bryant.mp4, wildtrack/, epfl_rlc/, epfl_lab/,
  caviar_shop/, caviar_meet/, caviar_leftbag/, aboda/
outputs/checkpoint_reid_crops/              # top-10 crop pairs saved for visual verification
outputs/plaza_dwell/dwell_overlay.png       # visual check of dwell classification
outputs/epfl_rlc_viz/, epfl_lab_viz/, caviar_shop_viz/  # rendered tracking/dwell overlay videos
~/.claude/plans/sparkling-sauteeing-boole.md  # PULSE metric expansion + global scheduler plan
```

Not committed (gitignored, regeneratable): `data/`, `outputs/`
(checkpoints, per-run result txts, plots), `.venv/`, `*.egg-info/`.
`results/tennis_video.mp4` is a real exception worth knowing about: it's
121MB third-party broadcast footage sitting in the committed `results/`
directory (not gitignored `data/`) from before this convention was fully
settled — explicitly excluded from every commit made so far in this
project, still untracked on disk.

## Blockers

- **CAVIAR's INRIA-lobby camera (`Meet_*`, `LeftBag*`) is fisheye and
  breaks YOLOv8s entirely** — not a bug in this project's code, a real
  property of that specific camera. Confirmed directly (conf=0.05 still
  returns mostly COCO "bird"/"clock" hallucinations, person confidence
  crashes to noise level). Worked around per-metric (see Key Decisions) —
  not something to keep retrying against.
- **No calibrated museum multi-camera dataset exists publicly** — checked
  directly, not assumed (see Key Decisions). Not a blocker on work
  already done (EPFL Lab covers the generalization-validation need), but
  worth knowing before promising a "real museum" result to anyone.
- `powermetrics` energy sampling needs passwordless `sudo` (`sudo -n`),
  not configured in this environment — this is a system/sudoers change
  the user should make deliberately if energy numbers are wanted, not
  something to script around.
- The standing MOT17 blocker (`motchallenge.net` down) is resolved via the
  Kaggle mirror. The WildTrack download blocker from the previous version
  of this handoff is also resolved — it finished and was built into a
  real fusion demo before the pivot to EPFL-RLC's higher framerate.

## Next Steps

**The global cross-stream scheduler is done through Phase C, with a real
negative result** — see findings.md #18 for the full writeup. Phases D/E
(live multiprocess arbiter, live throughput demo) were deliberately not
built: doing so would demo a mechanism the accuracy experiment couldn't
validate, not validate it further. Not "still to do" — a closed, reported
result. If picked back up later, it should be framed as a pure
throughput/systems demo with no accuracy claim, not a rescue attempt.

**Active**: commit everything from the application-exploration pass
onward. Nothing has been pushed since the follow-on accuracy pass's 18
commits — that now includes the full PULSE/multicam generalization
pass, the 5-metric expansion, and the global scheduler's Phases A-C.

Both the 14-week plan and the follow-on accuracy pass are complete. What's
left there is optional polish, not core scope:

- **Optional**: `mv-learned`'s CorrectionNet still slightly trails
  mv-fixed on every metric even after re-testing on top of both fixes
  (findings.md #4-5, #10, #11 don't change this verdict) — would need
  appearance features or a confidence gate to plausibly close, not
  attempted.
- **Optional**: improve `propagate_boxes` itself (scale correction for
  subjects moving toward/away from camera, robust MV statistics instead
  of plain median) — explicitly out of scope for the follow-on pass, the
  next lever if more accuracy is wanted without new modalities.
- **Optional**: set up `powermetrics` passwordless sudo if real energy
  numbers (not just fps) are wanted for the multi-stream Pareto plot.
- **Optional**: extend the ablation to UA-DETRAC (vehicles, per the
  original plan's dataset list) if a second domain is wanted for
  robustness — not attempted, MOT17 alone was the whole plan's dataset in
  practice.
- **Optional**: multi-stream and anchor-interval-ablation numbers in
  `results/` still reflect the pre-fix tracker/YOLOv8n — rerunning them
  would give a fully consistent picture, but wasn't required since the
  qualitative comparisons (baseline vs mv-*) still hold.
- **Known rough edge**: a hung multi-stream sweep process once occurred
  (see CLAUDE.md gotcha) — not root-caused, workaround (kill + rerun) is
  documented, low priority unless it recurs.
