"""Cross-checkpoint runner re-identification.

Now pointed at a genuine same-race pair: OKC Memorial Marathon start line
and finish line, same event, same day (2026-04-26), same uploader --
unlike the earlier okc/pikespeak run (two different races, guaranteed zero
overlap, used there only to check the matcher doesn't hallucinate). Here a
true positive is *possible* -- real runners visible at the start are
somewhere in the finish stream -- but not *guaranteed* in these specific
90s windows, which were picked independently, not clock-aligned to any
known individual (chasing an isolated elite-leader frame to get a
guaranteed match was tried and abandoned -- the start camera stays crowded
all the way back past the 8-minute mark, no clean lone-runner moment
available from this angle).

So this run can't be graded against ground truth. What it can do: surface
the top-scoring cross-clip pairs as actual crop images side by side, so a
human can look and judge whether any are plausibly the same runner --
the strongest verification available without bib-level identity data.

Per clip: track with ultralytics' own ByteTrack (same as eval/run.py's
run_baseline -- this step doesn't need mv-tracking's box-propagation, just
clean per-clip identities), then average each track's OSNet/MSMT17 ReID
embedding (mvtrack.track.reid) over several sampled crops for robustness
against single-frame motion blur / partial occlusion.
"""

import pathlib

import cv2
import numpy as np
from ultralytics import YOLO

from mvtrack.detect import pick_device
from mvtrack.track.reid import ReIDEmbedder

PERSON_CLS = 0
MAX_CROPS_PER_TRACK = 5
# Median box size on these wide elevated race-cams is ~37x37px (measured
# directly), upscaled 5-8x to OSNet's 256x128 input -- real detail loss vs.
# the MOT17 crops the same model was validated on in findings.md #13.
# Gating out sub-50x50 boxes tests whether that resolution gap, not a
# tracking bug, is what's driving elevated cross-clip similarity below.
MIN_BOX_AREA = 2500  # 50x50px
# reid.py's own docstring: same-identity cosine sim ~0.511, different-identity
# ~0.428 on real MOT17 crops. Every pair here is different-identity by
# construction (different races) -- a score above this ballpark would be a
# real false-positive risk (e.g. matching outfit colors), not a recognized
# person.
FALSE_POSITIVE_THRESHOLD = 0.5

REPO_ROOT = pathlib.Path(__file__).resolve().parent.parent
CLIPS = {
    "okc_start": REPO_ROOT / "data" / "checkpoint_start_okc.mp4",
    "okc_finish": REPO_ROOT / "data" / "checkpoint_finish_okc.mp4",
}
CROPS_DIR = REPO_ROOT / "outputs" / "checkpoint_reid_crops"
TOP_N_TO_SAVE = 10


def extract_track_embeddings(video_path, reid, weights="yolov8s.pt"):
    model = YOLO(weights)
    device = pick_device()
    track_embs = {}
    track_best_crop = {}  # tid -> (area, crop_bgr) -- largest crop seen, for visual review
    for res in model.track(
        source=str(video_path), device=device, tracker="bytetrack.yaml",
        stream=True, verbose=False, conf=0.25, classes=[PERSON_CLS], persist=True,
    ):
        if res.boxes.id is None:
            continue
        ids = res.boxes.id.int().tolist()
        xyxy = res.boxes.xyxy.cpu().numpy()
        img = res.orig_img
        for tid, box in zip(ids, xyxy):
            area = (box[2] - box[0]) * (box[3] - box[1])
            if area < MIN_BOX_AREA:
                continue
            bucket = track_embs.setdefault(tid, [])
            if len(bucket) < MAX_CROPS_PER_TRACK:
                bucket.append(reid(img, box[None, :])[0])
            best = track_best_crop.get(tid)
            if best is None or area > best[0]:
                x0, y0, x1, y1 = box.astype(int)
                track_best_crop[tid] = (area, img[max(y0, 0):y1, max(x0, 0):x1].copy())

    embeddings = {
        tid: (lambda e: e / (np.linalg.norm(e) + 1e-8))(np.mean(embs, axis=0))
        for tid, embs in track_embs.items()
    }
    crops = {tid: crop for tid, (_, crop) in track_best_crop.items()}
    return embeddings, crops


def main():
    reid = ReIDEmbedder(device=pick_device())
    per_clip = {}
    per_clip_crops = {}
    for name, path in CLIPS.items():
        print(f"tracking + embedding {name}...")
        embs, crops = extract_track_embeddings(path, reid)
        print(f"  {len(embs)} tracks")
        per_clip[name] = embs
        per_clip_crops[name] = crops

    names = list(CLIPS.keys())
    a_ids, a_embs = zip(*per_clip[names[0]].items())
    b_ids, b_embs = zip(*per_clip[names[1]].items())
    A, B = np.stack(a_embs), np.stack(b_embs)
    # Apple Accelerate BLAS raises spurious divide-by-zero/overflow warnings on
    # some normalized-embedding matmuls (verified cosmetic: fires even on
    # synthetic random unit vectors, output matches manual dot product to fp32
    # precision -- see mvtrack/track/reid.py's own note on the same issue).
    with np.errstate(all="ignore"):
        sim = A @ B.T

    print(f"\ncross-checkpoint similarity: {sim.shape[0]} {names[0]} tracks x "
          f"{sim.shape[1]} {names[1]} tracks")
    print(f"mean={sim.mean():.3f}  max={sim.max():.3f}  min={sim.min():.3f}")

    flat = sorted(
        ((sim[i, j], a_ids[i], b_ids[j]) for i in range(sim.shape[0]) for j in range(sim.shape[1])),
        reverse=True,
    )
    risky = [t for t in flat if t[0] > FALSE_POSITIVE_THRESHOLD]
    print(f"\npairs above {FALSE_POSITIVE_THRESHOLD} "
          f"(known-same-identity ballpark from findings.md #13): {len(risky)}")

    print(f"\nsaving top {TOP_N_TO_SAVE} pairs as side-by-side crops to {CROPS_DIR} "
          "for visual review -- similarity alone can't confirm a real match, a human "
          "checking the actual crops can:")
    CROPS_DIR.mkdir(parents=True, exist_ok=True)
    for rank, (score, aid, bid) in enumerate(flat[:TOP_N_TO_SAVE]):
        crop_a = per_clip_crops[names[0]][aid]
        crop_b = per_clip_crops[names[1]][bid]
        h = max(crop_a.shape[0], crop_b.shape[0], 128)
        pad = lambda c: cv2.copyMakeBorder(
            cv2.resize(c, (int(c.shape[1] * h / c.shape[0]), h)),
            0, 0, 0, 0, cv2.BORDER_CONSTANT)
        combined = np.hstack([pad(crop_a), np.full((h, 10, 3), 255, np.uint8), pad(crop_b)])
        out_path = CROPS_DIR / f"{rank:02d}_sim{score:.3f}_{names[0]}{aid}_{names[1]}{bid}.png"
        cv2.imwrite(str(out_path), combined)
        print(f"  [{rank}] {names[0]} track {aid} <-> {names[1]} track {bid}: "
              f"{score:.3f} -> {out_path.name}")


if __name__ == "__main__":
    main()
