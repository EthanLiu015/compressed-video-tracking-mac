"""Re-encode WildTrack's per-camera 2fps PNG sequences into real H.264
bitstreams -- what actually makes this data usable by mvtrack.

WildTrack ships pre-decoded frames, not a compressed bitstream, so
MVTracker has nothing to propagate on as-is (see conversation: the
WildTrack-based fusion work built earlier used generic YOLO+ByteTrack,
not the project's actual compressed-domain pipeline, at all). Re-encoding
restores a real bitstream to extract motion vectors from.

Baseline-profile H.264, no B-frames (`-bf 0`), same convention as
`scripts/get_test_clip.py`/`scripts/prep_mot17.py` -- MV propagation
hasn't been validated on backward-referencing vectors. Input frame rate
set to 2 (matches WildTrack's own annotated-subset sampling, confirmed
directly from the paper: 400 frames GT'd at 2fps, subsampled from a 10fps
extraction we deliberately did *not* download given a tight local disk
budget -- see conversation).

Deliberately reuses only the already-downloaded 2fps PNGs already on
disk -- zero new downloads.
"""

import pathlib
import subprocess

REPO_ROOT = pathlib.Path(__file__).resolve().parent.parent
IMG_ROOT = REPO_ROOT / "data" / "wildtrack" / "Wildtrack_dataset" / "Image_subsets"
OUT_DIR = REPO_ROOT / "data" / "wildtrack" / "videos"
INPUT_FPS = 2


def reencode_camera(cam: int):
    src_glob = str(IMG_ROOT / f"C{cam}" / "*.png")
    out_path = OUT_DIR / f"C{cam}.mp4"
    cmd = [
        "ffmpeg", "-y",
        "-framerate", str(INPUT_FPS),
        "-pattern_type", "glob", "-i", src_glob,
        "-c:v", "libx264", "-profile:v", "baseline", "-bf", "0",
        "-g", "30", "-crf", "23", "-pix_fmt", "yuv420p",
        str(out_path),
    ]
    subprocess.run(cmd, check=True, capture_output=True)
    return out_path


def main():
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    for cam in range(1, 8):
        out_path = reencode_camera(cam)
        size_mb = out_path.stat().st_size / 1e6
        print(f"C{cam} -> {out_path} ({size_mb:.1f} MB)")


if __name__ == "__main__":
    main()
