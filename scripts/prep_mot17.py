"""Unzip MOT17 and re-encode train sequences to H.264 videos.

MOT17 ships as JPEG image sequences; the whole point of this project is
operating on compressed bitstreams, so each sequence is encoded to
baseline-profile H.264 (no B-frames) at its native fps. The three detector
variants (DPM/FRCNN/SDP) share identical frames and ground truth, so only
FRCNN directories are encoded.

Usage: python scripts/prep_mot17.py [--crf 23]
Expects data/MOT17.zip (see plan); outputs data/MOT17/videos/<seq>.mp4
"""

import argparse
import configparser
import pathlib
import subprocess
import zipfile

ROOT = pathlib.Path(__file__).resolve().parents[1]
DATA = ROOT / "data"
ZIP = DATA / "MOT17.zip"
MOT = DATA / "MOT17"


def encode(seq_dir: pathlib.Path, out: pathlib.Path, crf: int) -> None:
    ini = configparser.ConfigParser()
    ini.read(seq_dir / "seqinfo.ini")
    fps = ini["Sequence"]["frameRate"]
    subprocess.run(
        [
            "ffmpeg", "-y", "-v", "error",
            "-framerate", fps,
            "-i", str(seq_dir / "img1" / "%06d.jpg"),
            "-c:v", "libx264", "-profile:v", "baseline",
            "-bf", "0", "-g", "30", "-crf", str(crf),
            "-pix_fmt", "yuv420p",
            str(out),
        ],
        check=True,
    )


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--crf", type=int, default=23)
    args = ap.parse_args()

    if not MOT.exists():
        print(f"unzipping {ZIP}")
        with zipfile.ZipFile(ZIP) as z:
            z.extractall(DATA)

    videos = MOT / "videos"
    videos.mkdir(exist_ok=True)
    seqs = sorted((MOT / "train").glob("MOT17-*-FRCNN"))
    for seq in seqs:
        out = videos / f"{seq.name}.mp4"
        if out.exists():
            print(f"skip {out.name}")
            continue
        print(f"encoding {seq.name} @ crf {args.crf}")
        encode(seq, out, args.crf)
    print(f"done: {len(seqs)} sequences in {videos}")
