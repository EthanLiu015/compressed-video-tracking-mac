"""Fetch a small pedestrian clip and re-encode it for MV experiments.

Re-encode targets baseline-profile H.264 (no B-frames, GOP 30) so early MV
propagation only has to handle forward prediction — per the plan's risk
mitigation. Uses PyAV for the re-encode so it works before `brew install
ffmpeg` finishes.
"""

import pathlib
import ssl
import urllib.request

import av
import certifi

URL = (
    "https://github.com/intel-iot-devkit/sample-videos/raw/master/"
    "people-detection.mp4"
)
DATA = pathlib.Path(__file__).resolve().parents[1] / "data"
RAW = DATA / "people_raw.mp4"
OUT = DATA / "people_baseline.mp4"


def reencode(src: pathlib.Path, dst: pathlib.Path) -> None:
    inp = av.open(str(src))
    out = av.open(str(dst), "w")
    istream = inp.streams.video[0]
    ostream = out.add_stream(
        "libx264",
        rate=istream.average_rate or 30,
        options={"profile": "baseline", "bf": "0", "g": "30", "crf": "23"},
    )
    ostream.width = istream.codec_context.width
    ostream.height = istream.codec_context.height
    ostream.pix_fmt = "yuv420p"
    for frame in inp.decode(istream):
        out.mux(ostream.encode(frame))
    out.mux(ostream.encode())  # flush
    out.close()
    inp.close()


if __name__ == "__main__":
    DATA.mkdir(exist_ok=True)
    if not RAW.exists():
        print(f"downloading {URL}")
        ctx = ssl.create_default_context(cafile=certifi.where())
        with urllib.request.urlopen(URL, context=ctx) as resp:
            RAW.write_bytes(resp.read())
    print(f"re-encoding -> {OUT}")
    reencode(RAW, OUT)
    print(f"done: {OUT} ({OUT.stat().st_size / 1e6:.1f} MB)")
