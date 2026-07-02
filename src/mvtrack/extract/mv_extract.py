"""Motion-vector extraction from H.264/HEVC bitstreams via PyAV.

This is the correctness-baseline path: FFmpeg decodes frames fully with
`flags2 +export_mvs`, and we read AVMotionVector side data off each frame.
The cheap path (skip pixel reconstruction with skip_idct/skip_loop_filter)
replaces this later without changing the FrameMV interface.

MV convention: FFmpeg reports (src_x, src_y) -> (dst_x, dst_y) per predicted
block. The displacement stored in `mv_grid` is (dst - src), i.e. how far the
content of the block moved *forward* into the current frame — the quantity
needed to shift a track box from the reference frame to this one.
"""

from dataclasses import dataclass
from typing import Iterator, Tuple

import av
import numpy as np
from av.video.frame import PictureType

BLOCK = 16  # grid cell size in pixels

# Memory layout of libavutil's AVMotionVector (40-byte stride, 8-byte aligned
# uint64 flags). PyAV 17 exposes the side data only as a raw buffer.
MV_DTYPE = np.dtype(
    {
        "names": [
            "source", "w", "h", "src_x", "src_y", "dst_x", "dst_y",
            "flags", "motion_x", "motion_y", "motion_scale",
        ],
        "formats": [
            "<i4", "u1", "u1", "<i2", "<i2", "<i2", "<i2",
            "<u8", "<i4", "<i4", "<u2",
        ],
        "offsets": [0, 4, 5, 6, 8, 10, 12, 16, 24, 28, 32],
        "itemsize": 40,
    }
)


@dataclass
class FrameMV:
    index: int
    pict_type: str  # 'I', 'P', 'B', ...
    width: int
    height: int
    mv_grid: np.ndarray  # (H/16, W/16, 2) float32, (dx, dy) in pixels
    occupancy: np.ndarray  # (H/16, W/16) uint8, 1 where a block carried an MV


def _motion_vectors(frame: av.VideoFrame):
    for sd in frame.side_data:
        if "MOTION_VECTORS" in str(sd.type):
            return np.frombuffer(bytes(sd), dtype=MV_DTYPE)
    return None


def _frame_mv(index: int, frame: av.VideoFrame) -> FrameMV:
    h_cells = (frame.height + BLOCK - 1) // BLOCK
    w_cells = (frame.width + BLOCK - 1) // BLOCK
    grid = np.zeros((h_cells, w_cells, 2), np.float32)
    occ = np.zeros((h_cells, w_cells), np.uint8)

    mvs = _motion_vectors(frame)
    if mvs is not None and len(mvs):
        dx = (mvs["dst_x"] - mvs["src_x"]).astype(np.float32)
        dy = (mvs["dst_y"] - mvs["src_y"]).astype(np.float32)
        # Backward-referencing vectors (source > 0, i.e. predicted from a
        # future frame in B-frames) point the wrong way for forward motion.
        back = mvs["source"] > 0
        dx[back] *= -1.0
        dy[back] *= -1.0

        # dst_x/dst_y are block centers; blocks can be 4..16 px, so paint
        # every grid cell the block overlaps.
        x0 = np.clip(
            mvs["dst_x"].astype(np.int32) - mvs["w"] // 2, 0, frame.width - 1
        )
        y0 = np.clip(
            mvs["dst_y"].astype(np.int32) - mvs["h"] // 2, 0, frame.height - 1
        )
        x1 = np.clip(x0 + mvs["w"], 1, frame.width)
        y1 = np.clip(y0 + mvs["h"], 1, frame.height)
        for i in range(len(mvs)):
            cy0, cy1 = y0[i] // BLOCK, (y1[i] - 1) // BLOCK + 1
            cx0, cx1 = x0[i] // BLOCK, (x1[i] - 1) // BLOCK + 1
            grid[cy0:cy1, cx0:cx1, 0] = dx[i]
            grid[cy0:cy1, cx0:cx1, 1] = dy[i]
            occ[cy0:cy1, cx0:cx1] = 1

    return FrameMV(
        index=index,
        pict_type=PictureType(frame.pict_type).name,
        width=frame.width,
        height=frame.height,
        mv_grid=grid,
        occupancy=occ,
    )


def iter_frames_with_mvs(
    path: str,
) -> Iterator[Tuple[FrameMV, av.VideoFrame]]:
    """Yield (FrameMV, decoded frame) pairs for every video frame in `path`."""
    container = av.open(path)
    try:
        stream = container.streams.video[0]
        # Container-level options don't reach the decoder in PyAV 17; the
        # flag must be set on the codec context before the first decode.
        stream.codec_context.flags2 |= av.codec.context.Flags2.export_mvs
        for i, frame in enumerate(container.decode(stream)):
            yield _frame_mv(i, frame), frame
    finally:
        container.close()
