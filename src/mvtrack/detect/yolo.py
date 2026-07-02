"""YOLO detector wrapper (PyTorch MPS today, Core ML export later)."""

import numpy as np
import torch
from ultralytics import YOLO


def pick_device() -> str:
    return "mps" if torch.backends.mps.is_available() else "cpu"


class Detector:
    def __init__(self, weights: str = "yolov8n.pt", device: str | None = None, conf: float = 0.25):
        self.model = YOLO(weights)
        self.device = device or pick_device()
        self.conf = conf

    def __call__(self, frame_bgr: np.ndarray):
        """Return (boxes_xyxy [N,4], scores [N], class_ids [N])."""
        res = self.model.predict(
            frame_bgr, device=self.device, conf=self.conf, verbose=False
        )[0]
        return (
            res.boxes.xyxy.cpu().numpy(),
            res.boxes.conf.cpu().numpy(),
            res.boxes.cls.cpu().numpy().astype(int),
        )
