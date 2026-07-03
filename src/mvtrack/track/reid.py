"""Lightweight appearance embedding for re-identification.

Motion vectors alone are a structural ceiling: they carry no information
about what something looks like, so IoU-based association can't
distinguish two overlapping people or recover an identity across a gap
too large for spatial overlap alone. This wraps a generic
ImageNet-pretrained backbone (no time budget in this pass for training a
person-ReID-specific model on MOT17 triplets) as a cosine-similarity
appearance cost, used alongside IoU in MVTracker's association instead of
IoU alone.
"""

import ssl

import certifi
import cv2
import numpy as np
import torch
from torch import nn
from torchvision.models import MobileNet_V3_Small_Weights, mobilenet_v3_small

# macOS python.org installs lack system CA certs, breaking the first-time
# pretrained-weights download via torch.hub (same root cause as the fix in
# scripts/get_test_clip.py, applied globally here since torch.hub's
# downloader doesn't take an explicit ssl context).
ssl._create_default_https_context = lambda: ssl.create_default_context(cafile=certifi.where())

CROP_SIZE = (128, 64)  # (h, w), standard ReID aspect ratio


class ReIDEmbedder:
    def __init__(self, device: str = "cpu"):
        self.device = device
        weights = MobileNet_V3_Small_Weights.IMAGENET1K_V1
        backbone = mobilenet_v3_small(weights=weights)
        self.model = nn.Sequential(backbone.features, nn.AdaptiveAvgPool2d(1)).to(device).eval()
        tf = weights.transforms()
        self.mean = torch.tensor(tf.mean, device=device).view(1, 3, 1, 1)
        self.std = torch.tensor(tf.std, device=device).view(1, 3, 1, 1)
        with torch.no_grad():
            dummy = torch.zeros(1, 3, *CROP_SIZE, device=device)
            self.dim = self.model(dummy).squeeze(-1).squeeze(-1).shape[1]

    def __call__(self, frame_bgr: np.ndarray, boxes_xyxy: np.ndarray) -> np.ndarray:
        """Returns L2-normalized embeddings [N, dim] for each box crop."""
        if len(boxes_xyxy) == 0:
            return np.zeros((0, self.dim), np.float32)
        h, w = frame_bgr.shape[:2]
        crops = []
        for x0, y0, x1, y1 in boxes_xyxy:
            x0, y0 = max(int(x0), 0), max(int(y0), 0)
            x1, y1 = min(int(x1), w), min(int(y1), h)
            if x1 <= x0 or y1 <= y0:
                crops.append(np.zeros((*CROP_SIZE, 3), np.uint8))
                continue
            crop = cv2.resize(frame_bgr[y0:y1, x0:x1], (CROP_SIZE[1], CROP_SIZE[0]))
            crops.append(cv2.cvtColor(crop, cv2.COLOR_BGR2RGB))
        batch = torch.from_numpy(np.stack(crops)).to(self.device)
        batch = batch.permute(0, 3, 1, 2).float() / 255.0
        batch = (batch - self.mean) / self.std
        with torch.no_grad():
            feats = self.model(batch).squeeze(-1).squeeze(-1)
            feats = torch.nn.functional.normalize(feats, dim=1)
        return feats.cpu().numpy().astype(np.float32)
