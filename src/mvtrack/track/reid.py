"""Appearance embedding for re-identification, via a real ReID-trained model.

Motion vectors alone are a structural ceiling: they carry no information
about what something looks like, so IoU-based association can't
distinguish two overlapping people or recover an identity across a gap
too large for spatial overlap alone (see findings.md #9-12).

A first pass used a generic ImageNet-pretrained classifier backbone
(MobileNetV3) as a quick, no-training-required embedding -- findings.md
#13 measured it as genuinely influencing ~26% of real assignment
decisions but net-flat on tracking accuracy, and diagnosed why: a
classifier backbone learns "this is a person" (category), not "this is
*this* person" (identity), which is exactly what dedicated ReID
metric-learning training exists to fix.

This uses OSNet (Zhou et al., ICCV'19), pretrained by the original authors
on MSMT17 (4101 identities, 15 cameras -- an actual person-ReID benchmark,
not a classification dataset), sourced from a public checkpoint (no
training on MOT17 at all -- zero overfitting risk to MOT17's small
identity pool, unlike fine-tuning would carry). Verified directly: on real
MOT17 crops, OSNet's same-identity vs. different-identity cosine
similarity gap (0.511) is meaningfully larger than the generic backbone's
(0.428) before this was wired into tracking at all.
"""

import cv2
import numpy as np
import torch
import torch.nn.functional as F
from huggingface_hub import hf_hub_download
from torchreid.reid.models import build_model

CROP_SIZE = (256, 128)  # (h, w) -- OSNet's MSMT17 training resolution
_MSMT17_NUM_CLASSES = 4101  # only used to build a matching classifier head, then discarded
_HF_REPO = "kaiyangzhou/osnet"
_HF_FILENAME = (
    "osnet_x0_25_msmt17_combineall_256x128_amsgrad_ep150_stp60_"
    "lr0.0015_b64_fb10_softmax_labelsmooth_flip_jitter.pth"
)
_IMAGENET_MEAN = (0.485, 0.456, 0.406)
_IMAGENET_STD = (0.229, 0.224, 0.225)


class ReIDEmbedder:
    def __init__(self, device: str = "cpu"):
        self.device = device
        ckpt_path = hf_hub_download(repo_id=_HF_REPO, filename=_HF_FILENAME)
        model = build_model("osnet_x0_25", num_classes=_MSMT17_NUM_CLASSES, pretrained=False)
        state = torch.load(ckpt_path, map_location="cpu")
        missing, unexpected = model.load_state_dict(state, strict=False)
        assert not missing and not unexpected, (
            f"OSNet checkpoint didn't match the architecture exactly "
            f"(missing={missing}, unexpected={unexpected}) -- likely a "
            f"num_classes mismatch or a changed checkpoint filename."
        )
        self.model = model.to(device).eval()
        self.mean = torch.tensor(_IMAGENET_MEAN, device=device).view(1, 3, 1, 1)
        self.std = torch.tensor(_IMAGENET_STD, device=device).view(1, 3, 1, 1)
        with torch.no_grad():
            dummy = torch.zeros(1, 3, *CROP_SIZE, device=device)
            self.dim = self.model(dummy).shape[1]

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
            feats = self.model(batch)
            feats = F.normalize(feats, dim=1)
        return feats.cpu().numpy().astype(np.float32)
