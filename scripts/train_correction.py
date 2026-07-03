"""Train CorrectionNet on the dataset from build_correction_dataset.py.

Holds out one full sequence (not a random row split) so validation loss
reflects generalization to an unseen scene, not just unseen frames of a
scene the net has already partially seen.

Usage: python scripts/train_correction.py [--epochs 30] [--holdout MOT17-11-FRCNN]
Writes outputs/correction_net.pt
"""

import argparse
import pathlib
import sys

import numpy as np
import torch
from torch import nn

ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from mvtrack.detect import pick_device
from mvtrack.track.correct import CorrectionNet


def sanity_check(device: str) -> None:
    """Overfit a tiny random batch to near-zero loss before trusting the
    training loop on the real dataset -- catches shape/wiring bugs early."""
    torch.manual_seed(0)
    model = CorrectionNet().to(device)
    x = torch.randn(32, model.net[0].in_features, device=device)
    y = torch.randn(32, 4, device=device)
    opt = torch.optim.Adam(model.parameters(), lr=1e-2)
    loss_fn = nn.MSELoss()
    for _ in range(300):
        opt.zero_grad()
        loss = loss_fn(model(x), y)
        loss.backward()
        opt.step()
    final = loss.item()
    assert final < 0.05, f"sanity overfit failed, loss={final:.4f}"
    print(f"sanity check OK: overfit 32-sample batch to loss={final:.4f}")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--epochs", type=int, default=30)
    ap.add_argument("--holdout", default="MOT17-11-FRCNN")
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--batch-size", type=int, default=256)
    args = ap.parse_args()

    device = pick_device()
    sanity_check(device)

    data = np.load(ROOT / "outputs" / "correction_dataset.npz", allow_pickle=True)
    feats, labels, seq_ids = data["feats"], data["labels"], data["seq_ids"]
    seq_names = list(data["seq_names"])
    holdout_idx = seq_names.index(args.holdout)

    train_mask = seq_ids != holdout_idx
    val_mask = ~train_mask
    x_train = torch.from_numpy(feats[train_mask]).to(device)
    y_train = torch.from_numpy(labels[train_mask]).to(device)
    x_val = torch.from_numpy(feats[val_mask]).to(device)
    y_val = torch.from_numpy(labels[val_mask]).to(device)
    print(f"train: {len(x_train)} pairs, val (held-out {args.holdout}): {len(x_val)} pairs")

    # Baseline to beat: "do nothing" (predict zero residual, i.e. trust
    # propagate_boxes as-is) -- the net should learn to beat this on val.
    zero_val_mse = (y_val**2).mean().item()
    print(f"zero-residual baseline val MSE: {zero_val_mse:.5f}")

    model = CorrectionNet().to(device)
    opt = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=1e-4)
    loss_fn = nn.MSELoss()
    n = len(x_train)

    best_val, best_state = float("inf"), None
    for epoch in range(args.epochs):
        perm = torch.randperm(n, device=device)
        model.train()
        total = 0.0
        for i in range(0, n, args.batch_size):
            idx = perm[i : i + args.batch_size]
            opt.zero_grad()
            loss = loss_fn(model(x_train[idx]), y_train[idx])
            loss.backward()
            opt.step()
            total += loss.item() * len(idx)
        model.eval()
        with torch.no_grad():
            val_mse = loss_fn(model(x_val), y_val).item()
        if val_mse < best_val:
            best_val = val_mse
            best_state = {k: v.clone() for k, v in model.state_dict().items()}
        print(f"epoch {epoch+1}/{args.epochs}: train MSE={total/n:.5f} val MSE={val_mse:.5f}")

    print(f"best val MSE={best_val:.5f} (zero-residual baseline={zero_val_mse:.5f})")
    out = ROOT / "outputs" / "correction_net.pt"
    torch.save(best_state, out)
    print(f"saved {out}")


if __name__ == "__main__":
    main()
