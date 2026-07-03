"""Train CorrectionNet on dataset(s) from build_correction_dataset.py and/or
build_rollout_dataset.py.

Holds out one full sequence (not a random row split) so validation loss
reflects generalization to an unseen scene, not just unseen frames of a
scene the net has already partially seen.

Usage:
    python scripts/train_correction.py [--epochs 30] [--holdout MOT17-11-FRCNN]
    python scripts/train_correction.py --datasets outputs/correction_dataset.npz \
        outputs/correction_dataset_rollout.npz --out outputs/correction_net_v2.pt
Writes outputs/correction_net.pt (or --out).
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
    ap.add_argument(
        "--datasets", nargs="+",
        default=[str(ROOT / "outputs" / "correction_dataset.npz")],
        help="one or more .npz files (same schema); concatenated for training",
    )
    ap.add_argument("--out", default=str(ROOT / "outputs" / "correction_net.pt"))
    args = ap.parse_args()

    device = pick_device()
    sanity_check(device)

    datasets = [np.load(p, allow_pickle=True) for p in args.datasets]
    seq_names = list(datasets[0]["seq_names"])
    for d, p in zip(datasets[1:], args.datasets[1:]):
        assert list(d["seq_names"]) == seq_names, f"{p} has different seq_names ordering"
    feats = np.concatenate([d["feats"] for d in datasets])
    labels = np.concatenate([d["labels"] for d in datasets])
    seq_ids = np.concatenate([d["seq_ids"] for d in datasets])
    print(f"loaded {len(feats)} pairs from {len(datasets)} dataset(s): {args.datasets}")
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
    # Huber loss for training: rollout data has heavy-tailed targets (a
    # chained correction occasionally drifts a track far off, especially in
    # crowded/occluded cases), and plain MSE on those outliers destabilized
    # training (val loss spiking 10x between epochs). MSE is still used for
    # val reporting so it stays comparable to the zero-residual baseline.
    train_loss_fn = nn.SmoothL1Loss(beta=0.5)
    mse_fn = nn.MSELoss()
    n = len(x_train)

    best_val, best_state = float("inf"), None
    for epoch in range(args.epochs):
        perm = torch.randperm(n, device=device)
        model.train()
        total = 0.0
        for i in range(0, n, args.batch_size):
            idx = perm[i : i + args.batch_size]
            opt.zero_grad()
            loss = train_loss_fn(model(x_train[idx]), y_train[idx])
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            opt.step()
            total += loss.item() * len(idx)
        model.eval()
        with torch.no_grad():
            val_mse = mse_fn(model(x_val), y_val).item()
        if val_mse < best_val:
            best_val = val_mse
            best_state = {k: v.clone() for k, v in model.state_dict().items()}
        print(f"epoch {epoch+1}/{args.epochs}: train MSE={total/n:.5f} val MSE={val_mse:.5f}")

    print(f"best val MSE={best_val:.5f} (zero-residual baseline={zero_val_mse:.5f})")
    out = pathlib.Path(args.out)
    torch.save(best_state, out)
    print(f"saved {out}")


if __name__ == "__main__":
    main()
