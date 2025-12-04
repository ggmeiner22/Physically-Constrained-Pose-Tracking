#!/usr/bin/env python
"""
Train position regressor from crops + bboxes.

Supports:
  • Single manifest + internal split:
        --manifest data/manifest_train.json --val_split 0.1

  • Explicit train + val manifests:
        --manifest data/manifest_train.json --val_manifest data/manifest_val.json
"""

from __future__ import annotations
import os
import argparse
from typing import Optional, Tuple

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, random_split
from tqdm.auto import tqdm
from torchvision import transforms

from pcpose.data.position_dataset import PositionDataset
from pcpose.models.position_head import PositionNet


# -----------------------------------------------------------------------------


def build_args() -> argparse.Namespace:
    p = argparse.ArgumentParser("Train position regressor from crops + bboxes")

    # Data
    p.add_argument(
        "--manifest",
        type=str,
        required=True,
        help="Train manifest JSON path (also used for val if --val_manifest not set).",
    )
    p.add_argument(
        "--val_manifest",
        type=str,
        default=None,
        help="Optional separate validation manifest JSON path.",
    )
    p.add_argument(
        "--outdir",
        type=str,
        default="outputs/position",
        help="Directory to save checkpoints.",
    )

    # Training hyperparameters
    p.add_argument("--epochs", type=int, default=20)
    p.add_argument("--batch_size", type=int, default=64)
    p.add_argument(
        "--val_split",
        type=float,
        default=0.1,
        help="Validation fraction when using a single manifest (ignored if --val_manifest is given).",
    )
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--num_workers", type=int, default=4)
    p.add_argument("--device", type=str, default=None)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--save_every", type=int, default=5)

    # Cropping parameters
    p.add_argument(
        "--crop_size",
        type=int,
        default=128,
        help="Size (pixels) of square crop for the position regressor.",
    )
    p.add_argument(
        "--pad",
        type=float,
        default=0.15,
        help="Relative padding around bbox when cropping.",
    )

    return p.parse_args()


# -----------------------------------------------------------------------------


def build_dataloaders(
    args: argparse.Namespace, device: torch.device
) -> Tuple[DataLoader, Optional[DataLoader]]:
    """
    Build train and val dataloaders.

    If args.val_manifest is provided:
        train_ds = PositionDataset(args.manifest)
        val_ds   = PositionDataset(args.val_manifest)

    Else:
        full_ds is split into train/val using args.val_split.
    """

    # Basic transform: ToTensor (optionally you can add Normalize here)
    transform = transforms.ToTensor()

    if args.val_manifest is not None:
        # Explicit train + val manifests
        train_ds = PositionDataset(
            manifest_path=args.manifest,
            crop_size=args.crop_size,
            pad=args.pad,
            transform=transform,
        )
        val_ds = PositionDataset(
            manifest_path=args.val_manifest,
            crop_size=args.crop_size,
            pad=args.pad,
            transform=transform,
        )
        print(
            f"[INFO] Using explicit manifests: "
            f"train={args.manifest}, val={args.val_manifest}"
        )
    else:
        # Single manifest + random split
        full_ds = PositionDataset(
            manifest_path=args.manifest,
            crop_size=args.crop_size,
            pad=args.pad,
            transform=transform,
        )
        n_total = len(full_ds)
        n_val = int(n_total * args.val_split)
        n_train = n_total - n_val
        if n_train <= 0:
            raise ValueError(
                f"val_split={args.val_split} too large: total={n_total}, "
                f"train={n_train}, val={n_val}"
            )
        print(
            f"[INFO] Using single manifest with val_split={args.val_split:.3f} "
            f"→ train={n_train}, val={n_val}"
        )
        g = torch.Generator().manual_seed(args.seed)
        train_ds, val_ds = random_split(full_ds, [n_train, n_val], generator=g)

    train_loader = DataLoader(
        train_ds,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=True,
    )

    val_loader: Optional[DataLoader]
    if len(val_ds) == 0:
        val_loader = None
    else:
        val_loader = DataLoader(
            val_ds,
            batch_size=args.batch_size,
            shuffle=False,
            num_workers=args.num_workers,
            pin_memory=True,
        )

    return train_loader, val_loader


# -----------------------------------------------------------------------------


def train_epoch(
    model: nn.Module,
    loader: DataLoader,
    opt: optim.Optimizer,
    criterion: nn.Module,
    device: torch.device,
) -> float:
    model.train()
    total_loss = 0.0
    n_batches = 0

    for batch in tqdm(loader, desc="Train", leave=False):
        crops, bbox_norm, pos = batch  # (B,3,Hc,Wc)
        crops, bbox_norm, pos = crops.to(device), bbox_norm.to(device), pos.to(device)

        opt.zero_grad(set_to_none=True)
        pred = model(crops, bbox_norm)            # (B,3)
        loss = criterion(pred, pos)

        loss.backward()
        opt.step()

        total_loss += float(loss.item())
        n_batches += 1

    return total_loss / max(n_batches, 1)


@torch.no_grad()
def eval_epoch(
    model: nn.Module,
    loader: Optional[DataLoader],
    criterion: nn.Module,
    device: torch.device,
) -> float:
    if loader is None:
        return 0.0

    model.eval()
    total_loss = 0.0
    n_batches = 0

    for batch in tqdm(loader, desc="Val", leave=False):
        crops, bbox_norm, pos = batch          # (B,3,Hc,Wc)
        crops, bbox_norm, pos = crops.to(device), bbox_norm.to(device), pos.to(device)

        pred = model(crops, bbox_norm)
        loss = criterion(pred, pos)

        total_loss += float(loss.item())
        n_batches += 1

    return total_loss / max(n_batches, 1)


# -----------------------------------------------------------------------------


def main():
    args = build_args()
    device_str = args.device or ("cuda" if torch.cuda.is_available() else "cpu")
    device = torch.device(device_str)
    os.makedirs(args.outdir, exist_ok=True)
    torch.manual_seed(args.seed)

    print(f"[INFO] Using device: {device}")

    # Build dataloaders
    train_loader, val_loader = build_dataloaders(args, device)

    # Model, optimizer, loss
    model = PositionNet(out_dim=3).to(device)
    optimizer = optim.Adam(model.parameters(), lr=args.lr)
    criterion = nn.MSELoss()

    best_val = float("inf")
    best_path = os.path.join(args.outdir, "best_position.pt")

    for epoch in range(1, args.epochs + 1):
        print(f"\n=== Epoch {epoch}/{args.epochs} ===")
        train_loss = train_epoch(model, train_loader, optimizer, criterion, device)
        val_loss = eval_epoch(model, val_loader, criterion, device)

        print(f"  train_loss = {train_loss:.6f}")
        if val_loader is not None:
            print(f"  val_loss   = {val_loss:.6f}")
        else:
            print("  val_loss   = (no val loader)")

        # Save periodic checkpoints
        if epoch % args.save_every == 0:
            ckpt_path = os.path.join(args.outdir, f"position_epoch{epoch:03d}.pt")
            torch.save(
                {
                    "epoch": epoch,
                    "state_dict": model.state_dict(),
                    "train_loss": train_loss,
                    "val_loss": val_loss,
                    "args": vars(args),
                },
                ckpt_path,
            )
            print(f"  [CKPT] Saved checkpoint to {ckpt_path}")

        # Track best model by val_loss (if val_loader exists)
        if val_loader is not None and val_loss < best_val:
            best_val = val_loss
            torch.save(
                {
                    "epoch": epoch,
                    "state_dict": model.state_dict(),
                    "train_loss": train_loss,
                    "val_loss": val_loss,
                    "args": vars(args),
                },
                best_path,
            )
            print(f"  [BEST] Updated best model at {best_path} (val_loss={best_val:.6f})")

    print("\n[DONE] Training finished.")
    if val_loader is not None:
        print(f"[BEST] Best val_loss = {best_val:.6f}")
        print(f"[BEST] Best model saved at: {best_path}")


if __name__ == "__main__":
    main()
