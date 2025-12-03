#!/usr/bin/env python
"""
Train bounding box detector (image → bbox).

Supports:
  • Single manifest + internal split:
        --manifest data/manifest_train.json --val_split 0.1

  • Explicit train + val manifests:
        --manifest data/manifest_train.json --val_manifest data/manifest_val.json
"""

from __future__ import annotations

import argparse
import os

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, random_split
from torchvision import transforms
from tqdm.auto import tqdm

from pcpose.data.detector_dataset import DetectorDataset
from pcpose.models.detector import BoundingBoxNet


# -----------------------------------------------------------------------------


def build_args() -> argparse.Namespace:
    p = argparse.ArgumentParser("Train bounding box detector")
    p.add_argument(
        "--manifest",
        type=str,
        required=True,
        help="Train manifest JSON (image_path, bbox, pos).",
    )
    p.add_argument(
        "--val_manifest",
        type=str,
        default=None,
        help="Optional separate validation manifest JSON.",
    )
    p.add_argument(
        "--outdir",
        type=str,
        default="outputs/detector",
        help="Where to save detector checkpoints.",
    )
    p.add_argument("--epochs", type=int, default=10)
    p.add_argument("--batch_size", type=int, default=32)
    p.add_argument(
        "--val_split",
        type=float,
        default=0.1,
        help="Validation fraction when using a single manifest "
             "(ignored if --val_manifest is given).",
    )
    p.add_argument("--lr", type=float, default=1e-4)
    p.add_argument("--num_workers", type=int, default=0)
    p.add_argument("--device", type=str, default=None)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--save_every", type=int, default=5)
    return p.parse_args()


# -----------------------------------------------------------------------------


def build_dataloaders(args: argparse.Namespace) -> tuple[DataLoader, DataLoader]:
    """
    Build train and val DataLoaders.

    If args.val_manifest is provided:
        train_ds = DetectorDataset(args.manifest)
        val_ds   = DetectorDataset(args.val_manifest)

    Else:
        full_ds is split into train/val using args.val_split.
    """

    # Simple HWC uint8 -> CHW float32 [0,1]
    # (You can add Normalize if you want ImageNet-style preproc)
    transform = transforms.ToTensor()

    if args.val_manifest is not None:
        print(
            f"[INFO] Using explicit manifests: "
            f"train={args.manifest}, val={args.val_manifest}"
        )
        train_ds = DetectorDataset(args.manifest, transform=transform)
        val_ds = DetectorDataset(args.val_manifest, transform=transform)
    else:
        full_ds = DetectorDataset(args.manifest, transform=transform)
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

    device = args.device or ("cuda" if torch.cuda.is_available() else "cpu")
    use_pin = device != "cpu"

    train_loader = DataLoader(
        train_ds,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=use_pin,
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=use_pin,
    )

    return train_loader, val_loader


# -----------------------------------------------------------------------------


def train_epoch(
    model: nn.Module,
    loader: DataLoader,
    opt: torch.optim.Optimizer,
    criterion: nn.Module,
    device: torch.device,
) -> float:
    model.train()
    running_loss = 0.0
    n_batches = 0

    for imgs, bboxes in tqdm(loader, desc="Train", leave=False):
        imgs = imgs.to(device)
        bboxes = bboxes.to(device)

        opt.zero_grad(set_to_none=True)
        preds = model(imgs)
        loss = criterion(preds, bboxes)
        loss.backward()
        opt.step()

        running_loss += loss.item()
        n_batches += 1

    return running_loss / max(n_batches, 1)


@torch.no_grad()
def eval_epoch(
    model: nn.Module,
    loader: DataLoader,
    criterion: nn.Module,
    device: torch.device,
) -> float:
    model.eval()
    running_val = 0.0
    n_batches = 0

    for imgs, bboxes in tqdm(loader, desc="Val", leave=False):
        imgs = imgs.to(device)
        bboxes = bboxes.to(device)
        preds = model(imgs)
        loss = criterion(preds, bboxes)
        running_val += loss.item()
        n_batches += 1

    return running_val / max(n_batches, 1)


# -----------------------------------------------------------------------------


def main() -> None:
    args = build_args()
    os.makedirs(args.outdir, exist_ok=True)

    device_str = args.device or ("cuda" if torch.cuda.is_available() else "cpu")
    device = torch.device(device_str)
    print(f"[train_detector] Using device: {device}")

    torch.manual_seed(args.seed)

    train_loader, val_loader = build_dataloaders(args)

    model = BoundingBoxNet(pretrained=True).to(device)
    criterion = nn.SmoothL1Loss(beta=1.0)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)

    best_val = float("inf")

    for epoch in range(1, args.epochs + 1):
        print(f"\n=== Epoch {epoch}/{args.epochs} ===")
        train_loss = train_epoch(model, train_loader, optimizer, criterion, device)
        val_loss = eval_epoch(model, val_loader, criterion, device)

        print(f"[Epoch {epoch:03d}] train_loss={train_loss:.4f}  val_loss={val_loss:.4f}")

        # Save best
        if val_loss < best_val:
            best_val = val_loss
            ckpt_path = os.path.join(args.outdir, "best_detector.pt")
            torch.save(
                {"epoch": epoch, "state_dict": model.state_dict(), "val_loss": val_loss},
                ckpt_path,
            )
            print(f"  → Saved new best to {ckpt_path}")

        # Save periodic checkpoints
        if epoch % args.save_every == 0:
            ckpt_path = os.path.join(args.outdir, f"detector_epoch{epoch:03d}.pt")
            torch.save(
                {"epoch": epoch, "state_dict": model.state_dict(), "val_loss": val_loss},
                ckpt_path,
            )
            print(f"  → Saved checkpoint to {ckpt_path}")

    print(f"\n[DONE] Best val_loss = {best_val:.4f}")


if __name__ == "__main__":
    main()
