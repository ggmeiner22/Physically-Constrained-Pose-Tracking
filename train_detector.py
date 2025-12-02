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


def build_args() -> argparse.Namespace:
    p = argparse.ArgumentParser("Train bounding box detector")
    p.add_argument("--manifest", type=str, required=True,
                   help="JSON manifest with fields: image_path, bbox, pos")
    p.add_argument("--outdir", type=str, default="outputs/detector",
                   help="Where to save detector checkpoints")
    p.add_argument("--epochs", type=int, default=10)
    p.add_argument("--batch_size", type=int, default=32)
    p.add_argument("--val_split", type=float, default=0.1)
    p.add_argument("--lr", type=float, default=1e-4)
    p.add_argument("--num_workers", type=int, default=0)
    p.add_argument("--device", type=str, default=None)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--save_every", type=int, default=5)
    return p.parse_args()


def main() -> None:
    args = build_args()
    os.makedirs(args.outdir, exist_ok=True)

    device = args.device or ("cuda" if torch.cuda.is_available() else "cpu")
    use_pin = device != "cpu"
    print(f"[train_detector] Using device: {device}")

    # Simple HWC uint8 -> CHW float32 [0,1]
    transform = transforms.ToTensor()

    full_ds = DetectorDataset(args.manifest, transform=transform)

    n_total = len(full_ds)
    n_val = int(n_total * args.val_split)
    n_train = n_total - n_val

    g = torch.Generator().manual_seed(args.seed)
    train_ds, val_ds = random_split(full_ds, [n_train, n_val], generator=g)

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

    model = BoundingBoxNet(pretrained=True).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    criterion = nn.SmoothL1Loss()

    best_val = float("inf")

    for epoch in range(1, args.epochs + 1):
        # ---- Train ----
        model.train()
        running_loss = 0.0
        n_batches = 0

        for imgs, bboxes in tqdm(train_loader, desc=f"Train {epoch}", leave=False):
            imgs = imgs.to(device)          # (B,3,H,W)
            bboxes = bboxes.to(device)      # (B,4)

            preds = model(imgs)             # (B,4)
            loss = criterion(preds, bboxes)

            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()

            running_loss += loss.item()
            n_batches += 1

        train_loss = running_loss / max(n_batches, 1)

        # ---- Val ----
        model.eval()
        running_val = 0.0
        n_batches = 0
        with torch.no_grad():
            for imgs, bboxes in tqdm(val_loader, desc=f"Val {epoch}", leave=False):
                imgs = imgs.to(device)
                bboxes = bboxes.to(device)
                preds = model(imgs)
                loss = criterion(preds, bboxes)
                running_val += loss.item()
                n_batches += 1
        val_loss = running_val / max(n_batches, 1)

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


if __name__ == "__main__":
    main()
