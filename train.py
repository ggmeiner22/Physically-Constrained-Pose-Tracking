from __future__ import annotations
import os
import argparse
import math
from typing import Optional, Dict

import torch
import torch.optim as optim
from torch.utils.data import DataLoader, random_split
from tqdm.auto import tqdm

from pcpose.models.two_stage_backbone import TwoStageBackbone
from pcpose.pipeline import _positions_to_measurements
from pcpose.pipeline import PosePipeline
from pcpose.losses import total_loss
from pcpose.data_video_dataset import VideoClipDataset


def build_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    # Single-video legacy option still supported
    p.add_argument("--video_path", type=str, default=None)
    # Manifest-driven training (recommended)
    p.add_argument("--manifest", type=str, default=None, help="CSV manifest path")

    p.add_argument("--scenario", type=str, choices=["hanging", "sliding", "dropping"], default="hanging")
    p.add_argument("--device", type=str, default=None)
    p.add_argument("--epochs", type=int, default=10)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--lambda_phys", type=float, default=5.0)
    p.add_argument("--lambda_smooth", type=float, default=0.5)
    p.add_argument("--outdir", type=str, default="outputs/runs")
    p.add_argument("--save_every", type=int, default=5)
    p.add_argument("--max_frames", type=int, default=0)

    p.add_argument("--filter", type=str, choices=["ekf", "ema"], default="ekf")
    p.add_argument("--use_vel_meas", action="store_true")

    p.add_argument("--model", type=str, choices=["temporal", "tiny", "two_stage"], default="temporal")
    p.add_argument("--batch_size", type=int, default=4)
    p.add_argument("--clip_len", type=int, default=32)
    p.add_argument("--num_workers", type=int, default=2)
    p.add_argument("--val_split", type=float, default=0.1)
    p.add_argument("--test_split", type=float, default=0.1)
    p.add_argument("--seed", type=int, default=42)

    p.add_argument("--detector_ckpt", type=str, default=None)
    p.add_argument("--position_ckpt", type=str, default=None)

    # Explicit split manifests
    p.add_argument("--train_manifest", type=str, default=None)
    p.add_argument("--val_manifest",   type=str, default=None)
    p.add_argument("--test_manifest",  type=str, default=None)
    return p.parse_args()


def _build_dataloaders(args, device: str):
    """Fallback: single manifest with internal 80/10/10 split."""
    ds = VideoClipDataset(
        manifest_path=args.manifest,
        clip_len=args.clip_len,
        resize=(224, 224),
        device=device,
        sampling="random",
        seed=args.seed,
    )
    n_total = len(ds)
    n_test = int(math.floor(n_total * args.test_split))
    n_val = int(math.floor(n_total * args.val_split))
    n_train = n_total - n_val - n_test
    if n_train <= 0:
        raise ValueError(f"Split too small: total={n_total}, train={n_train}, val={n_val}, test={n_test}")

    g = torch.Generator().manual_seed(args.seed)
    ds_train, ds_val, ds_test = random_split(ds, [n_train, n_val, n_test], generator=g)

    def collate(batch):
        frames = torch.stack([b["frames"] for b in batch], dim=0)  # (B,T,C,H,W)
        scenarios = [b["scenario"] for b in batch]
        params = [b["params"] for b in batch]
        videos = [b["video_path"] for b in batch]
        return {"frames": frames, "scenarios": scenarios, "params": params, "videos": videos}

    loader_args = dict(batch_size=args.batch_size, num_workers=args.num_workers, collate_fn=collate, shuffle=True, drop_last=False)
    train_loader = DataLoader(ds_train, **loader_args)
    val_loader = DataLoader(ds_val, **{**loader_args, "shuffle": False})
    test_loader = DataLoader(ds_test, **{**loader_args, "shuffle": False})
    return train_loader, val_loader, test_loader


def _loader_from_manifest(path, args, device, shuffle: bool):
    ds = VideoClipDataset(
        manifest_path=path,
        clip_len=args.clip_len,
        resize=(224, 224),
        device=device,
        sampling="random",
        seed=args.seed,
    )

    def collate(batch):
        frames = torch.stack([b["frames"] for b in batch], dim=0)  # (B,T,C,H,W)
        scenarios = [b["scenario"] for b in batch]
        params = [b["params"] for b in batch]
        videos = [b["video_path"] for b in batch]
        return {"frames": frames, "scenarios": scenarios, "params": params, "videos": videos}

    return DataLoader(ds, batch_size=args.batch_size, num_workers=args.num_workers,
                      collate_fn=collate, shuffle=shuffle, drop_last=False)


def train_epoch_seq_batches(model: PosePipeline, loader, opt: optim.Optimizer) -> Dict[str, float]:
    """Train over a DataLoader of (B,T,C,H,W) clips with tqdm progress."""
    model.train()
    logs_accum = {"L": 0.0, "L_vision": 0.0, "L_phys": 0.0, "L_smooth": 0.0}
    n_batches = 0

    for batch in tqdm(loader, desc="Train", leave=False):
        x = batch["frames"].to(model.device)  # (B,T,C,H,W)
        B, T = x.shape[:2]

        if isinstance(model.backbone, TwoStageBackbone):
            # two-stage backbone returns positions (B,T,3)
            pos_bt = model.backbone(x)  # (B,T,3)

            # Convert each sequence to [x,y,vx,vy] using the same helper as predict_and_smooth
            y_list = []
            for i in range(B):
                pos_seq = pos_bt[i]  # (T,3)
                y_seq = _positions_to_measurements(pos_seq, dt=model.params.dt)  # (T,4)
                y_list.append(y_seq)

            y_bt = torch.stack(y_list, dim=0)  # (B,T,4)
        elif getattr(model.backbone, "expects_sequence", False):
            y_bt = model.backbone(x)  # (B,T,4)
        else:
            y_bt = torch.stack([model.backbone(x[:, t]) for t in range(T)], dim=1)  # (B,T,4)


        opt.zero_grad(set_to_none=True)
        L_total = torch.tensor(0.0, device=model.device)

        for i in range(B):
            scenario_i = batch["scenarios"][i]
            # Clone params and apply per-video overrides
            params_i = type(model.params)()
            params_i.gravity = model.params.gravity
            params_i.dt = model.params.dt
            params_i.anchor = model.params.anchor
            params_i.cable_length = model.params.cable_length
            params_i.slide_y = model.params.slide_y

            p = batch["params"][i]
            if "anchor_x" in p and "anchor_y" in p:
                params_i.anchor = (float(p["anchor_x"]), float(p["anchor_y"]))
            if "cable_length" in p:
                params_i.cable_length = float(p["cable_length"])
            if "slide_y" in p:
                params_i.slide_y = float(p["slide_y"])
            if "mu_slide" in p:
                params_i.mu_slide = float(p["mu_slide"])  # type: ignore[attr-defined]
            if "pendulum_damping" in p:
                params_i.pendulum_damping = float(p["pendulum_damping"])  # type: ignore[attr-defined]

            y_prev = None
            for t in range(T):
                y_t = y_bt[i, t:t+1, :]  # (1,4)
                gt_t: Optional[torch.Tensor] = None
                L_t, logs = total_loss(y_t, y_prev, scenario_i, params_i, model.weights, gt_t)
                L_total = L_total + L_t
                y_prev = y_t
                for k, v in logs.items():
                    logs_accum[k] += float(v.item())

        L_total.backward()
        opt.step()
        n_batches += 1

    if n_batches == 0:
        return {k: 0.0 for k in logs_accum}
    return {k: v / n_batches for k, v in logs_accum.items()}


@torch.no_grad()
def eval_epoch_seq_batches(model: PosePipeline, loader) -> Dict[str, float]:
    model.eval()
    logs_accum = {"L": 0.0, "L_vision": 0.0, "L_phys": 0.0, "L_smooth": 0.0}
    n_batches = 0

    for batch in tqdm(loader, desc="Eval", leave=False):
        x = batch["frames"].to(model.device)
        B, T = x.shape[:2]
        if isinstance(model.backbone, TwoStageBackbone):
            pos_bt = model.backbone(x)  # (B,T,3)
            y_list = []
            for i in range(B):
                pos_seq = pos_bt[i]
                y_seq = _positions_to_measurements(pos_seq, dt=model.params.dt)
                y_list.append(y_seq)
            y_bt = torch.stack(y_list, dim=0)  # (B,T,4)
        elif getattr(model.backbone, "expects_sequence", False):
            y_bt = model.backbone(x)
        else:
            y_bt = torch.stack([model.backbone(x[:, t]) for t in range(T)], dim=1)


        for i in range(B):
            scenario_i = batch["scenarios"][i]
            params_i = type(model.params)()
            params_i.gravity = model.params.gravity
            params_i.dt = model.params.dt
            params_i.anchor = model.params.anchor
            params_i.cable_length = model.params.cable_length
            params_i.slide_y = model.params.slide_y

            p = batch["params"][i]
            if "anchor_x" in p and "anchor_y" in p:
                params_i.anchor = (float(p["anchor_x"]), float(p["anchor_y"]))
            if "cable_length" in p:
                params_i.cable_length = float(p["cable_length"])
            if "slide_y" in p:
                params_i.slide_y = float(p["slide_y"])
            if "mu_slide" in p:
                params_i.mu_slide = float(p["mu_slide"])  # type: ignore[attr-defined]
            if "pendulum_damping" in p:
                params_i.pendulum_damping = float(p["pendulum_damping"])  # type: ignore[attr-defined]

            y_prev = None
            for t in range(T):
                y_t = y_bt[i, t:t+1, :]
                gt_t: Optional[torch.Tensor] = None
                L_t, logs = total_loss(y_t, y_prev, scenario_i, params_i, model.weights, gt_t)
                y_prev = y_t
                for k, v in logs.items():
                    logs_accum[k] += float(v.item())
        n_batches += 1

    if n_batches == 0:
        return {k: 0.0 for k in logs_accum}
    return {k: v / n_batches for k, v in logs_accum.items()}


def main():
    args = build_args()
    device = args.device or ("cuda" if torch.cuda.is_available() else "cpu")
    os.makedirs(args.outdir, exist_ok=True)

    # Build model
    model = PosePipeline(
        scenario=args.scenario,
        device=device,
        filter_type=args.filter,
        use_vel_meas=args.use_vel_meas,
        model_type=args.model,
        detector_ckpt=args.detector_ckpt,
        position_ckpt=args.position_ckpt,
    )

    model.weights.lambda_phys = args.lambda_phys
    model.weights.lambda_smooth = args.lambda_smooth

    opt = optim.Adam(model.parameters(), lr=args.lr)

    # ---- Prefer explicit manifests if provided; otherwise use single manifest with internal split ----
    if args.train_manifest or args.manifest:
        if args.train_manifest:
            train_loader = _loader_from_manifest(args.train_manifest, args, device, shuffle=True)
            val_loader = _loader_from_manifest(args.val_manifest, args, device, shuffle=False) if args.val_manifest else None
            test_loader = _loader_from_manifest(args.test_manifest, args, device, shuffle=False) if args.test_manifest else None
        else:
            train_loader, val_loader, test_loader = _build_dataloaders(args, device)

        for epoch in range(1, args.epochs + 1):
            tr_logs = train_epoch_seq_batches(model, train_loader, opt)
            va_logs = eval_epoch_seq_batches(model, val_loader) if val_loader else {"L": 0, "L_vision": 0, "L_phys": 0, "L_smooth": 0}
            print(f"Epoch {epoch:03d} | train: {tr_logs} | val: {va_logs}")

            if epoch % args.save_every == 0:
                ckpt = {"epoch": epoch, "state_dict": model.state_dict(), "args": vars(args),
                        "logs": {"train": tr_logs, "val": va_logs}}
                torch.save(ckpt, os.path.join(args.outdir, f"ckpt_epoch{epoch:03d}.pt"))

        if test_loader:
            te_logs = eval_epoch_seq_batches(model, test_loader)
            print(f"[TEST] {te_logs}")
        return

    # ---- Legacy single-video path ----
    if args.video_path is None:
        raise ValueError("Provide --train_manifest/--manifest for multi-video training, or --video_path for legacy mode.")

    from pcpose.data import load_video_frames, to_tensor_batch
    frames_np = load_video_frames(args.video_path, max_frames=args.max_frames, resize=(224, 224))
    batch = to_tensor_batch(frames_np, device)

    y_gt = None
    for epoch in range(1, args.epochs + 1):
        # Reuse older per-frame training if you kept it around; else call the new functions with a faux loader
        tr_logs = train_epoch_seq_batches(model, [{"frames": batch.frames.unsqueeze(0),
                                                   "scenarios": [args.scenario], "params": [{}], "videos": [args.video_path]}], opt)
        ev_logs = eval_epoch_seq_batches(model, [{"frames": batch.frames.unsqueeze(0),
                                                  "scenarios": [args.scenario], "params": [{}], "videos": [args.video_path]}])
        print(f"Epoch {epoch:03d} | train: {tr_logs} | eval: {ev_logs}")

        if epoch % args.save_every == 0:
            ckpt = {"epoch": epoch, "state_dict": model.state_dict(), "args": vars(args),
                    "logs": {"train": tr_logs, "eval": ev_logs}}
            torch.save(ckpt, os.path.join(args.outdir, f"ckpt_epoch{epoch:03d}.pt"))


if __name__ == "__main__":
    main()
