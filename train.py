from __future__ import annotations
import os
import argparse
import numpy as np
import torch
import torch.optim as optim
from pcpose.config import CLIConfig
from pcpose.data import load_video_frames, to_tensor_batch
from pcpose.pipeline import PosePipeline
from pcpose.engine import train_one_epoch, evaluate
from pcpose.viz import render_overlay_sequence


def build_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--video_path", type=str, default=None)
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
    p.add_argument("--use_vel_meas", action="store_true", help="EKF consumes [x,y,vx,vy]")
    p.add_argument("--model", type=str, choices=["temporal", "tiny"], default="temporal",
                   help="Choose the visual backbone")
    return p.parse_args()


def main():
    args = build_args()
    cfg = CLIConfig(
        video_path=args.video_path,
        scenario=args.scenario,
        device=args.device or ("cuda" if torch.cuda.is_available() else "cpu"),
        epochs=args.epochs,
        lr=args.lr,
        lambda_phys=args.lambda_phys,
        lambda_smooth=args.lambda_smooth,
        outdir=args.outdir,
        save_every=args.save_every,
        max_frames=args.max_frames,
    )

    os.makedirs(cfg.outdir, exist_ok=True)
    
    # Load frames
    if cfg.video_path is None:
        print("[WARN] --video_path not provided. Creating synthetic frames.")
        T = 90
        frames_np = np.full((T, 256, 256, 3), 200, dtype=np.uint8)
    else:
        frames_np = load_video_frames(cfg.video_path, max_frames=cfg.max_frames)
    
    batch = to_tensor_batch(frames_np, cfg.device)
    
    # Create model
    model = PosePipeline(cfg.scenario, cfg.device,
                     filter_type=args.filter,
                     use_vel_meas=args.use_vel_meas,
                     model_type=args.model)

    # update weights from CLI
    model.weights.lambda_phys = cfg.lambda_phys
    model.weights.lambda_smooth = cfg.lambda_smooth
    
    opt = optim.Adam(model.parameters(), lr=cfg.lr)
    y_gt = None # Optional: provide tensor of shape (T,4)
    
    # Train loop
    for epoch in range(1, cfg.epochs + 1):
        tr_logs = train_one_epoch(model, batch, y_gt, opt)
        ev_logs = evaluate(model, batch, y_gt)
        print(f"Epoch {epoch:03d} | train: {tr_logs} | eval: {ev_logs}")
    
        if epoch % cfg.save_every == 0:
            ckpt = {
            "epoch": epoch,
            "state_dict": model.state_dict(),
            "cfg": cfg.__dict__,
            "logs": {"train": tr_logs, "eval": ev_logs},
            }
            ckpt_path = os.path.join(cfg.outdir, f"ckpt_epoch{epoch:03d}.pt")
            torch.save(ckpt, ckpt_path)
    
    # Predict & visualize
    with torch.no_grad():
        y_all = model.predict_and_smooth(batch.frames) # (T,4)
        xy = y_all[:, :2].detach().cpu().numpy()
    
    out_video = os.path.join(cfg.outdir, "viz", "overlay.mp4")
    render_overlay_sequence(frames_np, xy, out_video)
    print(f"[OK] Wrote visualization to: {out_video}")


if __name__ == "__main__":
    main()
