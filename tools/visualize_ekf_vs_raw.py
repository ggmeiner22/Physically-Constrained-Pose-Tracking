from __future__ import annotations
import os
import argparse
import numpy as np
import torch
import cv2

from pcpose import data as data_mod
from pcpose.pipeline import PosePipeline


def build_args():
    p = argparse.ArgumentParser("Visualize RAW vs EKF-smoothed trajectory")
    p.add_argument("--video_path", type=str, required=True,
                   help="Input video file (mp4, etc.)")
    p.add_argument("--scenario", type=str,
                   choices=["pendulum", "sliding", "dropping"],
                   default="pendulum")
    p.add_argument("--device", type=str, default=None)
    p.add_argument("--max_frames", type=int, default=0,
                   help="Optional cap on number of frames (0 = all)")
    p.add_argument("--out", type=str,
                   default="outputs/runs/viz/ekf_vs_raw.png",
                   help="Output PNG path for visualization")
    p.add_argument("--use_vel_meas", action="store_true",
                   help="Whether EKF uses velocity measurements")
    p.add_argument("--model", type=str,
                   choices=["temporal", "tiny", "two_stage"],
                   default="temporal",
                   help="Which backbone type PosePipeline should use")
    p.add_argument("--detector_ckpt", type=str, default=None,
                   help="Path to detector ckpt (required if --model two_stage)")
    p.add_argument("--position_ckpt", type=str, default=None,
                   help="Path to position ckpt (required if --model two_stage)")
    return p.parse_args()


def polyline(img: np.ndarray, pts: np.ndarray, color, thickness: int = 2):
    """
    Draw a polyline through pts (N,2) on img in-place.
    color is BGR tuple, e.g. (0,0,255) for red.
    """
    if pts.shape[0] < 2:
        return
    pts_i32 = np.round(pts).astype(np.int32).reshape(-1, 1, 2)
    cv2.polylines(img, [pts_i32], isClosed=False, color=color, thickness=thickness)


def main():
    args = build_args()
    device = args.device or ("cuda" if torch.cuda.is_available() else "cpu")

    # -------------------------------------------------------------------------
    # Load frames from video
    # -------------------------------------------------------------------------
    frames_np = data_mod.load_video_frames(
        args.video_path,
        max_frames=args.max_frames,
        resize=(224, 224),
    )
    if frames_np.shape[0] == 0:
        raise RuntimeError(f"No frames loaded from {args.video_path}")
    T, H, W, _ = frames_np.shape

    batch = data_mod.to_tensor_batch(frames_np, device)  # FrameBatch with .frames (T,C,H,W)

    # -------------------------------------------------------------------------
    # Build model (with EKF for filtered pass)
    # -------------------------------------------------------------------------
    if args.model == "two_stage":
        if args.detector_ckpt is None or args.position_ckpt is None:
            raise ValueError("two_stage model requires --detector_ckpt and --position_ckpt")

    model = PosePipeline(
        scenario=args.scenario,
        device=device,
        filter_type="ekf",
        use_vel_meas=args.use_vel_meas,
        model_type=args.model,
        detector_ckpt=args.detector_ckpt,
        position_ckpt=args.position_ckpt,
    )
    model.eval()

    # -------------------------------------------------------------------------
    # Run raw backbone vs EKF-smoothed trajectories
    # -------------------------------------------------------------------------
    with torch.no_grad():
        # Raw backbone predictions (no temporal filter)
        x_seq = batch.frames.unsqueeze(0)               # (1, T, C, H, W)
        y_raw = model.backbone(x_seq).squeeze(0)        # (T, D), D>=2 (4 for temporal/tiny, 3 for two_stage)
        xy_raw = y_raw[:, :2].detach().cpu().numpy()    # (T,2)

        # EKF-smoothed predictions: predict_and_smooth expects (T,C,H,W)
        y_ekf = model.predict_and_smooth(batch.frames)  # (T,4)
        xy_ekf = y_ekf[:, :2].detach().cpu().numpy()    # (T,2)

    # -------------------------------------------------------------------------
    # Build visualization canvas
    # -------------------------------------------------------------------------
    # Start from the first video frame as backdrop
    # frames_np is BGR from cv2; make sure it's uint8
    backdrop = frames_np[0].copy()
    if backdrop.dtype != np.uint8:
        backdrop = np.clip(backdrop, 0, 255).astype(np.uint8)

    # Draw trajectories (BGR colors)
    polyline(backdrop, xy_raw, color=(60, 60, 220), thickness=2)   # red-ish for RAW
    polyline(backdrop, xy_ekf, color=(60, 180, 60), thickness=3)   # green-ish for EKF

    # Legend box
    cv2.rectangle(backdrop, (10, 10), (260, 85), (255, 255, 255), -1)
    cv2.putText(backdrop, "RAW (backbone)", (20, 35),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (60, 60, 220), 2)
    cv2.putText(backdrop, "EKF-smoothed", (20, 65),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (60, 180, 60), 2)

    # -------------------------------------------------------------------------
    # Save result
    # -------------------------------------------------------------------------
    out_dir = os.path.dirname(args.out)
    if out_dir:
        os.makedirs(out_dir, exist_ok=True)
    cv2.imwrite(args.out, backdrop)
    print(f"[OK] Wrote comparison image to: {args.out}")


if __name__ == "__main__":
    main()
