from __future__ import annotations
import os
import argparse
import numpy as np
import torch
import cv2


from pcpose.data import load_video_frames, to_tensor_batch
from pcpose.pipeline import PosePipeline


def build_args():
    p = argparse.ArgumentParser()
    p.add_argument("--video_path", type=str, required=True)
    p.add_argument("--scenario", type=str, choices=["hanging", "sliding", "dropping"], default="hanging")
    p.add_argument("--device", type=str, default=None)
    p.add_argument("--max_frames", type=int, default=0)
    p.add_argument("--out", type=str, default="outputs/runs/viz/ekf_vs_raw.png")
    p.add_argument("--use_vel_meas", action="store_true")
    return p.parse_args()


def polyline(img: np.ndarray, pts: np.ndarray, color, thickness: int = 2):
    pts_i32 = np.round(pts).astype(np.int32).reshape(-1, 1, 2)
    cv2.polylines(img, [pts_i32], isClosed=False, color=color, thickness=thickness)


def main():
    args = build_args()
    device = args.device or ("cuda" if torch.cuda.is_available() else "cpu")


    frames_np = load_video_frames(args.video_path, max_frames=args.max_frames, resize=(224, 224))
    if frames_np.shape[0] == 0:
        raise RuntimeError("No frames loaded.")
    T, H, W, _ = frames_np.shape

    batch = to_tensor_batch(frames_np, device)

    # Model with EKF for filtered pass
    model = PosePipeline(args.scenario, device, filter_type="ekf", use_vel_meas=args.use_vel_meas)

    with torch.no_grad():
        # Raw backbone predictions (no temporal filter)
        x_seq = batch.frames.unsqueeze(0)           # (1, T, C, H, W)
        y_raw = model.backbone(x_seq).squeeze(0)    # (T, 4)
        xy_raw = y_raw[:, :2].detach().cpu().numpy()
        # EKF-smoothed predictions
        try:
            y_ekf = model.predict_and_smooth(x_seq).squeeze(0)   # (T,4)
        except Exception:
            y_ekf = model.predict_and_smooth(batch.frames)       # (T,4)
        xy_ekf = y_ekf[:, :2].detach().cpu().numpy()

    # Build a side-by-side canvas for comparison
    canvas = np.full((H, W, 3), 255, dtype=np.uint8)
    # Draw faint frame backdrop (first frame)
    backdrop = cv2.addWeighted(frames_np[0], 0.6, canvas, 0.4, 0)

    # Draw trajectories
    polyline(backdrop, xy_raw, color=(60, 60, 220), thickness=2) # red-ish for RAW
    polyline(backdrop, xy_ekf, color=(60, 180, 60), thickness=3) # green-ish for EKF

    # Legend
    cv2.rectangle(backdrop, (10, 10), (230, 80), (255, 255, 255), -1)
    cv2.putText(backdrop, "RAW (backbone)", (20, 35), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (60, 60, 220), 2)
    cv2.putText(backdrop, "EKF-smoothed", (20, 65), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (60, 180, 60), 2)


    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    cv2.imwrite(args.out, backdrop)
    print(f"[OK] Wrote comparison image to: {args.out}")


if __name__ == "__main__":
    main()