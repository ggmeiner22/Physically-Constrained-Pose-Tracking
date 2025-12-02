from __future__ import annotations

import argparse
from pathlib import Path

import cv2
import numpy as np


def build_args() -> argparse.Namespace:
    p = argparse.ArgumentParser("Visualize one sim with bbox + position overlay")
    p.add_argument("--video", type=str, required=True,
                   help="Path to the .mp4 video (e.g., datasets/marble/train/...mp4)")
    p.add_argument("--info", type=str, required=True,
                   help="Path to the matching .npy info file (shape (N,16))")
    p.add_argument("--out", type=str, default=None,
                   help="Output video path (default: <video_stem>_viz.mp4 in same folder)")
    p.add_argument("--draw_pos", action="store_true",
                   help="Also draw (x,y) position as a point (using cols 0:2).")
    p.add_argument("--scale", type=float, default=1.0,
                   help="Optional resize factor for output (e.g., 0.5).")
    return p.parse_args()


def main() -> None:
    args = build_args()
    video_path = Path(args.video)
    info_path = Path(args.info)

    if args.out is None:
        out_path = video_path.with_name(video_path.stem + "_viz.mp4")
    else:
        out_path = Path(args.out)

    print(f"[INFO] Video: {video_path}")
    print(f"[INFO] Info : {info_path}")
    print(f"[INFO] Out  : {out_path}")

    # ---- Load metadata (.npy) ----
    meta = np.load(str(info_path))  # (N, 16)
    if meta.ndim != 2 or meta.shape[1] < 16:
        raise ValueError(f"Expected (N,16) array, got {meta.shape}")

    # Positions (optional)
    pos_xyz = meta[:, 0:3]          # (N,3)
    # Bboxes in pixel coords
    bboxes = meta[:, 12:16]         # (N,4) = x_min, y_min, x_max, y_max

    # ---- Open video ----
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise IOError(f"Could not open video: {video_path}")

    fps = cap.get(cv2.CAP_PROP_FPS)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    print(f"[INFO] Video FPS={fps}, frames={total_frames}, size=({width}x{height})")
    print(f"[INFO] Meta frames={meta.shape[0]}")

    n_meta = meta.shape[0]
    n_frames = min(total_frames, n_meta)
    if total_frames != n_meta:
        print(f"[WARN] Video frames ({total_frames}) != meta rows ({n_meta}). "
              f"Using min={n_frames} frames.")

    # Output size
    out_w = int(width * args.scale)
    out_h = int(height * args.scale)

    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    out_vid = cv2.VideoWriter(str(out_path), fourcc, fps, (out_w, out_h))

    # ---- Iterate frames ----
    frame_idx = 0
    while frame_idx < n_frames:
        ok, frame = cap.read()
        if not ok:
            print("[WARN] Ran out of video frames early.")
            break

        # Draw bbox
        x_min, y_min, x_max, y_max = bboxes[frame_idx]
        p1 = (int(x_min), int(y_min))
        p2 = (int(x_max), int(y_max))

        cv2.rectangle(frame, p1, p2, (0, 255, 0), 2)  # green bbox

        # Optionally draw projected (x,y) as a point
        if args.draw_pos:
            x, y, _ = pos_xyz[frame_idx]
            # If your (x,y) are in some other coordinate system, you may
            # need to scale/shift; for now assume they're in pixel-ish units.
            cx = int(x)
            cy = int(y)
            cv2.circle(frame, (cx, cy), 4, (0, 0, 255), -1)  # red dot
            cv2.putText(frame, f"({x:.2f}, {y:.2f})", (cx + 5, cy - 5),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 0, 255), 1, cv2.LINE_AA)

        # Label frame index
        cv2.putText(frame, f"frame {frame_idx}", (10, 20),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2, cv2.LINE_AA)

        # Resize if needed
        if args.scale != 1.0:
            frame = cv2.resize(frame, (out_w, out_h), interpolation=cv2.INTER_AREA)

        out_vid.write(frame)
        frame_idx += 1

    cap.release()
    out_vid.release()
    print(f"[OK] Wrote annotated video to {out_path}")


if __name__ == "__main__":
    main()
