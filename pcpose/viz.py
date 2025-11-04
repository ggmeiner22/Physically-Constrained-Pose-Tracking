from __future__ import annotations
from typing import Tuple
import os
import numpy as np
import cv2


def overlay_prediction(frame: np.ndarray, xy: Tuple[float, float], color=(0, 255, 0)) -> np.ndarray:
    vis = frame.copy()
    x, y = int(xy[0]), int(xy[1])
    cv2.circle(vis, (x, y), 4, color, -1)
    return vis


def render_overlay_sequence(frames_np: np.ndarray, traj_xy: np.ndarray, out_path: str, fps: int = 30):
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    H, W = frames_np.shape[1:3]
    writer = cv2.VideoWriter(out_path, cv2.VideoWriter_fourcc(*"mp4v"), fps, (W, H))
    for t in range(len(frames_np)):
        xy = traj_xy[t]
        vis = overlay_prediction(frames_np[t], (xy[0], xy[1]))
        writer.write(vis)
    writer.release()
    