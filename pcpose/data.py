from __future__ import annotations
from dataclasses import dataclass
from typing import Optional, Tuple
import os
import cv2
import numpy as np
import torch


@dataclass
class FrameBatch:
    frames: torch.Tensor # (T,C,H,W)
    indices: torch.Tensor # (T,)


def load_video_frames(path: str, max_frames: int = 0, resize: Optional[Tuple[int, int]] = (256, 256)) -> np.ndarray:
    if path is None or not os.path.exists(path):
        raise FileNotFoundError(f"Video not found: {path}")
    cap = cv2.VideoCapture(path)
    frames = []
    t = 0
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        if resize is not None:
            frame = cv2.resize(frame, resize, interpolation=cv2.INTER_AREA)
            frames.append(frame)
            t += 1
        if max_frames > 0 and t >= max_frames:
            break
    cap.release()
    return np.stack(frames, axis=0) if frames else np.zeros((0, *(resize or (0,0)), 3), dtype=np.uint8)


def to_tensor_batch(frames_np: np.ndarray, device: str) -> FrameBatch:
    T, H, W, C = frames_np.shape
    x = torch.from_numpy(frames_np).float() / 255.0 # (T,H,W,C)
    x = x.permute(0, 3, 1, 2).contiguous() # (T,C,H,W)
    idx = torch.arange(T, device=device)
    return FrameBatch(frames=x.to(device), indices=idx)
