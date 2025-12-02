from __future__ import annotations
from typing import Tuple

import numpy as np
import torch
import cv2


def to_tensor(img: np.ndarray) -> torch.Tensor:
    """
    Convert a single HxWx3 uint8 or float image (RGB) into a
    torch tensor (3, H, W) normalized to [0,1].
    """
    if not isinstance(img, np.ndarray):
        raise TypeError("to_tensor expects a numpy array (H, W, 3).")

    if img.ndim != 3 or img.shape[2] != 3:
        raise ValueError(f"to_tensor expects shape (H, W, 3), got {img.shape}.")

    if img.dtype != np.float32 and img.dtype != np.float64:
        img = img.astype(np.float32) / 255.0
    img = np.clip(img, 0.0, 1.0)

    tensor = torch.from_numpy(img).float()        # (H,W,3)
    tensor = tensor.permute(2, 0, 1).contiguous() # (3,H,W)
    return tensor


def build_crops(
    frames: torch.Tensor,
    bboxes: torch.Tensor,
    H: int,
    W: int,
    pad: float = 0.1,
    out_size: int = 128,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Build cropped image tensors and normalized bbox features.

    Args:
        frames: (N, C, H, W) float tensor in [0,1] or [0,255].
        bboxes: (N, 4) tensor in pixel coords [x_min, y_min, x_max, y_max].
        H, W : original frame height/width.
        pad  : extra padding around the bbox as a fraction of the max side.
        out_size: crop output size (square, out_size x out_size).

    Returns:
        crops: (N, 3, out_size, out_size) float tensor in [0,1]
        bbox_norm: (N, 4) tensor with coords normalized to [0,1]
    """
    if not isinstance(frames, torch.Tensor):
        raise TypeError("frames must be a torch.Tensor of shape (N, C, H, W).")
    if not isinstance(bboxes, torch.Tensor):
        bboxes = torch.as_tensor(bboxes, dtype=torch.float32)

    frames_cpu = frames.detach().cpu()           # (N,C,H,W)
    bboxes_cpu = bboxes.detach().cpu()           # (N,4)

    N = frames_cpu.shape[0]
    crops = []
    bbox_norm = []

    for i in range(N):
        x_min, y_min, x_max, y_max = bboxes_cpu[i].tolist()

        w_box = x_max - x_min
        h_box = y_max - y_min
        cx = (x_min + x_max) / 2.0
        cy = (y_min + y_max) / 2.0

        scale = 1.0 + pad
        half = 0.5 * scale * max(w_box, h_box)

        x0 = int(max(0, cx - half))
        x1 = int(min(W, cx + half))
        y0 = int(max(0, cy - half))
        y1 = int(min(H, cy + half))

        # (C,H,W) -> (H,W,C)
        frame_chw = frames_cpu[i]
        frame_hwc = frame_chw.permute(1, 2, 0).numpy()

        # Assume frames are in [0,1]; convert to uint8 for OpenCV
        if frame_hwc.dtype != np.uint8:
            frame_hwc = np.clip(frame_hwc * 255.0, 0, 255).astype(np.uint8)

        crop = frame_hwc[y0:y1, x0:x1]
        if crop.size == 0:
            crop = np.zeros((out_size, out_size, 3), dtype=np.uint8)
        else:
            crop = cv2.resize(crop, (out_size, out_size), interpolation=cv2.INTER_LINEAR)

        crop_t = torch.from_numpy(crop).float() / 255.0
        crop_t = crop_t.permute(2, 0, 1).contiguous()  # (3,hc,wc)
        crops.append(crop_t)

        bbox_norm.append([
            x_min / W,
            y_min / H,
            x_max / W,
            y_max / H,
        ])

    crops_t = torch.stack(crops, dim=0)                 # (N,3,out_size,out_size)
    bbox_norm_t = torch.tensor(bbox_norm, dtype=torch.float32)  # (N,4)

    return crops_t, bbox_norm_t
