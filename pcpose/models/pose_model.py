from __future__ import annotations
from typing import Optional, Tuple

import torch
import torch.nn as nn

from pcpose.data.utils import build_crops


class PoseModel(nn.Module):
    """
    Two-stage perception + optional physics wrapper:

        frames (B,T,C,H,W)
          -> detector         : (B*T, 4) bounding boxes
          -> build_crops      : crops (B*T,3,hc,wc), bbox_norm (B*T,4)
          -> position_net     : positions (B*T, 3)
          -> reshape          : (B,T,3)
          -> physics_layer?   : (B,T,3) filtered / constrained

    Returns:
        positions: (B, T, 3)
        bboxes  : (B, T, 4)
    """
    def __init__(
        self,
        detector: nn.Module,
        position_net: nn.Module,
        physics_layer: Optional[nn.Module] = None,
        crop_pad: float = 0.1,
        crop_size: int = 128,
    ) -> None:
        super().__init__()
        self.detector = detector
        self.position_net = position_net
        self.physics_layer = physics_layer
        self.crop_pad = crop_pad
        self.crop_size = crop_size

    def forward(self, frames: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        frames: (B, T, C, H, W) float in [0,1] or [0,255]
        """
        if frames.dim() != 5:
            raise ValueError("PoseModel.forward expects frames with shape (B, T, C, H, W)")

        B, T, C, H, W = frames.shape
        device = frames.device

        # Flatten time into batch for detector
        frames_flat = frames.view(B * T, C, H, W)

        # --- Stage 1: bounding boxes ---
        bboxes = self.detector(frames_flat)  # (B*T, 4) in pixel coords

        # --- Stage 2: crops + normalized bboxes ---
        crops, bbox_norm = build_crops(
            frames_flat, bboxes, H=H, W=W, pad=self.crop_pad, out_size=self.crop_size
        )  # crops: (B*T,3,hc,wc), bbox_norm: (B*T,4)

        crops = crops.to(device)
        bbox_norm = bbox_norm.to(device)

        # --- Stage 3: positions ---
        positions_flat = self.position_net(crops, bbox_norm)  # (B*T, 3)

        # Reshape back to per-sequence
        positions = positions_flat.view(B, T, -1)             # (B,T,3)
        bboxes_seq = bboxes.view(B, T, -1)                    # (B,T,4)

        # Optional physics / filtering over time
        if self.physics_layer is not None:
            positions = self.physics_layer(positions)

        return positions, bboxes_seq
