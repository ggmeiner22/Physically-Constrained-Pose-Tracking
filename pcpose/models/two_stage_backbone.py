from __future__ import annotations
from typing import Tuple

import torch
import torch.nn as nn

from pcpose.models.detector import BoundingBoxNet
from pcpose.models.position_head import PositionNet
from pcpose.data.utils import build_crops


class TwoStageBackbone(nn.Module):
    """
    Backbone that:
      frames -> detector -> bboxes
      frames + bboxes -> crops -> PositionNet -> positions

    forward(x) expects (B,T,C,H,W) and returns (B,T,3) positions.
    """
    expects_sequence = True  # so PosePipeline treats it as sequence-aware

    def __init__(
        self,
        detector_ckpt: str,
        position_ckpt: str,
        device: torch.device,
        crop_size: int = 128,
        pad: float = 0.15,
    ):
        super().__init__()
        self.device = device
        self.crop_size = crop_size
        self.pad = pad

        # ---- Detector ----
        self.detector = BoundingBoxNet()
        det_state = torch.load(detector_ckpt, map_location=device)
        # handle both {"state_dict": ...} and raw state dict
        det_sd = det_state["state_dict"] if "state_dict" in det_state else det_state
        self.detector.load_state_dict(det_sd)
        self.detector.to(device)
        self.detector.eval()  # we don't train this inside PosePipeline

        # ---- Position regressor ----
        # out_dim=3 in your PositionNet (x,y,z)
        self.position_net = PositionNet(out_dim=3)
        pos_state = torch.load(position_ckpt, map_location=device)
        pos_sd = pos_state["state_dict"] if "state_dict" in pos_state else pos_state
        self.position_net.load_state_dict(pos_sd)
        self.position_net.to(device)
        self.position_net.eval()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        x: (B,T,C,H,W) or (T,C,H,W)
        returns: (B,T,3) positions
        """
        if x.dim() == 4:
            # treat as single sequence (T,C,H,W)
            x = x.unsqueeze(0)  # (1,T,C,H,W)
        elif x.dim() != 5:
            raise ValueError(f"TwoStageBackbone expected (B,T,C,H,W) or (T,C,H,W), got {tuple(x.shape)}")

        B, T, C, H, W = x.shape
        device = self.device

        # Flatten time + batch
        frames_flat = x.view(B * T, C, H, W)  # (N,3,H,W)
        frames_flat = frames_flat.to(device)

        with torch.no_grad():
            # 1) detector: frames -> bboxes in pixel coords
            bboxes = self.detector(frames_flat)         # (N,4)
            # 2) crops + normalized bboxes
            crops, bbox_norm = build_crops(
                frames_flat,
                bboxes,
                H=H,
                W=W,
                pad=self.pad,
                out_size=self.crop_size,
            )  # crops: (N,3,c,c), bbox_norm: (N,4)

            # 3) position regressor
            pos_flat = self.position_net(crops.to(device), bbox_norm.to(device))  # (N,3)

        pos_bt = pos_flat.view(B, T, -1)  # (B,T,3)
        return pos_bt
