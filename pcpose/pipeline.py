from __future__ import annotations
from typing import Optional

import torch
import torch.nn as nn

from .models.backbone import TinyBackbone
from .config import PhysicsParams, LossWeights
from .filtering import SimpleSmoother, EKFTracker, EKFConfig

try:
    from .models.pose_temporal import PoseTemporal
except Exception:  # noqa: BLE001
    PoseTemporal = None  # handled at runtime


class PosePipeline(nn.Module):
    """
    Wraps the visual backbone (tiny / resnet_temporal) and a temporal filter (EKF / EMA).

    model_type:
        - "tiny"     -> TinyBackbone (per-frame)
        - "temporal" -> ResNet18 + GRU (sequence-aware)
    """
    def __init__(
        self,
        scenario: str,
        device: str,
        filter_type: str = "ekf",
        use_vel_meas: bool = False,
        model_type: str = "temporal",
    ) -> None:
        super().__init__()
        self.scenario = scenario
        self.params = PhysicsParams()
        self.weights = LossWeights()
        self.device = device
        self.model_type = model_type

        # ---- Visual backbone ----
        if model_type == "temporal":
            if PoseTemporal is None:
                raise RuntimeError(
                    "PoseTemporal unavailable. Install torchvision and add models/pose_temporal.py."
                )
            self.backbone: nn.Module = PoseTemporal(pretrained=True).to(device)
            # used by train.py to know it can feed (B,T,C,H,W)
            self.backbone.expects_sequence = True  # type: ignore[attr-defined]
        elif model_type == "tiny":
            self.backbone = TinyBackbone().to(device)
            self.backbone.expects_sequence = False  # type: ignore[attr-defined]
        else:
            raise ValueError("model_type must be 'temporal' or 'tiny'")

        # ---- Temporal filter ----
        if filter_type == "ekf":
            self._filter_cfg = EKFConfig(
                dt=self.params.dt,
                gravity=self.params.gravity,
                use_velocity_measurement=use_vel_meas,
            )
            self.filter_type = "ekf"
            # We will re-create EKFTracker in predict_and_smooth() to avoid stale state
        elif filter_type == "ema":
            self.tracker = SimpleSmoother(alpha=0.25)
            self.filter_type = "ema"
        else:
            raise ValueError("filter_type must be 'ekf' or 'ema'")

    # -------------------------------------------------------------------------
    # Core forward: used by training. train.py calls model.backbone(...) directly,
    # but we keep this for completeness.
    # -------------------------------------------------------------------------
    def forward(self, frames: torch.Tensor) -> torch.Tensor:
        """
        For per-frame backbones:
            frames: (B, C, H, W) -> (B, 4)
        For temporal backbone:
            frames: (B, T, C, H, W) or (B, C, H, W) -> (B, T, 4) or (B, 4)
        """
        return self.backbone(frames)

    # -------------------------------------------------------------------------
    # Inference helper for a *single* sequence of frames (no batching)
    # -------------------------------------------------------------------------
    @torch.no_grad()
    def predict_and_smooth(self, frames: torch.Tensor) -> torch.Tensor:
        """
        Convenience for inference on 1 sequence of T frames:

            input:  frames (T, C, H, W)
            output: (T, 4) filtered states [x, y, vx, vy]
        """
        if frames.dim() != 4:
            raise ValueError("predict_and_smooth expects frames of shape (T, C, H, W)")

        T, C, H, W = frames.shape
        device = self.device

        # ---- 1) Run backbone to get raw measurements y_seq (T,4) ----
        if getattr(self.backbone, "expects_sequence", False):
            # backbone takes (B,T,C,H,W)
            y_bt = self.backbone(frames.unsqueeze(0))  # (1, T, 4)
            y_seq = y_bt[0]                            # (T, 4)
        else:
            # backbone is per-frame: loop over time
            ys = []
            for t in range(T):
                y_t = self.backbone(frames[t : t + 1].to(device))  # (1, 4)
                ys.append(y_t[0])
            y_seq = torch.stack(ys, dim=0)  # (T, 4)

        # ---- 2) Apply temporal filter (EKF or EMA) ----
        outs = []

        if self.filter_type == "ekf":
            # fresh EKF instance to avoid reusing old state
            tracker = EKFTracker(self._filter_cfg)
            for t in range(T):
                x_f = tracker.step(y_seq[t])          # (4,)
                outs.append(x_f.unsqueeze(0))
        else:
            # EMA smoother
            self.tracker.prev = None  # reset
            for t in range(T):
                outs.append(self.tracker(y_seq[t : t + 1]))

        return torch.cat(outs, dim=0)  # (T, 4)
