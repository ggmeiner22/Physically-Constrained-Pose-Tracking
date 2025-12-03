from __future__ import annotations
from typing import Optional

import torch
import torch.nn as nn

from .models.backbone import TinyBackbone
from .models.two_stage_backbone import TwoStageBackbone
from .config import PhysicsParams, LossWeights
from .filtering import SimpleSmoother, EKFTracker, EKFConfig

try:
    from .models.pose_temporal import PoseTemporal
except Exception:  # noqa: BLE001
    PoseTemporal = None  # handled at runtime


class PosePipeline(nn.Module):
    def __init__(self, scenario: str, device: str, filter_type: str, use_vel_meas: bool, model_type: str = "temporal",
                 detector_ckpt: str | None = None,
                 position_ckpt: str | None = None):
        super().__init__()
        self.device = torch.device(device)
        self.filter_type = filter_type
        self.use_vel_meas = use_vel_meas
        ...
        if model_type == "tiny":
            self.backbone = TinyBackbone().to(self.device)
        elif model_type == "temporal":
            if PoseTemporal is None:
                raise RuntimeError("PoseTemporal not available, check torchvision install")
            self.backbone = PoseTemporal(out_dim=4).to(self.device)
        elif model_type == "two_stage":
            if detector_ckpt is None or position_ckpt is None:
                raise ValueError("two_stage model_type requires --detector_ckpt and --position_ckpt")
            self.backbone = TwoStageBackbone(
                detector_ckpt=detector_ckpt,
                position_ckpt=position_ckpt,
                device=self.device,
            )
        else:
            raise ValueError(f"Unknown model_type: {model_type}")


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

        # ---- 1) Run backbone ----
        if isinstance(self.backbone, TwoStageBackbone):
            # two-stage: returns positions (B,T,3)
            pos_bt = self.backbone(frames.unsqueeze(0))  # (1,T,3)
            pos_seq = pos_bt[0]                          # (T,3)
            y_seq = _positions_to_measurements(pos_seq, dt=self.params.dt)  # (T,4)
        elif getattr(self.backbone, "expects_sequence", False):
            # temporal backbones: already output (B,T,4)
            y_bt = self.backbone(frames.unsqueeze(0))  # (1,T,4)
            y_seq = y_bt[0]
        else:
            # per-frame backbone: map each frame to (4,)
            ys = []
            for t in range(T):
                yt = self.backbone(frames[t : t + 1])  # (1,4)
                ys.append(yt[0])
            y_seq = torch.stack(ys, dim=0)             # (T,4)

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


def _positions_to_measurements(pos_seq: torch.Tensor, dt: float) -> torch.Tensor:
    """
    pos_seq: (T,3) or (T,2) positions
    returns: (T,4) [x, y, vx, vy]
    """
    # take only x,y
    xy = pos_seq[:, :2]                     # (T,2)
    T = xy.shape[0]
    v = torch.zeros_like(xy)               # (T,2)

    if T > 1:
        v[1:] = (xy[1:] - xy[:-1]) / max(dt, 1e-6)
        v[0] = v[1]                        # copy first velocity for simplicity

    y_seq = torch.cat([xy, v], dim=-1)     # (T,4)
    return y_seq
