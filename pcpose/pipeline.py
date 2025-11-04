from __future__ import annotations
import torch
import torch.nn as nn
from .models.backbone import TinyBackbone
from .config import PhysicsParams, LossWeights
from .filtering import SimpleSmoother, EKFTracker, EKFConfig

try:
    from .models.pose_temporal import PoseTemporal
except Exception:
    PoseTemporal = None  # handled at runtime


class PosePipeline(nn.Module):
    """
    Wraps the visual backbone (tiny/resnet_temporal) and a temporal filter (ekf/ema).
    model_type:
        - "tiny"              -> TinyBackbone (per-frame)
        - "temporal"          -> ResNet18 + GRU (sequence-aware)
    """
    def __init__(self, scenario: str, device: str,
                 filter_type: str = "ekf",
                 use_vel_meas: bool = False,
                 model_type: str = "temporal"):
        super().__init__()
        self.scenario = scenario
        self.params = PhysicsParams()
        self.weights = LossWeights()
        self.device = device
        self.model_type = model_type

        # Choose backbone
        if model_type == "temporal":
            if PoseTemporal is None:
                raise RuntimeError("PoseTemporal unavailable. Install torchvision and add models/pose_temporal.py.")
            self.backbone = PoseTemporal(pretrained=True).to(device)
        elif model_type == "tiny":
            self.backbone = TinyBackbone().to(device)
        else:
            raise ValueError("model_type must be 'temporal' or 'tiny'")

        # Choose temporal filter
        if filter_type == "ekf":
            self.tracker = EKFTracker(EKFConfig(dt=self.params.dt,
                                               gravity=self.params.gravity,
                                               use_velocity_measurement=use_vel_meas))
            self.filter_type = "ekf"
        elif filter_type == "ema":
            self.tracker = SimpleSmoother(alpha=0.25)
            self.filter_type = "ema"
        else:
            raise ValueError("filter_type must be 'ekf' or 'ema'")

    def forward(self, frames: torch.Tensor) -> torch.Tensor:
        """
        For per-frame backbones:
            frames: (B, C, H, W) -> (B, 4)
        For temporal backbone:
            frames: (B, T, C, H, W) or (B, C, H, W) -> (B, T, 4)
        """
        return self.backbone(frames)

    @torch.no_grad()
    def predict_and_smooth(self, frames: torch.Tensor) -> torch.Tensor:
        """
        Convenience for inference on 1 sequence of T frames:
            input: frames (T, C, H, W)
            returns: (T, 4) filtered states
        """
        if frames.dim() != 4:
            raise ValueError("predict_and_smooth expects (T, C, H, W)")

        # Forward through backbone
        if getattr(self.backbone, "expects_sequence", False):
            # Make a batch of 1 sequence
            y_seq = self.backbone(frames.unsqueeze(0))  # (1, T, 4)
            y_seq = y_seq.squeeze(0)                    # (T, 4)
        else:
            # Per-frame pass
            y_list = [self.backbone(frames[t:t+1]) for t in range(frames.shape[0])]
            y_seq = torch.cat(y_list, dim=0)           # (T, 4)

        # Run filter sequentially
        outs = []
        if self.filter_type == "ekf":
            for t in range(y_seq.shape[0]):
                x_f = self.tracker.step(y_seq[t])      # (4,)
                outs.append(x_f.unsqueeze(0))
            return torch.cat(outs, dim=0)              # (T, 4)
        else:
            for t in range(y_seq.shape[0]):
                outs.append(self.tracker(y_seq[t:t+1]))
            return torch.cat(outs, dim=0)              # (T, 4)
        