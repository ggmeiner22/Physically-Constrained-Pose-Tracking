from __future__ import annotations
import torch
import torch.nn as nn
from .models.backbone import TinyBackbone
from .config import PhysicsParams, LossWeights
from .filtering import SimpleSmoother, EKFTracker, EKFConfig


class PosePipeline(nn.Module):
    def __init__(self, scenario: str, device: str, filter_type: str = "ekf", use_vel_meas: bool = False):
        super().__init__()
        self.backbone = TinyBackbone().to(device)
        self.scenario = scenario
        self.params = PhysicsParams()
        self.weights = LossWeights()
        self.smoother = SimpleSmoother(alpha=0.25)
        self.device = device
        # Choose temporal filter
        if filter_type == "ekf":
            self.tracker = EKFTracker(EKFConfig(dt=self.params.dt, gravity=self.params.gravity, use_velocity_measurement=use_vel_meas))
            self.filter_type = "ekf"
        elif filter_type == "ema":
            self.tracker = SimpleSmoother(alpha=0.25)
            self.filter_type = "ema"
        else:
            raise ValueError("filter_type must be 'ekf' or 'ema'")
        
    def forward(self, frames: torch.Tensor) -> torch.Tensor:
        return self.backbone(frames)
        
        
    @torch.no_grad()
    def predict_and_smooth(self, frames: torch.Tensor) -> torch.Tensor:
        y = self.forward(frames)
        outs = []
        for i in range(y.shape[0]):
            outs.append(self.smoother(y[i:i+1]))
        return torch.cat(outs, dim=0)