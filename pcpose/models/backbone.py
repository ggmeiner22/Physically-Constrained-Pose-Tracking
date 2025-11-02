from __future__ import annotations
import torch
import torch.nn as nn




class TinyBackbone(nn.Module):
    """Very small CNN placeholder. Replace with ViT/OpenPose/keypoints regressor."""
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
        nn.Conv2d(3, 16, 3, padding=1), nn.ReLU(),
        nn.MaxPool2d(2),
        nn.Conv2d(16, 32, 3, padding=1), nn.ReLU(),
        nn.MaxPool2d(2),
        nn.Conv2d(32, 64, 3, padding=1), nn.ReLU(),
        nn.AdaptiveAvgPool2d((1, 1))
        )
        self.head = nn.Sequential(
        nn.Flatten(),
        nn.Linear(64, 64), nn.ReLU(),
        nn.Linear(64, 4) # [x,y,vx,vy]
        )


def forward(self, x: torch.Tensor) -> torch.Tensor:
    feats = self.net(x)
    return self.head(feats)