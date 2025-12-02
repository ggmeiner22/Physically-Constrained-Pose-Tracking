import torch.nn as nn
import torch
import torchvision.models as models


class PositionNet(nn.Module):
    def __init__(self, backbone_name="resnet18", in_bbox_feats=4, out_dim=3):
        super().__init__()
        backbone = models.resnet18(weights="IMAGENET1K_V1")
        in_feats = backbone.fc.in_features
        backbone.fc = nn.Identity()
        self.backbone = backbone
        self.mlp = nn.Sequential(
            nn.Linear(in_feats + in_bbox_feats, 256),
            nn.ReLU(),
            nn.Linear(256, out_dim)
        )

    def forward(self, crop, bbox_norm):
        feats = self.backbone(crop)
        x = torch.cat([feats, bbox_norm], dim=-1)
        pos = self.mlp(x)
        return pos
