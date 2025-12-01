import torch
import torch.nn as nn
from torchvision.models import resnet18

class BoundingBoxNet(nn.Module):
    def __init__(self, pretrained=True):
        super().__init__()
        backbone = resnet18(weights="IMAGENET1K_V1" if pretrained else None)
        in_feats = backbone.fc.in_features
        backbone.fc = nn.Identity()
        self.backbone = backbone
        self.head = nn.Linear(in_feats, 4)  # x_min, y_min, x_max, y_max

    def forward(self, x):
        feats = self.backbone(x)
        bbox = self.head(feats)
        return bbox
