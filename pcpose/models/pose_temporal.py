from __future__ import annotations
import torch
import torch.nn as nn
import torchvision.models as models

# We import torchvision lazily to avoid hard version pinning issues.
def _resnet18_backbone(pretrained: bool = True) -> nn.Module:
    try:
        weights = models.ResNet18_Weights.IMAGENET1K_V1 if pretrained else None
        net = models.resnet18(weights=weights)
    except Exception:
        net = models.resnet18(pretrained=pretrained)
    # Strip classification head; keep global pooled 512-dim feature
    net.fc = nn.Identity()
    return net

class PoseTemporal(nn.Module):
    """
    ResNet18 (per-frame features) + GRU over time -> (x, y, vx, vy) per frame.
    Expects inputs as (B, T, C, H, W). If you pass (B, C, H, W) it will assume T=1.
    """
    expects_sequence: bool = True

    def __init__(self, hidden_size: int = 128, out_dim: int = 4, pretrained: bool = True):
        super().__init__()
        self.backbone = _resnet18_backbone(pretrained=pretrained)  # outputs 512-dim per frame
        self.temporal = nn.GRU(input_size=512, hidden_size=hidden_size, batch_first=True)
        self.head = nn.Linear(hidden_size, out_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        x: (B, T, C, H, W) or (B, C, H, W)
        returns: (B, T, out_dim)
        """
        # Normalize shapes:
        # - (B,C,H,W) -> (B,1,C,H,W)
        # - (B,1,T,C,H,W) -> (B,T,C,H,W)
        if x.dim() == 4:
            x = x.unsqueeze(1)
        elif x.dim() == 6 and x.size(1) == 1:
            x = x.squeeze(1)
        elif x.dim() != 5:
            raise ValueError(f"PoseTemporal expected (B,T,C,H,W) or (B,C,H,W), got {tuple(x.shape)}")
    
        B, T, C, H, W = x.shape
    
        # Vectorized CNN pass over all frames (faster than looping over T)
        x_flat = x.reshape(B * T, C, H, W)          # (B*T,C,H,W)
        feats_flat = self.backbone(x_flat)          # (B*T,512)
        feats = feats_flat.view(B, T, -1)           # (B,T,512)
    
        out, _ = self.temporal(feats)               # (B,T,hidden)
        preds = self.head(out)                      # (B,T,out_dim)
        return preds

    