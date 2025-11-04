from __future__ import annotations
import torch
import torch.nn as nn

# We import torchvision lazily to avoid hard version pinning issues.
def _resnet18_backbone(pretrained: bool = True) -> nn.Module:
    try:
        import torchvision.models as models
        # Handle both old and new torchvision weights APIs
        try:
            weights = models.ResNet18_Weights.IMAGENET1K_V1 if pretrained else None
            net = models.resnet18(weights=weights)
        except Exception:
            net = models.resnet18(pretrained=pretrained)
    except Exception as e:
        raise RuntimeError(
            "torchvision is required for PoseTemporal (ResNet18 backbone). "
            "Install with `pip install torchvision`."
        ) from e
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
        if x.dim() == 4:
            # (B, C, H, W) -> add time dim
            x = x.unsqueeze(1)

        B, T, C, H, W = x.shape

        # Extract per-frame features with the CNN
        feat_list = []
        for t in range(T):
            ft = self.backbone(x[:, t])   # (B, 512)
            feat_list.append(ft)
        feats = torch.stack(feat_list, dim=1)  # (B, T, 512)

        # GRU over time
        out, _ = self.temporal(feats)          # (B, T, hidden)
        preds = self.head(out)                 # (B, T, out_dim)
        return preds
    