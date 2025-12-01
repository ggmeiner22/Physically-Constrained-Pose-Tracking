import torch.nn as nn

class PoseModel(nn.Module):
    def __init__(self, detector, position_net, physics_layer=None):
        super().__init__()
        self.detector = detector
        self.position_net = position_net
        self.physics_layer = physics_layer  # e.g., EKF or differentiable physics

    def forward(self, frames):
        # frames: (B, T, C, H, W) maybe
        B, T, C, H, W = frames.shape
        frames_flat = frames.view(B*T, C, H, W)

        bboxes = self.detector(frames_flat)           # (B*T, 4)
        # compute crops & bbox_norm here, or use a separate util function
        crops, bbox_norm = build_crops(frames_flat, bboxes, H, W)
        pos = self.position_net(crops, bbox_norm)    # (B*T, 3)
        pos_seq = pos.view(B, T, -1)

        if self.physics_layer is not None:
            pos_seq = self.physics_layer(pos_seq)

        return pos_seq, bboxes.view(B, T, -1)
