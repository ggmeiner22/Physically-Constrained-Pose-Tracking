from __future__ import annotations

import argparse
import os

import torch
from torch.utils.data import DataLoader
from tqdm.auto import tqdm

from pcpose.data_video_dataset import VideoClipDataset
from pcpose.models.detector import BoundingBoxNet
from pcpose.models.position_head import PositionNet
from pcpose.models.pose_model import PoseModel
from pcpose.models.physics_layers import EMAPositionSmoother, EKFPositionLayer
from pcpose.config import PhysicsParams


def build_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        "Run PoseModel (detector + position + physics) on video clips"
    )
    p.add_argument(
        "--manifest",
        type=str,
        required=True,
        help="CSV manifest for VideoClipDataset (same as train.py uses)",
    )
    p.add_argument(
        "--scenario",
        type=str,
        default="hanging",
        choices=["hanging", "sliding", "dropping"],
        help="Scenario (matches how you configure things in train.py)",
    )
    p.add_argument(
        "--detector_ckpt",
        type=str,
        default="outputs/detector/best_detector.pt",
    )
    p.add_argument(
        "--position_ckpt",
        type=str,
        default="outputs/position/best_position.pt",
    )
    p.add_argument("--clip_len", type=int, default=32)
    p.add_argument("--batch_size", type=int, default=2)
    p.add_argument("--num_workers", type=int, default=2)
    p.add_argument(
        "--device",
        type=str,
        default=None,
        help="Override device (default: auto cuda/cpu)",
    )
    p.add_argument(
        "--resize",
        type=int,
        nargs=2,
        default=[224, 224],
        help="Resize (H W) used by VideoClipDataset",
    )
    p.add_argument(
        "--sampling",
        type=str,
        default="sequential",
        choices=["sequential", "random"],
    )
    p.add_argument(
        "--use_ema",
        action="store_true",
        help="If set, use EMA smoother instead of EKF (for debugging).",
    )
    return p.parse_args()


def _build_loader(args: argparse.Namespace, device: str) -> DataLoader:
    ds = VideoClipDataset(
        manifest_path=args.manifest,
        clip_len=args.clip_len,
        resize=tuple(args.resize),
        device=device,
        sampling=args.sampling,
        seed=42,
    )

    def collate(batch):
        # Matches train.py collate
        frames = torch.stack([b["frames"] for b in batch], dim=0)  # (B,T,C,H,W)
        scenarios = [b["scenario"] for b in batch]
        params = [b["params"] for b in batch]
        videos = [b["video_path"] for b in batch]
        return {
            "frames": frames,
            "scenarios": scenarios,
            "params": params,
            "videos": videos,
        }

    loader = DataLoader(
        ds,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        collate_fn=collate,
        drop_last=False,
    )
    return loader


def load_detector(path: str, device: str) -> BoundingBoxNet:
    model = BoundingBoxNet(pretrained=False).to(device)
    ckpt = torch.load(path, map_location=device)
    state = ckpt.get("state_dict", ckpt)
    model.load_state_dict(state)
    model.eval()
    return model


def load_position_net(path: str, device: str) -> PositionNet:
    model = PositionNet(backbone_name="resnet18", in_bbox_feats=4, out_dim=3).to(device)
    ckpt = torch.load(path, map_location=device)
    state = ckpt.get("state_dict", ckpt)
    model.load_state_dict(state)
    model.eval()
    return model


def main() -> None:
    args = build_args()
    device = args.device or ("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[run_pose_model] Using device: {device}")
    print(f"[run_pose_model] Scenario: {args.scenario}")

    if not os.path.exists(args.detector_ckpt):
        raise FileNotFoundError(f"Detector checkpoint not found: {args.detector_ckpt}")
    if not os.path.exists(args.position_ckpt):
        raise FileNotFoundError(f"Position checkpoint not found: {args.position_ckpt}")

    # 1) Load trained networks
    detector = load_detector(args.detector_ckpt, device)
    position_net = load_position_net(args.position_ckpt, device)

    # 2) Pull dt, gravity from PhysicsParams so we match train.py
    phys_params = PhysicsParams()
    dt = phys_params.dt
    gravity = phys_params.gravity

    print(f"[run_pose_model] PhysicsParams: dt={dt}, gravity={gravity}")

    if args.use_ema:
        physics_layer = EMAPositionSmoother(alpha=0.25).to(device)
        print("[run_pose_model] Using EMAPositionSmoother (no EKF).")
    else:
        physics_layer = EKFPositionLayer(
            dt=dt,
            gravity=gravity,
            use_velocity_measurement=True,
            q_pos=1e-2,
            q_vel=1e-1,
            r_pos=5.0,
            r_vel=1.0,
        ).to(device)
        print("[run_pose_model] Using EKFPositionLayer with PhysicsParams dt/gravity.")

    # 3) Build PoseModel
    pose_model = PoseModel(
        detector=detector,
        position_net=position_net,
        physics_layer=physics_layer,
        crop_pad=0.1,
        crop_size=128,
    ).to(device)
    pose_model.eval()

    # 4) Data loader of clips
    loader = _build_loader(args, device=device)

    # 5) Run a couple batches for sanity-check
    for batch_idx, batch in enumerate(tqdm(loader, desc="PoseModel inference")):
        frames = batch["frames"].to(device)  # (B,T,C,H,W)

        with torch.no_grad():
            positions, bboxes = pose_model(frames)  # positions: (B,T,3), bboxes: (B,T,4)

        print(f"\nBatch {batch_idx}:")
        print(f"  frames:    {frames.shape}")
        print(f"  positions: {positions.shape}  (B,T,3)")
        print(f"  bboxes:    {bboxes.shape}     (B,T,4)")
        print(f"  first sample, first frame position: {positions[0,0].cpu().numpy()}")
        print(f"  first sample, first frame bbox:     {bboxes[0,0].cpu().numpy()}")

        # Just do a couple for debugging
        if batch_idx >= 1:
            break


if __name__ == "__main__":
    main()
