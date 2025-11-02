from __future__ import annotations
from dataclasses import dataclass
from typing import Tuple
import torch


def device_default() -> str:
    return "cuda" if torch.cuda.is_available() else "cpu"


@dataclass
class CLIConfig:
    video_path: str | None = None
    scenario: str = "hanging" # choices: hanging, sliding, dropping
    device: str = device_default()
    epochs: int = 10
    lr: float = 1e-3
    lambda_phys: float = 5.0
    lambda_smooth: float = 0.5
    outdir: str = "outputs/runs"
    save_every: int = 5
    max_frames: int = 0
    filter_type: str = "ekf" # choices: ekf, ema
    use_vel_meas: bool = False # if True, EKF consumes (x,y,vx,vy) as measurement; else only (x,y)


@dataclass
class PhysicsParams:
    gravity: float = 9.81
    dt: float = 1.0 / 30.0
    # hanging
    anchor: Tuple[float, float] = (128.0, 0.0)
    cable_length: float = 120.0
    # sliding
    slide_y: float = 128.0


@dataclass
class LossWeights:
    lambda_phys: float = 5.0
    lambda_smooth: float = 0.5
