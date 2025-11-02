from __future__ import annotations
from dataclasses import dataclass
from typing import Optional
import torch


class SimpleSmoother:
    """Lightweight EMA smoother on (x,y,vx,vy). Replace with EKF/UKF/PF."""
    def __init__(self, alpha: float = 0.25):
        self.alpha = alpha
        self.prev: Optional[torch.Tensor] = None


    def __call__(self, y: torch.Tensor) -> torch.Tensor:
        if self.prev is None:
            self.prev = y.detach()
            return y
        self.prev = self.alpha * y + (1 - self.alpha) * self.prev
        return self.prev
    

# -----------------------------
# Extended Kalman Filter (CV model)
# State: x = [px, py, vx, vy]^T
# Process: constant-velocity with gravity on vy
# Measurement: position only (default) or position+velocity (optional)
# -----------------------------


@dataclass
class EKFConfig:
    dt: float = 1.0 / 30.0
    q_pos: float = 1e-2 # process noise (position)
    q_vel: float = 1e-1 # process noise (velocity)
    r_pos: float = 5.0 # measurement noise (pixels)
    r_vel: float = 1.0 # measurement noise for velocity (if used)
    gravity: float = 9.81 # pixels/s^2 downward (screen +y)
    use_velocity_measurement: bool = False


class EKFTracker:
    def __init__(self, cfg: EKFConfig):
        self.cfg = cfg
        self.I = torch.eye(4)
        # Constant parts of F and Q (except gravity term, handled in control u)
        dt = cfg.dt
        self.F = torch.tensor([
            [1.0, 0.0, dt, 0.0],
            [0.0, 1.0, 0.0, dt],
            [0.0, 0.0, 1.0, 0.0],
            [0.0, 0.0, 0.0, 1.0],
        ])
        self.Q = torch.diag(torch.tensor([cfg.q_pos, cfg.q_pos, cfg.q_vel, cfg.q_vel]))
        # Measurement matrix H depends on whether we observe velocity
        if cfg.use_velocity_measurement:
            self.H = torch.eye(4)
            self.R = torch.diag(torch.tensor([cfg.r_pos, cfg.r_pos, cfg.r_vel, cfg.r_vel]))
        else:
            self.H = torch.tensor([
                [1.0, 0.0, 0.0, 0.0],
                [0.0, 1.0, 0.0, 0.0],
            ])
            self.R = torch.diag(torch.tensor([cfg.r_pos, cfg.r_pos]))
        self.x: Optional[torch.Tensor] = None # (4,)
        self.P: Optional[torch.Tensor] = None # (4,4)


def _to_device(self, device: torch.device):
    self.I = self.I.to(device)
    self.F = self.F.to(device)
    self.Q = self.Q.to(device)
    self.H = self.H.to(device)
    self.R = self.R.to(device)
    if self.x is not None:
        self.x = self.x.to(device)
    if self.P is not None:
        self.P = self.P.to(device)


def reset(self, init_state: torch.Tensor, init_cov: float = 10.0):
    """init_state: (4,) tensor"""
    self.x = init_state.clone()
    self.P = torch.eye(4, device=self.x.device) * init_cov


def predict(self):
    assert self.x is not None and self.P is not None
    # gravity control vector approximated as u = [0, 0, 0, g*dt]
    u = torch.tensor([0.0, 0.0, 0.0, self.cfg.gravity * self.cfg.dt], device=self.x.device)
    self.x = self.F @ self.x + u
    self.P = self.F @ self.P @ self.F.T + self.Q


def update(self, z: torch.Tensor):
    assert self.x is not None and self.P is not None
    # z shape: (2,) for position-only or (4,) if using velocity also
    H = self.H
    R = self.R
    y = z - (H @ self.x)
    S = H @ self.P @ H.T + R
    K = self.P @ H.T @ torch.linalg.inv(S)
    self.x = self.x + K @ y
    self.P = (self.I - K @ H) @ self.P


def step(self, meas_xyvxvy: torch.Tensor) -> torch.Tensor:
    """meas_xyvxvy: (4,) measurement from backbone; we may drop velocity if configured.
    Returns filtered state (4,).
    """
    device = meas_xyvxvy.device
    self._to_device(device)
    if self.x is None:
        # initialize from first measurement
        if self.cfg.use_velocity_measurement:
            self.reset(meas_xyvxvy.detach(), init_cov=25.0)
        else:
            m = meas_xyvxvy.detach()
            init = torch.tensor([m[0], m[1], 0.0, 0.0], device=device)
            self.reset(init, init_cov=25.0)
        return self.x
    self.predict()
    z = meas_xyvxvy if self.cfg.use_velocity_measurement else meas_xyvxvy[:2]
    self.update(z)
    return self.x