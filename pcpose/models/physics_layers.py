from __future__ import annotations

import torch
import torch.nn as nn

from pcpose.filtering import EKFTracker, EKFConfig


class EMAPositionSmoother(nn.Module):
    """
    Simple exponential moving average smoother over time.

    Input:  pos_seq (B, T, D)
    Output: smoothed (B, T, D) with EMA along the T dimension.
    """
    def __init__(self, alpha: float = 0.25) -> None:
        super().__init__()
        self.alpha = alpha

    def forward(self, pos_seq: torch.Tensor) -> torch.Tensor:
        if pos_seq.dim() != 3:
            raise ValueError("EMAPositionSmoother expects (B, T, D)")

        B, T, D = pos_seq.shape
        out = pos_seq.clone()

        for b in range(B):
            for t in range(1, T):
                out[b, t] = (
                    self.alpha * pos_seq[b, t] +
                    (1.0 - self.alpha) * out[b, t - 1]
                )

        return out


class EKFPositionLayer(nn.Module):
    """
    Wraps EKFTracker to run over a batch of position sequences.

    Assumptions:
      - Input is image-plane positions (x, y, [z]) from the network.
      - EKF state is [x, y, vx, vy].
      - We optionally build a velocity measurement by finite differences.

    Input:
        pos_seq: (B, T, D) where D>=2 (x,y[,z,...])

    Output:
        pos_filt: (B, T, D)
          - x,y filtered by EKF
          - remaining dims (like z) are passed through unchanged.
    """
    def __init__(
        self,
        dt: float = 1.0 / 30.0,
        gravity: float = 9.81,
        use_velocity_measurement: bool = True,
        q_pos: float = 1e-2,
        q_vel: float = 1e-1,
        r_pos: float = 5.0,
        r_vel: float = 1.0,
    ) -> None:
        super().__init__()
        self.cfg = EKFConfig(
            dt=dt,
            q_pos=q_pos,
            q_vel=q_vel,
            r_pos=r_pos,
            r_vel=r_vel,
            gravity=gravity,
            use_velocity_measurement=use_velocity_measurement,
        )

    def forward(self, pos_seq: torch.Tensor) -> torch.Tensor:
        """
        pos_seq: (B, T, D>=2).
        Returns (B, T, D) with x,y filtered, others unchanged.
        """
        if pos_seq.dim() != 3:
            raise ValueError("EKFPositionLayer expects (B, T, D)")

        device = pos_seq.device
        B, T, D = pos_seq.shape

        out_list = []

        for b in range(B):
            # New EKF for each sequence to avoid shared state
            tracker = EKFTracker(self.cfg)
            # tracker._to_device will be called inside step()

            seq_out = []
            prev_xy = None

            for t in range(T):
                xy = pos_seq[b, t, :2]  # (2,)
                xy = xy.to(device)

                if self.cfg.use_velocity_measurement:
                    if prev_xy is None:
                        v = torch.zeros_like(xy)
                    else:
                        v = (xy - prev_xy) / self.cfg.dt
                    meas = torch.cat([xy, v], dim=0)  # (4,)
                else:
                    # Only position measurement; velocities left to the model
                    # meas still has length 4, but EKF will only use first 2
                    meas = torch.cat([xy, torch.zeros_like(xy)], dim=0)

                state = tracker.step(meas)  # (4,) -> [x,y,vx,vy]
                xy_filt = state[:2]         # filtered x,y

                # Keep any extra dims (e.g., z) from the original sequence
                if D > 2:
                    extra = pos_seq[b, t, 2:].to(device)
                    pos_filt = torch.cat([xy_filt, extra], dim=0)
                else:
                    pos_filt = xy_filt

                seq_out.append(pos_filt.unsqueeze(0))
                prev_xy = xy

            seq_out_t = torch.cat(seq_out, dim=0)   # (T, D)
            out_list.append(seq_out_t.unsqueeze(0)) # (1,T,D)

        out = torch.cat(out_list, dim=0)            # (B,T,D)
        return out
