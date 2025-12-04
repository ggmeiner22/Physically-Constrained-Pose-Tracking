from __future__ import annotations
from typing import Optional
import torch
from .config import PhysicsParams

# --- Helpers for pendulum dynamics ---
def _pendulum_angle(r: torch.Tensor) -> torch.Tensor:
    return torch.atan2(r[:, 1], r[:, 0])  # (B,)

def _angular_velocity(r: torch.Tensor, v: torch.Tensor, L: float) -> torch.Tensor:
    rxv_z = r[:, 0] * v[:, 1] - r[:, 1] * v[:, 0]
    return rxv_z / (L * L + 1e-6)

def physics_penalty(
    xt: torch.Tensor,        # (B,2)
    vt: torch.Tensor,        # (B,2)
    xt_prev: Optional[torch.Tensor],
    vt_prev: Optional[torch.Tensor],
    scenario: str,
    params: PhysicsParams,
) -> torch.Tensor:
    penalties = []

    # Temporal dynamics baseline
    if xt_prev is not None and vt_prev is not None:
        gvec = torch.tensor([0.0, params.gravity], device=xt.device)
        v_expected = vt_prev + gvec * params.dt
        penalties.append(((vt - v_expected) ** 2).sum(dim=1))
        x_expected = xt_prev + vt_prev * params.dt
        penalties.append(((xt - x_expected) ** 2).sum(dim=1))

    if scenario == "pendulum":
        anchor = torch.tensor(params.anchor, device=xt.device).unsqueeze(0)
        r = xt - anchor
        L = params.cable_length
        length = torch.linalg.norm(r, dim=1)
        penalties.append((length - L).abs())
        penalties.append(torch.clamp(params.anchor[1] - xt[:, 1], min=0.0))

        # --- Pendulum angle dynamics ---
        c = getattr(params, "pendulum_damping", 0.05)
        theta = _pendulum_angle(r)
        omega = _angular_velocity(r, vt, L)
        alpha_expected = -(params.gravity / (L + 1e-6)) * torch.sin(theta) - c * omega
        if xt_prev is not None and vt_prev is not None:
            r_prev = xt_prev - anchor
            omega_prev = _angular_velocity(r_prev, vt_prev, L)
            alpha_obs = (omega - omega_prev) / max(params.dt, 1e-6)
            penalties.append((alpha_obs - alpha_expected) ** 2)

        # Inextensible string: (r · v) ≈ 0
        radial_v = (r * vt).sum(dim=1) / (L + 1e-6)
        penalties.append(radial_v ** 2)

    elif scenario == "marble":
        penalties.append((xt[:, 1] - params.slide_y).abs())
        # --- Friction cone on flat surface ---
        if vt_prev is not None:
            a = (vt - vt_prev) / max(params.dt, 1e-6)
            ax = a[:, 0].abs()
            mu = torch.as_tensor(getattr(params, "mu_slide", 0.5), device=xt.device)
            thresh = mu * params.gravity
            penalties.append(torch.clamp(ax - thresh, min=0.0) ** 2)

    elif scenario == "dropping":
        penalties.append(torch.clamp(-vt[:, 1], min=0.0))

    else:
        raise ValueError(f"Unknown scenario: {scenario}")

    return sum(penalties) if penalties else torch.zeros(xt.shape[0], device=xt.device)
