from __future__ import annotations
from typing import Optional
import torch
from .config import PhysicsParams


def physics_penalty(
    xt: torch.Tensor, # (B,2)
    vt: torch.Tensor, # (B,2)
    xt_prev: Optional[torch.Tensor],
    vt_prev: Optional[torch.Tensor],
    scenario: str,
    params: PhysicsParams,
) -> torch.Tensor:
    penalties = []


    # Temporal dynamics: constant-accel with gravity in +y (screen coords)
    if xt_prev is not None and vt_prev is not None:
        g = torch.tensor([0.0, params.gravity], device=xt.device)
        v_expected = vt_prev + g * params.dt
        penalties.append(((vt - v_expected) ** 2).sum(dim=1))
        x_expected = xt_prev + vt_prev * params.dt
        penalties.append(((xt - x_expected) ** 2).sum(dim=1))
    
    
    if scenario == "hanging":
        anchor = torch.tensor(params.anchor, device=xt.device).unsqueeze(0)
        length = torch.linalg.norm(xt - anchor, dim=1)
        penalties.append((length - params.cable_length).abs())
        penalties.append(torch.clamp(params.anchor[1] - xt[:, 1], min=0.0))
    
    
    elif scenario == "sliding":
        penalties.append((xt[:, 1] - params.slide_y).abs())
    
    
    elif scenario == "dropping":
        penalties.append(torch.clamp(-(vt[:, 1]), min=0.0))
    
    
    else:
        raise ValueError(f"Unknown scenario: {scenario}")
    
    
    return sum(penalties) if penalties else torch.zeros(xt.shape[0], device=xt.device)