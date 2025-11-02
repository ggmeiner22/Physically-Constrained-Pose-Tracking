from __future__ import annotations
from typing import Optional, Dict, Tuple
import torch
from .config import PhysicsParams, LossWeights
from .physics import physics_penalty




def total_loss(
    y_pred_t: torch.Tensor, # (B,4) -> [x,y,vx,vy]
    y_pred_t_prev: Optional[torch.Tensor],
    scenario: str,
    params: PhysicsParams,
    w: LossWeights,
    y_gt_t: Optional[torch.Tensor] = None,
) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
    xt = y_pred_t[:, :2]
    vt = y_pred_t[:, 2:]
    xt_prev = y_pred_t_prev[:, :2] if y_pred_t_prev is not None else None
    vt_prev = y_pred_t_prev[:, 2:] if y_pred_t_prev is not None else None


    L_vision = torch.tensor(0.0, device=y_pred_t.device)
    if y_gt_t is not None:
        L_vision = ((y_pred_t - y_gt_t) ** 2).mean()
    
    
    L_phys_vec = physics_penalty(xt, vt, xt_prev, vt_prev, scenario, params)
    L_phys = L_phys_vec.mean()
    
    
    L_smooth = torch.tensor(0.0, device=y_pred_t.device)
    if y_pred_t_prev is not None:
        L_smooth = ((y_pred_t - y_pred_t_prev) ** 2).mean()
    
    
    L = L_vision + w.lambda_phys * L_phys + w.lambda_smooth * L_smooth
    logs = {
    "L": L.detach(),
    "L_vision": L_vision.detach(),
    "L_phys": L_phys.detach(),
    "L_smooth": L_smooth.detach(),
    }
    return L, logs