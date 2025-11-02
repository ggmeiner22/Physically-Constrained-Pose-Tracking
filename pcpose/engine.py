from __future__ import annotations
from typing import Dict, Optional
import torch
import torch.optim as optim
from .pipeline import PosePipeline
from .data import FrameBatch
from .losses import total_loss




def train_one_epoch(model: PosePipeline, batch: FrameBatch, y_gt: Optional[torch.Tensor], opt: optim.Optimizer) -> Dict[str, float]:
    model.train()
    logs_accum = {"L": 0.0, "L_vision": 0.0, "L_phys": 0.0, "L_smooth": 0.0}
    y_prev = None
    for i in range(batch.frames.shape[0]):
        x_t = batch.frames[i:i+1]
        y_t = model(x_t)
        gt_t = y_gt[i:i+1] if y_gt is not None else None
        L, logs = total_loss(y_t, y_prev, model.scenario, model.params, model.weights, gt_t)
        opt.zero_grad(set_to_none=True)
        L.backward()
        opt.step()
        y_prev = y_t.detach()
        for k, v in logs.items():
            logs_accum[k] += float(v.item())
    n = batch.frames.shape[0]
    return {k: v / max(n, 1) for k, v in logs_accum.items()}




@torch.no_grad()
def evaluate(model: PosePipeline, batch: FrameBatch, y_gt: Optional[torch.Tensor]) -> Dict[str, float]:
    model.eval()
    logs_accum = {"L": 0.0, "L_vision": 0.0, "L_phys": 0.0, "L_smooth": 0.0}
    y_prev = None
    for i in range(batch.frames.shape[0]):
        x_t = batch.frames[i:i+1]
        y_t = model(x_t)
        gt_t = y_gt[i:i+1] if y_gt is not None else None
        L, logs = total_loss(y_t, y_prev, model.scenario, model.params, model.weights, gt_t)
        y_prev = y_t
        for k, v in logs.items():
            logs_accum[k] += float(v.item())
    n = batch.frames.shape[0]
    return {k: v / max(n, 1) for k, v in logs_accum.items()}