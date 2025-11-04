from __future__ import annotations
from typing import Dict, Optional
import torch
import torch.optim as optim
from .pipeline import PosePipeline
from .data import FrameBatch
from .losses import total_loss


def _train_eval_sequence(model: PosePipeline,
                         batch: FrameBatch,
                         y_gt: Optional[torch.Tensor],
                         is_train: bool,
                         opt: Optional[optim.Optimizer]) -> Dict[str, float]:
    """
    Runs one pass over the whole sequence at once if the backbone supports sequences.
    Otherwise falls back to per-frame.
    """
    mode = "train" if is_train else "eval"
    if is_train:
        model.train()
    else:
        model.eval()

    logs_accum = {"L": 0.0, "L_vision": 0.0, "L_phys": 0.0, "L_smooth": 0.0}

    frames = batch.frames  # (T, C, H, W)
    T = frames.shape[0]

    # If backbone expects a sequence, do one forward for all T
    if getattr(model.backbone, "expects_sequence", False):
        x_in = frames.unsqueeze(0)               # (1, T, C, H, W)
        y_seq = model.backbone(x_in).squeeze(0)  # (T, 4)
        y_prev = None
        for t in range(T):
            y_t = y_seq[t:t+1]
            gt_t = y_gt[t:t+1] if y_gt is not None else None
            L, logs = total_loss(y_t, y_prev, model.scenario, model.params, model.weights, gt_t)
            if is_train:
                opt.zero_grad(set_to_none=True)
                L.backward()
                opt.step()
            y_prev = y_t.detach()
            for k, v in logs.items():
                logs_accum[k] += float(v.item())
    else:
        # Per-frame pass (original behavior)
        y_prev = None
        for t in range(T):
            x_t = frames[t:t+1]                  # (1, C, H, W)
            y_t = model(x_t)                     # (1, 4)
            gt_t = y_gt[t:t+1] if y_gt is not None else None
            L, logs = total_loss(y_t, y_prev, model.scenario, model.params, model.weights, gt_t)
            if is_train:
                opt.zero_grad(set_to_none=True)
                L.backward()
                opt.step()
            y_prev = y_t.detach()
            for k, v in logs.items():
                logs_accum[k] += float(v.item())

    n = max(T, 1)
    return {k: v / n for k, v in logs_accum.items()}


def train_one_epoch(model: PosePipeline, batch: FrameBatch,
                    y_gt: Optional[torch.Tensor], opt: optim.Optimizer) -> Dict[str, float]:
    return _train_eval_sequence(model, batch, y_gt, is_train=True, opt=opt)


@torch.no_grad()
def evaluate(model: PosePipeline, batch: FrameBatch,
             y_gt: Optional[torch.Tensor]) -> Dict[str, float]:
    return _train_eval_sequence(model, batch, y_gt, is_train=False, opt=None)
