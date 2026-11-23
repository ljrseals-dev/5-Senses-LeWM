"""Minimal local training loop for the educational LeWM-style model."""

from __future__ import annotations

from typing import Callable, Dict, List, Optional

import torch

from .data import PairDataset


def train_model(model, dataset: PairDataset, steps: int = 2000, batch_size: int = 32,
                lr: float = 3e-4, device: str = "cpu", log_every: int = 50, seed: int = 0,
                log_fn: Optional[Callable[[str], None]] = print) -> List[Dict[str, float]]:
    dev = torch.device(device)
    model.to(dev).train()
    gen = torch.Generator().manual_seed(seed)
    opt = torch.optim.AdamW(model.parameters(), lr=lr)
    history: List[Dict[str, float]] = []
    for step in range(1, steps + 1):
        obs_t, action, obs_tp1 = dataset.sample(batch_size, dev, gen)
        opt.zero_grad(set_to_none=True)
        total, _, _, pred, reg = model(obs_t, action, obs_tp1)
        total.backward()
        opt.step()
        if step == 1 or step % log_every == 0 or step == steps:
            rec = {"step": step, "total": total.item(), "pred": pred.item(), "reg": reg.item()}
            history.append(rec)
            if log_fn:
                log_fn(f"step {step:5d} | total={rec['total']:.4f} pred={rec['pred']:.4f} reg={rec['reg']:.4f}")
    model.eval()
    return history
