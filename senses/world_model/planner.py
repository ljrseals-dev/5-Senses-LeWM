"""
Simple latent-space Cross-Entropy Method (CEM) planner.
"""

from __future__ import annotations

import math
from typing import Tuple

import torch

from .jepa import LeWMStyle


class LatentCEMPlanner:
    """Plan bounded action sequences by rolling the learned dynamics model."""

    def __init__(
        self,
        model: LeWMStyle,
        horizon: int = 8,
        n_candidates: int = 64,
        n_elite: int = 8,
        n_iters: int = 4,
        action_low: float = -1.0,
        action_high: float = 1.0,
    ):
        if not isinstance(horizon, int) or horizon < 1:
            raise ValueError("horizon must be an integer >= 1")
        if not isinstance(n_candidates, int) or n_candidates < 2:
            raise ValueError("n_candidates must be an integer >= 2")
        if not isinstance(n_elite, int) or not 1 <= n_elite < n_candidates:
            raise ValueError("n_elite must be in [1, n_candidates)")
        if not isinstance(n_iters, int) or n_iters < 1:
            raise ValueError("n_iters must be an integer >= 1")
        if (
            not math.isfinite(action_low)
            or not math.isfinite(action_high)
            or not action_low < action_high
        ):
            raise ValueError("action bounds must be finite and action_low < action_high")
        self.model = model
        self.horizon = horizon
        self.n_candidates = n_candidates
        self.n_elite = n_elite
        self.n_iters = n_iters
        self.action_low = float(action_low)
        self.action_high = float(action_high)
        self.action_dim = model.action_dim

    @torch.no_grad()
    def plan(self, z_start: torch.Tensor, z_goal: torch.Tensor) -> Tuple[torch.Tensor, float]:
        if not isinstance(z_start, torch.Tensor) or not isinstance(z_goal, torch.Tensor):
            raise TypeError("z_start and z_goal must be torch.Tensor instances")
        if z_start.ndim != 1 or z_goal.ndim != 1:
            raise ValueError("z_start and z_goal must be one-dimensional latent vectors")
        if z_start.shape != z_goal.shape:
            raise ValueError("z_start and z_goal must have identical shapes")
        if z_start.numel() != self.model.latent_dim:
            raise ValueError("latent vector size does not match model.latent_dim")
        if z_start.device != z_goal.device:
            raise ValueError("z_start and z_goal must be on the same device")
        parameter = next(self.model.parameters())
        if z_start.device != parameter.device or z_start.dtype != parameter.dtype:
            raise ValueError("latent vectors must match the model parameter device and dtype")
        if not torch.is_floating_point(z_start):
            raise TypeError("latent vectors must be floating-point tensors")
        if not torch.isfinite(z_start).all() or not torch.isfinite(z_goal).all():
            raise ValueError("latent vectors must contain only finite values")

        mean = torch.zeros(
            self.horizon, self.action_dim, device=z_start.device, dtype=z_start.dtype
        )
        std = torch.full_like(mean, 0.5)
        best_actions = None
        best_cost = float("inf")

        for _ in range(self.n_iters):
            noise = torch.randn(
                self.n_candidates,
                self.horizon,
                self.action_dim,
                device=z_start.device,
                dtype=z_start.dtype,
            )
            actions = (mean.unsqueeze(0) + std.unsqueeze(0) * noise).clamp(
                self.action_low, self.action_high
            )
            z = z_start.unsqueeze(0).expand(self.n_candidates, -1)
            for step in range(self.horizon):
                z = self.model.predict(z, actions[:, step])

            costs = (z - z_goal.unsqueeze(0)).pow(2).sum(dim=-1)
            elite_idx = costs.argsort()[: self.n_elite]
            elite_actions = actions[elite_idx]
            elite_costs = costs[elite_idx]
            mean = elite_actions.mean(dim=0)
            std = elite_actions.std(dim=0, unbiased=False).clamp(min=0.05)

            if elite_costs[0].item() < best_cost:
                best_cost = float(elite_costs[0].item())
                best_actions = elite_actions[0].clone()

        if best_actions is None:
            raise RuntimeError("CEM failed to produce a candidate action sequence")
        return best_actions, best_cost
