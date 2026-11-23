"""
Simple latent-space Cross-Entropy Method (CEM) planner.
"""

from __future__ import annotations

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
        if horizon < 1:
            raise ValueError("horizon must be >= 1")
        if n_candidates < 2:
            raise ValueError("n_candidates must be >= 2")
        if not 1 <= n_elite < n_candidates:
            raise ValueError("n_elite must be in [1, n_candidates)")
        if n_iters < 1:
            raise ValueError("n_iters must be >= 1")
        if not action_low < action_high:
            raise ValueError("action_low must be less than action_high")
        self.model = model
        self.horizon = horizon
        self.n_candidates = n_candidates
        self.n_elite = n_elite
        self.n_iters = n_iters
        self.action_low = action_low
        self.action_high = action_high
        self.action_dim = model.action_dim

    @torch.no_grad()
    def plan(self, z_start: torch.Tensor, z_goal: torch.Tensor) -> Tuple[torch.Tensor, float]:
        if z_start.ndim != 1 or z_goal.ndim != 1:
            raise ValueError("z_start and z_goal must be one-dimensional latent vectors")
        if z_start.shape != z_goal.shape:
            raise ValueError("z_start and z_goal must have identical shapes")
        if z_start.numel() != self.model.latent_dim:
            raise ValueError("latent vector size does not match model.latent_dim")
        if z_start.device != z_goal.device:
            raise ValueError("z_start and z_goal must be on the same device")
        if not torch.isfinite(z_start).all() or not torch.isfinite(z_goal).all():
            raise ValueError("latent vectors must contain only finite values")

        device = z_start.device
        mean = torch.zeros(self.horizon, self.action_dim, device=device)
        std = torch.full_like(mean, 0.5)
        best_actions = None
        best_cost = float("inf")

        for _ in range(self.n_iters):
            noise = torch.randn(
                self.n_candidates, self.horizon, self.action_dim, device=device
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
            # unbiased=True yields NaN for a single elite; population std is
            # the appropriate statistic for fitting this sampling distribution.
            std = elite_actions.std(dim=0, unbiased=False).clamp(min=0.05)

            if elite_costs[0].item() < best_cost:
                best_cost = float(elite_costs[0].item())
                best_actions = elite_actions[0].clone()

        if best_actions is None:
            raise RuntimeError("CEM failed to produce a candidate action sequence")
        return best_actions, best_cost
