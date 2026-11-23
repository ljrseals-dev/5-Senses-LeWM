"""
Simple latent-space Cross-Entropy Method (CEM) planner.
"""

from __future__ import annotations

from typing import Optional, Tuple
import torch
from .jepa import LeWMStyle


class LatentCEMPlanner:
    """
    Plan action sequences entirely in latent space.

    1. Encode start and goal observations.
    2. Sample candidate action sequences.
    3. Roll them out with the predictor.
    4. Score by distance of final embedding to goal embedding.
    5. Keep elite samples and refine the distribution.
    """

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
        self.model = model
        self.horizon = horizon
        self.n_candidates = n_candidates
        self.n_elite = n_elite
        self.n_iters = n_iters
        self.action_low = action_low
        self.action_high = action_high
        self.action_dim = model.action_dim

    @torch.no_grad()
    def plan(
        self,
        z_start: torch.Tensor,
        z_goal: torch.Tensor,
    ) -> Tuple[torch.Tensor, float]:
        """
        Returns:
            best_actions : (horizon, action_dim)
            best_cost    : scalar distance to goal
        """
        device = z_start.device
        # mean / std of the sampling distribution
        mean = torch.zeros(self.horizon, self.action_dim, device=device)
        std = torch.ones(self.horizon, self.action_dim, device=device) * 0.5

        best_actions = None
        best_cost = float("inf")

        for _ in range(self.n_iters):
            # sample
            noise = torch.randn(self.n_candidates, self.horizon, self.action_dim, device=device)
            actions = mean.unsqueeze(0) + std.unsqueeze(0) * noise
            actions = actions.clamp(self.action_low, self.action_high)

            # rollout
            z = z_start.expand(self.n_candidates, -1)
            for t in range(self.horizon):
                z = self.model.predict(z, actions[:, t])

            # cost = L2 distance to goal
            costs = (z - z_goal).pow(2).sum(dim=-1)  # (N,)

            # elite
            elite_idx = costs.argsort()[: self.n_elite]
            elite_actions = actions[elite_idx]
            elite_costs = costs[elite_idx]

            # update distribution
            mean = elite_actions.mean(dim=0)
            std = elite_actions.std(dim=0).clamp(min=0.05)

            if elite_costs[0] < best_cost:
                best_cost = float(elite_costs[0].item())
                best_actions = elite_actions[0].clone()

        assert best_actions is not None
        return best_actions, best_cost