"""Latent-space Cross-Entropy Method (CEM) planner with an auditable result."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

import torch


@dataclass
class PlanResult:
    actions: torch.Tensor                  # (horizon, action_dim)
    cost: float                            # squared L2 distance of predicted final latent to goal
    initial_cost: float                    # squared distance start -> goal (no action taken)
    history: List[Dict[str, float]] = field(default_factory=list)
    params: Dict[str, Any] = field(default_factory=dict)

    def __iter__(self):                    # backward compatible: actions, cost = planner.plan(...)
        yield self.actions
        yield self.cost

    def to_dict(self) -> Dict[str, Any]:
        return {"actions": self.actions.detach().cpu().tolist(), "cost": self.cost,
                "initial_cost": self.initial_cost, "history": self.history, "params": self.params}


class LatentCEMPlanner:
    """
    1. Encode start and goal (done by the caller). 2. Sample action sequences.
    3. Roll out with the predictor. 4. Score by distance of the final latent to the goal.
    5. Refit the sampling distribution to the elites. Everything is recorded in ``PlanResult``.

    ``model`` needs ``predict(z, a)`` and ``action_dim`` (any backend works).
    """

    def __init__(self, model, horizon: int = 8, n_candidates: int = 64, n_elite: int = 8,
                 n_iters: int = 4, action_low: float = -1.0, action_high: float = 1.0,
                 seed: Optional[int] = None):
        if n_elite > n_candidates:
            raise ValueError("n_elite must be <= n_candidates")
        self.model = model
        self.horizon, self.n_candidates, self.n_elite, self.n_iters = horizon, n_candidates, n_elite, n_iters
        self.action_low, self.action_high = action_low, action_high
        self.action_dim = model.action_dim
        self.seed = seed

    @torch.no_grad()
    def rollout(self, z_start: torch.Tensor, actions: torch.Tensor) -> torch.Tensor:
        """z_start (D,) or (1,D); actions (N,H,A) or (H,A) -> final latent (N,D) / (1,D)."""
        if actions.dim() == 2:
            actions = actions.unsqueeze(0)
        z = z_start.reshape(1, -1).expand(actions.shape[0], -1)
        for t in range(actions.shape[1]):
            z = self.model.predict(z, actions[:, t])
        return z

    @torch.no_grad()
    def plan(self, z_start: torch.Tensor, z_goal: torch.Tensor) -> PlanResult:
        device = z_start.device
        z_start = z_start.reshape(1, -1)
        z_goal = z_goal.reshape(1, -1).to(device)
        gen = None
        if self.seed is not None:
            gen = torch.Generator(device=device)
            gen.manual_seed(self.seed)

        mean = torch.zeros(self.horizon, self.action_dim, device=device)
        std = torch.full((self.horizon, self.action_dim), 0.5, device=device)
        best_actions, best_cost = None, float("inf")
        history: List[Dict[str, float]] = []

        for it in range(self.n_iters):
            noise = torch.randn(self.n_candidates, self.horizon, self.action_dim,
                                device=device, generator=gen)
            actions = (mean.unsqueeze(0) + std.unsqueeze(0) * noise).clamp(self.action_low, self.action_high)
            z_final = self.rollout(z_start, actions)
            costs = (z_final - z_goal).pow(2).sum(dim=-1)                  # (N,)

            order = costs.argsort()
            elite = actions[order[: self.n_elite]]
            mean = elite.mean(dim=0)
            std = elite.std(dim=0).clamp(min=0.05) if self.n_elite > 1 else std
            if costs[order[0]].item() < best_cost:
                best_cost = float(costs[order[0]].item())
                best_actions = elite[0].clone()
            history.append({"iter": it, "best_cost": float(costs[order[0]].item()),
                            "elite_mean_cost": float(costs[order[: self.n_elite]].mean().item()),
                            "all_mean_cost": float(costs.mean().item()),
                            "action_std_mean": float(std.mean().item())})

        assert best_actions is not None
        return PlanResult(
            actions=best_actions, cost=best_cost,
            initial_cost=float((z_start - z_goal).pow(2).sum().item()),
            history=history,
            params={"horizon": self.horizon, "n_candidates": self.n_candidates,
                    "n_elite": self.n_elite, "n_iters": self.n_iters, "seed": self.seed,
                    "action_low": self.action_low, "action_high": self.action_high})
