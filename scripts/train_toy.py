#!/usr/bin/env python3
"""
Toy training loop for the educational LeWM-style model.
Generates synthetic random trajectories so the script runs without external data.
"""

from __future__ import annotations

import torch
import torch.optim as optim
from senses.world_model.jepa import LeWMStyle


def make_batch(batch_size: int = 16, size: int = 64, action_dim: int = 2):
    """Synthetic (obs_t, action, obs_tp1) batch."""
    obs_t = torch.randn(batch_size, 3, size, size)
    action = torch.randn(batch_size, action_dim) * 0.5
    # crude “physics”: next frame = current + small noise modulated by action
    noise = 0.1 * torch.randn_like(obs_t)
    obs_tp1 = obs_t + noise + action.view(batch_size, action_dim, 1, 1)[:, :1] * 0.05
    return obs_t, action, obs_tp1


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    model = LeWMStyle(latent_dim=64, action_dim=2, lambda_reg=0.1, n_projections=64)
    model.to(device)
    opt = optim.Adam(model.parameters(), lr=3e-4)

    steps = 200
    log_every = 20

    print("Starting toy training …")
    for step in range(1, steps + 1):
        obs_t, action, obs_tp1 = make_batch()
        obs_t, action, obs_tp1 = obs_t.to(device), action.to(device), obs_tp1.to(device)

        opt.zero_grad()
        total, _, _, pred_loss, reg_loss = model(obs_t, action, obs_tp1)
        total.backward()
        opt.step()

        if step % log_every == 0 or step == 1:
            print(
                f"step {step:4d} | total={total.item():.4f} "
                f"pred={pred_loss.item():.4f} reg={reg_loss.item():.4f}"
            )

    print("Toy training finished.")
    # quick smoke test of planner
    from senses.world_model.planner import LatentCEMPlanner

    model.eval()
    with torch.no_grad():
        z0 = model.encode(torch.randn(1, 3, 64, 64, device=device))
        zg = model.encode(torch.randn(1, 3, 64, 64, device=device))
        planner = LatentCEMPlanner(model, horizon=4, n_candidates=32, n_elite=4, n_iters=2)
        actions, cost = planner.plan(z0.squeeze(0), zg.squeeze(0))
        print(f"Planner smoke test → actions shape {tuple(actions.shape)}, cost={cost:.4f}")


if __name__ == "__main__":
    main()