"""
Simplified educational LeWorldModel-style JEPA.

Inspired by:
Maes, L., Le Lidec, Q., Scieur, D., LeCun, Y., & Balestriero, R. (2026).
LeWorldModel: Stable End-to-End Joint-Embedding Predictive Architecture from Pixels.
arXiv:2603.19312. https://arxiv.org/abs/2603.19312

Official code: https://github.com/lucas-maes/le-wm

This is deliberately small and fully end-to-end so it can be trained or run
on a single GPU / CPU inside the offline-first 5-Senses lab.
"""

from __future__ import annotations

import torch
import torch.nn as nn
from .sigreg import SIGReg


class SimpleEncoder(nn.Module):
    """Tiny CNN encoder that maps (B,3,H,W) → (B, latent_dim)."""

    def __init__(self, latent_dim: int = 192):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(3, 32, 4, 2, 1),
            nn.ReLU(inplace=True),
            nn.Conv2d(32, 64, 4, 2, 1),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 128, 4, 2, 1),
            nn.ReLU(inplace=True),
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.Linear(128, latent_dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class ARPredictor(nn.Module):
    """Action-conditioned autoregressive predictor: (z_t, a_t) → ẑ_{t+1}."""

    def __init__(self, latent_dim: int = 192, action_dim: int = 2, hidden: int = 256):
        super().__init__()
        self.action_embed = nn.Linear(action_dim, latent_dim)
        self.net = nn.Sequential(
            nn.Linear(latent_dim * 2, hidden),
            nn.ReLU(inplace=True),
            nn.Linear(hidden, latent_dim),
        )

    def forward(self, z: torch.Tensor, a: torch.Tensor) -> torch.Tensor:
        a_emb = self.action_embed(a)
        return self.net(torch.cat([z, a_emb], dim=-1))


class LeWMStyle(nn.Module):
    """
    Simplified LeWorldModel-style JEPA.

    Objective:
        L = MSE(predictor(encoder(o_t), a_t), encoder(o_{t+1}).detach())
            + λ * SIGReg(encoder(o_t))
    """

    def __init__(
        self,
        latent_dim: int = 192,
        action_dim: int = 2,
        lambda_reg: float = 0.1,
        n_projections: int = 256,
    ):
        super().__init__()
        self.encoder = SimpleEncoder(latent_dim)
        self.predictor = ARPredictor(latent_dim, action_dim)
        self.sigreg = SIGReg(n_projections=n_projections)
        self.lambda_reg = lambda_reg
        self.latent_dim = latent_dim
        self.action_dim = action_dim

    def encode(self, obs: torch.Tensor) -> torch.Tensor:
        return self.encoder(obs)

    def predict(self, z: torch.Tensor, action: torch.Tensor) -> torch.Tensor:
        return self.predictor(z, action)

    def forward(
        self,
        obs_t: torch.Tensor,
        action: torch.Tensor,
        obs_tp1: torch.Tensor | None = None,
    ):
        z_t = self.encoder(obs_t)
        z_pred = self.predictor(z_t, action)

        if obs_tp1 is None:
            return z_pred

        z_tp1 = self.encoder(obs_tp1)
        pred_loss = nn.functional.mse_loss(z_pred, z_tp1.detach())
        reg_loss = self.sigreg(z_t)
        total = pred_loss + self.lambda_reg * reg_loss
        return total, z_pred, z_tp1, pred_loss, reg_loss