"""
Educational LeWorldModel-style joint-embedding predictive architecture.

This is a compact independent teaching implementation, not the official
LeWorldModel training stack or a numerically faithful reproduction.
"""

from __future__ import annotations

import math

import torch
import torch.nn as nn

from .sigreg import SIGReg


class SimpleEncoder(nn.Module):
    """Tiny CNN encoder mapping (B, 3, H, W) to (B, latent_dim)."""

    def __init__(self, latent_dim: int = 192):
        super().__init__()
        if not isinstance(latent_dim, int) or latent_dim < 1:
            raise ValueError("latent_dim must be a positive integer")
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
        if x.ndim != 4 or x.shape[1] != 3:
            raise ValueError("observations must have shape (B, 3, H, W)")
        if x.shape[0] < 1 or x.shape[2] < 8 or x.shape[3] < 8:
            raise ValueError("observation batch must be non-empty and H,W >= 8")
        if not torch.is_floating_point(x):
            raise TypeError("observations must be floating-point tensors")
        if not torch.isfinite(x).all():
            raise ValueError("observations must contain only finite values")
        return self.net(x)


class ARPredictor(nn.Module):
    """Action-conditioned predictor: (z_t, a_t) -> predicted z_(t+1)."""

    def __init__(self, latent_dim: int = 192, action_dim: int = 2, hidden: int = 256):
        super().__init__()
        if min(latent_dim, action_dim, hidden) < 1:
            raise ValueError("latent_dim, action_dim, and hidden must be positive")
        self.action_embed = nn.Linear(action_dim, latent_dim)
        self.net = nn.Sequential(
            nn.Linear(latent_dim * 2, hidden),
            nn.ReLU(inplace=True),
            nn.Linear(hidden, latent_dim),
        )

    def forward(self, z: torch.Tensor, a: torch.Tensor) -> torch.Tensor:
        if z.ndim != 2:
            raise ValueError("z must have shape (B, latent_dim)")
        if a.ndim != 2 or a.shape != (z.shape[0], self.action_embed.in_features):
            raise ValueError("actions must have shape (B, action_dim)")
        if not torch.is_floating_point(z) or not torch.is_floating_point(a):
            raise TypeError("latents and actions must be floating-point tensors")
        if z.device != a.device:
            raise ValueError("latents and actions must be on the same device")
        if not torch.isfinite(z).all() or not torch.isfinite(a).all():
            raise ValueError("latents and actions must contain only finite values")
        action_embedding = self.action_embed(a)
        return self.net(torch.cat([z, action_embedding], dim=-1))


class LeWMStyle(nn.Module):
    """
    Educational JEPA-style model.

    Objective:
        MSE(predictor(encoder(o_t), a_t), stop_gradient(encoder(o_(t+1))))
        + lambda_reg * SIGReg(encoder(o_t))
    """

    def __init__(
        self,
        latent_dim: int = 192,
        action_dim: int = 2,
        lambda_reg: float = 0.1,
        n_projections: int = 256,
    ):
        super().__init__()
        if not isinstance(latent_dim, int) or latent_dim < 1:
            raise ValueError("latent_dim must be a positive integer")
        if not isinstance(action_dim, int) or action_dim < 1:
            raise ValueError("action_dim must be a positive integer")
        if not math.isfinite(lambda_reg) or lambda_reg < 0:
            raise ValueError("lambda_reg must be finite and non-negative")
        self.encoder = SimpleEncoder(latent_dim)
        self.predictor = ARPredictor(latent_dim, action_dim)
        self.sigreg = SIGReg(n_projections=n_projections)
        self.lambda_reg = float(lambda_reg)
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
        if obs_tp1.shape != obs_t.shape:
            raise ValueError("obs_t and obs_tp1 must have identical shapes")
        z_tp1 = self.encoder(obs_tp1)
        pred_loss = nn.functional.mse_loss(z_pred, z_tp1.detach())
        reg_loss = self.sigreg(z_t)
        total = pred_loss + self.lambda_reg * reg_loss
        return total, z_pred, z_tp1, pred_loss, reg_loss
