"""
Simplified educational LeWorldModel-style JEPA.

Inspired by:
Maes, L., Le Lidec, Q., Scieur, D., LeCun, Y., & Balestriero, R. (2026).
LeWorldModel: Stable End-to-End Joint-Embedding Predictive Architecture from Pixels.
arXiv:2603.19312. https://arxiv.org/abs/2603.19312
Official code: https://github.com/lucas-maes/le-wm

The paper trains end-to-end with two loss terms (next-embedding prediction + SIGReg on the
embeddings) and no stop-gradient or EMA target. This educational model follows that recipe
by default (``detach_target=False``). It is far smaller than the official ~15M-parameter model.
"""

from __future__ import annotations

from typing import Any, Dict, Sequence

import torch
import torch.nn as nn
import torch.nn.functional as F

from .sigreg import SIGReg


class SimpleEncoder(nn.Module):
    """Small CNN: (B,3,H,W) -> (B, latent_dim). Keeps spatial layout (no global pooling) so
    object *position* stays encodable — global average pooling would throw it away."""

    def __init__(self, latent_dim: int = 192, channels: Sequence[int] = (32, 64, 128),
                 image_size: int = 64):
        super().__init__()
        down = 2 ** len(channels)
        if image_size % down != 0 or image_size < down:
            raise ValueError(f"image_size={image_size} must be a multiple of {down}")
        layers, c_in = [], 3
        for c in channels:
            layers += [nn.Conv2d(c_in, c, 4, 2, 1), nn.ReLU(inplace=True)]
            c_in = c
        side = image_size // down
        self.conv = nn.Sequential(*layers)
        self.head = nn.Sequential(nn.Flatten(), nn.Linear(c_in * side * side, latent_dim))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.head(self.conv(x))


class ARPredictor(nn.Module):
    """Action-conditioned predictor: (z_t, a_t) -> ẑ_{t+1} (residual: z_t + f(z_t, a_t))."""

    def __init__(self, latent_dim: int = 192, action_dim: int = 2, hidden: int = 256):
        super().__init__()
        self.action_embed = nn.Linear(action_dim, latent_dim)
        self.net = nn.Sequential(
            nn.Linear(latent_dim * 2, hidden), nn.ReLU(inplace=True), nn.Linear(hidden, latent_dim)
        )

    def forward(self, z: torch.Tensor, a: torch.Tensor) -> torch.Tensor:
        return z + self.net(torch.cat([z, self.action_embed(a)], dim=-1))


class LeWMStyle(nn.Module):
    """
    L = MSE(predictor(z_t, a_t), z_{t+1}) + λ · SIGReg([z_t; z_{t+1}])

    ``detach_target=True`` adds a stop-gradient on z_{t+1} (a common JEPA trick the paper avoids).
    """

    def __init__(self, latent_dim: int = 192, action_dim: int = 2, lambda_reg: float = 0.1,
                 n_projections: int = 256, image_size: int = 64,
                 channels: Sequence[int] = (32, 64, 128), hidden: int = 256,
                 detach_target: bool = False):
        super().__init__()
        self.encoder = SimpleEncoder(latent_dim, channels, image_size)
        self.predictor = ARPredictor(latent_dim, action_dim, hidden)
        self.sigreg = SIGReg(n_projections=n_projections)
        self.lambda_reg = lambda_reg
        self.latent_dim = latent_dim
        self.action_dim = action_dim
        self.image_size = image_size
        self.detach_target = detach_target
        self._cfg: Dict[str, Any] = dict(
            latent_dim=latent_dim, action_dim=action_dim, lambda_reg=lambda_reg,
            n_projections=n_projections, image_size=image_size, channels=list(channels),
            hidden=hidden, detach_target=detach_target)

    # ------------------------------------------------------------- config
    def config_dict(self) -> Dict[str, Any]:
        return dict(self._cfg)

    @classmethod
    def from_config(cls, cfg: Dict[str, Any]) -> "LeWMStyle":
        m = cfg["model"]
        return cls(latent_dim=m["latent_dim"], action_dim=m["action_dim"],
                   lambda_reg=m["lambda_reg"], n_projections=m.get("n_projections", 256),
                   image_size=cfg["data"]["image_size"],
                   channels=m["encoder"]["channels"], hidden=m["predictor"]["hidden"],
                   detach_target=m.get("detach_target", False))

    def num_parameters(self) -> int:
        return sum(p.numel() for p in self.parameters())

    # ------------------------------------------------------------ forward
    def encode(self, obs: torch.Tensor) -> torch.Tensor:
        return self.encoder(obs)

    def predict(self, z: torch.Tensor, action: torch.Tensor) -> torch.Tensor:
        return self.predictor(z, action)

    def forward(self, obs_t: torch.Tensor, action: torch.Tensor,
                obs_tp1: torch.Tensor | None = None):
        z_t = self.encoder(obs_t)
        z_pred = self.predictor(z_t, action)
        if obs_tp1 is None:
            return z_pred
        z_tp1 = self.encoder(obs_tp1)
        target = z_tp1.detach() if self.detach_target else z_tp1
        pred_loss = F.mse_loss(z_pred, target)
        reg_loss = self.sigreg(torch.cat([z_t, z_tp1], dim=0))
        total = pred_loss + self.lambda_reg * reg_loss
        return total, z_pred, z_tp1, pred_loss, reg_loss
