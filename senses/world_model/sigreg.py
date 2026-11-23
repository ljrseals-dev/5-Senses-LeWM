"""
Simplified educational implementation of SIGReg (Sketched Isotropic Gaussian Regularizer)
inspired by:

Maes, L., Le Lidec, Q., Scieur, D., LeCun, Y., & Balestriero, R. (2026).
LeWorldModel: Stable End-to-End Joint-Embedding Predictive Architecture from Pixels.
arXiv:2603.19312. https://arxiv.org/abs/2603.19312

Official code: https://github.com/lucas-maes/le-wm

The original paper uses a differentiable Epps–Pulley-style statistic on random
projections.  This module provides both a lightweight mean/variance version and
a more faithful Epps–Pulley implementation for educational purposes.
"""

from __future__ import annotations

import torch
import torch.nn as nn


class EppsPulley(nn.Module):
    """
    Differentiable Epps–Pulley normality statistic
    (the univariate test inside SIGReg).

    Based on the classical test of Epps & Pulley (1983) that compares the
    empirical characteristic function to that of a standard normal.
    """

    def __init__(self, t_max: float = 4.0, n_points: int = 17):
        super().__init__()
        # Positive frequencies only (even integrand → double later)
        self.register_buffer("t", torch.linspace(0.0, t_max, n_points))
        # Gaussian weight φ(t) = exp(-t²/2)
        self.register_buffer("weight", torch.exp(-0.5 * self.t**2))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        x : (N,) or (N, M)   — 1-D projections
        Returns scalar (mean over projections) or (M,) tensor.
        """
        if x.dim() == 1:
            x = x.unsqueeze(1)  # (N, 1)

        N, M = x.shape
        t = self.t.view(1, 1, -1)  # (1, 1, K)
        x_t = x.unsqueeze(2) * t  # (N, M, K)

        # Empirical characteristic function
        cos_term = torch.cos(x_t).mean(dim=0)  # (M, K)
        sin_term = torch.sin(x_t).mean(dim=0)  # (M, K)

        # Target CF of N(0,1)
        phi = torch.exp(-0.5 * self.t**2)  # (K,)

        # |ECF - φ|²
        diff_sq = (cos_term - phi) ** 2 + sin_term**2

        # Weighted integral (trapezoidal) + factor 2 for symmetry + *N
        integral = torch.trapz(diff_sq * self.weight, self.t, dim=1)
        T = N * 2.0 * integral  # (M,)

        return T.mean() if M > 1 else T.squeeze()


class SIGReg(nn.Module):
    """
    Sketched Isotropic Gaussian Regularizer.

    Projects the batch of embeddings onto many random unit directions and
    encourages each 1-D marginal to be close to a standard normal
    (via Epps–Pulley or a simple mean/variance penalty).
    """

    def __init__(
        self,
        n_projections: int = 256,
        use_epps: bool = True,
        **epps_kwargs,
    ):
        super().__init__()
        self.n_projections = n_projections
        self.use_epps = use_epps
        if use_epps:
            self.epps = EppsPulley(**epps_kwargs)

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        """
        z : (N, D)  batch of embeddings
        """
        N, D = z.shape
        # Random unit directions
        u = torch.randn(D, self.n_projections, device=z.device, dtype=z.dtype)
        u = u / (u.norm(dim=0, keepdim=True) + 1e-8)

        # Project
        projections = z @ u  # (N, M)

        if self.use_epps:
            return self.epps(projections)

        # Fallback: simple isotropic Gaussian penalty (mean ≈ 0, var ≈ 1)
        mean = projections.mean(dim=0)
        var = projections.var(dim=0, unbiased=False)
        loss = mean.pow(2).mean() + (var - 1.0).pow(2).mean()
        return loss