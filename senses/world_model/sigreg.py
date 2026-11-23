"""
Educational SIGReg implementation using random one-dimensional projections
and an Epps–Pulley-style characteristic-function statistic.
"""

from __future__ import annotations

import torch
import torch.nn as nn


class EppsPulley(nn.Module):
    """Differentiable univariate Epps–Pulley normality statistic."""

    def __init__(self, t_max: float = 4.0, n_points: int = 17):
        super().__init__()
        if not isinstance(n_points, int) or n_points < 2:
            raise ValueError("n_points must be an integer >= 2")
        if not t_max > 0:
            raise ValueError("t_max must be positive")
        self.register_buffer("t", torch.linspace(0.0, t_max, n_points))
        self.register_buffer("weight", torch.exp(-0.5 * self.t**2))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.ndim == 1:
            x = x.unsqueeze(1)
        if x.ndim != 2 or x.shape[0] < 2 or x.shape[1] < 1:
            raise ValueError("x must have shape (N, M), with N >= 2 and M >= 1")
        if not torch.is_floating_point(x):
            raise TypeError("x must be a floating-point tensor")

        n = x.shape[0]
        t = self.t.to(device=x.device, dtype=x.dtype).view(1, 1, -1)
        x_t = x.unsqueeze(2) * t
        cos_term = torch.cos(x_t).mean(dim=0)
        sin_term = torch.sin(x_t).mean(dim=0)
        phi = torch.exp(-0.5 * self.t.to(device=x.device, dtype=x.dtype) ** 2)
        diff_sq = (cos_term - phi) ** 2 + sin_term**2
        weight = self.weight.to(device=x.device, dtype=x.dtype)
        statistic = n * 2.0 * torch.trapz(diff_sq * weight, self.t.to(x), dim=1)
        return statistic.mean()


class SIGReg(nn.Module):
    """
    Sketched Isotropic Gaussian Regularizer.

    Each random projection is standardized using its batch mean and standard
    deviation before the Epps–Pulley test. Without this step, the test compares
    arbitrary-scale projections against N(0, 1), conflating scale/offset with
    non-Gaussian shape and making the regularizer's behavior poorly calibrated.
    """

    def __init__(
        self,
        n_projections: int = 256,
        use_epps: bool = True,
        **epps_kwargs,
    ):
        super().__init__()
        if not isinstance(n_projections, int) or n_projections < 1:
            raise ValueError("n_projections must be an integer >= 1")
        self.n_projections = n_projections
        self.use_epps = use_epps
        if use_epps:
            self.epps = EppsPulley(**epps_kwargs)

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        if z.ndim != 2 or z.shape[0] < 2 or z.shape[1] < 1:
            raise ValueError("z must have shape (N, D), with N >= 2 and D >= 1")
        if not torch.is_floating_point(z):
            raise TypeError("z must be a floating-point tensor")
        if not torch.isfinite(z).all():
            raise ValueError("z must contain only finite values")

        _, dim = z.shape
        directions = torch.randn(
            dim, self.n_projections, device=z.device, dtype=z.dtype
        )
        directions = directions / directions.norm(dim=0, keepdim=True).clamp_min(1e-8)
        projections = z @ directions

        if self.use_epps:
            mean = projections.mean(dim=0, keepdim=True)
            std = projections.std(dim=0, unbiased=False, keepdim=True).clamp_min(1e-6)
            standardized = (projections - mean) / std
            return self.epps(standardized)

        mean = projections.mean(dim=0)
        var = projections.var(dim=0, unbiased=False)
        return mean.pow(2).mean() + (var - 1.0).pow(2).mean()
