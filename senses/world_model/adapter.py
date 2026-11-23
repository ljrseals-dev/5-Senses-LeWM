"""Bridge between visual world-model latents and the classical brain."""

from __future__ import annotations

from typing import Optional

import torch

from .jepa import LeWMStyle


class WorldModelEye:
    """Visual sensory module; encodes observations and predicts their futures."""

    def __init__(self, model: Optional[LeWMStyle] = None, device: str = "cpu"):
        self.device = torch.device(device)
        self.model = model if model is not None else LeWMStyle()
        self.model.to(self.device)
        self.model.eval()

    @torch.no_grad()
    def encode(self, observation: torch.Tensor) -> torch.Tensor:
        """Encode (3,H,W) or (B,3,H,W) observations."""
        if not isinstance(observation, torch.Tensor):
            raise TypeError("observation must be a torch.Tensor")
        if observation.ndim == 3:
            observation = observation.unsqueeze(0)
        if observation.ndim != 4 or observation.shape[1] != 3:
            raise ValueError("observation must have shape (3,H,W) or (B,3,H,W)")
        if observation.shape[0] < 1 or not torch.is_floating_point(observation):
            raise ValueError("observation batch must be non-empty and floating point")
        if not torch.isfinite(observation).all():
            raise ValueError("observation must contain only finite values")
        return self.model.encode(observation.to(self.device))

    @torch.no_grad()
    def predict_next(self, z: torch.Tensor, action: torch.Tensor) -> torch.Tensor:
        if z.ndim != 2 or z.shape[-1] != self.model.latent_dim:
            raise ValueError("z must have shape (B, model.latent_dim)")
        if action.ndim != 2 or action.shape != (z.shape[0], self.model.action_dim):
            raise ValueError("action must have shape (B, model.action_dim)")
        if not torch.isfinite(z).all() or not torch.isfinite(action).all():
            raise ValueError("z and action must contain only finite values")
        return self.model.predict(z.to(self.device), action.to(self.device))

    @torch.no_grad()
    def surprise(
        self,
        obs_t: torch.Tensor,
        action: torch.Tensor,
        obs_tp1: torch.Tensor,
    ) -> float:
        """
        Return prediction error, not the training objective.

        SIGReg is a batch-distribution regularizer; it is not evidence that a
        particular transition was surprising. The prediction MSE is the
        transition-specific signal suitable for anomaly triage.
        """
        was_training = self.model.training
        try:
            self.model.eval()
            obs_t = obs_t.to(self.device)
            action = action.to(self.device)
            obs_tp1 = obs_tp1.to(self.device)
            _, _, _, pred_loss, _ = self.model(obs_t, action, obs_tp1)
            return float(pred_loss.item())
        finally:
            self.model.train(was_training)

    def train_mode(self) -> None:
        self.model.train()

    def eval_mode(self) -> None:
        self.model.eval()
