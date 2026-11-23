"""
Bridge between world-model latents and the classical brain.
"""

from __future__ import annotations

from typing import Optional
import torch
from .jepa import LeWMStyle


class WorldModelEye:
    """
    “Eyes” sensory module.

    Can be used standalone or to feed latent physics embeddings
    into the classical brain (body_of_secrets.brain.ClassicalBrain).
    """

    def __init__(self, model: Optional[LeWMStyle] = None, device: str = "cpu"):
        self.device = torch.device(device)
        self.model = model or LeWMStyle()
        self.model.to(self.device)
        self.model.eval()

    @torch.no_grad()
    def encode(self, observation: torch.Tensor) -> torch.Tensor:
        """observation: (B, 3, H, W) or (3, H, W)"""
        if observation.dim() == 3:
            observation = observation.unsqueeze(0)
        observation = observation.to(self.device)
        return self.model.encode(observation)

    @torch.no_grad()
    def predict_next(self, z: torch.Tensor, action: torch.Tensor) -> torch.Tensor:
        z = z.to(self.device)
        action = action.to(self.device)
        return self.model.predict(z, action)

    def surprise(
        self,
        obs_t: torch.Tensor,
        action: torch.Tensor,
        obs_tp1: torch.Tensor,
    ) -> float:
        """Return scalar surprise (prediction + reg loss)."""
        self.model.eval()
        with torch.no_grad():
            loss, *_ = self.model(obs_t.to(self.device), action.to(self.device), obs_tp1.to(self.device))
        return float(loss.item())

    def train_mode(self) -> None:
        self.model.train()

    def eval_mode(self) -> None:
        self.model.eval()