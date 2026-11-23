"""
Bridge between a world-model backend and the classical brain.

A *backend* is anything with ``latent_dim``, ``action_dim``, ``encode(obs)->z`` and
``predict(z, action)->z`` (torch tensors). The educational ``LeWMStyle`` is one; the official
LeWorldModel (lucas-maes/le-wm) can be plugged in the same way by wrapping it in
``CallableBackend`` (see the README — that glue is yours to verify against the official API).
"""

from __future__ import annotations

from typing import Any, Callable, Dict, Optional

import numpy as np
import torch

from ..audit import append_jsonl
from .jepa import LeWMStyle
from .planner import LatentCEMPlanner, PlanResult


class CallableBackend:
    """Wrap any encode/predict callables (e.g. an official LeWM model) as a backend."""

    def __init__(self, encode_fn: Callable, predict_fn: Callable, latent_dim: int, action_dim: int):
        self._enc, self._pred = encode_fn, predict_fn
        self.latent_dim, self.action_dim = latent_dim, action_dim

    def encode(self, obs):
        return self._enc(obs)

    def predict(self, z, action):
        return self._pred(z, action)

    def to(self, device):          # nn.Module-like no-ops so WorldModelEye can treat it uniformly
        return self

    def eval(self):
        return self

    def train(self, mode: bool = True):
        return self


def _is_uint8(obs: Any) -> bool:
    if isinstance(obs, torch.Tensor):
        return obs.dtype == torch.uint8
    return isinstance(obs, np.ndarray) and obs.dtype == np.uint8


def _as_frames(obs: Any) -> torch.Tensor:
    """numpy/tensor frames -> float32 (B,3,H,W) in [0,1]. Accepts uint8, HWC or CHW, single or batch."""
    u8 = _is_uint8(obs)
    t = torch.as_tensor(obs)
    if t.dim() == 3:
        t = t.unsqueeze(0)
    if t.shape[-1] == 3 and t.shape[1] != 3:          # (B,H,W,3) -> (B,3,H,W)
        t = t.permute(0, 3, 1, 2)
    t = t.float()
    return t / 255.0 if u8 else t


class WorldModelEye:
    """“Eyes” sensory module: pixels -> latent physics embeddings, prediction, surprise, planning."""

    def __init__(self, model=None, device: str = "cpu"):
        self.device = torch.device(device)
        self.model = model if model is not None else LeWMStyle()
        self.model.to(self.device)
        self.model.eval()
        self._s_mean: Optional[float] = None
        self._s_std: Optional[float] = None

    @property
    def latent_dim(self) -> int:
        return self.model.latent_dim

    @property
    def action_dim(self) -> int:
        return self.model.action_dim

    # ---------------------------------------------------------- torch API
    @torch.no_grad()
    def encode(self, observation: Any) -> torch.Tensor:
        return self.model.encode(_as_frames(observation).to(self.device))

    @torch.no_grad()
    def predict_next(self, z: torch.Tensor, action: Any) -> torch.Tensor:
        a = torch.as_tensor(action, dtype=torch.float32).to(self.device)
        if a.dim() == 1:
            a = a.unsqueeze(0)
        return self.model.predict(z.to(self.device), a)

    # ---------------------------------------------------------- numpy API (used by physics.py)
    def encode_frames(self, frames: np.ndarray) -> np.ndarray:
        return self.encode(frames).cpu().numpy()

    def predict_latents(self, z: np.ndarray, action: np.ndarray) -> np.ndarray:
        return self.predict_next(torch.as_tensor(z, dtype=torch.float32), action).cpu().numpy()

    # ---------------------------------------------------------- surprise
    @torch.no_grad()
    def latent_error(self, obs_t: Any, action: Any, obs_tp1: Any) -> torch.Tensor:
        """Per-sample mean squared error between predicted and actual next latent, shape (B,)."""
        z_pred = self.predict_next(self.encode(obs_t), action)
        z_next = self.encode(obs_tp1)
        return (z_pred - z_next).pow(2).mean(dim=-1)

    def surprise(self, obs_t: Any, action: Any, obs_tp1: Any) -> float:
        """Scalar surprise = mean latent prediction error (no regulariser term mixed in)."""
        return float(self.latent_error(obs_t, action, obs_tp1).mean().item())

    def calibrate(self, obs_t: Any, action: Any, obs_tp1: Any) -> None:
        """Record the error distribution on *normal* transitions so surprise can be z-scored."""
        e = self.latent_error(obs_t, action, obs_tp1).cpu().numpy()
        self._s_mean, self._s_std = float(e.mean()), float(e.std() + 1e-12)

    def surprise_z(self, obs_t: Any, action: Any, obs_tp1: Any) -> np.ndarray:
        """Per-sample z-score of the latent error versus the calibrated normal distribution."""
        if self._s_mean is None:
            raise RuntimeError("call calibrate() on normal transitions first")
        e = self.latent_error(obs_t, action, obs_tp1).cpu().numpy()
        return (e - self._s_mean) / self._s_std

    # ---------------------------------------------------------- planning
    def plan(self, obs_start: Any, obs_goal: Any, audit_path: Optional[str] = None,
             **planner_kwargs) -> PlanResult:
        planner = LatentCEMPlanner(self.model, **planner_kwargs)
        res = planner.plan(self.encode(obs_start)[0], self.encode(obs_goal)[0])
        if audit_path:
            append_jsonl(audit_path, {"event": "plan", **res.to_dict()})
        return res

    # ---------------------------------------------------------- persistence
    def save_checkpoint(self, path: str) -> None:
        if not isinstance(self.model, LeWMStyle):
            raise TypeError("save_checkpoint supports the educational LeWMStyle model only")
        torch.save({"state_dict": self.model.state_dict(), "config": self.model.config_dict(),
                    "surprise": {"mean": self._s_mean, "std": self._s_std}}, path)

    @classmethod
    def from_checkpoint(cls, path: str, device: str = "cpu") -> "WorldModelEye":
        """Load a checkpoint written by ``save_checkpoint`` (weights-only, no pickle code run)."""
        ck: Dict[str, Any] = torch.load(path, map_location="cpu", weights_only=True)
        model = LeWMStyle(**ck["config"])
        model.load_state_dict(ck["state_dict"])
        eye = cls(model, device=device)
        s = ck.get("surprise") or {}
        eye._s_mean, eye._s_std = s.get("mean"), s.get("std")
        return eye

    # ---------------------------------------------------------- misc
    def train_mode(self) -> None:
        self.model.train()

    def eval_mode(self) -> None:
        self.model.eval()
