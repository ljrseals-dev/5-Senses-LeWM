"""
BallWorld: a tiny first-order physics world used for training and for checking
what the world model has learned. Numpy only (no torch).

Laws (all observable from single frames, which a one-frame JEPA needs):
  * action:  position shifts by ``action_scale * clip(a, -1, 1)`` per step
  * drift:   a constant per-step displacement ("gravity"), default none
  * walls:   position is clamped to ``[radius, 1 - radius]`` (objects are blocked)
Image coordinates: x to the right, y downward, both in [0, 1].
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from typing import Dict, Optional, Tuple

import numpy as np


@dataclass
class BallWorldConfig:
    image_size: int = 64
    radius: float = 0.08            # fraction of the image side
    action_scale: float = 0.08      # max displacement per step
    drift_x: float = 0.0
    drift_y: float = 0.02           # gravity pulls toward +y (down)

    @property
    def lo(self) -> float:
        return self.radius

    @property
    def hi(self) -> float:
        return 1.0 - self.radius


class BallWorld:
    def __init__(self, cfg: Optional[BallWorldConfig] = None, seed: int = 0):
        self.cfg = cfg or BallWorldConfig()
        self.rng = np.random.default_rng(seed)
        self.pos = np.array([0.5, 0.5])

    def reset(self, pos: Optional[np.ndarray] = None) -> np.ndarray:
        self.pos = (self.rng.uniform(self.cfg.lo, self.cfg.hi, 2)
                    if pos is None else np.asarray(pos, dtype=np.float64).copy())
        return self.pos.copy()

    def step(self, action: np.ndarray) -> np.ndarray:
        c = self.cfg
        a = np.clip(np.asarray(action, dtype=np.float64), -1.0, 1.0)
        self.pos = np.clip(self.pos + c.action_scale * a + np.array([c.drift_x, c.drift_y]),
                           c.lo, c.hi)
        return self.pos.copy()

    def render(self, pos: Optional[np.ndarray] = None) -> np.ndarray:
        return render_batch(np.asarray(self.pos if pos is None else pos)[None], self.cfg)[0]


def render_batch(pos: np.ndarray, cfg: BallWorldConfig) -> np.ndarray:
    """pos: (N, 2) in [0,1] -> float32 frames (N, 3, H, W) in [0, 1]."""
    pos = np.asarray(pos, dtype=np.float64).reshape(-1, 2)
    S = cfg.image_size
    grid = np.arange(S, dtype=np.float64)
    px = pos[:, 0] * (S - 1)
    py = pos[:, 1] * (S - 1)
    dx = grid[None, None, :] - px[:, None, None]          # (N, 1, W)
    dy = grid[None, :, None] - py[:, None, None]          # (N, H, 1)
    dist = np.sqrt(dx ** 2 + dy ** 2)                     # (N, H, W)
    cover = np.clip(cfg.radius * (S - 1) - dist + 0.5, 0.0, 1.0)
    colour = np.array([1.0, 0.35, 0.2]).reshape(1, 3, 1, 1)
    return (cover[:, None] * colour).astype(np.float32)


def collect(n_traj: int, steps: int, cfg: Optional[BallWorldConfig] = None, seed: int = 0,
            zero_action_prob: float = 0.2, stickiness: float = 0.7) -> Dict[str, np.ndarray]:
    """
    Random-policy trajectories. Actions persist (``stickiness``) so objects reach
    walls; some steps use zero action so the drift law is observable.
    Returns obs uint8 (n, T+1, 3, H, W), actions float32 (n, T, 2), states float32 (n, T+1, 2).
    """
    cfg = cfg or BallWorldConfig()
    env = BallWorld(cfg, seed)
    rng = np.random.default_rng(seed + 1)
    obs = np.zeros((n_traj, steps + 1, 3, cfg.image_size, cfg.image_size), np.uint8)
    acts = np.zeros((n_traj, steps, 2), np.float32)
    states = np.zeros((n_traj, steps + 1, 2), np.float32)
    for i in range(n_traj):
        states[i, 0] = env.reset()
        a = rng.uniform(-1, 1, 2)
        for t in range(steps):
            if rng.random() > stickiness:
                a = rng.uniform(-1, 1, 2)
            a_t = np.zeros(2) if rng.random() < zero_action_prob else a
            acts[i, t] = a_t
            states[i, t + 1] = env.step(a_t)
        frames = render_batch(states[i], cfg)
        obs[i] = np.round(frames * 255).astype(np.uint8)
    return {"obs": obs, "actions": acts, "states": states}


def save_dataset(path: str, data: Dict[str, np.ndarray], cfg: BallWorldConfig) -> None:
    np.savez_compressed(path, config=np.array(json.dumps(asdict(cfg))), **data)


def load_dataset(path: str) -> Tuple[Dict[str, np.ndarray], BallWorldConfig]:
    d = np.load(path, allow_pickle=False)
    cfg = BallWorldConfig(**json.loads(str(d["config"])))
    return {k: d[k] for k in ("obs", "actions", "states")}, cfg


def to_float(obs: np.ndarray) -> np.ndarray:
    """uint8 frames -> float32 in [0, 1] (float input passes through)."""
    return obs.astype(np.float32) / 255.0 if obs.dtype == np.uint8 else obs.astype(np.float32)
