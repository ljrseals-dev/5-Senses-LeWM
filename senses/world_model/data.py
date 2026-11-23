"""Datasets for local training: synthetic trajectories (.npz) or your own image folders."""

from __future__ import annotations

import os
from typing import Optional, Tuple

import numpy as np
import torch
from torch.utils.data import Dataset


class PairDataset(Dataset):
    """(obs_t, action_t, obs_{t+1}) triples from trajectories. Frames stay uint8 in memory."""

    def __init__(self, obs: np.ndarray, actions: np.ndarray):
        if obs.ndim != 5 or obs.shape[2] != 3:
            raise ValueError("obs must be (n_traj, T+1, 3, H, W)")
        if obs.shape[1] != actions.shape[1] + 1:
            raise ValueError("need one more frame than actions per trajectory")
        n, T1 = obs.shape[:2]
        shape = obs.shape[2:]
        self.obs_t = torch.from_numpy(np.ascontiguousarray(obs[:, :-1]).reshape(-1, *shape))
        self.obs_tp1 = torch.from_numpy(np.ascontiguousarray(obs[:, 1:]).reshape(-1, *shape))
        self.actions = torch.from_numpy(np.ascontiguousarray(actions, dtype=np.float32)
                                        ).reshape(-1, actions.shape[-1])

    def __len__(self) -> int:
        return self.obs_t.shape[0]

    def __getitem__(self, i: int):
        return self.obs_t[i].float() / 255.0, self.actions[i], self.obs_tp1[i].float() / 255.0

    def sample(self, batch_size: int, device: torch.device,
               generator: Optional[torch.Generator] = None) -> Tuple[torch.Tensor, ...]:
        idx = torch.randint(len(self), (batch_size,), generator=generator)
        return (self.obs_t[idx].to(device).float().div_(255.0), self.actions[idx].to(device),
                self.obs_tp1[idx].to(device).float().div_(255.0))


def frames_from_folder(folder: str, size: int = 64) -> np.ndarray:
    """Load images (sorted by file name) as uint8 (N, 3, size, size). Needs Pillow."""
    from PIL import Image
    exts = (".png", ".jpg", ".jpeg", ".bmp", ".webp")
    names = sorted(f for f in os.listdir(folder) if f.lower().endswith(exts))
    if not names:
        raise FileNotFoundError(f"no images found in {folder}")
    arr = [np.asarray(Image.open(os.path.join(folder, n)).convert("RGB").resize((size, size)))
           for n in names]
    return np.stack(arr).transpose(0, 3, 1, 2).astype(np.uint8)


def pairs_from_frames(frames: np.ndarray, actions: Optional[np.ndarray] = None,
                      action_dim: int = 2) -> PairDataset:
    """One ordered frame sequence -> PairDataset. Without recorded actions, zero actions are used,
    so the model learns passive dynamics only (it cannot learn what actions do)."""
    if actions is None:
        actions = np.zeros((len(frames) - 1, action_dim), np.float32)
    return PairDataset(frames[None], np.asarray(actions, np.float32)[None])
