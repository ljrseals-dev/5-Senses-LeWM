"""Hopfield associative memory (classical, offline, transparent)."""

from __future__ import annotations

from dataclasses import dataclass
from typing import List, Optional

import numpy as np


@dataclass
class RecallResult:
    """Outcome of one recall, with enough detail to audit it."""

    pattern: Optional[np.ndarray]   # stored bipolar pattern that the state settled nearest to
    label: Optional[str]
    index: Optional[int]
    hamming: int                    # bits differing between settled state and that pattern
    overlap: float                  # 1.0 = identical, 0.0 = unrelated, -1.0 = inverted
    energy: float                   # Hopfield energy of the settled state
    converged: bool
    sweeps: int
    recognized: bool                # overlap >= recognition threshold


class AssociativeMemory:
    """
    Binary Hopfield network with Hebbian storage and asynchronous recall.

    * Real-valued inputs are binarised by sign (>= 0 -> +1, < 0 -> -1).
    * ``dim=None`` infers the dimension from the first stored vector. If a
      fixed ``dim`` is given, inputs of another length are mapped to it with a
      seeded random projection (instead of silently truncating).
    * Reliable capacity is roughly 0.138 * dim random patterns; beyond that
      recall degrades (see :attr:`overloaded`).
    """

    def __init__(self, dim: Optional[int] = None, seed: int = 0,
                 recognition_threshold: float = 0.6):
        self.dim = dim
        self.seed = seed
        self.recognition_threshold = recognition_threshold
        self.patterns: List[np.ndarray] = []
        self.labels: List[str] = []
        self._W: Optional[np.ndarray] = None
        self._proj_cache: dict[int, np.ndarray] = {}

    # ------------------------------------------------------------------ utils
    def _bipolar(self, x: np.ndarray) -> np.ndarray:
        v = np.asarray(x, dtype=np.float64).ravel()
        if v.size == 0:
            raise ValueError("empty pattern")
        if self.dim is None:
            self.dim = int(v.size)
        if v.size != self.dim:
            P = self._proj_cache.get(v.size)
            if P is None:
                rng = np.random.default_rng(self.seed + v.size)
                P = rng.standard_normal((self.dim, v.size)) / np.sqrt(v.size)
                self._proj_cache[v.size] = P
            v = P @ v
        return np.where(v >= 0, 1, -1).astype(np.int8)

    @property
    def capacity(self) -> int:
        return int(0.138 * self.dim) if self.dim else 0

    @property
    def overloaded(self) -> bool:
        return len(self.patterns) > self.capacity

    # ------------------------------------------------------------------- API
    def store(self, pattern: np.ndarray, label: str = "") -> int:
        p = self._bipolar(pattern)
        if self._W is None:
            self._W = np.zeros((self.dim, self.dim), dtype=np.float64)
        pf = p.astype(np.float64)
        self._W += np.outer(pf, pf) / self.dim
        np.fill_diagonal(self._W, 0.0)
        self.patterns.append(p)
        self.labels.append(label or f"pattern_{len(self.patterns)}")
        return len(self.patterns) - 1

    def energy(self, state: np.ndarray) -> float:
        s = state.astype(np.float64)
        return float(-0.5 * s @ self._W @ s)

    def recall(self, cue: np.ndarray, max_sweeps: int = 20, seed: Optional[int] = None
               ) -> RecallResult:
        if not self.patterns:
            return RecallResult(None, None, None, 0, 0.0, 0.0, True, 0, False)

        state = self._bipolar(cue).astype(np.float64)
        rng = np.random.default_rng(self.seed if seed is None else seed)
        converged, sweeps = False, 0
        for sweeps in range(1, max_sweeps + 1):
            changed = False
            for i in rng.permutation(self.dim):          # asynchronous updates
                h = self._W[i] @ state
                new = state[i] if h == 0 else (1.0 if h > 0 else -1.0)
                if new != state[i]:
                    state[i] = new
                    changed = True
            if not changed:
                converged = True
                break

        stored = np.stack(self.patterns)                  # (P, dim)
        s8 = state.astype(np.int8)
        ham = (stored != s8).sum(axis=1)
        idx = int(np.argmin(ham))
        overlap = 1.0 - 2.0 * float(ham[idx]) / self.dim
        return RecallResult(
            pattern=self.patterns[idx], label=self.labels[idx], index=idx,
            hamming=int(ham[idx]), overlap=overlap, energy=self.energy(s8),
            converged=converged, sweeps=sweeps,
            recognized=overlap >= self.recognition_threshold,
        )

    def __len__(self) -> int:
        return len(self.patterns)

    # ----------------------------------------------------------- persistence
    def save(self, path: str) -> None:
        np.savez_compressed(
            path,
            patterns=np.stack(self.patterns) if self.patterns else np.zeros((0, self.dim or 0), np.int8),
            labels=np.array(self.labels, dtype=str),
            dim=np.array(self.dim if self.dim is not None else -1),
            seed=np.array(self.seed),
        )

    @classmethod
    def load(cls, path: str) -> "AssociativeMemory":
        d = np.load(path, allow_pickle=False)
        dim = int(d["dim"])
        mem = cls(dim=None if dim < 0 else dim, seed=int(d["seed"]))
        for p, lab in zip(d["patterns"], d["labels"]):
            mem.store(p, str(lab))
        return mem
