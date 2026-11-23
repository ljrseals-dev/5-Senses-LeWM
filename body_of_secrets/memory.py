"""Lightweight Hopfield-style associative memory (educational stub)."""

from __future__ import annotations

from typing import List, Optional, Tuple
import numpy as np


class AssociativeMemory:
    """
    Extremely simplified binary Hopfield-style store.
    Not production-grade; exists only to keep the classical side transparent.
    """

    def __init__(self, dim: int = 64):
        self.dim = dim
        self.patterns: List[np.ndarray] = []
        self.labels: List[str] = []

    def store(self, pattern: np.ndarray, label: str = "") -> None:
        p = np.sign(pattern.flatten()[: self.dim])
        if p.shape[0] < self.dim:
            p = np.pad(p, (0, self.dim - p.shape[0]))
        self.patterns.append(p)
        self.labels.append(label or f"pattern_{len(self.patterns)}")

    def recall(self, cue: np.ndarray, steps: int = 5) -> Tuple[np.ndarray, Optional[str]]:
        if not self.patterns:
            return cue, None
        state = np.sign(cue.flatten()[: self.dim])
        if state.shape[0] < self.dim:
            state = np.pad(state, (0, self.dim - state.shape[0]))

        # Hebbian weight matrix (outer-product sum)
        W = np.zeros((self.dim, self.dim))
        for p in self.patterns:
            W += np.outer(p, p)
        np.fill_diagonal(W, 0)

        for _ in range(steps):
            state = np.sign(W @ state)

        # nearest stored pattern
        dists = [np.sum(state != p) for p in self.patterns]
        idx = int(np.argmin(dists))
        return self.patterns[idx], self.labels[idx]

    def __len__(self) -> int:
        return len(self.patterns)