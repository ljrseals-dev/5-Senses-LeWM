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
        if isinstance(dim, bool) or not isinstance(dim, int) or dim < 1:
            raise ValueError("dim must be a positive integer")
        self.dim = dim
        self.patterns: List[np.ndarray] = []
        self.labels: List[str] = []

    def _vector(self, value: np.ndarray, name: str) -> np.ndarray:
        vector = np.asarray(value)
        if not np.issubdtype(vector.dtype, np.number) or np.issubdtype(
            vector.dtype, np.complexfloating
        ):
            raise TypeError(f"{name} must contain real numeric values")
        vector = vector.reshape(-1)
        if vector.size == 0:
            raise ValueError(f"{name} must not be empty")
        if vector.size > self.dim:
            raise ValueError(
                f"{name} has {vector.size} values; memory dimension is {self.dim}; "
                "refusing to truncate data"
            )
        if not np.isfinite(vector).all():
            raise ValueError(f"{name} must contain only finite values")
        vector = np.sign(vector)
        if vector.size < self.dim:
            vector = np.pad(vector, (0, self.dim - vector.size))
        return vector

    def store(self, pattern: np.ndarray, label: str = "") -> None:
        if not isinstance(label, str):
            raise TypeError("label must be a string")
        p = self._vector(pattern, "pattern")
        self.patterns.append(p)
        self.labels.append(label or f"pattern_{len(self.patterns)}")

    def recall(self, cue: np.ndarray, steps: int = 5) -> Tuple[np.ndarray, Optional[str]]:
        if isinstance(steps, bool) or not isinstance(steps, int) or steps < 0:
            raise ValueError("steps must be a non-negative integer")
        if not self.patterns:
            return np.asarray(cue).copy(), None
        state = self._vector(cue, "cue")

        # Hebbian weight matrix (outer-product sum).
        W = np.zeros((self.dim, self.dim))
        for p in self.patterns:
            W += np.outer(p, p)
        np.fill_diagonal(W, 0)

        for _ in range(steps):
            updated = np.sign(W @ state)
            # Preserve the prior bit when activation is exactly zero.
            state = np.where(updated == 0, state, updated)

        dists = [np.sum(state != p) for p in self.patterns]
        idx = int(np.argmin(dists))
        return self.patterns[idx].copy(), self.labels[idx]

    def __len__(self) -> int:
        return len(self.patterns)
