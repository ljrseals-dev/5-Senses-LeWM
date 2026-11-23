"""Classical brain that can ingest sensory embeddings from the Eyes."""

from __future__ import annotations

from typing import Any, Optional

import numpy as np
import torch

from .expert import ExpertSystem
from .memory import AssociativeMemory


class ClassicalBrain:
    """
    Lightweight classical controller.

    Receives one latent embedding at a time from the Eyes module. The default
    memory dimension matches the default 192-dimensional Eye representation.
    """

    def __init__(self, memory_dim: int = 192):
        self.memory = AssociativeMemory(dim=memory_dim)
        self.expert = ExpertSystem()
        self._last_latent: Optional[np.ndarray] = None

        self.expert.add_rule(
            name="visual-surprise",
            condition="surprise high unexpected physics",
            action="flag anomaly and request human review",
            provenance="built-in",
            confidence=0.9,
        )
        self.expert.add_rule(
            name="goal-reached",
            condition="latent close to goal embedding",
            action="halt and report success",
            provenance="built-in",
            confidence=0.95,
        )

    def ingest_sensory(self, latent: torch.Tensor | np.ndarray, label: str = "obs") -> None:
        if isinstance(latent, torch.Tensor):
            if latent.ndim == 2:
                if latent.shape[0] != 1:
                    raise ValueError("ingest_sensory accepts exactly one latent vector")
                latent = latent.squeeze(0)
            elif latent.ndim != 1:
                raise ValueError("latent must have shape (D,) or (1, D)")
            arr = latent.detach().to(device="cpu", dtype=torch.float32).numpy()
        else:
            arr = np.asarray(latent)
            if arr.ndim == 2:
                if arr.shape[0] != 1:
                    raise ValueError("ingest_sensory accepts exactly one latent vector")
                arr = arr[0]
            elif arr.ndim != 1:
                raise ValueError("latent must have shape (D,) or (1, D)")
            if not np.issubdtype(arr.dtype, np.number):
                raise TypeError("latent must contain numeric values")
            arr = arr.astype(np.float32, copy=False)

        if not np.isfinite(arr).all():
            raise ValueError("latent must contain only finite values")
        arr = arr.copy()
        self.memory.store(arr, label=label)
        self._last_latent = arr
        self.expert.assert_fact("last_latent_norm", float(np.linalg.norm(arr)), source="eyes")

    def ask(self, question: str) -> str:
        answer = self.expert.query(question)
        if "No rule matched" not in answer:
            return answer
        if self._last_latent is not None:
            _, label = self.memory.recall(self._last_latent)
            return f"Memory recall → {label} (distance info available in full system)"
        return "Brain has no sensory data yet."

    def status(self) -> dict[str, Any]:
        return {
            "memory_patterns": len(self.memory),
            "rules": len(self.expert.rules),
            "facts": list(self.expert.facts.keys()),
            "has_last_latent": self._last_latent is not None,
        }
