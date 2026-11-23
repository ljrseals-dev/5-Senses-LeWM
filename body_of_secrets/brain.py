"""Classical brain that can ingest sensory embeddings from the Eyes."""

from __future__ import annotations

from typing import Any, Optional
import numpy as np
import torch

from .memory import AssociativeMemory
from .expert import ExpertSystem


class ClassicalBrain:
    """
    Lightweight classical controller.
    Receives latent embeddings from the world-model Eye and stores them
    in associative memory; expert rules can later query the memory.
    """

    def __init__(self, memory_dim: int = 64):
        self.memory = AssociativeMemory(dim=memory_dim)
        self.expert = ExpertSystem()
        self._last_latent: Optional[np.ndarray] = None

        # seed a couple of transparent rules
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
            arr = latent.detach().cpu().numpy().flatten()
        else:
            arr = np.asarray(latent).flatten()
        self._last_latent = arr
        self.memory.store(arr, label=label)
        self.expert.assert_fact("last_latent_norm", float(np.linalg.norm(arr)), source="eyes")

    def ask(self, question: str) -> str:
        # first try expert
        answer = self.expert.query(question)
        if "No rule matched" not in answer:
            return answer
        # fall back to memory recall
        if self._last_latent is not None:
            recalled, label = self.memory.recall(self._last_latent)
            return f"Memory recall → {label} (distance info available in full system)"
        return "Brain has no sensory data yet."

    def status(self) -> dict[str, Any]:
        return {
            "memory_patterns": len(self.memory),
            "rules": len(self.expert.rules),
            "facts": list(self.expert.facts.keys()),
            "has_last_latent": self._last_latent is not None,
        }