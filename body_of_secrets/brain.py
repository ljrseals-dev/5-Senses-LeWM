"""Classical brain that ingests sensory embeddings from the Eyes."""

from __future__ import annotations

import json
import os
from typing import Any, Dict, List, Optional

import numpy as np

from .expert import ExpertSystem, Firing
from .memory import AssociativeMemory
from .truth import TruthStatus


def _to_vec(x: Any) -> np.ndarray:
    """Accept numpy arrays, lists or torch tensors (duck-typed; no torch import)."""
    if hasattr(x, "detach"):
        x = x.detach().cpu().numpy()
    return np.asarray(x, dtype=np.float64).ravel()


class ClassicalBrain:
    """
    Receives latent embeddings from the world-model Eye, keeps them in Hopfield
    memory, tracks goal/surprise/physics facts and lets transparent rules reason
    over them.
    """

    def __init__(self, memory_dim: Optional[int] = None,
                 surprise_z_threshold: float = 3.0):
        self.memory = AssociativeMemory(dim=memory_dim)
        self.expert = ExpertSystem()
        self._last_latent: Optional[np.ndarray] = None
        self._goal: Optional[np.ndarray] = None

        e = self.expert
        e.add_rule("visual-surprise", "surprise_z >= %g" % surprise_z_threshold,
                   "flag anomaly and request human review",
                   provenance="built-in", confidence=0.9,
                   description="surprise high unexpected physics anomaly")
        e.add_rule("goal-reached", "goal_distance <= goal_tolerance",
                   "halt and report success",
                   provenance="built-in", confidence=0.95,
                   description="latent close to goal embedding reached")
        e.add_rule("plan-grounded", "min_law_score >= 0.8",
                   "latent plans agree with measured physics; allow plan execution",
                   provenance="built-in", confidence=0.9,
                   description="planning plan grounded physical laws trusted")
        e.add_rule("plan-ungrounded", "min_law_score < 0.5",
                   "do not execute latent plans; retrain or collect more data",
                   provenance="built-in", confidence=0.9,
                   description="planning plan ungrounded physical laws untrusted")

    # ------------------------------------------------------------- sensing
    def set_goal(self, latent: Any, tolerance: float) -> None:
        self._goal = _to_vec(latent)
        self.expert.assert_fact("goal_tolerance", float(tolerance), source="user",
                                status=TruthStatus.OFFICIAL)

    def ingest_sensory(self, latent: Any, label: str = "obs",
                       surprise_z: Optional[float] = None) -> None:
        arr = _to_vec(latent)
        if len(self.memory):                         # familiarity before storing
            rec = self.memory.recall(arr)
            self.expert.assert_fact("memory_overlap", float(rec.overlap), source="memory")
        self._last_latent = arr
        self.memory.store(arr, label=label)
        self.expert.assert_fact("last_latent_norm", float(np.linalg.norm(arr)), source="eyes")
        if self._goal is not None and self._goal.shape == arr.shape:
            self.expert.assert_fact("goal_distance", float(np.linalg.norm(arr - self._goal)),
                                    source="eyes")
        if surprise_z is not None:
            self.expert.assert_fact("surprise_z", float(surprise_z), source="eyes")

    def ingest_physics(self, report: Any) -> None:
        """Ground rules in measured laws: ``report`` needs ``assert_into(expert)``."""
        report.assert_into(self.expert)

    # ------------------------------------------------------------ reasoning
    def infer(self) -> List[Firing]:
        return self.expert.infer()

    def ask(self, question: str) -> str:
        if self.expert.query_rule(question) is not None:
            return self.expert.query(question)
        if self._last_latent is not None and len(self.memory):
            r = self.memory.recall(self._last_latent)
            return (f"Memory recall → {r.label} (overlap={r.overlap:.2f}, hamming={r.hamming}/"
                    f"{self.memory.dim}, converged={r.converged}, recognized={r.recognized})")
        return "Brain has no sensory data yet."

    def status(self) -> Dict[str, Any]:
        return {
            "memory_patterns": len(self.memory),
            "memory_capacity": self.memory.capacity,
            "rules": len(self.expert.rules),
            "facts": list(self.expert.facts.keys()),
            "has_last_latent": self._last_latent is not None,
            "has_goal": self._goal is not None,
        }

    # ---------------------------------------------------------- persistence
    def save(self, directory: str) -> None:
        os.makedirs(directory, exist_ok=True)
        if len(self.memory):
            self.memory.save(os.path.join(directory, "memory.npz"))
        facts = {k: {"value": (float(f.value) if isinstance(f.value, (int, float, np.floating)) else str(f.value)),
                     "source": f.source, "status": f.status.name}
                 for k, f in self.expert.facts.items()}
        with open(os.path.join(directory, "facts.json"), "w", encoding="utf-8") as fh:
            json.dump(facts, fh, indent=2)

    def load(self, directory: str) -> None:
        mp = os.path.join(directory, "memory.npz")
        if os.path.exists(mp):
            self.memory = AssociativeMemory.load(mp)
        fp = os.path.join(directory, "facts.json")
        if os.path.exists(fp):
            with open(fp, encoding="utf-8") as fh:
                for k, d in json.load(fh).items():
                    self.expert.assert_fact(k, d["value"], d["source"], d["status"])
