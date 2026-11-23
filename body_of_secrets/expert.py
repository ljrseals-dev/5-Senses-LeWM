"""Minimal rule-based expert system with provenance."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional


@dataclass
class Rule:
    name: str
    condition: str          # human-readable
    action: str             # human-readable
    provenance: str = "user"
    confidence: float = 1.0


@dataclass
class ExpertSystem:
    """Transparent, offline-first rule base."""

    rules: List[Rule] = field(default_factory=list)
    facts: Dict[str, Any] = field(default_factory=dict)

    def add_rule(self, name: str, condition: str, action: str,
                 provenance: str = "user", confidence: float = 1.0) -> None:
        self.rules.append(Rule(name, condition, action, provenance, confidence))

    def assert_fact(self, key: str, value: Any, source: str = "sensor") -> None:
        self.facts[key] = {"value": value, "source": source}

    def query(self, question: str) -> str:
        """Very simple keyword matching for demonstration."""
        q = question.lower()
        matched = []
        for r in self.rules:
            if any(tok in q for tok in r.condition.lower().split()):
                matched.append(r)
        if not matched:
            return f"No rule matched for: {question}"
        best = max(matched, key=lambda r: r.confidence)
        return (
            f"[{best.name}] {best.action} "
            f"(confidence={best.confidence:.2f}, provenance={best.provenance})"
        )

    def list_rules(self) -> List[str]:
        return [f"{r.name}: IF {r.condition} THEN {r.action}" for r in self.rules]