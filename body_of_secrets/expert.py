"""Minimal rule-based expert system with provenance."""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Any, Dict, List


@dataclass
class Rule:
    name: str
    condition: str
    action: str
    provenance: str = "user"
    confidence: float = 1.0

    def __post_init__(self) -> None:
        if not all(isinstance(value, str) and value.strip() for value in (
            self.name, self.condition, self.action, self.provenance
        )):
            raise ValueError("rule name, condition, action, and provenance must be non-empty strings")
        if (
            isinstance(self.confidence, bool)
            or not isinstance(self.confidence, (int, float))
            or not math.isfinite(self.confidence)
            or not 0.0 <= self.confidence <= 1.0
        ):
            raise ValueError("confidence must be finite and between 0 and 1")


@dataclass
class ExpertSystem:
    """Transparent, offline-first rule base."""

    rules: List[Rule] = field(default_factory=list)
    facts: Dict[str, Any] = field(default_factory=dict)

    def add_rule(
        self,
        name: str,
        condition: str,
        action: str,
        provenance: str = "user",
        confidence: float = 1.0,
    ) -> None:
        self.rules.append(Rule(name, condition, action, provenance, confidence))

    def assert_fact(self, key: str, value: Any, source: str = "sensor") -> None:
        if not isinstance(key, str) or not key.strip():
            raise ValueError("fact key must be a non-empty string")
        if not isinstance(source, str) or not source.strip():
            raise ValueError("fact source must be a non-empty string")
        self.facts[key] = {"value": value, "source": source}

    def query(self, question: str) -> str:
        """Very simple keyword matching for demonstration."""
        if not isinstance(question, str):
            raise TypeError("question must be a string")
        q = question.casefold()
        matched = [
            rule for rule in self.rules
            if any(token in q for token in rule.condition.casefold().split())
        ]
        if not matched:
            return f"No rule matched for: {question}"
        best = max(matched, key=lambda rule: rule.confidence)
        return (
            f"[{best.name}] {best.action} "
            f"(confidence={best.confidence:.2f}, provenance={best.provenance})"
        )

    def list_rules(self) -> List[str]:
        return [f"{rule.name}: IF {rule.condition} THEN {rule.action}" for rule in self.rules]
