"""Rule-based expert system with provenance, truth status and forward chaining."""

from __future__ import annotations

import json
import operator
import re
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple, Union

from .truth import TruthStatus, weakest

_OPS = {"<": operator.lt, "<=": operator.le, ">": operator.gt,
        ">=": operator.ge, "==": operator.eq, "!=": operator.ne}
_NUM = r"-?\d+(?:\.\d+)?(?:[eE][-+]?\d+)?"
_CLAUSE = re.compile(rf"^\s*([A-Za-z_][\w.]*)\s*(<=|>=|==|!=|<|>)\s*({_NUM}|[A-Za-z_][\w.]*)\s*$")
_STOP = frozenset("""the and for are was were what who why how when where which that this these those
there here with from have has had does did you your can could would should about into over under
than then them they their its not but any all out one two see""".split())


@dataclass(frozen=True)
class Clause:
    """``fact op rhs`` where rhs is a number or the name of another fact."""

    fact: str
    op: str
    rhs: Union[float, str]

    def __str__(self) -> str:
        return f"{self.fact} {self.op} {self.rhs}"


def parse_when(text: str) -> List[Clause]:
    """Parse ``"a > 3 and b <= c"``. Raises ValueError if not fully parsable."""
    clauses = []
    for part in re.split(r"\s+and\s+", text.strip(), flags=re.IGNORECASE):
        m = _CLAUSE.match(part)
        if not m:
            raise ValueError(f"cannot parse clause: {part!r}")
        fact, op, rhs = m.groups()
        try:
            rhs_v: Union[float, str] = float(rhs)
        except ValueError:
            rhs_v = rhs
        clauses.append(Clause(fact, op, rhs_v))
    return clauses


@dataclass
class Fact:
    value: Any
    source: str = "sensor"
    status: TruthStatus = TruthStatus.PLAUSIBLE


@dataclass
class Rule:
    name: str
    condition: str                  # human-readable
    action: str                     # human-readable
    provenance: str = "user"
    confidence: float = 1.0
    status: TruthStatus = TruthStatus.PLAUSIBLE
    description: str = ""           # extra keywords for question matching
    when: List[Clause] = field(default_factory=list)   # machine-checkable form (may be empty)

    def keywords(self) -> set:
        return _tokens(f"{self.name} {self.description} {self.condition}")


@dataclass
class Evaluation:
    satisfied: Optional[bool]       # None = cannot tell (missing facts or no clauses)
    missing: List[str]
    premises: List[Dict[str, Any]]


@dataclass
class Firing:
    rule: str
    action: str
    confidence: float
    status: TruthStatus
    provenance: str
    premises: List[Dict[str, Any]]

    def explain(self) -> str:
        prem = "; ".join(f"{p['fact']}={p['value']!r} (src={p['source']}, {p['status']})"
                         for p in self.premises) or "no premises"
        return (f"[{self.rule}] {self.action} — confidence={self.confidence:.2f}, "
                f"status={self.status.name}, provenance={self.provenance}; because {prem}")


def _tokens(text: str) -> set:
    return {t for t in re.findall(r"[a-z]+", text.lower()) if len(t) >= 3 and t not in _STOP}


@dataclass
class ExpertSystem:
    """Transparent, offline-first rule base."""

    rules: List[Rule] = field(default_factory=list)
    facts: Dict[str, Fact] = field(default_factory=dict)
    audit: List[Dict[str, Any]] = field(default_factory=list)

    # ---------------------------------------------------------------- editing
    def add_rule(self, name: str, condition: str, action: str,
                 provenance: str = "user", confidence: float = 1.0,
                 status: TruthStatus | str = TruthStatus.PLAUSIBLE,
                 description: str = "", when: Optional[str] = None) -> Rule:
        """
        ``when`` is a machine-checkable condition such as ``"surprise_z >= 3"``.
        If omitted, ``condition`` is tried as one; if it does not parse the
        rule stays a keyword-matched, human-readable rule.
        """
        clauses: List[Clause] = []
        source = when if when is not None else condition
        try:
            clauses = parse_when(source)
        except ValueError:
            if when is not None:
                raise
        rule = Rule(name, condition, action, provenance, float(confidence),
                    TruthStatus.parse(status), description, clauses)
        self.rules.append(rule)
        return rule

    def assert_fact(self, key: str, value: Any, source: str = "sensor",
                    status: TruthStatus | str = TruthStatus.PLAUSIBLE) -> None:
        self.facts[key] = Fact(value, source, TruthStatus.parse(status))

    # ------------------------------------------------------------- inference
    def evaluate(self, rule: Rule) -> Evaluation:
        if not rule.when:
            return Evaluation(None, [], [])
        missing: List[str] = []
        premises: List[Dict[str, Any]] = []
        ok = True
        for c in rule.when:
            names = [c.fact] + ([c.rhs] if isinstance(c.rhs, str) else [])
            absent = [n for n in names if n not in self.facts]
            if absent:
                missing.extend(absent)
                continue
            lhs = self.facts[c.fact]
            rhs_v = self.facts[c.rhs].value if isinstance(c.rhs, str) else c.rhs
            for n in names:
                f = self.facts[n]
                premises.append({"fact": n, "value": f.value, "source": f.source,
                                 "status": f.status.name})
            try:
                ok = ok and bool(_OPS[c.op](lhs.value, rhs_v))
            except TypeError:
                ok = False
        if missing:
            return Evaluation(None, sorted(set(missing)), premises)
        return Evaluation(ok, [], premises)

    def infer(self) -> List[Firing]:
        """Forward-chain once over all machine-checkable rules; return what fires."""
        fired: List[Firing] = []
        for r in self.rules:
            ev = self.evaluate(r)
            if ev.satisfied:
                prem_status = [TruthStatus[p["status"]] for p in ev.premises]
                f = Firing(r.name, r.action, r.confidence,
                           weakest(r.status, *prem_status), r.provenance, ev.premises)
                fired.append(f)
                self.audit.append({"event": "fire", "rule": r.name, "action": r.action,
                                   "status": f.status.name, "premises": ev.premises})
        return fired

    # -------------------------------------------------------------- querying
    def query_rule(self, question: str) -> Optional[Tuple[Rule, float]]:
        q = _tokens(question)
        best: Optional[Tuple[Rule, float]] = None
        for r in self.rules:
            kw = r.keywords()
            hit = len(q & kw)
            if not hit or not kw:
                continue
            score = (hit / len(kw)) * r.confidence
            if best is None or score > best[1]:
                best = (r, score)
        return best

    def query(self, question: str) -> str:
        hit = self.query_rule(question)
        if hit is None:
            return f"No rule matched for: {question}"
        rule, _ = hit
        ev = self.evaluate(rule)
        state = {True: "conditions currently met", False: "conditions currently NOT met",
                 None: ("conditions cannot be checked (missing facts: " + ", ".join(ev.missing) + ")")
                 if ev.missing else "keyword rule (no machine-checkable conditions)"}[ev.satisfied]
        return (f"[{rule.name}] {rule.action} (confidence={rule.confidence:.2f}, "
                f"status={rule.status.name}, provenance={rule.provenance}; {state})")

    def list_rules(self) -> List[str]:
        return [f"{r.name}: IF {r.condition} THEN {r.action} [{r.status.name}]" for r in self.rules]

    # ----------------------------------------------------------------- audit
    def export_audit(self, path: str) -> None:
        with open(path, "w", encoding="utf-8") as fh:
            for rec in self.audit:
                fh.write(json.dumps(rec, default=str) + "\n")
