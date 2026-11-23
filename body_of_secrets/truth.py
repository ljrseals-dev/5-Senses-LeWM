"""Truth-status vocabulary shared with the parent 5-Senses laboratory."""

from __future__ import annotations

from enum import IntEnum


class TruthStatus(IntEnum):
    """Ordered epistemic status: higher value = stronger warrant."""

    UNKNOWN = 0
    SPECULATIVE = 1
    PLAUSIBLE = 2
    CONSENSUS = 3
    OFFICIAL = 4

    @classmethod
    def parse(cls, value: "str | int | TruthStatus") -> "TruthStatus":
        if isinstance(value, cls):
            return value
        if isinstance(value, str):
            try:
                return cls[value.strip().upper()]
            except KeyError as exc:
                raise ValueError(f"unknown truth status: {value!r}") from exc
        return cls(int(value))


def weakest(*statuses: TruthStatus) -> TruthStatus:
    """A conclusion is never better warranted than its weakest premise."""
    return min(statuses) if statuses else TruthStatus.UNKNOWN
