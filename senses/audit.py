"""Append-only JSONL audit trail (plans, surprise, physics reports)."""

from __future__ import annotations

import json
import os
from datetime import datetime, timezone
from typing import Any, Dict


def _default(o: Any):
    if hasattr(o, "tolist"):
        return o.tolist()
    if hasattr(o, "item"):
        return o.item()
    return str(o)


def append_jsonl(path: str, record: Dict[str, Any]) -> None:
    d = os.path.dirname(path)
    if d:
        os.makedirs(d, exist_ok=True)
    rec = {"ts": datetime.now(timezone.utc).isoformat(), **record}
    with open(path, "a", encoding="utf-8") as fh:
        fh.write(json.dumps(rec, default=_default) + "\n")
