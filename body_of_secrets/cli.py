"""Command-line interface for the classical core (stdlib only)."""

from __future__ import annotations

import argparse
import sys
from typing import Optional, Sequence

from .brain import ClassicalBrain


def _brain(state: Optional[str]) -> ClassicalBrain:
    b = ClassicalBrain()
    if state:
        b.load(state)
    return b


def main(argv: Optional[Sequence[str]] = None) -> int:
    p = argparse.ArgumentParser(prog="body-of-secrets", description="BODY OF SECRETS – classical core CLI")
    p.add_argument("--state", help="directory to load/save brain state (memory + facts)")
    sub = p.add_subparsers(dest="cmd", required=True)
    sub.add_parser("status", help="show brain status")
    sub.add_parser("rules", help="list rules")
    sub.add_parser("infer", help="forward-chain over current facts and print what fires")
    a = sub.add_parser("ask", help="query the expert system / memory")
    a.add_argument("question")
    args = p.parse_args(argv)

    brain = _brain(args.state)
    if args.cmd == "status":
        print("Classical Brain Status")
        for k, v in brain.status().items():
            print(f"  {k}: {v}")
    elif args.cmd == "rules":
        print("\n".join(brain.expert.list_rules()))
    elif args.cmd == "infer":
        fired = brain.infer()
        print("\n".join(f.explain() for f in fired) if fired else "No rule fired.")
    elif args.cmd == "ask":
        print(f"Q: {args.question}")
        print(f"A: {brain.ask(args.question)}")
    if args.state:
        brain.save(args.state)
    return 0


app = main  # backward-compatible alias for the old Typer entry point

if __name__ == "__main__":
    sys.exit(main())
