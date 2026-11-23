"""Minimal Typer CLI for the classical core."""

from __future__ import annotations

import typer
from rich import print as rprint

from .brain import ClassicalBrain

app = typer.Typer(help="BODY OF SECRETS – classical core CLI")


@app.command()
def status():
    """Show current brain status."""
    brain = ClassicalBrain()
    s = brain.status()
    rprint("[bold green]Classical Brain Status[/bold green]")
    for k, v in s.items():
        rprint(f"  {k}: {v}")


@app.command()
def ask(question: str = typer.Argument(..., help="Question for the expert / memory")):
    """Query the classical brain."""
    brain = ClassicalBrain()
    answer = brain.ask(question)
    rprint(f"[cyan]Q:[/cyan] {question}")
    rprint(f"[green]A:[/green] {answer}")


if __name__ == "__main__":
    app()