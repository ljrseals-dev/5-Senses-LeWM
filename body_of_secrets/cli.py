"""Minimal Typer CLI for the classical core."""

from __future__ import annotations

import typer

from .brain import ClassicalBrain

app = typer.Typer(help="BODY OF SECRETS – classical core CLI")


@app.command()
def status() -> None:
    """Show current brain status."""
    brain = ClassicalBrain()
    for key, value in brain.status().items():
        typer.echo(f"{key}: {value}")


@app.command()
def ask(question: str = typer.Argument(..., help="Question for the expert / memory")) -> None:
    """Query the classical brain."""
    brain = ClassicalBrain()
    typer.echo(f"Q: {question}")
    typer.echo(f"A: {brain.ask(question)}")


if __name__ == "__main__":
    app()
