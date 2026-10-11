"""CLI module for pyrator."""

from __future__ import annotations

import typer

from pyrator.cli.drift import drift_app

app = typer.Typer(help="pyrator: Interrater reliability, agreement, and drift monitoring.")
app.add_typer(drift_app, name="drift")

__all__ = ["app", "drift_app"]
