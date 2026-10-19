"""CLI module for pyrator."""

from __future__ import annotations

import typer

from pyrator.cli.calculate import calculate_app
from pyrator.cli.drift import drift_app
from pyrator.mcp import run_mcp_stdio_server

app = typer.Typer(help="pyrator: Interrater reliability, agreement, and drift monitoring.")
app.add_typer(drift_app, name="drift")
app.add_typer(calculate_app, name="calculate")


@app.command("mcp")
def start_mcp() -> None:
    """Start the stdio Model Context Protocol (MCP) server for AI coding agents."""
    run_mcp_stdio_server()


__all__ = ["app", "calculate_app", "drift_app", "start_mcp"]
