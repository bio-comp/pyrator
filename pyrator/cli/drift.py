"""Drift monitoring CLI commands."""

from __future__ import annotations

import json
from typing import Optional

import pandas as pd
import typer

from pyrator.drift import (
    CramerVMonitorConfig,
    JsdMonitorConfig,
    MmdMonitorConfig,
    Monitor,
    MonitorConfig,
    PsiMonitorConfig,
    WassersteinMonitorConfig,
)

app = typer.Typer(help="Drift monitoring commands.")


def load_config(config_dict: dict) -> MonitorConfig:
    """Load a MonitorConfig from a dictionary."""
    metric = config_dict.get("metric")
    if metric is None:
        raise ValueError("Config must include 'metric' field")

    common = {
        "name": config_dict.get("name", "unnamed"),
        "warn": config_dict.get("warn"),
        "crit": config_dict.get("crit"),
        "semantics": config_dict.get("semantics", "abs"),
        "window_col": config_dict.get("window_col", "window_id"),
    }

    match metric:
        case "psi":
            return PsiMonitorConfig(
                **common,
                col=config_dict.get("col"),
                bins=config_dict.get("bins", "quantile"),
                n_bins=config_dict.get("n_bins", 10),
                cutpoints=config_dict.get("cutpoints"),
                stratify=config_dict.get("stratify"),
                eps=config_dict.get("eps", 1e-6),
            )
        case "cramer_v":
            return CramerVMonitorConfig(
                **common,
                x=config_dict.get("x"),
                y=config_dict.get("y"),
                bias_correct=config_dict.get("bias_correct", True),
                stratify=config_dict.get("stratify"),
            )
        case "jsd":
            return JsdMonitorConfig(
                **common,
                dist_cols=config_dict.get("dist_cols"),
                groupby=config_dict.get("groupby"),
                sqrt=config_dict.get("sqrt", True),
                eps=config_dict.get("eps", 1e-6),
            )
        case "wasserstein":
            return WassersteinMonitorConfig(
                **common,
                col=config_dict.get("col"),
                weight_type=config_dict.get("weight_type", "uniform"),
                stratify=config_dict.get("stratify"),
            )
        case "mmd":
            return MmdMonitorConfig(
                **common,
                emb_cols=config_dict.get("emb_cols"),
                kernel=config_dict.get("kernel", "rbf"),
                sigma=config_dict.get("sigma", "median_heuristic"),
                n_perm=config_dict.get("n_perm", 1000),
                seed=config_dict.get("seed"),
                stratify=config_dict.get("stratify"),
            )
        case _:
            raise ValueError(f"Unknown metric: {metric}")


@app.command("run")
def run_drift(  # noqa: C901
    config: str = typer.Option(
        ..., "--config", "-c", help="Path to drift monitoring configuration file (YAML or JSON)"
    ),
    data: str = typer.Option(
        ..., "--data", "-d", help="Path to input data file (CSV, Parquet, etc.)"
    ),
    baseline: str = typer.Option(..., "--baseline", "-b", help="Baseline window identifier"),
    current: str = typer.Option(..., "--current", "-t", help="Current window identifier"),
    output: Optional[str] = typer.Option(
        None, "--output", "-o", help="Path to output results file"
    ),
    format: str = typer.Option("json", "--format", "-f", help="Output format (json, csv)"),
):
    """
    Run drift monitoring analysis.

    Example:
        pyrator drift run --config drift.yml --data data.parquet --baseline 2025Q2 --current 2025-10
    """
    # Load data
    if data.endswith(".csv"):
        df = pd.read_csv(data)
    elif data.endswith(".parquet"):
        df = pd.read_parquet(data)
    else:
        raise ValueError("Unsupported data format. Use CSV or Parquet.")

    # Add window column if not present (assuming data already filtered for baseline/current)
    if "window_id" not in df.columns:
        # This is a simplified approach - in practice, you'd want to filter the data
        # based on baseline and current parameters
        pass

    # Load configuration
    if config.endswith(".json"):
        with open(config) as f:
            config_dict = json.load(f)
    elif config.endswith((".yml", ".yaml")):
        try:
            import yaml

            with open(config) as f:
                config_dict = yaml.safe_load(f)
        except ImportError:
            raise ImportError(
                "PyYAML is required for YAML configuration files. Install with: pip install pyyaml"
            )
    else:
        raise ValueError("Unsupported config format. Use JSON or YAML.")

    # Create monitor from config
    monitor_config = load_config(config_dict)
    monitor = Monitor(monitor_config)

    # Execute monitoring
    try:
        result = monitor.execute(df)

        # Format output
        if format == "json":
            if isinstance(result, pd.DataFrame):
                output_data = result.to_json(orient="records", indent=2)
            else:  # tuple for MMD
                output_data = json.dumps({"statistic": result[0], "p_value": result[1]}, indent=2)
        elif format == "csv":
            if isinstance(result, pd.DataFrame):
                output_data = result.to_csv(index=False)
            else:
                raise ValueError("Cannot convert MMD results to CSV format")
        else:
            raise ValueError(f"Unsupported format: {format}")

        # Output results
        if output:
            with open(output, "w") as f:
                f.write(output_data)
            typer.echo(f"Results written to {output}")
        else:
            typer.echo(output_data)

    except Exception as e:
        typer.echo(f"Error running drift monitoring: {e}", err=True)
        raise typer.Exit(1)


@app.command("explain")
def explain_drift(
    monitor_name: str = typer.Argument(..., help="Name of the monitor to explain"),
    data: str = typer.Option(..., "--data", "-d", help="Path to input data file"),
    window: str = typer.Option(..., "--window", "-w", help="Window to explain"),
    config: str = typer.Option(
        ..., "--config", "-c", help="Path to drift monitoring configuration file"
    ),
    output: Optional[str] = typer.Option(None, "--output", "-o", help="Path to output explanation"),
):
    """
    Explain drift monitoring results for a specific monitor and window.

    Example:
        pyrator drift explain case_mix_age_psi --data data.parquet \\
            --window 2025-10 --config drift.yml
    """
    typer.echo(f"Explanation for monitor '{monitor_name}' in window '{window}'")
    typer.echo("This feature is not yet implemented.")


@app.command("board")
def drift_board(
    serve: str = typer.Option(":8080", "--serve", "-s", help="Address to serve dashboard on"),
    data: str = typer.Option(..., "--data", "-d", help="Path to input data file"),
    config: str = typer.Option(
        ..., "--config", "-c", help="Path to drift monitoring configuration file"
    ),
):
    """
    Launch a drift monitoring dashboard.

    Example:
        pyrator drift board --serve :8080 --data data.parquet --config drift.yml
    """
    typer.echo(f"Starting drift monitoring dashboard on {serve}")
    typer.echo("This feature is not yet implemented.")


if __name__ == "__main__":
    app()
