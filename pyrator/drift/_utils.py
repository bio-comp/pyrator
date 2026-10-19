"""Shared utilities for drift monitoring metrics."""

from __future__ import annotations

from typing import Any

import narwhals.stable.v2 as nw
import pandas as pd

from pyrator.types import FrameLike


def _to_pandas_frame(data: FrameLike) -> pd.DataFrame:
    """Convert supported frame-like inputs to pandas DataFrame."""
    if isinstance(data, (nw.DataFrame, nw.LazyFrame)):
        if isinstance(data, nw.LazyFrame):
            data = data.collect()
        native = nw.to_native(data)
    else:
        native = data

    if hasattr(native, "collect"):
        native = native.collect()

    if isinstance(native, pd.DataFrame):
        return native
    if hasattr(native, "to_pandas"):
        return native.to_pandas()  # type: ignore[no-any-return]
    if hasattr(native, "df"):
        return native.df()  # type: ignore[no-any-return]
    return pd.DataFrame(native)


def create_result_dict(
    monitor_id: str,
    window_id: str,
    metric: str,
    value: float,
    delta_from_baseline: float | None = None,
    stratum: dict[str, Any] | None = None,
    ci_low: float | None = None,
    ci_high: float | None = None,
    p_value: float | None = None,
    threshold_level: str = "none",
) -> dict[str, Any]:
    """
    Create a standardized result dictionary for drift metrics.

    Args:
        monitor_id: Unique identifier for the monitor
        window_id: The window being compared against baseline
        metric: The metric name (e.g., "psi", "cramer_v")
        value: The computed metric value
        delta_from_baseline: Change from baseline (defaults to value)
        stratum: Stratification columns and values (empty dict if none)
        ci_low: Lower confidence interval bound
        ci_high: Upper confidence interval bound
        p_value: P-value from permutation/bootstrap test
        threshold_level: Threshold classification (default: "none")

    Returns:
        Dictionary with standardized result structure
    """
    if delta_from_baseline is None:
        delta_from_baseline = value
    if ci_low is None:
        ci_low = value
    if ci_high is None:
        ci_high = value
    if p_value is None:
        p_value = 0.0
    if stratum is None:
        stratum = {}

    return {
        "monitor_id": monitor_id,
        "window_id": window_id,
        "stratum": stratum,
        "metric": metric,
        "value": float(value),
        "delta_from_baseline": float(delta_from_baseline),
        "ci_low": float(ci_low),
        "ci_high": float(ci_high),
        "p_value": float(p_value),
        "threshold_level": threshold_level,
    }
