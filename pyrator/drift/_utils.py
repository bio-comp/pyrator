"""Shared utilities for drift monitoring metrics."""

from __future__ import annotations

from collections.abc import Iterator
from typing import Any

import narwhals.stable.v2 as nw
import numpy as np
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


def validate_windows(
    df: pd.DataFrame,
    window_col: str,
    baseline: str | None = None,
    metric_name: str = "Drift metric",
) -> tuple[str, list[str]]:
    """Validate window column and determine baseline and target windows."""
    if window_col not in df.columns:
        raise ValueError(f"Column '{window_col}' not found in data")

    # Preserve order of first appearance while deduplicating string representations
    windows = list(dict.fromkeys(str(w) for w in df[window_col].unique()))
    if len(windows) < 2:
        raise ValueError(f"{metric_name} requires at least 2 different windows")

    if baseline is not None:
        baseline_str = str(baseline)
        if baseline_str not in windows:
            raise ValueError(f"Baseline window '{baseline}' not found in data")
        baseline_window = baseline_str
    else:
        baseline_window = windows[0]

    current_windows = [w for w in windows if w != baseline_window]
    return baseline_window, current_windows


def iter_window_strata(
    df: pd.DataFrame,
    window_col: str,
    baseline_window: Any,
    current_window: Any,
    stratify: list[str] | None = None,
) -> Iterator[tuple[pd.DataFrame, pd.DataFrame, dict[str, Any]]]:
    """Iterate over pairs of (baseline, current) subsets for each stratum."""
    window_series = df[window_col].astype(str)
    baseline_mask = window_series == str(baseline_window)
    current_mask = window_series == str(current_window)

    if stratify:
        baseline_strata = df[baseline_mask][stratify].drop_duplicates()

        for _, stratum_row in baseline_strata.iterrows():
            stratum_cond = np.ones(len(df), dtype=bool)
            for stratum_col in stratify:
                val = stratum_row[stratum_col]
                if pd.isna(val):
                    stratum_cond &= df[stratum_col].isna()
                else:
                    stratum_cond &= df[stratum_col] == val

            stratum_baseline = df[stratum_cond & baseline_mask]
            stratum_current = df[stratum_cond & current_mask]

            if len(stratum_baseline) == 0 or len(stratum_current) == 0:
                continue

            stratum_dict = {str(k): v for k, v in stratum_row.items()}
            yield stratum_baseline, stratum_current, stratum_dict
    else:
        baseline_data = df[baseline_mask]
        current_data = df[current_mask]

        if len(baseline_data) > 0 and len(current_data) > 0:
            yield baseline_data, current_data, {}


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
