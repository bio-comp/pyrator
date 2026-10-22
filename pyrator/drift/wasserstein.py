"""Wasserstein-1 (Earth Mover's Distance) implementation for drift monitoring."""

from __future__ import annotations

from typing import Literal

import numpy as np
import pandas as pd

from pyrator.drift._utils import (
    _to_pandas_frame,
    create_result_dict,
    iter_window_strata,
    validate_windows,
)
from pyrator.types import FrameLike


def w1(  # noqa: C901
    data: FrameLike,
    col: str,
    window_col: str = "window_id",
    *,
    baseline: str | None = None,
    weight_type: Literal["uniform"] = "uniform",
    stratify: list[str] | None = None,
) -> pd.DataFrame:
    """
    Calculate Wasserstein-1 distance (Earth Mover's Distance)
    for monitoring drift in ordered variables.

    Args:
        data: Input data frame with window_id column
        col: Column name to calculate Wasserstein distance for (should be numeric/ordered)
        window_col: Column name indicating time window (default: "window_id")
        baseline: Specific window to use as baseline. If None, uses first unique window.
        weight_type: Type of weights to use (default: "uniform"). Note: "ic" weighting
            was a placeholder and has been removed.
        stratify: List of column names to stratify by (default: None)

    Returns:
        DataFrame with Wasserstein distance values and metadata

    Formula:
        W₁(P, Q) = ∫ |F_P(x) - F_Q(x)| dx
        where F_P and F_Q are cumulative distribution functions

    For discrete distributions with sorted values x₁ < x₂ < ... < xₖ:
        W₁(P, Q) = Σ |(xᵢ₊₁ - xᵢ) * (F_P(xᵢ) - F_Q(xᵢ))|
    """
    df = _to_pandas_frame(data)

    # Validate required columns exist
    required_cols = [col, window_col]
    if stratify:
        required_cols.extend(stratify)

    for rc in required_cols:
        if rc not in df.columns:
            raise ValueError(f"Column '{rc}' not found in data")

    # Validate that column is numeric
    if not pd.api.types.is_numeric_dtype(df[col]):
        raise ValueError(f"Column '{col}' must be numeric for Wasserstein distance")

    baseline_window, current_windows = validate_windows(
        df, window_col=window_col, baseline=baseline, metric_name="Wasserstein distance"
    )

    results = []

    for current_window in current_windows:
        for b_df, c_df, stratum in iter_window_strata(
            df,
            window_col=window_col,
            baseline_window=baseline_window,
            current_window=current_window,
            stratify=stratify,
        ):
            b_series = b_df[col]
            c_series = c_df[col]
            if len(b_series) == 0 or len(c_series) == 0:
                continue

            distance = _calculate_wasserstein_distance(b_series, c_series, weight_type)

            result = create_result_dict(
                monitor_id=f"wasserstein_{col}",
                window_id=current_window,
                metric="wasserstein",
                value=distance,
                stratum=stratum,
            )
            results.append(result)

    return pd.DataFrame(results)


def _calculate_wasserstein_distance(
    baseline: pd.Series, current: pd.Series, weight_type: str = "uniform"
) -> float:
    """Calculate Wasserstein-1 distance between two univariate distributions."""
    # Combine values from both distributions to get all possible points
    all_values = np.sort(
        np.unique(np.concatenate([baseline.dropna().values, current.dropna().values]))
    )

    if len(all_values) < 2:
        return 0.0

    # Calculate cumulative distribution functions
    baseline_sorted = np.sort(baseline.dropna())
    current_sorted = np.sort(current.dropna())

    # For each interval between consecutive values, calculate the CDF difference
    wasserstein_distance = 0.0

    for i in range(len(all_values) - 1):
        x_left = all_values[i]
        x_right = all_values[i + 1]
        interval_width = x_right - x_left

        if interval_width <= 0:
            continue

        # Calculate CDF at midpoint of interval
        x_mid = (x_left + x_right) / 2.0

        # CDF for baseline: proportion of values <= x_mid
        baseline_cdf = (
            np.sum(baseline_sorted <= x_mid) / len(baseline_sorted)
            if len(baseline_sorted) > 0
            else 0.0
        )

        # CDF for current: proportion of values <= x_mid
        current_cdf = (
            np.sum(current_sorted <= x_mid) / len(current_sorted)
            if len(current_sorted) > 0
            else 0.0
        )

        # Calculate weight for this interval
        if weight_type == "ic":
            # Use information content as weight (simplified approximation)
            # In practice, this would use the actual IC values from an ontology
            # For now, we'll use a placeholder that gives higher weight to extremes
            weight = 1.0 + abs(x_mid - np.mean(all_values)) / (np.std(all_values) + 1e-10)
        else:  # uniform
            weight = 1.0

        # Add weighted contribution to Wasserstein distance
        wasserstein_distance += weight * interval_width * abs(baseline_cdf - current_cdf)

    return wasserstein_distance
