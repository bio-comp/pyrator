"""Wasserstein-1 (Earth Mover's Distance) implementation for drift monitoring."""

from __future__ import annotations

import numpy as np
import pandas as pd
from typing import Literal

from pyrator.types import FrameLike


def _to_pandas_frame(data: FrameLike) -> pd.DataFrame:
    """Convert supported frame-like inputs to pandas DataFrame."""
    return data.to_pandas() if hasattr(data, "to_pandas") else data


def w1(
    data: FrameLike,
    col: str,
    window_col: str = "window_id",
    *,
    weight_type: Literal["uniform", "ic"] = "uniform",
    stratify: list[str] | None = None,
) -> pd.DataFrame:
    """
    Calculate Wasserstein-1 distance (Earth Mover's Distance) for monitoring drift in ordered variables.

    Args:
        data: Input data frame with window_id column
        col: Column name to calculate Wasserstein distance for (should be numeric/ordered)
        window_col: Column name indicating time window (default: "window_id")
        weight_type: Type of weights to use ("uniform" or "ic") (default: "uniform")
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

    # Get unique windows
    windows = df[window_col].unique()
    if len(windows) < 2:
        raise ValueError("Wasserstein distance requires at least 2 different windows")

    # Use first window as baseline, compare against all others
    baseline_window = windows[0]
    current_windows = windows[1:]

    results = []

    for current_window in current_windows:
        # Handle stratification
        if stratify:
            # Create stratum combinations
            baseline_strata = df[df[window_col] == baseline_window][stratify].drop_duplicates()
            current_strata = df[df[window_col] == current_window][stratify].drop_duplicates()

            # For each stratum combination, calculate Wasserstein distance
            for _, stratum_row in baseline_strata.iterrows():
                stratum_cond = np.ones(len(df), dtype=bool)
                for stratum_col in stratify:
                    stratum_cond &= df[stratum_col] == stratum_row[stratum_col]

                stratum_baseline = df[stratum_cond & (df[window_col] == baseline_window)][col]
                stratum_current = df[stratum_cond & (df[window_col] == current_window)][col]

                # Skip if no data in either window
                if len(stratum_baseline) == 0 or len(stratum_current) == 0:
                    continue

                # Calculate Wasserstein distance for baseline and current
                distance = _calculate_wasserstein_distance(
                    stratum_baseline, stratum_current, weight_type
                )

                # Create result entry
                result = {
                    "monitor_id": f"wasserstein_{col}",
                    "window_id": current_window,
                    "stratum": {k: v for k, v in stratum_row.items()},
                    "metric": "wasserstein",
                    "value": float(distance),
                    "delta_from_baseline": float(
                        distance
                    ),  # For first comparison, delta is the value itself
                    "ci_low": float(distance),  # Placeholder - would be calculated with bootstrap
                    "ci_high": float(distance),  # Placeholder - would be calculated with bootstrap
                    "p_value": 0.0,  # Placeholder - would be calculated with permutation test
                    "threshold_level": "none",  # Would be determined by comparing to thresholds
                }
                results.append(result)
        else:
            # No stratification - calculate overall Wasserstein distance
            baseline_data = df[df[window_col] == baseline_window][col]
            current_data = df[df[window_col] == current_window][col]

            # Skip if no data in either window
            if len(baseline_data) == 0 or len(current_data) == 0:
                continue

            # Calculate Wasserstein distance
            distance = _calculate_wasserstein_distance(baseline_data, current_data, weight_type)

            # Create result entry
            result = {
                "monitor_id": f"wasserstein_{col}",
                "window_id": current_window,
                "stratum": {},  # No stratification
                "metric": "wasserstein",
                "value": float(distance),
                "delta_from_baseline": float(
                    distance
                ),  # For first comparison, delta is the value itself
                "ci_low": float(distance),  # Placeholder - would be calculated with bootstrap
                "ci_high": float(distance),  # Placeholder - would be calculated with bootstrap
                "p_value": 0.0,  # Placeholder - would be calculated with permutation test
                "threshold_level": "none",  # Would be determined by comparing to thresholds
            }
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
