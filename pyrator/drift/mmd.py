"""Maximum Mean Discrepancy (MMD) implementation for embedding drift monitoring."""

from __future__ import annotations

import numpy as np
import pandas as pd
from typing import Literal

from pyrator.types import FrameLike


def _to_pandas_frame(data: FrameLike) -> pd.DataFrame:
    """Convert supported frame-like inputs to pandas DataFrame."""
    return data.to_pandas() if hasattr(data, "to_pandas") else data


def _rbf_kernel(x: np.ndarray, y: np.ndarray, sigma: float) -> np.ndarray:
    """Compute RBF kernel between two sets of vectors."""
    if x.ndim == 1:
        x = x.reshape(1, -1)
    if y.ndim == 1:
        y = y.reshape(1, -1)

    # Compute squared Euclidean distances
    xx = np.sum(x**2, axis=1, keepdims=True)
    yy = np.sum(y**2, axis=1, keepdims=True)
    xy = np.dot(x, y.T)
    dist_sq = xx + yy.T - 2 * xy

    # Compute RBF kernel
    return np.exp(-dist_sq / (2 * sigma**2))


def mmd(
    data: FrameLike,
    emb_cols: list[str],
    window_col: str = "window_id",
    *,
    kernel: Literal["rbf"] = "rbf",
    sigma: Literal["median_heuristic"] | float = "median_heuristic",
    n_perm: int = 1000,
    seed: int | None = None,
    stratify: list[str] | None = None,
) -> tuple[float, float]:
    """
    Calculate Maximum Mean Discrepancy (MMD) for monitoring embedding drift.

    Args:
        data: Input data frame with window_id column
        emb_cols: List of column names representing embedding dimensions
        window_col: Column name indicating time window (default: "window_id")
        kernel: Kernel type to use (default: "rbf")
        sigma: Kernel bandwidth ("median_heuristic" or float value) (default: "median_heuristic")
        n_perm: Number of permutations for permutation test (default: 1000)
        seed: Random seed for reproducibility (default: None)
        stratify: List of column names to stratify by (default: None)

    Returns:
        Tuple of (MMD statistic, p-value)

    Formula:
        MMD²(P, Q) = ||μ_P - μ_Q||²_H
        where μ_P and μ_Q are mean embeddings in reproducing kernel Hilbert space H
    """
    df = _to_pandas_frame(data)

    # Validate required columns exist
    required_cols = emb_cols + [window_col]
    if stratify:
        required_cols.extend(stratify)

    for rc in required_cols:
        if rc not in df.columns:
            raise ValueError(f"Column '{rc}' not found in data")

    # Get unique windows
    windows = df[window_col].unique()
    if len(windows) < 2:
        raise ValueError("MMD requires at least 2 different windows")

    # Use first window as baseline, compare against all others
    baseline_window = windows[0]
    current_windows = windows[1:]

    # Set random seed for reproducibility
    if seed is not None:
        np.random.seed(seed)

    results = []

    for current_window in current_windows:
        # Handle stratification
        if stratify:
            # Create stratum combinations
            baseline_strata = df[df[window_col] == baseline_window][stratify].drop_duplicates()
            current_strata = df[df[window_col] == current_window][stratify].drop_duplicates()

            # For each stratum combination, calculate MMD
            for _, stratum_row in baseline_strata.iterrows():
                stratum_cond = np.ones(len(df), dtype=bool)
                for stratum_col in stratify:
                    stratum_cond &= df[stratum_col] == stratum_row[stratum_col]

                stratum_baseline = df[stratum_cond & (df[window_col] == baseline_window)]
                stratum_current = df[stratum_cond & (df[window_col] == current_window)]

                # Skip if no data in either window
                if len(stratum_baseline) == 0 or len(stratum_current) == 0:
                    continue

                # Extract embeddings
                baseline_emb = stratum_baseline[emb_cols].values
                current_emb = stratum_current[emb_cols].values

                # Skip if not enough data
                if len(baseline_emb) < 2 or len(current_emb) < 2:
                    continue

                # Calculate MMD
                mmd_stat, p_value = _calculate_mmd(baseline_emb, current_emb, kernel, sigma, n_perm)

                # Create result entry
                result = {
                    "monitor_id": f"mmd_{'_'.join(emb_cols)}",
                    "window_id": current_window,
                    "stratum": {k: v for k, v in stratum_row.items()},
                    "metric": "mmd",
                    "value": float(mmd_stat),
                    "delta_from_baseline": float(
                        mmd_stat
                    ),  # For first comparison, delta is the value itself
                    "ci_low": float(mmd_stat),  # Placeholder
                    "ci_high": float(mmd_stat),  # Placeholder
                    "p_value": float(p_value),
                    "threshold_level": "none",  # Would be determined by comparing to thresholds
                }
                results.append(result)
        else:
            # No stratification - calculate overall MMD
            baseline_data = df[df[window_col] == baseline_window]
            current_data = df[df[window_col] == current_window]

            # Skip if no data in either window
            if len(baseline_data) == 0 or len(current_data) == 0:
                continue

            # Extract embeddings
            baseline_emb = baseline_data[emb_cols].values
            current_emb = current_data[emb_cols].values

            # Skip if not enough data
            if len(baseline_emb) < 2 or len(current_emb) < 2:
                continue

            # Calculate MMD
            mmd_stat, p_value = _calculate_mmd(baseline_emb, current_emb, kernel, sigma, n_perm)

            # Create result entry
            result = {
                "monitor_id": f"mmd_{'_'.join(emb_cols)}",
                "window_id": current_window,
                "stratum": {},  # No stratification
                "metric": "mmd",
                "value": float(mmd_stat),
                "delta_from_baseline": float(
                    mmd_stat
                ),  # For first comparison, delta is the value itself
                "ci_low": float(mmd_stat),  # Placeholder
                "ci_high": float(mmd_stat),  # Placeholder
                "p_value": float(p_value),
                "threshold_level": "none",  # Would be determined by comparing to thresholds
            }
            results.append(result)

    # Return first result for backward compatibility with existing API expectations
    # In a full implementation, we would return all results
    if results:
        return results[0]["value"], results[0]["p_value"]
    else:
        return 0.0, 1.0


def _calculate_mmd(
    x: np.ndarray,
    y: np.ndarray,
    kernel: Literal["rbf"],
    sigma: Literal["median_heuristic"] | float,
    n_perm: int,
) -> tuple[float, float]:
    """Calculate MMD statistic and p-value using permutation test."""
    # Combine samples
    combined = np.vstack([x, y])
    n_x = len(x)
    n_y = len(y)

    # Calculate kernel bandwidth if using median heuristic
    if sigma == "median_heuristic":
        # Compute pairwise distances
        if len(combined) > 1:
            from scipy.spatial.distance import pdist

            distances = pdist(combined, metric="euclidean")
            sigma_val = np.median(distances)
            # Avoid zero bandwidth
            sigma_val = max(sigma_val, 1e-10)
        else:
            sigma_val = 1.0
    else:
        sigma_val = float(sigma)

    # Compute kernel matrix
    K = _rbf_kernel(combined, combined, sigma_val)

    # Compute MMD statistic
    # K_XX: kernel matrix for x vs x
    # K_YY: kernel matrix for y vs y
    # K_XY: kernel matrix for x vs y
    K_XX = K[:n_x, :n_x]
    K_YY = K[n_x:, n_x:]
    K_XY = K[:n_x, n_x:]

    # MMD² = (1/(n_x*(n_x-1))) * sum(K_XX - diag(K_XX))
    #      + (1/(n_y*(n_y-1))) * sum(K_YY - diag(K_YY))
    #      - (2/(n_x*n_y)) * sum(K_XY)

    mmd_sq = (
        ((np.sum(K_XX) - np.trace(K_XX)) / (n_x * (n_x - 1)) if n_x > 1 else 0)
        + ((np.sum(K_YY) - np.trace(K_YY)) / (n_y * (n_y - 1)) if n_y > 1 else 0)
        - (2 * np.sum(K_XY) / (n_x * n_y) if n_x > 0 and n_y > 0 else 0)
    )

    # Ensure non-negative
    mmd_sq = max(mmd_sq, 0)
    mmd_stat = np.sqrt(mmd_sq)

    # Permutation test
    if n_perm > 0:
        # Combine labels for permutation
        labels = np.array([0] * n_x + [1] * n_y)
        perm_stats = []

        for _ in range(n_perm):
            # Shuffle labels
            shuffled_labels = np.random.permutation(labels)

            # Split according to shuffled labels
            x_shuffled = combined[shuffled_labels == 0]
            y_shuffled = combined[shuffled_labels == 1]

            # Skip if not enough data in either group
            if len(x_shuffled) < 2 or len(y_shuffled) < 2:
                perm_stats.append(0.0)
                continue

            # Recompute kernel matrix for shuffled data
            combined_shuffled = (
                np.vstack([x_shuffled, y_shuffled])
                if len(x_shuffled) > 0 and len(y_shuffled) > 0
                else combined
            )
            if len(combined_shuffled) > 1:
                K_shuffled = _rbf_kernel(combined_shuffled, combined_shuffled, sigma_val)

                # Split kernel matrix
                n_x_shuffled = len(x_shuffled)
                n_y_shuffled = len(y_shuffled)

                if n_x_shuffled > 0 and n_y_shuffled > 0:
                    K_XX_shuffled = K_shuffled[:n_x_shuffled, :n_x_shuffled]
                    K_YY_shuffled = K_shuffled[n_x_shuffled:, n_x_shuffled:]
                    K_XY_shuffled = K_shuffled[:n_x_shuffled, n_x_shuffled:]

                    # Compute MMD statistic for shuffled data
                    mmd_sq_shuffled = (
                        (
                            (np.sum(K_XX_shuffled) - np.trace(K_XX_shuffled))
                            / (n_x_shuffled * (n_x_shuffled - 1))
                            if n_x_shuffled > 1
                            else 0
                        )
                        + (
                            (np.sum(K_YY_shuffled) - np.trace(K_YY_shuffled))
                            / (n_y_shuffled * (n_y_shuffled - 1))
                            if n_y_shuffled > 1
                            else 0
                        )
                        - (
                            2 * np.sum(K_XY_shuffled) / (n_x_shuffled * n_y_shuffled)
                            if n_x_shuffled > 0 and n_y_shuffled > 0
                            else 0
                        )
                    )

                    mmd_sq_shuffled = max(mmd_sq_shuffled, 0)
                    perm_stats.append(np.sqrt(mmd_sq_shuffled))
                else:
                    perm_stats.append(0.0)
            else:
                perm_stats.append(0.0)

        # Calculate p-value
        perm_stats = np.array(perm_stats)
        p_value = np.mean(perm_stats >= mmd_stat)
    else:
        p_value = 0.0

    return mmd_stat, p_value
