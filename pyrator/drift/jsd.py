"""Jensen-Shannon Divergence (JSD) implementation for distribution drift monitoring."""

from __future__ import annotations

import numpy as np
import pandas as pd

from pyrator.drift._utils import _to_pandas_frame, create_result_dict
from pyrator.types import FrameLike


def jsd(
    data: FrameLike,
    dist_cols: list[str],
    window_col: str = "window_id",
    groupby: str | None = None,
    *,
    eps: float = 1e-6,
    sqrt: bool = True,
) -> pd.DataFrame:
    """
    Calculate Jensen-Shannon Divergence (JSD) for monitoring distribution drift.

    Args:
        data: Input data frame with window_id column
        dist_cols: List of column names representing distribution components
            (should sum to 1 per row)
        window_col: Column name indicating time window (default: "window_id")
        groupby: Column name to group by before calculating JSD (default: None)
        eps: Small value to avoid division by zero (default: 1e-6)
        sqrt: Whether to return square-rooted JSD (default: True)

    Returns:
        DataFrame with JSD values and metadata

    Formula:
        JSD(P || Q) = 0.5 * KL(P || M) + 0.5 * KL(Q || M)
        where M = 0.5*(P+Q)
        KL(P || Q) = Σ P(i) * log(P(i) / Q(i))

    With log₂ base, JSD ∈ [0,1]. When sqrt=True, returns √JSD ∈ [0,1].
    """
    df = _to_pandas_frame(data)

    # Validate required columns exist
    required_cols = dist_cols + [window_col]
    if groupby:
        required_cols.append(groupby)

    for rc in required_cols:
        if rc not in df.columns:
            raise ValueError(f"Column '{rc}' not found in data")

    # Validate that distribution columns sum to approximately 1 (within tolerance)
    row_sums = df[dist_cols].sum(axis=1)
    if not np.allclose(row_sums, 1.0, atol=1e-3):
        raise ValueError("Distribution columns must sum to 1 for each row")

    # Get unique windows
    windows = df[window_col].unique()
    if len(windows) < 2:
        raise ValueError("JSD requires at least 2 different windows")

    # Use first window as baseline, compare against all others
    baseline_window = windows[0]
    current_windows = windows[1:]

    results = []

    for current_window in current_windows:
        if groupby:
            baseline_groups = df[df[window_col] == baseline_window][groupby].unique()

            for group_value in baseline_groups:
                baseline_data = df[
                    (df[window_col] == baseline_window) & (df[groupby] == group_value)
                ]
                current_data = df[(df[window_col] == current_window) & (df[groupby] == group_value)]

                if len(baseline_data) == 0 or len(current_data) == 0:
                    continue

                baseline_dist = baseline_data[dist_cols].mean().values
                current_dist = current_data[dist_cols].mean().values

                jsd_value = _calculate_jsd(baseline_dist, current_dist, eps)
                if sqrt:
                    jsd_value = np.sqrt(jsd_value)

                result = create_result_dict(
                    monitor_id=f"jsd_{'_'.join(dist_cols)}",
                    window_id=current_window,
                    metric="jsd",
                    value=jsd_value,
                    stratum={groupby: group_value},
                )
                results.append(result)
        else:
            baseline_data = df[df[window_col] == baseline_window]
            current_data = df[df[window_col] == current_window]

            if len(baseline_data) == 0 or len(current_data) == 0:
                continue

            baseline_dist = baseline_data[dist_cols].mean().values
            current_dist = current_data[dist_cols].mean().values

            jsd_value = _calculate_jsd(baseline_dist, current_dist, eps)
            if sqrt:
                jsd_value = np.sqrt(jsd_value)

            result = create_result_dict(
                monitor_id=f"jsd_{'_'.join(dist_cols)}",
                window_id=current_window,
                metric="jsd",
                value=jsd_value,
            )
            results.append(result)

    return pd.DataFrame(results)


def _calculate_jsd(p: np.ndarray, q: np.ndarray, eps: float = 1e-6) -> float:
    """Calculate Jensen-Shannon Divergence between two distributions."""
    p = np.asarray(p) + eps
    q = np.asarray(q) + eps
    p = p / p.sum()
    q = q / q.sum()

    m = 0.5 * (p + q)

    kl_pm = np.sum(p * np.log(p / m))
    kl_qm = np.sum(q * np.log(q / m))

    jsd = 0.5 * kl_pm + 0.5 * kl_qm

    return float(jsd)
