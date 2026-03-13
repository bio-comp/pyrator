"""Cramér's V implementation for association drift monitoring."""

from __future__ import annotations

import numpy as np
import pandas as pd
from scipy.stats import chi2_contingency
from typing import Literal

from pyrator.types import FrameLike


def _to_pandas_frame(data: FrameLike) -> pd.DataFrame:
    """Convert supported frame-like inputs to pandas DataFrame."""
    return data.to_pandas() if hasattr(data, "to_pandas") else data


def cramer_v(
    data: FrameLike,
    x: str,
    y: str,
    window_col: str = "window_id",
    *,
    bias_correct: bool = True,
    stratify: list[str] | None = None,
) -> pd.DataFrame:
    """
    Calculate Cramér's V for monitoring association drift between two categorical variables.
    
    Args:
        data: Input data frame with window_id column
        x: First column name (categorical)
        y: Second column name (categorical)
        window_col: Column name indicating time window (default: "window_id")
        bias_correct: Whether to apply bias correction (default: True)
        stratify: List of column names to stratify by (default: None)
        
    Returns:
        DataFrame with Cramér's V values and metadata
        
    Formula:
        V = sqrt(χ² / (n * min(r-1, c-1)))
        where χ² is chi-square statistic, n is sample size, r and c are table dimensions
        
    Bias correction (Bergsma, 2013):
        V* = sqrt(χ² / (n * min(r-1, c-1))) * sqrt((min(r-1, c-1)) / (min(r-1, c-1) - 1))
    """
    df = _to_pandas_frame(data)
    
    # Validate required columns exist
    required_cols = [x, y, window_col]
    if stratify:
        required_cols.extend(stratify)
    
    for rc in required_cols:
        if rc not in df.columns:
            raise ValueError(f"Column '{rc}' not found in data")
    
    # Get unique windows
    windows = df[window_col].unique()
    if len(windows) < 2:
        raise ValueError("Cramér's V requires at least 2 different windows")
    
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
            
            # For each stratum combination, calculate Cramér's V
            for _, stratum_row in baseline_strata.iterrows():
                stratum_cond = np.ones(len(df), dtype=bool)
                for stratum_col in stratify:
                    stratum_cond &= (df[stratum_col] == stratum_row[stratum_col])
                
                stratum_baseline = df[stratum_cond & (df[window_col] == baseline_window)]
                stratum_current = df[stratum_cond & (df[window_col] == current_window)]
                
                # Skip if no data in either window
                if len(stratum_baseline) == 0 or len(stratum_current) == 0:
                    continue
                
                # Calculate Cramér's V for baseline and current
                baseline_v, baseline_chi2, baseline_n, baseline_r, baseline_c = _calculate_cramers_v(
                    stratum_baseline[x], stratum_baseline[y], bias_correct
                )
                current_v, current_chi2, current_n, current_r, current_c = _calculate_cramers_v(
                    stratum_current[x], stratum_current[y], bias_correct
                )
                
                # Calculate delta (current - baseline)
                delta_v = current_v - baseline_v
                
                # Create result entry
                result = {
                    "monitor_id": f"cramer_v_{x}_{y}",
                    "window_id": current_window,
                    "stratum": {k: v for k, v in stratum_row.items()},
                    "metric": "cramer_v",
                    "value": float(current_v),
                    "delta_from_baseline": float(delta_v),
                    "ci_low": float(delta_v),  # Placeholder - would be calculated with bootstrap
                    "ci_high": float(delta_v),  # Placeholder - would be calculated with bootstrap
                    "p_value": 0.0,  # Placeholder - would be calculated with permutation test
                    "threshold_level": "none",  # Would be determined by comparing to thresholds
                }
                results.append(result)
        else:
            # No stratification - calculate overall Cramér's V
            baseline_data = df[df[window_col] == baseline_window]
            current_data = df[df[window_col] == current_window]
            
            # Calculate Cramér's V for baseline and current
            baseline_v, baseline_chi2, baseline_n, baseline_r, baseline_c = _calculate_cramers_v(
                baseline_data[x], baseline_data[y], bias_correct
            )
            current_v, current_chi2, current_n, current_r, current_c = _calculate_cramers_v(
                current_data[x], current_data[y], bias_correct
            )
            
            # Calculate delta (current - baseline)
            delta_v = current_v - baseline_v
            
            # Create result entry
            result = {
                "monitor_id": f"cramer_v_{x}_{y}",
                "window_id": current_window,
                "stratum": {},  # No stratification
                "metric": "cramer_v",
                "value": float(current_v),
                "delta_from_baseline": float(delta_v),
                "ci_low": float(delta_v),  # Placeholder - would be calculated with bootstrap
                "ci_high": float(delta_v),  # Placeholder - would be calculated with bootstrap
                "p_value": 0.0,  # Placeholder - would be calculated with permutation test
                "threshold_level": "none",  # Would be determined by comparing to thresholds
            }
            results.append(result)
    
    return pd.DataFrame(results)


def _calculate_cramers_v(x_series: pd.Series, y_series: pd.Series, bias_correct: bool = True):
    """Calculate Cramér's V statistic for two categorical series."""
    # Create contingency table
    contingency_table = pd.crosstab(x_series, y_series)
    
    # Calculate chi-square statistic
    chi2, p, dof, expected = chi2_contingency(contingency_table, correction=False)
    
    # Get dimensions
    n = contingency_table.sum().sum()  # Total sample size
    r, c = contingency_table.shape  # Rows and columns
    
    # Calculate Cramér's V
    if n == 0 or min(r, c) <= 1:
        v = 0.0
    else:
        v = np.sqrt(chi2 / (n * min(r-1, c-1)))
    
    # Apply bias correction if requested
    if bias_correct and min(r, c) > 2:
        v = v * np.sqrt((min(r-1, c-1)) / (min(r-1, c-1) - 1))
    
    return v, chi2, n, r, c
