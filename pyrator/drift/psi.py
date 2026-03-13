"""Population Stability Index (PSI) implementation."""

from __future__ import annotations

import numpy as np
import pandas as pd
from typing import Literal

from pyrator.types import FrameLike


def _to_pandas_frame(data: FrameLike) -> pd.DataFrame:
    """Convert supported frame-like inputs to pandas DataFrame."""
    return data.to_pandas() if hasattr(data, "to_pandas") else data


def _get_bin_edges(
    data: pd.Series,
    bin_type: Literal["quantile", "fd", "scott", "rice", "sturges", "sqrt"] = "quantile",
    n_bins: int = 10,
    cutpoints: list[float] | None = None,
) -> np.ndarray:
    """Calculate bin edges for binning numeric data."""
    if cutpoints is not None:
        return np.array([-np.inf] + sorted(cutpoints) + [np.inf])
    
    if bin_type == "quantile":
        # Quantile bins (percentiles)
        percentiles = np.linspace(0, 100, n_bins + 1)
        return np.percentile(data.dropna(), percentiles)
    elif bin_type == "fd":  # Freedman-Diaconis rule
        q75, q25 = np.percentile(data.dropna(), [75, 25])
        iqr = q75 - q25
        bin_width = 2 * iqr * (len(data.dropna()) ** (-1/3))
        bin_width = max(bin_width, 1e-10)  # Avoid zero width
        data_range = data.dropna().max() - data.dropna().min()
        n_bins_calc = max(1, int(np.ceil(data_range / bin_width)))
        return np.linspace(data.dropna().min(), data.dropna().max(), n_bins_calc + 1)
    elif bin_type == "scott":  # Scott's rule
        sigma = data.dropna().std()
        bin_width = 3.5 * sigma * (len(data.dropna()) ** (-1/3))
        bin_width = max(bin_width, 1e-10)
        data_range = data.dropna().max() - data.dropna().min()
        n_bins_calc = max(1, int(np.ceil(data_range / bin_width)))
        return np.linspace(data.dropna().min(), data.dropna().max(), n_bins_calc + 1)
    elif bin_type == "rice":  # Rice rule
        n_bins_calc = max(1, int(np.ceil(2 * (len(data.dropna()) ** (1/3)))))
        return np.linspace(data.dropna().min(), data.dropna().max(), n_bins_calc + 1)
    elif bin_type == "sturges":  # Sturges' rule
        n_bins_calc = max(1, int(np.ceil(np.log2(len(data.dropna())) + 1)))
        return np.linspace(data.dropna().min(), data.dropna().max(), n_bins_calc + 1)
    elif bin_type == "sqrt":  # Square root rule
        n_bins_calc = max(1, int(np.ceil(np.sqrt(len(data.dropna())))))
        return np.linspace(data.dropna().min(), data.dropna().max(), n_bins_calc + 1)
    else:
        raise ValueError(f"Unsupported bin_type: {bin_type}")


def _bin_data(
    data: pd.Series,
    bin_edges: np.ndarray,
    categorical: bool = False,
) -> pd.Series:
    """Bin data according to bin edges."""
    if categorical:
        # For categorical data, just return the data as-is (already binned)
        return data
    
    # For numeric data, bin according to edges
    # Add small epsilon to rightmost edge to include max value
    bin_edges = bin_edges.copy()
    bin_edges[-1] += 1e-10
    
    return pd.cut(data, bins=bin_edges, labels=False, include_lowest=True)


def psi(
    data: FrameLike,
    col: str,
    window_col: str = "window_id",
    *,
    bins: Literal["quantile", "fd", "scott", "rice", "sturges", "sqrt"] = "quantile",
    n_bins: int = 10,
    cutpoints: list[float] | None = None,
    stratify: list[str] | None = None,
    eps: float = 1e-6,
) -> pd.DataFrame:
    """
    Calculate Population Stability Index (PSI) for monitoring distribution drift.
    
    Args:
        data: Input data frame with window_id column
        col: Column name to calculate PSI for
        window_col: Column name indicating time window (default: "window_id")
        bins: Binning strategy for numeric data (default: "quantile")
        n_bins: Number of bins for quantile-based binning (default: 10)
        cutpoints: Custom cutpoints for binning (overrides bins and n_bins if provided)
        stratify: List of column names to stratify by (default: None)
        eps: Small value to avoid division by zero (default: 1e-6)
        
    Returns:
        DataFrame with PSI values and metadata
        
    Formula:
        PSI = Σ (p_i - q_i) * ln(p_i / q_i)
        where p_i is baseline share and q_i is current share in bin i
    """
    df = _to_pandas_frame(data)
    
    # Validate required columns exist
    required_cols = [col, window_col]
    if stratify:
        required_cols.extend(stratify)
    
    for rc in required_cols:
        if rc not in df.columns:
            raise ValueError(f"Column '{rc}' not found in data")
    
    # Get unique windows
    windows = df[window_col].unique()
    if len(windows) < 2:
        raise ValueError("PSI requires at least 2 different windows")
    
    # Use first window as baseline, compare against all others
    baseline_window = windows[0]
    current_windows = windows[1:]
    
    results = []
    
    for current_window in current_windows:
        # Filter data for baseline and current windows
        baseline_data = df[df[window_col] == baseline_window][col]
        current_data = df[df[window_col] == current_window][col]
        
        # Handle stratification
        if stratify:
            # Create stratum combinations
            baseline_strata = df[df[window_col] == baseline_window][stratify].drop_duplicates()
            current_strata = df[df[window_col] == current_window][stratify].drop_duplicates()
            
            # For each stratum combination, calculate PSI
            for _, stratum_row in baseline_strata.iterrows():
                stratum_cond = np.ones(len(df), dtype=bool)
                for stratum_col in stratify:
                    stratum_cond &= (df[stratum_col] == stratum_row[stratum_col])
                
                stratum_baseline = df[stratum_cond & (df[window_col] == baseline_window)][col]
                stratum_current = df[stratum_cond & (df[window_col] == current_window)][col]
                
                # Skip if no data in either window
                if len(stratum_baseline) == 0 or len(stratum_current) == 0:
                    continue
                
                # Determine if column is numeric or categorical
                is_numeric = pd.api.types.is_numeric_dtype(stratum_baseline)
                
                # Get bin edges from baseline data
                if is_numeric:
                    bin_edges = _get_bin_edges(
                        stratum_baseline, 
                        bin_type=bins, 
                        n_bins=n_bins, 
                        cutpoints=cutpoints
                    )
                else:
                    # For categorical data, use unique values from baseline
                    bin_edges = np.array(sorted(stratum_baseline.dropna().unique()))
                
                # Bin the data
                try:
                    if is_numeric:
                        baseline_binned = _bin_data(stratum_baseline, bin_edges, categorical=False)
                        current_binned = _bin_data(stratum_current, bin_edges, categorical=False)
                    else:
                        baseline_binned = _bin_data(stratum_baseline, bin_edges, categorical=True)
                        current_binned = _bin_data(stratum_current, bin_edges, categorical=True)
                except Exception:
                    # Fallback: treat as categorical if binning fails
                    baseline_binned = _bin_data(stratum_baseline, bin_edges, categorical=True)
                    current_binned = _bin_data(stratum_current, bin_edges, categorical=True)
                
                # Calculate PSI for this stratum
                max_bin = max(
                    baseline_binned.max() if not baseline_binned.empty else 0,
                    current_binned.max() if not current_binned.empty else 0
                )
                n_bins_actual = int(max_bin) + 2  # +2 to account for 0-indexing and potential overflow
                
                # Count observations in each bin
                baseline_counts = np.bincount(
                    baseline_binned.dropna().astype(int), 
                    minlength=n_bins_actual
                )
                current_counts = np.bincount(
                    current_binned.dropna().astype(int), 
                    minlength=n_bins_actual
                )
                
                # Convert to proportions with smoothing
                baseline_props = (baseline_counts + eps) / (baseline_counts.sum() + eps * len(baseline_counts))
                current_props = (current_counts + eps) / (current_counts.sum() + eps * len(current_counts))
                
                # Calculate PSI
                psi_vals = (baseline_props - current_props) * np.log(baseline_props / current_props)
                psi_total = np.sum(psi_vals)
                
                # Create result entry
                result = {
                    "monitor_id": f"psi_{col}",
                    "window_id": current_window,
                    "stratum": {k: v for k, v in stratum_row.items()},
                    "metric": "psi",
                    "value": float(psi_total),
                    "delta_from_baseline": float(psi_total),  # For first comparison, delta is the value itself
                    "ci_low": float(psi_total),  # Placeholder - would be calculated with bootstrap
                    "ci_high": float(psi_total),  # Placeholder - would be calculated with bootstrap
                    "p_value": 0.0,  # Placeholder - would be calculated with permutation test
                    "threshold_level": "none",  # Would be determined by comparing to thresholds
                }
                results.append(result)
        else:
            # No stratification - calculate overall PSI
            # Determine if column is numeric or categorical
            is_numeric = pd.api.types.is_numeric_dtype(baseline_data)
            
            # Get bin edges from baseline data
            if is_numeric:
                bin_edges = _get_bin_edges(
                    baseline_data, 
                    bin_type=bins, 
                    n_bins=n_bins, 
                    cutpoints=cutpoints
                )
            else:
                # For categorical data, use unique values from baseline
                bin_edges = np.array(sorted(baseline_data.dropna().unique()))
            
            # Bin the data
            try:
                if is_numeric:
                    baseline_binned = _bin_data(baseline_data, bin_edges, categorical=False)
                    current_binned = _bin_data(current_data, bin_edges, categorical=False)
                else:
                    baseline_binned = _bin_data(baseline_data, bin_edges, categorical=True)
                    current_binned = _bin_data(current_data, bin_edges, categorical=True)
            except Exception:
                # Fallback: treat as categorical if binning fails
                baseline_binned = _bin_data(baseline_data, bin_edges, categorical=True)
                current_binned = _bin_data(current_data, bin_edges, categorical=True)
            
            # Calculate PSI
            max_bin = max(
                baseline_binned.max() if not baseline_binned.empty else 0,
                current_binned.max() if not current_binned.empty else 0
            )
            n_bins_actual = int(max_bin) + 2  # +2 to account for 0-indexing and potential overflow
            
            # Count observations in each bin
            baseline_counts = np.bincount(
                baseline_binned.dropna().astype(int), 
                minlength=n_bins_actual
            )
            current_counts = np.bincount(
                current_binned.dropna().astype(int), 
                minlength=n_bins_actual
            )
            
            # Convert to proportions with smoothing
            baseline_props = (baseline_counts + eps) / (baseline_counts.sum() + eps * len(baseline_counts))
            current_props = (current_counts + eps) / (current_counts.sum() + eps * len(current_counts))
            
            # Calculate PSI
            psi_vals = (baseline_props - current_props) * np.log(baseline_props / current_props)
            psi_total = np.sum(psi_vals)
            
            # Create result entry
            result = {
                "monitor_id": f"psi_{col}",
                "window_id": current_window,
                "stratum": {},  # No stratification
                "metric": "psi",
                "value": float(psi_total),
                "delta_from_baseline": float(psi_total),  # For first comparison, delta is the value itself
                "ci_low": float(psi_total),  # Placeholder - would be calculated with bootstrap
                "ci_high": float(psi_total),  # Placeholder - would be calculated with bootstrap
                "p_value": 0.0,  # Placeholder - would be calculated with permutation test
                "threshold_level": "none",  # Would be determined by comparing to thresholds
            }
            results.append(result)
    
    return pd.DataFrame(results)
