"""Drift monitoring configuration and execution framework."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal, Optional

import pandas as pd

from pyrator.drift.cramer_v import cramer_v
from pyrator.drift.jsd import jsd
from pyrator.drift.mmd import mmd
from pyrator.drift.psi import psi
from pyrator.drift.wasserstein import w1


@dataclass
class MonitorConfig:
    """Configuration for a drift monitor."""

    name: str
    metric: Literal["psi", "cramer_v", "jsd", "wasserstein", "mmd"]
    # For PSI
    col: Optional[str] = None
    bins: Literal["quantile", "fd", "scott", "rice", "sturges", "sqrt"] = "quantile"
    n_bins: int = 10
    cutpoints: Optional[list[float]] = None
    # For Cramér's V
    x: Optional[str] = None
    y: Optional[str] = None
    bias_correct: bool = True
    # For JSD
    dist_cols: Optional[list[str]] = None
    groupby: Optional[str] = None
    sqrt: bool = True
    # For Wasserstein
    weight_type: Literal["uniform", "ic"] = "uniform"
    # For MMD
    emb_cols: Optional[list[str]] = None
    kernel: Literal["rbf"] = "rbf"
    sigma: Literal["median_heuristic"] | float = "median_heuristic"
    n_perm: int = 1000
    seed: Optional[int] = None
    # Common parameters
    window_col: str = "window_id"
    stratify: Optional[list[str]] = None
    eps: float = 1e-6

    # Alert thresholds
    warn: Optional[float] = None
    crit: Optional[float] = None
    semantics: Literal["abs", "delta"] = "abs"


class Monitor:
    """Drift monitor that executes a specific metric calculation."""

    def __init__(self, config: MonitorConfig):
        self.config = config

    def execute(self, data: pd.DataFrame) -> pd.DataFrame | tuple[float, float]:
        """
        Execute the monitor on the provided data.

        Args:
            data: Input data frame

        Returns:
            For most metrics: DataFrame with results
            For MMD: tuple of (statistic, p-value)
        """
        if self.config.metric == "psi":
            if not self.config.col:
                raise ValueError("PSI monitor requires 'col' parameter")
            return psi(
                data,
                col=self.config.col,
                window_col=self.config.window_col,
                bins=self.config.bins,
                n_bins=self.config.n_bins,
                cutpoints=self.config.cutpoints,
                stratify=self.config.stratify,
                eps=self.config.eps,
            )
        elif self.config.metric == "cramer_v":
            if not self.config.x or not self.config.y:
                raise ValueError("Cramér's V monitor requires 'x' and 'y' parameters")
            return cramer_v(
                data,
                x=self.config.x,
                y=self.config.y,
                window_col=self.config.window_col,
                bias_correct=self.config.bias_correct,
                stratify=self.config.stratify,
            )
        elif self.config.metric == "jsd":
            if not self.config.dist_cols:
                raise ValueError("JSD monitor requires 'dist_cols' parameter")
            return jsd(
                data,
                dist_cols=self.config.dist_cols,
                window_col=self.config.window_col,
                groupby=self.config.groupby,
                eps=self.config.eps,
                sqrt=self.config.sqrt,
            )
        elif self.config.metric == "wasserstein":
            if not self.config.col:
                raise ValueError("Wasserstein monitor requires 'col' parameter")
            return w1(
                data,
                col=self.config.col,
                window_col=self.config.window_col,
                weight_type=self.config.weight_type,
                stratify=self.config.stratify,
            )
        elif self.config.metric == "mmd":
            if not self.config.emb_cols:
                raise ValueError("MMD monitor requires 'emb_cols' parameter")
            return mmd(
                data,
                emb_cols=self.config.emb_cols,
                window_col=self.config.window_col,
                kernel=self.config.kernel,
                sigma=self.config.sigma,
                n_perm=self.config.n_perm,
                seed=self.config.seed,
                stratify=self.config.stratify,
            )
        else:
            raise ValueError(f"Unsupported metric: {self.config.metric}")

    def evaluate_thresholds(self, value: float) -> str:
        """
        Evaluate threshold levels for a given value.

        Args:
            value: The metric value to evaluate

        Returns:
            Threshold level: "none", "warn", or "crit"
        """
        if self.config.warn is None and self.config.crit is None:
            return "none"

        # For delta semantics, we use absolute value
        if self.config.semantics == "delta":
            value = abs(value)

        if self.config.crit is not None and value >= self.config.crit:
            return "crit"
        elif self.config.warn is not None and value >= self.config.warn:
            return "warn"
        else:
            return "none"
