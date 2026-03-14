"""Drift monitoring configuration and execution framework."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable, Literal, Optional

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
    col: Optional[str] = None
    bins: Literal["quantile", "fd", "scott", "rice", "sturges", "sqrt"] = "quantile"
    n_bins: int = 10
    cutpoints: Optional[list[float]] = None
    x: Optional[str] = None
    y: Optional[str] = None
    bias_correct: bool = True
    dist_cols: Optional[list[str]] = None
    groupby: Optional[str] = None
    sqrt: bool = True
    weight_type: Literal["uniform", "ic"] = "uniform"
    emb_cols: Optional[list[str]] = None
    kernel: Literal["rbf"] = "rbf"
    sigma: Literal["median_heuristic"] | float = "median_heuristic"
    n_perm: int = 1000
    seed: Optional[int] = None
    window_col: str = "window_id"
    stratify: Optional[list[str]] = None
    eps: float = 1e-6
    warn: Optional[float] = None
    crit: Optional[float] = None
    semantics: Literal["abs", "delta"] = "abs"


class Monitor:
    """Drift monitor that executes a specific metric calculation."""

    _DISPATCH_MAP: dict[str, tuple[Callable[..., Any], dict[str, Any]]] = {
        "psi": (
            psi,
            {
                "required": {"col": "col"},
                "params": ["col", "bins", "n_bins", "cutpoints", "stratify", "eps"],
            },
        ),
        "cramer_v": (
            cramer_v,
            {"required": {"x": "x", "y": "y"}, "params": ["x", "y", "bias_correct", "stratify"]},
        ),
        "jsd": (
            jsd,
            {
                "required": {"dist_cols": "dist_cols"},
                "params": ["dist_cols", "groupby", "eps", "sqrt"],
            },
        ),
        "wasserstein": (
            w1,
            {"required": {"col": "col"}, "params": ["col", "weight_type", "stratify"]},
        ),
        "mmd": (
            mmd,
            {
                "required": {"emb_cols": "emb_cols"},
                "params": ["emb_cols", "kernel", "sigma", "n_perm", "seed", "stratify"],
            },
        ),
    }

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
        dispatch_entry = self._DISPATCH_MAP.get(self.config.metric)
        if dispatch_entry is None:
            raise ValueError(f"Unsupported metric: {self.config.metric}")

        func, config = dispatch_entry

        for param, attr in config["required"].items():
            if getattr(self.config, attr) is None:
                raise ValueError(
                    f"{self.config.metric.capitalize()} monitor requires '{param}' parameter"
                )

        kwargs: dict[str, Any] = {"data": data, "window_col": self.config.window_col}
        for param in config["params"]:
            value = getattr(self.config, param, None)
            if value is not None:
                kwargs[param] = value

        # The underlying metric functions can return DataFrame or tuple
        result = func(**kwargs)
        return result  # type: ignore[no-any-return]

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
