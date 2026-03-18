"""Drift monitoring configuration and execution framework."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal, Optional, cast

import pandas as pd

from pyrator.drift.cramer_v import cramer_v
from pyrator.drift.jsd import jsd
from pyrator.drift.mmd import mmd
from pyrator.drift.psi import psi
from pyrator.drift.wasserstein import w1
from pyrator.types import require_non_none


@dataclass
class MonitorConfig:
    """Base configuration for all drift monitors."""

    name: str
    metric: Literal["psi", "cramer_v", "jsd", "wasserstein", "mmd"]
    warn: Optional[float] = None
    crit: Optional[float] = None
    semantics: Literal["abs", "delta"] = "abs"
    window_col: str = "window_id"
    baseline: Optional[str] = None


@dataclass
class PsiMonitorConfig(MonitorConfig):
    """Configuration for PSI drift monitor."""

    metric: Literal["psi"] = "psi"
    col: Optional[str] = None
    bins: Literal["quantile", "fd", "scott", "rice", "sturges", "sqrt"] = "quantile"
    n_bins: int = 10
    cutpoints: Optional[list[float]] = None
    stratify: Optional[list[str]] = None
    eps: float = 1e-6


@dataclass
class CramerVMonitorConfig(MonitorConfig):
    """Configuration for Cramer's V drift monitor."""

    metric: Literal["cramer_v"] = "cramer_v"
    x: Optional[str] = None
    y: Optional[str] = None
    bias_correct: bool = True
    stratify: Optional[list[str]] = None


@dataclass
class JsdMonitorConfig(MonitorConfig):
    """Configuration for JSD drift monitor."""

    metric: Literal["jsd"] = "jsd"
    dist_cols: Optional[list[str]] = None
    groupby: Optional[str] = None
    sqrt: bool = True
    eps: float = 1e-6


@dataclass
class WassersteinMonitorConfig(MonitorConfig):
    """Configuration for Wasserstein distance drift monitor."""

    metric: Literal["wasserstein"] = "wasserstein"
    col: Optional[str] = None
    weight_type: Literal["uniform"] = "uniform"
    stratify: Optional[list[str]] = None


@dataclass
class MmdMonitorConfig(MonitorConfig):
    """Configuration for MMD drift monitor."""

    metric: Literal["mmd"] = "mmd"
    emb_cols: Optional[list[str]] = None
    kernel: Literal["rbf"] = "rbf"
    sigma: Literal["median_heuristic"] | float = "median_heuristic"
    n_perm: int = 1000
    seed: Optional[int] = None
    stratify: Optional[list[str]] = None


class Monitor:
    """Drift monitor that executes a specific metric calculation."""

    def __init__(self, config: MonitorConfig):
        self.config = config

    def execute(self, data: pd.DataFrame) -> pd.DataFrame | tuple[float, float]:
        """Execute the monitor on the provided data."""
        if isinstance(self.config, PsiMonitorConfig):
            return self._execute_psi(data)
        if isinstance(self.config, CramerVMonitorConfig):
            return self._execute_cramer_v(data)
        if isinstance(self.config, JsdMonitorConfig):
            return self._execute_jsd(data)
        if isinstance(self.config, WassersteinMonitorConfig):
            return self._execute_wasserstein(data)
        if isinstance(self.config, MmdMonitorConfig):
            return self._execute_mmd(data)
        raise ValueError(f"Unsupported metric: {type(self.config)}")

    def _execute_psi(self, data: pd.DataFrame) -> pd.DataFrame:
        cfg = cast(PsiMonitorConfig, self.config)
        require_non_none(cfg.col, "PSI monitor requires 'col' parameter")
        return psi(
            data=data,
            col=cfg.col,  # type: ignore[arg-type]
            window_col=cfg.window_col,
            baseline=cfg.baseline,
            bins=cfg.bins,
            n_bins=cfg.n_bins,
            cutpoints=cfg.cutpoints,
            stratify=cfg.stratify,
            eps=cfg.eps,
        )

    def _execute_cramer_v(self, data: pd.DataFrame) -> pd.DataFrame:
        cfg = cast(CramerVMonitorConfig, self.config)
        require_non_none(cfg.x, "CramerV monitor requires 'x' parameter")
        require_non_none(cfg.y, "CramerV monitor requires 'y' parameter")
        return cramer_v(
            data=data,
            x=cfg.x,  # type: ignore[arg-type]
            y=cfg.y,  # type: ignore[arg-type]
            window_col=cfg.window_col,
            baseline=cfg.baseline,
            bias_correct=cfg.bias_correct,
            stratify=cfg.stratify,
        )

    def _execute_jsd(self, data: pd.DataFrame) -> pd.DataFrame:
        cfg = cast(JsdMonitorConfig, self.config)
        require_non_none(cfg.dist_cols, "JSD monitor requires 'dist_cols' parameter")
        return jsd(
            data=data,
            dist_cols=cfg.dist_cols,  # type: ignore[arg-type]
            window_col=cfg.window_col,
            baseline=cfg.baseline,
            groupby=cfg.groupby,
            eps=cfg.eps,
            sqrt=cfg.sqrt,
        )

    def _execute_wasserstein(self, data: pd.DataFrame) -> pd.DataFrame:
        cfg = cast(WassersteinMonitorConfig, self.config)
        require_non_none(cfg.col, "Wasserstein monitor requires 'col' parameter")
        return w1(
            data=data,
            col=cfg.col,  # type: ignore[arg-type]
            window_col=cfg.window_col,
            baseline=cfg.baseline,
            weight_type=cfg.weight_type,
            stratify=cfg.stratify,
        )

    def _execute_mmd(self, data: pd.DataFrame) -> tuple[float, float]:
        cfg = cast(MmdMonitorConfig, self.config)
        require_non_none(cfg.emb_cols, "MMD monitor requires 'emb_cols' parameter")
        return mmd(  # type: ignore[return-value]
            data=data,
            emb_cols=cfg.emb_cols,  # type: ignore[arg-type]
            window_col=cfg.window_col,
            baseline=cfg.baseline,
            kernel=cfg.kernel,
            sigma=cfg.sigma,
            n_perm=cfg.n_perm,
            seed=cfg.seed,
            stratify=cfg.stratify,
        )

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
