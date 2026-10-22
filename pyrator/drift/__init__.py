"""Drift monitoring module for pyrator."""

from __future__ import annotations

from pyrator.drift.cramer_v import cramer_v
from pyrator.drift.jsd import jsd
from pyrator.drift.mmd import mmd
from pyrator.drift.monitor import (
    CramerVMonitorConfig,
    JsdMonitorConfig,
    MmdMonitorConfig,
    Monitor,
    MonitorConfig,
    PsiMonitorConfig,
    WassersteinMonitorConfig,
)
from pyrator.drift.psi import psi
from pyrator.drift.wasserstein import w1

# Common semantic aliases
wasserstein_distance = w1
cramers_v = cramer_v

__all__ = [
    "psi",
    "cramer_v",
    "cramers_v",
    "jsd",
    "w1",
    "wasserstein_distance",
    "mmd",
    "Monitor",
    "MonitorConfig",
    "PsiMonitorConfig",
    "CramerVMonitorConfig",
    "JsdMonitorConfig",
    "WassersteinMonitorConfig",
    "MmdMonitorConfig",
]
