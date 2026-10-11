# MonitorConfig ISP Refactor Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Refactor MonitorConfig from monolithic dataclass to discriminated union of metric-specific config classes.

**Architecture:** Base MonitorConfig class with metric-specific subclasses (PsiMonitorConfig, CramerVMonitorConfig, etc.). Monitor uses match/case for dispatch. CLI maps flat dict to correct config type.

**Tech Stack:** Python dataclasses, Literal types, match/case statements

---

### Task 1: Create Metric-Specific Config Classes

**Files:**
- Modify: `pyrator/drift/monitor.py:1-141`

**Step 1: Write the failing test**

```python
# tests/test_drift/test_monitor_config.py
import pytest
from pyrator.drift.monitor import (
    MonitorConfig,
    PsiMonitorConfig,
    CramerVMonitorConfig,
    JsdMonitorConfig,
    WassersteinMonitorConfig,
    MmdMonitorConfig,
)

def test_psi_monitor_config_creation():
    config = PsiMonitorConfig(name="test", col="value", n_bins=20)
    assert config.name == "test"
    assert config.col == "value"
    assert config.n_bins == 20

def test_cramer_v_monitor_config_creation():
    config = CramerVMonitorConfig(name="test", x="col1", y="col2")
    assert config.name == "test"
    assert config.x == "col1"
    assert config.y == "col2"

def test_monitor_dispatch_by_config_type():
    psi_config = PsiMonitorConfig(name="test", col="value")
    monitor = Monitor(psi_config)
    assert isinstance(monitor.config, PsiMonitorConfig)
```

**Step 2: Run test to verify it fails**

Run: `pytest tests/test_drift/test_monitor_config.py -v`
Expected: FAIL - classes don't exist

**Step 3: Write minimal implementation**

Replace the MonitorConfig dataclass with:

```python
from dataclasses import dataclass
from typing import Any, Callable, Literal, Optional
import pandas as pd

@dataclass
class MonitorConfig:
    """Base configuration for all drift monitors."""
    name: str
    metric: Literal["psi", "cramer_v", "jsd", "wasserstein", "mmd"]
    warn: Optional[float] = None
    crit: Optional[float] = None
    semantics: Literal["abs", "delta"] = "abs"
    window_col: str = "window_id"


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
    weight_type: Literal["uniform", "ic"] = "uniform"
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
```

**Step 4: Run test to verify it passes**

Run: `pytest tests/test_drift/test_monitor_config.py::test_psi_monitor_config_creation -v`
Expected: PASS

**Step 5: Commit**

```bash
git add pyrator/drift/monitor.py
git commit -m "feat: add metric-specific MonitorConfig classes"
```

---

### Task 2: Refactor Monitor.dispatch_map to use match/case

**Files:**
- Modify: `pyrator/drift/monitor.py:48-141`

**Step 1: Write the failing test**

```python
def test_monitor_execute_dispatches_correctly():
    import pandas as pd
    import numpy as np
    
    # Create sample data
    data = pd.DataFrame({
        "window_id": [1, 1, 2, 2],
        "value": [1.0, 2.0, 3.0, 4.0]
    })
    
    config = PsiMonitorConfig(name="test", col="value")
    monitor = Monitor(config)
    
    # Should dispatch to psi function and return DataFrame
    result = monitor.execute(data)
    assert isinstance(result, pd.DataFrame)
```

**Step 2: Run test to verify it fails**

Run: `pytest tests/test_drift/test_monitor_config.py::test_monitor_execute_dispatches_correctly -v`
Expected: FAIL - dispatch not implemented

**Step 3: Write minimal implementation**

Replace the _DISPATCH_MAP and execute method with:

```python
class Monitor:
    """Drift monitor that executes a specific metric calculation."""

    def __init__(self, config: MonitorConfig):
        self.config = config

    def execute(self, data: pd.DataFrame) -> pd.DataFrame | tuple[float, float]:
        """Execute the monitor on the provided data."""
        match self.config:
            case PsiMonitorConfig():
                return self._execute_psi(data)
            case CramerVMonitorConfig():
                return self._execute_cramer_v(data)
            case JsdMonitorConfig():
                return self._execute_jsd(data)
            case WassersteinMonitorConfig():
                return self._execute_wasserstein(data)
            case MmdMonitorConfig():
                return self._execute_mmd(data)
            case _:
                raise ValueError(f"Unsupported metric: {type(self.config)}")

    def _execute_psi(self, data: pd.DataFrame) -> pd.DataFrame:
        cfg = self.config
        require_non_none(cfg.col, "PSI monitor requires 'col' parameter")
        return psi(
            data=data,
            col=cfg.col,
            window_col=cfg.window_col,
            bins=cfg.bins,
            n_bins=cfg.n_bins,
            cutpoints=cfg.cutpoints,
            stratify=cfg.stratify,
            eps=cfg.eps,
        )

    def _execute_cramer_v(self, data: pd.DataFrame) -> pd.DataFrame:
        cfg = self.config
        require_non_none(cfg.x, "CramerV monitor requires 'x' parameter")
        require_non_none(cfg.y, "CramerV monitor requires 'y' parameter")
        return cramer_v(
            data=data,
            x=cfg.x,
            y=cfg.y,
            window_col=cfg.window_col,
            bias_correct=cfg.bias_correct,
            stratify=cfg.stratify,
        )

    def _execute_jsd(self, data: pd.DataFrame) -> pd.DataFrame:
        cfg = self.config
        require_non_none(cfg.dist_cols, "JSD monitor requires 'dist_cols' parameter")
        return jsd(
            data=data,
            dist_cols=cfg.dist_cols,
            window_col=cfg.window_col,
            groupby=cfg.groupby,
            eps=cfg.eps,
            sqrt=cfg.sqrt,
        )

    def _execute_wasserstein(self, data: pd.DataFrame) -> pd.DataFrame:
        cfg = self.config
        require_non_none(cfg.col, "Wasserstein monitor requires 'col' parameter")
        return w1(
            data=data,
            col=cfg.col,
            window_col=cfg.window_col,
            weight_type=cfg.weight_type,
            stratify=cfg.stratify,
        )

    def _execute_mmd(self, data: pd.DataFrame) -> tuple[float, float]:
        cfg = self.config
        require_non_none(cfg.emb_cols, "MMD monitor requires 'emb_cols' parameter")
        return mmd(
            data=data,
            emb_cols=cfg.emb_cols,
            window_col=cfg.window_col,
            kernel=cfg.kernel,
            sigma=cfg.sigma,
            n_perm=cfg.n_perm,
            seed=cfg.seed,
            stratify=cfg.stratify,
        )
```

**Step 4: Run test to verify it passes**

Run: `pytest tests/test_drift/test_monitor_config.py::test_monitor_execute_dispatches_correctly -v`
Expected: PASS

**Step 5: Commit**

```bash
git add pyrator/drift/monitor.py
git commit -m "feat: refactor Monitor to use match/case dispatch"
```

---

### Task 3: Update CLI to use new config classes

**Files:**
- Modify: `pyrator/cli/drift.py:68-70`

**Step 1: Write the failing test**

```python
def test_cli_loads_psi_config():
    from pyrator.cli.drift import load_config
    
    config_dict = {
        "name": "test",
        "metric": "psi",
        "col": "value",
    }
    
    config = load_config(config_dict)
    assert isinstance(config, PsiMonitorConfig)
    assert config.col == "value"
```

**Step 2: Run test to verify it fails**

Run: `pytest tests/test_drift/test_cli_drift.py::test_cli_loads_psi_config -v`
Expected: FAIL - load_config doesn't exist

**Step 3: Write minimal implementation**

Add to drift.py:

```python
def load_config(config_dict: dict) -> MonitorConfig:
    """Load a MonitorConfig from a dictionary."""
    metric = config_dict.get("metric")
    if metric is None:
        raise ValueError("Config must include 'metric' field")
    
    # Common fields
    common = {
        "name": config_dict.get("name", "unnamed"),
        "warn": config_dict.get("warn"),
        "crit": config_dict.get("crit"),
        "semantics": config_dict.get("semantics", "abs"),
        "window_col": config_dict.get("window_col", "window_id"),
    }
    
    match metric:
        case "psi":
            return PsiMonitorConfig(
                **common,
                col=config_dict.get("col"),
                bins=config_dict.get("bins", "quantile"),
                n_bins=config_dict.get("n_bins", 10),
                cutpoints=config_dict.get("cutpoints"),
                stratify=config_dict.get("stratify"),
                eps=config_dict.get("eps", 1e-6),
            )
        case "cramer_v":
            return CramerVMonitorConfig(
                **common,
                x=config_dict.get("x"),
                y=config_dict.get("y"),
                bias_correct=config_dict.get("bias_correct", True),
                stratify=config_dict.get("stratify"),
            )
        case "jsd":
            return JsdMonitorConfig(
                **common,
                dist_cols=config_dict.get("dist_cols"),
                groupby=config_dict.get("groupby"),
                sqrt=config_dict.get("sqrt", True),
                eps=config_dict.get("eps", 1e-6),
            )
        case "wasserstein":
            return WassersteinMonitorConfig(
                **common,
                col=config_dict.get("col"),
                weight_type=config_dict.get("weight_type", "uniform"),
                stratify=config_dict.get("stratify"),
            )
        case "mmd":
            return MmdMonitorConfig(
                **common,
                emb_cols=config_dict.get("emb_cols"),
                kernel=config_dict.get("kernel", "rbf"),
                sigma=config_dict.get("sigma", "median_heuristic"),
                n_perm=config_dict.get("n_perm", 1000),
                seed=config_dict.get("seed"),
                stratify=config_dict.get("stratify"),
            )
        case _:
            raise ValueError(f"Unknown metric: {metric}")
```

Then update line 69:
```python
monitor_config = load_config(config_dict)
```

**Step 4: Run test to verify it passes**

Run: `pytest tests/test_drift/test_cli_drift.py::test_cli_loads_psi_config -v`
Expected: PASS

**Step 5: Commit**

```bash
git add pyrator/cli/drift.py
git commit -m "feat: update CLI to use metric-specific config classes"
```

---

### Task 4: Update exports in __init__.py

**Files:**
- Modify: `pyrator/drift/__init__.py:1-20`

**Step 1: Update exports**

Add the new config classes to exports:

```python
from pyrator.drift.monitor import (
    Monitor,
    MonitorConfig,
    PsiMonitorConfig,
    CramerVMonitorConfig,
    JsdMonitorConfig,
    WassersteinMonitorConfig,
    MmdMonitorConfig,
)

__all__ = [
    "Monitor",
    "MonitorConfig",
    "PsiMonitorConfig",
    "CramerVMonitorConfig",
    "JsdMonitorConfig",
    "WassersteinMonitorConfig",
    "MmdMonitorConfig",
]
```

**Step 2: Run tests**

Run: `pytest tests/test_drift/ -v`
Expected: All pass

**Step 3: Commit**

```bash
git add pyrator/drift/__init__.py
git commit -m "feat: export new config classes"
```

---

### Task 5: Run quality gate

**Files:**
- N/A

**Step 1: Run lint**

```bash
uv run ruff check . --fix
```

**Step 2: Run format**

```bash
uv run ruff format .
```

**Step 3: Run type check**

```bash
uv run mypy pyrator/
```

**Step 4: Run full test suite**

```bash
uv run pytest tests/ -v
```

---

### Task 6: Push and create PR

**Files:**
- N/A

**Step 1: Create branch**

```bash
git checkout -b issue/38-monitor-config-isp
```

**Step 2: Push**

```bash
git push -u origin issue/38-monitor-config-isp
```

**Step 3: Create PR**

```bash
gh pr create --title "refactor: MonitorConfig ISP refactor (#38)" --body "$(cat <<'EOF'
## Summary
- Refactor MonitorConfig from monolithic dataclass to discriminated union
- Add metric-specific config classes (PsiMonitorConfig, CramerVMonitorConfig, etc.)
- Update Monitor to use match/case dispatch
- Update CLI to map flat dict to correct config type

## Testing
- All tests pass
- mypy and ruff clean
EOF
)"
```
