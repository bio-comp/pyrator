# MonitorConfig Refactor Design

## Overview

Refactor `MonitorConfig` from a monolithic dataclass to a discriminated union of metric-specific config classes, applying Interface Segregation Principle properly.

## Background

The current `MonitorConfig` dataclass has 26 fields covering 5 different metrics (psi, cramer_v, jsd, wasserstein, mmd). Each metric only uses a subset of fields, creating a confusing API where users must know which fields apply to their chosen metric.

## Architecture

### Base Configuration

```python
@dataclass
class MonitorConfig:
    """Base configuration for all drift monitors."""
    name: str
    metric: Literal["psi", "cramer_v", "jsd", "wasserstein", "mmd"]
    warn: Optional[float] = None
    crit: Optional[float] = None
    semantics: Literal["abs", "delta"] = "abs"
    window_col: str = "window_id"
```

### Metric-Specific Configs

Each metric extends the base with its required parameters:

- **PsiMonitorConfig**: `col`, `bins`, `n_bins`, `cutpoints`, `stratify`, `eps`
- **CramerVMonitorConfig**: `x`, `y`, `bias_correct`, `stratify`
- **JsdMonitorConfig**: `dist_cols`, `groupby`, `eps`, `sqrt`
- **WassersteinMonitorConfig**: `col`, `weight_type`, `stratify`
- **MmdMonitorConfig**: `emb_cols`, `kernel`, `sigma`, `n_perm`, `seed`, `stratify`

### Monitor Dispatch

The `Monitor` class uses `match/case` statement to dispatch based on config type:

```python
def execute(self, data: pd.DataFrame) -> pd.DataFrame | tuple[float, float]:
    match self.config:
        case PsiMonitorConfig():
            return psi(data, ...)
        case CramerVMonitorConfig():
            return cramer_v(data, ...)
        # etc.
```

### CLI Changes

The CLI maps flat config dict to correct config type:

```python
def _dict_to_config(config_dict: dict) -> MonitorConfig:
    metric = config_dict.pop("metric")
    match metric:
        case "psi": return PsiMonitorConfig(**config_dict)
        case "cramer_v": return CramerVMonitorConfig(**config_dict)
        # etc.
```

## Benefits

1. **Type safety**: Type checkers can validate that only valid params are passed per metric
2. **Self-documenting**: Config classes make required fields explicit
3. **ISP compliance**: Users only deal with fields relevant to their metric
4. **Extensibility**: Adding new metrics follows the same pattern

## Testing Strategy

1. Unit tests for each config class construction
2. Integration tests for Monitor dispatch
3. CLI tests with sample configs per metric type
