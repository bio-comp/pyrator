"""Structured results container for consensus estimation."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Literal

import numpy as np


@dataclass(frozen=True)
class ConsensusResult:
    """Container for consensus estimation results across multiple raters and items.

    Preserves backend affinity so users can export back to Polars, Pandas, or Arrow seamlessly.
    """

    item_ids: list[str]
    consensus_labels: list[str]
    confidences: np.ndarray
    class_probabilities: np.ndarray
    classes: list[str]
    annotator_confusion: dict[str, np.ndarray] = field(default_factory=dict)
    converged: bool = True
    iterations: int = 1
    log_likelihood: float = 0.0
    backend_affinity: Literal["polars", "pandas", "arrow", "dict"] = "polars"

    def to_pandas(self) -> Any:
        """Export consensus predictions as a pandas DataFrame."""
        import pandas as pd

        return pd.DataFrame(
            {
                "item_id": self.item_ids,
                "consensus_label": self.consensus_labels,
                "confidence": self.confidences,
            }
        )

    def to_polars(self) -> Any:
        """Export consensus predictions as a polars DataFrame."""
        import polars as pl

        return pl.DataFrame(
            {
                "item_id": self.item_ids,
                "consensus_label": self.consensus_labels,
                "confidence": self.confidences,
            }
        )

    def to_arrow(self) -> Any:
        """Export consensus predictions as a pyarrow Table."""
        import pyarrow as pa

        return pa.Table.from_pydict(
            {
                "item_id": self.item_ids,
                "consensus_label": self.consensus_labels,
                "confidence": self.confidences,
            }
        )

    def to_dict(self) -> dict[str, Any]:
        """Export summary results as a serializable Python dictionary."""
        return {
            "item_ids": self.item_ids,
            "consensus_labels": self.consensus_labels,
            "confidences": self.confidences.tolist(),
            "classes": self.classes,
            "converged": self.converged,
            "iterations": self.iterations,
            "log_likelihood": float(self.log_likelihood),
            "backend_affinity": self.backend_affinity,
        }
