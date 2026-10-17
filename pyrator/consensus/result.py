"""Structured results container for consensus estimation."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np

from pyrator.types import BackendAffinity


@dataclass(frozen=True)
class ConsensusResult:
    """Container for consensus estimation results across multiple raters and items.

    Preserves backend affinity so users can export back to Polars, Pandas, Arrow,
    DuckDB, or Ibis seamlessly.
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
    backend_affinity: BackendAffinity = "polars"

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

    def to_duckdb(self) -> Any:
        """Export consensus predictions as a DuckDB relation."""
        import duckdb

        arrow_tbl = self.to_arrow()
        return duckdb.from_arrow(arrow_tbl)

    def to_ibis(self) -> Any:
        """Export consensus predictions as an Ibis memtable."""
        import ibis

        arrow_tbl = self.to_arrow()
        return ibis.memtable(arrow_tbl)

    def to_native(self) -> Any:
        """Export consensus predictions matching the caller's detected input backend.

        Falls back to polars DataFrame if installed, or pandas DataFrame if polars is unavailable.
        """
        if self.backend_affinity == "pandas":
            return self.to_pandas()
        if self.backend_affinity == "arrow":
            return self.to_arrow()
        if self.backend_affinity == "duckdb":
            return self.to_duckdb()
        if self.backend_affinity == "ibis":
            return self.to_ibis()
        if self.backend_affinity == "dict":
            return self.to_dict()
        try:
            return self.to_polars()
        except ImportError:
            return self.to_pandas()

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
