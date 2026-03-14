"""Strictly typed result classes for the API."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

import pandas as pd


@dataclass(slots=True)
class AgreementResults:
    """Structured results produced by frequentist agreement estimators."""

    alpha: float
    mode: Literal["nominal", "semantic"]
    metric: str | None
    hard_items: pd.DataFrame
    annotator_profiles: pd.DataFrame
    consensus_labels: pd.Series

    def get_hard_items(self, top_n: int = 10) -> pd.DataFrame:
        """Return the highest-disagreement items."""
        if top_n <= 0:
            raise ValueError("top_n must be positive.")
        return self.hard_items.head(top_n).copy()

    def get_annotator_profiles(self) -> pd.DataFrame:
        """Return per-annotator agreement summaries."""
        return self.annotator_profiles.copy()

    def get_consensus_labels(self) -> pd.Series:
        """Return deterministic item-level consensus labels."""
        return self.consensus_labels.copy()
