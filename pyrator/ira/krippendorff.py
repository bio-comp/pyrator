"""Krippendorff's Alpha implementation using matrix operations."""

from __future__ import annotations

from typing import Literal

import numpy as np
from numpy.typing import NDArray

from pyrator.types import FrameLike


class KrippendorffAlpha:
    """
    Calculates Krippendorff's Alpha for inter-rater reliability.

    Supports:
    - Nominal, Interval metrics (via standard distance functions).
    - Semantic metrics (via custom distance matrices).
    - Missing data (handles variable number of raters per item).
    """

    def __init__(self, df: FrameLike, item_col: str, rater_col: str, label_col: str):
        """
        Initialize with a long-format DataFrame (Item, Rater, Label).

        Args:
            df: DataFrame containing the annotations.
            item_col: Column name for items/subjects.
            rater_col: Column name for raters/annotators.
            label_col: Column name for assigned labels/categories.
        """
        if hasattr(df, "to_pandas"):
            self.df = df.to_pandas()
        else:
            self.df = df

        self.item_col = item_col
        self.rater_col = rater_col
        self.label_col = label_col

        self.labels = sorted(self.df[label_col].unique())
        self.label_map = {label: i for i, label in enumerate(self.labels)}
        self.n_labels = len(self.labels)

    def calculate(
        self,
        metric: Literal["nominal", "interval", "custom"] = "nominal",
        distance_matrix: NDArray[np.float64] | None = None,
    ) -> float:
        """
        Compute Krippendorff's Alpha.

        Args:
            metric: The metric level of the data.
            distance_matrix: Optional pre-computed (V x V) distance matrix.
                             Required if metric is 'custom'.
                             If provided, 'metric' argument is ignored.
        """
        pivot = self.df.groupby([self.item_col, self.label_col]).size().unstack(fill_value=0)

        pivot = pivot.reindex(columns=self.labels, fill_value=0)

        U = pivot.to_numpy()

        m_u = U.sum(axis=1)

        valid_items = m_u >= 2
        U = U[valid_items]
        m_u = m_u[valid_items]

        if len(U) == 0:
            return 0.0

        N = len(U)
        n_total_assignments = m_u.sum()

        if distance_matrix is not None:
            D2 = distance_matrix
            if D2.shape != (self.n_labels, self.n_labels):
                raise ValueError(
                    f"Distance matrix shape {D2.shape} does not match "
                    f"unique label count ({self.n_labels})"
                )
        else:
            D2 = self._get_standard_metric(metric)  # type: ignore[arg-type]

        scale_per_item = 1.0 / (m_u * (m_u - 1))

        term1 = U @ D2
        row_disagreements = np.einsum("ij,ij->i", term1, U)

        total_observed_disagreement = np.sum(row_disagreements * scale_per_item)

        n_c = U.sum(axis=0)

        term2 = n_c @ D2
        total_expected_disagreement = np.dot(term2, n_c)

        d_o = total_observed_disagreement / N
        d_e = total_expected_disagreement / (n_total_assignments * (n_total_assignments - 1))

        if d_e == 0:
            return 1.0

        result: float = 1.0 - (d_o / d_e)
        return result

    def _get_standard_metric(self, metric: Literal["nominal", "interval"]) -> NDArray[np.float64]:
        """Generate V x V distance matrix for standard metrics."""
        v = self.n_labels

        if metric == "interval":
            try:
                num_vals = np.array([float(x) for x in self.labels])
            except ValueError as e:
                raise ValueError("Labels must be numeric for interval metric") from e

        d = np.zeros((v, v))

        if metric == "nominal":
            d = 1.0 - np.eye(v)

        elif metric == "interval":
            col, row = np.meshgrid(num_vals, num_vals)
            d = (col - row) ** 2

        return d
