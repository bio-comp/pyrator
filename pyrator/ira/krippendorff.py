"""Krippendorff's Alpha implementation using Narwhals pre-aggregation and matrix operations."""

from __future__ import annotations

from typing import Literal, cast

import numpy as np
from numpy.typing import NDArray

from pyrator.data.pipeline import compute_item_label_matrix
from pyrator.types import FrameLike


class KrippendorffAlpha:
    """Calculates Krippendorff's Alpha for inter-rater reliability.

    Supports:
    - Pre-aggregation in native dataframe engines (Polars/DuckDB/Pandas) via Narwhals.
    - Nominal and Interval metrics (standard distance functions).
    - Semantic metrics (via custom ontology distance matrices).
    - Variable number of raters per item (missing data).
    """

    def __init__(self, df: FrameLike, item_col: str, rater_col: str, label_col: str):
        self.raw_data = df
        self.item_col = item_col
        self.rater_col = rater_col
        self.label_col = label_col

        # Pre-aggregate directly through Narwhals without eager pandas materialization
        self.U, self.items, self.labels = compute_item_label_matrix(
            self.raw_data, self.item_col, self.label_col
        )
        self.label_map = {label: i for i, label in enumerate(self.labels)}
        self.n_labels = len(self.labels)

    def calculate(
        self,
        metric: Literal["nominal", "interval", "custom"] = "nominal",
        distance_matrix: NDArray[np.float64] | None = None,
    ) -> float:
        """Compute Krippendorff's Alpha."""
        U = self.U.copy()

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
            if not np.allclose(D2, D2.T):
                raise ValueError("Distance matrix must be symmetric.")
        else:
            if metric not in ("nominal", "interval"):
                raise ValueError(f"Unknown metric '{metric}' and no distance_matrix provided.")
            D2 = self._get_standard_metric(cast(Literal["nominal", "interval"], metric))

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

        return float(1.0 - (d_o / d_e))

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
