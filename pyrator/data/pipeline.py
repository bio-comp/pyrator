"""Narwhals-based expression pipeline and materialization boundary.

This module provides universal, backend-agnostic dataframe handling. It preserves
eager or lazy execution across Polars, Pandas, and PyArrow until reaching the
explicit Materialization Boundary, where arrays are extracted into NumPy/SciPy core.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, cast

import narwhals.stable.v2 as nw
import numpy as np

from pyrator.types import BackendAffinity


def detect_backend_affinity(
    data: Any,
    default: BackendAffinity | None = None,
) -> BackendAffinity:
    """Detect the originating dataframe backend to preserve caller affinity.

    Recognizes native Polars, Pandas, PyArrow, DuckDB, Ibis, and Python dict containers.
    If the originating backend cannot be detected (e.g. file paths, unknown objects),
    returns `default` if provided, or 'unknown'.
    """
    raw = data
    if isinstance(data, (nw.DataFrame, nw.LazyFrame)):
        raw = nw.to_native(data)

    if isinstance(raw, dict):
        return "dict"

    mod_name = type(raw).__module__
    if "polars" in mod_name:
        return "polars"
    if "pandas" in mod_name:
        return "pandas"
    if "pyarrow" in mod_name:
        return "arrow"
    if "duckdb" in mod_name:
        return "duckdb"
    if "ibis" in mod_name:
        return "ibis"

    if default is not None:
        return default
    return "unknown"


def resolve_affinity(
    data: Any,
    nw_df: Any,
    default: BackendAffinity = "polars",
) -> BackendAffinity:
    """Resolve backend affinity from raw data, inspecting the loaded frame if raw was unknown."""
    affinity = detect_backend_affinity(data)
    if affinity == "unknown":
        return detect_backend_affinity(nw_df, default=default)
    return affinity


def normalize_to_narwhals(data: Any) -> nw.DataFrame[Any] | nw.LazyFrame[Any]:
    """Normalize input to a Narwhals DataFrame or LazyFrame without unnecessary copies."""
    if isinstance(data, (nw.DataFrame, nw.LazyFrame)):
        return data

    # File path handling
    if isinstance(data, (str, Path)):
        path = str(data)
        if path.endswith(".parquet"):
            try:
                import polars as pl

                data = pl.scan_parquet(path)
            except ImportError:
                try:
                    import pyarrow.parquet as pq

                    data = pq.read_table(path)
                except ImportError:
                    import pandas as pd

                    data = pd.read_parquet(path)
        elif path.endswith(".csv"):
            try:
                import polars as pl

                data = pl.scan_csv(path)
            except ImportError:
                import pandas as pd

                data = pd.read_csv(path)

    return cast(nw.DataFrame[Any] | nw.LazyFrame[Any], nw.from_native(data))


def filter_windows(
    df: nw.DataFrame[Any] | nw.LazyFrame[Any],
    window_col: str,
    baseline_id: str,
    current_id: str,
) -> tuple[nw.DataFrame[Any] | nw.LazyFrame[Any], nw.DataFrame[Any] | nw.LazyFrame[Any]]:
    """Filter dataset into baseline and current windows using Narwhals expressions."""
    baseline_df = df.filter(nw.col(window_col) == baseline_id)
    current_df = df.filter(nw.col(window_col) == current_id)
    return baseline_df, current_df


class MaterializationBoundary:
    """The explicit boundary between Narwhals expressions and NumPy/SciPy computation."""

    @staticmethod
    def to_eager(df: nw.DataFrame[Any] | nw.LazyFrame[Any]) -> nw.DataFrame[Any]:
        """Materialize lazy frames to eager frames at the computation boundary."""
        if isinstance(df, nw.LazyFrame):
            return df.collect()
        return df

    @classmethod
    def extract_1d_array(
        cls,
        df: nw.DataFrame[Any] | nw.LazyFrame[Any],
        col: str,
        dtype: Any = None,
    ) -> np.ndarray:
        """Extract a single column as a 1D NumPy array with projection pushdown."""
        projected = df.select([col])
        eager_df = cls.to_eager(projected)
        arr = eager_df[col].to_numpy()
        if dtype is not None:
            return arr.astype(dtype)
        return arr

    @classmethod
    def extract_canonical_annotations(
        cls,
        df: nw.DataFrame[Any] | nw.LazyFrame[Any],
        item_col: str = "item_id",
        rater_col: str = "annotator_id",
        label_col: str = "label_id",
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray, dict[str, dict[str, Any]]]:
        """Extract and encode canonical (item, rater, label) annotations to int32 arrays.

        Applies column projection pushdown and null filtering in the native engine
        prior to materializing arrays into NumPy.

        Returns:
            Tuple of (items_idx, raters_idx, labels_idx, id_maps)
            where id_maps contains {"to_int": ..., "to_str": ...} for each entity.
        """
        projected = df.select([item_col, rater_col, label_col]).drop_nulls()
        eager_df = cls.to_eager(projected)

        raw_items = eager_df[item_col].to_numpy().astype(str)
        raw_raters = eager_df[rater_col].to_numpy().astype(str)
        raw_labels = eager_df[label_col].to_numpy().astype(str)

        items_idx, item_to_int, item_to_str = cls._encode_unique(raw_items)
        raters_idx, rater_to_int, rater_to_str = cls._encode_unique(raw_raters)
        labels_idx, label_to_int, label_to_str = cls._encode_unique(raw_labels)

        id_maps: dict[str, dict[str, Any]] = {
            "item": {"to_int": item_to_int, "to_str": item_to_str},
            "annotator": {"to_int": rater_to_int, "to_str": rater_to_str},
            "label": {"to_int": label_to_int, "to_str": label_to_str},
        }

        return items_idx, raters_idx, labels_idx, id_maps

    @staticmethod
    def _encode_unique(arr: np.ndarray) -> tuple[np.ndarray, dict[str, int], dict[int, str]]:
        uniques, inverse = np.unique(arr, return_inverse=True)
        to_int = {str(val): int(idx) for idx, val in enumerate(uniques)}
        to_str = {int(idx): str(val) for idx, val in enumerate(uniques)}
        return inverse.astype(np.int32), to_int, to_str


def compute_item_label_matrix(
    data: Any,
    item_col: str,
    label_col: str,
) -> tuple[np.ndarray, list[str], list[str]]:
    """Aggregate (item, label) counts natively and materialize into a dense (N, K) count matrix.

    This preserves projection and predicate pushdown in Polars/DuckDB without materializing
    millions of raw rows into Python memory.
    """
    df = normalize_to_narwhals(data)

    agg = (
        df.select([item_col, label_col])
        .drop_nulls()
        .group_by([item_col, label_col])
        .agg(count=nw.len())
    )

    eager_agg = MaterializationBoundary.to_eager(agg)
    raw_items = eager_agg[item_col].to_numpy().astype(str)
    raw_labels = eager_agg[label_col].to_numpy().astype(str)
    raw_counts = eager_agg["count"].to_numpy().astype(np.float64)

    unique_items, item_inv = np.unique(raw_items, return_inverse=True)
    unique_labels, label_inv = np.unique(raw_labels, return_inverse=True)

    matrix = np.zeros((len(unique_items), len(unique_labels)), dtype=np.float64)
    matrix[item_inv, label_inv] = raw_counts

    return matrix, list(unique_items), list(unique_labels)
