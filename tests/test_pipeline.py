"""Tests for the Narwhals expression layer and Materialization Boundary."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from pyrator.data.pipeline import (
    MaterializationBoundary,
    detect_backend_affinity,
    filter_windows,
    normalize_to_narwhals,
)


def test_normalize_pandas_dataframe() -> None:
    df = pd.DataFrame({"a": [1, 2, 3], "b": ["x", "y", "z"]})
    nw_df = normalize_to_narwhals(df)
    assert hasattr(nw_df, "columns")
    assert nw_df.columns == ["a", "b"]
    assert detect_backend_affinity(df) == "pandas"


def test_normalize_polars_eager_and_lazy() -> None:
    pl = pytest.importorskip("polars")

    # Eager
    df_eager = pl.DataFrame({"a": [10, 20], "b": ["cat", "dog"]})
    nw_eager = normalize_to_narwhals(df_eager)
    assert nw_eager.columns == ["a", "b"]
    assert detect_backend_affinity(df_eager) == "polars"

    # Lazy: verify it preserves laziness
    df_lazy = df_eager.lazy()
    nw_lazy = normalize_to_narwhals(df_lazy)
    assert hasattr(nw_lazy, "collect")
    assert detect_backend_affinity(df_lazy) == "polars"


def test_normalize_pyarrow_table() -> None:
    pa = pytest.importorskip("pyarrow")
    tbl = pa.Table.from_pydict({"a": [1, 2], "b": ["alpha", "beta"]})
    nw_tbl = normalize_to_narwhals(tbl)
    assert nw_tbl.columns == ["a", "b"]
    assert detect_backend_affinity(tbl) == "arrow"


def test_filter_windows() -> None:
    df = pd.DataFrame(
        {
            "window_id": ["W1", "W1", "W2", "W2", "W3"],
            "val": [1.0, 2.0, 10.0, 20.0, 99.0],
        }
    )
    nw_df = normalize_to_narwhals(df)
    base_nw, curr_nw = filter_windows(
        nw_df, window_col="window_id", baseline_id="W1", current_id="W2"
    )

    base_vals = MaterializationBoundary.extract_1d_array(base_nw, "val")
    curr_vals = MaterializationBoundary.extract_1d_array(curr_nw, "val")

    np.testing.assert_array_equal(base_vals, [1.0, 2.0])
    np.testing.assert_array_equal(curr_vals, [10.0, 20.0])


def test_extract_canonical_annotations() -> None:
    df = pd.DataFrame(
        {
            "item_id": ["item_1", "item_1", "item_2", "item_2"],
            "annotator_id": ["rater_A", "rater_B", "rater_A", "rater_B"],
            "label_id": ["cat", "dog", "dog", "dog"],
        }
    )
    nw_df = normalize_to_narwhals(df)

    items, raters, labels, id_maps = MaterializationBoundary.extract_canonical_annotations(
        nw_df, item_col="item_id", rater_col="annotator_id", label_col="label_id"
    )

    assert items.dtype == np.int32
    assert raters.dtype == np.int32
    assert labels.dtype == np.int32
    assert len(items) == 4
    assert len(raters) == 4
    assert len(labels) == 4

    # Verify ID map invertibility
    assert "item" in id_maps
    assert "annotator" in id_maps
    assert "label" in id_maps
    assert id_maps["label"]["to_int"]["cat"] == labels[0]
    assert id_maps["label"]["to_str"][labels[0]] == "cat"


def test_compute_item_label_matrix_pandas_and_polars_lazy() -> None:
    from pyrator.data.pipeline import compute_item_label_matrix

    data = {
        "item_id": ["i1", "i1", "i1", "i2", "i2"],
        "annotator_id": ["r1", "r2", "r3", "r1", "r2"],
        "label_id": ["A", "A", "B", "B", "B"],
    }
    df_pd = pd.DataFrame(data)

    mat_pd, items_pd, labels_pd = compute_item_label_matrix(
        df_pd, item_col="item_id", label_col="label_id"
    )

    assert items_pd == ["i1", "i2"]
    assert labels_pd == ["A", "B"]
    # i1: 2 'A', 1 'B'
    # i2: 0 'A', 2 'B'
    np.testing.assert_array_equal(mat_pd, [[2.0, 1.0], [0.0, 2.0]])

    # Polars LazyFrame test: verify lazy pre-aggregation works directly
    pl = pytest.importorskip("polars")
    df_lazy = pl.DataFrame(data).lazy()
    mat_pl, items_pl, labels_pl = compute_item_label_matrix(
        df_lazy, item_col="item_id", label_col="label_id"
    )

    assert items_pl == ["i1", "i2"]
    assert labels_pl == ["A", "B"]
    np.testing.assert_array_equal(mat_pl, [[2.0, 1.0], [0.0, 2.0]])

