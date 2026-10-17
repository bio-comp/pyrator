import numpy as np
import pandas as pd

from pyrator.ira.krippendorff import KrippendorffAlpha
from pyrator.ira.semantic import SemanticDistanceFactory


def test_krippendorff_nominal_canonical(krippendorff_data):
    """Test against Krippendorff's standard example (Alpha ~ 0.148)."""
    ka = KrippendorffAlpha(
        krippendorff_data,
        item_col="item",
        rater_col="rater",
        label_col="label",
    )
    alpha = ka.calculate(metric="nominal")

    assert 0.14 < alpha < 0.16


def test_krippendorff_interval_simple():
    """Test interval metric on perfect agreement (Alpha = 1.0)."""
    df = pd.DataFrame(
        {
            "item": ["i1", "i2", "i3", "i1", "i2", "i3"],
            "rater": ["A", "A", "A", "B", "B", "B"],
            "label": [1, 2, 3, 1, 2, 3],
        }
    )
    ka = KrippendorffAlpha(df, "item", "rater", "label")
    assert ka.calculate(metric="interval") == 1.0


def test_semantic_distance_matrix(simple_ontology):
    """Verify the distance matrix generation logic."""
    factory = SemanticDistanceFactory(simple_ontology)
    labels = ["A", "B", "C"]

    matrix = factory.compute_distance_matrix(labels, metric="path")

    expected = np.array(
        [
            [0.0, 2.0, 1.0],
            [2.0, 0.0, 1.0],
            [1.0, 1.0, 0.0],
        ]
    )

    np.testing.assert_array_equal(matrix, expected)


def test_semantic_alpha_integration(simple_ontology):
    """End-to-end test: Calculate Semantic Alpha using the Diamond graph."""
    df = pd.DataFrame(
        [
            {"item": "i1", "rater": "r1", "label": "A"},
            {"item": "i1", "rater": "r2", "label": "C"},
            {"item": "i2", "rater": "r1", "label": "A"},
            {"item": "i2", "rater": "r2", "label": "B"},
        ]
    )

    factory = SemanticDistanceFactory(simple_ontology)
    dist_matrix = factory.compute_distance_matrix(["A", "B", "C"], metric="path")

    ka = KrippendorffAlpha(df, "item", "rater", "label")
    alpha = ka.calculate(metric="custom", distance_matrix=dist_matrix)

    assert -1.0 <= alpha <= 1.0


def test_krippendorff_polars_lazy_and_arrow(krippendorff_data: pd.DataFrame) -> None:
    """Verify KrippendorffAlpha directly accepts Polars LazyFrame and PyArrow Table."""
    import pytest

    pl = pytest.importorskip("polars")
    pa = pytest.importorskip("pyarrow")

    # Baseline with pandas
    ka_pd = KrippendorffAlpha(
        krippendorff_data, item_col="item", rater_col="rater", label_col="label"
    )
    alpha_pd = ka_pd.calculate(metric="nominal")

    # Polars LazyFrame
    lazy_df = pl.DataFrame(krippendorff_data).lazy()
    ka_pl = KrippendorffAlpha(lazy_df, item_col="item", rater_col="rater", label_col="label")
    alpha_pl = ka_pl.calculate(metric="nominal")
    assert np.isclose(alpha_pl, alpha_pd)

    # PyArrow Table
    tbl = pa.Table.from_pandas(krippendorff_data)
    ka_pa = KrippendorffAlpha(tbl, item_col="item", rater_col="rater", label_col="label")
    alpha_pa = ka_pa.calculate(metric="nominal")
    assert np.isclose(alpha_pa, alpha_pd)


def test_krippendorff_duckdb_and_ibis(krippendorff_data: pd.DataFrame) -> None:
    """Verify KrippendorffAlpha directly accepts native DuckDB relation and Ibis table."""
    import pytest

    duckdb = pytest.importorskip("duckdb")
    ibis = pytest.importorskip("ibis")

    # Baseline with pandas
    ka_pd = KrippendorffAlpha(
        krippendorff_data, item_col="item", rater_col="rater", label_col="label"
    )
    alpha_pd = ka_pd.calculate(metric="nominal")

    # Native DuckDB relation
    duck_rel = duckdb.from_df(krippendorff_data)
    ka_duck = KrippendorffAlpha(duck_rel, item_col="item", rater_col="rater", label_col="label")
    alpha_duck = ka_duck.calculate(metric="nominal")
    assert np.isclose(alpha_duck, alpha_pd)

    # Native Ibis table
    ibis_tbl = ibis.memtable(krippendorff_data)
    ka_ibis = KrippendorffAlpha(ibis_tbl, item_col="item", rater_col="rater", label_col="label")
    alpha_ibis = ka_ibis.calculate(metric="nominal")
    assert np.isclose(alpha_ibis, alpha_pd)
