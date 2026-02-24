"""Tests for kappa-family agreement metrics."""

import pandas as pd
import pytest

from pyrator.ira.kappa import cohen_kappa


def test_cohen_kappa_nominal_reference_value(cohen_nominal_data: pd.DataFrame) -> None:
    """Cohen's kappa should match the known reference value for this fixture."""
    value = cohen_kappa(
        cohen_nominal_data,
        item_col="item",
        rater_col="rater",
        label_col="label",
    )

    assert value == pytest.approx(0.4, abs=1e-9)


def test_cohen_kappa_requires_exactly_two_raters() -> None:
    """Cohen's kappa must reject datasets with more than two raters."""
    df = pd.DataFrame(
        [
            {"item": "i1", "rater": "A", "label": "x"},
            {"item": "i1", "rater": "B", "label": "x"},
            {"item": "i1", "rater": "C", "label": "y"},
        ]
    )

    with pytest.raises(ValueError, match="exactly two raters"):
        cohen_kappa(df, item_col="item", rater_col="rater", label_col="label")


def test_cohen_kappa_rejects_missing_ratings_per_item() -> None:
    """Cohen's kappa requires both raters to annotate each item."""
    df = pd.DataFrame(
        [
            {"item": "i1", "rater": "A", "label": "x"},
            {"item": "i1", "rater": "B", "label": "x"},
            {"item": "i2", "rater": "A", "label": "y"},
        ]
    )

    with pytest.raises(ValueError, match="missing ratings"):
        cohen_kappa(df, item_col="item", rater_col="rater", label_col="label")
