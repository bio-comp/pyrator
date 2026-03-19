"""Tests for kappa-family agreement metrics."""

import pandas as pd
import pytest

from pyrator.ira.kappa import cohen_kappa, fleiss_kappa


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


def test_fleiss_kappa_nominal_reference_value(fleiss_nominal_data: pd.DataFrame) -> None:
    """Fleiss' kappa should match the known reference value for this fixture."""
    value = fleiss_kappa(
        fleiss_nominal_data,
        item_col="item",
        rater_col="rater",
        label_col="label",
    )

    assert value == pytest.approx(1.0 / 3.0, abs=1e-9)


def test_fleiss_kappa_requires_same_ratings_per_item() -> None:
    """Fleiss' kappa should reject datasets with variable ratings per item."""
    df = pd.DataFrame(
        [
            {"item": "i1", "rater": "A", "label": "x"},
            {"item": "i1", "rater": "B", "label": "x"},
            {"item": "i2", "rater": "A", "label": "y"},
        ]
    )

    with pytest.raises(ValueError, match="same number of ratings per item"):
        fleiss_kappa(df, item_col="item", rater_col="rater", label_col="label")


def test_fleiss_kappa_rejects_duplicate_item_rater_pairs() -> None:
    """Fleiss' kappa requires one rating per (item, rater) pair."""
    df = pd.DataFrame(
        [
            {"item": "i1", "rater": "A", "label": "x"},
            {"item": "i1", "rater": "A", "label": "y"},
            {"item": "i1", "rater": "B", "label": "x"},
            {"item": "i1", "rater": "C", "label": "x"},
        ]
    )

    with pytest.raises(ValueError, match="one rating per"):
        fleiss_kappa(df, item_col="item", rater_col="rater", label_col="label")


def test_fleiss_kappa_autodrop_incomplete_items() -> None:
    """Fleiss' kappa with autodrop should drop items with fewer ratings."""
    df = pd.DataFrame(
        [
            {"item": "i1", "rater": "A", "label": "x"},
            {"item": "i1", "rater": "B", "label": "x"},
            {"item": "i1", "rater": "C", "label": "x"},
            {"item": "i2", "rater": "A", "label": "y"},
            {"item": "i2", "rater": "B", "label": "y"},
        ]
    )

    value = fleiss_kappa(
        df,
        item_col="item",
        rater_col="rater",
        label_col="label",
        autodrop="incomplete_items",
    )
    assert isinstance(value, float)


def test_fleiss_kappa_autodrop_rare_labels() -> None:
    """Fleiss' kappa with autodrop should drop labels appearing in only one item."""
    df = pd.DataFrame(
        [
            {"item": "i1", "rater": "A", "label": "x"},
            {"item": "i1", "rater": "B", "label": "x"},
            {"item": "i1", "rater": "C", "label": "x"},
            {"item": "i2", "rater": "A", "label": "x"},
            {"item": "i2", "rater": "B", "label": "x"},
            {"item": "i2", "rater": "C", "label": "y"},
            {"item": "i3", "rater": "A", "label": "y"},
            {"item": "i3", "rater": "B", "label": "y"},
            {"item": "i3", "rater": "C", "label": "y"},
        ]
    )

    value = fleiss_kappa(
        df,
        item_col="item",
        rater_col="rater",
        label_col="label",
        autodrop="rare_labels",
    )
    assert isinstance(value, float)


def test_fleiss_kappa_autodrop_all() -> None:
    """Fleiss' kappa with autodrop='all' should apply both policies."""
    df = pd.DataFrame(
        [
            {"item": "i1", "rater": "A", "label": "x"},
            {"item": "i1", "rater": "B", "label": "x"},
            {"item": "i1", "rater": "C", "label": "y"},
            {"item": "i2", "rater": "A", "label": "x"},
            {"item": "i2", "rater": "B", "label": "y"},
            {"item": "i2", "rater": "C", "label": "y"},
        ]
    )

    value = fleiss_kappa(
        df,
        item_col="item",
        rater_col="rater",
        label_col="label",
        autodrop="all",
    )
    assert isinstance(value, float)


def test_fleiss_kappa_autodrop_empty_after_drop() -> None:
    """Fleiss' kappa should raise if autodrop removes all data."""
    df = pd.DataFrame(
        [
            {"item": "i1", "rater": "A", "label": "x"},
            {"item": "i2", "rater": "A", "label": "y"},
        ]
    )

    with pytest.raises(ValueError, match="[Nn]o items"):
        fleiss_kappa(
            df,
            item_col="item",
            rater_col="rater",
            label_col="label",
            autodrop="all",
        )
