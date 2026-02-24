"""Tests for the high-level AnnotatorModel facade."""

from __future__ import annotations

import pandas as pd
import pytest

from pyrator.api import AnnotatorModel, ModelResults
from pyrator.ontology.core import Ontology


def test_annotator_model_nominal_fit_with_configurable_columns() -> None:
    """Nominal fit should work with non-canonical column names."""
    df = pd.DataFrame(
        [
            {"item": "i1", "rater": "A", "label": "x"},
            {"item": "i1", "rater": "B", "label": "x"},
            {"item": "i2", "rater": "A", "label": "y"},
            {"item": "i2", "rater": "B", "label": "y"},
            {"item": "i3", "rater": "A", "label": "x"},
            {"item": "i3", "rater": "B", "label": "y"},
        ]
    )

    model = AnnotatorModel(
        mode="nominal",
        item_col="item",
        rater_col="rater",
        label_col="label",
    )
    results = model.fit(df)

    assert isinstance(results, ModelResults)
    assert results.mode == "nominal"
    assert results.metric is None
    assert -1.0 <= results.alpha <= 1.0

    consensus = results.get_consensus_labels()
    assert list(consensus.index) == ["i1", "i2", "i3"]
    assert list(consensus.values) == ["x", "y", "x"]

    hard_items = results.get_hard_items(top_n=2)
    assert list(hard_items["item_id"]) == ["i3", "i1"]
    assert hard_items.iloc[0]["disagreement_rate"] == pytest.approx(0.5, abs=1e-12)

    profiles = results.get_annotator_profiles()
    assert set(profiles["annotator_id"].tolist()) == {"A", "B"}
    agreement_a = profiles.loc[profiles["annotator_id"] == "A", "agreement_rate"].item()
    agreement_b = profiles.loc[profiles["annotator_id"] == "B", "agreement_rate"].item()
    assert agreement_a == pytest.approx(1.0, abs=1e-12)
    assert agreement_b == pytest.approx(2.0 / 3.0, abs=1e-12)


def test_annotator_model_nominal_fit_with_canonical_defaults() -> None:
    """Default canonical columns should work without explicit mapping."""
    df = pd.DataFrame(
        [
            {"item_id": "i1", "annotator_id": "A", "label_id": "yes"},
            {"item_id": "i1", "annotator_id": "B", "label_id": "yes"},
            {"item_id": "i2", "annotator_id": "A", "label_id": "no"},
            {"item_id": "i2", "annotator_id": "B", "label_id": "yes"},
        ]
    )

    results = AnnotatorModel(mode="nominal").fit(df)
    assert isinstance(results, ModelResults)
    assert len(results.get_consensus_labels()) == 2


@pytest.mark.parametrize(
    ("metric", "expected_alpha"),
    [
        ("path", -0.2857142857142856),
        ("lin", -0.19999999999999996),
        ("resnik_norm", 1.0),
    ],
)
def test_annotator_model_semantic_fit_with_supported_metrics(
    simple_ontology: Ontology,
    metric: str,
    expected_alpha: float,
) -> None:
    """Semantic mode should support path/lin/resnik_norm with strict metric validation."""
    df = pd.DataFrame(
        [
            {"item": "i1", "rater": "r1", "label": "A"},
            {"item": "i1", "rater": "r2", "label": "C"},
            {"item": "i2", "rater": "r1", "label": "A"},
            {"item": "i2", "rater": "r2", "label": "B"},
        ]
    )
    model = AnnotatorModel(
        ontology=simple_ontology,
        mode="semantic",
        metric=metric,
        item_col="item",
        rater_col="rater",
        label_col="label",
    )

    results = model.fit(df)
    assert results.mode == "semantic"
    assert results.metric == metric
    assert results.alpha == pytest.approx(expected_alpha, abs=1e-12)


def test_annotator_model_semantic_requires_ontology() -> None:
    """Semantic mode should fail fast if ontology is not provided."""
    df = pd.DataFrame(
        [
            {"item": "i1", "rater": "A", "label": "A"},
            {"item": "i1", "rater": "B", "label": "B"},
        ]
    )
    model = AnnotatorModel(
        mode="semantic",
        metric="path",
        item_col="item",
        rater_col="rater",
        label_col="label",
    )

    with pytest.raises(ValueError, match="requires an ontology"):
        model.fit(df)


def test_annotator_model_rejects_unknown_semantic_metric(simple_ontology: Ontology) -> None:
    """Only path/lin/resnik_norm should be accepted in semantic mode."""
    with pytest.raises(ValueError, match="Unsupported semantic metric"):
        AnnotatorModel(ontology=simple_ontology, mode="semantic", metric="lca")


def test_annotator_model_rejects_missing_required_columns() -> None:
    """Configured item/rater/label columns are required."""
    df = pd.DataFrame(
        [
            {"item": "i1", "rater": "A", "label": "x"},
            {"item": "i1", "rater": "B", "label": "x"},
        ]
    )

    model = AnnotatorModel(mode="nominal")
    with pytest.raises(ValueError, match="Missing required column"):
        model.fit(df)


def test_annotator_model_rejects_duplicate_item_rater_pairs() -> None:
    """Strict mode requires exactly one rating per (item, rater) pair."""
    df = pd.DataFrame(
        [
            {"item": "i1", "rater": "A", "label": "x"},
            {"item": "i1", "rater": "A", "label": "y"},
            {"item": "i1", "rater": "B", "label": "x"},
        ]
    )
    model = AnnotatorModel(
        mode="nominal",
        item_col="item",
        rater_col="rater",
        label_col="label",
    )

    with pytest.raises(ValueError, match="one rating per \\(item, rater\\) pair"):
        model.fit(df)


def test_model_results_get_hard_items_validates_top_n() -> None:
    """ModelResults.get_hard_items should reject non-positive top_n values."""
    df = pd.DataFrame(
        [
            {"item_id": "i1", "annotator_id": "A", "label_id": "x"},
            {"item_id": "i1", "annotator_id": "B", "label_id": "x"},
            {"item_id": "i2", "annotator_id": "A", "label_id": "y"},
            {"item_id": "i2", "annotator_id": "B", "label_id": "x"},
        ]
    )
    results = AnnotatorModel(mode="nominal").fit(df)

    with pytest.raises(ValueError, match="top_n must be positive"):
        results.get_hard_items(top_n=0)
