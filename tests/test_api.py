"""Tests for the high-level AnnotatorModel facade."""

from __future__ import annotations

import pandas as pd
import pandera as pa
import pytest

from pyrator.api import AgreementResults, KrippendorffEstimator
from pyrator.ontology.core import Ontology


def test_krippendorff_estimator_nominal_fit_with_canonical_columns() -> None:
    """Nominal fit should work with canonical column names."""
    df = pd.DataFrame(
        [
            {"item_id": "i1", "annotator_id": "A", "label_id": "x"},
            {"item_id": "i1", "annotator_id": "B", "label_id": "x"},
            {"item_id": "i2", "annotator_id": "A", "label_id": "y"},
            {"item_id": "i2", "annotator_id": "B", "label_id": "y"},
            {"item_id": "i3", "annotator_id": "A", "label_id": "x"},
            {"item_id": "i3", "annotator_id": "B", "label_id": "y"},
        ]
    )

    estimator = KrippendorffEstimator(mode="nominal")
    results = estimator.fit(df)

    assert isinstance(results, AgreementResults)
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


def test_krippendorff_estimator_with_default_mode() -> None:
    """Default mode should be nominal without explicit setting."""
    df = pd.DataFrame(
        [
            {"item_id": "i1", "annotator_id": "A", "label_id": "yes"},
            {"item_id": "i1", "annotator_id": "B", "label_id": "yes"},
            {"item_id": "i2", "annotator_id": "A", "label_id": "no"},
            {"item_id": "i2", "annotator_id": "B", "label_id": "yes"},
        ]
    )

    estimator = KrippendorffEstimator()
    results = estimator.fit(df)
    assert isinstance(results, AgreementResults)
    assert len(results.get_consensus_labels()) == 2


@pytest.mark.parametrize(
    ("metric", "expected_alpha"),
    [
        ("path", -0.2857142857142856),
        ("lin", -0.19999999999999996),
        ("resnik_norm", 1.0),
    ],
)
def test_krippendorff_estimator_semantic_fit_with_supported_metrics(
    simple_ontology: Ontology,
    metric: str,
    expected_alpha: float,
) -> None:
    """Semantic mode should support path/lin/resnik_norm with strict metric validation."""
    df = pd.DataFrame(
        [
            {"item_id": "i1", "annotator_id": "r1", "label_id": "A"},
            {"item_id": "i1", "annotator_id": "r2", "label_id": "C"},
            {"item_id": "i2", "annotator_id": "r1", "label_id": "A"},
            {"item_id": "i2", "annotator_id": "r2", "label_id": "B"},
        ]
    )
    estimator = KrippendorffEstimator(
        ontology=simple_ontology,
        mode="semantic",
        metric=metric,
    )

    results = estimator.fit(df)
    assert results.mode == "semantic"
    assert results.metric == metric
    assert results.alpha == pytest.approx(expected_alpha, abs=1e-12)


def test_krippendorff_estimator_semantic_requires_ontology() -> None:
    """Semantic mode should fail fast if ontology is not provided."""
    with pytest.raises(ValueError, match="requires an ontology"):
        KrippendorffEstimator(
            mode="semantic",
            metric="path",
        )


def test_krippendorff_estimator_rejects_unknown_semantic_metric(simple_ontology: Ontology) -> None:
    """Only path/lin/resnik_norm should be accepted in semantic mode."""
    with pytest.raises(ValueError, match="Unsupported metric"):
        KrippendorffEstimator(ontology=simple_ontology, mode="semantic", metric="lca")


def test_krippendorff_estimator_rejects_missing_required_columns() -> None:
    """Canonical item_id/annotator_id/label_id columns are required."""
    df = pd.DataFrame(
        [
            {"item": "i1", "rater": "A", "label": "x"},
            {"item": "i1", "rater": "B", "label": "x"},
        ]
    )

    estimator = KrippendorffEstimator(mode="nominal")
    with pytest.raises(pa.errors.SchemaError):
        estimator.fit(df)


def test_krippendorff_estimator_rejects_duplicate_item_rater_pairs() -> None:
    """Strict mode requires exactly one rating per (item, rater) pair."""
    df = pd.DataFrame(
        [
            {"item_id": "i1", "annotator_id": "A", "label_id": "x"},
            {"item_id": "i1", "annotator_id": "A", "label_id": "y"},
            {"item_id": "i1", "annotator_id": "B", "label_id": "x"},
        ]
    )
    estimator = KrippendorffEstimator(mode="nominal")

    with pytest.raises(ValueError, match="exactly one rating per"):
        estimator.fit(df)


def test_agreement_results_get_hard_items_validates_top_n() -> None:
    """AgreementResults.get_hard_items should reject non-positive top_n values."""
    df = pd.DataFrame(
        [
            {"item_id": "i1", "annotator_id": "A", "label_id": "x"},
            {"item_id": "i1", "annotator_id": "B", "label_id": "x"},
            {"item_id": "i2", "annotator_id": "A", "label_id": "y"},
            {"item_id": "i2", "annotator_id": "B", "label_id": "x"},
        ]
    )
    estimator = KrippendorffEstimator(mode="nominal")
    results = estimator.fit(df)

    with pytest.raises(ValueError, match="top_n must be positive"):
        results.get_hard_items(top_n=0)
