"""Tests for Multi-Tier Consensus Engine (Tree Consensus and Dawid-Skene with Hierarchy Prior)."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from pyrator.consensus import (
    ConsensusResult,
    DawidSkeneEstimator,
    TreeConsensusEstimator,
)
from pyrator.ontology.core import Ontology


@pytest.fixture
def clinical_ontology() -> Ontology:
    """Medical guideline hierarchy for testing consensus.

           Root
         /      \\
      Cardio     Neuro
      /    \\     /    \\
    CAD   Arrh  Stroke Migraine
    """
    nodes = {
        "Root": {"depth": 0},
        "Cardio": {"depth": 1},
        "Neuro": {"depth": 1},
        "CAD": {"depth": 2},
        "Arrh": {"depth": 2},
        "Stroke": {"depth": 2},
        "Migraine": {"depth": 2},
    }
    edges = [
        ("Root", "Cardio"),
        ("Root", "Neuro"),
        ("Cardio", "CAD"),
        ("Cardio", "Arrh"),
        ("Neuro", "Stroke"),
        ("Neuro", "Migraine"),
    ]
    return Ontology.build("clinical_v1", nodes=nodes, edges=edges)


def test_tree_consensus_lca_and_medoid(clinical_ontology: Ontology) -> None:
    # Item 1: Raters agree on CAD -> CAD
    # Item 2: CAD vs Arrh -> LCA is Cardio
    # Item 3: CAD vs Stroke -> LCA is Root
    df = pd.DataFrame(
        {
            "item_id": ["i1", "i1", "i2", "i2", "i3", "i3"],
            "annotator_id": ["r1", "r2", "r1", "r2", "r1", "r2"],
            "label_id": ["CAD", "CAD", "CAD", "Arrh", "CAD", "Stroke"],
        }
    )

    estimator_lca = TreeConsensusEstimator(
        ontology=clinical_ontology,
        strategy="lca",
    )
    res_lca = estimator_lca.fit_predict(df)

    assert isinstance(res_lca, ConsensusResult)
    preds = dict(zip(res_lca.item_ids, res_lca.consensus_labels))
    assert preds["i1"] == "CAD"
    assert preds["i2"] == "Cardio"
    assert preds["i3"] == "Root"

    # Medoid strategy: within label set or full tree
    estimator_medoid = TreeConsensusEstimator(
        ontology=clinical_ontology,
        strategy="medoid",
        metric="path",
    )
    res_medoid = estimator_medoid.fit_predict(df)
    preds_medoid = dict(zip(res_medoid.item_ids, res_medoid.consensus_labels))
    assert preds_medoid["i1"] == "CAD"


def test_dawid_skene_closed_form_and_em(clinical_ontology: Ontology) -> None:
    # 3 items, 3 raters.
    # rater 1 and 2 are reliable. rater 3 is biased/noisy.
    df = pd.DataFrame(
        {
            "item_id": ["i1", "i1", "i1", "i2", "i2", "i2", "i3", "i3", "i3"],
            "annotator_id": ["r1", "r2", "r3", "r1", "r2", "r3", "r1", "r2", "r3"],
            "label_id": [
                "CAD",
                "CAD",
                "Arrh",  # r3 made nearby mistake
                "Stroke",
                "Stroke",
                "Migraine",  # r3 made nearby mistake
                "CAD",
                "CAD",
                "CAD",
            ],
        }
    )

    ds = DawidSkeneEstimator(
        ontology=clinical_ontology,
        decay_metric="path",
        decay_lambda=0.5,
        max_iter=50,
        tol=1e-5,
    )
    res = ds.fit(df)

    assert isinstance(res, ConsensusResult)
    assert res.converged
    assert res.iterations > 0
    assert len(res.consensus_labels) == 3

    preds = dict(zip(res.item_ids, res.consensus_labels))
    assert preds["i1"] == "CAD"
    assert preds["i2"] == "Stroke"
    assert preds["i3"] == "CAD"

    # Confidences must be between 0 and 1
    assert all(0.0 <= conf <= 1.0 for conf in res.confidences)

    # Annotator confusion matrices must be normalized probabilities
    for rater, conf_mat in res.annotator_confusion.items():
        assert conf_mat.shape == (len(res.classes), len(res.classes))
        np.testing.assert_allclose(conf_mat.sum(axis=1), np.ones(len(res.classes)), atol=1e-5)


def test_consensus_result_exports() -> None:
    res = ConsensusResult(
        item_ids=["item_a", "item_b"],
        consensus_labels=["CAD", "Stroke"],
        confidences=np.array([0.95, 0.88]),
        class_probabilities=np.array([[0.95, 0.05], [0.12, 0.88]]),
        classes=["CAD", "Stroke"],
        annotator_confusion={"r1": np.eye(2)},
        converged=True,
        iterations=5,
        log_likelihood=-2.41,
        backend_affinity="pandas",
    )

    # Pandas export
    df_pd = res.to_pandas()
    assert isinstance(df_pd, pd.DataFrame)
    assert list(df_pd.columns) == ["item_id", "consensus_label", "confidence"]
    assert len(df_pd) == 2

    # Polars export (if available)
    pl = pytest.importorskip("polars")
    df_pl = res.to_polars()
    assert isinstance(df_pl, pl.DataFrame)
    assert df_pl.shape == (2, 3)

    # Dict export
    d = res.to_dict()
    assert d["converged"] is True
    assert d["iterations"] == 5


def test_dawid_skene_polars_native_input(clinical_ontology: Ontology) -> None:
    pl = pytest.importorskip("polars")

    df_pl = pl.DataFrame(
        {
            "item_id": ["i1", "i1", "i2", "i2"],
            "annotator_id": ["r1", "r2", "r1", "r2"],
            "label_id": ["CAD", "CAD", "Stroke", "Stroke"],
        }
    )

    ds = DawidSkeneEstimator(ontology=clinical_ontology)
    res = ds.fit(df_pl)

    assert res.backend_affinity == "polars"
    df_out = res.to_polars()
    assert isinstance(df_out, pl.DataFrame)
    assert df_out["consensus_label"].to_list() == ["CAD", "Stroke"]
