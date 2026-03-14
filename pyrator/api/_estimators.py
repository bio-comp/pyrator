"""Scikit-learn style estimators for agreement analysis."""

from __future__ import annotations

from typing import Literal

import pandas as pd
import pandera as pa
from pandera.typing import DataFrame

from pyrator.api._results import AgreementResults
from pyrator.api._schemas import AnnotationSchema
from pyrator.ira.krippendorff import KrippendorffAlpha
from pyrator.ira.semantic import SemanticDistanceFactory
from pyrator.ontology.core import Ontology


class KrippendorffEstimator:
    """Estimator for classical and semantic Krippendorff's Alpha."""

    _SUPPORTED_SEMANTIC_METRICS: set[str] = {"path", "lin", "resnik_norm"}

    def __init__(
        self,
        ontology: Ontology | None = None,
        mode: Literal["nominal", "semantic"] = "nominal",
        metric: str = "path",
    ):
        self.ontology = ontology
        self.mode = mode
        self.metric = metric

        if self.mode == "semantic":
            if self.ontology is None:
                raise ValueError("Semantic mode requires an ontology.")
            if self.metric not in self._SUPPORTED_SEMANTIC_METRICS:
                raise ValueError(
                    f"Unsupported metric. Supported: {self._SUPPORTED_SEMANTIC_METRICS}"
                )

    @pa.check_types
    def fit(self, data: DataFrame[AnnotationSchema]) -> AgreementResults:
        """Fit agreement analysis and return structured results."""
        duplicate_counts = data.groupby(["item_id", "annotator_id"]).size()
        if not duplicate_counts[duplicate_counts > 1].empty:
            raise ValueError("Data requires exactly one rating per (item, rater) pair.")

        ka = KrippendorffAlpha(
            data,
            item_col="item_id",
            rater_col="annotator_id",
            label_col="label_id",
        )

        metric_used: str | None = None
        if self.mode == "nominal":
            alpha = ka.calculate(metric="nominal")
        else:
            metric_used = self.metric
            labels = sorted(data["label_id"].unique(), key=lambda x: str(x))
            distance_matrix = SemanticDistanceFactory(
                self.ontology  # type: ignore[arg-type]
            ).compute_distance_matrix(
                labels,
                metric=metric_used,
            )
            alpha = ka.calculate(metric="custom", distance_matrix=distance_matrix)

        consensus = self._build_consensus_labels(data)
        hard_items = self._build_hard_items(data, consensus)
        profiles = self._build_annotator_profiles(data, consensus)

        return AgreementResults(
            alpha=float(alpha),
            mode=self.mode,
            metric=metric_used,
            hard_items=hard_items,
            annotator_profiles=profiles,
            consensus_labels=consensus,
        )

    def _build_consensus_labels(self, data: pd.DataFrame) -> pd.Series:
        consensus = data.groupby("item_id", sort=True)["label_id"].apply(self._deterministic_mode)
        consensus.name = "consensus_label"
        return consensus

    def _build_hard_items(self, data: pd.DataFrame, consensus: pd.Series) -> pd.DataFrame:
        with_consensus = data.merge(
            consensus.rename("consensus_label"),
            left_on="item_id",
            right_index=True,
            how="left",
            validate="many_to_one",
        )
        with_consensus["is_disagreement"] = (
            with_consensus["label_id"] != with_consensus["consensus_label"]
        )

        hard = (
            with_consensus.groupby("item_id", sort=True)
            .agg(
                consensus_label=("consensus_label", "first"),
                n_ratings=("label_id", "size"),
                disagreement_rate=("is_disagreement", "mean"),
            )
            .reset_index()
        )
        return hard.sort_values(
            by=["disagreement_rate", "n_ratings", "item_id"],
            ascending=[False, False, True],
            kind="mergesort",
        ).reset_index(drop=True)

    def _build_annotator_profiles(self, data: pd.DataFrame, consensus: pd.Series) -> pd.DataFrame:
        with_consensus = data.merge(
            consensus.rename("consensus_label"),
            left_on="item_id",
            right_index=True,
            how="left",
            validate="many_to_one",
        )
        with_consensus["is_disagreement"] = (
            with_consensus["label_id"] != with_consensus["consensus_label"]
        )

        profiles = (
            with_consensus.groupby("annotator_id", sort=True)
            .agg(
                n_items=("item_id", "size"),
                disagreement_rate=("is_disagreement", "mean"),
            )
            .reset_index()
        )
        profiles["agreement_rate"] = 1.0 - profiles["disagreement_rate"]
        profiles = profiles[["annotator_id", "n_items", "agreement_rate", "disagreement_rate"]]
        return profiles.sort_values(
            by=["agreement_rate", "annotator_id"],
            ascending=[False, True],
            kind="mergesort",
        ).reset_index(drop=True)

    @staticmethod
    def _deterministic_mode(values: pd.Series) -> object:
        counts = values.value_counts()
        candidates = counts[counts == counts.max()].index.tolist()
        return sorted(candidates, key=lambda x: str(x))[0]
