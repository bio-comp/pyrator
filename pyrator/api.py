"""High-level user facade for agreement analysis."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal, cast

import pandas as pd

from pyrator.ira.krippendorff import KrippendorffAlpha
from pyrator.ira.semantic import SemanticDistanceFactory
from pyrator.ontology.core import Ontology
from pyrator.types import FrameLike

AgreementMode = Literal["nominal", "semantic"]
SemanticMetric = Literal["path", "lin", "resnik_norm"]


@dataclass(slots=True)
class ModelResults:
    """Structured results produced by ``AnnotatorModel.fit``."""

    alpha: float
    mode: AgreementMode
    metric: SemanticMetric | None
    _hard_items: pd.DataFrame
    _annotator_profiles: pd.DataFrame
    _consensus_labels: pd.Series

    def get_hard_items(self, top_n: int = 10) -> pd.DataFrame:
        """Return the highest-disagreement items."""
        if top_n <= 0:
            raise ValueError("top_n must be positive.")
        return self._hard_items.head(top_n).copy()

    def get_annotator_profiles(self) -> pd.DataFrame:
        """Return per-annotator agreement summaries."""
        return self._annotator_profiles.copy()

    def get_consensus_labels(self) -> pd.Series:
        """Return deterministic item-level consensus labels."""
        return self._consensus_labels.copy()


class AnnotatorModel:
    """Thin facade around current Krippendorff and ontology-semantic components."""

    _SUPPORTED_SEMANTIC_METRICS: set[str] = {"path", "lin", "resnik_norm"}

    def __init__(
        self,
        ontology: Ontology | None = None,
        *,
        strategy: str = "auto",
        mode: str = "nominal",
        metric: str = "path",
        random_state: int | None = 0,
        item_col: str = "item_id",
        rater_col: str = "annotator_id",
        label_col: str = "label_id",
    ):
        self.ontology = ontology
        self.strategy = strategy
        self.mode = self._validate_mode(mode)
        self.metric = self._validate_semantic_metric(metric)
        self.random_state = random_state
        self.item_col = item_col
        self.rater_col = rater_col
        self.label_col = label_col

    def fit(self, data: FrameLike) -> ModelResults:
        """Fit agreement analysis and return structured results."""
        normalized = self._normalize_input_frame(self._to_pandas_frame(data))

        ka = KrippendorffAlpha(
            normalized,
            item_col="item_id",
            rater_col="annotator_id",
            label_col="label_id",
        )

        metric_used: SemanticMetric | None = None
        if self.mode == "nominal":
            alpha = ka.calculate(metric="nominal")
        else:
            if self.ontology is None:
                raise ValueError("Semantic mode requires an ontology.")
            metric_used = self.metric
            labels = sorted(normalized["label_id"].unique(), key=lambda value: str(value))
            distance_matrix = SemanticDistanceFactory(self.ontology).compute_distance_matrix(
                labels,
                metric=metric_used,
            )
            alpha = ka.calculate(metric="custom", distance_matrix=distance_matrix)

        consensus = self._build_consensus_labels(normalized)
        hard_items = self._build_hard_items(normalized, consensus)
        profiles = self._build_annotator_profiles(normalized, consensus)

        return ModelResults(
            alpha=float(alpha),
            mode=self.mode,
            metric=metric_used,
            _hard_items=hard_items,
            _annotator_profiles=profiles,
            _consensus_labels=consensus,
        )

    def _normalize_input_frame(self, frame: pd.DataFrame) -> pd.DataFrame:
        """Validate and normalize input columns to canonical internal names."""
        for col in (self.item_col, self.rater_col, self.label_col):
            if col not in frame.columns:
                raise ValueError(f"Missing required column: {col}")

        normalized = frame[[self.item_col, self.rater_col, self.label_col]].rename(
            columns={
                self.item_col: "item_id",
                self.rater_col: "annotator_id",
                self.label_col: "label_id",
            }
        )

        if normalized.empty:
            raise ValueError("AnnotatorModel.fit requires at least one annotation row.")

        if normalized[["item_id", "annotator_id", "label_id"]].isna().any().any():
            raise ValueError("Input contains nulls in required item/rater/label columns.")

        duplicate_counts = normalized.groupby(["item_id", "annotator_id"]).size()
        duplicate_pairs = duplicate_counts[duplicate_counts > 1]
        if not duplicate_pairs.empty:
            raise ValueError("AnnotatorModel.fit requires one rating per (item, rater) pair.")

        return normalized

    def _to_pandas_frame(self, data: FrameLike) -> pd.DataFrame:
        """Convert supported frame-like inputs to pandas DataFrame."""
        frame = data.to_pandas() if hasattr(data, "to_pandas") else data
        if not isinstance(frame, pd.DataFrame):
            raise ValueError("AnnotatorModel.fit expects a pandas-compatible frame-like input.")
        return frame

    def _validate_mode(self, mode: str) -> AgreementMode:
        if mode not in {"nominal", "semantic"}:
            raise ValueError("Unsupported mode. Supported modes: nominal, semantic.")
        return cast("AgreementMode", mode)

    def _validate_semantic_metric(self, metric: str) -> SemanticMetric:
        if metric not in self._SUPPORTED_SEMANTIC_METRICS:
            raise ValueError(
                "Unsupported semantic metric. Supported metrics: path, lin, resnik_norm."
            )
        return cast("SemanticMetric", metric)

    def _build_consensus_labels(self, normalized: pd.DataFrame) -> pd.Series:
        """Compute deterministic item-level consensus labels."""
        consensus = normalized.groupby("item_id", sort=True)["label_id"].apply(
            self._deterministic_mode
        )
        consensus.name = "consensus_label"
        return consensus

    def _build_hard_items(self, normalized: pd.DataFrame, consensus: pd.Series) -> pd.DataFrame:
        """Compute item hardness using disagreement rate with deterministic ordering."""
        with_consensus = normalized.merge(
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
        hard = hard.sort_values(
            by=["disagreement_rate", "n_ratings", "item_id"],
            ascending=[False, False, True],
            kind="mergesort",
        ).reset_index(drop=True)
        return hard

    def _build_annotator_profiles(
        self,
        normalized: pd.DataFrame,
        consensus: pd.Series,
    ) -> pd.DataFrame:
        """Compute per-annotator agreement profiles against consensus labels."""
        with_consensus = normalized.merge(
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
        profiles = profiles.sort_values(
            by=["agreement_rate", "annotator_id"],
            ascending=[False, True],
            kind="mergesort",
        ).reset_index(drop=True)
        return profiles

    @staticmethod
    def _deterministic_mode(values: pd.Series) -> object:
        """Return mode with string-ordered tie-break for deterministic outputs."""
        counts = values.value_counts()
        max_count = counts.max()
        candidates = counts[counts == max_count].index.tolist()
        return sorted(candidates, key=lambda value: str(value))[0]
