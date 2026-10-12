"""Closed-form tree and DAG consensus estimators."""

from __future__ import annotations

from collections import Counter
from typing import Any, Literal

import numpy as np

from pyrator.consensus.result import ConsensusResult
from pyrator.data.pipeline import (
    MaterializationBoundary,
    detect_backend_affinity,
    normalize_to_narwhals,
)
from pyrator.ontology.core import Ontology


class TreeConsensusEstimator:
    """Closed-form consensus estimator operating on ontology graphs (trees or DAGs).

    Supports LCA-based generalization, distance-minimizing medoid consensus, and
    majority voting with ontology tie-breaking.
    """

    def __init__(
        self,
        ontology: Ontology,
        strategy: Literal["lca", "medoid", "majority"] = "lca",
        metric: str = "path",
    ) -> None:
        self.ontology = ontology
        self.strategy = strategy
        self.metric = metric

    def fit_predict(
        self,
        data: Any,
        item_col: str = "item_id",
        rater_col: str = "annotator_id",
        label_col: str = "label_id",
    ) -> ConsensusResult:
        """Compute consensus labels for items in the dataset."""
        affinity = detect_backend_affinity(data, default="polars")
        nw_df = normalize_to_narwhals(data)
        projected = nw_df.select([item_col, label_col]).drop_nulls()
        eager_df = MaterializationBoundary.to_eager(projected)

        raw_items = eager_df[item_col].to_numpy().astype(str)
        raw_labels = eager_df[label_col].to_numpy().astype(str)

        # Group labels by item preserving order of appearance
        item_to_labels: dict[str, list[str]] = {}
        for item, label in zip(raw_items, raw_labels):
            item_to_labels.setdefault(item, []).append(label)

        unique_items = list(item_to_labels.keys())
        consensus_labels: list[str] = []
        confidences: list[float] = []

        for item in unique_items:
            labels = item_to_labels[item]
            cons_label, conf = self._compute_item_consensus(labels)
            consensus_labels.append(cons_label)
            confidences.append(conf)

        all_classes = sorted(list(set(raw_labels).union(consensus_labels)))
        class_to_idx = {c: i for i, c in enumerate(all_classes)}

        # Build class probabilities (1.0 at consensus label for deterministic estimator)
        n_items = len(unique_items)
        n_classes = len(all_classes)
        probs = np.zeros((n_items, n_classes), dtype=np.float64)
        for i, c in enumerate(consensus_labels):
            probs[i, class_to_idx[c]] = confidences[i]
            # Normalize row
            row_sum = probs[i].sum()
            if row_sum > 0:
                probs[i] /= row_sum

        return ConsensusResult(
            item_ids=unique_items,
            consensus_labels=consensus_labels,
            confidences=np.array(confidences, dtype=np.float64),
            class_probabilities=probs,
            classes=all_classes,
            converged=True,
            iterations=1,
            log_likelihood=0.0,
            backend_affinity=affinity,
        )

    def _compute_item_consensus(self, labels: list[str]) -> tuple[str, float]:
        """Compute consensus label and confidence for a single item's label set."""
        if not labels:
            raise ValueError("Cannot compute consensus for empty label set.")

        # Check if all labels are identical
        if len(set(labels)) == 1:
            return labels[0], 1.0

        if self.strategy == "lca":
            return self._lca_consensus(labels)
        if self.strategy == "medoid":
            return self._medoid_consensus(labels)
        if self.strategy == "majority":
            return self._majority_consensus(labels)

        raise ValueError(f"Unknown strategy: {self.strategy}")

    def _lca_consensus(self, labels: list[str]) -> tuple[str, float]:
        """Compute iterative Lowest Common Ancestor across all labels."""
        curr = labels[0]
        for nxt in labels[1:]:
            lca = self.ontology.get_lca(curr, nxt)
            if lca is not None:
                curr = lca
            else:
                curr = "Root"

        # Confidence: proportion of raters whose label is descended from or equals curr
        matches = 0
        for lbl in labels:
            if lbl == curr or self.ontology.is_ancestor(curr, lbl):
                matches += 1

        conf = matches / len(labels)
        return curr, conf

    def _medoid_consensus(self, labels: list[str]) -> tuple[str, float]:
        """Find candidate node minimizing total ontology distance to all annotations."""
        unique_labels = list(set(labels))
        counts = Counter(labels)

        best_candidate = unique_labels[0]
        best_cost = float("inf")

        for cand in unique_labels:
            cost = sum(self.ontology.get_distance(cand, lbl, metric=self.metric) for lbl in labels)
            # Tie-break: highest frequency, then lowest lexical
            if cost < best_cost or (cost == best_cost and counts[cand] > counts[best_candidate]):
                best_cost = cost
                best_candidate = cand

        mean_dist = best_cost / len(labels)
        conf = float(np.clip(1.0 - (mean_dist / 4.0), 0.1, 1.0))
        return best_candidate, conf

    def _majority_consensus(self, labels: list[str]) -> tuple[str, float]:
        """Plurality vote with LCA tie-breaking."""
        counts = Counter(labels)
        most_common = counts.most_common()
        top_count = most_common[0][1]
        tied_candidates = [label for label, count in most_common if count == top_count]

        if len(tied_candidates) == 1:
            return tied_candidates[0], top_count / len(labels)

        # Break tie using LCA
        lca, _ = self._lca_consensus(tied_candidates)
        return lca, top_count / len(labels)
