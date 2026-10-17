"""Dawid-Skene consensus estimator with hierarchy-aware decay prior."""

from __future__ import annotations

from typing import Any, Literal

import numpy as np

from pyrator.consensus.result import ConsensusResult
from pyrator.data.pipeline import (
    MaterializationBoundary,
    normalize_to_narwhals,
    resolve_affinity,
)
from pyrator.ontology.core import Ontology


class DawidSkeneEstimator:
    """Expectation-Maximization Dawid-Skene model with ontology-aware Dirichlet prior.

    Instead of penalizing all off-diagonal misclassifications equally, this estimator
    incorporates an exponential decay prior based on ontology distances:
        alpha_{k, l} = beta_diag if k == l else beta_off * exp(-lambda * d(k, l))
    which allows the model to differentiate between subtle near-ancestor disagreements
    versus catastrophic cross-branch confusion.
    """

    def __init__(
        self,
        ontology: Ontology | None = None,
        decay_metric: Literal["path", "resnik", "lin"] = "path",
        decay_lambda: float = 1.0,
        beta_diag: float = 5.0,
        beta_off: float = 1.0,
        max_iter: int = 100,
        tol: float = 1e-5,
    ) -> None:
        self.ontology = ontology
        self.decay_metric = decay_metric
        self.decay_lambda = decay_lambda
        self.beta_diag = beta_diag
        self.beta_off = beta_off
        self.max_iter = max_iter
        self.tol = tol

    def fit(
        self,
        data: Any,
        item_col: str = "item_id",
        rater_col: str = "annotator_id",
        label_col: str = "label_id",
    ) -> ConsensusResult:
        """Fit Dawid-Skene model via EM and return structured consensus results."""
        nw_df = normalize_to_narwhals(data)
        affinity = resolve_affinity(data, nw_df, default="polars")

        # Extract at Materialization Boundary
        items_idx, raters_idx, labels_idx, id_maps = (
            MaterializationBoundary.extract_canonical_annotations(
                nw_df, item_col=item_col, rater_col=rater_col, label_col=label_col
            )
        )

        n_items = len(id_maps["item"]["to_str"])
        n_raters = len(id_maps["annotator"]["to_str"])
        n_classes = len(id_maps["label"]["to_str"])

        classes = [id_maps["label"]["to_str"][k] for k in range(n_classes)]
        raters = [id_maps["annotator"]["to_str"][j] for j in range(n_raters)]
        items = [id_maps["item"]["to_str"][i] for i in range(n_items)]

        # Precompute hierarchy decay prior matrix alpha of shape (K, K)
        alpha = self._build_prior_matrix(classes)

        # Organize observations per item
        obs_by_item: list[list[tuple[int, int]]] = [[] for _ in range(n_items)]
        for it, rt, lb in zip(items_idx, raters_idx, labels_idx):
            obs_by_item[it].append((int(rt), int(lb)))

        # Initialize class priors p_k from label frequencies
        label_counts = np.bincount(labels_idx, minlength=n_classes)
        p = (label_counts + 1.0) / (len(labels_idx) + n_classes)

        # Initialize rater confusion matrices pi of shape (J, K, K) from prior
        pi = np.zeros((n_raters, n_classes, n_classes), dtype=np.float64)
        pi[:] = alpha / alpha.sum(axis=1, keepdims=True)

        prev_ll = -np.inf
        converged = False
        n_iter = 0
        q = np.zeros((n_items, n_classes), dtype=np.float64)

        for iteration in range(1, self.max_iter + 1):
            n_iter = iteration

            # --- E-Step: Compute posterior class probabilities q_i(k) ---
            for i in range(n_items):
                log_q_i = np.log(np.maximum(p, 1e-12))
                for j, obs_label in obs_by_item[i]:
                    log_q_i += np.log(np.maximum(pi[j, :, obs_label], 1e-12))

                max_log = np.max(log_q_i)
                exp_q = np.exp(log_q_i - max_log)
                q[i] = exp_q / np.maximum(np.sum(exp_q), 1e-12)

            # --- M-Step: Update priors p and confusion matrices pi ---
            p = (q.sum(axis=0) + 1.0) / (n_items + n_classes)

            new_pi = np.zeros_like(pi)
            for i in range(n_items):
                for j, obs_label in obs_by_item[i]:
                    new_pi[j, :, obs_label] += q[i]

            # Incorporate Dirichlet prior smoothing
            new_pi += alpha

            # Normalize rows
            row_sums = new_pi.sum(axis=2, keepdims=True)
            pi = new_pi / np.maximum(row_sums, 1e-12)

            # --- Convergence check via Log-Likelihood ---
            ll = 0.0
            for i in range(n_items):
                term = np.log(np.maximum(p, 1e-12))
                for j, obs_label in obs_by_item[i]:
                    term += np.log(np.maximum(pi[j, :, obs_label], 1e-12))
                max_t = np.max(term)
                ll += max_t + float(np.log(np.maximum(np.sum(np.exp(term - max_t)), 1e-12)))

            if iteration > 1 and abs(ll - prev_ll) < self.tol:
                converged = True
                break
            prev_ll = ll

        # Package results
        best_k = np.argmax(q, axis=1)
        consensus_labels = [classes[k] for k in best_k]
        confidences = np.max(q, axis=1)

        confusion_dict = {raters[j]: pi[j] for j in range(n_raters)}

        return ConsensusResult(
            item_ids=items,
            consensus_labels=consensus_labels,
            confidences=confidences,
            class_probabilities=q,
            classes=classes,
            annotator_confusion=confusion_dict,
            converged=converged,
            iterations=n_iter,
            log_likelihood=float(prev_ll),
            backend_affinity=affinity,
        )

    def _build_prior_matrix(self, classes: list[str]) -> np.ndarray:
        """Construct the (K, K) Dirichlet prior matrix based on ontology distances."""
        n_classes = len(classes)
        alpha = np.zeros((n_classes, n_classes), dtype=np.float64)

        if self.ontology is None:
            # Standard uninformative prior
            alpha[:] = self.beta_off
            np.fill_diagonal(alpha, self.beta_diag)
            return alpha

        for k in range(n_classes):
            for c in range(n_classes):
                if k == c:
                    alpha[k, c] = self.beta_diag
                else:
                    dist = self.ontology.get_distance(
                        classes[k], classes[c], metric=self.decay_metric
                    )
                    alpha[k, c] = self.beta_off * float(np.exp(-self.decay_lambda * dist))

        return alpha
