"""Semantic distance utilities for IRA calculations."""

from __future__ import annotations

from typing import Sequence

import numpy as np
from numpy.typing import NDArray

from pyrator.ontology.core import Ontology


class SemanticDistanceFactory:
    """Helper to generate distance matrices from Ontologies."""

    def __init__(self, ontology: Ontology):
        self.ont = ontology

    def compute_distance_matrix(
        self,
        labels: Sequence[str],
        metric: str = "path",
    ) -> NDArray[np.float64]:
        """
        Generate a symmetric distance matrix for the provided labels.

        Args:
            labels: List of label IDs (must exist in ontology).
            metric: Distance metric ('path', 'lin', 'resnik_norm', 'lca').

        Returns:
            Square numpy matrix of shape (len(labels), len(labels)).
        """
        n = len(labels)
        matrix = np.zeros((n, n), dtype=np.float64)

        self.ont.validate_labels_exist(labels)

        for i in range(n):
            u = labels[i]
            for j in range(i + 1, n):
                v = labels[j]

                try:
                    dist = self.ont.get_distance(u, v, metric=metric)
                except KeyError:
                    dist = 1.0

                matrix[i, j] = dist
                matrix[j, i] = dist

        return matrix
