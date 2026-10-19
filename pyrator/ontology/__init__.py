"""Ontology module for tree and DAG hierarchies, semantic similarity, and LCA queries."""

from __future__ import annotations

from pyrator.ontology.core import Node, Ontology, truncate_id_to_depth

__all__ = ["Node", "Ontology", "truncate_id_to_depth"]
