"""Additional ontology core coverage tests."""

from __future__ import annotations

import pytest

from pyrator.ontology.core import Ontology, _toposort, truncate_id_to_depth


def _build_small_tree() -> Ontology:
    nodes = {
        "root": {"name": "Root"},
        "a": {"name": "A"},
        "b": {"name": "B"},
        "c": {"name": "C"},
    }
    edges = [("root", "a"), ("root", "b"), ("a", "c")]
    return Ontology.build("v_extra", nodes, edges)


def test_build_rejects_missing_parent_and_child() -> None:
    nodes = {"a": {}, "b": {}}

    with pytest.raises(KeyError, match="Parent id not found"):
        Ontology.build("v", nodes, [("missing", "a")])

    with pytest.raises(KeyError, match="Child id not found"):
        Ontology.build("v", nodes, [("a", "missing")])


def test_distance_similarity_and_strict_validation_branches() -> None:
    ont = _build_small_tree()

    with pytest.raises(KeyError, match="labels not found"):
        ont.get_distance("missing", "a")

    with pytest.raises(KeyError, match="labels not found"):
        ont.get_similarity("missing", "a")

    disconnected = Ontology.build("v2", {"x": {}, "y": {}}, [])
    assert disconnected.get_distance("x", "y") == 0
    assert disconnected.get_similarity("x", "y") == 0.0

    with pytest.raises(KeyError, match="Unknown labels"):
        ont.expand_with_ancestors(["root", "unknown"], strict=True)


def test_frontier_lca_and_descendant_utilities() -> None:
    ont = _build_small_tree()

    assert ont.compress_to_frontier(["root", "a", "c"]) == {"c"}
    assert ont.lowest_common_ancestors(["c"]) == {"c"}
    assert ont.lowest_common_ancestors(["missing"]) == set()
    assert ont.ancestors("c") == {"root", "a"}
    assert ont.parents_of("c") == {"a"}
    assert ont.get_descendants("root") == {"a", "b", "c"}
    assert ont.get_descendants("missing") == set()


def test_truncate_and_toposort_cycle_guard() -> None:
    assert truncate_id_to_depth("a_b_c_d", depth=1) == "a_b"
    assert truncate_id_to_depth("a_b", depth=3) == "a_b"

    with pytest.raises(ValueError, match="Graph is not a DAG"):
        _toposort(
            nodes={"a", "b"},
            parents={"a": {"b"}, "b": {"a"}},
            children={"a": {"b"}, "b": {"a"}},
        )
