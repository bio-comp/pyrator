import pytest
import math
from pyrator.ontology.core import Ontology


def test_ontology_ic_calculation():
    """
    Test IC calculation in a simple tree.
    
         Root (10)
        /    \
       A (5)  B (5)
      / \      \
     A1(2) A2(3) B1(5)
    
    Counts (at nodes): Root:0, A:0, B:0, A1:2, A2:3, B1:5
    Descendant Counts:
    Root: 2+3+5 = 10
    A: 2+3 = 5
    B: 5
    A1: 2
    A2: 3
    B1: 5
    
    alpha = 1
    |V| = 6
    Denominator: 10 + 1 * 6 = 16
    
    p_hat(Root) = (10 + 1) / 16 = 11/16
    p_hat(A) = (5 + 1) / 16 = 6/16
    p_hat(B) = (5 + 1) / 16 = 6/16
    p_hat(A1) = (2 + 1) / 16 = 3/16
    p_hat(A2) = (3 + 1) / 16 = 4/16
    p_hat(B1) = (5 + 1) / 16 = 6/16
    """
    nodes = {n: {} for n in ["root", "a", "b", "a1", "a2", "b1"]}
    edges = [("root", "a"), ("root", "b"), ("a", "a1"), ("a", "a2"), ("b", "b1")]
    corpus_counts = {"a1": 2, "a2": 3, "b1": 5}

    ont = Ontology.build("v1", nodes, edges, corpus_counts=corpus_counts, smoothing_alpha=1.0)

    assert ont.nodes["root"].ic == -math.log(11 / 16)
    assert ont.nodes["a"].ic == -math.log(6 / 16)
    assert ont.nodes["b"].ic == -math.log(6 / 16)
    assert ont.nodes["a1"].ic == -math.log(3 / 16)
    assert ont.nodes["a2"].ic == -math.log(4 / 16)
    assert ont.nodes["b1"].ic == -math.log(6 / 16)


def test_ontology_semantic_metrics():
    nodes = {n: {} for n in ["root", "a", "b", "a1", "a2"]}
    edges = [("root", "a"), ("root", "b"), ("a", "a1"), ("a", "a2")]
    corpus_counts = {"a1": 2, "a2": 2, "b": 4}
    # Total count = 8, |V| = 5, alpha=1
    # Denom = 8 + 5 = 13
    # root desc: a1+a2+b = 8. p = (8+1)/13 = 9/13. IC = -log(9/13)
    # a desc: a1+a2 = 4. p = (4+1)/13 = 5/13. IC = -log(5/13)
    # b desc: 4. p = (4+1)/13 = 5/13. IC = -log(5/13)
    # a1 desc: 2. p = (2+1)/13 = 3/13. IC = -log(3/13)
    # a2 desc: 2. p = (2+1)/13 = 3/13. IC = -log(3/13)

    ont = Ontology.build("v1", nodes, edges, corpus_counts=corpus_counts, smoothing_alpha=1.0)

    # Lin similarity: 2 * IC(LCA) / (IC(u) + IC(v))
    # s_lin(a1, a2) = 2 * IC(a) / (IC(a1) + IC(a2))
    # IC(a) = -log(5/13)
    # IC(a1) = -log(3/13)
    expected_lin = (2 * -math.log(5 / 13)) / (-math.log(3 / 13) + -math.log(3 / 13))
    assert ont.get_similarity("a1", "a2", metric="lin") == pytest.approx(expected_lin)

    # Resnik similarity: IC(LCA)
    assert ont.get_similarity("a1", "a2", metric="resnik") == pytest.approx(-math.log(5 / 13))

    # Resnik norm: IC(LCA) / max_ic
    max_ic = max(n.ic for n in ont.nodes.values())
    assert ont.get_similarity("a1", "a2", metric="resnik_norm") == pytest.approx(
        -math.log(5 / 13) / max_ic
    )


def test_lca_policies():
    """
    DAG with multiple LCAs:
         Root
          |
          M
        /   \
       P1    P2
       | \  / |
       C1    C2
    
    LCA(C1, C2) = {P1, P2}
    """
    nodes = {n: {} for n in ["root", "m", "p1", "p2", "c1", "c2"]}
    edges = [
        ("root", "m"),
        ("m", "p1"),
        ("m", "p2"),
        ("p1", "c1"),
        ("p1", "c2"),
        ("p2", "c1"),
        ("p2", "c2"),
    ]

    # Policy: max_depth
    # p1 and p2 have same depth (3).
    ont_depth = Ontology.build("v1", nodes, edges, lca_policy="max_depth")
    lca = ont_depth.get_lca("c1", "c2")
    assert lca in ["p1", "p2"]

    # Policy: max_ic
    # Give p1 more counts so it has lower IC? Wait, IC = -log(p). Higher count -> higher p -> lower IC.
    # To have higher IC, it should have LOWER count.
    corpus_counts = {"c1": 1, "c2": 1}
    # p1 descendants: {p1, c1, c2}. count = 1+1 = 2
    # p2 descendants: {p2, c1, c2}. count = 1+1 = 2
    # If we add counts to p1 directly:
    corpus_counts_ic = {"c1": 1, "c2": 1, "p1": 10}  # p1 has more count -> lower IC
    ont_ic = Ontology.build("v1", nodes, edges, corpus_counts=corpus_counts_ic, lca_policy="max_ic")
    # p1 desc count = 10+1+1 = 12. p = (12+1)/denom. IC = -log(13/denom)
    # p2 desc count = 0+1+1 = 2. p = (2+1)/denom. IC = -log(3/denom)
    # IC(p2) > IC(p1) since 3/denom < 13/denom.
    assert ont_ic.get_lca("c1", "c2") == "p2"


def test_ontology_metrics_discovery():
    ont = Ontology.build("v1", {"a": {}}, [])
    metrics = ont.metrics()
    assert any(m["key"] == "lin" and m["type"] == "similarity" for m in metrics)
    assert any(m["key"] == "path" and m["type"] == "distance" for m in metrics)
