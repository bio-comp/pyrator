from pyrator.ontology.core import Ontology


def test_dag_diamond_distance():
    """
    Test distance calculation in a Diamond DAG:
         Root
        /    \
       A      B
        \\    /
          C

    Depths: Root=0, A=1, B=1, C=2
    """
    # 1. Setup Diamond Graph
    nodes = {
        "root": {"name": "Root"},
        "a": {"name": "A"},
        "b": {"name": "B"},
        "c": {"name": "C"},
    }
    # Edges: (Parent, Child)
    edges = [
        ("root", "a"),
        ("root", "b"),
        ("a", "c"),
        ("b", "c"),
    ]

    ont = Ontology.build(version="test_v1", nodes=nodes, edges=edges)

    # 2. Validate Depths (Base logic check)
    assert ont.nodes["root"].depth == 0
    assert ont.nodes["a"].depth == 1
    assert ont.nodes["b"].depth == 1
    assert ont.nodes["c"].depth == 2  # Max parent (1) + 1

    # 3. Test Distances

    # Case 1: Simple Parent-Child (Direct)
    # d(A, C) = 1 + 2 - 2*depth(A=1) = 1
    assert ont.get_distance("a", "c") == 1

    # Case 2: Siblings (A vs B)
    # LCA(A, B) = Root (depth 0)
    # d(A, B) = 1 + 1 - 2*(0) = 2
    assert ont.get_distance("a", "b") == 2

    # Case 3: The Diamond Bottom (Root vs C)
    # Path goes Root -> A -> C (or Root -> B -> C)
    # LCA(Root, C) = Root (depth 0)
    # d(Root, C) = 0 + 2 - 2*(0) = 2
    assert ont.get_distance("root", "c") == 2


def test_dag_ambiguous_lca():
    """
    Test a "Criss-Cross" DAG where LCAs have different depths (if depths were uneven),
    or simply multiple LCAs.

         Root
          |
          M (Middle, depth 1)
        /   \
       P1    P2 (Parents, depth 2)
       | \\  / |
       |  \\/  |
       |  /\\  |
       C1    C2 (Children, depth 3)

    LCA(C1, C2) includes {P1, P2, M, Root}.
    Deepest are P1 and P2 (depth 2).
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

    ont = Ontology.build("v1", nodes, edges)

    # Check Depths
    assert ont.nodes["c1"].depth == 3
    assert ont.nodes["c2"].depth == 3

    # LCA(C1, C2) set should contain P1, P2, M, Root
    # The method should pick max_depth(P1, P2) = 2
    # Distance = 3 + 3 - 2(2) = 2
    # Represents path: C1 -> P1 -> C2 (distance 2)
    assert ont.get_distance("c1", "c2") == 2
