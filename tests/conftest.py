# tests/conftest.py

from pathlib import Path

import httpx
import pandas as pd
import pytest
from loguru import logger

from pyrator.ontology.core import Ontology

# Define the cache path for your package's test data
CACHE_DIR = Path.home() / ".cache" / "pyrator"
DATA_URL = (
    "https://raw.githubusercontent.com/nyu-mll/crows-pairs/master/data/crows_pairs_anonymized.csv"
)
DATA_PATH = CACHE_DIR / "crows_pairs_anonymized.csv"


@pytest.fixture(scope="session")
def crows_pairs_path() -> str:
    """
    A pytest fixture that downloads the CrowS-Pairs dataset if it doesn't exist
    and returns the path to it. The scope='session' ensures this runs only
    once per test session.
    """
    if DATA_PATH.exists():
        logger.info(f"Found existing dataset at: {DATA_PATH}")
    return str(DATA_PATH)


@pytest.fixture
def simple_ontology():
    r"""
    A diamond graph for testing semantic distances.

          Root (0)
         /    \
      A (1)   B (1)
         \    /
          C (2)
    """
    nodes = {
        "Root": {"depth": 0},
        "A": {"depth": 1},
        "B": {"depth": 1},
        "C": {"depth": 2},
    }
    edges = [("Root", "A"), ("Root", "B"), ("A", "C"), ("B", "C")]
    return Ontology.build("test_v1", nodes=nodes, edges=edges)


@pytest.fixture
def dag_cross_branch_shortcut() -> tuple[dict[str, dict[str, str]], list[tuple[str, str]]]:
    r"""
    DAG where undirected shortest path is shorter than LCA up/down path.

          root
         /    \
        a      b
        |      |
        u      v
         \    /
           z

    d_spanning_tree(u, v) = 4 via root
    d_undirected_tr(u, v) = 2 via z
    """
    nodes = {node_id: {"name": node_id.upper()} for node_id in ["root", "a", "b", "u", "v", "z"]}
    edges = [
        ("root", "a"),
        ("root", "b"),
        ("a", "u"),
        ("b", "v"),
        ("u", "z"),
        ("v", "z"),
    ]
    return nodes, edges


@pytest.fixture
def dag_with_redundant_edge() -> tuple[dict[str, dict[str, str]], list[tuple[str, str]]]:
    r"""
    DAG with an explicit transitive edge.

    root -> a -> b -> c -> z
    root -> x -> y ----^
                   ^
                   |
                   b -> z  (redundant transitive edge via b -> c -> z)

    For distance between b and y:
      - raw edges shortest path is 2 (b-z-y),
      - transitive-reduction shortest path is 3 (b-c-z-y).
    """
    nodes = {
        node_id: {"name": node_id.upper()} for node_id in ["root", "a", "b", "c", "x", "y", "z"]
    }
    edges = [
        ("root", "a"),
        ("a", "b"),
        ("b", "c"),
        ("c", "z"),
        ("root", "x"),
        ("x", "y"),
        ("y", "z"),
        ("b", "z"),
    ]
    return nodes, edges


@pytest.fixture
def cohen_nominal_data() -> pd.DataFrame:
    """Canonical two-rater nominal dataset with expected Cohen's kappa = 0.4."""
    rows: list[dict[str, str]] = []

    # Confusion table:
    #               rater_B
    #           yes(30)  no(20)
    # rater_A yes  20       5
    #         no   10      15
    # po = 0.7, pe = 0.5 => kappa = 0.4
    item_id = 0
    for _ in range(20):
        item = f"item_{item_id}"
        rows.append({"item": item, "rater": "A", "label": "yes"})
        rows.append({"item": item, "rater": "B", "label": "yes"})
        item_id += 1

    for _ in range(5):
        item = f"item_{item_id}"
        rows.append({"item": item, "rater": "A", "label": "yes"})
        rows.append({"item": item, "rater": "B", "label": "no"})
        item_id += 1

    for _ in range(10):
        item = f"item_{item_id}"
        rows.append({"item": item, "rater": "A", "label": "no"})
        rows.append({"item": item, "rater": "B", "label": "yes"})
        item_id += 1

    for _ in range(15):
        item = f"item_{item_id}"
        rows.append({"item": item, "rater": "A", "label": "no"})
        rows.append({"item": item, "rater": "B", "label": "no"})
        item_id += 1

    return pd.DataFrame(rows)


@pytest.fixture
def krippendorff_data():
    """
    Canonical 12-item dataset from Krippendorff (1980).
    Expected Alpha (Nominal) approx 0.375.
    """
    data = []
    rater_a = [1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
    rater_b = [0, 1, 1, 0, 0, 1, 0, 1, 0, 0, 0, 0]

    for i, val in enumerate(rater_a, 1):
        data.append({"item": f"item_{i}", "rater": "A", "label": str(val)})
    for i, val in enumerate(rater_b, 1):
        data.append({"item": f"item_{i}", "rater": "B", "label": str(val)})

    return pd.DataFrame(data)

    logger.info("=" * 70)
    logger.info("Downloading CrowS-Pairs dataset for testing...")
    logger.info(f"Source: {DATA_URL}")
    logger.info("License: Creative Commons BY-SA 4.0")
    logger.warning("This dataset contains offensive content and has known reliability issues.")
    logger.info("=" * 70)

    CACHE_DIR.mkdir(parents=True, exist_ok=True)

    try:
        with httpx.stream("GET", DATA_URL, follow_redirects=True, timeout=30) as response:
            response.raise_for_status()  # Raise an exception for 4xx/5xx errors
            with open(DATA_PATH, "wb") as f:
                for chunk in response.iter_bytes():
                    f.write(chunk)
        logger.success(f"Successfully downloaded dataset to: {DATA_PATH}")
    except httpx.RequestError as e:
        logger.error(f"Error downloading data: {e}")
        pytest.fail(f"Failed to download test data from {DATA_URL}")

    return str(DATA_PATH)
