# tests/conftest.py

import pytest
import httpx
from pathlib import Path
from loguru import logger

# Define the cache path for your package's test data
CACHE_DIR = Path.home() / ".cache" / "pyrator"
DATA_URL = "https://raw.githubusercontent.com/nyu-mll/crows-pairs/master/data/crows_pairs_anonymized.csv"
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

    logger.info("=" * 70)
    logger.info("Downloading CrowS-Pairs dataset for testing...")
    logger.info(f"Source: {DATA_URL}")
    logger.info("License: Creative Commons BY-SA 4.0")
    logger.warning(
        "This dataset contains offensive content and has known reliability issues."
    )
    logger.info("=" * 70)

    CACHE_DIR.mkdir(parents=True, exist_ok=True)

    try:
        with httpx.stream(
            "GET", DATA_URL, follow_redirects=True, timeout=30
        ) as response:
            response.raise_for_status()  # Raise an exception for 4xx/5xx errors
            with open(DATA_PATH, "wb") as f:
                for chunk in response.iter_bytes():
                    f.write(chunk)
        logger.success(f"Successfully downloaded dataset to: {DATA_PATH}")
    except httpx.RequestError as e:
        logger.error(f"Error downloading data: {e}")
        pytest.fail(f"Failed to download test data from {DATA_URL}")

    return str(DATA_PATH)
