"""Tests for Wasserstein distance drift monitoring functionality."""

from __future__ import annotations

import pandas as pd
import pytest

from pyrator.drift.wasserstein import w1


def test_wasserstein_basic():
    """Test basic Wasserstein distance calculation."""
    data = pd.DataFrame(
        {
            "window_id": ["baseline", "baseline", "current", "current"],
            "value": [1.0, 3.0, 2.0, 4.0],
        }
    )

    result = w1(data, col="value", window_col="window_id")

    assert len(result) == 1
    assert result.iloc[0]["window_id"] == "current"
    assert result.iloc[0]["metric"] == "wasserstein"
    assert result.iloc[0]["value"] >= 0


def test_wasserstein_identical_distributions():
    """Test Wasserstein with identical distributions (should be 0.0)."""
    data = pd.DataFrame(
        {
            "window_id": ["baseline", "baseline", "current", "current"],
            "value": [1.0, 3.0, 1.0, 3.0],
        }
    )

    result = w1(data, col="value", window_col="window_id")

    assert result.iloc[0]["value"] == 0.0


def test_wasserstein_invalid_inputs():
    """Test Wasserstein with invalid inputs."""
    data = pd.DataFrame({"window_id": ["baseline", "current"], "value": [1.0, 2.0]})

    with pytest.raises(ValueError, match="Column 'nonexistent' not found"):
        w1(data, col="nonexistent", window_col="window_id")

    with pytest.raises(
        ValueError, match="Wasserstein distance requires at least 2 different windows"
    ):
        data_single = pd.DataFrame({"window_id": ["baseline", "baseline"], "value": [1.0, 2.0]})
        w1(data_single, col="value", window_col="window_id")
