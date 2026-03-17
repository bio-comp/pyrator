"""Tests for Cramer V drift monitoring functionality."""

from __future__ import annotations

import pandas as pd
import pytest

from pyrator.drift.cramer_v import cramer_v


def test_cramer_v_basic():
    """Test basic Cramer's V calculation."""
    data = pd.DataFrame(
        {
            "window_id": ["baseline", "baseline", "current", "current"],
            "category_a": ["x", "y", "x", "y"],
            "category_b": ["a", "b", "a", "b"],
        }
    )

    result = cramer_v(data, x="category_a", y="category_b", window_col="window_id")

    assert len(result) == 1
    assert result.iloc[0]["window_id"] == "current"
    assert result.iloc[0]["metric"] == "cramer_v"
    assert 0 <= result.iloc[0]["value"] <= 1


def test_cramer_v_perfect_association():
    """Test Cramer's V with perfect association (should be 1.0)."""
    data = pd.DataFrame(
        {
            "window_id": ["baseline", "baseline", "current", "current"],
            "category_a": ["x", "y", "x", "y"],
            "category_b": ["x", "y", "x", "y"],
        }
    )

    result = cramer_v(data, x="category_a", y="category_b", window_col="window_id")

    assert result.iloc[0]["value"] == 1.0


def test_cramer_v_no_association():
    """Test Cramer's V with no association (should be 0.0)."""
    data = pd.DataFrame(
        {
            "window_id": ["baseline"] * 4 + ["current"] * 4,
            "category_a": ["x", "y", "x", "y"] * 2,
            "category_b": ["a", "a", "b", "b"] * 2,
        }
    )

    result = cramer_v(data, x="category_a", y="category_b", window_col="window_id")

    assert result.iloc[0]["value"] == 0.0


def test_cramer_v_invalid_inputs():
    """Test Cramer's V with invalid inputs."""
    data = pd.DataFrame({"window_id": ["baseline", "current"], "cat": ["a", "b"]})

    with pytest.raises(ValueError, match="Column 'nonexistent' not found"):
        cramer_v(data, x="nonexistent", y="cat", window_col="window_id")

    with pytest.raises(ValueError, match="Column 'nonexistent' not found"):
        cramer_v(data, x="cat", y="nonexistent", window_col="window_id")
