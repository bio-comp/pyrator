"""Tests for MMD drift monitoring functionality."""

from __future__ import annotations

import pandas as pd
import pytest

from pyrator.drift.mmd import mmd


def test_mmd_basic():
    """Test basic MMD calculation."""
    data = pd.DataFrame(
        {
            "window_id": ["baseline", "baseline", "current", "current"],
            "emb_0": [0.1, 0.2, 0.3, 0.4],
            "emb_1": [0.4, 0.3, 0.2, 0.1],
        }
    )

    result = mmd(data, emb_cols=["emb_0", "emb_1"], window_col="window_id", n_perm=100)

    assert isinstance(result, pd.DataFrame)
    assert len(result) == 1
    assert result.iloc[0]["window_id"] == "current"
    assert result.iloc[0]["metric"] == "mmd"
    assert isinstance(result.iloc[0]["value"], float)


def test_mmd_identical_distributions():
    """Test MMD with identical distributions (should be ~0.0)."""
    data = pd.DataFrame(
        {
            "window_id": ["baseline", "baseline", "current", "current"],
            "emb_0": [0.1, 0.2, 0.1, 0.2],
            "emb_1": [0.3, 0.4, 0.3, 0.4],
        }
    )

    result = mmd(data, emb_cols=["emb_0", "emb_1"], window_col="window_id", n_perm=100, seed=42)

    assert result.iloc[0]["value"] < 0.1


def test_mmd_invalid_inputs():
    """Test MMD with invalid inputs."""
    data = pd.DataFrame({"window_id": ["baseline", "current"], "emb": [0.1, 0.2]})

    with pytest.raises(ValueError, match="Column 'nonexistent' not found"):
        mmd(data, emb_cols=["nonexistent"], window_col="window_id")

    with pytest.raises(ValueError, match="MMD requires at least 2 different windows"):
        data_single = pd.DataFrame({"window_id": ["baseline", "baseline"], "emb": [0.1, 0.2]})
        mmd(data_single, emb_cols=["emb"], window_col="window_id")
