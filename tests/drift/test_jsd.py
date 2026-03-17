"""Tests for JSD drift monitoring functionality."""

from __future__ import annotations

import pandas as pd
import pytest

from pyrator.drift.jsd import jsd


def test_jsd_basic():
    """Test basic JSD calculation."""
    data = pd.DataFrame(
        {
            "window_id": ["baseline", "baseline", "current", "current"],
            "dist_a": [0.5, 0.5, 0.7, 0.3],
            "dist_b": [0.5, 0.5, 0.3, 0.7],
        }
    )

    result = jsd(data, dist_cols=["dist_a", "dist_b"], window_col="window_id")

    assert len(result) == 1
    assert result.iloc[0]["window_id"] == "current"
    assert result.iloc[0]["metric"] == "jsd"
    assert result.iloc[0]["value"] >= 0


def test_jsd_identical_distributions():
    """Test JSD with identical distributions (should be 0.0)."""
    data = pd.DataFrame(
        {
            "window_id": ["baseline", "baseline", "current", "current"],
            "dist_a": [0.5, 0.5, 0.5, 0.5],
            "dist_b": [0.5, 0.5, 0.5, 0.5],
        }
    )

    result = jsd(data, dist_cols=["dist_a", "dist_b"], window_col="window_id")

    assert result.iloc[0]["value"] == 0.0


def test_jsd_invalid_inputs():
    """Test JSD with invalid inputs."""
    data = pd.DataFrame({"window_id": ["baseline", "current"], "dist": [0.5, 0.5]})

    with pytest.raises(ValueError, match="Column 'nonexistent' not found"):
        jsd(data, dist_cols=["nonexistent"], window_col="window_id")

    with pytest.raises(ValueError, match="Distribution columns must sum to 1"):
        jsd(data, dist_cols=["dist"], window_col="window_id")
