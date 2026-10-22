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


def test_jsd_with_groupby() -> None:
    """Test JSD calculation stratified by group."""
    data = pd.DataFrame(
        {
            "window_id": ["baseline", "baseline", "current", "current"],
            "group": ["g1", "g2", "g1", "g2"],
            "dist_a": [0.5, 0.4, 0.7, 0.6],
            "dist_b": [0.5, 0.6, 0.3, 0.4],
        }
    )

    result = jsd(data, dist_cols=["dist_a", "dist_b"], window_col="window_id", groupby="group")

    assert len(result) == 2
    assert set(result["stratum"].apply(lambda d: d.get("group"))) == {"g1", "g2"}
    assert all(result["metric"] == "jsd")
    assert all(result["value"] >= 0)
