"""Tests for PSI drift monitoring functionality."""

from __future__ import annotations

import pandas as pd
import pytest

from pyrator.drift.psi import psi


def test_psi_basic():
    """Test basic PSI calculation."""
    # Create test data with two windows
    data = pd.DataFrame(
        {
            "window_id": ["baseline", "baseline", "current", "current"],
            "age": [25, 35, 30, 40],  # Different distributions
        }
    )

    result = psi(data, col="age", window_col="window_id")

    # Should have one row (current vs baseline)
    assert len(result) == 1
    assert result.iloc[0]["window_id"] == "current"
    assert result.iloc[0]["metric"] == "psi"
    assert isinstance(result.iloc[0]["value"], float)
    assert result.iloc[0]["value"] >= 0  # PSI should be non-negative


def test_psi_identical_distributions():
    """Test PSI with identical distributions (should be near zero)."""
    # Create test data with identical distributions
    data = pd.DataFrame(
        {
            "window_id": ["baseline", "baseline", "current", "current"],
            "age": [25, 35, 25, 35],  # Same distribution
        }
    )

    result = psi(data, col="age", window_col="window_id")

    # PSI should be very close to zero for identical distributions
    assert len(result) == 1
    assert result.iloc[0]["value"] < 0.1  # Should be very small


def test_psi_with_stratify():
    """Test PSI with stratification."""
    # Create test data with stratification
    data = pd.DataFrame(
        {
            "window_id": ["baseline", "baseline", "current", "current"],
            "age": [25, 35, 30, 40],
            "gender": ["M", "F", "M", "F"],
        }
    )

    result = psi(data, col="age", window_col="window_id", stratify=["gender"])

    # Should have one row for each stratum
    assert len(result) == 2  # M and F strata
    assert all(result["metric"] == "psi")
    assert all(result["window_id"] == "current")


def test_psi_invalid_inputs():
    """Test PSI with invalid inputs."""
    # Missing column
    data = pd.DataFrame({"window_id": ["baseline", "current"], "age": [25, 30]})

    with pytest.raises(ValueError, match="Column 'nonexistent' not found"):
        psi(data, col="nonexistent", window_col="window_id")

    # Not enough windows
    data = pd.DataFrame({"window_id": ["baseline", "baseline"], "age": [25, 35]})

    with pytest.raises(ValueError, match="PSI requires at least 2 different windows"):
        psi(data, col="age", window_col="window_id")
