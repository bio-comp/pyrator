"""Tests for drift utility functions."""

from __future__ import annotations

import pandas as pd
import pytest

from pyrator.drift._utils import iter_window_strata, validate_windows


def test_validate_windows_success() -> None:
    df = pd.DataFrame({"window": ["w1", "w2", "w3"]})
    base, currents = validate_windows(df, window_col="window", baseline="w1", metric_name="Test")
    assert base == "w1"
    assert currents == ["w2", "w3"]


def test_validate_windows_default_baseline() -> None:
    df = pd.DataFrame({"window": ["w1", "w2"]})
    base, currents = validate_windows(df, window_col="window", baseline=None, metric_name="Test")
    assert base == "w1"
    assert currents == ["w2"]


def test_validate_windows_insufficient_windows() -> None:
    df = pd.DataFrame({"window": ["w1", "w1"]})
    with pytest.raises(ValueError, match="Test requires at least 2 different windows"):
        validate_windows(df, window_col="window", metric_name="Test")


def test_validate_windows_missing_baseline() -> None:
    df = pd.DataFrame({"window": ["w1", "w2"]})
    with pytest.raises(ValueError, match="Baseline window 'nonexistent' not found in data"):
        validate_windows(df, window_col="window", baseline="nonexistent", metric_name="Test")


def test_validate_windows_missing_column() -> None:
    df = pd.DataFrame({"other": ["w1", "w2"]})
    with pytest.raises(ValueError, match="Column 'window' not found in data"):
        validate_windows(df, window_col="window", metric_name="Test")


def test_validate_windows_deduplicates_mixed_types() -> None:
    df = pd.DataFrame({"window": [1, "1"]})
    with pytest.raises(ValueError, match="Test requires at least 2 different windows"):
        validate_windows(df, window_col="window", metric_name="Test")


def test_validate_windows_with_int_column() -> None:
    df = pd.DataFrame({"window": [1, 2, 3]})
    base, currents = validate_windows(df, window_col="window", baseline="1", metric_name="Test")
    assert base == "1"
    assert currents == ["2", "3"]


def test_iter_window_strata_unstratified() -> None:
    df = pd.DataFrame(
        {
            "window": ["w1", "w1", "w2", "w2"],
            "val": [1, 2, 3, 4],
        }
    )
    strata = list(
        iter_window_strata(df, window_col="window", baseline_window="w1", current_window="w2")
    )
    assert len(strata) == 1
    b_df, c_df, s_dict = strata[0]
    assert s_dict == {}
    assert list(b_df["val"]) == [1, 2]
    assert list(c_df["val"]) == [3, 4]


def test_iter_window_strata_stratified() -> None:
    df = pd.DataFrame(
        {
            "window": ["w1", "w1", "w2", "w2"],
            "grp": ["g1", "g2", "g1", "g2"],
            "val": [10, 20, 30, 40],
        }
    )
    strata = list(
        iter_window_strata(
            df,
            window_col="window",
            baseline_window="w1",
            current_window="w2",
            stratify=["grp"],
        )
    )
    assert len(strata) == 2
    assert strata[0][2] == {"grp": "g1"}
    assert list(strata[0][0]["val"]) == [10]
    assert list(strata[0][1]["val"]) == [30]
    assert strata[1][2] == {"grp": "g2"}
    assert list(strata[1][0]["val"]) == [20]
    assert list(strata[1][1]["val"]) == [40]


def test_iter_window_strata_with_nan_in_stratum() -> None:
    df = pd.DataFrame(
        {
            "window": ["w1", "w1", "w2", "w2"],
            "grp": [None, "g1", None, "g1"],
            "val": [10, 20, 30, 40],
        }
    )
    strata = list(
        iter_window_strata(
            df,
            window_col="window",
            baseline_window="w1",
            current_window="w2",
            stratify=["grp"],
        )
    )
    assert len(strata) == 2
    # Find stratum with None/NaN
    nan_strata = [s for s in strata if pd.isna(s[2]["grp"])]
    assert len(nan_strata) == 1
    assert list(nan_strata[0][0]["val"]) == [10]
    assert list(nan_strata[0][1]["val"]) == [30]
