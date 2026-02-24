"""Tests for intraclass correlation (ICC) agreement metrics."""

from __future__ import annotations

from typing import cast

import pandas as pd
import pytest

from pyrator.ira.icc import (
    ICCVariant,
    icc_2_1,
    icc_2_k,
    icc_3_1,
    icc_3_k,
    intraclass_correlation,
)


@pytest.mark.parametrize(
    ("variant", "expected"),
    [
        ("ICC2_1", 0.16806722689075623),
        ("ICC2_k", 0.37735849056603765),
        ("ICC3_1", 0.7142857142857142),
        ("ICC3_k", 0.8823529411764705),
    ],
)
def test_intraclass_correlation_reference_values(
    icc_continuous_data: pd.DataFrame,
    variant: ICCVariant,
    expected: float,
) -> None:
    """Each supported ICC variant should match known reference values."""
    value = intraclass_correlation(
        icc_continuous_data,
        item_col="item",
        rater_col="rater",
        score_col="score",
        variant=variant,
    )

    assert value == pytest.approx(expected, abs=1e-12)


def test_icc_wrapper_functions_match_dispatch(icc_continuous_data: pd.DataFrame) -> None:
    """Wrapper functions should match dispatcher outputs for each variant."""
    assert icc_2_1(
        icc_continuous_data, item_col="item", rater_col="rater", score_col="score"
    ) == pytest.approx(
        intraclass_correlation(
            icc_continuous_data,
            item_col="item",
            rater_col="rater",
            score_col="score",
            variant="ICC2_1",
        ),
        abs=1e-12,
    )
    assert icc_2_k(
        icc_continuous_data, item_col="item", rater_col="rater", score_col="score"
    ) == pytest.approx(
        intraclass_correlation(
            icc_continuous_data,
            item_col="item",
            rater_col="rater",
            score_col="score",
            variant="ICC2_k",
        ),
        abs=1e-12,
    )
    assert icc_3_1(
        icc_continuous_data, item_col="item", rater_col="rater", score_col="score"
    ) == pytest.approx(
        intraclass_correlation(
            icc_continuous_data,
            item_col="item",
            rater_col="rater",
            score_col="score",
            variant="ICC3_1",
        ),
        abs=1e-12,
    )
    assert icc_3_k(
        icc_continuous_data, item_col="item", rater_col="rater", score_col="score"
    ) == pytest.approx(
        intraclass_correlation(
            icc_continuous_data,
            item_col="item",
            rater_col="rater",
            score_col="score",
            variant="ICC3_k",
        ),
        abs=1e-12,
    )


def test_intraclass_correlation_rejects_missing_item_rater_cells() -> None:
    """ICC should require a complete item-by-rater matrix in strict mode."""
    df = pd.DataFrame(
        [
            {"item": "i1", "rater": "A", "score": 1.0},
            {"item": "i1", "rater": "B", "score": 2.0},
            {"item": "i2", "rater": "A", "score": 3.0},
        ]
    )

    with pytest.raises(ValueError, match="complete item-by-rater matrix"):
        intraclass_correlation(df, item_col="item", rater_col="rater", score_col="score")


def test_intraclass_correlation_rejects_duplicate_item_rater_pairs() -> None:
    """ICC should reject duplicate (item, rater) observations."""
    df = pd.DataFrame(
        [
            {"item": "i1", "rater": "A", "score": 1.0},
            {"item": "i1", "rater": "A", "score": 1.5},
            {"item": "i1", "rater": "B", "score": 2.0},
            {"item": "i2", "rater": "A", "score": 3.0},
            {"item": "i2", "rater": "B", "score": 4.0},
        ]
    )

    with pytest.raises(ValueError, match="one rating per \\(item, rater\\) pair"):
        intraclass_correlation(df, item_col="item", rater_col="rater", score_col="score")


def test_intraclass_correlation_rejects_non_numeric_scores() -> None:
    """ICC should reject non-numeric score columns."""
    df = pd.DataFrame(
        [
            {"item": "i1", "rater": "A", "score": "high"},
            {"item": "i1", "rater": "B", "score": "low"},
            {"item": "i2", "rater": "A", "score": "mid"},
            {"item": "i2", "rater": "B", "score": "mid"},
        ]
    )

    with pytest.raises(ValueError, match="numeric"):
        intraclass_correlation(df, item_col="item", rater_col="rater", score_col="score")


def test_intraclass_correlation_rejects_unknown_variant(
    icc_continuous_data: pd.DataFrame,
) -> None:
    """Only the explicitly supported four ICC variants should be accepted."""
    with pytest.raises(ValueError, match="Unsupported ICC variant"):
        intraclass_correlation(
            icc_continuous_data,
            item_col="item",
            rater_col="rater",
            score_col="score",
            variant=cast("ICCVariant", "ICC1_1"),
        )
