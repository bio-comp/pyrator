"""Intraclass correlation coefficient (ICC) metrics for continuous agreement."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

import numpy as np
from numpy.typing import NDArray

from pyrator.types import FrameLike

ICCVariant = Literal["ICC2_1", "ICC2_k", "ICC3_1", "ICC3_k"]


@dataclass(frozen=True)
class _TwoWayMeanSquares:
    """Container for two-way ANOVA mean squares used by ICC formulas."""

    ms_rows: float
    ms_cols: float
    ms_error: float


def intraclass_correlation(
    df: FrameLike,
    *,
    item_col: str,
    rater_col: str,
    score_col: str,
    variant: ICCVariant = "ICC2_1",
) -> float:
    """Compute Shrout/Fleiss-style two-way ICC variants for continuous ratings.

    Supported variants:
    - ``ICC2_1``: Two-way random effects, single rater/measurement.
    - ``ICC2_k``: Two-way random effects, average of k raters/measurements.
    - ``ICC3_1``: Two-way mixed effects, single rater/measurement.
    - ``ICC3_k``: Two-way mixed effects, average of k raters/measurements.

    Args:
        df: Long-format annotation data.
        item_col: Item/subject column.
        rater_col: Rater/annotator column.
        score_col: Continuous score column.
        variant: ICC variant name.

    Returns:
        ICC value for the selected variant.

    Raises:
        ValueError: If inputs are malformed or selected variant is unsupported.
    """
    if hasattr(df, "to_pandas"):
        frame = df.to_pandas()
    else:
        frame = df

    for col in (item_col, rater_col, score_col):
        if col not in frame.columns:
            raise ValueError(f"Missing required column: {col}")

    duplicate_counts = frame.groupby([item_col, rater_col]).size()
    duplicate_pairs = duplicate_counts[duplicate_counts > 1]
    if not duplicate_pairs.empty:
        raise ValueError("ICC requires one rating per (item, rater) pair.")

    matrix = frame.pivot(index=item_col, columns=rater_col, values=score_col)
    if matrix.isna().any().any():
        raise ValueError("ICC requires a complete item-by-rater matrix in strict mode.")

    values = matrix.to_numpy()
    try:
        ratings = values.astype(float)
    except (TypeError, ValueError) as error:
        raise ValueError("ICC requires numeric score values.") from error

    if not np.isfinite(ratings).all():
        raise ValueError("ICC requires finite numeric score values.")

    n_items, n_raters = ratings.shape
    if n_items < 2:
        raise ValueError("ICC requires at least two items.")
    if n_raters < 2:
        raise ValueError("ICC requires at least two raters.")

    mean_squares = _compute_two_way_mean_squares(ratings)
    return _compute_variant(
        variant=variant,
        mean_squares=mean_squares,
        n_items=n_items,
        n_raters=n_raters,
    )


def icc_2_1(
    df: FrameLike,
    *,
    item_col: str,
    rater_col: str,
    score_col: str,
) -> float:
    """Compute ``ICC(2,1)`` (two-way random, single measurement)."""
    return intraclass_correlation(
        df,
        item_col=item_col,
        rater_col=rater_col,
        score_col=score_col,
        variant="ICC2_1",
    )


def icc_2_k(
    df: FrameLike,
    *,
    item_col: str,
    rater_col: str,
    score_col: str,
) -> float:
    """Compute ``ICC(2,k)`` (two-way random, average measurement)."""
    return intraclass_correlation(
        df,
        item_col=item_col,
        rater_col=rater_col,
        score_col=score_col,
        variant="ICC2_k",
    )


def icc_3_1(
    df: FrameLike,
    *,
    item_col: str,
    rater_col: str,
    score_col: str,
) -> float:
    """Compute ``ICC(3,1)`` (two-way mixed, single measurement)."""
    return intraclass_correlation(
        df,
        item_col=item_col,
        rater_col=rater_col,
        score_col=score_col,
        variant="ICC3_1",
    )


def icc_3_k(
    df: FrameLike,
    *,
    item_col: str,
    rater_col: str,
    score_col: str,
) -> float:
    """Compute ``ICC(3,k)`` (two-way mixed, average measurement)."""
    return intraclass_correlation(
        df,
        item_col=item_col,
        rater_col=rater_col,
        score_col=score_col,
        variant="ICC3_k",
    )


def _compute_two_way_mean_squares(values: NDArray[np.float64]) -> _TwoWayMeanSquares:
    """Compute two-way ANOVA mean squares for item and rater effects.

    Notes:
        Uses Shrout and Fleiss (1979) decomposition for fully crossed designs.
    """
    n_items, n_raters = values.shape
    grand_mean = float(np.mean(values))
    item_means = np.mean(values, axis=1)
    rater_means = np.mean(values, axis=0)

    ss_rows = float(n_raters * np.sum((item_means - grand_mean) ** 2))
    ss_cols = float(n_items * np.sum((rater_means - grand_mean) ** 2))
    ss_total = float(np.sum((values - grand_mean) ** 2))
    ss_error = ss_total - ss_rows - ss_cols

    # Guard against tiny negative residuals from floating-point cancellation.
    if ss_error < 0.0 and abs(ss_error) < 1e-12:
        ss_error = 0.0

    ms_rows = ss_rows / float(n_items - 1)
    ms_cols = ss_cols / float(n_raters - 1)
    ms_error = ss_error / float((n_items - 1) * (n_raters - 1))

    return _TwoWayMeanSquares(ms_rows=ms_rows, ms_cols=ms_cols, ms_error=ms_error)


def _compute_variant(
    *,
    variant: ICCVariant,
    mean_squares: _TwoWayMeanSquares,
    n_items: int,
    n_raters: int,
) -> float:
    """Map supported ICC variants to their Shrout/Fleiss formulas."""
    msr = mean_squares.ms_rows
    msc = mean_squares.ms_cols
    mse = mean_squares.ms_error

    numerator = msr - mse
    n_items_f = float(n_items)
    n_raters_f = float(n_raters)

    if variant == "ICC2_1":
        denominator = msr + (n_raters_f - 1.0) * mse + (n_raters_f * (msc - mse) / n_items_f)
    elif variant == "ICC2_k":
        denominator = msr + ((msc - mse) / n_items_f)
    elif variant == "ICC3_1":
        denominator = msr + (n_raters_f - 1.0) * mse
    elif variant == "ICC3_k":
        denominator = msr
    else:
        raise ValueError(
            f"Unsupported ICC variant: {variant}. "
            "Supported variants: ICC2_1, ICC2_k, ICC3_1, ICC3_k."
        )

    if abs(denominator) < 1e-12:
        raise ValueError(f"ICC variant {variant} is undefined because its denominator is zero.")

    return float(numerator / denominator)
