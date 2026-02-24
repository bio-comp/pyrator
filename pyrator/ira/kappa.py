"""Kappa-family agreement metrics."""

from __future__ import annotations

import numpy as np

from pyrator.types import FrameLike


def cohen_kappa(
    df: FrameLike,
    *,
    item_col: str,
    rater_col: str,
    label_col: str,
) -> float:
    """Compute nominal Cohen's kappa for exactly two raters.

    Args:
        df: Long-format annotation data.
        item_col: Item/subject column.
        rater_col: Rater/annotator column.
        label_col: Nominal label column.

    Returns:
        Cohen's kappa value.

    Raises:
        ValueError: If input does not contain exactly two raters or has missing/duplicate ratings.
    """
    if hasattr(df, "to_pandas"):
        frame = df.to_pandas()
    else:
        frame = df

    for col in (item_col, rater_col, label_col):
        if col not in frame.columns:
            raise ValueError(f"Missing required column: {col}")

    raters = sorted(frame[rater_col].dropna().unique())
    if len(raters) != 2:
        raise ValueError(
            f"Cohen's kappa requires exactly two raters; found {len(raters)}: {raters}"
        )

    duplicate_counts = frame.groupby([item_col, rater_col]).size()
    duplicate_pairs = duplicate_counts[duplicate_counts > 1]
    if not duplicate_pairs.empty:
        raise ValueError("Cohen's kappa requires one rating per (item, rater) pair.")

    matrix = frame.pivot(index=item_col, columns=rater_col, values=label_col)
    if matrix[raters].isna().any().any():
        raise ValueError("Cohen's kappa cannot be computed with missing ratings per item.")

    first, second = raters
    labels = sorted(
        np.unique(np.concatenate([matrix[first].to_numpy(), matrix[second].to_numpy()]))
    )
    confusion = (
        matrix.groupby([first, second], observed=False).size().unstack(fill_value=0).reindex(
            index=labels, columns=labels, fill_value=0
        )
    )

    observed = confusion.to_numpy(dtype=float)
    n_items = float(observed.sum())
    if n_items == 0:
        raise ValueError("Cohen's kappa requires at least one paired item.")

    po = float(np.trace(observed) / n_items)
    row_marginals = observed.sum(axis=1) / n_items
    col_marginals = observed.sum(axis=0) / n_items
    pe = float(np.dot(row_marginals, col_marginals))

    denom = 1.0 - pe
    if abs(denom) < 1e-12:
        return 1.0 if abs(po - 1.0) < 1e-12 else 0.0

    return float((po - pe) / denom)
