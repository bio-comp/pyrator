"""Kappa-family agreement metrics."""

from __future__ import annotations

from typing import Literal, Optional

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
        matrix.groupby([first, second], observed=False)
        .size()
        .unstack(fill_value=0)
        .reindex(index=labels, columns=labels, fill_value=0)
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


def fleiss_kappa(  # noqa: C901
    df: FrameLike,
    *,
    item_col: str,
    rater_col: str,
    label_col: str,
    autodrop: Optional[Literal["incomplete_items", "rare_labels", "all"]] = None,
) -> float:
    """Compute Fleiss' kappa for many-rater nominal agreement.

    Args:
        df: Long-format annotation data.
        item_col: Item/subject column.
        rater_col: Rater/annotator column.
        label_col: Nominal label column.
        autodrop: Optional policy for handling incomplete data:
            - None: Strict mode (default) - require complete rating matrix
            - "incomplete_items": Drop items with fewer ratings than the maximum
            - "rare_labels": Drop labels that appear in fewer than 2 items
            - "all": Apply both policies

    Returns:
        Fleiss' kappa value.

    Raises:
        ValueError: If input is malformed or (in strict mode) items don't have equal ratings.
    """
    if hasattr(df, "to_pandas"):
        frame = df.to_pandas()
    else:
        frame = df

    for col in (item_col, rater_col, label_col):
        if col not in frame.columns:
            raise ValueError(f"Missing required column: {col}")

    duplicate_counts = frame.groupby([item_col, rater_col]).size()
    duplicate_pairs = duplicate_counts[duplicate_counts > 1]
    if not duplicate_pairs.empty:
        raise ValueError("Fleiss' kappa requires one rating per (item, rater) pair.")

    counts_by_item = frame.groupby(item_col).size()
    if counts_by_item.empty:
        raise ValueError("Fleiss' kappa requires at least one rated item.")

    unique_counts = counts_by_item.unique()
    max_ratings = int(unique_counts.max())

    if len(unique_counts) != 1:
        if autodrop is None:
            raise ValueError(
                "Fleiss' kappa strictly requires the same number of ratings per item. "
                "For varying raters or missing data, use Krippendorff's Alpha instead, "
                "or set autodrop to 'incomplete_items', 'rare_labels', or 'all'."
            )
        if autodrop in ("incomplete_items", "all"):
            frame = frame[frame[item_col].isin(counts_by_item[counts_by_item == max_ratings].index)]

    label_counts = frame.groupby(label_col)[item_col].nunique()
    if autodrop in ("rare_labels", "all"):
        frame = frame[frame[label_col].isin(label_counts[label_counts >= 2].index)]

    if frame.empty:
        raise ValueError("No items or labels remain after autodrop.")

    counts_by_item = frame.groupby(item_col).size()
    unique_counts = counts_by_item.unique()
    if len(unique_counts) != 1:
        raise ValueError(
            "After dropping rare labels, remaining items have unequal ratings. "
            "Consider using Krippendorff's Alpha instead."
        )

    ratings_per_item = int(unique_counts[0])
    if ratings_per_item < 2:
        raise ValueError("Fleiss' kappa requires at least two ratings per item.")

    item_label_counts = frame.groupby([item_col, label_col]).size().unstack(fill_value=0)
    observed = item_label_counts.to_numpy(dtype=float)

    n_items = observed.shape[0]
    n = float(ratings_per_item)

    # Fleiss (1971): per-item agreement P_i and expected agreement from pooled marginals.
    p_i = (np.sum(observed**2, axis=1) - n) / (n * (n - 1.0))
    p_bar = float(np.mean(p_i))

    p_j = np.sum(observed, axis=0) / (n_items * n)
    p_e_bar = float(np.sum(p_j**2))

    denom = 1.0 - p_e_bar
    if abs(denom) < 1e-12:
        return 1.0 if abs(p_bar - 1.0) < 1e-12 else 0.0

    return float((p_bar - p_e_bar) / denom)
