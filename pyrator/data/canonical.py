# pyrator/data/canonical.py
from typing import Literal

from loguru import logger

from pyrator.data.backends import has_polars, to_pandas, to_polars
from pyrator.types import FrameLike


def to_long_canonical(
    df_like: FrameLike,
    *,
    item_col: str,
    annotator_col: str,
    label_col: str,
    wide_annotator_cols: list[str] | None = None,
) -> FrameLike:
    """Convert data to long canonical format for interrater analysis."""

    # Preserve input type if it's already a DataFrame-like
    from pyrator.data.backends import _pd

    # Use pandas path for dict/list inputs, polars only for explicit polars DataFrames
    if (
        has_polars()
        and _pd is not None
        and not isinstance(df_like, _pd.DataFrame)
        and not isinstance(df_like, dict)
        and not isinstance(df_like, list)
    ):
        import polars as pl

        df = to_polars(df_like)

        if wide_annotator_cols:
            # Validate columns exist
            missing = set(wide_annotator_cols) - set(df.columns)
            if missing:
                raise ValueError(f"Missing annotator columns: {missing}")

            # Check if df is polars or pandas and use appropriate method
            if hasattr(df, "unpivot"):  # polars DataFrame
                return (
                    df.unpivot(
                        index=[item_col],
                        on=wide_annotator_cols,
                        variable_name=annotator_col,
                        value_name=label_col,
                    )
                    .drop_nulls(subset=[label_col])
                    .filter(pl.col("label").is_not_null())
                )
            else:  # pandas DataFrame
                return df.unpivot(
                    index=[item_col],
                    on=wide_annotator_cols,
                    variable_name=annotator_col,
                    value_name=label_col,
                ).dropna(subset=[label_col])

        # Validate columns for long format (when wide_annotator_cols is None)
        required = [item_col, annotator_col, label_col]
        missing = set(required) - set(df.columns)
        if missing:
            raise ValueError(f"Missing columns: {missing}")

        return df.select(required)

    # Pandas path
    df = to_pandas(df_like)

    if wide_annotator_cols:
        missing = set(wide_annotator_cols) - set(df.columns)
        if missing:
            raise ValueError(f"Missing annotator columns: {missing}")

        return df.melt(
            id_vars=[item_col],
            value_vars=wide_annotator_cols,
            var_name=annotator_col,
            value_name=label_col,
        ).dropna(subset=[label_col])[[item_col, annotator_col, label_col]]

    required = [item_col, annotator_col, label_col]
    missing = set(required) - set(df.columns)
    if missing:
        raise ValueError(f"Missing columns: {missing}")

    return df[required].copy()  # Return copy, not view


def explode_multilabel(  # noqa: C901
    df_like: FrameLike,
    *,
    label_col: str,
    mode: Literal["auto", "json", "delim", "list"] = "auto",
    delim: str = "|",
) -> FrameLike:
    """Explode multi-label columns into separate rows."""

    # Preserve input type if it's already a DataFrame-like
    from pyrator.data.backends import _pd

    if has_polars() and not (_pd is None or isinstance(df_like, _pd.DataFrame)):
        import polars as pl

        df = to_polars(df_like)

        if label_col not in df.columns:
            raise ValueError(f"Column '{label_col}' not found")

        col_type = df.schema.get(label_col)

        # Handle list types
        if mode in ("auto", "list") and isinstance(col_type, pl.List):
            logger.debug(f"Exploding {label_col} as list type")
            return df.explode(label_col)

        # Handle delimited strings
        if mode in ("auto", "delim"):
            logger.debug(f"Exploding {label_col} with delimiter '{delim}'")
            return df.with_columns(pl.col(label_col).cast(pl.Utf8).str.split(delim)).explode(
                label_col
            )

        # Handle JSON arrays
        if mode == "json":
            logger.debug(f"Exploding {label_col} as JSON")
            # Use native JSON extraction if possible
            return df.with_columns(
                pl.col(label_col).str.json_decode(dtype=pl.List(pl.Utf8))
            ).explode(label_col)

        return df

    # Pandas path
    df = to_pandas(df_like).copy()

    if label_col not in df.columns:
        raise ValueError(f"Column '{label_col}' not found")

    # Check for list type more efficiently
    if mode in ("auto", "list"):
        sample = df[label_col].dropna().iloc[0] if not df[label_col].dropna().empty else None
        if isinstance(sample, list):
            logger.debug(f"Exploding {label_col} as list type")
            return df.explode(label_col).dropna(subset=[label_col])

    # Handle delimited strings
    if mode in ("auto", "delim") and df[label_col].dtype == "object":
        logger.debug(f"Exploding {label_col} with delimiter '{delim}'")
        df[label_col] = df[label_col].astype(str).str.split(delim)
        return df.explode(label_col).dropna(subset=[label_col])

    # Handle JSON arrays
    if mode == "json":
        import orjson

        logger.debug(f"Exploding {label_col} as JSON")
        df[label_col] = df[label_col].apply(orjson.loads)
        return df.explode(label_col).dropna(subset=[label_col])

    return df
