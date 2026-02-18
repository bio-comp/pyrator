# pyrator/data/backends/__init__.py
"""Compatibility exports for backend availability and conversions."""

from __future__ import annotations

from typing import Any, cast

from pyrator.data.backends import duckdb, pandas, polars, pyarrow  # noqa: F401
from pyrator.data.backends.backends import _try_import
from pyrator.types import ArrayModule

# Keep module-level dependency handles patchable for tests and legacy callers.
_pd = _try_import("pandas")
_pl = _try_import("polars")
_duckdb = _try_import("duckdb")
_cupy = _try_import("cupy")


def has_pandas() -> bool:
    return _pd is not None


def has_polars() -> bool:
    return _pl is not None


def has_duckdb() -> bool:
    return _duckdb is not None


def has_cupy() -> bool:
    return _cupy is not None


def has_ijson() -> bool:
    return _try_import("ijson") is not None


def is_pandas_dataframe(obj: Any) -> bool:
    return _pd is not None and isinstance(obj, _pd.DataFrame)


def is_polars_dataframe(obj: Any) -> bool:
    return _pl is not None and isinstance(obj, _pl.DataFrame)


def get_xp(device: str | None = None) -> ArrayModule:
    """Return NumPy by default and CuPy when `device='gpu'`."""
    if device == "gpu":
        if _cupy is None:
            raise RuntimeError("device='gpu' requested but CuPy is not installed.")
        return cast(ArrayModule, _cupy)

    import numpy as _np

    return cast(ArrayModule, _np)


def to_pandas(df_like: Any) -> Any:
    """Convert DataFrame-like input to pandas DataFrame."""
    if _pd is None:
        raise RuntimeError("pandas is required for this operation.")
    if _pl is not None and isinstance(df_like, _pl.DataFrame):
        return df_like.to_pandas()
    if isinstance(df_like, _pd.DataFrame):
        return df_like
    return _pd.DataFrame(df_like)


def to_polars(df_like: Any) -> Any:
    """Convert DataFrame-like input to polars DataFrame."""
    if _pl is None:
        raise RuntimeError("polars is required for this operation.")
    if isinstance(df_like, _pl.DataFrame):
        return df_like
    return _pl.DataFrame(df_like)


__all__ = [
    "polars",
    "pandas",
    "duckdb",
    "pyarrow",
    "_try_import",
    "get_xp",
    "has_cupy",
    "has_duckdb",
    "has_ijson",
    "has_pandas",
    "has_polars",
    "is_pandas_dataframe",
    "is_polars_dataframe",
    "to_pandas",
    "to_polars",
]
