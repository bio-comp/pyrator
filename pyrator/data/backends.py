# pyrator/data/backends.py
from __future__ import annotations

from typing import Any, cast

from pyrator.types import ArrayModule


def _try_import(name: str) -> Any | None:
    """Attempts to import a library, returning None if it fails."""
    try:
        return __import__(name)
    except Exception:
        return None


# --- Attempt to import optional dependencies ---
_pd = _try_import("pandas")
_pl = _try_import("polars")
_duckdb = _try_import("duckdb")
_cupy = _try_import("cupy")


# --- Boolean checks for dependency availability ---
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


# --- Backend Dispatcher ---
def get_xp(device: str | None = None) -> ArrayModule:
    """
    Return the array module (NumPy default, CuPy if device='gpu').

    The 'xp' alias is a common convention for a swappable array library.
    """
    if device == "gpu":
        if _cupy is None:
            raise RuntimeError("device='gpu' requested but CuPy is not installed.")
        # Tell Mypy to trust that the _cupy module satisfies the protocol
        return cast(ArrayModule, _cupy)

    import numpy as _np

    # Tell Mypy to trust that the _np module satisfies the protocol
    return cast(ArrayModule, _np)


def to_pandas(df_like: Any) -> Any:
    """
    Robustly converts a DataFrame-like object to a pandas DataFrame.

    Raises:
        RuntimeError: If pandas is not installed.
    """
    # 1. First, guard against the primary dependency being missing.
    if _pd is None:
        raise RuntimeError("pandas is required for this operation.")

    # 2. Handle the special case: converting from a polars DataFrame.
    if _pl is not None and isinstance(df_like, _pl.DataFrame):
        return df_like.to_pandas()

    # 3. For all other cases (dicts, lists, or already a pandas DataFrame),
    return _pd.DataFrame(df_like)


def to_polars(df_like: Any) -> Any:
    """
    Robustly converts a DataFrame-like object to a Polars DataFrame.

    Raises:
        RuntimeError: If polars is not installed.
    """
    if _pl is not None:
        # Inside this block, Mypy knows _pl is the polars module.
        if isinstance(df_like, _pl.DataFrame):
            return df_like

        return _pl.DataFrame(df_like)
    else:
        # The None case is handled here.
        raise RuntimeError("polars is required for this operation.")
