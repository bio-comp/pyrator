"""Data backend implementations.

This package contains concrete implementations of the DataBackend protocol,
providing support for various data processing libraries like Polars,
Pandas, DuckDB, and PyArrow.
"""

# Import backend implementations directly to avoid circular imports
from pyrator.data.backends import duckdb, pandas, polars, pyarrow  # noqa: F401

# Import the compatibility functions from the main backends module
from pyrator.data.backends.backends import (
    get_xp,
    has_cupy,
    has_duckdb,
    has_pandas,
    has_polars,
    is_pandas_dataframe,
    is_polars_dataframe,
)

# Re-export for backward compatibility
__all__ = [
    "polars",
    "pandas",
    "duckdb",
    "pyarrow",
    "has_polars",
    "has_pandas",
    "has_duckdb",
    "get_xp",
    "has_cupy",
    "is_pandas_dataframe",
    "is_polars_dataframe",
]
