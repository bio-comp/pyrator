# pyrator/data/__init__.py
"""Data processing module for pyrator."""

# Trigger backend registration first.
# Importing these modules executes @BackendRegistry.register decorators.
from pyrator.data.backends import duckdb, pandas, polars, pyarrow  # noqa: F401
from pyrator.data.backends.backends import get_xp, has_cupy, has_polars
from pyrator.data.batching import batch_data, process_in_batches
from pyrator.data.canonical import explode_multilabel, to_long_canonical
from pyrator.data.encoders import CategoryEncoder
from pyrator.data.loaders import (
    load_any,
    load_csv,
    load_jsonl,
    load_parquet,
    scan_csv,
    scan_jsonl,
    scan_parquet,
)

__all__ = [
    "has_polars",
    "has_cupy",
    "get_xp",
    "load_any",
    "load_csv",
    "load_jsonl",
    "load_parquet",
    "scan_csv",
    "scan_jsonl",
    "scan_parquet",
    "to_long_canonical",
    "explode_multilabel",
    "CategoryEncoder",
    "batch_data",
    "process_in_batches",
]
