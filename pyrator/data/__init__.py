# pyrator/data/__init__.py
from pyrator.data.backends import get_xp, has_cupy, has_polars
from pyrator.data.batching import BatchIterator
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
    "BatchIterator",
]
