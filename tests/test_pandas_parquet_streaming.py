# tests/test_pandas_parquet_streaming.py
"""Regression tests for pandas parquet scanner behavior."""

from __future__ import annotations

from pathlib import Path
from unittest.mock import patch

import pandas as pd
import pytest

from pyrator.data.backends.backends import has_pandas
from pyrator.data.backends.pandas import PandasBackend


@pytest.fixture()
def parquet_file(tmp_path: Path) -> Path:
    """Create a small Parquet file for scanner tests."""
    if not has_pandas():
        pytest.skip("Pandas is not available")

    path = tmp_path / "test.parquet"
    frame = pd.DataFrame({"a": [1, 2, 3, 4, 5], "b": [10, 11, 12, 13, 14]})
    try:
        frame.to_parquet(path)
    except Exception as exc:
        pytest.skip(f"Unable to create parquet file for test: {exc}")

    return path


def test_pandas_scan_parquet_streams_in_batches(parquet_file: Path) -> None:
    """Pandas backend should stream parquet data by chunk size."""
    backend = PandasBackend()
    chunks = list(backend.scan_parquet(parquet_file, chunk_size=2))
    pd_chunks = [chunk for chunk in chunks if isinstance(chunk, pd.DataFrame)]

    assert len(chunks) == 3
    assert len(pd_chunks) == len(chunks)
    assert [len(chunk) for chunk in pd_chunks] == [2, 2, 1]
    assert sum((len(chunk) for chunk in pd_chunks), start=0) == 5


def test_pandas_scan_parquet_does_not_use_read_parquet(parquet_file: Path) -> None:
    """Scanner should not load the full parquet file through pandas.read_parquet."""
    backend = PandasBackend()

    with patch.object(
        backend._backend,
        "read_parquet",
        side_effect=AssertionError("read_parquet should not be used"),
    ):
        chunks = list(backend.scan_parquet(parquet_file, chunk_size=2))
    pd_chunks = [chunk for chunk in chunks if isinstance(chunk, pd.DataFrame)]

    assert len(pd_chunks) == len(chunks)
    assert sum((len(chunk) for chunk in pd_chunks), start=0) == 5


def test_pandas_scan_parquet_requires_pyarrow(parquet_file: Path) -> None:
    """Scanner should fail clearly when pyarrow is unavailable."""
    backend = PandasBackend()
    with patch.dict("sys.modules", {"pyarrow": None, "pyarrow.parquet": None}):
        with pytest.raises(RuntimeError, match="pyarrow is required for pandas parquet scanner"):
            next(backend.scan_parquet(parquet_file, chunk_size=2))
