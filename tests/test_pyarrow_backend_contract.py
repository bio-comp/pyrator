"""Tests for PyArrow backend contract and unsupported operations."""

from __future__ import annotations

from pathlib import Path

import pytest

from pyrator.data.backends.pyarrow import PyArrowBackend


@pytest.fixture()
def backend() -> PyArrowBackend:
    """Create a PyArrow backend instance or skip when unavailable."""
    instance = PyArrowBackend()
    if not instance.is_available():
        pytest.skip("PyArrow backend dependencies are unavailable")
    return instance


def test_pyarrow_backend_capabilities_are_parquet_only(backend: PyArrowBackend) -> None:
    """PyArrow backend should advertise only parquet-related capabilities."""
    assert backend.capabilities() == {"parquet", "streaming"}


def test_pyarrow_backend_unsupported_load_methods_raise(
    backend: PyArrowBackend, tmp_path: Path
) -> None:
    """Non-parquet load operations should raise NotImplementedError."""
    csv_file = tmp_path / "sample.csv"
    csv_file.write_text("a,b\n1,2\n")
    jsonl_file = tmp_path / "sample.jsonl"
    jsonl_file.write_text('{"a": 1}\n')

    with pytest.raises(NotImplementedError, match="Parquet only"):
        backend.load_csv(csv_file)
    with pytest.raises(NotImplementedError, match="Parquet only"):
        backend.load_jsonl(jsonl_file)


def test_pyarrow_backend_unsupported_scan_methods_raise(
    backend: PyArrowBackend, tmp_path: Path
) -> None:
    """Non-parquet scan operations should raise NotImplementedError."""
    csv_file = tmp_path / "sample.csv"
    csv_file.write_text("a,b\n1,2\n")
    jsonl_file = tmp_path / "sample.jsonl"
    jsonl_file.write_text('{"a": 1}\n')

    with pytest.raises(NotImplementedError, match="Parquet only"):
        next(backend.scan_csv(csv_file, chunk_size=1))
    with pytest.raises(NotImplementedError, match="Parquet only"):
        next(backend.scan_jsonl(jsonl_file, chunk_size=1))
