"""Coverage tests for DuckDB backend behavior."""

from __future__ import annotations

from pathlib import Path
from unittest.mock import patch

import pandas as pd
import pytest

from pyrator.data.backends.duckdb import DuckDBBackend


class _Relation:
    def __init__(
        self, *, as_df: pd.DataFrame | None = None, as_pl: pd.DataFrame | None = None
    ) -> None:
        self._as_df = as_df if as_df is not None else pd.DataFrame()
        self._as_pl = as_pl if as_pl is not None else pd.DataFrame()

    def df(self) -> pd.DataFrame:
        return self._as_df

    def pl(self) -> pd.DataFrame:
        return self._as_pl


class _FakeDuckDBModule:
    def __init__(self) -> None:
        self._csv_scan_calls = 0
        self._parquet_scan_calls = 0

    def read_csv(self, path: Path, sep: str = ",") -> _Relation:
        return _Relation(as_df=pd.DataFrame({"a": [1, 2], "sep": [sep, sep]}))

    def read_json_auto(self, path: str) -> _Relation:
        return _Relation(as_df=pd.DataFrame({"a": [1, 2, 3]}))

    def read_parquet(self, path: Path) -> _Relation:
        return _Relation(as_pl=pd.DataFrame({"a": [10, 11]}))

    def sql(self, query: str) -> _Relation:
        if "read_csv" in query:
            self._csv_scan_calls += 1
            if self._csv_scan_calls == 1:
                return _Relation(as_df=pd.DataFrame({"a": [1], "b": [2]}))
            return _Relation(as_df=pd.DataFrame({"a": [], "b": []}))

        self._parquet_scan_calls += 1
        if self._parquet_scan_calls == 1:
            return _Relation(as_pl=pd.DataFrame({"a": [3], "b": [4]}))
        return _Relation(as_pl=pd.DataFrame({"a": [], "b": []}))


def test_duckdb_backend_unavailable_when_import_fails() -> None:
    """DuckDBBackend should mark itself unavailable when import fails."""
    with patch.object(DuckDBBackend, "_create_backend", side_effect=ImportError("duckdb missing")):
        backend = DuckDBBackend()
    assert backend.is_available() is False


def test_duckdb_backend_real_backend_if_installed() -> None:
    """Real initialization should work when duckdb is installed."""
    pytest.importorskip("duckdb")
    backend = DuckDBBackend()
    assert backend.is_available() is True
    assert backend.name == "duckdb"
    assert {"csv", "parquet", "streaming"}.issubset(backend.capabilities())


def test_duckdb_backend_load_methods_with_stubbed_backend() -> None:
    """Load methods should delegate to DuckDB relation helpers."""
    fake = _FakeDuckDBModule()
    with patch.object(DuckDBBackend, "_create_backend", return_value=fake):
        backend = DuckDBBackend()

    csv_out = backend.load_csv(Path("x.csv"), sep=";")
    jsonl_out = backend.load_jsonl(Path("x.jsonl"))
    parquet_out = backend.load_parquet(Path("x.parquet"))

    assert len(csv_out) == 2
    assert len(jsonl_out) == 3
    assert len(parquet_out) == 2


def test_duckdb_backend_load_jsonl_wraps_errors() -> None:
    """JSONL loading errors should be wrapped as RuntimeError."""

    class _FailingJsonDuckDB(_FakeDuckDBModule):
        def read_json_auto(self, path: str) -> _Relation:
            raise ValueError("bad json")

    with patch.object(DuckDBBackend, "_create_backend", return_value=_FailingJsonDuckDB()):
        backend = DuckDBBackend()

    with pytest.raises(RuntimeError, match="DuckDB JSONL support failed: bad json"):
        backend.load_jsonl(Path("broken.jsonl"))


def test_duckdb_backend_scan_methods_with_stubbed_backend() -> None:
    """Scan methods should yield chunked results and stop on empty batches."""
    fake = _FakeDuckDBModule()
    with patch.object(DuckDBBackend, "_create_backend", return_value=fake):
        backend = DuckDBBackend()

    csv_chunks = list(backend.scan_csv(Path("x.csv"), chunk_size=1))
    jsonl_chunks = list(backend.scan_jsonl(Path("x.jsonl"), chunk_size=2))
    parquet_chunks = list(backend.scan_parquet(Path("x.parquet"), chunk_size=1))

    assert len(csv_chunks) == 1
    assert len(jsonl_chunks) == 2
    assert len(parquet_chunks) == 1
    assert sum(len(chunk) for chunk in jsonl_chunks) == 3
