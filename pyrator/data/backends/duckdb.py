"""DuckDB backend implementation.

This module provides a DuckDB-based implementation of the DataBackend protocol,
offering SQL-based data processing with direct API calls.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Iterator

from pyrator.data.backends.base import BaseBackend
from pyrator.data.registry import BackendRegistry
from pyrator.types import FrameLike


@BackendRegistry.register("duckdb", priority=2)
class DuckDBBackend(BaseBackend):
    """DuckDB-based data backend."""

    backend_module = "duckdb"

    def _create_backend(self):
        import duckdb

        return duckdb

    @property
    def name(self) -> str:
        return "duckdb"

    def capabilities(self) -> set[str]:
        return {"csv", "parquet", "streaming"}

    def _duckdb_backend(self) -> Any:
        backend = self._backend
        if backend is None:
            raise RuntimeError("duckdb backend is unavailable; install duckdb to use this backend.")
        return backend

    @staticmethod
    def _escape_sql_literal(value: str) -> str:
        """Escape single quotes for SQL string literals."""
        return value.replace("'", "''")

    @staticmethod
    def _fetch_df_chunk(fetch_df_chunk: Any, chunk_size: int) -> Any:
        """Call DuckDB chunk fetch with compatibility across API variants."""
        try:
            return fetch_df_chunk(chunk_size)
        except TypeError:
            pass

        try:
            return fetch_df_chunk(rows_per_batch=chunk_size)
        except TypeError:
            pass

        try:
            return fetch_df_chunk(chunk_size=chunk_size)
        except TypeError:
            pass

        return fetch_df_chunk()

    @staticmethod
    def _fetch_record_batch(fetch_record_batch: Any, chunk_size: int) -> Any:
        """Call DuckDB record-batch fetch with compatibility across API variants."""
        try:
            return fetch_record_batch(chunk_size)
        except TypeError:
            pass

        try:
            return fetch_record_batch(rows_per_batch=chunk_size)
        except TypeError:
            pass

        return fetch_record_batch()

    def _iter_relation_chunks(self, relation: Any, chunk_size: int) -> Iterator[Any]:
        """Yield relation data in chunks without repeated SQL re-scans."""
        fetch_df_chunk = getattr(relation, "fetch_df_chunk", None)
        if callable(fetch_df_chunk):
            while True:
                batch = self._fetch_df_chunk(fetch_df_chunk, chunk_size)
                if len(batch) == 0:
                    return
                yield batch

        fetch_record_batch = getattr(relation, "fetch_record_batch", None)
        if callable(fetch_record_batch):
            reader = self._fetch_record_batch(fetch_record_batch, chunk_size)
            for record_batch in reader:
                batch = record_batch.to_pandas()
                if len(batch) > 0:
                    yield batch
            return

        # Fallback for older APIs: one-pass materialization with local slicing.
        batch = relation.df()
        if len(batch) == 0:
            return
        for start in range(0, len(batch), chunk_size):
            yield batch.iloc[start : start + chunk_size]

    def load_csv(self, path: Path, sep: str = ",", **kwargs: Any) -> FrameLike:
        """Load CSV file using DuckDB."""
        from loguru import logger

        logger.debug(f"Loading CSV with DuckDB: {path.name}")
        backend = self._duckdb_backend()
        return backend.read_csv(path, sep=sep).df()

    def load_jsonl(self, path: Path, **kwargs: Any) -> FrameLike:
        """Load JSONL file using DuckDB.

        Note: DuckDB doesn't have native JSONL support, so this
        uses a read_jsonlines function if available.
        """
        from loguru import logger

        logger.warning("DuckDB backend: JSONL support limited, using experimental approach.")

        # Try to read as newline-delimited JSON
        try:
            backend = self._duckdb_backend()
            # DuckDB can parse JSON lines with read_json_auto
            return backend.read_json_auto(str(path)).df()
        except Exception as e:
            raise RuntimeError(f"DuckDB JSONL support failed: {e}")

    def load_parquet(self, path: Path, **kwargs: Any) -> FrameLike:
        """Load Parquet file using DuckDB."""
        from loguru import logger

        logger.debug(f"Loading Parquet with DuckDB: {path.name}")
        backend = self._duckdb_backend()
        # Use direct API call, note .pl() returns Polars DataFrame
        return backend.read_parquet(path).pl()

    def scan_csv(
        self, path: Path, chunk_size: int, sep: str = ",", **kwargs: Any
    ) -> Iterator[FrameLike]:
        """Scan CSV file in chunks using DuckDB."""
        from loguru import logger

        logger.debug(f"Scanning CSV with DuckDB: {path.name}")

        backend = self._duckdb_backend()
        chunk_size_int = int(chunk_size)
        path_sql = self._escape_sql_literal(str(path))
        sep_sql = self._escape_sql_literal(sep)
        query = f"SELECT * FROM read_csv('{path_sql}', header=True, sep='{sep_sql}')"
        relation = backend.sql(query)
        yield from self._iter_relation_chunks(relation, chunk_size_int)

    def scan_jsonl(self, path: Path, chunk_size: int, **kwargs: Any) -> Iterator[FrameLike]:
        """Scan JSONL file in chunks using DuckDB."""
        from loguru import logger

        logger.debug(f"Scanning JSONL with DuckDB: {path.name}")

        backend = self._duckdb_backend()
        chunk_size_int = int(chunk_size)
        path_sql = self._escape_sql_literal(str(path))
        query = f"SELECT * FROM read_json_auto('{path_sql}')"
        relation = backend.sql(query)
        yield from self._iter_relation_chunks(relation, chunk_size_int)

    def scan_parquet(self, path: Path, chunk_size: int, **kwargs: Any) -> Iterator[FrameLike]:
        """Scan Parquet file in chunks using DuckDB."""
        from loguru import logger

        logger.debug(f"Scanning Parquet with DuckDB: {path.name}")

        backend = self._duckdb_backend()
        chunk_size_int = int(chunk_size)
        path_sql = self._escape_sql_literal(str(path))
        query = f"SELECT * FROM read_parquet('{path_sql}')"
        relation = backend.sql(query)
        yield from self._iter_relation_chunks(relation, chunk_size_int)
