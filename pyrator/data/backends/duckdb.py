"""DuckDB backend implementation.

This module provides a DuckDB-based implementation of the DataBackend protocol,
offering SQL-based data processing with direct API calls.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Iterator, Set

from pyrator.data.backends.base import BaseBackend
from pyrator.data.registry import BackendRegistry
from pyrator.types import FrameLike


@BackendRegistry.register("duckdb", priority=2)
class DuckDBBackend(BaseBackend):
    """DuckDB-based data backend."""

    def _create_backend(self):
        import duckdb

        return duckdb

    @property
    def name(self) -> str:
        return "duckdb"

    def capabilities(self) -> Set[str]:
        return {"csv", "parquet", "streaming"}

    def load_csv(self, path: Path, sep: str = ",", **kwargs: Any) -> FrameLike:
        """Load CSV file using DuckDB."""
        from loguru import logger

        logger.debug(f"Loading CSV with DuckDB: {path.name}")
        return self._backend.read_csv(path, sep=sep).df()

    def load_jsonl(self, path: Path, **kwargs: Any) -> FrameLike:
        """Load JSONL file using DuckDB.

        Note: DuckDB doesn't have native JSONL support, so this
        uses a read_jsonlines function if available.
        """
        from loguru import logger

        logger.warning("DuckDB backend: JSONL support limited, using experimental approach.")

        # Try to read as newline-delimited JSON
        try:
            # DuckDB can parse JSON lines with read_json_auto
            return self._backend.read_json_auto(str(path)).df()
        except Exception as e:
            raise RuntimeError(f"DuckDB JSONL support failed: {e}")

    def load_parquet(self, path: Path, **kwargs: Any) -> FrameLike:
        """Load Parquet file using DuckDB."""
        from loguru import logger

        logger.debug(f"Loading Parquet with DuckDB: {path.name}")
        # Use direct API call, note .pl() returns Polars DataFrame
        return self._backend.read_parquet(path).pl()

    def scan_csv(
        self, path: Path, chunk_size: int, sep: str = ",", **kwargs: Any
    ) -> Iterator[FrameLike]:
        """Scan CSV file in chunks using DuckDB."""
        from loguru import logger

        logger.debug(f"Scanning CSV with DuckDB: {path.name}")

        chunk_size_int = int(chunk_size)
        offset = 0

        while True:
            query = (
                f"SELECT * FROM read_csv('{path}', header=True, sep='{sep}') "
                f"LIMIT {chunk_size_int} OFFSET {offset}"
            )
            batch = self._backend.sql(query).df()
            if len(batch) == 0:
                break
            yield batch
            offset += chunk_size_int

    def scan_jsonl(self, path: Path, chunk_size: int, **kwargs: Any) -> Iterator[FrameLike]:
        """Scan JSONL file in chunks using DuckDB.

        Uses LIMIT/OFFSET with row_number to stream without loading entire file.
        """
        from loguru import logger

        logger.debug(f"Scanning JSONL with DuckDB: {path.name}")

        chunk_size_int = int(chunk_size)
        offset = 0

        while True:
            query = (
                f"SELECT * FROM ("
                f"SELECT *, row_number() OVER() as _rn FROM read_json_auto('{path}')"
                f") WHERE _rn > {offset} AND _rn <= {offset + chunk_size_int}"
            )
            batch = self._backend.sql(query).df()
            if len(batch) == 0:
                break
            yield batch
            offset += chunk_size_int

    def scan_parquet(self, path: Path, chunk_size: int, **kwargs: Any) -> Iterator[FrameLike]:
        """Scan Parquet file in chunks using DuckDB."""
        from loguru import logger

        logger.debug(f"Scanning Parquet with DuckDB: {path.name}")

        chunk_size_int = int(chunk_size)
        offset = 0

        while True:
            query = f"SELECT * FROM read_parquet('{path}') LIMIT {chunk_size_int} OFFSET {offset}"
            batch = self._backend.sql(query).pl()
            if len(batch) == 0:
                break
            yield batch
            offset += chunk_size_int
