"""Polars backend implementation.

This module provides a Polars-based implementation of the DataBackend protocol,
offering high-performance data loading and streaming capabilities.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Iterator, Set

from pyrator.data.backends.base import BaseBackend
from pyrator.data.registry import BackendRegistry
from pyrator.types import FrameLike


@BackendRegistry.register("polars", priority=0)
class PolarsBackend(BaseBackend):
    """Polars-based data backend."""

    backend_module = "polars"

    def _create_backend(self):
        import polars

        return polars

    @property
    def name(self) -> str:
        return "polars"

    def capabilities(self) -> Set[str]:
        return {"csv", "jsonl", "parquet", "streaming"}

    def load_csv(self, path: Path, sep: str = ",", **kwargs: Any) -> FrameLike:
        """Load CSV file using Polars."""
        from loguru import logger

        logger.debug(f"Loading CSV with Polars: {path.name}")
        return self._backend.read_csv(path, separator=sep)

    def load_jsonl(self, path: Path, **kwargs: Any) -> FrameLike:
        """Load JSONL file using Polars."""
        return self._backend.read_ndjson(path)

    def load_parquet(self, path: Path, **kwargs: Any) -> FrameLike:
        """Load Parquet file using Polars."""
        return self._backend.read_parquet(path)

    def scan_csv(
        self, path: Path, chunk_size: int, sep: str = ",", **kwargs: Any
    ) -> Iterator[FrameLike]:
        """Scan CSV file in chunks using Polars."""
        from loguru import logger

        logger.debug(f"Scanning CSV with Polars: {path.name}")

        chunk_size_int = int(chunk_size)
        lf = self._backend.scan_csv(path, separator=sep)
        offset = 0
        while True:
            batch = lf.slice(offset, chunk_size_int).collect(engine="streaming")
            if not len(batch):
                break
            yield batch
            offset += chunk_size_int

    def scan_jsonl(self, path: Path, chunk_size: int, **kwargs: Any) -> Iterator[FrameLike]:
        """Scan JSONL file in chunks using Polars."""
        from loguru import logger

        logger.debug(f"Scanning JSONL with Polars: {path.name}")

        chunk_size_int = int(chunk_size)
        lf = self._backend.scan_ndjson(path)
        offset = 0
        while True:
            batch = lf.slice(offset, chunk_size_int).collect(engine="streaming")
            if not len(batch):
                break
            yield batch
            offset += chunk_size_int

    def scan_parquet(self, path: Path, chunk_size: int, **kwargs: Any) -> Iterator[FrameLike]:
        """Scan Parquet file in chunks using Polars."""
        from loguru import logger

        logger.debug(f"Scanning Parquet with Polars: {path.name}")

        chunk_size_int = int(chunk_size)
        lf = self._backend.scan_parquet(str(path))
        offset = 0
        while True:
            batch = lf.slice(offset, chunk_size_int).collect(engine="streaming")
            if not len(batch):
                break
            yield batch
            offset += chunk_size_int
