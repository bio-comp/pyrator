"""PyArrow backend implementation.

This module provides a PyArrow-based implementation of the DataBackend protocol,
specializing in efficient Parquet file streaming and processing.
"""

from __future__ import annotations

from collections.abc import Iterator
from pathlib import Path
from typing import Any

from pyrator.data.backends.base import BaseBackend
from pyrator.data.registry import BackendRegistry
from pyrator.types import FrameLike


@BackendRegistry.register("pyarrow", priority=3)
class PyArrowBackend(BaseBackend):
    """PyArrow-based data backend."""

    backend_module = "pyarrow.parquet"

    def _create_backend(self):
        import pyarrow.parquet as pq

        return pq

    @property
    def name(self) -> str:
        return "pyarrow"

    def capabilities(self) -> set[str]:
        return {"parquet", "streaming"}

    def load_csv(self, path: Path, sep: str = ",", **kwargs: Any) -> FrameLike:
        """CSV loading is intentionally not supported by the PyArrow backend."""
        raise NotImplementedError("PyArrow backend supports Parquet only for loading.")

    def load_jsonl(self, path: Path, **kwargs: Any) -> FrameLike:
        """JSONL loading is intentionally not supported by the PyArrow backend."""
        raise NotImplementedError("PyArrow backend supports Parquet only for loading.")

    def load_parquet(self, path: Path, **kwargs: Any) -> FrameLike:
        """Load Parquet file using PyArrow."""
        from loguru import logger

        logger.debug(f"Loading Parquet with PyArrow: {path.name}")

        parquet_file = self._backend.ParquetFile(path)
        table = parquet_file.read_table()
        return table.to_pandas()

    def scan_csv(
        self, path: Path, chunk_size: int, sep: str = ",", **kwargs: Any
    ) -> Iterator[FrameLike]:
        """CSV scanning is intentionally not supported by the PyArrow backend."""
        raise NotImplementedError("PyArrow backend supports Parquet only for scanning.")

    def scan_jsonl(self, path: Path, chunk_size: int, **kwargs: Any) -> Iterator[FrameLike]:
        """JSONL scanning is intentionally not supported by the PyArrow backend."""
        raise NotImplementedError("PyArrow backend supports Parquet only for scanning.")

    def scan_parquet(self, path: Path, chunk_size: int, **kwargs: Any) -> Iterator[FrameLike]:
        """Scan Parquet file in chunks using PyArrow."""
        from loguru import logger

        logger.debug(f"Scanning Parquet with PyArrow: {path.name}")

        chunk_size_int = int(chunk_size)
        parquet_file = self._backend.ParquetFile(path)
        for batch in parquet_file.iter_batches(batch_size=chunk_size_int):
            yield batch.to_pandas()
