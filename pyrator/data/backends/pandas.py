# data/backends/pandas.py
"""Pandas backend implementation.

This module provides a Pandas-based implementation of the DataBackend protocol,
offering broad compatibility and reliability.
"""

from __future__ import annotations

from collections.abc import Iterator
from pathlib import Path
from typing import Any

from pyrator.data.backends.base import BaseBackend
from pyrator.data.registry import BackendRegistry
from pyrator.types import FrameLike


@BackendRegistry.register("pandas", priority=1)
class PandasBackend(BaseBackend):
    """Pandas-based data backend."""

    backend_module = "pandas"

    def _create_backend(self) -> Any:
        import pandas

        return pandas

    @property
    def name(self) -> str:
        return "pandas"

    def capabilities(self) -> set[str]:
        # Pandas supports chunked CSV/JSONL scans without pyarrow.
        # Parquet scanning still validates pyarrow at call time.
        return {"csv", "jsonl", "parquet", "streaming"}

    def _pandas_backend(self) -> Any:
        """Return initialized pandas module or raise a dependency error."""
        backend = self._backend
        if backend is None:
            raise RuntimeError("pandas backend is unavailable; install pandas to use this backend.")
        return backend

    def load_csv(self, path: Path, sep: str = ",", **kwargs: Any) -> FrameLike:
        """Load CSV file using Pandas."""
        from loguru import logger

        logger.debug(f"Loading CSV with Pandas: {path.name}")
        backend = self._pandas_backend()
        return backend.read_csv(path, sep=sep)

    def load_jsonl(self, path: Path, **kwargs: Any) -> FrameLike:
        """Load JSONL file using Pandas."""
        backend = self._pandas_backend()
        return backend.read_json(str(path), lines=True)

    def load_parquet(self, path: Path, **kwargs: Any) -> FrameLike:
        """Load Parquet file using Pandas."""
        backend = self._pandas_backend()
        return backend.read_parquet(path)

    def scan_csv(
        self, path: Path, chunk_size: int, sep: str = ",", **kwargs: Any
    ) -> Iterator[FrameLike]:
        """Scan CSV file in chunks using Pandas."""
        from loguru import logger

        logger.debug(f"Scanning CSV with Pandas: {path.name}")

        backend = self._pandas_backend()
        chunk_size_int = int(chunk_size)
        reader = backend.read_csv(str(path), sep=sep, chunksize=chunk_size_int)
        for chunk in reader:
            yield chunk

    def scan_jsonl(self, path: Path, chunk_size: int, **kwargs: Any) -> Iterator[FrameLike]:
        """Scan JSONL file in chunks using Pandas."""
        from loguru import logger

        logger.debug(f"Scanning JSONL with Pandas: {path.name}")

        try:
            import orjson
        except ImportError:
            raise RuntimeError("orjson is required for pandas JSONL scanner.")

        backend = self._pandas_backend()
        chunk_size_int = int(chunk_size)
        buf = []
        with open(path, "rb") as f:
            for line in f:
                if line.strip():
                    buf.append(orjson.loads(line))
                    if len(buf) >= chunk_size_int:
                        yield backend.DataFrame(buf)
                        buf.clear()
        if buf:
            yield backend.DataFrame(buf)

    def scan_parquet(self, path: Path, chunk_size: int, **kwargs: Any) -> Iterator[FrameLike]:
        """Scan Parquet file in chunks using PyArrow batch iteration."""
        from loguru import logger

        try:
            import pyarrow.parquet as pq
        except ImportError as exc:
            raise RuntimeError(
                "pyarrow is required for pandas parquet scanner. "
                "Install pyarrow or use a different backend."
            ) from exc

        logger.debug(f"Scanning Parquet with Pandas via PyArrow batches: {path.name}")
        chunk_size_int = int(chunk_size)

        parquet_file = pq.ParquetFile(path)
        for batch in parquet_file.iter_batches(batch_size=chunk_size_int):
            yield batch.to_pandas()
