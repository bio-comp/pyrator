"""Pandas backend implementation.

This module provides a Pandas-based implementation of the DataBackend protocol,
offering broad compatibility and reliability.
"""

from __future__ import annotations

from typing import Iterator, Any, Set
from pathlib import Path

from pyrator.data.registry import BackendRegistry
from pyrator.data.backends.base import BaseBackend
from pyrator.types import FrameLike


@BackendRegistry.register("pandas", priority=1)
class PandasBackend(BaseBackend):
    """Pandas-based data backend."""

    def _create_backend(self):
        import pandas

        return pandas

    @property
    def name(self) -> str:
        return "pandas"

    def capabilities(self) -> Set[str]:
        return {"csv", "jsonl", "parquet"}

    def load_csv(self, path: Path, sep: str = ",", **kwargs: Any) -> FrameLike:
        """Load CSV file using Pandas."""
        from loguru import logger

        logger.debug(f"Loading CSV with Pandas: {path.name}")
        return self._backend.read_csv(path, sep=sep)

    def load_jsonl(self, path: Path, **kwargs: Any) -> FrameLike:
        """Load JSONL file using Pandas."""
        return self._backend.read_json(str(path), lines=True)

    def load_parquet(self, path: Path, **kwargs: Any) -> FrameLike:
        """Load Parquet file using Pandas."""
        return self._backend.read_parquet(path)

    def scan_csv(
        self, path: Path, chunk_size: int, sep: str = ",", **kwargs: Any
    ) -> Iterator[FrameLike]:
        """Scan CSV file in chunks using Pandas."""
        from loguru import logger

        logger.debug(f"Scanning CSV with Pandas: {path.name}")

        chunk_size_int = int(chunk_size)
        reader = self._backend.read_csv(str(path), sep=sep, chunksize=chunk_size_int)
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

        chunk_size_int = int(chunk_size)
        buf = []
        with open(path, "rb") as f:
            for line in f:
                if line.strip():
                    buf.append(orjson.loads(line))
                    if len(buf) >= chunk_size_int:
                        yield self._backend.DataFrame(buf)
                        buf.clear()
        if buf:
            yield self._backend.DataFrame(buf)

    def scan_parquet(self, path: Path, chunk_size: int, **kwargs: Any) -> Iterator[FrameLike]:
        """Scan Parquet file in chunks using Pandas.

        Note: Pandas doesn't have native Parquet streaming, so this
        loads the entire file into memory and yields chunks.
        """
        from loguru import logger

        logger.warning("Pandas backend: Parquet streaming not efficient, loading entire file.")

        df = self._backend.read_parquet(path)
        chunk_size_int = int(chunk_size)
        total_rows = len(df)

        for i in range(0, total_rows, chunk_size_int):
            yield df.iloc[i : i + chunk_size_int]
