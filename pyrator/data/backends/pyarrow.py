"""PyArrow backend implementation.

This module provides a PyArrow-based implementation of the DataBackend protocol,
specializing in efficient Parquet file streaming and processing.
"""

from __future__ import annotations

from typing import Iterator, Any, Set
from pathlib import Path

from pyrator.data.registry import BackendRegistry
from pyrator.data.backends.base import BaseBackend
from pyrator.types import FrameLike


@BackendRegistry.register("pyarrow", priority=3)
class PyArrowBackend(BaseBackend):
    """PyArrow-based data backend."""

    def _create_backend(self):
        import pyarrow.parquet as pq

        return pq

    @property
    def name(self) -> str:
        return "pyarrow"

    def capabilities(self) -> Set[str]:
        return {"parquet", "streaming"}

    def load_csv(self, path: Path, sep: str = ",", **kwargs: Any) -> FrameLike:
        """Load CSV file using PyArrow.

        Note: PyArrow can read CSV but focuses on Parquet efficiency.
        """
        from loguru import logger

        logger.warning("PyArrow backend: CSV support experimental, using pandas fallback.")

        # Fallback to pandas for CSV since PyArrow CSV support is limited
        import pandas as pd

        return pd.read_csv(path, sep=sep)

    def load_jsonl(self, path: Path, **kwargs: Any) -> FrameLike:
        """Load JSONL file using PyArrow.

        Note: PyArrow doesn't have native JSONL support, so falls back.
        """
        from loguru import logger

        logger.warning("PyArrow backend: JSONL not supported, using pandas fallback.")

        import pandas as pd

        return pd.read_json(str(path), lines=True)

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
        """Scan CSV file in chunks using PyArrow."""
        from loguru import logger

        logger.warning("PyArrow backend: CSV streaming experimental, using pandas fallback.")

        # Fallback to pandas for CSV streaming
        import pandas as pd

        chunk_size_int = int(chunk_size)
        reader = pd.read_csv(str(path), sep=sep, chunksize=chunk_size_int)
        for chunk in reader:
            yield chunk

    def scan_jsonl(self, path: Path, chunk_size: int, **kwargs: Any) -> Iterator[FrameLike]:
        """Scan JSONL file in chunks using PyArrow."""
        from loguru import logger

        logger.warning("PyArrow backend: JSONL not supported, using pandas fallback.")

        # Fallback to pandas for JSONL streaming
        import pandas as pd

        chunk_size_int = int(chunk_size)

        try:
            import orjson
        except ImportError:
            raise RuntimeError("orjson is required for pandas JSONL scanner.")

        buf = []
        with open(path, "rb") as f:
            for line in f:
                if line.strip():
                    buf.append(orjson.loads(line))
                    if len(buf) >= chunk_size_int:
                        yield pd.DataFrame(buf)
                        buf.clear()
        if buf:
            yield pd.DataFrame(buf)

    def scan_parquet(self, path: Path, chunk_size: int, **kwargs: Any) -> Iterator[FrameLike]:
        """Scan Parquet file in chunks using PyArrow."""
        from loguru import logger

        logger.debug(f"Scanning Parquet with PyArrow: {path.name}")

        chunk_size_int = int(chunk_size)
        parquet_file = self._backend.ParquetFile(path)
        for batch in parquet_file.iter_batches(batch_size=chunk_size_int):
            yield batch.to_pandas()
