"""Service Provider Interface for data loading backends.

This module defines the protocol that all data backends must implement,
providing a contract for extensible data loading without creating
circular import dependencies.
"""

from __future__ import annotations

from collections.abc import Iterator
from pathlib import Path
from typing import Any

from pyrator.types import FrameLike


class DataBackend:
    """Protocol that all data backends must implement."""

    @property
    def name(self) -> str:
        """Return the name of this backend."""
        ...

    def is_available(self) -> bool:
        """Check if backend dependencies are available.

        Returns:
            True if backend can be used, False otherwise.
        """
        ...

    def capabilities(self) -> set[str]:
        """Return the capabilities of this backend.

        Returns:
            Set of capability strings like 'csv', 'jsonl', 'parquet', 'streaming', etc.
            Methods without a matching capability should raise NotImplementedError.
        """
        ...

    # Loading operations
    def load_csv(self, path: Path, sep: str = ",", **kwargs: Any) -> FrameLike:
        """Load CSV file into DataFrame."""
        ...

    def load_jsonl(self, path: Path, **kwargs: Any) -> FrameLike:
        """Load JSONL file into DataFrame."""
        ...

    def load_parquet(self, path: Path, **kwargs: Any) -> FrameLike:
        """Load Parquet file into DataFrame."""
        ...

    # Scanning operations
    def scan_csv(
        self, path: Path, chunk_size: int, sep: str = ",", **kwargs: Any
    ) -> Iterator[FrameLike]:
        """Scan CSV file in chunks."""
        ...

    def scan_jsonl(self, path: Path, chunk_size: int, **kwargs: Any) -> Iterator[FrameLike]:
        """Scan JSONL file in chunks."""
        ...

    def scan_parquet(self, path: Path, chunk_size: int, **kwargs: Any) -> Iterator[FrameLike]:
        """Scan Parquet file in chunks."""
        ...
