# pyrator/data/loaders.py
from __future__ import annotations

import numbers
from collections.abc import Iterator
from pathlib import Path
from typing import TYPE_CHECKING, Literal

from loguru import logger

from pyrator.data.registry import BackendRegistry

# This block allows Mypy/Pylance to see types for static analysis
# without affecting runtime performance or causing circular imports.
if TYPE_CHECKING:
    from pyrator.types import FrameLike


def _validate_file(path: str | Path) -> None:
    """Validate file exists, is a file, and is not empty."""
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(f"File not found: {p}")
    if not p.is_file():
        raise ValueError(f"Path is not a file: {p}")
    if not p.stat().st_size:
        raise ValueError(f"File is empty: {p}")
    logger.debug(f"Validated file: {p} ({p.stat().st_size:,} bytes)")


def load_any(
    path: str | Path, *, prefer: Literal["auto", "polars", "pandas", "duckdb", "pyarrow"] = "auto"
) -> FrameLike:
    """
    Loads a file into a DataFrame, auto-detecting the format from extension.

    Args:
        path: Path to the file to load
        prefer: Backend preference - "auto" for automatic selection,
               or specific backend name

    Returns:
        DataFrame containing the loaded data
    """
    p = Path(path)
    _validate_file(p)

    ext = p.suffix.lower()
    logger.debug(f"Loading {ext} file: {p.name}")

    # Select backend
    if prefer == "auto":
        backend = BackendRegistry.auto_select()
    else:
        backend = BackendRegistry.get_backend(prefer)

    # Route to appropriate loader
    if ext in {".jsonl", ".json"}:
        return backend.load_jsonl(p, **{})
    if ext in {".parquet", ".pq"}:
        return backend.load_parquet(p, **{})
    if ext in {".csv", ".tsv"}:
        return backend.load_csv(p, sep="\t" if ext == ".tsv" else ",", **{})

    logger.error(f"Unsupported file extension: {ext}")
    raise ValueError(f"Unsupported file extension: {ext}")


def load_csv(
    path: str | Path,
    *,
    prefer: Literal["auto", "polars", "pandas", "duckdb"] = "auto",
    sep: str = ",",
) -> FrameLike:
    """
    Loads a CSV or TSV file into a DataFrame.

    Args:
        path: Path to the CSV file
        prefer: Backend preference - "auto" for automatic selection,
               or specific backend name that supports CSV
        sep: Field separator (default: comma)

    Returns:
        DataFrame containing the loaded data
    """
    p = Path(path)
    _validate_file(p)

    # Select backend
    if prefer == "auto":
        backend = BackendRegistry.auto_select()
    else:
        backend = BackendRegistry.get_backend(prefer)

    # Load with selected backend
    return backend.load_csv(p, sep=sep, **{})


def load_jsonl(
    path: str | Path, *, prefer: Literal["auto", "polars", "pandas"] = "auto"
) -> FrameLike:
    """
    Loads a JSONL (newline-delimited JSON) file into a DataFrame.

    Args:
        path: Path to the JSONL file
        prefer: Backend preference - "auto" for automatic selection,
               or specific backend name that supports JSONL

    Returns:
        DataFrame containing the loaded data
    """
    p = Path(path)
    _validate_file(p)

    # Select backend
    if prefer == "auto":
        backend = BackendRegistry.auto_select()
    else:
        backend = BackendRegistry.get_backend(prefer)

    # Load with selected backend
    return backend.load_jsonl(p, **{})


def load_parquet(
    path: str | Path, *, prefer: Literal["auto", "polars", "pandas", "duckdb", "pyarrow"] = "auto"
) -> FrameLike:
    """
    Loads a Parquet file into a DataFrame.

    Args:
        path: Path to the Parquet file
        prefer: Backend preference - "auto" for automatic selection,
               or specific backend name that supports Parquet

    Returns:
        DataFrame containing the loaded data
    """
    p = Path(path)
    _validate_file(p)

    # Select backend
    if prefer == "auto":
        backend = BackendRegistry.auto_select()
    else:
        backend = BackendRegistry.get_backend(prefer)

    # Load with selected backend
    return backend.load_parquet(p, **{})


def scan_csv(
    path: str | Path,
    *,
    chunk_size: numbers.Integral = 200_000,  # type: ignore[assignment]
    sep: str = ",",
    prefer: Literal["auto", "polars", "pandas"] = "auto",
) -> Iterator[FrameLike]:
    """
    Scans a CSV file in chunks, yielding DataFrames.

    Args:
        path: Path to the CSV file
        chunk_size: Number of rows per chunk
        prefer: Backend preference - "auto" for automatic selection,
               or specific backend name that supports CSV streaming
        sep: Field separator (default: comma)

    Yields:
        DataFrames containing chunks of the data
    """
    p = Path(path)
    _validate_file(p)
    chunk_size_int = int(chunk_size)

    # Select backend
    if prefer == "auto":
        backend = BackendRegistry.auto_select()
    else:
        backend = BackendRegistry.get_backend(prefer)

    # Scan with selected backend
    yield from backend.scan_csv(p, chunk_size=chunk_size_int, sep=sep, **{})


def scan_jsonl(
    path: str | Path,
    *,
    chunk_size: numbers.Integral = 50_000,  # type: ignore[assignment]
    prefer: Literal["auto", "polars", "pandas"] = "auto",
) -> Iterator[FrameLike]:
    """
    Scans a JSONL file in chunks, yielding DataFrames.

    Args:
        path: Path to the JSONL file
        chunk_size: Number of records per chunk
        prefer: Backend preference - "auto" for automatic selection,
               or specific backend name that supports JSONL streaming

    Yields:
        DataFrames containing chunks of the data
    """
    p = Path(path)
    _validate_file(p)
    chunk_size_int = int(chunk_size)

    # Select backend
    if prefer == "auto":
        backend = BackendRegistry.auto_select()
    else:
        backend = BackendRegistry.get_backend(prefer)

    # Scan with selected backend
    yield from backend.scan_jsonl(p, chunk_size=chunk_size_int, **{})


def scan_parquet(
    path: str | Path,
    *,
    chunk_size: numbers.Integral = 100_000,  # type: ignore[assignment]
    prefer: Literal["auto", "polars", "pyarrow"] = "auto",
) -> Iterator[FrameLike]:
    """
    Scans a Parquet file in chunks, yielding DataFrames.

    Args:
        path: Path to the Parquet file
        chunk_size: Number of rows per chunk
        prefer: Backend preference - "auto" for automatic selection,
               or specific backend name that supports Parquet streaming

    Yields:
        DataFrames containing chunks of the data
    """
    p = Path(path)
    _validate_file(p)
    chunk_size_int = int(chunk_size)

    # Select backend
    if prefer == "auto":
        backend = BackendRegistry.auto_select()
    else:
        backend = BackendRegistry.get_backend(prefer)

    # Scan with selected backend
    yield from backend.scan_parquet(p, chunk_size=chunk_size_int, **{})
