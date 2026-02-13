# pyrator/data/loaders.py
from __future__ import annotations

import csv
import json
import numbers
from collections.abc import Iterator
from pathlib import Path
from typing import TYPE_CHECKING, Literal

from loguru import logger

from pyrator.data.backends import has_duckdb, has_pandas, has_polars
from pyrator.data.registry import BackendRegistry
from pyrator.data.spi import DataBackend

# This block allows Mypy/Pylance to see types for static analysis
# without affecting runtime performance or causing circular imports.
if TYPE_CHECKING:
    from pyrator.types import FrameLike


def _select_backend(prefer: str, required_capabilities: set[str] | None = None) -> DataBackend:
    """Select backend, preferring capability-aware auto resolution."""
    if prefer == "auto":
        if required_capabilities:
            return BackendRegistry.get_best_backend_with_capabilities(required_capabilities)
        return BackendRegistry.auto_select()
    return BackendRegistry.get_backend(prefer)


def _has_pyarrow_parquet() -> bool:
    """Return True when pyarrow parquet support is importable."""
    try:
        __import__("pyarrow.parquet")
        return True
    except Exception:
        return False


def _validate_requested_capabilities(*, prefer: str, required: set[str]) -> None:
    """Fail fast with clear messages when required backends are unavailable."""
    if prefer != "auto":
        return

    if required == {"jsonl"} and not (has_polars() or has_pandas()):
        raise RuntimeError("Neither polars nor pandas is available for JSONL loading.")

    if required == {"csv", "streaming"} and not (has_polars() or has_pandas()):
        raise RuntimeError("Neither polars nor pandas is available for CSV streaming.")

    if required == {"jsonl", "streaming"} and not (has_polars() or has_pandas()):
        raise RuntimeError("Cannot stream JSONL: neither polars nor pandas is available.")

    if required == {"parquet"} and not (has_polars() or has_pandas() or has_duckdb()):
        raise RuntimeError("Need polars, pandas, or duckdb for parquet loading.")

    if required == {"parquet", "streaming"} and not (has_polars() or _has_pyarrow_parquet()):
        raise RuntimeError("Cannot stream Parquet without polars or pyarrow support.")


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


def _format_from_extension(ext: str) -> tuple[str, str | None] | None:
    """Return (format, delimiter) from file extension when known."""
    if ext == ".csv":
        return ("csv", ",")
    if ext == ".tsv":
        return ("csv", "\t")
    if ext in {".json", ".jsonl", ".ndjson"}:
        return ("jsonl", None)
    if ext in {".parquet", ".pq"}:
        return ("parquet", None)
    return None


def _looks_like_parquet(path: Path, sample_bytes: bytes) -> bool:
    """Detect parquet using magic bytes (header + footer)."""
    if len(sample_bytes) >= 4 and sample_bytes[:4] == b"PAR1":
        return True

    try:
        with path.open("rb") as handle:
            handle.seek(-4, 2)
            return handle.read(4) == b"PAR1"
    except OSError:
        return False


def _looks_like_jsonl(sample_text: str) -> bool:
    """Detect newline-delimited JSON by probing the first non-empty lines."""
    lines = [line.strip() for line in sample_text.splitlines() if line.strip()]
    if not lines:
        return False

    for line in lines[:5]:
        try:
            json.loads(line)
        except json.JSONDecodeError:
            return False
    return True


def _sniff_csv_delimiter(sample_text: str) -> str | None:
    """Infer a tabular delimiter from sample text."""
    if not sample_text.strip():
        return None
    try:
        dialect = csv.Sniffer().sniff(sample_text, delimiters=",\t;|")
    except csv.Error:
        return None
    return dialect.delimiter


def _detect_input_format(path: Path) -> tuple[str, str | None]:
    """Detect data format from extension and file content."""
    ext_info = _format_from_extension(path.suffix.lower())
    if ext_info and path.suffix.lower() not in {"", ".txt"}:
        return ext_info

    with path.open("rb") as handle:
        sample_bytes = handle.read(16_384)
    if _looks_like_parquet(path, sample_bytes):
        return ("parquet", None)

    sample_text = sample_bytes.decode("utf-8", errors="ignore")
    if _looks_like_jsonl(sample_text):
        return ("jsonl", None)

    delimiter = _sniff_csv_delimiter(sample_text)
    if delimiter is not None:
        return ("csv", delimiter)

    if ext_info:
        return ext_info

    ext = path.suffix.lower() or "<none>"
    raise ValueError(
        f"Unsupported file extension: {ext}. Could not determine file format from content."
    )


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

    detected_format, detected_sep = _detect_input_format(p)
    logger.debug(f"Loading detected format '{detected_format}' for file: {p.name}")

    _validate_requested_capabilities(prefer=prefer, required={detected_format})
    backend = _select_backend(prefer, {detected_format})

    if detected_format == "jsonl":
        return backend.load_jsonl(p, **{})
    if detected_format == "parquet":
        return backend.load_parquet(p, **{})
    if detected_format == "csv":
        sep = detected_sep if detected_sep is not None else ","
        return backend.load_csv(p, sep=sep, **{})

    raise ValueError(f"Unsupported detected format: {detected_format}")


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

    _validate_requested_capabilities(prefer=prefer, required={"csv"})
    # Select backend
    backend = _select_backend(prefer, {"csv"})

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

    _validate_requested_capabilities(prefer=prefer, required={"jsonl"})
    # Select backend
    backend = _select_backend(prefer, {"jsonl"})

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

    _validate_requested_capabilities(prefer=prefer, required={"parquet"})
    # Select backend
    backend = _select_backend(prefer, {"parquet"})

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

    _validate_requested_capabilities(prefer=prefer, required={"csv", "streaming"})
    # Select backend
    backend = _select_backend(prefer, {"csv", "streaming"})

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

    _validate_requested_capabilities(prefer=prefer, required={"jsonl", "streaming"})
    # Select backend
    backend = _select_backend(prefer, {"jsonl", "streaming"})

    # Scan with selected backend
    yield from backend.scan_jsonl(p, chunk_size=chunk_size_int, **{})


def scan_parquet(
    path: str | Path,
    *,
    chunk_size: numbers.Integral = 100_000,  # type: ignore[assignment]
    prefer: Literal["auto", "polars", "pandas", "pyarrow"] = "auto",
) -> Iterator[FrameLike]:
    """
    Scans a Parquet file in chunks, yielding DataFrames.

    Args:
        path: Path to the Parquet file
        chunk_size: Number of rows per chunk
        prefer: Backend preference - "auto" for automatic selection,
               or specific backend name that supports Parquet streaming.
               Pandas streaming requires pyarrow.

    Yields:
        DataFrames containing chunks of the data
    """
    p = Path(path)
    _validate_file(p)
    chunk_size_int = int(chunk_size)

    _validate_requested_capabilities(prefer=prefer, required={"parquet", "streaming"})
    # Select backend
    backend = _select_backend(prefer, {"parquet", "streaming"})

    # Scan with selected backend
    yield from backend.scan_parquet(p, chunk_size=chunk_size_int, **{})
