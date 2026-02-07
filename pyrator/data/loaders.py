# pyrator/data/loaders.py
from __future__ import annotations

import numbers
from collections.abc import Iterator
from pathlib import Path
from typing import TYPE_CHECKING, Any, Literal

from loguru import logger

from pyrator.data.backends import has_duckdb, has_pandas, has_polars
from pyrator.types import FrameLike

# This block allows Mypy/Pylance to see the types for static analysis
# without affecting runtime performance or causing circular imports.
if TYPE_CHECKING:
    import pandas as pd
    import polars as pl


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


def _escape_sql_path(path: str) -> str:
    """Escapes a path for inclusion in a DuckDB SQL string."""
    return path.replace("'", "''")


def load_any(path: str | Path, *, prefer: Literal["polars", "pandas"] = "polars") -> FrameLike:
    """
    Loads a file into a DataFrame, auto-detecting the format from the extension.
    """
    p = Path(path)
    _validate_file(p)

    ext = p.suffix.lower()
    logger.debug(f"Loading {ext} file: {p.name}")

    if ext in {".jsonl", ".json"}:
        return load_jsonl(str(p), prefer=prefer)
    if ext in {".parquet", ".pq"}:
        return load_parquet(str(p), prefer=prefer)
    if ext in {".csv", ".tsv"}:
        return load_csv(str(p), prefer=prefer, sep="\t" if ext == ".tsv" else ",")

    logger.error(f"Unsupported file extension: {ext}")
    raise ValueError(f"Unsupported file extension: {ext}")


def load_csv(
    path: str | Path, *, prefer: Literal["polars", "pandas"] = "polars", sep: str = ","
) -> FrameLike:
    """Loads a CSV or TSV file into a DataFrame."""
    p = str(path)
    if has_polars() and prefer == "polars":
        import polars as pl

        logger.debug(f"Loading CSV with Polars: {Path(p).name}")
        return pl.read_csv(p, separator=sep)
    if has_pandas():
        import pandas as pd

        logger.debug(f"Loading CSV with Pandas: {Path(p).name}")
        return pd.read_csv(p, sep=sep)
    if has_duckdb():
        import duckdb

        logger.debug(f"Loading CSV with DuckDB: {Path(p).name}")
        escaped_path = _escape_sql_path(p)
        return duckdb.sql(f"SELECT * FROM read_csv_auto('{escaped_path}')").df()

    logger.error("No CSV backend available")
    raise RuntimeError("Cannot read CSV: install polars, pandas, or duckdb")


def load_jsonl(path: str | Path, *, prefer: Literal["polars", "pandas"] = "polars") -> FrameLike:
    """Loads a JSONL (newline-delimited JSON) file into a DataFrame."""
    p = Path(path)
    _validate_file(p)
    p_str = str(p)

    if has_polars() and prefer == "polars":
        import polars as pl

        return pl.read_ndjson(p_str)
    if has_pandas():
        import pandas as pd

        return pd.read_json(p_str, lines=True)
    raise RuntimeError("Neither polars nor pandas is available to read JSONL.")


def load_parquet(path: str | Path, *, prefer: Literal["polars", "pandas"] = "polars") -> FrameLike:
    """Loads a Parquet file into a DataFrame."""
    p = str(path)
    if has_polars() and prefer == "polars":
        import polars as pl

        return pl.read_parquet(p)
    if has_pandas():
        import pandas as pd

        return pd.read_parquet(p)
    if has_duckdb():
        import duckdb

        escaped_path = _escape_sql_path(p)
        return duckdb.query(f"SELECT * FROM read_parquet('{escaped_path}')").pl()
    raise RuntimeError("Need polars, pandas, or duckdb to read Parquet.")


def scan_csv(
    path: str | Path,
    *,
    chunk_size: numbers.Integral = 200_000,  # type: ignore[assignment]
    sep: str = ",",
    prefer: Literal["polars", "pandas"] = "polars",
) -> Iterator[FrameLike]:
    """Scans a CSV file in chunks, yielding DataFrames."""
    p = Path(path)
    _validate_file(p)
    chunk_size_int = int(chunk_size)

    if has_polars() and prefer == "polars":
        import polars as pl

        logger.debug(f"Scanning CSV with Polars: {p.name}")
        lf = pl.scan_csv(p, separator=sep)
        offset = 0
        while True:
            batch: pl.DataFrame = lf.slice(offset, chunk_size_int).collect(engine="streaming")
            if not len(batch):
                break
            yield batch
            offset += chunk_size_int
        return

    if has_pandas():
        import pandas as pd

        logger.debug(f"Scanning CSV with Pandas: {p.name}")
        reader: Iterator["pd.DataFrame"] = pd.read_csv(str(p), sep=sep, chunksize=chunk_size_int)
        for chunk in reader:
            yield chunk
        return

    raise RuntimeError("Neither polars nor pandas is available to stream CSV.")


def scan_jsonl(
    path: str | Path,
    *,
    chunk_size: numbers.Integral = 50_000,  # type: ignore[assignment]
    prefer: Literal["polars", "pandas"] = "polars",
) -> Iterator[FrameLike]:
    """Scans a JSONL file in chunks, yielding DataFrames."""
    p = Path(path)
    _validate_file(p)
    chunk_size_int = int(chunk_size)

    if has_polars() and prefer == "polars":
        import polars as pl

        logger.debug(f"Scanning JSONL with Polars: {p.name}")
        lf = pl.scan_ndjson(p)
        offset = 0
        while True:
            batch: pl.DataFrame = lf.slice(offset, chunk_size_int).collect(engine="streaming")
            if not len(batch):
                break
            yield batch
            offset += chunk_size_int
        return

    if has_pandas():
        try:
            import orjson
        except ImportError:
            raise RuntimeError("orjson is required for the pandas JSONL scanner.")
        import pandas as pd

        logger.debug(f"Scanning JSONL with Pandas/orjson: {p.name}")
        buf: list[dict[str, Any]] = []
        with open(p, "rb") as f:
            for line in f:
                if line.strip():
                    buf.append(orjson.loads(line))
                    if len(buf) >= chunk_size_int:
                        yield pd.DataFrame(buf)
                        buf.clear()
        if buf:
            yield pd.DataFrame(buf)
        return

    raise RuntimeError("Cannot stream JSONL: install polars or pandas with orjson.")


def scan_parquet(
    path: str | Path,
    *,
    chunk_size: numbers.Integral = 100_000,  # type: ignore[assignment]
    prefer: Literal["polars", "pyarrow"] = "polars",
) -> Iterator[FrameLike]:
    """Scans a Parquet file in chunks, yielding DataFrames."""
    p = Path(path)
    _validate_file(p)
    chunk_size_int = int(chunk_size)

    if has_polars() and prefer == "polars":
        import polars as pl

        logger.debug(f"Scanning Parquet with Polars: {p.name}")
        lf = pl.scan_parquet(str(p))
        offset = 0
        while True:
            batch: "pl.DataFrame" = lf.slice(offset, chunk_size_int).collect(engine="streaming")
            if not len(batch):
                break
            yield batch
            offset += chunk_size_int
        return

    try:
        import pyarrow.parquet as pq

        logger.debug(f"Scanning Parquet with PyArrow: {p.name}")
        parquet_file = pq.ParquetFile(p)
        for batch in parquet_file.iter_batches(batch_size=chunk_size_int):
            yield batch.to_pandas()
        return
    except ImportError:
        logger.debug("PyArrow not available for streaming.")

    raise RuntimeError("Cannot stream Parquet: install polars or pyarrow.")
