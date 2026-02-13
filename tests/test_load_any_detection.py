"""Tests for load_any format detection with ambiguous extensions."""

from __future__ import annotations

import pandas as pd
import pytest

from pyrator.data.loaders import load_any


def test_load_any_detects_csv_in_txt_file(tmp_path) -> None:
    """load_any should parse CSV data from a .txt file."""
    txt_file = tmp_path / "records.txt"
    txt_file.write_text("a,b\n1,2\n3,4\n")

    result = load_any(txt_file, prefer="pandas")

    assert isinstance(result, pd.DataFrame)
    assert list(result.columns) == ["a", "b"]
    assert len(result) == 2


def test_load_any_detects_ndjson_extension(tmp_path) -> None:
    """load_any should treat .ndjson as JSONL."""
    ndjson_file = tmp_path / "records.ndjson"
    ndjson_file.write_text('{"a": 1, "b": 2}\n{"a": 3, "b": 4}\n')

    result = load_any(ndjson_file, prefer="pandas")

    assert isinstance(result, pd.DataFrame)
    assert len(result) == 2
    assert set(result.columns) == {"a", "b"}


def test_load_any_detects_parquet_without_extension(tmp_path) -> None:
    """load_any should detect parquet from file signature when extension is missing."""
    pytest.importorskip("pyarrow")

    path = tmp_path / "records"
    frame = pd.DataFrame({"a": [1, 2], "b": [3, 4]})
    frame.to_parquet(path)

    result = load_any(path, prefer="pandas")

    assert isinstance(result, pd.DataFrame)
    assert len(result) == 2


def test_load_any_unrecognized_format_raises(tmp_path) -> None:
    """Unknown content with unknown extension should raise ValueError."""
    unknown = tmp_path / "payload.unknown"
    unknown.write_bytes(b"\x00\xff\x01\x02\x03")

    with pytest.raises(ValueError, match="Could not determine file format"):
        load_any(unknown, prefer="pandas")
