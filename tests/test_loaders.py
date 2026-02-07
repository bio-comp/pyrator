"""Tests for data loading utilities."""

import json
import numbers
import pandas as pd
import pytest
from pathlib import Path
from unittest.mock import patch, MagicMock, mock_open

from pyrator.data.loaders import (
    _validate_file,
    _escape_sql_path,
    load_any,
    load_csv,
    load_jsonl,
    load_parquet,
    scan_csv,
    scan_jsonl,
    scan_parquet,
)
from pyrator.data.backends import has_polars, has_pandas, has_duckdb


class TestValidateFile:
    """Test the _validate_file utility function."""

    def test_validate_existing_file(self, tmp_path):
        """Test validation of existing file."""
        test_file = tmp_path / "test.txt"
        test_file.write_text("content")

        # Should not raise any exception
        _validate_file(test_file)

    def test_validate_nonexistent_file(self):
        """Test validation of nonexistent file."""
        nonexistent = Path("/nonexistent/file.txt")

        with pytest.raises(FileNotFoundError, match="File not found"):
            _validate_file(nonexistent)

    def test_validate_directory(self, tmp_path):
        """Test validation of directory (should fail)."""
        with pytest.raises(ValueError, match="Path is not a file"):
            _validate_file(tmp_path)

    def test_validate_empty_file(self, tmp_path):
        """Test validation of empty file."""
        empty_file = tmp_path / "empty.txt"
        empty_file.touch()

        with pytest.raises(ValueError, match="File is empty"):
            _validate_file(empty_file)

    def test_validate_with_path_string(self, tmp_path):
        """Test validation with path string instead of Path object."""
        test_file = tmp_path / "test.txt"
        test_file.write_text("content")

        # Should work with string path
        _validate_file(str(test_file))


class TestEscapeSqlPath:
    """Test the _escape_sql_path utility function."""

    def test_escape_normal_path(self):
        """Test escaping normal path."""
        path = "/path/to/file.csv"
        result = _escape_sql_path(path)
        assert result == "/path/to/file.csv"

    def test_escape_path_with_apostrophes(self):
        """Test escaping path with apostrophes."""
        path = "/path/to/John's file.csv"
        result = _escape_sql_path(path)
        assert result == "/path/to/John''s file.csv"

    def test_escape_path_with_multiple_apostrophes(self):
        """Test escaping path with multiple apostrophes."""
        path = "/path/to/what's up/John's file.csv"
        result = _escape_sql_path(path)
        assert result == "/path/to/what''s up/John''s file.csv"

    def test_escape_empty_path(self):
        """Test escaping empty path."""
        path = ""
        result = _escape_sql_path(path)
        assert result == ""

    def test_escape_path_with_only_apostrophes(self):
        """Test escaping path with only apostrophes."""
        path = "'''"
        result = _escape_sql_path(path)
        assert result == "''''''"


class TestLoadAny:
    """Test the load_any function."""

    def test_load_any_csv(self, tmp_path):
        """Test load_any with CSV file."""
        csv_file = tmp_path / "test.csv"
        csv_file.write_text("a,b,c\n1,2,3\n4,5,6\n")

        result = load_any(csv_file)

        assert isinstance(result, (pd.DataFrame,))
        if has_polars():
            import polars as pl

            assert isinstance(result, (pd.DataFrame, pl.DataFrame))
        assert len(result) == 2

    def test_load_any_tsv(self, tmp_path):
        """Test load_any with TSV file."""
        tsv_file = tmp_path / "test.tsv"
        tsv_file.write_text("a\tb\tc\n1\t2\t3\n4\t5\t6\n")

        result = load_any(tsv_file)

        assert isinstance(result, (pd.DataFrame,))
        assert len(result) == 2

    def test_load_any_json(self, tmp_path):
        """Test load_any with JSON file."""
        json_file = tmp_path / "test.json"
        json_file.write_text('{"a": 1, "b": 2}\n{"a": 3, "b": 4}\n')

        result = load_any(json_file)

        assert isinstance(result, (pd.DataFrame,))
        assert len(result) == 2

    def test_load_any_jsonl(self, tmp_path):
        """Test load_any with JSONL file."""
        jsonl_file = tmp_path / "test.jsonl"
        jsonl_file.write_text('{"a": 1, "b": 2}\n{"a": 3, "b": 4}\n')

        result = load_any(jsonl_file)

        assert isinstance(result, (pd.DataFrame,))
        assert len(result) == 2

    def test_load_any_parquet(self, tmp_path):
        """Test load_any with Parquet file."""
        if not has_pandas():
            pytest.skip("Pandas not available for creating test parquet")

        parquet_file = tmp_path / "test.parquet"
        df = pd.DataFrame({"a": [1, 2], "b": [3, 4]})
        df.to_parquet(parquet_file)

        result = load_any(parquet_file)

        assert isinstance(result, (pd.DataFrame,))
        assert len(result) == 2

    def test_load_any_unsupported_extension(self, tmp_path):
        """Test load_any with unsupported file extension."""
        txt_file = tmp_path / "test.txt"
        txt_file.write_text("some content")

        with pytest.raises(ValueError, match="Unsupported file extension"):
            load_any(txt_file)

    def test_load_any_nonexistent_file(self):
        """Test load_any with nonexistent file."""
        with pytest.raises(FileNotFoundError):
            load_any("/nonexistent/file.csv")

    def test_load_any_prefer_parameter(self, tmp_path):
        """Test load_any with prefer parameter."""
        csv_file = tmp_path / "test.csv"
        csv_file.write_text("a,b\n1,2\n")

        # Test preferring polars
        result_polars = load_any(csv_file, prefer="polars")
        assert isinstance(result_polars, (pd.DataFrame,))

        # Test preferring pandas
        result_pandas = load_any(csv_file, prefer="pandas")
        assert isinstance(result_pandas, (pd.DataFrame,))


class TestLoadCSV:
    """Test the load_csv function."""

    def test_load_csv_basic(self, tmp_path):
        """Test basic CSV loading."""
        csv_file = tmp_path / "test.csv"
        csv_file.write_text("a,b,c\n1,2,3\n4,5,6\n")

        result = load_csv(csv_file)

        assert isinstance(result, (pd.DataFrame,))
        if has_polars():
            import polars as pl

            assert isinstance(result, (pd.DataFrame, pl.DataFrame))
        assert len(result) == 2

    def test_load_csv_with_separator(self, tmp_path):
        """Test CSV loading with custom separator."""
        csv_file = tmp_path / "test.tsv"
        csv_file.write_text("a\tb\tc\n1\t2\t3\n4\t5\t6\n")

        result = load_csv(csv_file, sep="\t")

        assert isinstance(result, (pd.DataFrame,))
        assert len(result) == 2

    def test_load_csv_prefer_polars(self, tmp_path):
        """Test CSV loading preferring polars."""
        if not has_polars():
            pytest.skip("Polars not available")

        csv_file = tmp_path / "test.csv"
        csv_file.write_text("a,b\n1,2\n")

        result = load_csv(csv_file, prefer="polars")

        import polars as pl

        assert isinstance(result, pl.DataFrame)

    def test_load_csv_prefer_pandas(self, tmp_path):
        """Test CSV loading preferring pandas."""
        if not has_pandas():
            pytest.skip("Pandas not available")

        csv_file = tmp_path / "test.csv"
        csv_file.write_text("a,b\n1,2\n")

        result = load_csv(csv_file, prefer="pandas")

        assert isinstance(result, pd.DataFrame)

    def test_load_csv_fallback_duckdb(self, tmp_path):
        """Test CSV loading falling back to DuckDB."""
        if not has_duckdb():
            pytest.skip("DuckDB not available")

        csv_file = tmp_path / "test.csv"
        csv_file.write_text("a,b\n1,2\n")

        # Mock polars and pandas as unavailable
        with (
            patch("pyrator.data.loaders.has_polars", return_value=False),
            patch("pyrator.data.loaders.has_pandas", return_value=False),
        ):
            result = load_csv(csv_file)

            # Should return a pandas DataFrame from DuckDB (since polars is mocked as unavailable)
            import pandas as pd

            assert isinstance(result, pd.DataFrame)

    def test_load_csv_no_backends_available(self, tmp_path):
        """Test CSV loading with no backends available."""
        csv_file = tmp_path / "test.csv"
        csv_file.write_text("a,b\n1,2\n")

        with (
            patch("pyrator.data.loaders.has_polars", return_value=False),
            patch("pyrator.data.loaders.has_pandas", return_value=False),
            patch("pyrator.data.loaders.has_duckdb", return_value=False),
        ):
            with pytest.raises(
                RuntimeError, match="Cannot read CSV: install polars, pandas, or duckdb"
            ):
                load_csv(csv_file)


class TestLoadJSONL:
    """Test the load_jsonl function."""

    def test_load_jsonl_basic(self, tmp_path):
        """Test basic JSONL loading."""
        jsonl_file = tmp_path / "test.jsonl"
        jsonl_file.write_text('{"a": 1, "b": 2}\n{"a": 3, "b": 4}\n')

        result = load_jsonl(jsonl_file)

        assert isinstance(result, (pd.DataFrame,))
        assert len(result) == 2

    def test_load_jsonl_prefer_polars(self, tmp_path):
        """Test JSONL loading preferring polars."""
        if not has_polars():
            pytest.skip("Polars not available")

        jsonl_file = tmp_path / "test.jsonl"
        jsonl_file.write_text('{"a": 1, "b": 2}\n')

        result = load_jsonl(jsonl_file, prefer="polars")

        import polars as pl

        assert isinstance(result, pl.DataFrame)

    def test_load_jsonl_prefer_pandas(self, tmp_path):
        """Test JSONL loading preferring pandas."""
        if not has_pandas():
            pytest.skip("Pandas not available")

        jsonl_file = tmp_path / "test.jsonl"
        jsonl_file.write_text('{"a": 1, "b": 2}\n')

        result = load_jsonl(jsonl_file, prefer="pandas")

        assert isinstance(result, pd.DataFrame)

    def test_load_jsonl_no_backends_available(self, tmp_path):
        """Test JSONL loading with no backends available."""
        jsonl_file = tmp_path / "test.jsonl"
        jsonl_file.write_text('{"a": 1}\n')

        with (
            patch("pyrator.data.loaders.has_polars", return_value=False),
            patch("pyrator.data.loaders.has_pandas", return_value=False),
        ):
            with pytest.raises(RuntimeError, match="Neither polars nor pandas is available"):
                load_jsonl(jsonl_file)


class TestLoadParquet:
    """Test the load_parquet function."""

    def test_load_parquet_basic(self, tmp_path):
        """Test basic Parquet loading."""
        if not has_pandas():
            pytest.skip("Pandas not available for creating test parquet")

        parquet_file = tmp_path / "test.parquet"
        df = pd.DataFrame({"a": [1, 2], "b": [3, 4]})
        df.to_parquet(parquet_file)

        result = load_parquet(parquet_file)

        assert isinstance(result, (pd.DataFrame,))
        assert len(result) == 2

    def test_load_parquet_prefer_polars(self, tmp_path):
        """Test Parquet loading preferring polars."""
        if not has_polars() or not has_pandas():
            pytest.skip("Polars and Pandas not available")

        parquet_file = tmp_path / "test.parquet"
        df = pd.DataFrame({"a": [1], "b": [2]})
        df.to_parquet(parquet_file)

        result = load_parquet(parquet_file, prefer="polars")

        import polars as pl

        assert isinstance(result, pl.DataFrame)

    def test_load_parquet_prefer_pandas(self, tmp_path):
        """Test Parquet loading preferring pandas."""
        if not has_pandas():
            pytest.skip("Pandas not available")

        parquet_file = tmp_path / "test.parquet"
        df = pd.DataFrame({"a": [1], "b": [2]})
        df.to_parquet(parquet_file)

        result = load_parquet(parquet_file, prefer="pandas")

        assert isinstance(result, pd.DataFrame)

    def test_load_parquet_fallback_duckdb(self, tmp_path):
        """Test Parquet loading falling back to DuckDB."""
        import pandas as pd

        if not has_pandas() or not has_duckdb():
            pytest.skip("Pandas and DuckDB not available")

        parquet_file = tmp_path / "test.parquet"
        df = pd.DataFrame({"a": [1], "b": [2]})
        df.to_parquet(parquet_file)

        # Mock polars as unavailable
        with patch("pyrator.data.loaders.has_polars", return_value=False):
            result = load_parquet(parquet_file)

            # Should return a pandas DataFrame from DuckDB (since polars is mocked as unavailable)
            import pandas as pd

            assert isinstance(result, pd.DataFrame)

    def test_load_parquet_no_backends_available(self, tmp_path):
        """Test Parquet loading with no backends available."""
        parquet_file = tmp_path / "test.parquet"
        parquet_file.write_text("fake parquet content")

        with (
            patch("pyrator.data.loaders.has_polars", return_value=False),
            patch("pyrator.data.loaders.has_pandas", return_value=False),
            patch("pyrator.data.loaders.has_duckdb", return_value=False),
        ):
            with pytest.raises(RuntimeError, match="Need polars, pandas, or duckdb"):
                load_parquet(parquet_file)


class TestScanCSV:
    """Test the scan_csv function."""

    def test_scan_csv_basic(self, tmp_path):
        """Test basic CSV scanning."""
        csv_file = tmp_path / "test.csv"
        csv_file.write_text("a,b,c\n1,2,3\n4,5,6\n7,8,9\n")

        chunks = list(scan_csv(csv_file, chunk_size=2))

        assert len(chunks) == 2  # 3 rows split into chunks of 2
        total_rows = sum(len(chunk) for chunk in chunks)
        assert total_rows == 3

    def test_scan_csv_with_separator(self, tmp_path):
        """Test CSV scanning with custom separator."""
        csv_file = tmp_path / "test.tsv"
        csv_file.write_text("a\tb\tc\n1\t2\t3\n4\t5\t6\n")

        chunks = list(scan_csv(csv_file, chunk_size=1, sep="\t"))

        assert len(chunks) == 2
        total_rows = sum(len(chunk) for chunk in chunks)
        assert total_rows == 2

    def test_scan_csv_prefer_polars(self, tmp_path):
        """Test CSV scanning preferring polars."""
        if not has_polars():
            pytest.skip("Polars not available")

        csv_file = tmp_path / "test.csv"
        csv_file.write_text("a,b\n1,2\n3,4\n")

        chunks = list(scan_csv(csv_file, chunk_size=1, prefer="polars"))

        import polars as pl

        for chunk in chunks:
            assert isinstance(chunk, pl.DataFrame)

    def test_scan_csv_prefer_pandas(self, tmp_path):
        """Test CSV scanning preferring pandas."""
        if not has_pandas():
            pytest.skip("Pandas not available")

        csv_file = tmp_path / "test.csv"
        csv_file.write_text("a,b\n1,2\n3,4\n")

        chunks = list(scan_csv(csv_file, chunk_size=1, prefer="pandas"))

        for chunk in chunks:
            assert isinstance(chunk, pd.DataFrame)

    def test_scan_csv_chunk_size_validation(self, tmp_path):
        """Test CSV scanning with chunk size validation."""
        csv_file = tmp_path / "test.csv"
        csv_file.write_text("a,b\n1,2\n")

        # Test with integer chunk size
        chunks = list(scan_csv(csv_file, chunk_size=1))
        assert len(chunks) == 1

        # Test with numeric chunk size (using int which is a subclass of Integral)
        chunks = list(scan_csv(csv_file, chunk_size=int(1)))
        assert len(chunks) == 1

    def test_scan_csv_no_backends_available(self, tmp_path):
        """Test CSV scanning with no backends available."""
        csv_file = tmp_path / "test.csv"
        csv_file.write_text("a,b\n1,2\n")

        with (
            patch("pyrator.data.loaders.has_polars", return_value=False),
            patch("pyrator.data.loaders.has_pandas", return_value=False),
        ):
            with pytest.raises(RuntimeError, match="Neither polars nor pandas is available"):
                next(scan_csv(csv_file))


class TestScanJSONL:
    """Test the scan_jsonl function."""

    def test_scan_jsonl_basic(self, tmp_path):
        """Test basic JSONL scanning."""
        jsonl_file = tmp_path / "test.jsonl"
        jsonl_file.write_text('{"a": 1, "b": 2}\n{"a": 3, "b": 4}\n{"a": 5, "b": 6}\n')

        chunks = list(scan_jsonl(jsonl_file, chunk_size=2))

        assert len(chunks) == 2  # 3 rows split into chunks of 2
        total_rows = sum(len(chunk) for chunk in chunks)
        assert total_rows == 3

    def test_scan_jsonl_prefer_polars(self, tmp_path):
        """Test JSONL scanning preferring polars."""
        if not has_polars():
            pytest.skip("Polars not available")

        jsonl_file = tmp_path / "test.jsonl"
        jsonl_file.write_text('{"a": 1}\n{"a": 2}\n')

        chunks = list(scan_jsonl(jsonl_file, chunk_size=1, prefer="polars"))

        import polars as pl

        for chunk in chunks:
            assert isinstance(chunk, pl.DataFrame)

    def test_scan_jsonl_prefer_pandas(self, tmp_path):
        """Test JSONL scanning preferring pandas."""
        if not has_pandas():
            pytest.skip("Pandas not available")

        try:
            import orjson
        except ImportError:
            pytest.skip("orjson not available for pandas JSONL scanning")

        jsonl_file = tmp_path / "test.jsonl"
        jsonl_file.write_text('{"a": 1}\n{"a": 2}\n')

        chunks = list(scan_jsonl(jsonl_file, chunk_size=1, prefer="pandas"))

        for chunk in chunks:
            assert isinstance(chunk, pd.DataFrame)

    def test_scan_jsonl_pandas_without_orjson(self, tmp_path):
        """Test JSONL scanning with pandas but without orjson."""
        if not has_pandas():
            pytest.skip("Pandas not available")

        jsonl_file = tmp_path / "test.jsonl"
        jsonl_file.write_text('{"a": 1}\n')

        with (
            patch("pyrator.data.loaders.has_polars", return_value=False),
            patch("pyrator.data.loaders.has_pandas", return_value=True),
            patch.dict("sys.modules", {"orjson": None}),
        ):
            with pytest.raises(RuntimeError, match="orjson is required"):
                next(scan_jsonl(jsonl_file, prefer="pandas"))

    def test_scan_jsonl_no_backends_available(self, tmp_path):
        """Test JSONL scanning with no backends available."""
        jsonl_file = tmp_path / "test.jsonl"
        jsonl_file.write_text('{"a": 1}\n')

        with (
            patch("pyrator.data.loaders.has_polars", return_value=False),
            patch("pyrator.data.loaders.has_pandas", return_value=False),
        ):
            with pytest.raises(RuntimeError, match="Cannot stream JSONL"):
                next(scan_jsonl(jsonl_file))


class TestScanParquet:
    """Test the scan_parquet function."""

    def test_scan_parquet_basic(self, tmp_path):
        """Test basic Parquet scanning."""
        if not has_pandas():
            pytest.skip("Pandas not available for creating test parquet")

        parquet_file = tmp_path / "test.parquet"
        df = pd.DataFrame({"a": [1, 2, 3], "b": [4, 5, 6]})
        df.to_parquet(parquet_file)

        chunks = list(scan_parquet(parquet_file, chunk_size=2))

        assert len(chunks) >= 1  # At least one chunk
        total_rows = sum(len(chunk) for chunk in chunks)
        assert total_rows == 3

    def test_scan_parquet_prefer_polars(self, tmp_path):
        """Test Parquet scanning preferring polars."""
        if not has_polars() or not has_pandas():
            pytest.skip("Polars and Pandas not available")

        parquet_file = tmp_path / "test.parquet"
        df = pd.DataFrame({"a": [1, 2], "b": [3, 4]})
        df.to_parquet(parquet_file)

        chunks = list(scan_parquet(parquet_file, chunk_size=1, prefer="polars"))

        import polars as pl

        for chunk in chunks:
            assert isinstance(chunk, pl.DataFrame)

    def test_scan_parquet_prefer_pyarrow(self, tmp_path):
        """Test Parquet scanning preferring pyarrow."""
        try:
            import pyarrow.parquet as pq
        except ImportError:
            pytest.skip("PyArrow not available")

        if not has_pandas():
            pytest.skip("Pandas not available for creating test parquet")

        parquet_file = tmp_path / "test.parquet"
        df = pd.DataFrame({"a": [1, 2], "b": [3, 4]})
        df.to_parquet(parquet_file)

        chunks = list(scan_parquet(parquet_file, chunk_size=1, prefer="pyarrow"))

        for chunk in chunks:
            assert isinstance(chunk, pd.DataFrame)

    def test_scan_parquet_no_backends_available(self, tmp_path):
        """Test Parquet scanning with no backends available."""
        parquet_file = tmp_path / "test.parquet"
        parquet_file.write_text("fake parquet content")

        with (
            patch("pyrator.data.loaders.has_polars", return_value=False),
            patch.dict("sys.modules", {"pyarrow": None, "pyarrow.parquet": None}),
        ):
            with pytest.raises(RuntimeError, match="Cannot stream Parquet"):
                next(scan_parquet(parquet_file))


class TestIntegration:
    """Test integration between different loading functions."""

    def test_load_and_scan_consistency(self, tmp_path):
        """Test that load and scan functions give consistent results."""
        csv_file = tmp_path / "test.csv"
        csv_file.write_text("a,b,c\n1,2,3\n4,5,6\n7,8,9\n")

        # Load all at once
        full_data = load_csv(csv_file)

        # Scan in chunks and combine
        chunks = list(scan_csv(csv_file, chunk_size=2))
        combined_data = pd.concat(chunks, ignore_index=True) if has_pandas() else None

        if combined_data is not None:
            # Should have same number of rows
            assert len(full_data) == len(combined_data)

    def test_different_format_consistency(self, tmp_path):
        """Test consistency across different file formats."""
        # Create same data in different formats
        data = [{"a": 1, "b": 2}, {"a": 3, "b": 4}]

        # CSV
        csv_file = tmp_path / "test.csv"
        csv_file.write_text("a,b\n1,2\n3,4\n")

        # JSONL
        jsonl_file = tmp_path / "test.jsonl"
        jsonl_file.write_text("\n".join(json.dumps(record) for record in data))

        # Load both
        csv_data = load_csv(csv_file)
        jsonl_data = load_jsonl(jsonl_file)

        # Should have same number of rows
        assert len(csv_data) == len(jsonl_data) == 2

    def test_error_handling_consistency(self, tmp_path):
        """Test consistent error handling across loaders."""
        # Test with nonexistent file
        nonexistent = "/nonexistent/file.csv"

        with pytest.raises(FileNotFoundError):
            load_csv(nonexistent)

        with pytest.raises(FileNotFoundError):
            load_jsonl(nonexistent)

        with pytest.raises(FileNotFoundError):
            load_parquet(nonexistent)

        with pytest.raises(FileNotFoundError):
            next(scan_csv(nonexistent))

        with pytest.raises(FileNotFoundError):
            next(scan_jsonl(nonexistent))

        with pytest.raises(FileNotFoundError):
            next(scan_parquet(nonexistent))
