"""Tests for backend availability and dispatcher functionality."""

from unittest.mock import MagicMock, patch

import numpy as np
import pytest

from pyrator.data.backends import (
    _try_import,
    get_xp,
    has_cupy,
    has_duckdb,
    has_ijson,
    has_pandas,
    has_polars,
    to_pandas,
    to_polars,
)


class TestTryImport:
    """Test the _try_import utility function."""

    def test_successful_import(self):
        """Test successful import of available module."""
        result = _try_import("sys")
        assert result is not None
        assert hasattr(result, "version_info")

    def test_failed_import(self):
        """Test failed import of unavailable module."""
        result = _try_import("definitely_nonexistent_module_xyz123")
        assert result is None

    def test_import_with_exception(self):
        """Test import that raises an exception."""
        with patch("builtins.__import__", side_effect=ImportError("Test error")):
            result = _try_import("some_module")
            assert result is None

    def test_import_with_runtime_error(self):
        """Test import that raises a runtime error."""
        with patch("builtins.__import__", side_effect=RuntimeError("Runtime error")):
            result = _try_import("some_module")
            assert result is None


class TestBackendAvailability:
    """Test backend availability check functions."""

    def test_has_pandas(self):
        """Test pandas availability check."""
        # This should be True in the test environment
        result = has_pandas()
        assert isinstance(result, bool)

    def test_has_polars(self):
        """Test polars availability check."""
        result = has_polars()
        assert isinstance(result, bool)

    def test_has_duckdb(self):
        """Test duckdb availability check."""
        result = has_duckdb()
        assert isinstance(result, bool)

    def test_has_cupy(self):
        """Test cupy availability check."""
        result = has_cupy()
        assert isinstance(result, bool)

    def test_has_ijson(self):
        """Test ijson availability check."""
        result = has_ijson()
        assert isinstance(result, bool)

    def test_availability_return_types(self):
        """Test that all availability functions return booleans."""
        functions = [has_pandas, has_polars, has_duckdb, has_cupy, has_ijson]

        for func in functions:
            result = func()
            assert isinstance(result, bool), f"{func.__name__} should return bool"


class TestGetXp:
    """Test the get_xp array module dispatcher."""

    def test_get_xp_default(self):
        """Test get_xp with default parameters (should return NumPy)."""
        xp = get_xp()
        assert xp is np
        assert hasattr(xp, "asarray")
        assert hasattr(xp, "empty")
        assert hasattr(xp, "zeros")
        assert hasattr(xp, "ones")

    def test_get_xp_cpu(self):
        """Test get_xp with device='cpu'."""
        xp = get_xp(device="cpu")
        assert xp is np

    def test_get_xp_gpu_available(self):
        """Test get_xp with device='gpu' when CuPy is available."""
        # Mock CuPy as available
        mock_cupy = MagicMock()
        mock_cupy.asarray = MagicMock(return_value=np.array([1, 2, 3]))
        mock_cupy.empty = MagicMock(return_value=np.empty((3, 2)))
        mock_cupy.zeros = MagicMock(return_value=np.zeros((2, 3)))
        mock_cupy.ones = MagicMock(return_value=np.ones((2, 2)))
        mock_cupy.int32 = "int32"
        mock_cupy.float32 = "float32"
        mock_cupy.float64 = "float64"

        with patch("pyrator.data.backends._cupy", mock_cupy):
            xp = get_xp(device="gpu")
            assert xp is mock_cupy

    def test_get_xp_gpu_unavailable(self):
        """Test get_xp with device='gpu' when CuPy is not available."""
        with patch("pyrator.data.backends._cupy", None):
            with pytest.raises(
                RuntimeError, match="device='gpu' requested but CuPy is not installed"
            ):
                get_xp(device="gpu")

    def test_get_xp_gpu_import_error(self):
        """Test get_xp with device='gpu' when CuPy import fails."""
        with patch("pyrator.data.backends._cupy", None):
            with pytest.raises(
                RuntimeError, match="device='gpu' requested but CuPy is not installed"
            ):
                get_xp(device="gpu")

    def test_get_xp_protocol_compliance(self):
        """Test that get_xp returns ArrayModule protocol compliant object."""
        from pyrator.types import ArrayModule

        xp = get_xp()
        assert isinstance(xp, ArrayModule)

        # Test that it has required attributes
        assert hasattr(xp, "asarray")
        assert hasattr(xp, "empty")
        assert hasattr(xp, "zeros")
        assert hasattr(xp, "ones")
        assert hasattr(xp, "int32")
        assert hasattr(xp, "float32")
        assert hasattr(xp, "float64")


class TestToPandas:
    """Test the to_pandas conversion function."""

    def test_to_pandas_with_pandas_unavailable(self):
        """Test to_pandas when pandas is not available."""
        with patch("pyrator.data.backends._pd", None):
            with pytest.raises(RuntimeError, match="pandas is required for this operation"):
                to_pandas({"a": [1, 2, 3]})

    def test_to_pandas_with_polars_dataframe(self):
        """Test to_pandas conversion from polars DataFrame."""
        if not has_polars():
            pytest.skip("Polars not available")

        import polars as pl

        # Create a polars DataFrame
        pl_df = pl.DataFrame({"a": [1, 2, 3], "b": ["x", "y", "z"]})

        # Convert to pandas
        pd_df = to_pandas(pl_df)

        # Verify conversion
        import pandas as pd

        assert isinstance(pd_df, pd.DataFrame)
        assert list(pd_df.columns) == ["a", "b"]
        assert len(pd_df) == 3

    def test_to_pandas_with_dict(self):
        """Test to_pandas conversion from dict."""
        data = {"a": [1, 2, 3], "b": ["x", "y", "z"]}

        pd_df = to_pandas(data)

        import pandas as pd

        assert isinstance(pd_df, pd.DataFrame)
        assert list(pd_df.columns) == ["a", "b"]
        assert len(pd_df) == 3
        assert pd_df["a"].tolist() == [1, 2, 3]
        assert pd_df["b"].tolist() == ["x", "y", "z"]

    def test_to_pandas_with_list(self):
        """Test to_pandas conversion from list."""
        data = [[1, "x"], [2, "y"], [3, "z"]]

        pd_df = to_pandas(data)

        import pandas as pd

        assert isinstance(pd_df, pd.DataFrame)
        assert len(pd_df) == 3
        assert len(pd_df.columns) == 2

    def test_to_pandas_with_pandas_dataframe(self):
        """Test to_pandas with existing pandas DataFrame."""
        import pandas as pd

        original_df = pd.DataFrame({"a": [1, 2, 3]})
        result_df = to_pandas(original_df)

        assert isinstance(result_df, pd.DataFrame)
        assert result_df is original_df  # Should return the same object

    def test_to_pandas_with_different_data_types(self):
        """Test to_pandas with various data types."""
        import pandas as pd

        # Test with numpy array
        np_data = np.array([[1, 2], [3, 4]])
        pd_df = to_pandas(np_data)
        assert isinstance(pd_df, pd.DataFrame)

        # Test with series
        pd_series = pd.Series([1, 2, 3], name="test")
        pd_df = to_pandas(pd_series)
        assert isinstance(pd_df, pd.DataFrame)


class TestToPolars:
    """Test the to_polars conversion function."""

    def test_to_polars_unavailable(self):
        """Test to_polars when polars is not available."""
        with patch("pyrator.data.backends._pl", None):
            with pytest.raises(RuntimeError, match="polars is required for this operation"):
                to_polars({"a": [1, 2, 3]})

    def test_to_polars_with_polars_dataframe(self):
        """Test to_polars with existing polars DataFrame."""
        if not has_polars():
            pytest.skip("Polars not available")

        import polars as pl

        original_df = pl.DataFrame({"a": [1, 2, 3]})
        result_df = to_polars(original_df)

        assert isinstance(result_df, pl.DataFrame)
        assert result_df is original_df  # Should return the same object

    def test_to_polars_with_dict(self):
        """Test to_polars conversion from dict."""
        if not has_polars():
            pytest.skip("Polars not available")

        import polars as pl

        data = {"a": [1, 2, 3], "b": ["x", "y", "z"]}
        pl_df = to_polars(data)

        assert isinstance(pl_df, pl.DataFrame)
        assert list(pl_df.columns) == ["a", "b"]
        assert len(pl_df) == 3

    def test_to_polars_with_pandas_dataframe(self):
        """Test to_polars conversion from pandas DataFrame."""
        if not has_polars():
            pytest.skip("Polars not available")

        import pandas as pd
        import polars as pl

        pd_df = pd.DataFrame({"a": [1, 2, 3], "b": ["x", "y", "z"]})
        pl_df = to_polars(pd_df)

        assert isinstance(pl_df, pl.DataFrame)
        assert list(pl_df.columns) == ["a", "b"]
        assert len(pl_df) == 3

    def test_to_polars_with_list(self):
        """Test to_polars conversion from list."""
        if not has_polars():
            pytest.skip("Polars not available")

        import polars as pl

        data = {"col1": [1, 2, 3], "col2": ["x", "y", "z"]}
        pl_df = to_polars(data)

        assert isinstance(pl_df, pl.DataFrame)
        assert len(pl_df) == 3

    def test_to_polars_with_different_data_types(self):
        """Test to_polars with various data types."""
        if not has_polars():
            pytest.skip("Polars not available")

        import polars as pl

        # Test with numpy array
        np_data = np.array([[1, 2], [3, 4]])
        pl_df = to_polars(np_data)
        assert isinstance(pl_df, pl.DataFrame)


class TestBackendIntegration:
    """Test integration between different backend functions."""

    def test_roundtrip_conversion(self):
        """Test round-trip conversion between pandas and polars."""
        if not has_polars():
            pytest.skip("Polars not available")

        import pandas as pd
        import polars as pl

        # Start with pandas
        original_pd = pd.DataFrame({"a": [1, 2, 3], "b": ["x", "y", "z"], "c": [1.1, 2.2, 3.3]})

        # Convert to polars and back
        pl_df = to_polars(original_pd)
        back_to_pd = to_pandas(pl_df)

        # Verify data integrity
        assert isinstance(pl_df, pl.DataFrame)
        assert isinstance(back_to_pd, pd.DataFrame)
        assert back_to_pd.equals(original_pd)

    def test_array_module_with_conversions(self):
        """Test array module usage with DataFrame conversions."""
        xp = get_xp()

        # Test creating arrays with the array module
        data = [1, 2, 3, 4, 5]
        array = xp.asarray(data, dtype=xp.float32)

        if xp is np:
            assert isinstance(array, np.ndarray)
            assert array.dtype == np.float32

    def test_backend_consistency(self):
        """Test that backend functions work consistently."""
        # Test that all availability functions return without error
        availability_results = {
            "pandas": has_pandas(),
            "polars": has_polars(),
            "duckdb": has_duckdb(),
            "cupy": has_cupy(),
            "ijson": has_ijson(),
        }

        # All should be boolean
        for backend, result in availability_results.items():
            assert isinstance(result, bool), f"{backend} availability should be bool"

    def test_error_handling_consistency(self):
        """Test consistent error handling across backend functions."""
        # Test that missing dependencies raise appropriate errors
        with patch("pyrator.data.backends._pd", None):
            with pytest.raises(RuntimeError, match="pandas is required"):
                to_pandas({"a": [1, 2, 3]})

        with patch("pyrator.data.backends._pl", None):
            with pytest.raises(RuntimeError, match="polars is required"):
                to_polars({"a": [1, 2, 3]})

        with patch("pyrator.data.backends._cupy", None):
            with pytest.raises(RuntimeError, match="CuPy is not installed"):
                get_xp(device="gpu")
