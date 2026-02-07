"""Tests for type definitions and protocols."""

import numpy as np

from pyrator.types import ArrayLike, ArrayModule, FrameLike, IntLike, RealLike


class TestTypeAliases:
    """Test type aliases."""

    def test_int_like_types(self):
        """Test IntLike type alias with various integer types."""
        # Python int
        assert isinstance(42, IntLike)

        # NumPy integer types
        assert isinstance(np.int32(5), IntLike)
        assert isinstance(np.int64(10), IntLike)
        assert isinstance(np.int32(5), np.integer)  # Should be IntLike

        # Non-integer types should not be IntLike
        assert not isinstance(3.14, IntLike)
        assert not isinstance("hello", IntLike)

    def test_real_like_types(self):
        """Test RealLike type alias with various real number types."""
        # Python int and float
        assert isinstance(42, RealLike)
        assert isinstance(3.14, RealLike)

        # NumPy numeric types
        assert isinstance(np.int32(5), RealLike)
        assert isinstance(np.int64(10), RealLike)
        assert isinstance(np.float32(2.5), RealLike)
        assert isinstance(np.float64(1.5), RealLike)

        # Non-numeric types should not be RealLike
        assert not isinstance("hello", RealLike)
        assert not isinstance([1, 2, 3], RealLike)

    def test_array_like_type(self):
        """Test ArrayLike type alias."""
        # This is mainly for type checking, but we can test it's usable
        array: ArrayLike = np.array([1, 2, 3])
        assert isinstance(array, np.ndarray)

        # Lists should also be acceptable as ArrayLike
        list_array: ArrayLike = [1, 2, 3]
        assert isinstance(list_array, list)

    def test_frame_like_type(self):
        """Test FrameLike type alias."""
        # This is mainly for type checking
        import pandas as pd

        df: FrameLike = pd.DataFrame({"a": [1, 2, 3]})
        assert isinstance(df, pd.DataFrame)


class TestArrayModuleProtocol:
    """Test ArrayModule protocol implementation."""

    def test_numpy_array_module_compliance(self):
        """Test that NumPy module complies with ArrayModule protocol."""
        np_module = np

        # Check required methods exist
        assert hasattr(np_module, "asarray")
        assert hasattr(np_module, "empty")
        assert hasattr(np_module, "zeros")
        assert hasattr(np_module, "ones")

        # Check required dtypes exist
        assert hasattr(np_module, "int32")
        assert hasattr(np_module, "float32")
        assert hasattr(np_module, "float64")

        # Test method functionality
        test_array = [1, 2, 3]
        result = np_module.asarray(test_array)
        assert isinstance(result, np.ndarray)
        assert np.array_equal(result, [1, 2, 3])

        empty = np_module.empty((3, 2))
        assert empty.shape == (3, 2)

        zeros = np_module.zeros((2, 3))
        assert zeros.shape == (2, 3)
        assert np.all(zeros == 0)

        ones = np_module.ones((2, 2))
        assert ones.shape == (2, 2)
        assert np.all(ones == 1)

    def test_array_module_protocol_checking(self):
        """Test runtime checking of ArrayModule protocol."""

        # Verify the protocol is runtime checkable
        assert hasattr(ArrayModule, "__instancecheck__")
        assert hasattr(ArrayModule, "__subclasscheck__")

        # Test NumPy module compliance
        assert isinstance(np, ArrayModule)

        # Test non-compliant object
        class NotArrayModule:
            pass

        not_module = NotArrayModule()
        assert not isinstance(not_module, ArrayModule)

    def test_array_module_with_custom_backend(self):
        """Test ArrayModule protocol with a custom minimal backend."""

        class MinimalArrayBackend:
            """Minimal implementation of ArrayModule protocol."""

            def __init__(self):
                self.int32 = "int32"
                self.float32 = "float32"
                self.float64 = "float64"

            def asarray(self, obj, dtype=None):
                return f"asarray({obj}, dtype={dtype})"

            def empty(self, shape, dtype=None):
                return f"empty({shape}, dtype={dtype})"

            def zeros(self, shape, dtype=None):
                return f"zeros({shape}, dtype={dtype})"

            def ones(self, shape, dtype=None):
                return f"ones({shape}, dtype={dtype})"

        backend = MinimalArrayBackend()
        assert isinstance(backend, ArrayModule)

        # Test method calls
        assert backend.asarray([1, 2, 3]) == "asarray([1, 2, 3], dtype=None)"
        assert backend.empty((3, 2)) == "empty((3, 2), dtype=None)"
        assert backend.zeros((2, 3)) == "zeros((2, 3), dtype=None)"
        assert backend.ones((2, 2)) == "ones((2, 2), dtype=None)"

        # Test dtype attributes
        assert backend.int32 == "int32"
        assert backend.float32 == "float32"
        assert backend.float64 == "float64"

    def test_array_module_missing_methods(self):
        """Test that objects missing required methods don't comply."""

        class IncompleteBackend:
            """Missing some required methods."""

            def asarray(self, obj, dtype=None):
                return obj

            def empty(self, shape, dtype=None):
                return shape

        incomplete = IncompleteBackend()
        # Should not be considered an ArrayModule due to missing methods
        assert not isinstance(incomplete, ArrayModule)

    def test_array_module_method_signatures(self):
        """Test that ArrayModule methods have correct signatures."""
        # This is mainly for static type checking, but we can verify
        # the methods are callable with expected arguments

        # Test asarray
        result = np.asarray([1, 2, 3])
        assert isinstance(result, np.ndarray)

        result_with_dtype = np.asarray([1, 2, 3], dtype=np.float32)
        assert result_with_dtype.dtype == np.float32

        # Test empty
        empty_result = np.empty((3, 2))
        assert empty_result.shape == (3, 2)

        empty_with_dtype = np.empty((2, 3), dtype=np.int32)
        assert empty_with_dtype.dtype == np.int32

        # Test zeros
        zeros_result = np.zeros((2, 3))
        assert np.all(zeros_result == 0)

        zeros_with_dtype = np.zeros((2, 3), dtype=np.float64)
        assert zeros_with_dtype.dtype == np.float64

        # Test ones
        ones_result = np.ones((2, 2))
        assert np.all(ones_result == 1)

        ones_with_dtype = np.ones((2, 2), dtype=np.int32)
        assert ones_with_dtype.dtype == np.int32


class TestTypeUsage:
    """Test usage of types in practical scenarios."""

    def test_array_like_in_function_signatures(self):
        """Test ArrayLike in function signatures."""

        def process_array(data: ArrayLike) -> ArrayLike:
            """Process an array-like object."""
            return np.asarray(data) * 2

        # Test with list
        result = process_array([1, 2, 3])
        assert np.array_equal(result, [2, 4, 6])

        # Test with NumPy array
        result = process_array(np.array([4, 5, 6]))
        assert np.array_equal(result, [8, 10, 12])

    def test_frame_like_in_function_signatures(self):
        """Test FrameLike in function signatures."""
        import pandas as pd

        def process_frame(data: FrameLike) -> FrameLike:
            """Process a frame-like object."""
            if isinstance(data, pd.DataFrame):
                return data.copy()
            else:
                return pd.DataFrame(data)

        # Test with dict
        result = process_frame({"a": [1, 2, 3]})
        assert isinstance(result, pd.DataFrame)
        assert list(result.columns) == ["a"]

        # Test with DataFrame
        df = pd.DataFrame({"b": [4, 5, 6]})
        result = process_frame(df)
        assert isinstance(result, pd.DataFrame)
        assert result is not df  # Should be a copy

    def test_numeric_type_validation(self):
        """Test numeric type validation functions."""

        def is_valid_integer(value: IntLike) -> bool:
            """Check if value is a valid integer."""
            return isinstance(value, IntLike)

        def is_valid_real(value: RealLike) -> bool:
            """Check if value is a valid real number."""
            return isinstance(value, RealLike)

        # Test integer validation
        assert is_valid_integer(42)
        assert is_valid_integer(np.int32(5))
        assert not is_valid_integer(3.14)
        assert not is_valid_integer("42")

        # Test real number validation
        assert is_valid_real(42)
        assert is_valid_real(3.14)
        assert is_valid_real(np.int64(10))
        assert is_valid_real(np.float32(2.5))
        assert not is_valid_real("3.14")
        assert not is_valid_real([1, 2, 3])
