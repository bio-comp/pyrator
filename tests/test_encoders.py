"""Tests for category encoder functionality."""

from unittest.mock import MagicMock, patch

import numpy as np
import pandas as pd
import pytest

from pyrator.data.backends import has_polars
from pyrator.data.encoders import CategoryEncoder, Encoded


class TestEncoded:
    """Test the Encoded dataclass."""

    def test_encoded_creation(self):
        """Test creating an Encoded object."""
        items = np.array([0, 1, 2], dtype=np.int32)
        annotators = np.array([0, 1, 0], dtype=np.int32)
        labels = np.array([1, 0, 1], dtype=np.int32)
        maps = {
            "item": {1: 0, 2: 1, 3: 2},
            "annotator": {"A": 0, "B": 1},
            "label": {"pos": 0, "neg": 1},
        }

        encoded = Encoded(
            items=items, annotators=annotators, labels=labels, maps=maps, backend="numpy"
        )

        assert np.array_equal(encoded.items, items)
        assert np.array_equal(encoded.annotators, annotators)
        assert np.array_equal(encoded.labels, labels)
        assert encoded.maps == maps
        assert encoded.backend == "numpy"

    def test_encoded_attributes(self):
        """Test Encoded attributes are accessible."""
        encoded = Encoded(
            items=np.array([1, 2], dtype=np.int32),
            annotators=np.array([0, 1], dtype=np.int32),
            labels=np.array([1, 0], dtype=np.int32),
            maps={"test": {"a": 0}},
            backend="cupy",
        )

        assert hasattr(encoded, "items")
        assert hasattr(encoded, "annotators")
        assert hasattr(encoded, "labels")
        assert hasattr(encoded, "maps")
        assert hasattr(encoded, "backend")


class TestCategoryEncoder:
    """Test CategoryEncoder class."""

    def test_encoder_init_default(self):
        """Test CategoryEncoder initialization with default parameters."""
        encoder = CategoryEncoder()
        assert encoder.device is None
        assert encoder.backend == "numpy"
        assert encoder._maps is None

    def test_encoder_init_with_device(self):
        """Test CategoryEncoder initialization with device parameter."""
        encoder = CategoryEncoder(device="cpu")
        assert encoder.device == "cpu"
        assert encoder.backend == "numpy"

    def test_encoder_init_gpu(self):
        """Test CategoryEncoder initialization with GPU device."""
        encoder = CategoryEncoder(device="gpu")
        assert encoder.device == "gpu"
        assert encoder.backend == "cupy"

    def test_fit_pandas(self):
        """Test fitting encoder with pandas DataFrame."""
        encoder = CategoryEncoder()

        # Create test data
        df = pd.DataFrame(
            {
                "item": ["doc1", "doc2", "doc1", "doc3"],
                "annotator": ["A", "B", "A", "C"],
                "label": ["pos", "neg", "pos", "neutral"],
            }
        )

        # Fit the encoder
        fitted_encoder = encoder.fit(
            df, item_col="item", annotator_col="annotator", label_col="label"
        )

        # Check that fit returns self
        assert fitted_encoder is encoder
        assert encoder._maps is not None

        # Check maps structure
        assert "item" in encoder._maps
        assert "annotator" in encoder._maps
        assert "label" in encoder._maps

        # Check map contents (should be sorted unique values)
        expected_items = {"doc1": 0, "doc2": 1, "doc3": 2}
        expected_annotators = {"A": 0, "B": 1, "C": 2}
        expected_labels = {"neg": 0, "neutral": 1, "pos": 2}

        assert encoder._maps["item"] == expected_items
        assert encoder._maps["annotator"] == expected_annotators
        assert encoder._maps["label"] == expected_labels

    def test_fit_polars(self):
        """Test fitting encoder with polars DataFrame."""
        if not has_polars():
            pytest.skip("Polars not available")

        import polars as pl

        encoder = CategoryEncoder()

        # Create test data
        df = pl.DataFrame(
            {
                "item": ["doc1", "doc2", "doc1", "doc3"],
                "annotator": ["A", "B", "A", "C"],
                "label": ["pos", "neg", "pos", "neutral"],
            }
        )

        # Fit the encoder
        encoder.fit(df, item_col="item", annotator_col="annotator", label_col="label")

        # Check maps structure
        assert encoder._maps is not None
        assert "item" in encoder._maps
        assert "annotator" in encoder._maps
        assert "label" in encoder._maps

    def test_fit_with_numeric_data(self):
        """Test fitting encoder with numeric data."""
        encoder = CategoryEncoder()

        df = pd.DataFrame(
            {"item": [1, 2, 1, 3], "annotator": [10, 20, 10, 30], "label": [0, 1, 0, 2]}
        )

        encoder.fit(df, item_col="item", annotator_col="annotator", label_col="label")

        # Check that numeric values are handled correctly
        assert encoder._maps["item"] == {1: 0, 2: 1, 3: 2}
        assert encoder._maps["annotator"] == {10: 0, 20: 1, 30: 2}
        assert encoder._maps["label"] == {0: 0, 1: 1, 2: 2}

    def test_fit_with_mixed_types(self):
        """Test fitting encoder with mixed data types."""
        encoder = CategoryEncoder()

        df = pd.DataFrame(
            {
                "item": ["doc1", "doc2", "doc3"],
                "annotator": [1, 2, 3],  # numeric
                "label": ["pos", "neg", "pos"],
            }
        )

        encoder.fit(df, item_col="item", annotator_col="annotator", label_col="label")

        assert encoder._maps["item"] == {"doc1": 0, "doc2": 1, "doc3": 2}
        assert encoder._maps["annotator"] == {1: 0, 2: 1, 3: 2}
        assert encoder._maps["label"] == {"neg": 0, "pos": 1}

    def test_transform_without_fit(self):
        """Test transform without fitting first."""
        encoder = CategoryEncoder()

        df = pd.DataFrame(
            {"item": ["doc1", "doc2"], "annotator": ["A", "B"], "label": ["pos", "neg"]}
        )

        with pytest.raises(
            RuntimeError, match="CategoryEncoder must be fit\\(\\) before transform\\(\\)"
        ):
            encoder.transform(df, item_col="item", annotator_col="annotator", label_col="label")

    def test_transform_pandas(self):
        """Test transform with pandas DataFrame."""
        encoder = CategoryEncoder()

        # Fit first
        train_df = pd.DataFrame(
            {
                "item": ["doc1", "doc2", "doc3"],
                "annotator": ["A", "B", "C"],
                "label": ["pos", "neg", "neutral"],
            }
        )
        encoder.fit(train_df, item_col="item", annotator_col="annotator", label_col="label")

        # Transform with same data
        encoded = encoder.transform(
            train_df, item_col="item", annotator_col="annotator", label_col="label"
        )

        # Check encoded structure
        assert isinstance(encoded, Encoded)
        assert encoded.backend == "numpy"
        assert encoded.maps == encoder._maps

        # Check array types and shapes
        assert encoded.items.dtype == np.int32
        assert encoded.annotators.dtype == np.int32
        assert encoded.labels.dtype == np.int32
        assert len(encoded.items) == 3
        assert len(encoded.annotators) == 3
        assert len(encoded.labels) == 3

        # Check encoded values
        expected_items = np.array([0, 1, 2], dtype=np.int32)
        expected_annotators = np.array([0, 1, 2], dtype=np.int32)
        expected_labels = np.array([2, 0, 1], dtype=np.int32)  # neg:0, neutral:1, pos:2

        assert np.array_equal(encoded.items, expected_items)
        assert np.array_equal(encoded.annotators, expected_annotators)
        assert np.array_equal(encoded.labels, expected_labels)

    def test_transform_polars(self):
        """Test transform with polars DataFrame."""
        if not has_polars():
            pytest.skip("Polars not available")

        import polars as pl

        encoder = CategoryEncoder()

        # Fit first
        train_df = pl.DataFrame(
            {
                "item": ["doc1", "doc2", "doc3"],
                "annotator": ["A", "B", "C"],
                "label": ["pos", "neg", "neutral"],
            }
        )
        encoder.fit(train_df, item_col="item", annotator_col="annotator", label_col="label")

        # Transform
        encoded = encoder.transform(
            train_df, item_col="item", annotator_col="annotator", label_col="label"
        )

        # Check encoded structure
        assert isinstance(encoded, Encoded)
        assert encoded.backend == "numpy"  # Default backend
        assert len(encoded.items) == 3

    def test_transform_with_unknown_values(self):
        """Test transform with unknown values (should map to -1)."""
        encoder = CategoryEncoder()

        # Fit with training data
        train_df = pd.DataFrame(
            {"item": ["doc1", "doc2"], "annotator": ["A", "B"], "label": ["pos", "neg"]}
        )
        encoder.fit(train_df, item_col="item", annotator_col="annotator", label_col="label")

        # Transform with test data containing unknown values
        test_df = pd.DataFrame(
            {
                "item": ["doc1", "doc3"],  # doc3 is unknown
                "annotator": ["A", "C"],  # C is unknown
                "label": ["pos", "neutral"],  # neutral is unknown
            }
        )

        encoded = encoder.transform(
            test_df, item_col="item", annotator_col="annotator", label_col="label"
        )

        # Check that unknown values are mapped to -1
        expected_items = np.array([0, -1], dtype=np.int32)  # doc1:0, doc3:-1
        expected_annotators = np.array([0, -1], dtype=np.int32)  # A:0, C:-1
        expected_labels = np.array([1, -1], dtype=np.int32)  # pos:1, neutral:-1

        assert np.array_equal(encoded.items, expected_items)
        assert np.array_equal(encoded.annotators, expected_annotators)
        assert np.array_equal(encoded.labels, expected_labels)

    def test_transform_gpu(self):
        """Test transform with GPU backend."""
        # Mock CuPy as available
        mock_cupy = MagicMock()
        mock_cupy.asarray = MagicMock(side_effect=lambda x, dtype=None: np.asarray(x, dtype=dtype))
        mock_cupy.int32 = np.int32

        with patch("pyrator.data.backends.get_xp", return_value=mock_cupy):
            # Also mock the _cupy check to avoid the RuntimeError
            with patch("pyrator.data.backends._cupy", mock_cupy):
                encoder = CategoryEncoder(device="gpu")

                # Fit with pandas
                train_df = pd.DataFrame(
                    {"item": ["doc1", "doc2"], "annotator": ["A", "B"], "label": ["pos", "neg"]}
                )
                encoder.fit(train_df, item_col="item", annotator_col="annotator", label_col="label")

                # Transform
                encoded = encoder.transform(
                    train_df, item_col="item", annotator_col="annotator", label_col="label"
                )

                assert encoded.backend == "cupy"
                assert mock_cupy.asarray.called

    def test_fit_transform_consistency(self):
        """Test that fit and transform are consistent across calls."""
        encoder = CategoryEncoder()

        df = pd.DataFrame(
            {
                "item": ["doc1", "doc2", "doc1", "doc3"],
                "annotator": ["A", "B", "A", "C"],
                "label": ["pos", "neg", "pos", "neutral"],
            }
        )

        # Fit and transform
        encoder.fit(df, item_col="item", annotator_col="annotator", label_col="label")
        encoded1 = encoder.transform(
            df, item_col="item", annotator_col="annotator", label_col="label"
        )

        # Transform again (should be identical)
        encoded2 = encoder.transform(
            df, item_col="item", annotator_col="annotator", label_col="label"
        )

        # Check results are identical
        assert np.array_equal(encoded1.items, encoded2.items)
        assert np.array_equal(encoded1.annotators, encoded2.annotators)
        assert np.array_equal(encoded1.labels, encoded2.labels)
        assert encoded1.maps == encoded2.maps
        assert encoded1.backend == encoded2.backend

    def test_encoder_with_different_column_names(self):
        """Test encoder with non-standard column names."""
        encoder = CategoryEncoder()

        df = pd.DataFrame(
            {"document": ["doc1", "doc2"], "rater": ["A", "B"], "classification": ["pos", "neg"]}
        )

        # Fit with custom column names
        encoder.fit(df, item_col="document", annotator_col="rater", label_col="classification")

        # Check maps use the correct keys
        assert "item" in encoder._maps
        assert "annotator" in encoder._maps
        assert "label" in encoder._maps

        # But the values should come from the specified columns
        assert encoder._maps["item"] == {"doc1": 0, "doc2": 1}
        assert encoder._maps["annotator"] == {"A": 0, "B": 1}
        assert encoder._maps["label"] == {"neg": 0, "pos": 1}

    def test_encoder_with_duplicate_values(self):
        """Test encoder with duplicate values in columns."""
        encoder = CategoryEncoder()

        df = pd.DataFrame(
            {
                "item": ["doc1", "doc1", "doc2", "doc1"],  # doc1 appears 3 times
                "annotator": ["A", "A", "B", "A"],  # A appears 3 times
                "label": ["pos", "pos", "neg", "pos"],  # pos appears 3 times
            }
        )

        encoder.fit(df, item_col="item", annotator_col="annotator", label_col="label")

        # Check that duplicates are handled correctly (unique values only)
        assert encoder._maps["item"] == {"doc1": 0, "doc2": 1}
        assert encoder._maps["annotator"] == {"A": 0, "B": 1}
        assert encoder._maps["label"] == {"neg": 0, "pos": 1}

    def test_encoder_with_empty_dataframe(self):
        """Test encoder with empty DataFrame."""
        encoder = CategoryEncoder()

        df = pd.DataFrame({"item": [], "annotator": [], "label": []})

        # Should handle empty DataFrame gracefully
        encoder.fit(df, item_col="item", annotator_col="annotator", label_col="label")

        # Maps should be empty
        assert encoder._maps["item"] == {}
        assert encoder._maps["annotator"] == {}
        assert encoder._maps["label"] == {}

        # Transform should work with empty DataFrame
        encoded = encoder.transform(
            df, item_col="item", annotator_col="annotator", label_col="label"
        )

        assert len(encoded.items) == 0
        assert len(encoded.annotators) == 0
        assert len(encoded.labels) == 0

    def test_encoder_error_handling(self):
        """Test encoder error handling."""
        encoder = CategoryEncoder()

        df = pd.DataFrame(
            {"item": ["doc1", "doc2"], "annotator": ["A", "B"], "label": ["pos", "neg"]}
        )

        # Test with missing columns
        with pytest.raises(KeyError):
            encoder.fit(df, item_col="nonexistent", annotator_col="annotator", label_col="label")

        # Fit properly first
        encoder.fit(df, item_col="item", annotator_col="annotator", label_col="label")

        # Test transform with missing columns
        with pytest.raises(KeyError):
            encoder.transform(
                df, item_col="nonexistent", annotator_col="annotator", label_col="label"
            )
