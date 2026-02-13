"""Tests for data canonicalization functionality."""

import pandas as pd
import pytest

from pyrator.data.backends import has_polars
from pyrator.data.canonical import explode_multilabel, to_long_canonical


class TestToLongCanonical:
    """Test the to_long_canonical function."""

    def test_long_format_pandas(self):
        """Test converting pandas DataFrame to long format."""
        df = pd.DataFrame(
            {
                "item": ["doc1", "doc2", "doc3"],
                "annotator": ["A", "B", "A"],
                "label": ["pos", "neg", "pos"],
            }
        )

        result = to_long_canonical(
            df, item_col="item", annotator_col="annotator", label_col="label"
        )

        # Should return the same columns in the same order
        assert isinstance(result, pd.DataFrame)
        assert list(result.columns) == ["item", "annotator", "label"]
        assert len(result) == 3
        assert result["item"].tolist() == ["doc1", "doc2", "doc3"]
        assert result["annotator"].tolist() == ["A", "B", "A"]
        assert result["label"].tolist() == ["pos", "neg", "pos"]

    def test_long_format_polars(self):
        """Test converting polars DataFrame to long format."""
        if not has_polars():
            pytest.skip("Polars not available")

        import polars as pl

        df = pl.DataFrame(
            {
                "item": ["doc1", "doc2", "doc3"],
                "annotator": ["A", "B", "A"],
                "label": ["pos", "neg", "pos"],
            }
        )

        result = to_long_canonical(
            df, item_col="item", annotator_col="annotator", label_col="label"
        )

        # Should return the same columns in the same order
        assert isinstance(result, pl.DataFrame)
        assert list(result.columns) == ["item", "annotator", "label"]
        assert len(result) == 3

    def test_wide_format_pandas(self):
        """Test converting wide format pandas DataFrame to long format."""
        df = pd.DataFrame(
            {
                "item": ["doc1", "doc2", "doc3"],
                "annotator_A": ["pos", "neg", "pos"],
                "annotator_B": ["pos", "pos", "neg"],
                "annotator_C": [None, "neg", "pos"],
            }
        )

        result = to_long_canonical(
            df,
            item_col="item",
            annotator_col="annotator",
            label_col="label",
            wide_annotator_cols=["annotator_A", "annotator_B", "annotator_C"],
        )

        # Should melt to long format and drop nulls
        assert isinstance(result, pd.DataFrame)
        assert list(result.columns) == ["item", "annotator", "label"]
        assert len(result) == 8  # 3 items * 3 annotators - 1 null = 8

        # Check specific values
        doc1_rows = result[result["item"] == "doc1"]
        assert len(doc1_rows) == 2  # A and B (C is null)

        doc2_rows = result[result["item"] == "doc2"]
        assert len(doc2_rows) == 3  # A, B, C all present

    def test_wide_format_polars(self):
        """Test converting wide format polars DataFrame to long format."""
        if not has_polars():
            pytest.skip("Polars not available")

        import polars as pl

        df = pl.DataFrame(
            {"item": ["doc1", "doc2"], "annotator_A": ["pos", "neg"], "annotator_B": ["pos", None]}
        )

        result = to_long_canonical(
            df,
            item_col="item",
            annotator_col="annotator",
            label_col="label",
            wide_annotator_cols=["annotator_A", "annotator_B"],
        )

        # Should melt to long format and drop nulls
        assert isinstance(result, pl.DataFrame)
        assert list(result.columns) == ["item", "annotator", "label"]
        assert len(result) == 3  # 2 items * 2 annotators - 1 null = 3

    def test_missing_columns_error(self):
        """Test error when required columns are missing."""
        df = pd.DataFrame(
            {
                "item": ["doc1", "doc2"],
                "annotator": ["A", "B"],
                # Missing "label" column
            }
        )

        with pytest.raises(ValueError, match="Missing columns: {'label'}"):
            to_long_canonical(df, item_col="item", annotator_col="annotator", label_col="label")

    def test_missing_wide_columns_error(self):
        """Test error when wide annotator columns are missing."""
        df = pd.DataFrame(
            {
                "item": ["doc1", "doc2"],
                "annotator_A": ["pos", "neg"],
                # Missing "annotator_B" column
            }
        )

        with pytest.raises(ValueError, match="Missing annotator columns: {'annotator_B'}"):
            to_long_canonical(
                df,
                item_col="item",
                annotator_col="annotator",
                label_col="label",
                wide_annotator_cols=["annotator_A", "annotator_B"],
            )

    def test_dict_input(self):
        """Test converting dict input to long format."""
        data = {"item": ["doc1", "doc2"], "annotator": ["A", "B"], "label": ["pos", "neg"]}

        result = to_long_canonical(
            data, item_col="item", annotator_col="annotator", label_col="label"
        )

        assert isinstance(result, pd.DataFrame)
        assert list(result.columns) == ["item", "annotator", "label"]
        assert len(result) == 2

    def test_extra_columns_dropped(self):
        """Test that extra columns are dropped in long format."""
        df = pd.DataFrame(
            {
                "item": ["doc1", "doc2"],
                "annotator": ["A", "B"],
                "label": ["pos", "neg"],
                "extra_col": ["x", "y"],  # This should be dropped
                "another_extra": [1, 2],  # This should be dropped
            }
        )

        result = to_long_canonical(
            df, item_col="item", annotator_col="annotator", label_col="label"
        )

        # Should only have the required columns
        assert list(result.columns) == ["item", "annotator", "label"]
        assert "extra_col" not in result.columns
        assert "another_extra" not in result.columns

    def test_empty_dataframe(self):
        """Test handling empty DataFrame."""
        df = pd.DataFrame({"item": [], "annotator": [], "label": []})

        result = to_long_canonical(
            df, item_col="item", annotator_col="annotator", label_col="label"
        )

        assert isinstance(result, pd.DataFrame)
        assert len(result) == 0
        assert list(result.columns) == ["item", "annotator", "label"]

    def test_copy_not_view(self):
        """Test that pandas returns a copy, not a view."""
        df = pd.DataFrame(
            {"item": ["doc1", "doc2"], "annotator": ["A", "B"], "label": ["pos", "neg"]}
        )

        result = to_long_canonical(
            df, item_col="item", annotator_col="annotator", label_col="label"
        )

        # Modify original
        df.loc[0, "label"] = "changed"

        # Result should not be affected
        assert result.loc[0, "label"] == "pos"

    def test_different_column_orders(self):
        """Test with different column orders."""
        df = pd.DataFrame(
            {"label": ["pos", "neg"], "item": ["doc1", "doc2"], "annotator": ["A", "B"]}
        )

        result = to_long_canonical(
            df, item_col="item", annotator_col="annotator", label_col="label"
        )

        # Should return columns in the specified order
        assert list(result.columns) == ["item", "annotator", "label"]


class TestExplodeMultilabel:
    """Test the explode_multilabel function."""

    def test_explode_list_type_polars(self):
        """Test exploding list type with polars."""
        if not has_polars():
            pytest.skip("Polars not available")

        import polars as pl

        df = pl.DataFrame({"item": ["doc1", "doc2"], "label": [["pos", "neg"], ["neutral"]]})

        result = explode_multilabel(df, label_col="label")

        assert isinstance(result, pl.DataFrame)
        assert len(result) == 3  # 2 + 1 labels
        assert result["item"].to_list() == ["doc1", "doc1", "doc2"]
        assert result["label"].to_list() == ["pos", "neg", "neutral"]

    def test_explode_delimited_polars(self):
        """Test exploding delimited strings with polars."""
        if not has_polars():
            pytest.skip("Polars not available")

        import polars as pl

        df = pl.DataFrame({"item": ["doc1", "doc2"], "label": ["pos|neg", "neutral"]})

        result = explode_multilabel(df, label_col="label", mode="delim", delim="|")

        assert isinstance(result, pl.DataFrame)
        assert len(result) == 3
        assert result["item"].to_list() == ["doc1", "doc1", "doc2"]
        assert result["label"].to_list() == ["pos", "neg", "neutral"]

    def test_explode_json_polars(self):
        """Test exploding JSON arrays with polars."""
        if not has_polars():
            pytest.skip("Polars not available")

        import polars as pl

        df = pl.DataFrame({"item": ["doc1", "doc2"], "label": ['["pos", "neg"]', '["neutral"]']})

        result = explode_multilabel(df, label_col="label", mode="json")

        assert isinstance(result, pl.DataFrame)
        assert len(result) == 3
        assert result["item"].to_list() == ["doc1", "doc1", "doc2"]

    def test_explode_list_type_pandas(self):
        """Test exploding list type with pandas."""
        df = pd.DataFrame({"item": ["doc1", "doc2"], "label": [["pos", "neg"], ["neutral"]]})

        result = explode_multilabel(df, label_col="label")

        assert isinstance(result, pd.DataFrame)
        assert len(result) == 3
        assert result["item"].tolist() == ["doc1", "doc1", "doc2"]
        assert result["label"].tolist() == ["pos", "neg", "neutral"]

    def test_explode_delimited_pandas(self):
        """Test exploding delimited strings with pandas."""
        df = pd.DataFrame({"item": ["doc1", "doc2"], "label": ["pos|neg", "neutral"]})

        result = explode_multilabel(df, label_col="label", mode="delim", delim="|")

        assert isinstance(result, pd.DataFrame)
        assert len(result) == 3
        assert result["item"].tolist() == ["doc1", "doc1", "doc2"]
        assert result["label"].tolist() == ["pos", "neg", "neutral"]

    def test_explode_json_pandas(self):
        """Test exploding JSON arrays with pandas."""
        df = pd.DataFrame({"item": ["doc1", "doc2"], "label": ['["pos", "neg"]', '["neutral"]']})

        result = explode_multilabel(df, label_col="label", mode="json")

        assert isinstance(result, pd.DataFrame)
        assert len(result) == 3
        assert result["item"].tolist() == ["doc1", "doc1", "doc2"]

    def test_explode_auto_mode_list(self):
        """Test auto mode with list type."""
        df = pd.DataFrame({"item": ["doc1", "doc2"], "label": [["pos", "neg"], ["neutral"]]})

        result = explode_multilabel(df, label_col="label", mode="auto")

        assert len(result) == 3
        assert result["label"].tolist() == ["pos", "neg", "neutral"]

    def test_explode_auto_mode_delimited(self):
        """Test auto mode with delimited strings."""
        df = pd.DataFrame({"item": ["doc1", "doc2"], "label": ["pos|neg", "neutral"]})

        result = explode_multilabel(df, label_col="label", mode="auto", delim="|")

        assert len(result) == 3
        assert result["label"].tolist() == ["pos", "neg", "neutral"]

    def test_explode_missing_column_error(self):
        """Test error when label column is missing."""
        df = pd.DataFrame({"item": ["doc1", "doc2"], "other_col": ["x", "y"]})

        with pytest.raises(ValueError, match="Column 'label' not found"):
            explode_multilabel(df, label_col="label")

    def test_explode_empty_dataframe(self):
        """Test exploding empty DataFrame."""
        df = pd.DataFrame({"item": [], "label": []})

        result = explode_multilabel(df, label_col="label")

        assert isinstance(result, pd.DataFrame)
        assert len(result) == 0

    def test_explode_with_nulls(self):
        """Test exploding with null values."""
        df = pd.DataFrame(
            {"item": ["doc1", "doc2", "doc3"], "label": [["pos", "neg"], None, ["neutral"]]}
        )

        result = explode_multilabel(df, label_col="label")

        # Should handle nulls appropriately
        assert len(result) == 3  # 2 + 0 + 1 (null row becomes empty)
        assert result["item"].tolist() == ["doc1", "doc1", "doc3"]

    def test_explode_custom_delimiter(self):
        """Test exploding with custom delimiter."""
        df = pd.DataFrame({"item": ["doc1", "doc2"], "label": ["pos;neg;neutral", "positive"]})

        result = explode_multilabel(df, label_col="label", mode="delim", delim=";")

        assert len(result) == 4
        assert result["label"].tolist() == ["pos", "neg", "neutral", "positive"]

    def test_explode_single_labels(self):
        """Test exploding single labels (no change)."""
        df = pd.DataFrame({"item": ["doc1", "doc2"], "label": ["pos", "neg"]})

        result = explode_multilabel(df, label_col="label", mode="delim", delim="|")

        # Should remain unchanged
        assert len(result) == 2
        assert result["label"].tolist() == ["pos", "neg"]

    def test_explode_copy_not_view(self):
        """Test that pandas returns a copy, not a view."""
        df = pd.DataFrame({"item": ["doc1", "doc2"], "label": ["pos|neg", "neutral"]})

        result = explode_multilabel(df, label_col="label", mode="delim", delim="|")

        # Modify original
        df.loc[0, "label"] = "changed"

        # Result should not be affected (using .iloc to get first row)
        assert result.iloc[0]["label"] == "pos"

    def test_explode_with_numeric_data(self):
        """Test exploding with numeric data."""
        df = pd.DataFrame({"item": ["doc1", "doc2"], "label": ["1|2|3", "4"]})

        result = explode_multilabel(df, label_col="label", mode="delim", delim="|")

        assert len(result) == 4
        assert result["label"].tolist() == ["1", "2", "3", "4"]

    def test_explode_mixed_data_types(self):
        """Test exploding with mixed data types in label column."""
        df = pd.DataFrame(
            {"item": ["doc1", "doc2", "doc3"], "label": [["pos", "neg"], "single", "multi|label"]}
        )

        # Should handle mixed types gracefully
        result = explode_multilabel(df, label_col="label", mode="auto", delim="|")

        assert isinstance(result, pd.DataFrame)
        # The exact behavior depends on auto-detection, but it shouldn't crash

    def test_explode_with_empty_strings(self):
        """Test exploding with empty strings."""
        df = pd.DataFrame({"item": ["doc1", "doc2"], "label": ["pos|", "|neg"]})

        result = explode_multilabel(df, label_col="label", mode="delim", delim="|")

        # Should handle empty strings
        assert len(result) >= 2  # At least the non-empty parts
