"""Tests for batch processing utilities."""

from typing import Any, List

import pytest

from pyrator.data.batching import batch_data, process_in_batches


class TestBatchData:
    """Test the batch_data function."""

    def test_batch_data_exact_multiple(self):
        """Test batching when data size is exact multiple of batch size."""
        data = [1, 2, 3, 4, 5, 6]
        batch_size = 2

        batches = list(batch_data(data, batch_size))

        assert len(batches) == 3
        assert batches[0] == [1, 2]
        assert batches[1] == [3, 4]
        assert batches[2] == [5, 6]

    def test_batch_data_partial_last_batch(self):
        """Test batching when last batch is partial."""
        data = [1, 2, 3, 4, 5]
        batch_size = 2

        batches = list(batch_data(data, batch_size))

        assert len(batches) == 3
        assert batches[0] == [1, 2]
        assert batches[1] == [3, 4]
        assert batches[2] == [5]

    def test_batch_data_single_item(self):
        """Test batching with batch size of 1."""
        data = [1, 2, 3]
        batch_size = 1

        batches = list(batch_data(data, batch_size))

        assert len(batches) == 3
        assert batches[0] == [1]
        assert batches[1] == [2]
        assert batches[2] == [3]

    def test_batch_data_larger_batch_size(self):
        """Test batching when batch size is larger than data."""
        data = [1, 2, 3]
        batch_size = 10

        batches = list(batch_data(data, batch_size))

        assert len(batches) == 1
        assert batches[0] == [1, 2, 3]

    def test_batch_data_empty_data(self):
        """Test batching with empty data."""
        data = []
        batch_size = 3

        batches = list(batch_data(data, batch_size))

        assert len(batches) == 0

    def test_batch_data_different_data_types(self):
        """Test batching with different data types."""
        # Test with strings
        string_data = ["a", "b", "c", "d"]
        batches = list(batch_data(string_data, 2))
        assert batches == [["a", "b"], ["c", "d"]]

        # Test with tuples
        tuple_data = [(1, "a"), (2, "b"), (3, "c")]
        batches = list(batch_data(tuple_data, 2))
        assert batches == [[(1, "a"), (2, "b")], [(3, "c")]]

        # Test with objects
        class TestObj:
            def __init__(self, val):
                self.val = val

        obj_data = [TestObj(1), TestObj(2), TestObj(3), TestObj(4)]
        batches = list(batch_data(obj_data, 2))
        assert len(batches) == 2
        assert len(batches[0]) == 2
        assert len(batches[1]) == 2

    def test_batch_data_generator(self):
        """Test batching with generator input."""

        def number_generator():
            for i in range(10):
                yield i

        batches = list(batch_data(number_generator(), 3))

        assert len(batches) == 4
        assert batches[0] == [0, 1, 2]
        assert batches[1] == [3, 4, 5]
        assert batches[2] == [6, 7, 8]
        assert batches[3] == [9]

    def test_batch_data_iterator_behavior(self):
        """Test that batch_data returns an iterator."""
        data = [1, 2, 3, 4, 5]
        batch_size = 2

        batch_iter = batch_data(data, batch_size)

        # Should be an iterator
        assert hasattr(batch_iter, "__iter__")
        assert hasattr(batch_iter, "__next__")

        # Should be able to iterate multiple times
        first_pass = list(batch_iter)
        second_pass = list(batch_data(data, batch_size))

        assert first_pass == second_pass

    def test_batch_data_large_dataset(self):
        """Test batching with a large dataset."""
        data = list(range(1000))
        batch_size = 100

        batches = list(batch_data(data, batch_size))

        assert len(batches) == 10
        for i, batch in enumerate(batches):
            assert len(batch) == 100
            assert batch[0] == i * 100
            assert batch[-1] == i * 100 + 99

    def test_batch_data_invalid_batch_size(self):
        """Test batching with invalid batch sizes."""
        data = [1, 2, 3]

        # Batch size of 0 should work but create no batches
        batches = list(batch_data(data, 0))
        assert len(batches) == 0

        # Negative batch size should also work but create no batches
        batches = list(batch_data(data, -1))
        assert len(batches) == 0


class TestProcessInBatches:
    """Test the process_in_batches function."""

    def test_process_in_batches_basic(self):
        """Test basic batch processing."""
        data = [1, 2, 3, 4, 5]
        batch_size = 2

        def processor(batch: List[int]) -> int:
            return sum(batch)

        results = list(process_in_batches(data, processor, batch_size))

        assert len(results) == 3
        assert results[0] == 3  # 1 + 2
        assert results[1] == 7  # 3 + 4
        assert results[2] == 5  # 5

    def test_process_in_batches_different_processor(self):
        """Test batch processing with different processor types."""
        data = ["hello", "world", "test", "batch"]
        batch_size = 2

        def concat_processor(batch: List[str]) -> str:
            return " ".join(batch)

        results = list(process_in_batches(data, concat_processor, batch_size))

        assert len(results) == 2
        assert results[0] == "hello world"
        assert results[1] == "test batch"

    def test_process_in_batches_with_partial_last_batch(self):
        """Test batch processing with partial last batch."""
        data = [1, 2, 3, 4, 5]
        batch_size = 3

        def count_processor(batch: List[int]) -> int:
            return len(batch)

        results = list(process_in_batches(data, count_processor, batch_size))

        assert len(results) == 2
        assert results[0] == 3  # First batch has 3 items
        assert results[1] == 2  # Second batch has 2 items

    def test_process_in_batches_empty_data(self):
        """Test batch processing with empty data."""
        data = []
        batch_size = 2

        def processor(batch: List[Any]) -> str:
            return f"processed_{len(batch)}"

        results = list(process_in_batches(data, processor, batch_size))

        assert len(results) == 0

    def test_process_in_batches_single_item_batches(self):
        """Test batch processing with batch size of 1."""
        data = [10, 20, 30]
        batch_size = 1

        def multiply_processor(batch: List[int]) -> int:
            return batch[0] * 2

        results = list(process_in_batches(data, multiply_processor, batch_size))

        assert len(results) == 3
        assert results == [20, 40, 60]

    def test_process_in_batches_large_batch_size(self):
        """Test batch processing when batch size exceeds data size."""
        data = [1, 2, 3]
        batch_size = 10

        def sum_processor(batch: List[int]) -> int:
            return sum(batch)

        results = list(process_in_batches(data, sum_processor, batch_size))

        assert len(results) == 1
        assert results[0] == 6

    def test_process_in_batches_with_generator(self):
        """Test batch processing with generator input."""

        def number_generator():
            for i in range(6):
                yield i

        def sum_processor(batch: List[int]) -> int:
            return sum(batch)

        results = list(process_in_batches(number_generator(), sum_processor, 2))

        assert len(results) == 3
        assert results == [1, 5, 9]  # (0+1), (2+3), (4+5)

    def test_process_in_batches_processor_exception(self):
        """Test batch processing when processor raises exception."""
        data = [1, 2, 3, 4]
        batch_size = 2

        def failing_processor(batch: List[int]) -> int:
            if sum(batch) > 3:
                raise ValueError("Sum too large")
            return sum(batch)

        with pytest.raises(ValueError, match="Sum too large"):
            list(process_in_batches(data, failing_processor, batch_size))

    def test_process_in_batches_complex_processor(self):
        """Test batch processing with complex processor logic."""
        data = [1, 2, 3, 4, 5, 6]
        batch_size = 3

        def complex_processor(batch: List[int]) -> dict:
            return {
                "sum": sum(batch),
                "mean": sum(batch) / len(batch),
                "max": max(batch),
                "min": min(batch),
                "count": len(batch),
            }

        results = list(process_in_batches(data, complex_processor, batch_size))

        assert len(results) == 2
        assert results[0] == {"sum": 6, "mean": 2.0, "max": 3, "min": 1, "count": 3}
        assert results[1] == {"sum": 15, "mean": 5.0, "max": 6, "min": 4, "count": 3}

    def test_process_in_batches_stateful_processor(self):
        """Test batch processing with stateful processor."""
        data = [1, 2, 3, 4, 5]
        batch_size = 2

        class StatefulProcessor:
            def __init__(self):
                self.call_count = 0

            def __call__(self, batch: List[int]) -> str:
                self.call_count += 1
                return f"batch_{self.call_count}_sum_{sum(batch)}"

        processor = StatefulProcessor()
        results = list(process_in_batches(data, processor, batch_size))

        assert len(results) == 3
        assert results[0] == "batch_1_sum_3"
        assert results[1] == "batch_2_sum_7"
        assert results[2] == "batch_3_sum_5"
        assert processor.call_count == 3

    def test_process_in_batches_lazy_evaluation(self):
        """Test that process_in_batches is lazy."""
        data = [1, 2, 3, 4, 5, 6]
        batch_size = 2

        call_count = 0

        def counting_processor(batch: List[int]) -> int:
            nonlocal call_count
            call_count += 1
            return sum(batch)

        # Create iterator but don't consume
        results_iter = process_in_batches(data, counting_processor, batch_size)

        # Processor should not have been called yet
        assert call_count == 0

        # Consume first result
        first_result = next(results_iter)
        assert call_count == 1
        assert first_result == 3

        # Consume remaining results
        remaining_results = list(results_iter)
        assert call_count == 3
        assert remaining_results == [7, 11]


class TestIntegration:
    """Test integration between batching functions."""

    def test_batch_data_and_process_in_batches(self):
        """Test integration between batch_data and process_in_batches."""
        data = list(range(10))
        batch_size = 3

        # First create batches
        batches = list(batch_data(data, batch_size))

        # Then process those batches
        def sum_processor(batch: List[int]) -> int:
            return sum(batch)

        results = [sum_processor(batch) for batch in batches]

        # Should be same as using process_in_batches directly
        direct_results = list(process_in_batches(data, sum_processor, batch_size))

        assert results == direct_results

    def test_chained_batch_processing(self):
        """Test chaining multiple batch processing operations."""
        data = list(range(12))

        # First level: group into batches of 3 and sum
        def first_level_processor(batch: List[int]) -> int:
            return sum(batch)

        first_results = list(process_in_batches(data, first_level_processor, 3))
        assert first_results == [3, 12, 21, 30]  # (0+1+2), (3+4+5), etc.

        # Second level: group those results into batches of 2 and sum
        def second_level_processor(batch: List[int]) -> int:
            return sum(batch)

        final_results = list(process_in_batches(first_results, second_level_processor, 2))
        assert final_results == [15, 51]  # (3+12), (21+30)

    def test_memory_efficiency_with_large_data(self):
        """Test that batching is memory efficient with large data."""

        # Create a large generator
        def large_data_generator():
            for i in range(10000):
                yield i

        def simple_processor(batch: List[int]) -> int:
            return len(batch)

        # Process without loading all data into memory
        results = list(process_in_batches(large_data_generator(), simple_processor, 1000))

        assert len(results) == 10
        assert all(result == 1000 for result in results)  # All batches including last
