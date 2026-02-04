from typing import Any, Callable, Iterable, List

def batch_data(data: Iterable[Any], batch_size: int) -> Iterable[List[Any]]:
    """Batch data into chunks of a specified size."""
    batch = []
    for item in data:
        batch.append(item)
        if len(batch) == batch_size:
            yield batch
            batch = []
    if batch:
        yield batch

def process_in_batches(
    data: Iterable[Any],
    processor: Callable[[List[Any]], Any],
    batch_size: int,
) -> Iterable[Any]:
    """Process data in batches using the provided processor function."""
    for batch in batch_data(data, batch_size):
        yield processor(batch)
