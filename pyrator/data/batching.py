from typing import Any, Callable, Iterable, TypeVar

T = TypeVar('T')
U = TypeVar('U')

def batch_data(data: Iterable[T], batch_size: int) -> Iterable[list[T]]:
    """Batch data into chunks of a specified size."""
    batch: list[T] = []
    for item in data:
        batch.append(item)
        if len(batch) == batch_size:
            yield batch
            batch = []
    if batch:
        yield batch

def process_in_batches(
    data: Iterable[T],
    processor: Callable[[list[T]], U],
    batch_size: int,
) -> Iterable[U]:
    """Process data in batches using the provided processor function."""
    for batch in batch_data(data, batch_size):
        yield processor(batch)
