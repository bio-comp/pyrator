# pyrator/types.py
from __future__ import annotations

import numbers
from typing import Any, Protocol, runtime_checkable

# Numeric ABCs in stdlib (like scikit-learn):
IntLike = numbers.Integral  # Any integer-like type, e.g., int, numpy.int64.
RealLike = numbers.Real  # Any real number type, e.g., float, int, numpy.float64.


# Array module protocol (NumPy/CuPy-like)
@runtime_checkable
class ArrayModule(Protocol):
    """A protocol for array modules like NumPy or CuPy.

    This defines the minimal API surface that pyrator relies on to perform
    array operations, allowing the computational backend to be swappable.
    """

    # minimal surface we rely on
    def asarray(self, obj: Any, dtype: Any | None = None) -> Any: ...
    def empty(self, shape: Any, dtype: Any | None = None) -> Any: ...
    def zeros(self, shape: Any, dtype: Any | None = None) -> Any: ...
    def ones(self, shape: Any, dtype: Any | None = None) -> Any: ...

    int32: Any
    float32: Any
    float64: Any


# Type alias for array-like objects. Opaque to avoid a hard numpy dependency.
ArrayLike = Any

# Type alias for DataFrame-like objects. Opaque to avoid pandas/polars deps.
FrameLike = Any
