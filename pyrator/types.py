"""Ibis-based unified DataFrame protocol.

This module provides a protocol that wraps Ibis table expressions,
eliminating the need for pandas/polars imports in the type system.
"""

from __future__ import annotations

from typing import Any, Protocol, runtime_checkable, Union

# Try to import ibis, but make it optional for type checking
try:
    import ibis
    from ibis.types import Expr as IbisExpr
    has_ibis = True
except ImportError:
    has_ibis = False
    IbisExpr = Any


@runtime_checkable
class DataFrameProtocol(Protocol):
    """Protocol that defines a unified DataFrame interface."""
    @property
    def _expr(self) -> Any: ...
    @property
    def _schema(self) -> Any: ...
    def to_pandas(self) -> Any: ...
    def to_polars(self) -> Any: ...
    def __len__(self) -> int: ...
    def columns(self) -> list[str]: ...
    # ... (rest of protocol methods same as before)


class IbisDataFrame:
    """Ibis-based implementation of DataFrameProtocol."""
    # ... (Implementation same as before)
    def __init__(self, expr: Any):
        if not has_ibis:
            raise RuntimeError("Ibis is required for IbisDataFrame")
        # ... rest of init


# Define FrameLike based on availability
if has_ibis:
    # Use IbisDataFrame as the primary type, but allow Protocol for flexibility
    FrameLike = Union[IbisDataFrame, DataFrameProtocol]
else:
    # Fallback to Any or the Protocol when Ibis is missing
    FrameLike = Union[DataFrameProtocol, Any]

# Basic Types
ArrayLike = Any
ArrayModule = Any
IntLike = Union[int, Any]  # simplified for brevity
RealLike = Union[float, Any]
