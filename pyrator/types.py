"""Shared typing protocols and aliases for pyrator."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any, Protocol, TypeAlias, TypeVar, runtime_checkable

if TYPE_CHECKING:
    # Kept in TYPE_CHECKING to avoid a runtime dependency on ibis.
    from ibis.expr.types import Expr as IbisExpr
else:
    IbisExpr = object


T = TypeVar("T")


def require_non_none(value: T | None, msg: str = "unexpected None") -> T:
    """Assert that a value is not None, raising ValueError if it is."""
    if value is None:
        raise ValueError(msg)
    return value


@runtime_checkable
class ArrayModule(Protocol):
    """Protocol for NumPy/CuPy-like array modules."""

    int32: Any
    float32: Any
    float64: Any

    def asarray(self, obj: Any, dtype: Any | None = None) -> Any: ...
    def empty(self, shape: Any, dtype: Any | None = None) -> Any: ...
    def zeros(self, shape: Any, dtype: Any | None = None) -> Any: ...
    def ones(self, shape: Any, dtype: Any | None = None) -> Any: ...


@runtime_checkable
class DataFrameProtocol(Protocol):
    """Protocol for frame-like objects used across the data layer."""

    def __len__(self) -> int: ...


class IbisDataFrame:
    """Lightweight ibis expression wrapper used for typing boundaries."""

    def __init__(self, expr: IbisExpr):
        self._expr = expr

    def to_pandas(self) -> Any:
        """Convert ibis expression to pandas DataFrame."""
        return self._expr.to_pandas()


FrameLike: TypeAlias = DataFrameProtocol | IbisDataFrame
ArrayLike: TypeAlias = Any
IntLike: TypeAlias = int
RealLike: TypeAlias = int | float
