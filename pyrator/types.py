"""Type definitions and protocols for pyrator data structures.

This module provides unified type definitions that work across multiple
backends (pandas, polars, duckdb) with graceful fallbacks when dependencies
are missing.
"""

from __future__ import annotations

from typing import Protocol, runtime_checkable, Union
from typing_extensions import Any


# Import optional dependencies using the _try_import pattern
def _try_import(name: str) -> Any | None:
    """Attempts to import a library, returning None if it fails."""
    try:
        return __import__(name)
    except Exception:
        return None


_ibis = _try_import("ibis")
_pd = _try_import("pandas")
_pl = _try_import("polars")


# Boolean checks for dependency availability
def has_ibis() -> bool:
    return _ibis is not None


def has_pandas() -> bool:
    return _pd is not None


def has_polars() -> bool:
    return _pl is not None


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

    def __getitem__(self, key) -> Any: ...

    def columns(self) -> list[str]: ...

    def shape(self) -> tuple[int, ...]: ...

    def head(self, n: int = 5) -> Any: ...

    def tail(self, n: int = 5) -> Any: ...

    def filter(self, predicate) -> Any: ...

    def group_by(self, *args, **kwargs) -> Any: ...

    def sort_values(self, *args, **kwargs) -> Any: ...

    def concat(self, other: Any, **kwargs) -> Any: ...

    def assign(self, **kwargs) -> Any: ...

    def drop(self, *args, **kwargs) -> Any: ...

    def rename(self, *args, **kwargs) -> Any: ...

    def describe(self, **kwargs) -> Any: ...


class IbisDataFrame:
    """Ibis-based implementation of DataFrameProtocol."""

    def __init__(self, expr: Any):
        if not has_ibis():
            raise RuntimeError("Ibis is required for IbisDataFrame")

        # Handle different input types
        if isinstance(expr, DataFrameProtocol):
            self.__expr = expr._expr
        elif hasattr(expr, "execute"):
            # Ibis expression
            self.__expr = expr
        else:
            # Try to convert to Ibis expression
            try:
                self.__expr = _ibis.table(expr)
            except Exception as e:
                raise RuntimeError(f"Failed to create Ibis table: {e}")

        self.__cached_pandas = None
        self.__cached_polars = None
        self.__cached_len = None
        self.__schema = None

    @property
    def _expr(self) -> Any:
        return self.__expr

    @property
    def _schema(self) -> Any:
        if self.__schema is None:
            try:
                table = self.__expr.execute()
                self.__schema = getattr(table, "schema", lambda: None)()
                if self.__schema is None:
                    self.__schema = {}
            except Exception:
                self.__schema = {}
        return self.__schema

    def _get_table(self) -> Any:
        """Get the executed table."""
        return self.__expr.execute()

    def to_pandas(self) -> Any:
        if not has_ibis():
            raise RuntimeError("Ibis is required for IbisDataFrame")

        if self.__cached_pandas is not None:
            return self.__cached_pandas

        table = self._get_table()
        self.__cached_pandas = table.to_pandas()
        self.__cached_len = len(self.__cached_pandas)
        return self.__cached_pandas

    def to_polars(self) -> Any:
        if not has_ibis():
            raise RuntimeError("Ibis is required for IbisDataFrame")

        if self.__cached_polars is not None:
            return self.__cached_polars

        table = self._get_table()
        self.__cached_polars = table.to_polars()
        self.__cached_len = len(self.__cached_polars)
        return self.__cached_polars

    def __len__(self) -> int:
        if self.__cached_len is not None:
            return self.__cached_len

        # Get length by converting to pandas first
        pandas_df = self.to_pandas()
        return len(pandas_df)

    def __getitem__(self, key) -> Any:
        table = self._get_table()
        return table[key]

    def columns(self) -> list[str]:
        try:
            table = self._get_table()
            cols = getattr(table, "columns", None)
            if cols is not None:
                return list(cols)
            # Fallback: try to get columns from pandas conversion
            pandas_df = self.to_pandas()
            return list(pandas_df.columns)
        except Exception:
            return []

    def shape(self) -> tuple[int, ...]:
        try:
            pandas_df = self.to_pandas()
            return pandas_df.shape
        except Exception:
            return (0,)

    def head(self, n: int = 5) -> Any:
        pandas_df = self.to_pandas()
        return pandas_df.head(n)

    def tail(self, n: int = 5) -> Any:
        pandas_df = self.to_pandas()
        return pandas_df.tail(n)

    def filter(self, predicate, **kwargs) -> Any:
        # Convert to pandas, filter, then wrap back
        pandas_df = self.to_pandas()
        filtered = (
            pandas_df.query(predicate) if isinstance(predicate, str) else pandas_df[predicate]
        )
        return IbisDataFrame(_ibis.from_pandas(filtered))

    def group_by(self, *args, **kwargs) -> Any:
        # Return a wrapper that can handle grouping operations
        return IbisGroupByWrapper(self, *args, **kwargs)

    def sort_values(self, *args, **kwargs) -> Any:
        pandas_df = self.to_pandas()
        sorted_df = pandas_df.sort_values(*args, **kwargs)
        return IbisDataFrame(_ibis.from_pandas(sorted_df))

    def concat(self, other: Any, **kwargs) -> Any:
        if isinstance(other, IbisDataFrame):
            other_pandas = other.to_pandas()
        else:
            other_pandas = other

        this_pandas = self.to_pandas()
        combined = _pd.concat([this_pandas, other_pandas], **kwargs)
        return IbisDataFrame(_ibis.from_pandas(combined))

    def assign(self, **kwargs) -> Any:
        pandas_df = self.to_pandas()
        result = pandas_df.assign(**kwargs)
        return IbisDataFrame(_ibis.from_pandas(result))

    def drop(self, *args, **kwargs) -> Any:
        pandas_df = self.to_pandas()
        result = pandas_df.drop(*args, **kwargs)
        return IbisDataFrame(_ibis.from_pandas(result))

    def rename(self, *args, **kwargs) -> Any:
        pandas_df = self.to_pandas()
        result = pandas_df.rename(*args, **kwargs)
        return IbisDataFrame(_ibis.from_pandas(result))

    def describe(self, **kwargs) -> Any:
        pandas_df = self.to_pandas()
        result = pandas_df.describe(**kwargs)
        return IbisDataFrame(_ibis.from_pandas(result))


class IbisGroupByWrapper:
    """Wrapper for group by operations on IbisDataFrame."""

    def __init__(self, dataframe: IbisDataFrame, *args, **kwargs):
        self.dataframe = dataframe
        self.args = args
        self.kwargs = kwargs

    def aggregate(self, **kwargs) -> IbisDataFrame:
        pandas_df = self.dataframe.to_pandas()
        grouped = pandas_df.groupby(*self.args, **self.kwargs)
        result = grouped.agg(kwargs)
        return IbisDataFrame(_ibis.from_pandas(result))

    def assign(self, **kwargs) -> Any:
        self._ensure_table()
        result = self.__table.mutate(**kwargs)
        return IbisDataFrame(result)

    def drop(self, *args, **kwargs) -> Any:
        self._ensure_table()
        result = self.__table.drop(*args, **kwargs)
        return IbisDataFrame(result)

    def rename(self, *args, **kwargs) -> Any:
        self._ensure_table()
        result = self.__table.rename(*args, **kwargs)
        return IbisDataFrame(result)

    def describe(self, **kwargs) -> Any:
        self._ensure_table()
        result = self.__table.aggregate(**kwargs)
        return IbisDataFrame(result)

    def assign(self, **kwargs) -> Any:
        self._ensure_table()
        result = self._table.mutate(**kwargs)
        return IbisDataFrame(result)

    def drop(self, *args, **kwargs) -> Any:
        self._ensure_table()
        result = self._table.drop(*args, **kwargs)
        return IbisDataFrame(result)

    def rename(self, *args, **kwargs) -> Any:
        self._ensure_table()
        result = self._table.rename(*args, **kwargs)
        return IbisDataFrame(result)

    def describe(self, **kwargs) -> Any:
        self._ensure_table()
        result = self._table.aggregate(**kwargs)
        return IbisDataFrame(result)


# Type aliases with proper fallbacks
if has_ibis():
    FrameLike = Union[IbisDataFrame, DataFrameProtocol]
else:
    FrameLike = Union[DataFrameProtocol, Any]

ArrayLike = Any
ArrayModule = Any
IntLike = Union[int, Any]
RealLike = Union[float, Any]

# Public API
__all__ = [
    "DataFrameProtocol",
    "IbisDataFrame",
    "FrameLike",
    "ArrayLike",
    "ArrayModule",
    "IntLike",
    "RealLike",
    "has_ibis",
    "has_pandas",
    "has_polars",
]
