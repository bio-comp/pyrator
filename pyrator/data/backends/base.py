"""Abstract base class for data loading backends.

This module provides the base implementation that concrete backends should inherit from,
ensuring consistent behavior and interface compliance.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from importlib.util import find_spec
from typing import Any

from pyrator.data.spi import DataBackend


class BaseBackend(DataBackend, ABC):
    """Abstract base class for data backends."""

    backend_module: str | None = None

    def __init__(self) -> None:
        """Initialize backend state without crashing on missing optional deps."""
        self._backend: Any | None = None
        self._available = False
        try:
            self._backend = self._create_backend()
            self._available = True
        except ImportError:
            self._backend = None
            self._available = False

    @classmethod
    def is_available_class(cls) -> bool:
        """Cheap class-level dependency probe without backend instantiation."""
        if cls.backend_module is None:
            return True
        try:
            return find_spec(cls.backend_module) is not None
        except (ImportError, ValueError):
            return False

    @abstractmethod
    def _create_backend(self) -> Any:
        """Create and return the actual backend library instance.

        This should handle imports and return the library object
        that will be used for actual operations.
        """
        raise NotImplementedError

    def is_available(self) -> bool:
        """Check if backend dependencies are available.

        Returns:
            True when optional dependencies for this backend are installed.
        """
        return self._available

    def capabilities(self) -> set[str]:
        """Return default capabilities for most backends."""
        return {"csv", "jsonl", "parquet"}
