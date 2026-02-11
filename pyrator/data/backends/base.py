"""Abstract base class for data loading backends.

This module provides the base implementation that concrete backends should inherit from,
ensuring consistent behavior and interface compliance.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any, Set

from pyrator.data.spi import DataBackend


class BaseBackend(DataBackend, ABC):
    """Abstract base class for data backends."""

    def __init__(self):
        """Initialize the backend, creating the actual backend instance."""
        self._backend = self._create_backend()

    @abstractmethod
    def _create_backend(self):
        """Create and return the actual backend library instance.

        This should handle imports and return the library object
        that will be used for actual operations.
        """
        pass

    def is_available(self) -> bool:
        """Check if backend dependencies are available.

        Default implementation tries to create backend and catches import errors.
        """
        try:
            self._backend
            return True
        except (ImportError, RuntimeError):
            return False

    def capabilities(self) -> Set[str]:
        """Return default capabilities for most backends."""
        return {"csv", "jsonl", "parquet"}
