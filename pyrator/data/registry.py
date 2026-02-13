"""Backend registry for data loading system.

This module provides a centralized registry for data backends,
allowing for dynamic registration, auto-selection, and capability-based
routing of backend implementations.
"""

from __future__ import annotations

from typing import Dict, List, Optional, Set, Type

from pyrator.data.spi import DataBackend


class BackendRegistry:
    """Central registry for data loading backends."""

    _backends: Dict[str, Type[DataBackend]] = {}
    _priority: List[str] = ["polars", "pandas", "duckdb", "pyarrow"]
    _capabilities: Dict[str, Set[str]] = {}

    @classmethod
    def register(cls, name: str, priority: Optional[int] = None):
        """Decorator for registering backends.

        Args:
            name: Name of the backend
            priority: Optional priority in selection list (None = append)

        Returns:
            Decorator function for backend registration
        """

        def decorator(backend_cls: Type[DataBackend]):
            cls._backends[name] = backend_cls

            # Insert at correct priority position
            if priority is not None and name not in cls._priority:
                cls._priority.insert(priority, name)
            elif name not in cls._priority:
                cls._priority.append(name)

            return backend_cls

        return decorator

    @classmethod
    def get_backend(cls, name: str) -> DataBackend:
        """Get specific backend by name.

        Args:
            name: Name of the backend to retrieve

        Returns:
            Backend instance

        Raises:
            ValueError: If backend is not registered
            RuntimeError: If backend is registered but dependencies missing
        """
        if name not in cls._backends:
            raise ValueError(f"Unknown backend: {name}")

        try:
            backend = cls._backends[name]()
        except ImportError as exc:
            raise RuntimeError(
                f"Backend '{name}' is registered but dependencies are missing."
            ) from exc
        if not backend.is_available():
            raise RuntimeError(f"Backend '{name}' is registered but dependencies are missing.")
        return backend

    @classmethod
    def auto_select(cls) -> DataBackend:
        """Auto-select best available backend in priority order.

        Returns:
            First available backend in priority order

        Raises:
            RuntimeError: If no backends are available
        """
        for name in cls._priority:
            if name in cls._backends:
                try:
                    backend = cls._backends[name]()
                except ImportError:
                    continue
                if backend.is_available():
                    return backend
        raise RuntimeError("No data backends found (install polars, pandas, or duckdb).")

    @classmethod
    def get_best_backend_with_capability(cls, capability: str) -> DataBackend:
        """Get best backend that supports a specific capability.

        Args:
            capability: Required capability (e.g., 'streaming', 'distributed')

        Returns:
            First available backend with the requested capability
        """
        for name in cls._priority:
            if name in cls._backends:
                try:
                    backend = cls._backends[name]()
                except ImportError:
                    continue
                if backend.is_available() and capability in backend.capabilities():
                    return backend
        raise RuntimeError(f"No backend available with capability '{capability}'.")

    @classmethod
    def get_best_backend_with_capabilities(cls, required: set[str]) -> DataBackend:
        """Get best backend that supports all required capabilities."""
        for name in cls._priority:
            if name in cls._backends:
                try:
                    backend = cls._backends[name]()
                except ImportError:
                    continue
                if backend.is_available() and required.issubset(backend.capabilities()):
                    return backend
        joined = ", ".join(sorted(required))
        raise RuntimeError(f"No backend available with capabilities: {joined}.")

    @classmethod
    def get_backends_with_capability(cls, capability: str) -> List[str]:
        """Get list of backends that support specific capability.

        Args:
            capability: Required capability

        Returns:
            List of backend names that support the capability
        """
        available = []
        for name in cls._priority:
            if name in cls._backends and capability in cls._capabilities.get(name, set()):
                try:
                    backend = cls._backends[name]()
                except ImportError:
                    continue
                if backend.is_available():
                    available.append(name)
        return available

    @classmethod
    def available_backends(cls) -> List[str]:
        """List all registered backends.

        Returns:
            List of all registered backend names
        """
        return list(cls._backends.keys())

    @classmethod
    def register_capabilities(cls, name: str, capabilities: Set[str]):
        """Register capabilities for a backend."""
        cls._capabilities[name] = capabilities

    @classmethod
    def clear(cls):
        """Clear all registered backends (for testing)."""
        cls._backends.clear()
        cls._priority.clear()
        cls._capabilities.clear()
