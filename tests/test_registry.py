"""Tests for backend registry selection and dependency handling."""

from __future__ import annotations

from collections.abc import Iterator
from pathlib import Path
from typing import Any

import pytest

from pyrator.data.backends.base import BaseBackend
from pyrator.data.registry import BackendRegistry
from pyrator.types import FrameLike


@pytest.fixture()
def isolated_registry() -> Iterator[None]:
    """Keep backend registry mutations scoped to each test."""
    original_backends = BackendRegistry._backends.copy()
    original_priority = BackendRegistry._priority.copy()
    original_capabilities = BackendRegistry._capabilities.copy()
    try:
        BackendRegistry.clear()
        yield
    finally:
        BackendRegistry.clear()
        BackendRegistry._backends.update(original_backends)
        BackendRegistry._priority.extend(original_priority)
        BackendRegistry._capabilities.update(original_capabilities)


class ImportErrorBackend:
    """Backend stub that crashes during construction."""

    def __init__(self):
        raise ImportError("optional dependency missing")

    @property
    def name(self) -> str:
        return "import_error"

    def is_available(self) -> bool:
        return False

    def capabilities(self) -> set[str]:
        return {"csv"}

    def load_csv(self, path: Path, sep: str = ",", **kwargs: Any) -> FrameLike:
        raise NotImplementedError

    def load_jsonl(self, path: Path, **kwargs: Any) -> FrameLike:
        raise NotImplementedError

    def load_parquet(self, path: Path, **kwargs: Any) -> FrameLike:
        raise NotImplementedError

    def scan_csv(
        self, path: Path, chunk_size: int, sep: str = ",", **kwargs: Any
    ) -> Iterator[FrameLike]:
        raise NotImplementedError

    def scan_jsonl(self, path: Path, chunk_size: int, **kwargs: Any) -> Iterator[FrameLike]:
        raise NotImplementedError

    def scan_parquet(self, path: Path, chunk_size: int, **kwargs: Any) -> Iterator[FrameLike]:
        raise NotImplementedError


class AvailableBackend:
    """Backend stub that is always available."""

    @property
    def name(self) -> str:
        return "available"

    def is_available(self) -> bool:
        return True

    def capabilities(self) -> set[str]:
        return {"csv"}

    def load_csv(self, path: Path, sep: str = ",", **kwargs: Any) -> FrameLike:
        raise NotImplementedError

    def load_jsonl(self, path: Path, **kwargs: Any) -> FrameLike:
        raise NotImplementedError

    def load_parquet(self, path: Path, **kwargs: Any) -> FrameLike:
        raise NotImplementedError

    def scan_csv(
        self, path: Path, chunk_size: int, sep: str = ",", **kwargs: Any
    ) -> Iterator[FrameLike]:
        raise NotImplementedError

    def scan_jsonl(self, path: Path, chunk_size: int, **kwargs: Any) -> Iterator[FrameLike]:
        raise NotImplementedError

    def scan_parquet(self, path: Path, chunk_size: int, **kwargs: Any) -> Iterator[FrameLike]:
        raise NotImplementedError


class MissingOptionalDependencyBackend(BaseBackend):
    """Concrete backend used to test BaseBackend import handling."""

    @property
    def name(self) -> str:
        return "missing_optional"

    def _create_backend(self) -> Any:
        raise ImportError("missing optional dependency")

    def load_csv(self, path: Path, sep: str = ",", **kwargs: Any) -> FrameLike:
        raise NotImplementedError

    def load_jsonl(self, path: Path, **kwargs: Any) -> FrameLike:
        raise NotImplementedError

    def load_parquet(self, path: Path, **kwargs: Any) -> FrameLike:
        raise NotImplementedError

    def scan_csv(
        self, path: Path, chunk_size: int, sep: str = ",", **kwargs: Any
    ) -> Iterator[FrameLike]:
        raise NotImplementedError

    def scan_jsonl(self, path: Path, chunk_size: int, **kwargs: Any) -> Iterator[FrameLike]:
        raise NotImplementedError

    def scan_parquet(self, path: Path, chunk_size: int, **kwargs: Any) -> Iterator[FrameLike]:
        raise NotImplementedError


class StreamingCsvBackend(AvailableBackend):
    """Backend stub that supports both csv and streaming."""

    @property
    def name(self) -> str:
        return "streaming_csv"

    def capabilities(self) -> set[str]:
        return {"csv", "streaming"}


class UnavailableBackend(AvailableBackend):
    """Backend stub that exists but reports unavailable."""

    @property
    def name(self) -> str:
        return "unavailable"

    def is_available(self) -> bool:
        return False


class TestBackendRegistryDependencyHandling:
    """Regression tests for optional dependency behavior in registry selection."""

    def test_auto_select_skips_backend_that_raises_import_error(
        self, isolated_registry: None
    ) -> None:
        BackendRegistry.register("broken", priority=0)(ImportErrorBackend)
        BackendRegistry.register("ok", priority=1)(AvailableBackend)

        selected = BackendRegistry.auto_select()
        assert selected.name == "available"

    def test_get_backend_wraps_import_error_as_runtime_error(self, isolated_registry: None) -> None:
        BackendRegistry.register("broken")(ImportErrorBackend)

        with pytest.raises(RuntimeError, match="dependencies are missing"):
            BackendRegistry.get_backend("broken")

    def test_select_backend_by_multiple_capabilities(self, isolated_registry: None) -> None:
        BackendRegistry.register("csv_only", priority=0)(AvailableBackend)
        BackendRegistry.register("csv_streaming", priority=1)(StreamingCsvBackend)

        selected = BackendRegistry.get_best_backend_with_capabilities({"csv", "streaming"})
        assert selected.name == "streaming_csv"


class TestBaseBackendDependencyHandling:
    """Tests for BaseBackend availability state when imports are missing."""

    def test_base_backend_marks_missing_dependency_as_unavailable(self) -> None:
        backend = MissingOptionalDependencyBackend()
        assert backend.is_available() is False


class TestBackendRegistryAdditionalCoverage:
    """Additional coverage for registry utility methods and error branches."""

    def test_get_backend_unknown_name(self, isolated_registry: None) -> None:
        with pytest.raises(ValueError, match="Unknown backend"):
            BackendRegistry.get_backend("does_not_exist")

    def test_get_backend_reports_unavailable_backend(self, isolated_registry: None) -> None:
        BackendRegistry.register("down")(UnavailableBackend)
        with pytest.raises(RuntimeError, match="dependencies are missing"):
            BackendRegistry.get_backend("down")

    def test_auto_select_raises_when_empty(self, isolated_registry: None) -> None:
        with pytest.raises(RuntimeError, match="No data backends found"):
            BackendRegistry.auto_select()

    def test_get_best_backend_with_capability_raises_when_missing(
        self, isolated_registry: None
    ) -> None:
        BackendRegistry.register("csv_only")(AvailableBackend)
        with pytest.raises(RuntimeError, match="No backend available with capability 'jsonl'"):
            BackendRegistry.get_best_backend_with_capability("jsonl")

    def test_get_backends_with_capability_and_capability_registration(
        self, isolated_registry: None
    ) -> None:
        BackendRegistry.register("available_a")(AvailableBackend)
        BackendRegistry.register("available_b")(UnavailableBackend)

        BackendRegistry.register_capabilities("available_a", {"csv"})
        BackendRegistry.register_capabilities("available_b", {"csv"})

        result = BackendRegistry.get_backends_with_capability("csv")
        assert result == ["available_a"]
        assert set(BackendRegistry.available_backends()) == {"available_a", "available_b"}

    def test_get_best_backend_with_capabilities_raises_when_missing(
        self, isolated_registry: None
    ) -> None:
        BackendRegistry.register("csv_only")(AvailableBackend)
        with pytest.raises(RuntimeError, match="No backend available with capabilities"):
            BackendRegistry.get_best_backend_with_capabilities({"csv", "streaming"})
