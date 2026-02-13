"""Additional coverage for backend utility modules."""

from __future__ import annotations

import runpy
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import pytest

from pyrator.data.backends import backends as backend_utils


def _load_legacy_module_namespace() -> dict[str, Any]:
    module_path = Path(__file__).resolve().parents[1] / "pyrator" / "data" / "backends.py"
    return runpy.run_path(str(module_path))


def test_legacy_backends_module_basic_paths() -> None:
    """Exercise the legacy `pyrator/data/backends.py` compatibility module."""
    ns = _load_legacy_module_namespace()

    assert isinstance(ns["has_pandas"](), bool)
    assert isinstance(ns["has_polars"](), bool)
    assert isinstance(ns["has_duckdb"](), bool)
    assert ns["get_xp"]() is np

    ns["get_xp"].__globals__["_cupy"] = None
    with pytest.raises(RuntimeError, match="CuPy is not installed"):
        ns["get_xp"]("gpu")

    ns["to_pandas"].__globals__["_pd"] = None
    with pytest.raises(RuntimeError, match="pandas is required for this operation"):
        ns["to_pandas"]({"a": [1]})


def test_legacy_backends_module_polars_conversion() -> None:
    """Legacy compatibility module should support polars conversion when available."""
    pl = pytest.importorskip("polars")

    ns = _load_legacy_module_namespace()
    polars_df = ns["to_polars"]({"a": [1, 2]})
    assert isinstance(polars_df, pl.DataFrame)

    ns["to_polars"].__globals__["_pl"] = None
    with pytest.raises(RuntimeError, match="polars is required"):
        ns["to_polars"]({"a": [1]})


def test_backend_utils_error_paths_and_dataframe_checks(monkeypatch: pytest.MonkeyPatch) -> None:
    """Exercise rarely used branches in `pyrator.data.backends.backends`."""
    assert isinstance(backend_utils.has_ijson(), bool)

    frame = pd.DataFrame({"a": [1]})
    assert backend_utils.is_pandas_dataframe(frame) is True
    assert backend_utils.is_polars_dataframe(frame) is False

    monkeypatch.setattr(backend_utils, "_cupy", None)
    assert backend_utils.has_cupy() is False
    with pytest.raises(RuntimeError, match="CuPy is not installed"):
        backend_utils.get_xp(device="gpu")

    monkeypatch.setattr(backend_utils, "_pd", None)
    with pytest.raises(RuntimeError, match="pandas is required"):
        backend_utils.to_pandas({"a": [1]})


def test_backend_utils_polars_and_gpu_happy_paths(monkeypatch: pytest.MonkeyPatch) -> None:
    """Exercise conversion and GPU success branches in backend utils."""
    pl = pytest.importorskip("polars")
    fake_cupy = object()
    monkeypatch.setattr(backend_utils, "_cupy", fake_cupy)
    assert backend_utils.get_xp(device="gpu") is fake_cupy

    polars_df = pl.DataFrame({"a": [1, 2]})
    assert backend_utils.is_polars_dataframe(polars_df) is True
    assert isinstance(backend_utils.to_pandas(polars_df), pd.DataFrame)
    assert isinstance(backend_utils.to_polars({"a": [1, 2]}), pl.DataFrame)
