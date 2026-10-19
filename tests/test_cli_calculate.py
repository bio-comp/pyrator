"""Tests for the pyrator calculate CLI commands."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from typer.testing import CliRunner

from pyrator.cli import app

runner = CliRunner()


@pytest.fixture
def annotations_csv(tmp_path: Path) -> Path:
    file_path = tmp_path / "annotations.csv"
    df = pd.DataFrame(
        {
            "item_id": ["i1", "i1", "i1", "i2", "i2", "i2"],
            "annotator_id": ["r1", "r2", "r3", "r1", "r2", "r3"],
            "label_id": ["cat", "cat", "cat", "dog", "dog", "cat"],
        }
    )
    df.to_csv(file_path, index=False)
    return file_path


@pytest.fixture
def drift_csv(tmp_path: Path) -> Path:
    file_path = tmp_path / "drift_data.csv"
    np.random.seed(42)
    base_vals = np.random.normal(0.0, 1.0, 100)
    target_vals = np.random.normal(1.5, 1.0, 100)
    df = pd.DataFrame(
        {
            "score": np.concatenate([base_vals, target_vals]),
            "window_id": ["base"] * 100 + ["curr"] * 100,
        }
    )
    df.to_csv(file_path, index=False)
    return file_path


def test_cli_calculate_agreement_json(annotations_csv: Path) -> None:
    result = runner.invoke(
        app,
        [
            "calculate",
            "agreement",
            str(annotations_csv),
            "--format",
            "json",
        ],
    )
    assert result.exit_code == 0, result.output
    data = json.loads(result.output)
    assert "krippendorff_alpha" in data
    assert "fleiss_kappa" in data
    assert isinstance(data["krippendorff_alpha"], float)
    assert data["n_items"] == 2
    assert data["n_raters"] == 3


def test_cli_calculate_consensus_dawid_skene(annotations_csv: Path) -> None:
    result = runner.invoke(
        app,
        [
            "calculate",
            "consensus",
            str(annotations_csv),
            "--method",
            "dawid_skene",
            "--format",
            "json",
        ],
    )
    assert result.exit_code == 0, result.output
    data = json.loads(result.output)
    assert "item_ids" in data
    assert "consensus_labels" in data
    assert data["item_ids"] == ["i1", "i2"]
    assert data["consensus_labels"] == ["cat", "dog"]


def test_cli_calculate_consensus_tree_majority(annotations_csv: Path) -> None:
    result = runner.invoke(
        app,
        [
            "calculate",
            "consensus",
            str(annotations_csv),
            "--method",
            "tree",
            "--strategy",
            "majority",
            "--format",
            "json",
        ],
    )
    assert result.exit_code == 0, result.output
    data = json.loads(result.output)
    assert data["consensus_labels"] == ["cat", "dog"]


def test_cli_calculate_drift_wasserstein(drift_csv: Path) -> None:
    result = runner.invoke(
        app,
        [
            "calculate",
            "drift",
            str(drift_csv),
            "--col",
            "score",
            "--window-col",
            "window_id",
            "--baseline",
            "base",
            "--target",
            "curr",
            "--metric",
            "wasserstein",
            "--format",
            "json",
        ],
    )
    assert result.exit_code == 0, result.output
    data = json.loads(result.output)
    assert data["metric"] == "wasserstein"
    assert data["value"] > 1.0  # Means differ by ~1.5


def test_cli_calculate_consensus_hyphen_dawid_skene(annotations_csv: Path) -> None:
    result = runner.invoke(
        app,
        [
            "calculate",
            "consensus",
            str(annotations_csv),
            "--method",
            "dawid-skene",
            "--format",
            "json",
        ],
    )
    assert result.exit_code == 0, result.output
    data = json.loads(result.output)
    assert data["consensus_labels"] == ["cat", "dog"]


def test_cli_calculate_agreement_table_and_output(annotations_csv: Path, tmp_path: Path) -> None:
    out_file = tmp_path / "agreement_table.txt"
    result = runner.invoke(
        app,
        [
            "calculate",
            "agreement",
            str(annotations_csv),
            "--format",
            "table",
            "--output",
            str(out_file),
        ],
    )
    assert result.exit_code == 0, result.output
    assert out_file.exists()
    content = out_file.read_text()
    assert "Field | Value" in content
    assert "krippendorff_alpha" in content


def test_cli_calculate_agreement_cohen_two_raters(tmp_path: Path) -> None:
    file_path = tmp_path / "two_raters.csv"
    df = pd.DataFrame(
        {
            "item_id": ["i1", "i2", "i3", "i1", "i2", "i3"],
            "annotator_id": ["r1", "r1", "r1", "r2", "r2", "r2"],
            "label_id": ["cat", "dog", "bird", "cat", "dog", "dog"],
        }
    )
    df.to_csv(file_path, index=False)
    result = runner.invoke(
        app,
        ["calculate", "agreement", str(file_path), "--metric", "cohen", "--format", "json"],
    )
    assert result.exit_code == 0, result.output
    data = json.loads(result.output)
    assert "cohen_kappa" in data
    assert isinstance(data["cohen_kappa"], float)


def test_cli_calculate_drift_cramers_v(tmp_path: Path) -> None:
    file_path = tmp_path / "cramers_data.csv"
    df = pd.DataFrame(
        {
            "cat": ["A", "A", "B", "B", "A", "B", "B", "B"],
            "window_id": ["w1", "w1", "w1", "w1", "w2", "w2", "w2", "w2"],
        }
    )
    df.to_csv(file_path, index=False)
    result = runner.invoke(
        app,
        [
            "calculate",
            "drift",
            str(file_path),
            "--col",
            "cat",
            "--window-col",
            "window_id",
            "--baseline",
            "w1",
            "--target",
            "w2",
            "--metric",
            "cramers_v",
            "--format",
            "json",
        ],
    )
    assert result.exit_code == 0, result.output
    data = json.loads(result.output)
    assert data["metric"] == "cramer_v"
    assert "value" in data
