"""Tests for the pyrator MCP (Model Context Protocol) calculation server."""

from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
import pytest

from pyrator.mcp import handle_mcp_message


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


def test_mcp_initialize() -> None:
    req = {
        "jsonrpc": "2.0",
        "id": 1,
        "method": "initialize",
        "params": {"protocolVersion": "2024-11-05"},
    }
    resp = handle_mcp_message(req)
    assert resp is not None
    assert resp["id"] == 1
    assert "result" in resp
    assert resp["result"]["protocolVersion"] == "2024-11-05"
    assert "tools" in resp["result"]["capabilities"]
    assert resp["result"]["serverInfo"]["name"] == "pyrator-mcp"


def test_mcp_tools_list() -> None:
    req = {
        "jsonrpc": "2.0",
        "id": 2,
        "method": "tools/list",
    }
    resp = handle_mcp_message(req)
    assert resp is not None
    assert resp["id"] == 2
    tools = resp["result"]["tools"]
    tool_names = [t["name"] for t in tools]
    assert "calculate_agreement" in tool_names
    assert "calculate_consensus" in tool_names
    assert "calculate_drift" in tool_names
    assert "query_ontology_distance" in tool_names


def test_mcp_call_calculate_agreement(annotations_csv: Path) -> None:
    req = {
        "jsonrpc": "2.0",
        "id": 3,
        "method": "tools/call",
        "params": {
            "name": "calculate_agreement",
            "arguments": {"data_path": str(annotations_csv)},
        },
    }
    resp = handle_mcp_message(req)
    assert resp is not None
    assert resp["id"] == 3
    assert resp["result"]["isError"] is False
    content_text = resp["result"]["content"][0]["text"]
    data = json.loads(content_text)
    assert "krippendorff_alpha" in data
    assert "fleiss_kappa" in data
    assert data["n_items"] == 2


def test_mcp_call_calculate_consensus(annotations_csv: Path) -> None:
    req = {
        "jsonrpc": "2.0",
        "id": 4,
        "method": "tools/call",
        "params": {
            "name": "calculate_consensus",
            "arguments": {
                "data_path": str(annotations_csv),
                "method": "dawid_skene",
            },
        },
    }
    resp = handle_mcp_message(req)
    assert resp is not None
    assert resp["id"] == 4
    assert resp["result"]["isError"] is False
    content_text = resp["result"]["content"][0]["text"]
    data = json.loads(content_text)
    assert data["consensus_labels"] == ["cat", "dog"]


def test_mcp_ping() -> None:
    req = {
        "jsonrpc": "2.0",
        "id": 5,
        "method": "ping",
    }
    resp = handle_mcp_message(req)
    assert resp is not None
    assert resp["id"] == 5
    assert resp["result"] == {}


def test_mcp_call_calculate_drift(tmp_path: Path) -> None:
    file_path = tmp_path / "drift.csv"
    df = pd.DataFrame(
        {
            "score": [1.0, 1.2, 1.1, 5.0, 5.1, 5.2],
            "window_id": ["base", "base", "base", "curr", "curr", "curr"],
        }
    )
    df.to_csv(file_path, index=False)

    req = {
        "jsonrpc": "2.0",
        "id": 6,
        "method": "tools/call",
        "params": {
            "name": "calculate_drift",
            "arguments": {
                "data_path": str(file_path),
                "col": "score",
                "baseline": "base",
                "target": "curr",
                "metric": "wasserstein",
            },
        },
    }
    resp = handle_mcp_message(req)
    assert resp is not None
    assert resp["id"] == 6
    assert resp["result"]["isError"] is False
    data = json.loads(resp["result"]["content"][0]["text"])
    assert data["metric"] == "wasserstein"
    assert data["value"] > 3.0


def test_mcp_call_query_ontology_distance(tmp_path: Path) -> None:
    onto_path = tmp_path / "onto.json"
    onto_data = {
        "version": "1.0",
        "nodes": {
            "ROOT": {"depth": 0},
            "animal": {"depth": 1},
            "cat": {"depth": 2},
            "dog": {"depth": 2},
        },
        "edges": [
            ["ROOT", "animal"],
            ["animal", "cat"],
            ["animal", "dog"],
        ],
    }
    onto_path.write_text(json.dumps(onto_data))

    req = {
        "jsonrpc": "2.0",
        "id": 7,
        "method": "tools/call",
        "params": {
            "name": "query_ontology_distance",
            "arguments": {
                "ontology_path": str(onto_path),
                "source_node": "cat",
                "target_node": "dog",
                "metric": "path",
            },
        },
    }
    resp = handle_mcp_message(req)
    assert resp is not None
    assert resp["id"] == 7
    assert resp["result"]["isError"] is False
    data = json.loads(resp["result"]["content"][0]["text"])
    assert data["distance"] == 2.0  # cat -> animal -> dog


def test_mcp_call_unknown_tool() -> None:
    req = {
        "jsonrpc": "2.0",
        "id": 8,
        "method": "tools/call",
        "params": {"name": "non_existent_tool", "arguments": {}},
    }
    resp = handle_mcp_message(req)
    assert resp is not None
    assert resp["id"] == 8
    assert "error" in resp
    assert resp["error"]["code"] == -32601


def test_mcp_call_calculation_error() -> None:
    req = {
        "jsonrpc": "2.0",
        "id": 9,
        "method": "tools/call",
        "params": {
            "name": "calculate_agreement",
            "arguments": {"data_path": "non_existent_file.csv"},
        },
    }
    resp = handle_mcp_message(req)
    assert resp is not None
    assert resp["id"] == 9
    assert resp["result"]["isError"] is True
    assert "Calculation error" in resp["result"]["content"][0]["text"]


def test_mcp_invalid_request_not_dict() -> None:
    resp = handle_mcp_message("not a dict")
    assert resp is not None
    assert "error" in resp
    assert resp["error"]["code"] == -32600
