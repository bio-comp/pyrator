"""Model Context Protocol (MCP) stdio calculation server for pyrator.

Provides deterministic mathematical calculation tools to AI coding agents
(Google Antigravity, Claude Code, Codex), preventing speculative hallucinations
of inter-rater reliability, consensus classifications, and drift statistics.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any

from pyrator.consensus import DawidSkeneEstimator, TreeConsensusEstimator
from pyrator.data.pipeline import (
    MaterializationBoundary,
    normalize_to_narwhals,
)
from pyrator.drift import (
    cramers_v,
    jsd,
    mmd,
    psi,
    wasserstein_distance,
)
from pyrator.ira import (
    KrippendorffAlpha,
    SemanticDistanceFactory,
    cohen_kappa,
    fleiss_kappa,
    intraclass_correlation,
)
from pyrator.ontology import Ontology

PROTOCOL_VERSION = "2024-11-05"
SERVER_INFO = {
    "name": "pyrator-mcp",
    "version": "0.1.0",
}

TOOLS = [
    {
        "name": "calculate_agreement",
        "description": (
            "Deterministically calculate inter-rater agreement and reliability metrics "
            "(Krippendorff's Alpha, Cohen's Kappa, Fleiss' Kappa, and ICC) on long-format "
            "annotation data from a CSV, Parquet, or Arrow file."
        ),
        "inputSchema": {
            "type": "object",
            "properties": {
                "data_path": {
                    "type": "string",
                    "description": "Path to input dataset file (CSV, Parquet, or Arrow).",
                },
                "item_col": {
                    "type": "string",
                    "default": "item_id",
                    "description": "Name of the item/subject ID column.",
                },
                "rater_col": {
                    "type": "string",
                    "default": "annotator_id",
                    "description": "Name of the rater/annotator ID column.",
                },
                "label_col": {
                    "type": "string",
                    "default": "label_id",
                    "description": "Name of the label or score column.",
                },
                "metric": {
                    "type": "string",
                    "enum": ["all", "krippendorff", "cohen", "fleiss", "icc"],
                    "default": "all",
                    "description": "Agreement metric to calculate.",
                },
                "ontology_path": {
                    "type": "string",
                    "description": "Optional path to ontology JSON/CSV for semantic metrics.",
                },
            },
            "required": ["data_path"],
        },
    },
    {
        "name": "calculate_consensus",
        "description": (
            "Deterministically resolve item consensus labels and class posterior probabilities "
            "using Dawid-Skene EM (with hierarchy decay priors) or Tree LCA/Medoid graph consensus."
        ),
        "inputSchema": {
            "type": "object",
            "properties": {
                "data_path": {
                    "type": "string",
                    "description": "Path to input dataset file (CSV, Parquet, or Arrow).",
                },
                "method": {
                    "type": "string",
                    "enum": ["dawid_skene", "dawid-skene", "tree"],
                    "default": "dawid_skene",
                    "description": "Consensus estimation algorithm.",
                },
                "strategy": {
                    "type": "string",
                    "enum": ["lca", "medoid", "majority"],
                    "default": "lca",
                    "description": "Ontology strategy if using tree consensus.",
                },
                "item_col": {
                    "type": "string",
                    "default": "item_id",
                    "description": "Name of the item ID column.",
                },
                "rater_col": {
                    "type": "string",
                    "default": "annotator_id",
                    "description": "Name of the rater ID column.",
                },
                "label_col": {
                    "type": "string",
                    "default": "label_id",
                    "description": "Name of the label column.",
                },
                "ontology_path": {
                    "type": "string",
                    "description": "Optional path to ontology JSON/CSV for hierarchy support.",
                },
            },
            "required": ["data_path"],
        },
    },
    {
        "name": "calculate_drift",
        "description": (
            "Deterministically calculate distribution drift (Wasserstein distance, PSI, "
            "Jensen-Shannon divergence, MMD, or Cramer's V) between baseline and target windows."
        ),
        "inputSchema": {
            "type": "object",
            "properties": {
                "data_path": {
                    "type": "string",
                    "description": "Path to input dataset file (CSV, Parquet, or Arrow).",
                },
                "col": {
                    "type": "string",
                    "description": "Column to evaluate for drift.",
                },
                "window_col": {
                    "type": "string",
                    "default": "window_id",
                    "description": "Column identifying batch or time window.",
                },
                "baseline": {
                    "type": "string",
                    "description": "Value of window_col for baseline data.",
                },
                "target": {
                    "type": "string",
                    "description": "Value of window_col for current/target data.",
                },
                "metric": {
                    "type": "string",
                    "enum": ["wasserstein", "psi", "jsd", "mmd", "cramer_v", "cramers_v"],
                    "default": "wasserstein",
                    "description": "Statistical drift metric.",
                },
            },
            "required": ["data_path", "col", "baseline", "target"],
        },
    },
    {
        "name": "query_ontology_distance",
        "description": "Compute the semantic graph distance between two labels in an ontology.",
        "inputSchema": {
            "type": "object",
            "properties": {
                "ontology_path": {
                    "type": "string",
                    "description": "Path to ontology JSON or CSV file.",
                },
                "source_node": {"type": "string", "description": "Source label ID."},
                "target_node": {"type": "string", "description": "Target label ID."},
                "metric": {
                    "type": "string",
                    "enum": ["path", "resnik", "lin", "wpath"],
                    "default": "path",
                    "description": "Ontology distance metric.",
                },
            },
            "required": ["ontology_path", "source_node", "target_node"],
        },
    },
]


def _load_ontology(path_str: str | None) -> Ontology | None:
    if not path_str:
        return None
    p = Path(path_str)
    if not p.exists():
        raise FileNotFoundError(f"Ontology file not found: {path_str}")
    if p.suffix == ".json":
        return Ontology.from_json(str(p))
    if p.suffix == ".csv":
        return Ontology.from_csv(str(p))
    raise ValueError("Ontology file must have .json or .csv extension")


def _execute_calculate_agreement(args: dict[str, Any]) -> dict[str, Any]:
    file_path = args["data_path"]
    item_col = args.get("item_col", "item_id")
    rater_col = args.get("rater_col", "annotator_id")
    label_col = args.get("label_col", "label_id")
    metric = args.get("metric", "all")
    onto = _load_ontology(args.get("ontology_path"))

    nw_df = normalize_to_narwhals(file_path)
    items_idx, raters_idx, labels_idx, id_maps = (
        MaterializationBoundary.extract_canonical_annotations(
            nw_df, item_col=item_col, rater_col=rater_col, label_col=label_col
        )
    )

    classes = [id_maps["label"]["to_str"][i] for i in range(len(id_maps["label"]["to_str"]))]
    out: dict[str, Any] = {
        "dataset": file_path,
        "n_items": len(id_maps["item"]["to_str"]),
        "n_raters": len(id_maps["annotator"]["to_str"]),
        "n_ratings": len(items_idx),
        "classes": classes,
    }

    if metric in ("all", "krippendorff"):
        ka = KrippendorffAlpha(nw_df, item_col=item_col, rater_col=rater_col, label_col=label_col)
        if onto is not None:
            dist_factory = SemanticDistanceFactory(onto)
            dist_mat = dist_factory.compute_distance_matrix(classes, metric="path")
            out["semantic_krippendorff_alpha"] = float(
                ka.calculate(metric="custom", distance_matrix=dist_mat)
            )
        else:
            out["krippendorff_alpha"] = float(ka.calculate(metric="nominal"))

    eager_df = MaterializationBoundary.to_eager(
        nw_df.select([item_col, rater_col, label_col]).drop_nulls()
    )

    if metric in ("all", "fleiss"):
        try:
            out["fleiss_kappa"] = float(
                fleiss_kappa(eager_df, item_col=item_col, rater_col=rater_col, label_col=label_col)
            )
        except Exception as e:
            out["fleiss_kappa_note"] = str(e)

    if metric in ("all", "cohen"):
        if out["n_raters"] == 2:
            try:
                out["cohen_kappa"] = float(
                    cohen_kappa(
                        eager_df,
                        item_col=item_col,
                        rater_col=rater_col,
                        label_col=label_col,
                    )
                )
            except Exception as e:
                out["cohen_kappa_note"] = str(e)
        else:
            out["cohen_kappa_note"] = f"Requires exactly 2 raters (found {out['n_raters']})"

    if metric in ("all", "icc"):
        try:
            out["icc2_1"] = float(
                intraclass_correlation(
                    eager_df,
                    item_col=item_col,
                    rater_col=rater_col,
                    score_col=label_col,
                    variant="ICC2_1",
                )
            )
        except Exception as e:
            out["icc_note"] = f"ICC not applicable: {e}"

    return out


def _execute_calculate_consensus(args: dict[str, Any]) -> dict[str, Any]:
    file_path = args["data_path"]
    method = str(args.get("method", "dawid_skene")).replace("-", "_")
    strategy = args.get("strategy", "lca")
    item_col = args.get("item_col", "item_id")
    rater_col = args.get("rater_col", "annotator_id")
    label_col = args.get("label_col", "label_id")
    onto = _load_ontology(args.get("ontology_path"))

    if method == "tree":
        if onto is None:
            # Fallback trivial ontology
            unique_labels = normalize_to_narwhals(file_path).select([label_col]).drop_nulls()
            eager_labels = MaterializationBoundary.to_eager(unique_labels)
            distinct = sorted(list(set(eager_labels[label_col].to_numpy().astype(str))))
            nodes = {c: {"depth": 1} for c in distinct}
            nodes["ROOT"] = {"depth": 0}
            edges = [("ROOT", c) for c in distinct]
            onto = Ontology.build("auto_tree", nodes=nodes, edges=edges)

        estimator = TreeConsensusEstimator(ontology=onto, strategy=strategy)
        res = estimator.fit_predict(
            file_path, item_col=item_col, rater_col=rater_col, label_col=label_col
        )
    else:
        ds_est = DawidSkeneEstimator(ontology=onto)
        res = ds_est.fit(file_path, item_col=item_col, rater_col=rater_col, label_col=label_col)

    return res.to_dict()


def _execute_calculate_drift(args: dict[str, Any]) -> dict[str, Any]:
    file_path = args["data_path"]
    col = args["col"]
    window_col = args.get("window_col", "window_id")
    baseline = str(args["baseline"])
    target = str(args["target"])
    metric = str(args.get("metric", "wasserstein")).replace("-", "_")

    nw_df = normalize_to_narwhals(file_path)

    cols = [c.strip() for c in col.split(",")]

    if metric == "wasserstein":
        df_res = wasserstein_distance(nw_df, col=col, window_col=window_col, baseline=baseline)
    elif metric == "psi":
        df_res = psi(nw_df, col=col, window_col=window_col, baseline=baseline)
    elif metric == "jsd":
        df_res = jsd(nw_df, dist_cols=cols, window_col=window_col, baseline=baseline)
    elif metric == "mmd":
        df_res = mmd(nw_df, emb_cols=cols, window_col=window_col, baseline=baseline)
    elif metric in ("cramer_v", "cramers_v"):
        df_res = cramers_v(nw_df, x=col, y=col, window_col=window_col, baseline=baseline)
    else:
        raise ValueError(f"Unknown drift metric: {metric}")

    import pandas as pd

    if isinstance(df_res, pd.DataFrame):
        target_df = df_res[df_res["window_id"] == target]
        if not target_df.empty:
            return {str(k): v for k, v in target_df.iloc[0].to_dict().items()}
        if not df_res.empty:
            return {str(k): v for k, v in df_res.iloc[0].to_dict().items()}
        return {"metric": metric, "value": 0.0}

    return {"metric": metric, "value": float(df_res)}


def _execute_query_ontology_distance(args: dict[str, Any]) -> dict[str, Any]:
    onto = _load_ontology(args["ontology_path"])
    if onto is None:
        raise ValueError("ontology_path is required")
    u = str(args["source_node"])
    v = str(args["target_node"])
    m = args.get("metric", "path")
    dist = onto.get_distance(u, v, metric=m)
    return {
        "source": u,
        "target": v,
        "metric": m,
        "distance": float(dist),
    }


def handle_mcp_message(msg: Any) -> dict[str, Any] | None:  # noqa: C901
    """Process a single JSON-RPC MCP message and return the response."""
    if not isinstance(msg, dict):
        return {
            "jsonrpc": "2.0",
            "id": None,
            "error": {"code": -32600, "message": "Invalid Request: Expected JSON object"},
        }

    method = msg.get("method")
    msg_id = msg.get("id")

    if method == "initialize":
        return {
            "jsonrpc": "2.0",
            "id": msg_id,
            "result": {
                "protocolVersion": PROTOCOL_VERSION,
                "capabilities": {"tools": {}},
                "serverInfo": SERVER_INFO,
            },
        }

    if method == "notifications/initialized":
        return None

    if method == "ping":
        return {"jsonrpc": "2.0", "id": msg_id, "result": {}}

    if method == "tools/list":
        return {
            "jsonrpc": "2.0",
            "id": msg_id,
            "result": {"tools": TOOLS},
        }

    if method == "tools/call":
        params = msg.get("params", {})
        tool_name = params.get("name")
        tool_args = params.get("arguments", {})

        try:
            if tool_name == "calculate_agreement":
                result_data = _execute_calculate_agreement(tool_args)
            elif tool_name == "calculate_consensus":
                result_data = _execute_calculate_consensus(tool_args)
            elif tool_name == "calculate_drift":
                result_data = _execute_calculate_drift(tool_args)
            elif tool_name == "query_ontology_distance":
                result_data = _execute_query_ontology_distance(tool_args)
            else:
                return {
                    "jsonrpc": "2.0",
                    "id": msg_id,
                    "error": {"code": -32601, "message": f"Unknown tool: {tool_name}"},
                }

            return {
                "jsonrpc": "2.0",
                "id": msg_id,
                "result": {
                    "content": [{"type": "text", "text": json.dumps(result_data, indent=2)}],
                    "isError": False,
                },
            }
        except Exception as e:
            return {
                "jsonrpc": "2.0",
                "id": msg_id,
                "result": {
                    "content": [{"type": "text", "text": f"Calculation error: {e}"}],
                    "isError": True,
                },
            }

    # Unknown method
    if msg_id is not None:
        return {
            "jsonrpc": "2.0",
            "id": msg_id,
            "error": {"code": -32601, "message": f"Method not found: {method}"},
        }
    return None


def run_mcp_stdio_server() -> None:
    """Run MCP server over standard input and output."""
    for line in sys.stdin:
        line = line.strip()
        if not line:
            continue
        try:
            req = json.loads(line)
            resp = handle_mcp_message(req)
            if resp is not None:
                sys.stdout.write(json.dumps(resp) + "\n")
                sys.stdout.flush()
        except json.JSONDecodeError:
            resp = {
                "jsonrpc": "2.0",
                "id": None,
                "error": {"code": -32700, "message": "Parse error: Invalid JSON"},
            }
            sys.stdout.write(json.dumps(resp) + "\n")
            sys.stdout.flush()


if __name__ == "__main__":
    run_mcp_stdio_server()
