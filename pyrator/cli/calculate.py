"""Calculation CLI subcommands for deterministic reliability, consensus, and drift analysis."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Literal, Optional

import typer

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

calculate_app = typer.Typer(
    help="Deterministic calculation commands for agreement, consensus, and drift."
)


def _load_optional_ontology(ontology_path: str | None) -> Ontology | None:
    if not ontology_path:
        return None
    path = Path(ontology_path)
    if not path.exists():
        raise typer.BadParameter(f"Ontology file does not exist: {ontology_path}")
    if path.suffix == ".json":
        return Ontology.from_json(str(path))
    if path.suffix == ".csv":
        return Ontology.from_csv(str(path))
    raise typer.BadParameter("Ontology file must be .json or .csv")


@calculate_app.command("agreement")
def calculate_agreement(  # noqa: C901
    file_path: str = typer.Argument(..., help="Path to input dataset (CSV, Parquet, Arrow)"),
    item_col: str = typer.Option("item_id", "--item-col", help="Item / subject column"),
    rater_col: str = typer.Option("annotator_id", "--rater-col", help="Rater column"),
    label_col: str = typer.Option("label_id", "--label-col", help="Label or score column"),
    metric: str = typer.Option(
        "all",
        "--metric",
        "-m",
        help="Metric to calculate: all, krippendorff, cohen, fleiss, icc",
    ),
    ontology: Optional[str] = typer.Option(None, "--ontology", help="Path to ontology file"),
    format: str = typer.Option("json", "--format", "-f", help="Output format: json or table"),
    output: Optional[str] = typer.Option(None, "--output", "-o", help="File to write results to"),
) -> None:
    """Calculate deterministic inter-rater agreement metrics on long-format annotations."""
    try:
        nw_df = normalize_to_narwhals(file_path)
        onto = _load_optional_ontology(ontology)

        results: dict[str, Any] = {
            "dataset": file_path,
        }

        # Materialization boundary: extract canonical indices and mappings
        items_idx, raters_idx, labels_idx, id_maps = (
            MaterializationBoundary.extract_canonical_annotations(
                nw_df, item_col=item_col, rater_col=rater_col, label_col=label_col
            )
        )
        results["n_items"] = len(id_maps["item"]["to_str"])
        results["n_raters"] = len(id_maps["annotator"]["to_str"])
        results["n_ratings"] = len(items_idx)
        labels_to_str = id_maps["label"]["to_str"]
        results["classes"] = [labels_to_str[i] for i in range(len(labels_to_str))]

        # Krippendorff Alpha
        if metric in ("all", "krippendorff"):
            ka = KrippendorffAlpha(
                nw_df, item_col=item_col, rater_col=rater_col, label_col=label_col
            )
            if onto is not None:
                dist_factory = SemanticDistanceFactory(onto)
                dist_mat = dist_factory.compute_distance_matrix(results["classes"], metric="path")
                alpha_val = ka.calculate(metric="custom", distance_matrix=dist_mat)
                results["semantic_krippendorff_alpha"] = float(alpha_val)
            else:
                alpha_val = ka.calculate(metric="nominal")
                results["krippendorff_alpha"] = float(alpha_val)

        # Fleiss Kappa
        if metric in ("all", "fleiss"):
            try:
                # Use eager frame for pandas-based classical kappa
                eager_df = MaterializationBoundary.to_eager(
                    nw_df.select([item_col, rater_col, label_col]).drop_nulls()
                )
                fk = fleiss_kappa(
                    eager_df,
                    item_col=item_col,
                    rater_col=rater_col,
                    label_col=label_col,
                )
                results["fleiss_kappa"] = float(fk)
            except Exception as e:
                results["fleiss_kappa_note"] = str(e)

        # Cohen Kappa (requires exactly 2 raters)
        if metric in ("all", "cohen"):
            if results["n_raters"] == 2:
                try:
                    eager_df = MaterializationBoundary.to_eager(
                        nw_df.select([item_col, rater_col, label_col]).drop_nulls()
                    )
                    ck = cohen_kappa(
                        eager_df,
                        item_col=item_col,
                        rater_col=rater_col,
                        label_col=label_col,
                    )
                    results["cohen_kappa"] = float(ck)
                except Exception as e:
                    results["cohen_kappa_note"] = str(e)
            else:
                results["cohen_kappa_note"] = (
                    f"Cohen's kappa requires exactly 2 raters (found {results['n_raters']})"
                )

        # ICC (Intraclass correlation)
        if metric in ("all", "icc"):
            try:
                eager_df = MaterializationBoundary.to_eager(
                    nw_df.select([item_col, rater_col, label_col]).drop_nulls()
                )
                icc_val = intraclass_correlation(
                    eager_df,
                    item_col=item_col,
                    rater_col=rater_col,
                    score_col=label_col,
                    variant="ICC2_1",
                )
                results["icc2_1"] = float(icc_val)
            except Exception as e:
                results["icc_note"] = f"ICC not applicable: {e}"

        out_text = json.dumps(results, indent=2) if format == "json" else _format_table(results)
        if output:
            Path(output).write_text(out_text)
            typer.echo(f"Results written to {output}")
        else:
            typer.echo(out_text)

    except Exception as e:
        typer.echo(f"Error calculating agreement: {e}", err=True)
        raise typer.Exit(1)


@calculate_app.command("consensus")
def calculate_consensus(
    file_path: str = typer.Argument(..., help="Path to input dataset (CSV, Parquet, Arrow)"),
    method: Literal["dawid_skene", "dawid-skene", "tree"] = typer.Option(
        "dawid_skene", "--method", "-m", help="Consensus algorithm: dawid_skene or tree"
    ),
    strategy: Literal["lca", "medoid", "majority"] = typer.Option(
        "lca", "--strategy", "-s", help="Tree strategy: lca, medoid, or majority"
    ),
    item_col: str = typer.Option("item_id", "--item-col", help="Item column"),
    rater_col: str = typer.Option("annotator_id", "--rater-col", help="Rater column"),
    label_col: str = typer.Option("label_id", "--label-col", help="Label column"),
    ontology: Optional[str] = typer.Option(None, "--ontology", help="Path to ontology file"),
    format: str = typer.Option("json", "--format", "-f", help="Output format: json or table"),
    output: Optional[str] = typer.Option(None, "--output", "-o", help="File to write results to"),
) -> None:
    """Calculate consensus item labels and class distributions deterministically."""
    try:
        onto = _load_optional_ontology(ontology)
        clean_method = method.replace("-", "_")

        if clean_method == "tree":
            if onto is None:
                # Build trivial fallback ontology if tree majority is requested without ontology
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

        d = res.to_dict()
        out_text = json.dumps(d, indent=2) if format == "json" else _format_table(d)
        if output:
            Path(output).write_text(out_text)
            typer.echo(f"Results written to {output}")
        else:
            typer.echo(out_text)

    except Exception as e:
        typer.echo(f"Error calculating consensus: {e}", err=True)
        raise typer.Exit(1)


@calculate_app.command("drift")
def calculate_drift(  # noqa: C901
    file_path: str = typer.Argument(..., help="Path to input dataset (CSV, Parquet, Arrow)"),
    col: str = typer.Option(..., "--col", "-c", help="Column to monitor for drift"),
    window_col: str = typer.Option("window_id", "--window-col", help="Window identifier column"),
    baseline: str = typer.Option(..., "--baseline", "-b", help="Baseline window identifier"),
    target: str = typer.Option(..., "--target", "-t", help="Target window identifier"),
    metric: Literal[
        "wasserstein", "psi", "jsd", "mmd", "cramer_v", "cramers_v", "cramer-v", "cramers-v"
    ] = typer.Option(
        "wasserstein",
        "--metric",
        "-m",
        help="Drift metric: wasserstein, psi, jsd, mmd, cramer_v",
    ),
    format: str = typer.Option("json", "--format", "-f", help="Output format: json or table"),
    output: Optional[str] = typer.Option(None, "--output", "-o", help="File to write results to"),
) -> None:
    """Calculate statistical distribution drift between baseline and target windows."""
    try:
        nw_df = normalize_to_narwhals(file_path)

        cols = [c.strip() for c in col.split(",")]
        clean_metric = metric.replace("-", "_")

        if clean_metric == "wasserstein":
            df_res = wasserstein_distance(nw_df, col=col, window_col=window_col, baseline=baseline)
        elif clean_metric == "psi":
            df_res = psi(nw_df, col=col, window_col=window_col, baseline=baseline)
        elif clean_metric == "jsd":
            df_res = jsd(nw_df, dist_cols=cols, window_col=window_col, baseline=baseline)
        elif clean_metric == "mmd":
            df_res = mmd(nw_df, emb_cols=cols, window_col=window_col, baseline=baseline)
        elif clean_metric in ("cramer_v", "cramers_v"):
            df_res = cramers_v(nw_df, x=col, y=col, window_col=window_col, baseline=baseline)

        # Extract target window row
        import pandas as pd

        if isinstance(df_res, pd.DataFrame):
            target_df = df_res[df_res["window_id"] == target]
            if not target_df.empty:
                result = {str(k): v for k, v in target_df.iloc[0].to_dict().items()}
            elif not df_res.empty:
                result = {str(k): v for k, v in df_res.iloc[0].to_dict().items()}
            else:
                result = {"metric": metric, "value": 0.0}
        else:
            result = {"metric": metric, "value": float(df_res)}

        out_text = json.dumps(result, indent=2) if format == "json" else _format_table(result)
        if output:
            Path(output).write_text(out_text)
            typer.echo(f"Results written to {output}")
        else:
            typer.echo(out_text)

    except Exception as e:
        typer.echo(f"Error calculating drift: {e}", err=True)
        raise typer.Exit(1)


def _format_table(data: dict[str, Any]) -> str:
    lines = ["Field | Value", "---|---"]
    for k, v in data.items():
        if isinstance(v, (list, dict)) and len(str(v)) > 60:
            lines.append(f"{k} | <{len(v)} items>")
        else:
            lines.append(f"{k} | {v}")
    return "\n".join(lines)
