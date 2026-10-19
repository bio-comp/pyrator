---
name: "pyrator-calc"
description: "Use when calculating inter-rater reliability, agreement metrics (Krippendorff's alpha, Cohen's kappa, Fleiss' kappa, ICC), item consensus (Dawid-Skene, Tree LCA), or statistical distribution drift (Wasserstein distance, PSI, JSD, MMD, Cramer's V) from datasets to eliminate LLM mathematical calculation hallucinations."
---

# Pyrator Calculation Skill

Deterministic mathematical calculation tools and CLI recipes for inter-rater reliability, annotation consensus, semantic ontology distance, and dataset distribution drift.

## Core Principle

**Never guess, estimate, or hallucinate statistical metrics.**
LLMs lack floating-point precision and matrix decomposition capabilities required to accurately compute Krippendorff's Alpha, Fleiss' Kappa, Dawid-Skene expectation-maximization posteriors, or Wasserstein distances.

Always delegate mathematical calculations directly to `pyrator calculate` CLI subcommands or the stdio MCP calculation server.

---

## When to Use

- Calculating inter-rater agreement (Krippendorff's Alpha, Cohen's Kappa, Fleiss' Kappa, Intraclass Correlation / ICC).
- Computing semantic or hierarchical distance-weighted agreement with an ontology tree/DAG.
- Resolving ground truth or consensus labels across noisy annotators using Dawid-Skene EM or Tree LCA / Medoid consensus.
- Monitoring data drift across evaluation windows or production inference batches (Wasserstein-1 / EMD, Population Stability Index / PSI, Jensen-Shannon Divergence / JSD, Maximum Mean Discrepancy / MMD, Cramer's V).
- Querying exact graph path distances or semantic similarity between ontology nodes.

---

## CLI Recipes (`pyrator calculate`)

The `pyrator calculate` command suite works on CSV, Parquet, and Arrow files with lazy execution and zero unneeded copies.

### 1. Inter-Rater Agreement & Reliability

```bash
# Calculate all classical agreement metrics (Krippendorff, Fleiss, Cohen, ICC) in JSON format
pyrator calculate agreement annotations.parquet \
  --item-col item_id \
  --rater-col annotator_id \
  --label-col label_id \
  --metric all \
  --format json

# Calculate semantic Krippendorff's alpha weighted by an ontology tree/DAG
pyrator calculate agreement annotations.csv \
  --item-col item_id \
  --rater-col annotator_id \
  --label-col label_id \
  --ontology ontology.json \
  --metric krippendorff \
  --format table
```

Supported metrics: `all`, `krippendorff`, `cohen`, `fleiss`, `icc`.

### 2. Annotation Consensus Resolution

```bash
# Dawid-Skene EM with class posterior probabilities
pyrator calculate consensus annotations.parquet \
  --item-col item_id \
  --rater-col annotator_id \
  --label-col label_id \
  --method dawid-skene \
  --max-iter 100 \
  --tol 1e-4 \
  --format json \
  --output consensus_results.json

# Tree-based lowest common ancestor (LCA) consensus using an ontology
pyrator calculate consensus annotations.csv \
  --item-col item_id \
  --rater-col annotator_id \
  --label-col label_id \
  --method tree \
  --ontology ontology.json \
  --format table
```

### 3. Statistical Distribution Drift

```bash
# Wasserstein-1 distance (Earth Mover's Distance) for continuous or ordered scores
pyrator calculate drift monitoring_scores.parquet \
  --col score \
  --window-col window_id \
  --baseline baseline_v1 \
  --target curr_v2 \
  --metric wasserstein \
  --format json

# Population Stability Index (PSI) for binned drift
pyrator calculate drift inference_log.csv \
  --col prediction_score \
  --window-col window_id \
  --baseline 2026-w01 \
  --target 2026-w02 \
  --metric psi

# Categorical drift with Cramer's V
pyrator calculate drift predictions.parquet \
  --col predicted_category \
  --window-col batch_id \
  --baseline batch_0 \
  --target batch_1 \
  --metric cramer_v
```

Supported drift metrics: `wasserstein`, `psi`, `jsd`, `mmd`, `cramer_v`.

---

## Model Context Protocol (MCP) Server

Pyrator includes a stdio MCP server (`pyrator mcp` or `python -m pyrator.mcp`) implementing JSON-RPC 2.0 (protocol version `2024-11-05`). It provides 4 deterministic tools:

1. `calculate_agreement`: Computes Krippendorff, Fleiss, Cohen, ICC, and semantic Alpha.
2. `calculate_consensus`: Runs Dawid-Skene EM or Tree consensus and returns item-level consensus labels.
3. `calculate_drift`: Computes distribution drift metrics between baseline and target windows.
4. `query_ontology_distance`: Returns shortest path or semantic distance between ontology concepts.

### Configuring in Coding Agents

Add to your agent configuration (`claude_desktop_config.json`, `.gemini/settings.json`, or Codex agent manifest):

```json
{
  "mcpServers": {
    "pyrator": {
      "command": "uv",
      "args": ["run", "pyrator", "mcp"]
    }
  }
}
```

### Agent Tool Call Examples

#### Calculating Agreement
```json
{
  "name": "calculate_agreement",
  "arguments": {
    "data_path": "data/eval_annotations.parquet",
    "item_col": "item_id",
    "rater_col": "annotator_id",
    "label_col": "label_id",
    "metric": "all"
  }
}
```

#### Calculating Consensus
```json
{
  "name": "calculate_consensus",
  "arguments": {
    "data_path": "data/eval_annotations.parquet",
    "method": "dawid-skene",
    "item_col": "item_id",
    "rater_col": "annotator_id",
    "label_col": "label_id"
  }
}
```

#### Calculating Drift
```json
{
  "name": "calculate_drift",
  "arguments": {
    "data_path": "data/drift_log.parquet",
    "col": "confidence_score",
    "window_col": "window_id",
    "baseline": "baseline_q1",
    "target": "target_q2",
    "metric": "wasserstein"
  }
}
```
