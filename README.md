# pyrator

**Inter-rater reliability, consensus, ontology-aware agreement, and drift analysis in Python.**

> **Status:** Alpha (`0.1.0`) · Python 3.12+

Pyrator is a research-oriented Python library for analyzing how humans, models, or other raters agree, disagree, and change over time. It is designed for problems where labels may have structure—such as taxonomies, ontologies, or hierarchical guidelines—and where exact-match agreement alone can hide useful information.

The project combines classical reliability statistics with structured/semantic agreement, consensus analysis, scalable data backends, and drift monitoring.

---

## Why Pyrator?

Traditional agreement metrics are useful, but many real annotation and evaluation problems are more complicated than “same label” versus “different label.” Two labels may be neighbors in a hierarchy, one annotator may systematically choose broader concepts, a subset of items may be intrinsically ambiguous, or rater/model behavior may shift over time.

Pyrator is intended to help answer questions such as:

- How reliable are two or more raters?
- Which items create the most disagreement?
- Are disagreements small semantic differences or major cross-branch errors?
- What consensus label is best supported by the observations?
- Are particular raters systematically more or less specific?
- Has the data, label distribution, model behavior, or annotation process drifted over time?

---

## Core areas

### Inter-rater reliability

The codebase includes support and tests for core agreement/reliability workflows, including:

- Cohen's kappa and weighted kappa
- multi-rater agreement workflows
- Krippendorff-style reliability analysis
- intraclass correlation (ICC)
- missing-data-aware analysis
- configurable distance/difference functions

### Ontology-aware agreement

Pyrator is designed to work with labels that live in trees or directed acyclic graphs rather than treating every disagreement as equally severe.

Examples of supported or actively developed concepts include:

- ancestor and descendant relationships
- lowest/common-ancestor reasoning
- path-based semantic distance
- information-content-based similarity
- hierarchy-aware agreement and consensus
- semantic partial credit for nearby concepts

For example, choosing a parent concept rather than its child can be treated as a smaller disagreement than selecting an unrelated branch of the ontology.

### Consensus and disagreement analysis

The project includes dedicated consensus functionality and infrastructure for identifying hard or ambiguous items, comparing annotator behavior, and summarizing disagreement patterns.

### Drift monitoring

Pyrator also includes a drift-analysis subsystem for tracking changes in data and rater/model behavior over time. The codebase contains dedicated drift modules and tests for monitoring distributional and operational changes.

Typical targets include:

- feature/case-mix drift
- label-distribution drift
- rater behavior drift
- model-output drift
- association and calibration changes

### Data and execution backends

The library is designed to work across practical analysis environments rather than assuming every dataset fits comfortably in one in-memory pandas DataFrame.

Current project dependencies and optional backends include:

- pandas / NumPy / SciPy
- Polars
- DuckDB / Ibis / SQLAlchemy
- PyArrow
- scikit-learn for clustering extensions
- PyMC / JAX for Bayesian extensions

The repository also includes a CLI, Python API, configuration system, batching support, and tests for multiple data backends.

---

## Installation

The project currently targets Python 3.12+.

```bash
git clone https://github.com/bio-comp/pyrator.git
cd pyrator
pip install -e .
```

For development with `uv`:

```bash
uv sync --group dev
```

Optional capability groups are defined in `pyproject.toml`, including plotting, Bayesian methods, clustering, database/backends, and Polars support.

---

## Development

Run the test suite:

```bash
uv run pytest
```

Lint:

```bash
uv run ruff check .
```

Type-check:

```bash
uv run mypy pyrator
```

The project uses strict typing, pytest coverage, Ruff, and a modular package layout.

---

## Repository structure

```text
pyrator/
├── api/        # public Python API
├── cli/        # command-line interface
├── consensus/  # consensus and rater aggregation
├── data/       # data access and backend abstractions
├── drift/      # drift and monitoring functionality
├── ira/        # inter-rater agreement / reliability
├── ontology/   # hierarchical label and semantic structures
├── config.py   # configuration models
└── mcp.py      # integration surface
```

Implementation and design notes are kept under [`docs/plans/`](docs/plans/).

---

## Design principles

**Structured disagreement matters.** A near miss in a hierarchy should be distinguishable from a completely unrelated label.

**Evaluation should be inspectable.** Metrics, thresholds, configuration, and data transformations should be reproducible and explainable.

**Scale should not change the conceptual model.** The same analysis should be usable from small annotation studies through larger tabular datasets, with backend choices changing as needed.

**Uncertainty is information.** Difficult items, inconsistent raters, and unstable consensus should be surfaced rather than hidden behind a single summary statistic.

**Monitoring is part of evaluation.** Agreement measured once does not guarantee that the data, raters, or models behave the same way later.

---

## Public-data evaluation

Pyrator is intended to be evaluated with public or synthetic datasets that make assumptions and ground truth inspectable. Candidate public domains include pharmacogenomic recommendations, public clinical-practice guidelines, structured taxonomies, and other multi-rater or hierarchical-label datasets.

No employer-proprietary data, methods, or internal artifacts are required to use or evaluate this project.

---

## Project maturity

Pyrator is an **alpha-stage research/software project**. APIs and architecture may still change as the reliability, ontology, consensus, drift, and backend layers are consolidated and validated.

The repository intentionally separates implemented code from design/implementation notes so planned ideas are not presented as completed functionality.

---

## License

MIT License. See [`LICENSE`](LICENSE).
