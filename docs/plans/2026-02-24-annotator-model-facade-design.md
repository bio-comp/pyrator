# AnnotatorModel Facade Design

## Goal
Add a minimal but functional user-facing API that wraps current IRA components and returns structured results, while keeping behavior strict and forward-compatible with README v1.0 contracts.

## Scope
- Add `AnnotatorModel` facade and `ModelResults` container in `pyrator/api.py`.
- Support two fitting modes:
  - `nominal`: Krippendorff alpha with nominal distance.
  - `semantic`: Krippendorff alpha with ontology-derived distance matrix.
- Support semantic metrics: `path`, `lin`, `resnik_norm` with strict validation.
- Support configurable input column names with canonical defaults:
  - `item_col="item_id"`
  - `rater_col="annotator_id"`
  - `label_col="label_id"`

## Architecture
`AnnotatorModel.fit(...)` converts incoming frame-like data to pandas, validates schema/content, computes alpha through existing IRA modules, and derives deterministic summary artifacts for `ModelResults`.

`ModelResults` provides minimal functional accessors:
- `get_hard_items(top_n=10)`: highest disagreement-rate items.
- `get_annotator_profiles()`: per-rater agreement/disagreement with consensus.
- `get_consensus_labels()`: deterministic item-level consensus series.

## Validation Rules (Strict)
- Required configured columns must exist.
- No null values in required columns.
- Exactly one rating per `(item, rater)` pair.
- Non-empty dataset.
- Semantic mode requires ontology.
- Semantic metric must be one of: `path`, `lin`, `resnik_norm`.

## Deterministic Result Rules
- Consensus label per item is mode with lexical tie-break (`str` sort).
- Hard item ranking order:
  1. `disagreement_rate` descending
  2. `n_ratings` descending
  3. `item_id` ascending (string order)
- Annotator profiles include `n_items`, `agreement_rate`, `disagreement_rate`.

## Tradeoffs
- This facade intentionally does not introduce `AnnotationData` class yet; it accepts current frame-like inputs to minimize churn.
- It focuses on alpha-based reliability only for now; future facade extensions can add kappa/ICC and strategy pipelines without breaking signatures.

## Testing Strategy
- TDD with failing API tests first.
- Cover nominal and semantic happy paths.
- Cover strict validation failures.
- Cover deterministic behavior for consensus and ranking.
