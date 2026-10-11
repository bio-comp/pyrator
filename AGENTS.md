# AGENTS.md - pyrator Development Guide

This file defines how agentic coding assistants must operate in `pyrator`.

## Project Overview

`pyrator` is a scientific Python library for inter-rater reliability with ontology-aware metrics and backend-agnostic data loading.

Primary goals:

- Correctness first (strict TDD and reproducible scientific results).
- Modern Python (3.12+), strict typing, clear API boundaries.
- Backend-agnostic behavior across `pandas`, `polars`, `duckdb`, and `pyarrow` where applicable.

## Mandatory Toolchain

- Package manager: `uv` only.
- Lint/format: `ruff`.
- Type checking: `mypy` in strict mode.
- Logging: `loguru` (not stdlib `logging`).
- Testing: `pytest` and `pytest-cov`.

Never use `pip` or `poetry` directly for project tasks.

## Setup and Verification Commands

```bash
# Sync environment (recommended full optional dependency set)
uv sync --all-extras

# Run tests
uv run pytest tests/ -v
uv run pytest tests/test_ira.py -v

# Quality gate (must pass before PR handoff)
uv run ruff check . --fix
uv run ruff format .
uv run mypy pyrator/
```

## Required Development Workflow (Strict TDD)

For every bugfix or feature:

1. Define or update minimal fixtures in `tests/conftest.py`.
2. Add a failing test in `tests/` asserting expected behavior or scientific value.
3. Implement the minimal change in `pyrator/` to make tests pass.
4. Refactor for clarity and typing after green.
5. Re-run targeted tests, then full quality gate.

Do not implement behavior changes without a failing test first.

## Code Standards

### Python and Style

- Use built-in type syntax (`list[str]`, `X | None`).
- Use double quotes consistently.
- Use Google-style docstrings.
- Keep line length at `<= 100` characters.
- Prefer vectorized operations for numerical/dataframe paths.

### Architecture Rules

- Use `BackendRegistry` for backend selection; avoid hardcoded backend condition chains.
- Prefer `Protocol` interfaces for backend contracts and cross-module boundaries.
- Import optional dependencies lazily inside backend methods or guarded blocks.
- Keep module boundaries clean:
  - `pyrator/data`: I/O, validation, canonicalization, backend dispatch.
  - `pyrator/ontology`: ontology graph logic and semantic distances.
  - `pyrator/ira`: reliability metrics and distance-based alpha logic.
  - `pyrator/bayes`: Bayesian models (current/future implementation area).

### Scientific Rigor

- Reference formulas/papers in comments when implementing IRA/Bayesian metrics.
- Keep metric behavior explicit and reproducible (normalization and policy choices must be encoded and testable).
- Favor strict validation as the default for scientific metrics (fail fast on malformed inputs).
- Treat permissive handling (for example auto-dropping incomplete rows) as explicit, opt-in enhancements
  tracked in separate issues and never enabled by default.

### Safety Constraints

- Use `pathlib.Path` for file paths.
- Check `pyproject.toml` or `uv.lock` before adding imports.
- Avoid `typing.Any` unless unavoidable and documented.

## Optional Dependency and Backend Policy

- Optional backends must degrade gracefully when dependencies are missing.
- Tests that require optional dependencies must skip cleanly when unavailable.
- Capability declarations must reflect actual per-format behavior.
- Auto-selection should prefer capability-compatible available backends.

## Git and Collaboration Workflow

Use this exact cycle for each work item:

1. Create or pick a GitHub issue (small and atomic).
2. Create a PR branch:

   ```bash
   git checkout -b issue/<number>-short-description
   ```

3. Implement with strict TDD and run verification commands.
4. Push branch:

   ```bash
   git push -u origin issue/<number>-short-description
   ```

5. Open PR and include:
   - Issue reference.
   - Test evidence (commands and outcomes).
   - Risk/rollback notes if behavior changes.
6. User merges and deletes remote branch.
7. Local cleanup:

   ```bash
   git checkout main
   git pull --ff-only origin main
   git branch -d issue/<number>-short-description
   ```

8. Repeat on next issue.

## Agent-Specific Operating Rules

- Do not commit unless the user explicitly asks for a commit.
- Keep a local WIP tracker at `TODO.codex.md` with:
  - Current task state.
  - Next planned action.
  - Verification status.
- Before claiming completion, always run and report relevant verification commands.
- If a command cannot run (missing deps/sandbox/etc.), report the blocker explicitly.

## Dependency Reference

Core dependencies currently include:

- `numpy`, `scipy`, `pandas`, `krippendorff`, `pydantic`, `pydantic-settings`, `pandera`, `scikit-learn`, `orjson`, `loguru`.

Optional groups include:

- Plotting: `matplotlib`, `seaborn`.
- Bayesian: `pymc`, `jax`.
- Database/backends: `duckdb`, `duckdb-engine`, `ibis-framework`, `sqlalchemy`, `pyarrow`, `polars`.

## Local File Note

This `AGENTS.md` is local agent guidance. It should remain uncommitted unless the user explicitly requests committing it.
