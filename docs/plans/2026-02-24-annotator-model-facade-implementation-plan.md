# AnnotatorModel Facade Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Implement `AnnotatorModel` + `ModelResults` with strict nominal/semantic alpha support and deterministic result accessors.

**Architecture:** Add a new facade module (`pyrator/api.py`) that delegates alpha computation to existing IRA classes and builds deterministic summary outputs. Keep API small, typed, and validation-first.

**Tech Stack:** Python 3.12, pandas/numpy, existing `pyrator.ira` + `pyrator.ontology`, pytest, ruff, mypy.

---

### Task 1: RED tests for facade contract

**Files:**
- Create: `tests/test_api.py`
- Modify: `tests/conftest.py` (only if extra fixtures needed)

1. Add failing tests for:
   - nominal fit with configurable columns and functional `ModelResults` accessors
   - semantic fit for `path`, `lin`, `resnik_norm`
   - strict validation errors (missing columns, duplicates, invalid metric, missing ontology)
   - deterministic consensus tie-break behavior
2. Run: `uv run pytest tests/test_api.py -q` and confirm failure.

### Task 2: GREEN implementation for API module

**Files:**
- Create: `pyrator/api.py`

1. Add typed `AnnotatorModel` and `ModelResults`.
2. Implement strict input validation and normalized internal frame handling.
3. Wire nominal and semantic alpha computations through existing modules.
4. Implement deterministic accessors.

### Task 3: Export facade from package root

**Files:**
- Modify: `pyrator/__init__.py`

1. Export `AnnotatorModel` and `ModelResults` from top-level package.

### Task 4: Verification

**Files:**
- `pyrator/api.py`
- `pyrator/__init__.py`
- `tests/test_api.py`
- any touched fixture file

1. `uv run pytest tests/test_api.py -q`
2. `uv run ruff check pyrator/api.py pyrator/__init__.py tests/test_api.py tests/conftest.py`
3. `uv run mypy pyrator/api.py pyrator/__init__.py tests/test_api.py tests/conftest.py`
4. `uv run pytest tests/ -q`

### Task 5: Branch handoff

1. Update `TODO.codex.md` (local only) with status and test evidence.
2. Commit only tracked code/test files.
3. Push branch and open PR linking `#9`.
