# ICC Variants Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Add strict continuous-agreement ICC support for `ICC(2,1)`, `ICC(2,k)`, `ICC(3,1)`, and `ICC(3,k)` in `pyrator.ira`.

**Architecture:** Implement a typed public dispatcher `intraclass_correlation(..., variant=...)` plus explicit wrapper functions (`icc_2_1`, `icc_2_k`, `icc_3_1`, `icc_3_k`). Internally compute two-way ANOVA mean squares from a strict complete item×rater matrix and map formulas to Shrout/Fleiss variants.

**Tech Stack:** Python 3.12, numpy, pandas-compatible `FrameLike`, pytest, ruff, mypy.

---

### Task 1: Add failing ICC tests (RED)

**Files:**
- Modify: `tests/conftest.py`
- Create: `tests/test_icc.py`

1. Add `icc_continuous_data` fixture with deterministic numeric long-format ratings.
2. Add variant reference tests for all four ICC variants.
3. Add strict-validation tests for malformed inputs (duplicates, missing ratings, non-numeric values, unknown variant).
4. Run targeted tests and confirm failure due missing ICC implementation.

### Task 2: Implement ICC module (GREEN)

**Files:**
- Create: `pyrator/ira/icc.py`

1. Add typed public API with `ICCVariant` literal set.
2. Implement strict validation and complete matrix construction.
3. Compute ANOVA mean squares and apply four Shrout/Fleiss formulas.
4. Add wrapper functions for each variant.

### Task 3: Wire exports and pass tests

**Files:**
- Modify: `pyrator/ira/__init__.py`

1. Export dispatcher and wrapper functions.
2. Run targeted tests until green.

### Task 4: Quality gate for touched files

**Files:**
- `pyrator/ira/icc.py`
- `pyrator/ira/__init__.py`
- `tests/test_icc.py`
- `tests/conftest.py`

1. `uv run ruff check ...`
2. `uv run mypy ...`
3. `uv run pytest tests/test_icc.py -q`
4. `uv run pytest tests/ -q`

### Task 5: Prep for PR handoff

**Files:**
- Modify: `TODO.codex.md` (local, uncommitted)

1. Record implementation status and verification evidence.
2. Stage only tracked source/test files for commit.
3. Commit + push + open PR (when requested).
