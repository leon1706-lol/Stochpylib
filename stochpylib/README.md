# stochpylib/ — the installable package

One subpackage per library module. Status:

- `probability/` — complete, tested (21 public names, 25 tests).
- `distributions/` — **work in progress**: distribution classes are written but not yet
  exported via `__init__.py`, not tested, and not checked off in the implementation checklist.
  Do not rely on it yet.

Public API is re-exported from `stochpylib/__init__.py`; currently only `probability`.
Target API for every planned module lives in the private Obsidian vault
(`../Stochpylib-Obsidian-Vault/Modules/<name>.md`).

Run all tests from the repo root: `pytest tests/ -v`.
