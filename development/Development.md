# Development Notes

This folder tracks the actual build-out of `stochpylib`, as distinct from the
design spec in [`../Stochpylib-Obsidian-Vault/`](../Stochpylib-Obsidian-Vault/README.md).
The vault tells you what's *supposed* to exist; this folder tells you what
*actually happened* getting there — what was built, when, and what went wrong
along the way.

## Files

- [`architecture.md`](architecture.md) — the current system architecture:
  objective, module map, the common distribution contract, cross-cutting
  conventions, package layout rules.
- [`infrastructure.md`](infrastructure.md) — the build/packaging/CI/release
  runbook: local setup, the `spl` CLI, GitHub Actions workflows, the PyPI
  Trusted-Publisher pipeline.
- [`project_structure.md`](project_structure.md) — the annotated directory tree.
- [`CHANGELOG.md`](CHANGELOG.md) — append-only log of build phases, in order.
  One entry per meaningful chunk of work (a module, a tooling change, a
  packaging fix) — not one entry per commit.
- [`Probleme.md`](Probleme.md) — audit log of bugs/issues found during
  implementation or manual testing, with a severity score (1–10) and a status
  from the legend at the top. Skip an entry here if nothing was actually found
  during a task — this is not a general changelog (that's CHANGELOG.md).
- [`Implementation-Checklist.md`](Implementation-Checklist.md) — every planned
  public name as a checkbox; the single-glance progress tracker (currently
  317/794). Update it whenever you implement.

## Package layout decisions (for context)

- **Flat-per-module packages**: each top-level module is `stochpylib/<module>/`
  with flat submodule files inside, an optional `_base.py` only when the module
  has real shared base-class behavior (e.g. `distributions`, `copulas`,
  `information_theory`; `probability` is plain functions, no base class needed).
- **Tests live outside the package**: `tests/<module>/tests.py`, one file per
  module — not colocated inside the package. This avoids test files shipping in
  the built wheel and keeps `stochpylib/` free of test-only imports like
  `pytest` (see `Probleme.md` [3] for the packaging bug that forced the move).
- **`stochpylib/` stays nested under the repo root**, not flattened to the repo
  root itself — a normal nested package directory avoids non-standard
  `package-dir` mapping in `pyproject.toml`. This was tried and deliberately
  reverted — see `Probleme.md` for why.

## Workflow

See [`Essential-Tasks.md`](../Stochpylib-Obsidian-Vault/Essential-Tasks.md) in
the vault for the full step-by-step wrap-up checklist (spec check → manual test
→ automated tests → docs → `Implementation-Checklist.md` → this folder → vault
regeneration → `HANDOFF.MD`). Operational details (local setup, CLI, CI,
releases) live in [`infrastructure.md`](infrastructure.md).
