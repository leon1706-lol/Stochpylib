# Development Notes

This folder tracks the actual build-out of `stochpylib`, as distinct from the design spec in
[`Stochpylib-Obsidian-Vault/`](../Stochpylib-Obsidian-Vault/README.md). Use it to record what
was built, when, and what went wrong along the way — the vault tells you what's *supposed* to
exist; this folder tells you what *actually happened* getting there.

## Files

- [CHANGELOG.md](CHANGELOG.md) — append-only log of build phases, in order. One entry per
  meaningful chunk of work (a module, a tooling change, a packaging fix) — not one entry per
  commit.
- [Probleme.md](Probleme.md) — audit log of bugs/issues found during implementation or manual
  testing, with a severity score (1–10) and a Fixed/Open status. Skip an entry here if nothing
  was actually found during a task — this is not a general changelog (that's CHANGELOG.md).

## Package layout decisions (for context)

- **Flat-per-module packages**: each top-level module is `stochpylib/<module>/` with flat
  submodule files inside, an optional `_base.py` only when the module has real shared base-class
  behavior (none yet — `probability` is plain functions, no base class needed).
- **Tests live outside the package**: `tests/<module>/tests.py`, one file per module — not
  colocated inside the package (avoids the test file ending up bundled in the built wheel, and
  keeps `stochpylib/` itself free of test-only imports like `pytest`).
- **`stochpylib/` stays nested under the repo root**, not flattened to the repo root itself —
  even though the repo root will eventually be renamed to match the package name, a normal
  nested package directory is simpler and avoids non-standard `package-dir` mapping in
  `pyproject.toml`. This was tried and deliberately reverted — see
  [Probleme.md](Probleme.md) for why.

## Workflow

See [Essential-Tasks.md](../Stochpylib-Obsidian-Vault/Essential-Tasks.md) in the vault for the
full step-by-step wrap-up checklist (spec check → manual test → automated tests → docs →
[Implementation-Checklist.md](Implementation-Checklist.md) → this folder → vault regeneration →
`HANDOFF.MD`).
