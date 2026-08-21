# stochpylib — Obsidian Vault

## Purpose

This vault is agent context for **stochpylib**: a Python library covering probability,
distributions, time series, Gaussian processes, stochastic processes, financial stochastics,
and a long tail of statistical/numerical tooling. Implementation has started —
`stochpylib/probability/` is real and tested — but the vast majority of the module map in
[Module-Map.md](Module-Map.md) is still just design spec, originally captured from a design-spec
React component (a self-contained preview/pitch page enumerating the intended module tree,
ratings, and usage examples). This vault turns that spec into navigable Markdown so a future
agent picking up implementation work has the full target shape without re-reading the React file,
and tracks real progress against it in [Implementation-Checklist.md](../development/Implementation-Checklist.md).

**Naming note:** the library was originally drafted under the name "stochpy" (see the vault's own
folder name, `Stochpylib-Obsidian-Vault`, and the design-spec React component this vault was built
from). `stochpy` was already taken on PyPI by an unrelated package, as was `pystoch`, so the real
package was renamed to **`stochpylib`** before any code was written. All content here uses
`stochpylib`; only the vault's own folder name was left as `Stochpylib-Obsidian-Vault` for historical
continuity.

## What you'll find

- [Module-Map.md](Module-Map.md) — index of all 23 planned top-level modules with submodule/item
  counts and design-completeness scores.
- [Implementation-Checklist.md](../development/Implementation-Checklist.md) — single-file checkbox list of every
  module/submodule/public name across the whole spec, checked off as things actually get
  implemented and tested. The fastest way to see real progress at a glance.
- `Modules/` — one file per top-level module (`stochpylib.<name>.md`), listing its submodules and
  every public class/function the spec calls for. **Hand-authored from the spec, not generated
  from code** — there is nothing to scan yet.
- [ARCHITECTURE.md](ARCHITECTURE.md) — package layout conventions, design philosophy, and what
  "done" looks like for a module.
- [Ratings.md](Ratings.md) — the 23-area design-completeness scorecard (9.3/10 overall per the
  spec) with the reasoning behind each score and the gaps it flags.
- [Quickstart-Examples.md](Quickstart-Examples.md) — the usage examples from the spec, kept as a
  reference for the intended public API shape (install, GARCH, GP, copulas, survival, Hawkes,
  financial stochastics, NUTS, RMT, variance reduction).
- [Dependencies.md](Dependencies.md) — planned runtime/dev dependencies.
- [Essential-Tasks.md](Essential-Tasks.md) — wrap-up checklist for when a module actually gets
  implemented (tests, docs, vault regeneration).
- `scripts/` — generic repo-scanning vault generators (`regenerate_vault.py`,
  `generate_code_graph.py`), inherited from a vault template for an unrelated project. They scan
  whatever `--repo-root` you point them at; they are **not yet pointed at stochpylib source** because
  none exists. Once real implementation starts, rerun them to generate per-file/per-function
  `code/` notes that supplement — not replace — the hand-authored `Modules/` files.

## Quick start (agents & humans)

There is nothing to regenerate yet. When real implementation begins:

1. Check [ARCHITECTURE.md](ARCHITECTURE.md) and the relevant `Modules/<name>.md` file for the
   target API before writing code, so the implementation matches the spec (or the spec gets
   updated to match a deliberate deviation — update the `Modules/` file in that case).
2. Once a module has real source, generate code notes:

   ```powershell
   python scripts/generate_code_graph.py --repo-root .. --vault . --append-handoff `
     --agent <your-name> --summary "Generated code notes for <module>"
   ```
3. Refresh the top-level index:

   ```powershell
   python scripts/regenerate_vault.py --repo-root .. --vault . --append-handoff `
     --agent <your-name> --action updated --summary "Regenerated project map"
   ```

## Notes and warnings

- `Modules/*.md` describes **intent**, not reality. As soon as a module is implemented, diff the
  real public API against its `Modules/<name>.md` file and update whichever is wrong — don't let
  them silently diverge.
- The previous version of this vault documented an unrelated project (a git-like ML artifact
  version-control tool). That content has been removed; if you find stray references to it,
  they're leftovers that should be deleted.
