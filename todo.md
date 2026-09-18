# To-Do — Objectives Canvas

This is the owner's planning space, not a generated backlog. Whatever is written below is
the current objective(s) and any personal notes/context for it — read it before starting
work in this repo, and treat it as the live brief for what an AI agent should do next.
Expect this file to be rewritten or cleared out entirely as objectives change; it does not
accumulate history (that's what `development/CHANGELOG.md` and `development/Probleme.md`
are for — see `AGENTS.md`).

-----

## V0.9.0 — complete

`stochpylib.statistics` (twelfth module, 48/48 spec names) is implemented, tested, and
documented — see `development/CHANGELOG.md` Phase 26. Full verification: `tests/statistics`,
`tests/library`, `tests/docs`, `tests/cli` all green; `spl --test` reports 160/160 checks
passing; per-module/chunked `pytest tests/ -v` runs green throughout (see the note below on
why chunked, not monolithic).

No new objective has been set yet — treat an otherwise-empty canvas as a prompt to *ask* the
owner what's next (or fall back to the roadmap order in `README.md`'s Roadmap section:
advanced MCMC, Bayesian inference, nonparametric methods, robust statistics, ...).

-----

## Blocked by environment, not by choice

- Full monolithic `pytest tests/ -v` run: this dev sandbox's ~4GB RAM reliably OOM-kills
  it partway through; verified in per-module/per-chunk batches instead. Not expected
  to be an issue in real CI (GitHub Actions runners have far more headroom).

-----

### Future work not in scope for current plans

-//-
