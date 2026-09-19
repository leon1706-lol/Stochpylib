# To-Do — Objectives Canvas

This is the owner's planning space, not a generated backlog. Whatever is written below is
the current objective(s) and any personal notes/context for it — read it before starting
work in this repo, and treat it as the live brief for what an AI agent should do next.
Expect this file to be rewritten or cleared out entirely as objectives change; it does not
accumulate history (that's what `development/CHANGELOG.md` and `development/Probleme.md`
are for — see `AGENTS.md`).

-----

## V0.10.0 — complete

`stochpylib.random_matrix` (thirteenth module, 23/23 spec names) is implemented, tested and
documented — see `development/CHANGELOG.md` Phase 27. The testing infrastructure the brief
asked for is in place: every module has an end-to-end API sweep (`tests/<module>/e2e.py`,
one exercise per public name, guarded), CI runs one visible `smoke (<module>)` job per
module plus `cross-suite` and `install-smoke`, and the sweeps surfaced and fixed twelve shipped
bugs (`Probleme.md` #71–#82). Verification: every `tests/<module>` pair green, cross suites
green, `spl --test` 168/168, wheel build + spec-name check green.

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
