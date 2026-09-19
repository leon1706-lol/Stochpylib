# To-Do — Objectives Canvas

This is the owner's planning space, not a generated backlog. Whatever is written below is
the current objective(s) and any personal notes/context for it — read it before starting
work in this repo, and treat it as the live brief for what an AI agent should do next.
Expect this file to be rewritten or cleared out entirely as objectives change; it does not
accumulate history (that's what `development/CHANGELOG.md` and `development/Probleme.md`
are for — see `AGENTS.md`).

-----

## V0.11.0 — advanced_mcmc — DONE, awaiting commit confirmation

Implemented `stochpylib.advanced_mcmc` (35/35 spec names, 6 submodules) end to end:
`tests/advanced_mcmc/{tests.py,e2e.py}` (48 + 39 tests), the `smoke (advanced_mcmc)`
CI job, `spl demo`/selftest coverage (177 checks), all docs/counters synced, vault
handoff done. Full suite verified green in per-module chunks (1538 passed / 2 skipped
of 1540 collected — matches the README exactly) plus a from-scratch wheel install-smoke
check in a clean venv.

Status: [x] 1 package  [x] 2 manual debug  [x] 3 tests  [x] 4 wiring  [x] 5 docs
        [x] 6 vault handoff  [ ] 7 commit — proposed, waiting on the user (AGENTS.md §5.6)

Two library bugs found and fixed while testing (see `development/Probleme.md` #83-84):
`NoUTurnSampler` never counted its own divergences into the public `divergences_`
attribute, and the diagnostics rank-normalization transform used the wrong Blom-formula
denominator (gave `NaN`/`inf` R-hat on perfectly good chains).

-----

## Blocked by environment, not by choice

- Full monolithic `pytest tests/ -v` run: this dev sandbox's ~4GB RAM reliably OOM-kills
  it partway through; verified in per-module/per-chunk batches instead. Not expected
  to be an issue in real CI (GitHub Actions runners have far more headroom).

-----

### Future work not in scope for current plans

- V0.12.0: next module per the README roadmap is `bayesian` (25 names, 4 submodules).
  `Ratings.md` notes its variational-inference submodule is thin relative to the rest
  and suggests it could delegate to `advanced_mcmc.variational` (now implemented)
  instead of duplicating scope — worth reconsidering when that module is picked up.
