# To-Do — Objectives Canvas

This is the owner's planning space, not a generated backlog. Whatever is written below is
the current objective(s) and any personal notes/context for it — read it before starting
work in this repo, and treat it as the live brief for what an AI agent should do next.
Expect this file to be rewritten or cleared out entirely as objectives change; it does not
accumulate history (that's what `development/CHANGELOG.md` and `development/Probleme.md`
are for — see `AGENTS.md`).

-----

## V0.17.0 — done

`experimental_design` shipped end to end (29/29 spec names, 689/794 overall, twenty
modules): classical/screening/RSM designs, optimal designs, space-filling designs, response
surfaces and surrogates, and DOE/sensitivity analysis, plus its `smoke (experimental_design)`
CI job. Two pre-existing library bugs surfaced and were fixed on the way (Probleme.md #109
Matern hyperparameters never optimized, #110 `MCResult` intervals wrong for levels != 0.95).

## Next candidate

Three modules remain: `spatial_statistics` (32), `viz` (35), `utils` (38). Each needs an
explicit decision before it can start:

- `spatial_statistics`: its `GaussianRandomField` collides with the one already shipped in
  `levy_processes` (rename, alias, or re-export).
- `viz`: a plotting-backend decision — matplotlib as a dev-only oracle/example dependency,
  or a from-scratch SVG/ASCII renderer to keep the zero-runtime-dep policy. It would also
  render `experimental_design`'s data-only `InteractionPlot`/`NormalPlot`.
- `utils.performance` (`GPUBackend`, `JIT_compile()`): optional torch/cupy/numba deps and a
  lazy-import design pass first.

Hygiene follow-up worth a patch release: a few modules still import `scipy.stats` in library
code (`montecarlo.sensitivity_analysis`, `gaussian_processes/inference.py`,
`information_theory` Wasserstein, `levy_processes/advanced.py`) despite the oracle-only rule.

-----

## Blocked by environment, not by choice

- Full monolithic `pytest tests/ -v` run: this dev sandbox's ~4GB RAM reliably OOM-kills
  it partway through; verified in per-module/per-chunk batches instead. Not expected
  to be an issue in real CI (GitHub Actions runners have far more headroom).
