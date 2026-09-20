# To-Do — Objectives Canvas

This is the owner's planning space, not a generated backlog. Whatever is written below is
the current objective(s) and any personal notes/context for it — read it before starting
work in this repo, and treat it as the live brief for what an AI agent should do next.
Expect this file to be rewritten or cleared out entirely as objectives change; it does not
accumulate history (that's what `development/CHANGELOG.md` and `development/Probleme.md`
are for — see `AGENTS.md`).

-----

## V0.14.0 

Biggest remaining module by name count is `utils` (38 names, tied with where
`numerical_methods` started) — `viz` (35), `spatial_statistics`/`optimization` (32 each),
`nonparametric` (31), `experimental_design` (29), `robust_statistics` (28) round out the
rest. `utils.performance` (`GPUBackend`, `JIT_compile()`) implies optional third-party
deps (torch/cupy/numba) that would need a lazy-import pattern to respect the "NumPy +
SciPy, nothing else" runtime rule — worth a design pass before picking it up. Given
`bayesian` was picked ahead of `nonparametric`/`robust_statistics` for being smallest,
the same size-first heuristic points at `robust_statistics` (28) next, unless the owner
prefers a different order.


- please now start the full end to end implementation of the next biggest module end to end without deffering anything
- than add extentsive testing and smoke github actions test
- than do the verification tasksk as outlined in agents md without missing anything
-----

## Blocked by environment, not by choice

- Full monolithic `pytest tests/ -v` run: this dev sandbox's ~4GB RAM reliably OOM-kills
  it partway through; verified in per-module/per-chunk batches instead. Not expected
  to be an issue in real CI (GitHub Actions runners have far more headroom).
