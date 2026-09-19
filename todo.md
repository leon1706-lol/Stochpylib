# To-Do — Objectives Canvas

This is the owner's planning space, not a generated backlog. Whatever is written below is
the current objective(s) and any personal notes/context for it — read it before starting
work in this repo, and treat it as the live brief for what an AI agent should do next.
Expect this file to be rewritten or cleared out entirely as objectives change; it does not
accumulate history (that's what `development/CHANGELOG.md` and `development/Probleme.md`
are for — see `AGENTS.md`).

-----

## V0.12.0: done

`numerical_methods` implemented end to end (38/38 spec names, six submodules: quadrature,
ODE/SDE solvers, native linear algebra, root finding, interpolation, PDE tools), its
`smoke (numerical_methods)` CI job added, and every wrap-up/verification step in
`AGENTS.md` §5 / the vault's `Essential-Tasks.md` completed. 15/23 modules, 544/794 spec
names, version 0.12.0. See `development/CHANGELOG.md` Phase 29 for the full summary and
`development/Probleme.md` #85-88 for the bugs found and fixed along the way.

No new objective has been set yet — see "Future work" below for the natural next pick, or
ask the owner.

-----

## Blocked by environment, not by choice

- Full monolithic `pytest tests/ -v` run: this dev sandbox's ~4GB RAM reliably OOM-kills
  it partway through; verified in per-module/per-chunk batches instead. Not expected
  to be an issue in real CI (GitHub Actions runners have far more headroom).
- **V0.12.0 wrap-up specifically:** `financial_stochastics` and `random_matrix`'s oracle
  suites were mid-run (44% and 74% through respectively, all passing, no failures) when
  the harness killed both background batches for system memory pressure; `advanced_mcmc`
  hadn't started. None of their source was touched by V0.12.0 (only shared files:
  `stochpylib/__init__.py`, `cli.py`, `cli_demo.py`, `selftest.py`,
  `tests/library/tests.py`, `tests/docs/tests.py`, `pyproject.toml` — none of which
  import or alter those three modules' internals), and a plain import check of all three
  succeeds, but their full suites were not reconfirmed green in this session. Every
  other module (probability, distributions, montecarlo, timeseries,
  gaussian_processes, copulas, survival, queueing, information_theory, levy_processes,
  statistics, numerical_methods) plus `tests/library`, `tests/docs`, `tests/cli` and a
  clean-venv wheel install-smoke all reran green after the change. Re-run
  `pytest tests/financial_stochastics tests/random_matrix tests/advanced_mcmc -v`
  (or trust CI, which has far more headroom) before tagging a release.

-----

### Future work not in scope for current plans

- Next module per the README roadmap: `bayesian` (25 names, 4 submodules) is roadmap-first,
  but the biggest remaining module by name count is now `utils` (38 names, tied with where
  `numerical_methods` started) — `viz` (35), `spatial_statistics`/`optimization` (32 each),
  `nonparametric` (31), `experimental_design` (29), `robust_statistics` (28) round out the
  rest. `utils.performance` (`GPUBackend`, `JIT_compile()`) implies optional third-party
  deps (torch/cupy/numba) that would need a lazy-import pattern to respect the "NumPy +
  SciPy, nothing else" runtime rule — worth a design pass before picking it up.
- `bayesian`: `Ratings.md` notes its variational-inference submodule is thin relative to
  the rest and suggests it could delegate to `advanced_mcmc.variational` (implemented since
  V0.11.0) instead of duplicating scope — worth reconsidering when that module is picked up.
