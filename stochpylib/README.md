# stochpylib/ — the installable package

One subpackage per library module. Status:

- `probability/` — complete, tested (21 public names).
- `distributions/` — complete, tested (60 public names: 47 distributions + 13 shared methods).
- `montecarlo/` — complete, tested (25 public names: sequences, estimators, variance
  reduction, applications).
- `timeseries/` — complete, tested (61 public names: linear models, volatility, state
  space, latent regimes, changepoints, spectral analysis, forecasting).
- `gaussian_processes/` — complete, tested (36 public names: composable kernel zoo,
  exact/sparse GP regression, Laplace/EP/VI classification, hyperparameter optimization).
- `copulas/` — complete, tested (26 public names: elliptical/Archimedean/empirical
  copula families, C-/D-/R-vines, dependence measures and CopulaFit dispatcher).

Public API is re-exported from `stochpylib/__init__.py`; currently `probability`,
`distributions`, `montecarlo`, `timeseries`, `gaussian_processes`, and `copulas`.
The package also ships a CLI (`spl`, see
`cli.py`) and an embedded self-check suite (`selftest.py`, run via `spl --test`).

Target API for every planned module lives in the private Obsidian vault
(`../Stochpylib-Obsidian-Vault/Modules/<name>.md`).

Run all tests from the repo root: `pytest tests/ -v`.
