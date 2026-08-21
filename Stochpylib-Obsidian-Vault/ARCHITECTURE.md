# stochpylib — Architecture & Design Intent

## Status

**Early implementation.** Most of this document still describes target design, not a built
system — but `stochpylib/probability/` is real, implemented, and tested (see
[Implementation-Checklist.md](../development/Implementation-Checklist.md) for exact progress). Everything else
in the module map remains planned.

## What stochpylib is meant to be

A single Python library spanning the full stack of stochastic/statistical computing — from
combinatorics and Bayes' theorem up through Lévy processes, random matrix theory, and
GPU-accelerated MCMC — so a user doesn't need to stitch together `scipy.stats` + `statsmodels` +
`pymc` + `arch` + `lifelines` + `copulas` separately. See [Module-Map.md](Module-Map.md) for the
full inventory: 23 top-level modules, ~120 submodules, ~790 public names.

## Package layout convention

Each top-level module in [Module-Map.md](Module-Map.md) maps to one `stochpylib/<module>/` package.
Within a module, submodules are flat sibling files (e.g. `stochpylib/timeseries/volatility_models.py`
holds the GARCH family), not deeply nested. The spec's own examples
([Quickstart-Examples.md](Quickstart-Examples.md)) import directly from the top-level module
(`from stochpylib.timeseries import GARCH, ARIMA`), so submodule names should stay as implementation
detail re-exported at the module `__init__.py` rather than required import paths.

## Cross-cutting conventions implied by the spec

- **Distribution interface**: every distribution in `distributions/` exposes the same method set —
  `.pdf()/.pmf()`, `.cdf()`, `.ppf()`, `.rvs()`, `.mean()`, `.var()`, `.skewness()`, `.kurtosis()`,
  `.entropy()`, `.mgf()`, `.cf()`, `.fit()`, `.ks_test()`. Any new distribution added later should
  satisfy this same surface — it's the load-bearing abstraction the rest of the library (fitting,
  goodness-of-fit, plotting) is built against.
- **Model fit/predict symmetry**: time series, GP, survival, and financial models all follow
  `.fit(data)` then query (`.predict()`, `.forecast()`, `.simulate()`) — see the
  [Quickstart examples](Quickstart-Examples.md) for ARIMA/GARCH, GP regression, Cox PH, and Heston.
- **Kernel composability**: Gaussian process kernels support algebraic composition
  (`RBFKernel(...) + MaternKernel(...)`) per `gaussian_processes.kernel_ops` — kernels should be
  designed as composable objects from the start, not single monolithic classes.
- **Diagnostics live next to the algorithm they check**: e.g. `advanced_mcmc.diagnostics`
  (`Rhat()`, `ESS()`) sits inside the MCMC module rather than a generic stats-testing module;
  `timeseries.tests` (ADF, KPSS, Ljung-Box) sits inside time series. Follow this pattern for new
  modules — don't centralize all hypothesis tests into `statistics` unless they're
  general-purpose.
- **GPU/parallel backend is a utility, not per-module**: `utils.performance` (`GPUBackend`,
  `ParallelSimulation`, `JIT_compile()`) is meant to be a shared backend that simulation-heavy
  modules (Monte Carlo, MCMC, financial stochastics) opt into, not something each module
  reimplements.

## Known gaps (from the design scorecard)

See [Ratings.md](Ratings.md) for the full reasoning. The two lowest-scored areas as currently
specced:
- **Spatial statistics (8/10)** — has kriging/variogram/point processes but no CAR/SAR models.
- **Bayesian inference (9/10)** — core prior/posterior/model-selection is covered but variational
  inference is thin relative to `bayesian.computation`'s other methods.

If/when those modules are implemented, treat these as the first follow-up additions rather than
re-deriving scope from scratch.

## How to navigate this vault

- [Module-Map.md](Module-Map.md) — index, scores, and links into `Modules/`.
- `Modules/<name>.md` — per-module target API (submodules + public names).
- [Ratings.md](Ratings.md) — design-completeness scorecard and the reasoning behind each score.
- [Quickstart-Examples.md](Quickstart-Examples.md) — intended public API usage patterns.
- [Dependencies.md](Dependencies.md) — planned runtime/dev/system dependencies.
- [Essential-Tasks.md](Essential-Tasks.md) — checklist to run through once a module is actually
  implemented.
