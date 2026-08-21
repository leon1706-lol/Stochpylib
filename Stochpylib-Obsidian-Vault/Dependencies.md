# Dependencies — stochpylib (planned)

No `setup.py`/`pyproject.toml` exists yet. These are the dependencies implied by the design spec's
module list and quickstart examples — use as a starting point when scaffolding the package, not
as an already-pinned manifest.

## Core runtime (implied by nearly every module)

```
numpy            # array backend for every module
scipy            # distributions, optimization, integration, linear algebra baselines
```

## Per-module implied dependencies

- **timeseries / advanced_mcmc (diagnostics)**: `statsmodels` overlaps heavily with `timeseries`
  (ARIMA/GARCH/state-space) and `statistics.hypothesis` — decide early whether stochpylib wraps it or
  reimplements natively, since the spec implies a from-scratch implementation ("rivals
  statsmodels + scipy + pymc3 combined").
- **gaussian_processes**: no hard dependency required if kernels/inference are implemented
  natively; `scipy.linalg` (Cholesky) covers exact GP inference.
- **advanced_mcmc (variational)**: `NormalizingFlows` likely needs a tensor/autodiff backend
  (`torch` or `jax`) — see `utils.compat.torch_interface()` / `jax_interface()`, which suggests
  these are optional, lazily-imported extras rather than hard dependencies.
- **utils.performance.GPUBackend**: optional GPU extra — likely `cupy` or `torch`, lazily imported
  like the framework plugins pattern (raise a clear `ImportError` pointing at the extra if
  missing, don't hard-depend).
- **viz**: `matplotlib` for all `plot_*()` functions.
- **utils.io**: stdlib `json`/`pickle` cover `to_json()`/`to_pickle()`; no extra dependency needed.
- **utils.compat**: `pandas` for `pandas_interface()`; `torch`/`jax` as optional extras for their
  respective interfaces — mirrors the lazy-import-with-clear-error pattern.

## Build / packaging

```
setuptools>=68.0
```

No C++ extension is implied by the spec (unlike the previous, unrelated project this vault used to
document) — everything described is pure-Python/NumPy/SciPy in scope, except optionally GPU/autodiff
backends as extras.

## System requirements

- **Python:** 3.10+ (consistent with current NumPy/SciPy support windows)

## Open question for whoever scaffolds the package

Decide and record here once decided: does stochpylib vendor/reimplement everything natively (per the
spec's "rivals statsmodels + scipy + pymc3 combined" framing), or does it wrap existing libraries
for the well-trodden areas (ARIMA, GARCH, KDE) and add native implementations only where the spec
goes beyond what those libraries offer (Hawkes processes, vine copulas, RMT, NUTS)? This materially
changes the dependency list above.
