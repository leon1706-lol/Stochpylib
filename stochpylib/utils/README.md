# stochpylib.utils

Infrastructure & utilities: seeding/reproducible streams, benchmarking/profiling and
parallel/GPU/JIT execution, reproducibility scaffolding, distribution fitting & data
cleaning, serialization, and interop with numpy/scipy/pandas/torch/jax — 38 spec names
across six submodules. `FitResult`/`OutlierResult`/`from_pickle` are documented extras
beyond the spec (a fit-comparison table, an outlier report, and the untrusted-pickle
counterpart to `to_pickle`).

**Status:** implemented & tested (38/38 spec names).

```python
import numpy as np
from stochpylib.utils import set_seed, fit, outlier_detection, ParallelSimulation, to_json
from stochpylib.montecarlo import MCResult

set_seed(2026)                         # library-wide root: random_state=None becomes reproducible

data = np.concatenate([np.random.default_rng(0).gamma(3.0, 2.0, 950), [80, -40, 90]])
best = fit(data)                       # AIC-ranked distribution comparison
print(best.best_name_, best.summary())

out = outlier_detection(data, method="mad")
clean = data[~out.mask]

def sim(n, rng):
    x = rng.standard_normal(n)
    return MCResult(float(x.mean()), float(x.std(ddof=1) / np.sqrt(n)), n, "demo")

est = ParallelSimulation(n_jobs=4).estimate(sim, n=200_000, random_state=1)
to_json(est, path="estimate.json")
```

## Files

- `random.py` — `set_seed()`/`random_state()`/`spawn_generator()` (the public face of
  `stochpylib._rng`), plus `Generator`/`SeedSequence` (identity re-exports of numpy's own
  classes, not stochpylib wrappers).
- `performance.py` — `Benchmark`/`Profiler` (stdlib `time`/`cProfile`/`tracemalloc`);
  `ParallelSimulation` (the public face of `stochpylib._parallel` — thread/process/auto
  backends, worker-count-invariant chunking); `GPUBackend` (numpy/cupy/torch/jax array
  ops, `simulate_gbm`); `JIT_compile` (numba/jax, falls back to plain Python with one
  warning on any compile/first-call failure — including a backend that is installed but
  broken for the running environment); `VectorizedOps` (log-domain stability, rolling
  stats, pairwise distances); `MemoryPool` (LRU ndarray buffer reuse).
- `reproducibility.py` — `Reproducibility` (seeding context manager + reproducibility
  self-check), `RandomStream` (named, checkpointable/jumpable stream), `VersionLock`
  (dependency-version snapshot/check), `EnvironmentCapture` (Python/numpy/scipy/BLAS/
  optional-backend snapshot), `ExperimentLogger` (JSON-Lines run logging).
- `data.py` — `fit()`/`goodness_of_fit()`/`moment_matching()`/`ecdf()` (all delegate to
  `distributions`/`statistics`/`nonparametric` rather than re-deriving anything);
  `DataValidation`; `outlier_detection()` (MAD/IQR/z-score/Grubbs/Hampel/MCD);
  `missing_imputation()` (mean/median/mode/constant/LOCF/NOCB/linear/kNN/EM/MICE).
- `io.py` — `to_dict()`/`from_dict()`/`to_json()`/`from_json()`/`to_pickle()` (+
  `from_pickle()`); `Serialization` (format-dispatching json/pickle/npz);
  `Configuration` (nested, dot-path, env-overridable, freezable); `Logging` (a facade
  over the stdlib `"stochpylib"` logger); `summary()` (a universal dispatcher over
  results/distributions/arrays/dicts).
- `compat.py` — `numpy_interface()` (duck-typed, no import needed); `scipy_interface()`
  (a read-only `scipy.stats`-frozen-distribution-shaped view over a stochpylib
  `Distribution` — never imports `scipy.stats` itself); `pandas_interface()`;
  `torch_interface()`/`jax_interface()` (arrays, native distribution objects, and
  autodiff-gradient wrappers usable as `advanced_mcmc`'s `grad_log_prob=`).
- `_backends.py` — the **only** file in the package allowed to import
  pandas/torch/jax/jaxlib/numba/cupy, and only lazily, inside function bodies (an AST
  guard in `tests/library/tests.py` enforces both rules, mirroring `viz/_mpl.py`'s
  matplotlib discipline). `import_backend()` turns any import failure — missing *or*
  installed-but-broken (e.g. numba against a newer numpy than it supports) — into one
  clear `ImportError` naming the pip extra.
- `_results.py` — `FitResult`/`OutlierResult` (private module; re-exported at the
  package level as documented extras).

## Conventions

- **Reuse, not re-derivation.** `fit()`/`goodness_of_fit()`/`moment_matching()`/`ecdf()`
  delegate to `distributions.*.fit()`, `statistics.{ks_test,chi2_test,MOM}`,
  `nonparametric.{AndersenDarling,CramerVonMises,EmpiricalCDF}` and
  `robust_statistics.{_mad,MCD}`; estimates come back as `montecarlo.MCResult` or
  `statistics.TestResult`, never a new ad hoc result type (`FitResult`/`OutlierResult`
  are reports, not estimates).
- **Every stochastic function takes `random_state=`**, resolved through the shared
  `stochpylib._rng.as_generator` (accepts `None`, an int/seed sequence, a `Generator`,
  a `RandomState`, or anything with `_as_generator()` such as `RandomStream`).
- **`set_seed()` is opt-in, not global by default.** It sets a library-wide root that
  `random_state=None` calls spawn independent children from; it never touches numpy's
  or Python's own global random state unless `numpy_global=`/`python_random=True` is
  passed explicitly.
- **Parallel results are reproducible but not legacy-equivalent.** `ParallelSimulation`/
  `n_jobs=` on `montecarlo`/`financial_stochastics`/`advanced_mcmc` give the same answer
  for any worker count at a fixed `(seed, n, chunk_size)`, because each chunk draws an
  independently spawned child stream — but that answer differs from the pre-V0.20.0
  single-stream serial result at the same seed. `n_jobs=None`/`backend=None` (the
  defaults everywhere) keep the exact legacy single-stream behavior.
- **Optional backends never gate import.** `import stochpylib.utils` (and the rest of
  the library) needs none of pandas/torch/jax/numba/cupy; each is imported only when a
  function that actually needs it is called, and a missing or broken one raises one
  clear `ImportError` naming the extra to install.

## Known limitations

- **`Generator`/`SeedSequence` are numpy's own classes**, re-exported for convenience —
  not stochpylib wrappers, so numpy's own documentation/semantics apply directly.
- **cupy is exercised in CI only through an injected fake module** (GitHub Actions
  runners have no GPU); `GPUBackend("cupy")` itself is untested against real hardware.
- **`goodness_of_fit`'s KS/Anderson-Darling/Cramer-von Mises p-values are not corrected
  for parameters estimated from the same data** (no Lilliefors-type correction) — treat
  them as approximate when `dist` was fit to `data` itself, as `fit()` -> `goodness_of_fit()`
  naturally does.
- **`missing_imputation`'s `"mice"` is capped at 20 chained iterations** regardless of
  `max_iter`, and its per-column model is always OLS (no logistic/count-model chaining
  for categorical/count columns).
- **`JIT_compile(backend="jax")`/`jax_interface`/`GPUBackend("jax")` enable
  `jax_enable_x64` globally** the first time any of them runs, a documented jax-wide
  side effect (jax defaults to float32 otherwise).
- **`torch_interface`/`jax_interface` distribution mapping is a fixed list** (Normal,
  Exponential, Gamma, Beta, Uniform, LogNormal, Student_t, Laplace, Cauchy, Poisson,
  Bernoulli, Binomial for torch; a smaller `jax.scipy.stats`-backed subset for jax) — any
  other `Distribution` raises a `TypeError` naming the supported set.

Spec: vault `Modules/utils.md` (private). Tests: `tests/utils/tests.py` (oracles:
`scipy.stats`, `scipy.special`, real pandas), `tests/utils/e2e.py` (API sweep), and
`tests/utils/backend_optional.py` (real torch/jax/numba/pandas checks, run only by the
`utils-optional` CI job — every other test in this module passes with none of them
installed at all).
