# stochpylib/

The installable package: one subpackage per library module behind a single
load-bearing contract — every distribution exposes the same method set, every
stochastic method takes `random_state=`, every Monte Carlo estimator returns a
shared result object. Twenty-three subpackages live today (794/794 spec names):

| Subpackage | Spec names | What it owns | Guide |
|---|---|---|---|
| `probability/` | 21 | Sample spaces, Bayes' theorem, exact combinatorics, independence | [README](probability/README.md) |
| `distributions/` | 60 | 47 distributions behind the common interface | [README](distributions/README.md) |
| `montecarlo/` | 25 | QMC sequences, MC estimators, variance reduction, applications | [README](montecarlo/README.md) |
| `timeseries/` | 61 | Linear/volatility models, filters, changepoints, spectral analysis | [README](timeseries/README.md) |
| `gaussian_processes/` | 36 | Kernel zoo, exact/sparse GP regression & classification | [README](gaussian_processes/README.md) |
| `copulas/` | 26 | Elliptical/Archimedean/empirical copulas, vines, dependence measures | [README](copulas/README.md) |
| `survival/` | 28 | Kaplan-Meier, parametric fits, Cox/AFT/FineGray, competing risks | [README](survival/README.md) |
| `queueing/` | 29 | M/M/1-Jackson networks, blocking formulas, discrete-event simulation | [README](queueing/README.md) |
| `information_theory/` | 31 | Entropy, divergences, mutual information, channels, coding | [README](information_theory/README.md) |
| `levy_processes/` | 33 | Jump-diffusion pricing, subordinators, Hawkes/branching processes, SDE solvers | [README](levy_processes/README.md) |
| `financial_stochastics/` | 50 | Option pricing & Greeks, stochastic/local vol, rate models, risk, credit, portfolio | [README](financial_stochastics/README.md) |
| `statistics/` | 48 | Estimation, hypothesis tests, regression, multivariate methods | [README](statistics/README.md) |
| `random_matrix/` | 23 | Classical ensembles, limit laws, Haar rotations, spectral statistics | [README](random_matrix/README.md) |
| `advanced_mcmc/` | 35 | MH/Gibbs/HMC/NUTS/SMC samplers, diagnostics, variational inference | [README](advanced_mcmc/README.md) |
| `numerical_methods/` | 38 | Quadrature, ODE/SDE solvers, linear algebra, root finding, PDE tools | [README](numerical_methods/README.md) |
| `bayesian/` | 25 | Conjugate/grid/Laplace/VI posteriors, Bayesian models, model selection | [README](bayesian/README.md) |
| `robust_statistics/` | 28 | Robust location/scale, high-breakdown regression, MCD/MVE/OGK, bootstraps | [README](robust_statistics/README.md) |
| `nonparametric/` | 31 | Density estimation, resampling/rank tests, dependence measures, local regression | [README](nonparametric/README.md) |
| `optimization/` | 32 | Gradient/quasi-Newton methods, metaheuristics, stochastic approximation, constrained solvers | [README](optimization/README.md) |
| `experimental_design/` | 29 | Classical/optimal/space-filling designs, response surfaces, sensitivity analysis | [README](experimental_design/README.md) |
| `spatial_statistics/` | 32 | Variograms, kriging, random fields, point processes, spatial autocorrelation | [README](spatial_statistics/README.md) |
| `viz/` | 35 | SVG-native statistical plots, matplotlib optional | [README](viz/README.md) |
| `utils/` | 38 | Seeding, parallel/GPU/JIT backends, reproducibility, data cleaning, interop | [README](utils/README.md) |

Package-level files:

- `__init__.py` — re-exports every subpackage; `__version__` lives here.
- `_rng.py` — the shared `random_state=` resolution every module routes through
  (`as_generator`/`spawn`); the public face is `utils.random`.
- `_parallel.py` — the shared `n_jobs=`/`backend=` chunking, pooling and
  thread/process dispatch behind `utils.ParallelSimulation` and the
  `montecarlo`/`financial_stochastics`/`advanced_mcmc` retrofit.
- `cli.py` — the `spl` console command: `--help` library inventory (generated
  from each module's `__all__`), `--version [--list]` with PyPI awareness,
  `--test` embedded self-check, and the `update` / `info` / `show` / `demo` /
  `cite` subcommands.
- `cli_pypi.py` — offline-safe PyPI metadata access (fetch, 24 h cache,
  version parsing, install-mode detection) behind `--version` and `update`.
- `cli_demo.py` — the twenty-three live mini-examples behind `spl demo <module>`.
- `selftest.py` — the 291-check self-check suite shipped inside the wheel,
  runnable from any pip install without pytest or a source checkout.

Target API for every planned module lives in the private Obsidian vault
(`../Stochpylib-Obsidian-Vault/Modules/<name>.md`); exact progress against the
full design spec is tracked in
[`../development/Implementation-Checklist.md`](../development/Implementation-Checklist.md).

Run all tests from the repo root: `pytest tests/ -v`.
