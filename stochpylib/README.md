# stochpylib/

The installable package: one subpackage per library module behind a single
load-bearing contract — every distribution exposes the same method set, every
stochastic method takes `random_state=`, every Monte Carlo estimator returns a
shared result object. Nineteen subpackages live today (660/794 spec names):

| Subpackage | Spec names | What it owns | Guide |
|---|---|---|---|
| `probability/` | 21 | sample spaces, Bayes, exact combinatorics, independence | [README](probability/README.md) |
| `distributions/` | 60 | 47 distributions behind the common interface | [README](distributions/README.md) |
| `montecarlo/` | 25 | quasi-random sequences, estimators, variance reduction, applications | [README](montecarlo/README.md) |
| `timeseries/` | 61 | linear/volatility models, filters, changepoints, spectral analysis | [README](timeseries/README.md) |
| `gaussian_processes/` | 36 | composable kernel zoo, exact/sparse GP regression & classification | [README](gaussian_processes/README.md) |
| `copulas/` | 26 | elliptical/Archimedean/empirical copulas, vines, dependence measures | [README](copulas/README.md) |
| `survival/` | 28 | Kaplan-Meier, parametric fits, Cox/AFT/FineGray, competing risks | [README](survival/README.md) |
| `queueing/` | 29 | M/M/1 to Jackson networks, birth-death formulas, discrete-event simulation | [README](queueing/README.md) |
| `information_theory/` | 31 | entropy, divergences, mutual information, channels, coding | [README](information_theory/README.md) |
| `levy_processes/` | 33 | Lévy-Khintchine core, jump-diffusion pricing, subordinators, advanced point/branching/field processes, SDE solvers | [README](levy_processes/README.md) |
| `financial_stochastics/` | 50 | option pricing & Greeks, stochastic/local vol, short-rate models, risk, credit, portfolio construction | [README](financial_stochastics/README.md) |
| `statistics/` | 48 | descriptive stats, MLE/MOM/Bayesian/bootstrap estimation, hypothesis tests, OLS/GLM/penalized/quantile regression, multivariate methods | [README](statistics/README.md) |
| `random_matrix/` | 23 | GOE/GUE/GSE, Wigner, Wishart, CUE, Ginibre-type ensembles; semicircle/Marchenko-Pastur/Tracy-Widom laws; Haar rotations; spacing & edge statistics | [README](random_matrix/README.md) |
| `advanced_mcmc/` | 35 | MCMC samplers (MH, Gibbs, adaptive, HMC/NUTS/MALA/RMHMC/NeuTra, slice, tempering, SMC, particle, transdimensional), diagnostics, variational inference | [README](advanced_mcmc/README.md) |
| `numerical_methods/` | 38 | Gauss quadrature & adaptive/cubature integration, ODE/SDE solvers, native linear algebra (expm/logm, eigendecomposition, SVD, QR, Schur), root finding, interpolation, PDE tools (FD/FEM/BEM/spectral, FEniCS-style adapter) | [README](numerical_methods/README.md) |
| `bayesian/` | 25 | priors/likelihoods/posteriors, ten conjugate families, Bayesian linear/logistic regression, naive Bayes, hierarchical models, mixtures, Bayesian networks, Dirichlet processes, AIC/BIC/DIC/WAIC/PSIS-LOO/TIC & Bayes factors, Laplace/EP/VI/importance posteriors | [README](bayesian/README.md) |
| `robust_statistics/` | 28 | trimmed/winsorized means, median, Hodges-Lehmann, L/M/R-estimators, MAD/Qn/Sn/IQR/biweight/tau scales, Theil-Sen/Siegel/RANSAC/LTS/MM/Huber regression, MCD/MVE/OGK covariance, robust correlation, Ledoit-Wolf/OAS shrinkage, robust/wild/block/stationary bootstraps | [README](robust_statistics/README.md) |
| `nonparametric/` | 31 | kernel/adaptive/kNN/orthogonal-series/log-spline density estimation, empirical distribution/CDF/characteristic-function estimators, Glivenko-Cantelli & empirical likelihood, permutation/bootstrap/rank hypothesis tests, Spearman/Kendall/distance/Hoeffding dependence measures, local-polynomial/isotonic/spline/quantile regression | [README](nonparametric/README.md) |
| `optimization/` | 32 | gradient descent and the adaptive-step family, Newton/BFGS/L-BFGS/conjugate-gradient/trust-region/Levenberg-Marquardt, simulated annealing, genetic algorithms, particle swarm, differential evolution, ant colony, CMA-ES, GP-surrogate Bayesian optimization, Robbins-Monro/Kiefer-Wolfowitz/SPSA, cross-entropy method, sample-average approximation, penalty/augmented-Lagrangian/active-set/interior-point solvers | [README](optimization/README.md) |

Package-level files:

- `__init__.py` — re-exports every subpackage; `__version__` lives here.
- `cli.py` — the `spl` console command: `--help` library inventory (generated
  from each module's `__all__`), `--version [--list]` with PyPI awareness,
  `--test` embedded self-check, and the `update` / `info` / `show` / `demo` /
  `cite` subcommands.
- `cli_pypi.py` — offline-safe PyPI metadata access (fetch, 24 h cache,
  version parsing, install-mode detection) behind `--version` and `update`.
- `cli_demo.py` — the nineteen live mini-examples behind `spl demo <module>`.
- `selftest.py` — the 235-check self-check suite shipped inside the wheel,
  runnable from any pip install without pytest or a source checkout.

Target API for every planned module lives in the private Obsidian vault
(`../Stochpylib-Obsidian-Vault/Modules/<name>.md`); exact progress against the
full design spec is tracked in
[`../development/Implementation-Checklist.md`](../development/Implementation-Checklist.md).

Run all tests from the repo root: `pytest tests/ -v`.
