# Changelog — Build Phases & Development Walkthrough

Stochpylib was built incrementally across the following phases. This is the full
development history; see [`README.md`](../README.md) for current usage docs and
[`Probleme.md`](Probleme.md) for the audit log of bugs found and fixed.

## Phase 1 — Vault digitization
- **Design spec converted to Obsidian vault** at `Stochpylib-Obsidian-Vault/`: module map, one spec file per module (~794 public names), ratings, quickstart examples, dependencies, architecture, and essential-tasks checklist.

## Phase 2 — Naming and package scaffold
- **Renamed to `stochpylib`** after discovering `stochpy` and variants were taken on PyPI; propagated through every vault file.
- **Scaffolded the package**: `pyproject.toml`, `.gitignore`, README, CI workflow (test matrix), and publish workflow (tag-triggered PyPI Trusted Publisher).

## Phase 3 — First module: `stochpylib.probability`
- **Implemented 21 public names** across sample spaces, combinatorics (exact integer arithmetic), and independence checks.
- **Two math bugs** caught by doctests before shipping (Probleme #1-2).

## Phase 4 — Test relocation, progress tracking, dev-folder setup
- **Tests moved out of the package** to `tests/<module>/tests.py` so they never ship in the wheel.
- **Implementation-Checklist.md** created as the single source of truth for spec conformance.
- **`development/` folder** created for changelog, problems, and dev docs.

## Phase 5 — Second module: `stochpylib.distributions` + `spl` CLI
- **47 distribution classes** implemented with the full 13-method contract (pdf/cdf/ppf/rvs/moments/entropy/mgf/cf/fit/ks_test).
- **Four library bugs** found during scipy cross-check audit and fixed (Probleme #5-8).
- **`spl` CLI** created with `--version` and `--test` (embedded self-check suite).
- **81/794 public names.**

## Phase 8 — Third module: `stochpylib.montecarlo`
- **25 spec names**: Halton/Faure/Sobol/Niederreiter QMC with programmatically-verified GF(2) generator polynomials, crude/QMC/stratified/importance/rejection estimators, variance-reduction toolkit, and applications (integration, option pricing, VaR/ES, reliability).
- **Two construction bugs** caught by exactness checks (Probleme #10); known limitation on (t,m,s)-net certification logged (#11).
- **106/794 public names.**

## Phase 6 — Open-source hygiene
- **Community layer**: CONTRIBUTING, Code of Conduct, SECURITY, issue/PR templates.
- **Release workflow** creating GitHub Releases on every tag.
- **Latent bug fixed**: publish.yml ran pytest against the old in-package test location.

## Phase 7 — `spl --help` library overview
- **CLI help** now prints a full inventory of implemented modules, distribution classes, common interface, and quickstart — dynamically from `__all__` so it never goes stale.

## Phase 9 — Resolving documented open items
- **Joe-Kuo direction numbers** embedded (64 dims x 30 cols) for exact Sobol nets (Probleme #11).
- **Alpha=1 skewed stable sampling** replaced with a cached numerical quantile table after a failed closed-form hunt (Probleme #12).

## Phase 10 — Fourth module: `stochpylib.timeseries`
- **61 spec names across nine submodules**: ARMA/SARIMA/ARFIMA, VAR/VARMA/VECM, GARCH family, Kalman/RTS/EKF/UKF/particle filters, HMM, changepoint detection, spectral methods, forecasting diagnostics.
- **Eleven construction bugs** caught by smoke tests (Probleme #13-19).
- **Fluent `.fit()` and `ForecastResult`** conventions introduced.
- **167/794 public names. Version 0.2.0.**

## Phase 11 — Fifth module: `stochpylib.gaussian_processes`
- **36 spec names**: 10-kernel zoo with operator overloading, kernel ops, exact/sparse/deep GP models, Laplace/EP/VI classification, hyperparameter optimization.
- **Two construction bugs** caught during implementation (Probleme #21-22).
- **203/794 public names.**

## Phase 12 — Library audit: completing GP delivery + stability fixes
- **GPClassification/SparseGaussianProcess/InducingPointGP** added (36/36 spec names complete); module wired into package root.
- **Three real defects** fixed: broken duplicate FITC/VFE copy, sparse posterior exploding for large inducing counts (rewritten in whitened parameterization), and `BaseKernel.diag` crashing for most kernels (Probleme #23-24).
- **203/794 public names.**

## Phase 13 — Sixth module: `stochpylib.copulas`
- **26 spec names**: elliptical (recursive-integration CDF), Archimedean (generator framework, exact bivariate densities), empirical, vines (C/D/R with AIC selection), and methods (fit, sample, tail dependence).
- **Six construction defects** caught and fixed (Probleme #25-30).
- **229/794 public names. Version 0.3.0.**

## Phase 14 — V0.3.1 audit: spec conformance + cross-module tests + doc sync
- **Cross-module test suite** created (20 cases) verifying spec-name conformance and end-to-end workflows.
- **Selftest extended** to 130 checks.
- **Documentation synced** across all READMEs and development docs.

## Phase 15 — Seventh module: `stochpylib.survival`
- **28 spec names**: Kaplan-Meier, Nelson-Aalen, parametric fits with censored likelihood, Cox PH, AFT, Fine-Gray, log-rank family, competing risks.
- **257/794 public names. Version 0.4.0.**

## Phase 16 — Eighth module: `stochpylib.queueing`
- **29 spec names**: M/M/1, M/M/c, M/D/1, M/G/1, GI/G/1, priority queues, birth-death, Jackson/closed/BCMP networks, discrete-event simulation, Little's law.
- **287/794 public names. Version 0.5.0.**

## Phase 17 — V0.5.1 audit: bug fixes + edge-case tests + doc sync
- **Four survival bugs** fixed (step-evaluator default, integration grid, hazard wrapper, Gompertz overflow — Probleme #31-34).
- **20 new edge-case and cross-module tests** added.
- **406 passed / 2 skipped.**

## Phase 18 — Ninth module: `stochpylib.information_theory`
- **31 spec names**: entropy family, divergences, mutual information, channel capacity, Huffman coding, typical sets.
- **317/794 public names (progress accounting corrected). Version 0.6.0.**

## Phase 19 — V0.6.1 audit: bug fixes + edge-case tests
- **Two information-theory bugs** fixed: InformationGain using raw labels instead of counts, and Renyi alpha=0 in nats instead of bits (Probleme #35-36).
- **12 new edge-case tests** added.

## Phase 20 — V0.6.2 documentation overhaul
- **All nine module READMEs** rewritten to a common template; main README rebuilt with status table, known limitations, and quickstart snippets.
- **Doc-consistency suite** created (14 cases) enforcing that badges, counts, versions, and links never drift.
- **Progress accounting error** found and corrected (288/794 was arithmetically wrong; true total 317/794 — Probleme #37).
- **spl --help gap** fixed (missing queueing/information_theory blocks — Probleme #38).

## Phase 21 — V0.6.3 CI fix
- **Version-literal test** made bump-proof after it broke CI on every version bump (Probleme #39).
- **README CLI reference** gained TOC sub-links.

## Phase 22 — V0.6.4 CLI expansion: `spl update`, `info`, `show`, `demo`, `cite`
- **spl --version** gained PyPI awareness with offline-safe update checking.
- **spl update** added for switching pip versions.
- **spl info** added for environment report.
- **spl show** added for looking up any public name.
- **spl demo** added with nine deterministic mini-examples.
- **spl cite** added for plain-text + BibTeX citation.
- **New CLI test suite** (50 cases) with zero network access.
- **Version 0.6.4.**

## Phase 23 — V0.7.0 `levy_processes`
- **33 spec names**: Levy-Khintchine core, subordinators (gamma, inverse-Gaussian, stable, tempered), jump-diffusion pricing (Merton, Kou, Bates), advanced point processes (Hawkes, Cox, branching), SDE solvers (EM through strong-order 1.5).
- **Eleven bugs** found and fixed while writing tests (Probleme #41-51), including a Carr-Madan log-strike convention error and a biased truncated-jump quantile grid.
- **Version 0.7.0.**

## Phase 24 — CI hotfix: statsmodels 0.15.0 compatibility
- **Dropped `old_names` kwarg** from AutoReg oracle call (Probleme #52).

## Phase 25 — V0.8.0 `financial_stochastics`
- **50 spec names**: Black-Scholes, American (BAW + binomial), lattices, MC pricing (incl. Asian), Longstaff-Schwartz, Fourier methods, Greeks (closed-form/MC/FD), stochastic vol (Heston, SABR, rough Heston/Bergomi, local vol), rate models (Vasicek, CIR, Hull-White, HJM, LMM), risk (VaR, ES, stress testing), credit (CDS, Merton, migration, copula), portfolio optimization.
- **Ten bugs** found and fixed (Probleme #53-62), including a COS truncation-range error pricing a call at 2.7e19 and an inverted SABR skew factor.
- **Version 0.8.0.**

## Phase 26 — V0.9.0 `statistics`
- **48 spec names**: descriptive stats (weighted/trimmed, all nine Hyndman-Fan quantile types), estimation (MLE, MOM, Bayesian, bootstrap, jackknife, profile likelihood), hypothesis tests (z/t/chi2/F, ANOVA, MANOVA, nonparametric), regression (OLS/WLS/GLM/ridge/lasso/quantile), multivariate (PCA, factor analysis, discriminant, clustering, MDS).
- **Seven bugs** found and fixed (Probleme #63-69), including IRLS inverting the link derivative and OLS/GLM AIC over-counting parameters.
- **Version 0.9.0.**

## Phase 27 — V0.10.0 `random_matrix` + CI restructure
- **23 spec names**: GOE/GUE/GSE, Wishart, CUE, empirical spectra (semicircle, Marchenko-Pastur, Tracy-Widom), random rotations, eigenvalue statistics.
- **End-to-end API sweeps** (`tests/<module>/e2e.py`, 486 exercises across 13 modules) created with a guard that fails if any public name ships without one — caught **twelve shipped bugs** (Probleme #71-82).
- **CI restructured**: per-module `smoke` matrix, `fail-fast: false`, `cross-suite`, `install-smoke` wheel verification.
- **Version 0.10.0.**

## Phase 28 — V0.11.0 `advanced_mcmc`
- **35 spec names**: Metropolis-Hastings, Gibbs, adaptive variants, HMC, NUTS, MALA, manifold MALA, Riemannian HMC, Neutra-HMC, slice sampling, elliptical slice, replica exchange, parallel tempering, SMC, particle MCMC, reversible-jump, transdimensional, diagnostics (R-hat, ESS, Geweke, Raftery-Lewis), variational inference (mean-field, ADVI, black-box, normalizing flows, Stein VI).
- **Two bugs** found while testing (Probleme #83-84): NUTS never counting divergences, and rank-normalization using the wrong Blom denominator.
- **Version 0.11.0.**

## Phase 29 — V0.12.0 `numerical_methods`
- **38 spec names**: Gaussian quadrature (Golub-Welsch), adaptive integration, cubature, ODE solvers (Euler through Dormand-Prince, Adams, BDF), SDE solvers, matrix exponential/logarithm, decompositions (Cholesky, eigen, SVD, QR, Schur), root finding (Brent, Newton, fixed-point), interpolation (splines, PCHIP, barycentric, Chebyshev, NURBS), PDE tools (FEM, finite difference, spectral).
- **Five bugs** found and fixed (Probleme #85-89), including FEM silently discarding P2 DOFs and a complex-Schur reconstruction failure.
- **Version 0.12.0.**

## Phase 30 — V0.13.0 `bayesian`
- **25 spec names**: prior/likelihood framework, conjugate engine (10 families), posterior computation (grid, Laplace, EP, importance, SMC, MCMC), model selection (AIC/BIC/DIC/WAIC/LOO/TIC/Bayes factor), Bayesian models (linear, logistic, naive Bayes, hierarchical, mixtures, networks, Dirichlet process).
- **Three overflow bugs** in existing distributions fixed (Probleme #90-92): NegBinomial, Gamma, BetaBinomial all overflowed for large conjugate-posterior shapes.
- **Post-push CI fix**: GLM IRLS domain violation + flaky test seed (Probleme #93).
- **Version 0.13.0.**

## Phase 31 — V0.14.0 `robust_statistics`
- **28 spec names**: robust location (trimmed/Winsorized mean, median, Hodges-Lehmann, L/M/R estimators), robust scale (MAD, Qn, Sn, IQR), robust regression (Theil-Sen, Siegal, RANSAC, LTS, MM, Huber), robust covariance (MCD, MVE, OGK, shrinkage), robust bootstrap (percentile, wild, block, stationary).
- **Four bugs** found and fixed (Probleme #94-97).
- **Version 0.14.0.**

## Phase 32 — V0.15.0 `nonparametric`
- **31 spec names**: density estimation (KDE, adaptive KDE, k-NN, orthogonal series, log-spline), empirical methods (ECDF, empirical likelihood), resampling/rank tests (permutation, bootstrap, Mood, Kruskal-Wallis, Friedman, sign, runs, Anderson-Darling, Cramer-von Mises), dependence measures (Spearman, Kendall, distance correlation, Brownian, Hoeffding), local regression (local polynomial, isotonic, spline, GP, quantile).
- **Four bugs** found and fixed (Probleme #98-101), including a two-sample Cramer-von Mises statistic off by orders of magnitude.
- **Version 0.15.0.**

## Phase 33 — V0.16.0 `optimization`
- **32 spec names**: gradient methods (GD, SGD, AdaGrad, RMSProp, Adadelta, Adam, Nadam, AMSGrad), second-order (Newton, BFGS, LBFGS, CG, trust-region, Levenberg-Marquardt), metaheuristics (simulated annealing, genetic algorithm, PSO, differential evolution, CMA-ES, Bayesian optimization, ant colony), stochastic approximation (Robbins-Monro, Kiefer-Wolfowitz, SPSA, CEM, SAA), constrained (penalty, augmented Lagrangian, Lagrangian relaxation, active-set, interior-point).
- **Six bugs** found and fixed (Probleme #103-108), including Lagrangian dual ascent climbing the wrong way.
- **CI green-up**: Anderson-Darling critical values unpinned from version-changing scipy internals (Probleme #102).
- **Version 0.16.0.**

## Phase 34 — V0.17.0 `experimental_design`
- **29 spec names**: classical designs (full/fractional factorial, Plackett-Burman, CCD, Box-Behnken, Latin/Graeco-Latin squares), optimal designs (D/A/G/I/T, Bayesian), space-filling (LHD, maximin, minimax, uniform, orthogonal array), response surfaces (polynomial, RSM-ANOVA, polynomial chaos, kriging surrogate), analysis (ANOVA, main effects, interaction plots, sensitivity/Sobol indices).
- **Two existing library bugs** fixed (Probleme #109-110): GP hyperparameter optimization never moving Matern kernels, and MCResult confidence intervals wrong for non-0.95 levels.
- **Two construction bugs** caught before shipping (Probleme #111-112).
- **Post-push CI fix**: flaky MetaModel test with unseeded 3-fold split (Probleme #113).
- **Version 0.17.0.**

## Phase 35 — V0.18.0 `spatial_statistics`
- **32 spec names**: variograms (spherical/exponential/gaussian/Matern), kriging (simple/ordinary/universal/co/indicator/disjunctive), random fields (Gaussian, Matern, OU, Brownian/fractional-Brownian sheets), point processes (Poisson, inhomogeneous, Thomas, Matern cluster, LGCP), spatial autocorrelation tests (Moran's I, Geary's c, nearest-neighbor), CAR/SAR lattice models.
- **scipy.stats removed from all 9 remaining library files**; new AST guard prevents return (Probleme #114).
- **Six real bugs** fixed (Probleme #115-119), including a variogram fit reporting physically meaningless range on weak data.
- **`spl show` disambiguation** added for names exported by multiple modules.
- **Version 0.18.0.**

## Phase 36 — V0.19.0 `viz`
- **35 spec names**: SVG-native statistical plots (zero-dependency renderer) with optional matplotlib backend. Submodules: distributions, processes (ACF/PACF, periodogram, spectrogram, wavelet, trajectory), diagnostics (MCMC trace/posterior, regression residual/leverage/influence, funnel), multivariate (heatmap, correlation, copula, scatter matrix, biplot, dendrogram), special (Markov chain, Brownian/GBM, GP, survival KM, variogram, eigenvalues).
- **Cross-module hooks**: `InteractionPlot`/`NormalPlot` gained `to_figure()`.
- **Two bugs** found and fixed (Probleme #120-121): CWT default scale range crashing, and plot_markov_chain dropping self-loops.
- **`viz-matplotlib` CI job** created (the only place matplotlib is installed).
- **Version 0.19.0.**

## Phase 37 — V0.20.0 `utils` + library-wide RNG/parallel retrofit
- **38 spec names**: seed/stream management (`stochpylib._rng`), benchmarking/profiling, parallel/GPU/JIT backends (`stochpylib._parallel`), reproducibility (version lock, environment capture, experiment logger), data utilities (fit, goodness-of-fit, validation, imputation), serialization, interop (numpy/scipy/pandas/torch/jax).
- **794/794 public names complete** — all 23 modules shipped.
- **Every module's `random_state=`** now routes through the shared `_rng` helper; ~130 inline `default_rng` calls replaced; new AST guard enforces this.
- **`n_jobs=`/`parallel_backend=`** retrofitted onto every MC function returning `MCResult`; golden-stream harness verified reproducibility.
- **Optional backends** (pandas/torch/jax/numba/cupy) lazily imported in `utils/_backends.py` only, mirroring viz's matplotlib discipline.
- **Seven real bugs** found and fixed (Probleme #122-128).
- **Version 0.20.0.**

## Phase 38 — V0.20.1 test-infrastructure fixes
- **Three test-only bugs** fixed (Probleme #129-131): torch-backend tests assuming torch is installed, JIT backend test checking before first call, and jax-interface test not hiding jax.
- **One flaky test** fixed with MCSE-based threshold (Probleme #132).
