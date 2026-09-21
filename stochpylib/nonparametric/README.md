# stochpylib.nonparametric

Distribution-free methods: kernel/series density estimation, empirical distribution/CDF/
characteristic-function estimators and Glivenko-Cantelli bounds, Owen's empirical
likelihood, resampling and rank-based hypothesis tests, rank/distance dependence measures,
and local/isotonic/spline regression. 31 public names across five submodules, natively on
numpy/scipy. Density estimators satisfy the full 13-method distribution contract; hypothesis
tests and dependence measures reuse `stochpylib.statistics.TestResult`/`EstimateResult`
rather than defining new result types.

**Status:** implemented & tested (31/31 spec names).

## Files

- `_common.py` — private helpers: the six kernel functions (gaussian/epanechnikov/biweight/
  triangular/uniform/cosine) with their `R(K)`/`mu2(K)` constants, bandwidth rules
  (`scott`/`silverman` matching `scipy.stats.gaussian_kde`'s 1-D factors exactly,
  `normal_reference` matching `statsmodels.nonparametric.bandwidths`, grid-search `lscv`),
  weighted PAVA (`_pava_weighted`), B-spline knot/design-matrix construction, Owen's
  pseudo-log family (`_el_logstar*`), the DKW band, and pairwise-distance/double-/U-centering
  helpers for the distance-covariance family.
- `_base.py` — `NonparametricDensity` (subclasses `distributions.Distribution`; `fit(x)` is a
  fluent *instance* method, the one documented deviation from `Distribution.fit`'s classmethod
  contract, since these estimators carry bandwidth/kernel state), `NonparametricTest`
  (`fit(*samples)` → `TestResult`), `DependenceMeasure` (`fit(x, y)` → `EstimateResult`),
  `NonparametricRegressor` (`fit(X, y)`/`predict(X, return_std=)`).
- `density.py` — `KernelDensityEstimate` (1-D exact cdf/mean/var/rvs; d>1 diagonal product
  kernel), `AdaptiveKDE` (Abramson two-stage local bandwidths, `alpha=0` exactly equals
  `KernelDensityEstimate`), `NearestNeighborDensity` (k-NN density, truncated-and-renormalized
  for the distribution contract), `OrthogonalSeriesDensity` (cosine-basis series with
  Kronmal-Tarter term selection), `LogsplineEstimator` (Kooperberg-Stone log-spline MLE via
  Newton + Gauss-grid quadrature for the normalizing constant).
- `empirical.py` — `EmpiricalDistribution` (full distribution contract over sample atoms),
  `EmpiricalCDF` (fluent step-function, DKW band), `EmpiricalCharFn` (empirical
  characteristic function + distance to a reference cf), `GlivenkoCantelli` (sup-norm
  distance = the KS statistic + DKW bound + a convergence-rate helper), `EmpiricalLikelihood`
  (Owen's EL for a vector mean, Newton on the dual, matches
  `statsmodels.emplike.DescStat` exactly).
- `tests.py` — `PermutationTest` (independent/pairings/samples permutation types, exact
  enumeration below 20000 permutations else Monte Carlo, two-sided p-value = `2 *
  min(p_greater, p_less)` matching `scipy.stats.permutation_test` exactly),
  `BootstrapTest` (Efron-Tibshirani recentered bootstrap), `MoodTest` (scale/median,
  match `scipy.stats.mood`/`median_test` exactly), `KruskalWallis` (+ `posthoc_dunn`,
  matches `scipy.stats.kruskal` exactly), `FriedmanTest` (matches
  `scipy.stats.friedmanchisquare` exactly, tie-corrected), `SignTest`, `RunsTest`/
  `WaldWolfowitz` (match `statsmodels.sandbox.stats.runs` exactly), `AndersenDarling`
  (`dist="norm"`/`"expon"` match `scipy.stats.anderson` exactly; a fully-specified
  reference distribution uses an approximate case-0 critical-value table; `k_sample=True`
  matches `scipy.stats.anderson_ksamp(variant="midrank")` exactly), `CramerVonMises`
  (one-sample statistic matches `scipy.stats.cramervonmises` exactly, asymptotic p-value;
  two-sample matches `scipy.stats.cramervonmises_2samp(method="asymptotic")` exactly).
  `AndersonDarling` is the correctly-spelled alias of `AndersenDarling` (the spec's own
  spelling).
- `correlation.py` — `SpearmanCorrelation`/`KendallTau` (reuse
  `copulas._utils.spearman_rho_estimate`/`kendall_tau_estimate`; match `scipy.stats`
  exactly, including the exact small-sample Kendall test), `RankCorrelation` (dispatcher
  + Goodman-Kruskal gamma + Somers' D matching `scipy.stats.somersd` exactly),
  `DistanceCorrelation` (Szekely-Rizzo, biased or U-centered unbiased estimator),
  `BrownianCorrelation` (fractional-BM generalization, `hurst=0.5` exactly equals
  `DistanceCorrelation`), `HoeffdingD` (Hollander-Wolfe/Hmisc formula).
- `regression.py` — `LocalPolynomialReg` (degree 0/1 = Nadaraya-Watson/local-linear,
  matches `statsmodels.nonparametric.kernel_regression.KernelReg` exactly at a fixed
  bandwidth; GCV bandwidth selection), `IsotonicRegression` (weighted PAVA, matches
  `scipy.optimize.isotonic_regression` exactly), `SplineRegression` (`"pspline"`:
  B-spline + 2nd-difference penalty with GCV `lam`; `"smoothing"`: matches
  `scipy.interpolate.make_smoothing_spline` exactly), `GPR_Nonparametric` (facade over
  `gaussian_processes.GPRegression`), `QuantileRegression` (`"local"`: kernel-weighted
  check-loss LP with relative-weight truncation for tractable per-query cost; `"linear"`
  delegates to `statistics.quantile_regression`).

## Conventions

- **Density estimators are full distributions.** Every class in `density.py`/
  `EmpiricalDistribution` satisfies `.pdf()/.cdf()/.ppf()/.rvs()/.mean()/.var()/
  .skewness()/.kurtosis()/.entropy()/.mgf()/.cf()/.fit()/.ks_test()` via
  `NonparametricDensity`/`Distribution`'s generic fallbacks plus per-class closed forms;
  `fit(x)` is a fluent instance method rather than `Distribution`'s classmethod.
- **Result objects reused, not reinvented.** `NonparametricTest.to_result()` returns a
  `statistics.TestResult`; `DependenceMeasure.to_result()` returns an `EstimateResult`.
- **Every stochastic method takes `random_state=`.**
- **Kernel locality is exploited for tractability, not accuracy.** `QuantileRegression`'s
  local method drops points whose kernel weight is below `1e-6` of the query's maximum
  weight before solving the per-point LP — exact for a huge bandwidth (nothing is below
  threshold) and a bounded-cost approximation otherwise.

## Known limitations

- **`NearestNeighborDensity`'s raw k-NN estimate does not integrate to 1** over an
  unbounded support; `pdf`/`cdf`/moments operate on a truncated-and-renormalized version
  (support padded 3 sample standard deviations beyond the data range). Pointwise bias
  near the mode is higher than for kernel methods at comparable `k`; use the default
  contract methods for shape/tail behavior, not exact pointwise density values.
- **`LogsplineEstimator` has no knot-deletion search** — a fixed, evenly-spaced knot count
  (`n_knots`), unlike the adaptive-knot-selection Kooperberg-Stone algorithm.
- **`AndersenDarling` with a fully-specified reference distribution** (a `Distribution`
  instance or callable CDF, as opposed to `dist="norm"`/`"expon"`) uses an *approximate*
  p-value from the case-0 asymptotic critical-value table (D'Agostino & Stephens 1986,
  Table 4.2) via log-linear interpolation, not an exact closed-form p-value.
- **`CramerVonMises`'s one-sample p-value is the asymptotic (`n -> infinity`) null
  distribution**, not `scipy.stats.cramervonmises`'s finite-sample correction term — the
  statistic itself matches exactly; the two p-values agree closely for moderate/large `n`
  (the same documented asymptotic-vs-finite-sample deviation as `statistics.ks_test`).
- **`RankCorrelation(method="gamma")`'s tie handling is the simplified convention**
  (pairs tied on both variables are double-subtracted from the pair count) — exact for
  continuous (tie-free) data, an approximation otherwise.
- **`QuantileRegression`'s `"local"` method is O(active points) per query, no
  tree-based neighbor search** — the relative-weight truncation keeps per-query cost
  bounded for typical bandwidths, but a bandwidth comparable to the data's whole range
  still includes most points in every local LP.
- **`GPR_Nonparametric` is a thin facade, not a distinct algorithm** — it reuses
  `gaussian_processes.GPRegression` verbatim (see that module's own known limitations,
  e.g. no autodiff-based hyperparameter gradients).

Spec: vault `Modules/nonparametric.md` (private). Tests: `tests/nonparametric/tests.py`
(oracles: `scipy.stats`, `statsmodels`, `scipy.optimize`/`scipy.interpolate`, brute-force
enumeration) and `tests/nonparametric/e2e.py` (API sweep).
