# stochpylib.statistics

Classical statistical toolkit: descriptive statistics, point/interval
estimation (MLE/MOM/Bayesian conjugate/bootstrap/jackknife/delta method/
profile likelihood), hypothesis tests (z/t/chi2/F/ANOVA/MANOVA/rank tests/
normality/multiple comparisons), regression (OLS/GLM/ridge/lasso/elastic
net/quantile), and multivariate methods (PCA/factor analysis/canonical
correlation/discriminant analysis/clustering/MDS). 48 public names across
five submodules, natively on numpy/scipy — no wrapper dependencies.

**Status:** implemented & tested (48/48 spec names).

**Note:** the module name shadows the Python standard library's `statistics`
module if imported bare from inside this package's own directory — always
`import stochpylib.statistics` (or `from stochpylib import statistics`),
never a plain `import statistics` from code that also needs the stdlib
module in the same scope.

## Files

- `_common.py` — private shared helpers: normal/Student-t/chi2/F CDFs and
  quantiles via `scipy.special` (never `scipy.stats`), the Kolmogorov
  survival function, a from-scratch studentized-range CDF/quantile
  (`_ptukey`/`_qtukey`, log-space Gauss-Legendre double quadrature, matches
  `scipy.stats.studentized_range` to ~1e-11), numerical gradient/Jacobian/
  Hessian, and a jittered Cholesky factor.
- `_result.py` — `TestResult`, `EstimateResult`, `DescribeResult`,
  `RegressionResult`, and six multivariate result classes.
- `descriptive.py` — `mean`/`median`/`mode`/`variance`/`std`/`quantile`
  (all nine Hyndman-Fan interpolation types)/`iqr`/`skewness`/`kurtosis`/
  `covariance`/`correlation` (Pearson/Spearman/Kendall tau-b, tie-corrected)/
  `describe`.
- `estimation.py` — `MLE` (any `Distribution` subclass or a
  `loglik(params, data)` callable; Hessian-based standard errors), `MOM`
  (closed forms for 8 named distributions, least-squares fallback for any
  other), `bayesian_estimator` (normal-normal/beta-binomial/gamma-Poisson/
  gamma-exponential conjugate families, or a generic grid posterior),
  `confidence_interval` (mean/proportion/variance/median/mean-difference/
  correlation), `bootstrap_ci` (percentile/basic/normal/BCa),
  `jackknife`, `delta_method`, `profile_likelihood` (Wilks' theorem).
- `hypothesis.py` — `z_test`, `t_test`, `chi2_test`, `f_test` (variance
  ratio or nested-model), `ANOVA` (one-way, Welch, or two-way Type-II SS via
  nested-model comparison), `MANOVA` (Wilks/Pillai/Hotelling-Lawley/Roy with
  their F-approximations), `mann_whitney`/`wilcoxon` (exact DP null
  distributions or tie-corrected asymptotic normal), `ks_test`,
  `shapiro_wilk` (Royston 1992 AS R94), `levene`/`bartlett`, `tukey_hsd`,
  `bonferroni` (bonferroni/holm/sidak/holm-sidak/fdr_bh).
- `regression.py` — `linear_regression` (OLS/WLS, nonrobust or HC0-HC3
  sandwich SEs), `glm` (IRLS; gaussian/binomial/poisson/gamma/
  inverse_gaussian/negative_binomial families x 8 links), `logistic_regression`/
  `poisson_regression` (GLM facades), `ridge` (closed-form SVD, optional
  GCV), `lasso`/`elastic_net` (coordinate descent, warm-started paths),
  `quantile_regression` (exact LP + Koenker-Bassett kernel sandwich SEs).
- `multivariate.py` — `PCA`, `factor_analysis` (ML via profiled Jöreskog
  concentration, or iterated principal axis; varimax/quartimax rotation),
  `canonical_correlation` (with Wilks sequential-dimensionality tests),
  `discriminant_analysis` (LDA/QDA, direct Gaussian Bayes classification),
  `cluster_analysis` (k-means++ or agglomerative Lance-Williams linkage),
  `MDS` (classical Torgerson or SMACOF, metric or non-metric).

## Conventions

- Every function that returns a fitted/tested quantity returns one of the
  ten shared result classes (`TestResult`, `EstimateResult`,
  `RegressionResult`, `DescribeResult`, or a multivariate result) —
  point estimate + standard error/p-value + confidence interval, matching
  `stochpylib`'s library-wide convention (`MCResult`, `ForecastResult`).
  `TestResult.table` carries multi-row outputs (ANOVA/MANOVA/pairwise/
  multiple-testing) as a list of dicts.
- `alternative` (`"two-sided"`/`"greater"`/`"less"`) is consistent across
  every test that supports directional alternatives.
- Every stochastic function (`bootstrap_ci`, `cluster_analysis`,
  `MDS(method="smacof")`) takes `random_state=`.
- Library code never imports `scipy.stats` — every distribution CDF/PPF is
  built from `scipy.special` directly; `scipy.stats`/`scipy.cluster`/
  `statsmodels` are the test suite's independent oracles only.
- `spl show TestResult` (and `EstimateResult`, etc.) resolves to this
  module's class, since `cmd_show` walks `stochpylib.__all__` in order and
  `statistics` is the only module exporting those names.

## Known limitations

- `shapiro_wilk` is accurate for `3 <= n <= 5000` (Royston's polynomial
  approximations are not calibrated outside that range) and raises above it.
- `mann_whitney`/`wilcoxon`'s exact null distributions require no ties (or
  zeros, for Wilcoxon) in the data and are computed via dynamic programming
  over Python big integers — fast for the library's documented `n <= 50`
  range, but not intended for much larger exact computations.
- `ks_test`'s two-sample p-value uses the classical asymptotic (`N ->
  infinity`) Kolmogorov formula (matching R's `ks.test` and this module's
  own one-sample case exactly) rather than `scipy.stats.ks_2samp`'s more
  refined finite-sample `kstwo` correction, which has no `scipy.special`
  equivalent; the two converge for large samples and always agree on the
  statistic itself.
- `glm`'s IRLS is not guaranteed to converge for family/link combinations
  outside each family's natural domain (e.g. `family="poisson",
  link="identity"`) — `statsmodels` documents the same caveat for those
  pairings. Use each family's canonical (or another domain-respecting) link
  for guaranteed convergence.
- `factor_analysis(method="ml")`'s BFGS optimizer, like `statsmodels`' own,
  is not guaranteed to converge to the global optimum on ill-conditioned
  correlation matrices (near-Heywood cases, communalities close to 1).
- `MDS(method="smacof", metric=False)`'s non-metric fit (isotonic
  regression inside stress majorization) can converge to a local optimum;
  run with several `random_state` seeds and keep the lowest-stress result
  for a production fit.

Spec: vault `Modules/statistics.md` (private). Tests:
`tests/statistics/tests.py`.
