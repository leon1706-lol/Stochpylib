# stochpylib.robust_statistics

Outlier-resistant estimators: robust location (trimmed/winsorized means, median with a
Maritz-Jarrett standard error and a distribution-free order-statistic confidence interval,
Hodges-Lehmann, L/M/R-estimators), robust scale (MAD, Qn, Sn, IQR, and a Huber/biweight/
tau/Qn/Sn dispatcher), high-breakdown regression (Theil-Sen, Siegel repeated medians,
RANSAC, FAST-LTS, FAST-S+MM, Huber), robust covariance (FAST-MCD, MVE, OGK, robust
correlation), covariance shrinkage (Ledoit-Wolf, OAS, constant correlation), and resampling
(i.i.d./wild/block/stationary bootstraps). 28 public names across five submodules, natively
on numpy/scipy. Location/scale estimators and regressors reuse
`stochpylib.statistics.EstimateResult`/`RegressionResult` as result objects rather than
defining new ones.

**Status:** implemented & tested (28/28 spec names).

## Files

- `_common.py` — private helpers: seeded RNG, `_kth_pairwise`/`_kth_cross_diff` (exact for
  small n, an O(n log n) monotone bisection above it, used by Hodges-Lehmann/Qn/R-estimator
  confidence intervals), the M-estimator psi/weight family (Huber, Tukey biweight, Hampel,
  Andrews, Cauchy, Welsch, Fair, logistic), `_m_scale` (S-scale fixed point),
  `_huber_proposal2` (Huber 1981 joint location/scale, transcribed from
  `statsmodels.robust.scale.Huber` to match it exactly), `_tau_scale`/`_biweight_midvariance`
  (M-scales), `_chi2_consistency` (MCD/MVE/OGK raw-scatter correction), `_subsets`/
  `_weiszfeld` (randomized subsampling, spatial median).
- `_base.py` — `RobustEstimator` (location/scale: `fit(x)`, `estimate_`/`std_error_`,
  `confidence_interval()`, `to_result()` -> `EstimateResult`), `RobustRegressor`
  (`fit(X, y)`, `coef_` intercept-first, `predict`/`score`/`conf_int`/`to_result()` ->
  `RegressionResult`), `RobustCovarianceEstimator` (`location_`/`covariance_`,
  `mahalanobis()`, `outliers()`), `Resampler` (the shared base of the four bootstraps).
- `location.py` — `TrimmedMean`/`WinsorizedMean` (Tukey-McLaughlin SE), `Median`
  (Maritz-Jarrett SE, order-statistic CI), `HodgesLehmann` (one/two-sample, Hollander-Wolfe
  signed-rank/rank-sum CI), `L_Estimator` (trimean/Gastwirth/midhinge/trimmed/winsorized/
  callable/explicit weights), `M_Estimator` (Huber-family IRLS, H1 asymptotic SE, matches
  `statsmodels.RLM` exactly), `R_Estimator` (Wilcoxon score exactly equals `HodgesLehmann`;
  sign score exactly equals the median; normal/van-der-Waerden scores via root-finding).
- `scale.py` — `MedianAbsoluteDeviation`/`IQR_Scale` (match `scipy.stats` exactly),
  `Qn_Estimator`/`Sn_Estimator` (Rousseeuw & Croux 1993; `Qn_Estimator` matches
  `statsmodels.robust.scale.qn_scale` exactly -- no small-sample correction beyond the
  asymptotic constant, matching that oracle's own convention), `RobustStd` (dispatcher:
  mad/qn/sn/iqr/huber/biweight/tau).
- `regression.py` — `TheilSenRegression` (simple regression matches
  `scipy.stats.theilslopes` exactly including its Sen 1968 CI; multiple predictors via the
  spatial median of random `p+1`-point subset fits), `SiegalRegression` (+ `SiegelRegression`
  alias; matches `scipy.stats.siegelslopes` exactly; one predictor only), `RANSACRegression`
  (adaptive trial-count stopping; default residual threshold from a Theil-Sen pilot fit's
  MAD, not the raw response -- see Known limitations), `LTS_Regression` (FAST-LTS with exact
  h-subset enumeration below 5000 subsets), `MMRegression` (FAST-S scale-equivariant start +
  one M-step at fixed scale), `HuberRegression` (matches `statsmodels.RLM` with
  `M=HuberT()`, `scale_est="mad"`, `cov="H1"` exactly).
- `covariance.py` — `MCD`/`MVE` (exact subset enumeration below 5000 subsets, else FAST
  resampling with concentration/volume-shrinking steps; chi-square consistency-corrected,
  reweighted by default), `OGK` (Maronna & Zamar 2002 pairwise orthogonalization, composed
  over `n_iter` passes), `RobustCovariance`/`RobustCorrelation` (dispatchers; the rank
  correlations reuse `stochpylib.copulas._utils.kendall_tau_estimate`/
  `spearman_rho_estimate` rather than reimplementing them), `CovShrinkage` (Ledoit-Wolf
  toward identity/diagonal/constant-correlation targets, OAS, or a fixed intensity; the
  identity-target Ledoit-Wolf path matches `financial_stochastics.CovarianceEstimation`
  exactly).
- `bootstrap.py` — `RobustBootstrap` (any callable, `RobustEstimator` instance, or named
  statistic; percentile/basic/normal/BCa intervals), `WildBootstrap` (Rademacher/Mammen/
  normal/Webb multipliers on regression residuals, refitting OLS or any `RobustRegressor`),
  `BlockBootstrap` (moving/circular/nonoverlapping blocks), `StationaryBootstrap`
  (Politis-Romano geometric block lengths).

## Conventions

- **Result objects reused, not reinvented.** `RobustEstimator.to_result()` returns a
  `statistics.EstimateResult`; `RobustRegressor.to_result()` returns a
  `statistics.RegressionResult` — this module's siblings of `bayesian.Posterior`.
- **`.fit(X, y)`/`.fit(x)` returns `self`**, fitted attributes end in `_`, every stochastic
  method takes `random_state=`.
- **`coef_` is intercept-first** when `fit_intercept=True` (matching
  `statistics.RegressionResult`), so `coef_[1:]` are the slopes.
- **Exact enumeration below 5000 subsets, seeded resampling above it.** `MCD`/`MVE`/
  `LTS_Regression` and multi-predictor `TheilSenRegression` all follow this rule; `MCD`/`MVE`
  additionally apply the chi-square raw-scatter consistency correction (`_chi2_consistency`)
  both to the raw fit and to the reweighted refit.
- **Scale estimators report the sigma scale by default** (`normal=True` multiplies the raw
  statistic by its Gaussian consistency constant); pass `normal=False` for the raw statistic.

## Known limitations

- **`MCD`/`MVE`/`OGK`'s consistency correction is the plain chi-square factor, not the
  Pison/Van Aelst/Willems finite-sample refinement** — asymptotically correct, slightly
  conservative at very small n.
- **`SiegalRegression`/`TheilSenRegression`'s exact confidence interval is simple-regression
  only** (`p == 1`); multi-predictor `TheilSenRegression` reports `std_errors_` as `nan`.
- **`RANSACRegression`'s default `residual_threshold` comes from a Theil-Sen pilot fit's
  MAD**, not `MAD(y)` directly — for data where the response's own range reflects the
  regression slope rather than just noise, `MAD(y)` badly overstates the inlier band; pass
  `residual_threshold=` explicitly for full control.
- **`OGK` is not affine equivariant** by construction (a known property of the
  Gnanadesikan-Kettenring pairwise approach), unlike `MCD`/`MVE`.
- **`Sn_Estimator` is O(n^2)** (row-chunked); **`Qn_Estimator`/Hodges-Lehmann/two-sample
  R-estimator confidence intervals fall back to an O(n log n) monotone bisection above
  n = 1000** rather than materializing the full pairwise multiset.
- **`R_Estimator(score="normal")`'s two-sample shift falls back to `median(x) - median(y)`**
  when the pooled-rank statistic's bracket isn't sign-changing (rare, only for pathological
  or heavily tied data).
- **No autodiff or general M-estimator framework** — every `psi` family in `M_Estimator`/
  `MMRegression`/`HuberRegression` is a closed-form implementation, not a user-suppliable
  arbitrary loss.

Spec: vault `Modules/robust_statistics.md` (private). Tests:
`tests/robust_statistics/tests.py` (oracles: `scipy.stats`, `statsmodels`, brute-force
enumeration, `scipy.integrate`) and `tests/robust_statistics/e2e.py` (API sweep).
