# stochpylib.robust_statistics

*Outlier-resistant estimators*

**Design-completeness score:** 8/10 — Huber, RANSAC, Theil-Sen — essential for real-world data

Status: **planned** — not yet implemented.

## Submodules

### `robust_statistics.location`

- `TrimmedMean`
- `WinsorizedMean`
- `Median`
- `HodgesLehmann`
- `L_Estimator`
- `M_Estimator`
- `R_Estimator`

### `robust_statistics.scale`

- `MedianAbsoluteDeviation`
- `Qn_Estimator`
- `Sn_Estimator`
- `RobustStd`
- `IQR_Scale`

### `robust_statistics.regression`

- `TheilSenRegression`
- `RANSACRegression`
- `LTS_Regression`
- `MMRegression`
- `HuberRegression`
- `SiegalRegression`

### `robust_statistics.covariance`

- `RobustCovariance`
- `MCD`
- `MVE`
- `OGK`
- `RobustCorrelation`
- `CovShrinkage`

### `robust_statistics.bootstrap`

- `RobustBootstrap`
- `WildBootstrap`
- `BlockBootstrap`
- `StationaryBootstrap`

---

[[Module-Map]] · [[Ratings]] · [[ARCHITECTURE]]
