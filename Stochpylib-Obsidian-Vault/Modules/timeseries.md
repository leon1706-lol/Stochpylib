# stochpylib.timeseries

*Complete time series toolkit*

**Design-completeness score:** 10/10 — AR to GARCH family, state-space, HMM, wavelets, spectral — nothing missing

Status: **planned** — not yet implemented.

## Submodules

### `timeseries.linear_models`

- `AR(p)`
- `MA(q)`
- `ARMA(p,q)`
- `ARIMA(p,d,q)`
- `SARIMA(p,d,q)(P,D,Q,s)`
- `ARFIMA(p,d,q)`
- `VARMA`
- `VAR(p)`
- `VECM`

### `timeseries.volatility_models`

- `ARCH(q)`
- `GARCH(p,q)`
- `EGARCH`
- `TGARCH`
- `FIGARCH`
- `APARCH`
- `IGARCH`
- `GJRGARCH`
- `MGARCH`
- `DCC_GARCH`

### `timeseries.state_space`

- `StateSpaceModel`
- `KalmanFilter`
- `KalmanSmoother`
- `ExtendedKalmanFilter`
- `UnscentedKalmanFilter`
- `ParticleFilter`
- `RaoBlackwellFilter`

### `timeseries.latent_models`

- `HiddenMarkovModel`
- `SwitchingRegression`
- `RegimeSwitching`
- `MixtureAutoregressive`

### `timeseries.changepoint`

- `ChangePointDetection`
- `BayesianChangePoint`
- `PELT`
- `BinarySegmentation`
- `BottomUp`

### `timeseries.decomposition`

- `SeasonalDecomposition`
- `STLDecomposition`
- `X11Decomposition`
- `TrendFilter`
- `HPFilter`

### `timeseries.spectral`

- `SpectralAnalysis`
- `Periodogram`
- `PowerSpectrum`
- `WaveletTransform`
- `CWTTransform`
- `DWTTransform`
- `STFT`
- `Hilbert`

### `timeseries.tests`

- `adf_test()`
- `kpss_test()`
- `pp_test()`
- `ljung_box()`
- `durbin_watson()`
- `arch_test()`
- `granger_causality()`
- `johansen_test()`

### `timeseries.forecasting`

- `forecast()`
- `predict()`
- `confidence_bands()`
- `backtesting()`
- `cross_validation_ts()`

---

[[Module-Map]] · [[Ratings]] · [[ARCHITECTURE]]
