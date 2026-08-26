# stochpylib.timeseries

The complete time-series toolkit: linear models (ARIMA family), volatility
models (GARCH family), state-space filtering, latent-regime models,
changepoint detection, spectral analysis, diagnostics and forecasting —
61 spec names across nine submodules, natively on numpy/scipy with
statsmodels as a dev-only test oracle.

**Status:** implemented & tested (61/61 spec names).

## Files

- `linear_models.py` — AR, MA, ARMA (Hannan-Rissanen + CSS), ARIMA, SARIMA,
  ARFIMA (fixed-width binomial fractional filtering), VAR, VARMA, VECM
  (reduced-rank regression).
- `volatility_models.py` — GARCH family via Gaussian QMLE: ARCH/GARCH/IGARCH,
  TGARCH & GJRGARCH (shared leverage core), EGARCH, APARCH, FIGARCH (truncated
  fractional weights), MGARCH (CCC) and scalar two-step DCC_GARCH.
- `state_space.py` — KalmanFilter + RTS smoother (`KalmanSmoother`),
  Extended/Unscented KFs, bootstrap ParticleFilter, GPB(1)-style
  RaoBlackwellFilter.
- `latent.py` — Gaussian HMM (Baum-Welch + Viterbi), Markov-switching
  regression / switching AR / mixture AR via Hamilton-style EM.
- `changepoint.py` — PELT (exact pruned DP), binary segmentation, bottom-up,
  and Adams–MacKay Bayesian online changepoint detection.
- `spectral.py` — Periodogram, Welch PowerSpectrum, Morlet CWT/scalogram,
  DWT/IDWT with perfect reconstruction (haar/db2), STFT, Hilbert transform.
- `tests.py` — ADF/KPSS/PP unit-root tests, Ljung-Box, Durbin-Watson,
  ARCH-LM, Granger causality, Johansen cointegration.
- `forecasting.py` — forecast/predict dispatchers, confidence bands,
  walk-forward backtesting, rolling-origin cross-validation.

## Conventions

- Fluent `.fit()` returning self; forecasts wrapped in a shared
  `ForecastResult` (`.mean`, `.std`, `.confidence_interval()`).
- `random_state=` on all stochastic routines.
- Oracle checks: AR/VAR coefficients match statsmodels exactly (shared OLS);
  ADF/Ljung-Box statistics to 1e-8; documented simplifications live in the
  vault module spec.

Spec: vault `Modules/timeseries.md` (private). Tests:
`tests/timeseries/tests.py`.
