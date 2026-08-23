# stochpylib.timeseries — complete time-series toolkit

**Status: implemented and tested** (61/61 spec names).

Layout:

- `linear_models.py` — AR, MA, ARMA (Hannan-Rissanen + CSS), ARIMA, SARIMA, ARFIMA,
  VAR, VARMA, VECM (reduced-rank regression).
- `volatility_models.py` — GARCH family via Gaussian QMLE: ARCH/GARCH/IGARCH,
  TGARCH & GJRGARCH (shared leverage core), EGARCH, APARCH, FIGARCH (truncated
  fractional weights), MGARCH (CCC) and scalar two-step DCC_GARCH.
- `state_space.py` — KalmanFilter + RTS smoother (`KalmanSmoother`), Extended/Unscented
  KFs, bootstrap ParticleFilter, GPB(1)-style RaoBlackwellFilter.
- `latent.py` — Gaussian HMM (Baum-Welch + Viterbi), Markov-switching regression /
  switching AR / mixture AR via Hamilton-style EM.
- `changepoint.py` — PELT (exact pruned DP), binary segmentation, bottom-up, and
  Adams–MacKay Bayesian online changepoint detection.
- `spectral.py` — Periodogram, Welch PowerSpectrum, Morlet CWT/scalogram,
  DWT/IDWT with perfect reconstruction (haar/db2), STFT, Hilbert transform.
- `tests.py` — ADF/KPSS/PP unit-root tests, Ljung-Box, Durbin-Watson, ARCH-LM,
  Granger causality, Johansen cointegration test.
- `forecasting.py` — forecast/predict dispatchers, confidence bands, walk-forward
  backtesting, rolling-origin cross-validation.

Conventions: fluent `.fit()` returning self; forecasts wrapped in a shared
`ForecastResult` (`.mean`, `.std`, `.confidence_interval()`); `random_state=` on all
simulations; native numpy/scipy implementations with **statsmodels as a dev-only test
oracle**. Documented simplifications are listed in the vault module spec.

Spec: vault `Modules/timeseries.md` (private).
