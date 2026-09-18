# stochpylib.financial_stochastics

Quantitative finance: option pricing (closed-form, lattice, Monte Carlo,
Fourier), Greeks, stochastic/local volatility, short-rate models, risk
(VaR/ES/stress/scenario), credit risk, and portfolio construction. 50 public
names across seven submodules, natively on numpy/scipy — no wrapper
dependencies.

**Status:** implemented & tested (50/50 spec names).

**Note:** the vault Quickstart's `BlackScholes(S=100, K=105, ...).call_price()
# 10.45` comment is a spec typo — the true value at `K=105` is `8.0214`
(`10.4506` is the `K=100` price); the tests assert the correct numbers.

## Files

- `_common.py` — private shared helpers: the Black-76 pricing kernel
  (`_black_price`) that every closed-form price in this module ultimately
  calls, Black-Scholes d1/d2, an implied-vol solver, an `MCResult` builder,
  a jittered Cholesky factor, a tridiagonal (Thomas) solver, and numpy-only
  matrix log/exp (for `CreditMigration.generator()`).
- `option_pricing.py` — `BlackScholes` (closed form + Greeks properties),
  `BlackScholes_American` (Barone-Adesi-Whaley), `BinomialTree` (CRR
  lattice), `TrinomialTree` (Kamrad-Ritchken), `MonteCarloOptionPricing`
  (plain/antithetic/control-variate/QMC, path-dependent and Asian payoffs),
  `LongstaffSchwartz` (American least-squares MC), `FourierOptionPricing`
  (Carr-Madan or Fang-Oosterlee COS inversion of any characteristic
  function).
- `greeks.py` — `Delta`/`Gamma`/`Vega`/`Theta`/`Rho`/`Vanna`/`Volga` closed
  forms; `Greeks_FD` (central finite differences around any pricer);
  `Greeks_MC` (pathwise, likelihood-ratio, and common-random-number bump
  estimators).
- `stochastic_vol.py` — `HestonModel` ("little trap" characteristic function,
  QE or full-truncation Euler simulation, Carr-Madan pricing, least-squares
  calibration), `SABRModel` (Hagan 2002 implied vol), `RoughHeston`
  (fractional Adams-scheme Riccati equation), `RoughBergomi` (exact-Cholesky
  Riemann-Liouville fBM simulation), `LocalVol` (Crank-Nicolson PDE or MC),
  `Dupire` (local vol from an implied-vol surface), `LVSV` (local-stochastic
  vol with a histogram-calibrated leverage function), `VarianceSwap` (closed
  form, replication, or realized MC).
- `rate_models.py` — `VasicekModel`, `CIRProcess` (exact noncentral-chi-square
  simulation), `HullWhiteModel` and `HoLeeModel` (constant-theta or
  curve-fitted), `G2ppModel` (two-factor Gaussian), `BlackKarasinski`
  (log-OU, Monte Carlo pricing — no closed form), `LMM` (spot-measure
  log-Euler LIBOR Market Model), `HJM` (Gaussian one-factor Musiela
  parametrization).
- `risk.py` — `HistoricalVaR`, `ParametricVaR` (normal/Student-t/
  Cornish-Fisher/EWMA/GARCH), `ValueAtRisk` (facade + Kupiec backtest),
  `ExpectedShortfall`, `ConditionalVaR` (+ Rockafellar-Uryasev CVaR portfolio
  optimization), `StressTest`, `ScenarioAnalysis`.
- `credit.py` — `DefaultIntensity` (piecewise-constant hazard curve),
  `CDSPricing` (premium/protection legs, bootstrapping), `MertonCreditModel`
  (structural PD, distance-to-default, equity-implied calibration),
  `CreditMigration` (rating transition matrix, generator via matrix
  logarithm), `CreditRiskModel` (independent-default portfolio loss),
  `CopulaCreditModel` (Gaussian/Student-t copula-dependent defaults, reusing
  `copulas.elliptical`).
- `portfolio.py` — `CovarianceEstimation` (sample/EWMA/Ledoit-Wolf
  shrinkage), `MeanVariance`, `PortfolioOptimization` (fluent fit/optimize),
  `BlackLitterman` (He-Litterman posterior), `RiskParity` (Spinu convex
  coordinate descent).

## Conventions

- Every stochastic method takes `random_state=`; Monte Carlo estimators
  return the shared `MCResult` (`montecarlo._result`), and VaR/ES estimators
  return `RiskResult` (`montecarlo.applications`) — both library-wide
  conventions, not reimplemented here.
- Closed-form prices/rates return `float` (or an array for vectorized
  strikes); nothing that isn't a Monte Carlo/statistical estimate returns a
  result object.
- `BlackScholes.Delta/.Gamma/.Vega/.Theta/.Rho/.Vanna/.Volga` are call-side
  properties delegating to the `greeks.py` functions of the same name (per
  the vault Quickstart); `.delta(kind)`/`.theta(kind)`/`.rho(kind)` take an
  explicit `kind` for the put side.
- `CopulaCreditModel` sets `correlation_`/`df_` directly on a
  `GaussianCopula`/`StudentTCopula` instance before `.sample()` — those
  classes have no constructor path for a fixed correlation matrix (see
  `copulas/elliptical.py`).
- `RiskParity.optimize()` solves Spinu's convex objective over *unnormalized*
  weights and normalizes only once, after convergence — renormalizing every
  coordinate-descent sweep breaks the fixed point (see `Probleme.md`).

## Known limitations

- `BlackKarasinski` has no closed-form zero-coupon bond price (log-OU short
  rate); `zcb_price` is Monte Carlo only.
- `RoughBergomi` uses the exact joint-Gaussian Cholesky simulation scheme,
  not the faster hybrid scheme of Bennedsen-Lunde-Pakkanen (2017) — accurate
  but `O(N^2)` in the number of time steps per path batch.
- `LongstaffSchwartz.price()`'s standard error is the plain in-sample
  Monte Carlo standard error of the realized cash flows, which is known to
  be slightly optimistic (the regression is fit and evaluated on the same
  paths) — adequate for the library's stated ≥3 SE testing convention, not a
  duality-gap-certified bound.
- `LMM.simulate` drift is the spot-measure Euler approximation (not exact);
  caplet prices from it match Black's formula to a few basis points at
  `substeps=4`, not to machine precision.

Spec: vault `Modules/financial_stochastics.md` (private). Tests:
`tests/financial_stochastics/tests.py`.
