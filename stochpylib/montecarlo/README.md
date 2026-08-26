# stochpylib.montecarlo

Quasi-random sequences, Monte Carlo estimators, variance-reduction techniques,
and applications — option pricing validated against Black–Scholes, historical
VaR/ES, reliability driven by the library's own distribution objects. Every
estimator returns a shared `MCResult` carrying its point estimate together with
an honest standard error and confidence interval.

**Status:** implemented & tested (25/25 spec names).

## Files

- `quasi_random.py` — Halton, Faure, Sobol and Niederreiter base-2 digital nets
  driven by programmatically-verified GF(2) generator polynomials, plus the
  general `DigitalNetBase2` engine (spec alias `DigitalNet`) and the
  `LowDiscrepancy` facade. Sobol embeds the standard Joe-Kuo direction-number
  table (64 dims): every `generate_block(m)` is an *exactly* balanced net block
  and set-identical to scipy's (`generate(n)` streams from index 1, so plain
  windows are ±1 off by construction). Niederreiter and custom-polynomial nets
  use the GF(2) machinery (balance within ±1, documented).
- `simulation.py` — `simulate()`, `crude_mc()`, `importance_sampling()`
  (self-normalized, with ESS), `rejection_sampling()`, `stratified_sampling()`,
  `quasi_montecarlo()`.
- `variance_reduction.py` — AntitheticVariates (incl. European call/put pricing),
  ControlVariates (optional custom sampler hook), StratifiedSampling,
  LatinHypercubeSampling, OrthogonalSampling, ConditionedMC, RejectionControl.
- `applications.py` — MonteCarloIntegration, pi_estimation(), option_pricing_mc(),
  risk_analysis() (historical VaR/ES → `RiskResult`), reliability_mc() driven by
  library distribution objects, sensitivity_analysis().
- `_result.py` — shared `MCResult` (`.estimate`, `.std_error`,
  `.confidence_interval()`, float coercion).

## Conventions

- Every sampler takes `random_state=`; sequences advance on successive
  `generate(n)` calls (`reset()` restarts); integrands map `(m, dim) -> (m,)`.
- No scipy.stats in library code — it is the test oracle in
  `tests/montecarlo/tests.py`.

## Known limitations

- Exact (t,m,s)-net balance for non-Sobol digital nets in dimensions >= 2 is
  within ±1 rather than certified (GF(2) fallback path only).
- Historical VaR/ES are quantile-based on simulated paths, not filtered
  (no GARCH-coupled simulation yet).

Spec: vault `Modules/montecarlo.md` (private). Tests:
`tests/montecarlo/tests.py`.
