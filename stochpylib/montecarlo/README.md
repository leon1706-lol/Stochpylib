# stochpylib.montecarlo — simulation & variance reduction

**Status: implemented and tested** (25/25 spec names).

Layout:

- `quasi_random.py` — Halton, Faure, Sobol, Niederreiter sequences + the general GF(2)
  digital-net engine (`DigitalNetBase2`, spec alias `DigitalNet`) and the `LowDiscrepancy`
  facade. Native implementations; dimension 1 of the base-2 nets is exactly van der Corput
  and the canonical 2-D prefix matches standard tables (higher dimensions are within +-1 of
  exact net balance — see `development/Probleme.md` [11]).
- `simulation.py` — `simulate()`, `crude_mc()`, `importance_sampling()` (self-normalized,
  with ESS), `rejection_sampling()`, `stratified_sampling()`, `quasi_montecarlo()`.
- `variance_reduction.py` — AntitheticVariates (incl. European call/put pricing),
  ControlVariates (optional custom sampler hook), StratifiedSampling, LatinHypercubeSampling,
  OrthogonalSampling, ConditionedMC, RejectionControl.
- `applications.py` — MonteCarloIntegration, pi_estimation(), option_pricing_mc(),
  risk_analysis() (historical VaR/ES), reliability_mc() (driven by library distributions!),
  sensitivity_analysis().
- `_result.py` — shared `MCResult` (`.estimate`, `.std_error`, `.confidence_interval()`,
  `float()` coercion).

Conventions: every sampler takes `random_state=`; sequences advance on successive
`generate(n)` calls (`reset()` restarts); integrands map `(m, dim) -> (m,)`.
No scipy.stats in library code — it is the test oracle in `tests/montecarlo/tests.py`.

Spec: vault `Modules/montecarlo.md` (private).
