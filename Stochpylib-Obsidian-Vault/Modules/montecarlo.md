# stochpylib.montecarlo

*Simulation & variance reduction*

**Design-completeness score:** 9/10 — Sobol, Halton, LHS, antithetic — professional simulation toolkit

Status: **planned** — not yet implemented.

## Submodules

### `montecarlo.simulation`

- `simulate()`
- `importance_sampling()`
- `rejection_sampling()`
- `stratified_sampling()`
- `quasi_montecarlo()`
- `crude_mc()`

### `montecarlo.variance_reduction`

- `AntitheticVariates`
- `ControlVariates`
- `StratifiedSampling`
- `LatinHypercubeSampling`
- `OrthogonalSampling`
- `ConditionedMC`
- `RejectionControl`

### `montecarlo.quasi_random`

- `SobolSequence`
- `HaltonSequence`
- `FaureSequence`
- `NiederreiterSequence`
- `DigitalNet`
- `LowDiscrepancy`

### `montecarlo.applications`

- `MonteCarloIntegration`
- `pi_estimation()`
- `option_pricing_mc()`
- `risk_analysis()`
- `reliability_mc()`
- `sensitivity_analysis()`

---

[[Module-Map]] · [[Ratings]] · [[ARCHITECTURE]]
