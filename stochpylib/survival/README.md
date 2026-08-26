# stochpylib.survival

Survival and reliability analysis: Kaplan-Meier and Nelson-Aalen estimators,
parametric censored fits, Cox proportional hazards regression, the log-rank
test family, and competing-risks modeling — with `lifelines` as a dev-only
test oracle for the nonparametric core.

**Status:** implemented & tested (28/28 spec names).

## Files

- `nonparametric.py` — Kaplan-Meier (Greenwood CI, log-log or linear),
  Nelson-Aalen cumulative hazard, actuarial `LifeTable`, `EmpiricalSurvival`,
  `BreslowEstimator` for baseline hazards.
- `functions.py` — wrappers bridging data fits and distribution objects:
  `SurvivalFunction`, `HazardFunction` (generic pdf/(1-cdf) fallback for
  library distributions), `CumulativeHazard` (integration grid starts near
  zero — see `development/Probleme.md` [32]), `ResidualLifetime`,
  `MeanResidualLife` (log-time quadrature).
- `parametric.py` — Weibull/Exponential (closed-form rate)/LogNormal/
  LogLogistic/Gompertz censored MLE with AIC.
- `regression.py` — Cox PH (Breslow/Efron ties, vectorised suffix-sum risk
  sums), `StratifiedCox` with per-stratum baselines, Weibull
  `AcceleratedFailureTime`, `AalenAdditiveModel` (per-event-time LS with
  stabilisation guards), `FineGrayModel` (weighted-Cox scoring with
  inverse-KM-of-censoring weights).
- `tests.py` — log-rank family: `LogRankTest`, `WilcoxonSurvival`
  (Gehan-Breslow), `TaroneWareTest`, `PetoTest`, `FlemingHarrington(rho, gamma)`.
- `competing_risks.py` — `CauseSpecificHazard`, `CumulativeIncidenceFunction`
  (Aalen-Johansen, exact CIF+KM=1 identity), `CompetingRisksModel` facade.

## Conventions

- Fluent `.fit(durations, events)` returning self; `random_state=` on sampling
  methods; native numpy/scipy only.

## Known limitations

- FineGray uses naive inverse-information variance (documented).
- Aalen increments are skipped when the risk-set design becomes rank-deficient
  or too small (stabilisation guards, not silent zeros).
- LogLogistic/Gompertz MLEs use Nelder-Mead (no analytic gradients);
  Gompertz exponent is clipped against overflow ([34]).

Spec: vault `Modules/survival.md` (private). Tests:
`tests/survival/tests.py`.
