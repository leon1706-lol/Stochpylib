# stochpylib.survival — survival & reliability analysis

**Status: implemented and tested** (28/28 spec names).

Layout:

- `nonparametric.py` — Kaplan-Meier (Greenwood CI, log-log or linear),
  Nelson-Aalen cumulative hazard, actuarial LifeTable, EmpiricalSurvival,
  BreslowEstimator for baseline hazards.
- `functions.py` — wrappers bridging data fits and distribution objects:
  SurvivalFunction, HazardFunction, CumulativeHazard, ResidualLifetime,
  MeanResidualLife (log-time quadrature).
- `parametric.py` — Weibull/Exponential/LogNormal/LogLogistic/Gompertz
  censored MLE with AIC; exponential has closed-form rate.
- `regression.py` — Cox PH (Breslow/Efron ties, vectorised suffix-sum risk
  sums), StratifiedCox, Weibull AcceleratedFailureTime, AalenAdditiveModel
  (per-event-time LS with stabilisation guards), FineGrayModel (weighted-Cox
  scoring with inverse-KM-of-censoring weights).
- `tests.py` — log-rank family: LogRankTest, WilcoxonSurvival,
  TaroneWareTest, PetoTest, FlemingHarrington(rho, gamma).
- `competing_risks.py` — CauseSpecificHazard, CumulativeIncidenceFunction
  (Aalen-Johansen), CompetingRisksModel facade with exact CIF+KM=1 identity.

Conventions: fluent `.fit(durations, events)` returning self;
`random_state=` on sampling methods; native numpy/scipy only.

Known limitations: FineGray uses naive inverse-information variance
(documented); Aalen increments are skipped when the risk-set design becomes
rank-deficient or too small (stabilisation guards); LogLogistic/Gompertz MLEs
use Nelder-Mead (no analytic gradients).

Spec: vault `Modules/survival.md` (private). Tests: `tests/survival/tests.py`.
