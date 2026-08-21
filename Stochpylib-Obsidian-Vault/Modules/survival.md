# stochpylib.survival

*Survival & reliability analysis*

**Design-completeness score:** 9/10 — KM, Nelson-Aalen, Cox PH, AFT — covers clinical + reliability

Status: **planned** — not yet implemented.

## Submodules

### `survival.nonparametric`

- `KaplanMeier`
- `NelsonAalen`
- `LifeTable`
- `EmpiricalSurvival`
- `BreslowEstimator`

### `survival.functions`

- `SurvivalFunction`
- `HazardFunction`
- `CumulativeHazard`
- `ResidualLifetime`
- `MeanResidualLife`

### `survival.parametric`

- `WeibullSurvival`
- `ExponentialSurvival`
- `LogNormalSurvival`
- `LogLogisticSurvival`
- `GompertzSurvival`

### `survival.regression`

- `CoxProportionalHazards`
- `AcceleratedFailureTime`
- `AalenAdditiveModel`
- `FineGrayModel`
- `StratifiedCox`

### `survival.tests`

- `LogRankTest`
- `WilcoxonSurvival`
- `TaroneWareTest`
- `PetoTest`
- `FlemingHarrington`

### `survival.competing_risks`

- `CumulativeIncidenceFunction`
- `CompetingRisksModel`
- `CauseSpecificHazard`

---

[[Module-Map]] · [[Ratings]] · [[ARCHITECTURE]]
