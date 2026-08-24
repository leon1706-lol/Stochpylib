import io

p = "development/Implementation-Checklist.md"
s = io.open(p, encoding="utf-8").read()

old = (
    "- [ ] **stochpylib.survival** module overall\n"
    "  - [ ] `survival.nonparametric`\n"
    "    - [ ] `KaplanMeier`\n"
    "    - [ ] `NelsonAalen`\n"
    "    - [ ] `LifeTable`\n"
    "    - [ ] `EmpiricalSurvival`\n"
    "    - [ ] `BreslowEstimator`\n"
    "  - [ ] `survival.functions`\n"
    "    - [ ] `SurvivalFunction`\n"
    "    - [ ] `HazardFunction`\n"
    "    - [ ] `CumulativeHazard`\n"
    "    - [ ] `ResidualLifetime`\n"
    "    - [ ] `MeanResidualLife`\n"
    "  - [ ] `survival.parametric`\n"
    "    - [ ] `WeibullSurvival`\n"
    "    - [ ] `ExponentialSurvival`\n"
    "    - [ ] `LogNormalSurvival`\n"
    "    - [ ] `LogLogisticSurvival`\n"
    "    - [ ] `GompertzSurvival`\n"
    "  - [ ] `survival.regression`\n"
    "    - [ ] `CoxProportionalHazards`\n"
    "    - [ ] `AcceleratedFailureTime`\n"
    "    - [ ] `AalenAdditiveModel`\n"
    "    - [ ] `FineGrayModel`\n"
    "    - [ ] `StratifiedCox`\n"
    "  - [ ] `survival.tests`\n"
    "    - [ ] `LogRankTest`\n"
    "    - [ ] `WilcoxonSurvival`\n"
    "    - [ ] `TaroneWareTest`\n"
    "    - [ ] `PetoTest`\n"
    "    - [ ] `FlemingHarrington`\n"
    "  - [ ] `survival.competing_risks`\n"
    "    - [ ] `CumulativeIncidenceFunction`\n"
    "    - [ ] `CompetingRisksModel`\n"
    "    - [ ] `CauseSpecificHazard`"
)
new = old.replace("- [ ]", "- [x]")
if old in s:
    s = s.replace(old, new)
    s = s.replace("(0/28)", "(28/28)")
    print("replaced")
else:
    print("NOT FOUND")
io.open(p, "w", encoding="utf-8").write(s)
