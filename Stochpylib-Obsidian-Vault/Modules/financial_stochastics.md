# stochpylib.financial_stochastics

*Quantitative finance models*

**Design-completeness score:** 10/10 — Black-Scholes to SABR, Greeks, VaR, credit risk — quant-ready

Status: **planned** — not yet implemented.

## Submodules

### `financial_stochastics.option_pricing`

- `BlackScholes`
- `BlackScholes_American`
- `BinomialTree`
- `TrinomialTree`
- `MonteCarloOptionPricing`
- `LongstaffSchwartz`
- `FourierOptionPricing`

### `financial_stochastics.stochastic_vol`

- `HestonModel`
- `SABRModel`
- `RoughHeston`
- `RoughBergomi`
- `LVSV`
- `LocalVol`
- `Dupire`
- `VarianceSwap`

### `financial_stochastics.rate_models`

- `HullWhiteModel`
- `CIRProcess`
- `VasicekModel`
- `HoLeeModel`
- `LMM`
- `HJM`
- `G2ppModel`
- `BlackKarasinski`

### `financial_stochastics.risk`

- `ValueAtRisk`
- `ExpectedShortfall`
- `ConditionalVaR`
- `HistoricalVaR`
- `ParametricVaR`
- `StressTest`
- `ScenarioAnalysis`

### `financial_stochastics.credit`

- `CreditRiskModel`
- `DefaultIntensity`
- `CreditMigration`
- `CDSPricing`
- `MertonCreditModel`
- `CopulaCreditModel`

### `financial_stochastics.greeks`

- `Delta`
- `Gamma`
- `Vega`
- `Theta`
- `Rho`
- `Vanna`
- `Volga`
- `Greeks_MC`
- `Greeks_FD`

### `financial_stochastics.portfolio`

- `PortfolioOptimization`
- `MeanVariance`
- `BlackLitterman`
- `RiskParity`
- `CovarianceEstimation`

---

[[Module-Map]] · [[Ratings]] · [[ARCHITECTURE]]
