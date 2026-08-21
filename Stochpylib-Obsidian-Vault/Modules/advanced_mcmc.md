# stochpylib.advanced_mcmc

*State-of-the-art sampling algorithms*

**Design-completeness score:** 10/10 — NUTS, HMC, SMC, RJMCMC, particle MCMC — state of the art

Status: **planned** — not yet implemented.

## Submodules

### `advanced_mcmc.standard`

- `MetropolisHastings`
- `GibbsSampler`
- `IndependenceSampler`
- `AdaptiveMetropolis`
- `RobustAdaptiveMetropolis`

### `advanced_mcmc.gradient_based`

- `HamiltonianMonteCarlo`
- `NoUTurnSampler`
- `MALA`
- `MMALA`
- `RiemannianHMC`
- `NeutraHMC`

### `advanced_mcmc.slice_sampling`

- `SliceSampling`
- `EllipticalSliceSampling`
- `Doubling`
- `Stepping`
- `Polar_Slice`

### `advanced_mcmc.advanced`

- `ParallelTempering`
- `ReplicaExchange`
- `SequentialMonteCarlo`
- `ParticleMCMC`
- `ReversibleJumpMCMC`
- `TransdimensionalMCMC`

### `advanced_mcmc.diagnostics`

- `Rhat()`
- `ESS()`
- `GelmanRubin()`
- `geweke_test()`
- `raftery_lewis()`
- `autocorr_time()`
- `PSRF()`
- `TraceAnalysis`

### `advanced_mcmc.variational`

- `MeanFieldVI`
- `ADVI`
- `BlackBoxVI`
- `NormalizingFlows`
- `SteinVI`

---

[[Module-Map]] · [[Ratings]] · [[ARCHITECTURE]]
