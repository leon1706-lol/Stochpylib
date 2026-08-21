# stochpylib.levy_processes

*Levy & advanced stochastic processes*

**Design-completeness score:** 10/10 — Levy, Hawkes, Cox, jump-diffusion, semi-Markov — exhaustive

Status: **planned** — not yet implemented.

## Submodules

### `levy_processes.levy`

- `LevyProcess`
- `StableProcess`
- `AlphaStableDistribution`
- `SpectrallPositive`
- `SubordinatedProcess`
- `LevyKhintchine`

### `levy_processes.jump_diffusion`

- `JumpDiffusion`
- `MertonJumpDiffusion`
- `KouJumpDiffusion`
- `BatesModel`
- `VarianceGammaProcess`
- `CGMYProcess`
- `NormalInverseGaussianProcess`

### `levy_processes.subordinators`

- `Subordinator`
- `GammaSubordinator`
- `InverseGaussianSubordinator`
- `StableSubordinator`
- `TemperingSubordinator`

### `levy_processes.advanced`

- `SemiMarkovProcess`
- `RenewalProcess`
- `BranchingProcess`
- `HawkesProcess`
- `MultivariateHawkes`
- `CoxProcess`
- `GaussianRandomField`
- `RandomMeasure`

### `levy_processes.sde`

- `SDE()`
- `Euler_Maruyama()`
- `Milstein()`
- `Runge_Kutta_SDE()`
- `StochasticTaylor()`
- `WeakApproximation()`
- `StrongApproximation()`

---

[[Module-Map]] · [[Ratings]] · [[ARCHITECTURE]]
