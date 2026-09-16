# stochpylib.levy_processes

Lévy processes and their relatives: the core Lévy-Khintchine machinery, jump-
diffusion pricing models, subordinators (nondecreasing time changes), advanced
point/branching/field processes, and a small SDE-solver toolkit. 33 public
names across five submodules, natively on numpy/scipy — no wrapper
dependencies.

**Status:** implemented & tested (33/33 spec names).

## Files

- `levy.py` — `LevyProcess` (Brownian + compound-Poisson base),
  `StableProcess` (alpha-stable Lévy motion via the library's validated
  Chambers-Mallows-Stuck sampler), `AlphaStableDistribution` (adapter over
  `distributions.AlphaStable`), `SpectrallyPositive` (nondecreasing
  alpha-stable, 0 < alpha < 1), `SubordinatedProcess` (a base process
  time-changed by any subordinator, `X_t = B_{T_t}`), `LevyKhintchine` (the
  (b, sigma^2, nu) triplet with an analytic characteristic exponent for both
  compound-Poisson and pure-subordinator jump components).
- `subordinators.py` — `Subordinator` base (path simulation machinery),
  `GammaSubordinator` (the Variance-Gamma time change), `InverseGaussianSubordinator`
  (the NIG time change, `E[T_t] = t`), `StableSubordinator` (positive
  1/alpha-stable, `E[exp(-lam T_t)] = exp(-t lam**alpha)` exactly), and
  `TemperingSubordinator` (the CGMY/tempered-stable time change, simulated by
  a truncated compound-Poisson approximation with analytic mean
  compensation).
- `jump_diffusion.py` — `JumpDiffusion` base (drift + diffusion + compound
  Poisson), `MertonJumpDiffusion` (lognormal jumps, closed-form call series),
  `KouJumpDiffusion` (double-exponential jumps, Carr-Madan Fourier call
  price), `BatesModel` (Heston stochastic volatility + lognormal jumps, MC
  pricing), `VarianceGammaProcess`, `CGMYProcess` (Carr-Geman-Madan-Yor
  pure-jump process, closed-form CF, truncated-compound-Poisson simulation
  for Y < 1), `NormalInverseGaussianProcess` (exact IG-subordination
  simulation). `carr_madan_call` is the shared damped-Fourier call-pricing
  routine.
- `advanced.py` — `HawkesProcess` (exponential-kernel self-exciting process:
  Ogata-thinning simulation, exact recursive MLE, `branching_ratio()`,
  time-rescaling `ks_residuals()`), `MultivariateHawkes` (vector intensity,
  thinning simulation, `branching_matrix()`), `CoxProcess` (doubly-stochastic
  Poisson via thinning), `RenewalProcess` (i.i.d. inter-arrivals from any
  library distribution), `BranchingProcess` (Galton-Watson with a custom
  offspring sampler), `SemiMarkovProcess` (embedded chain + state-dependent
  holding distributions), `GaussianRandomField` (FFT spectral synthesis),
  `RandomMeasure` (independent gamma or spectrally-positive stable mass on
  intervals).
- `sde.py` — `SDE` (drift/diffusion definition with finite-difference
  derivatives for the higher-order schemes), `Euler_Maruyama` (strong order
  0.5), `Milstein` (strong order 1.0), `Runge_Kutta_SDE` (strong order 1.0,
  Komori-Burrage-style stage scheme), `StochasticTaylor` (Kloeden-Platen
  strong order 1.5), `WeakApproximation` (Talay-Tubaro weak order 2),
  `StrongApproximation` (strong-error convergence study across step sizes).

## Conventions

- Every stochastic method takes `random_state=` for reproducibility.
- Process classes expose `simulate(T, n_steps, n_paths, random_state)`
  returning `(n_paths, n_steps + 1)` path arrays (origin included);
  subordinators additionally expose `sample(times, random_state)` for
  irregular grids and `increments(dt, n, random_state)` for i.i.d. draws.
- Pricing models (`jump_diffusion.py`) return risk-neutral European call
  values from `call_price`; `call_price_mc` cross-checks via Monte Carlo and
  returns the shared `MCResult`. `call_price_mc` prices under the model's own
  `b` (set from the constructor's `mu` at construction time) — pass
  `mu=r` at construction for the two to represent the same risk-neutral
  dynamics used by `call_price(..., r=...)`.
- Truncated compound-Poisson simulators (`TemperingSubordinator`, `CGMYProcess`)
  sample jump sizes from a log-spaced, cumulative-trapezoid quantile grid —
  required because the tempered-stable/CGMY Lévy density is singular at the
  origin and a linear grid badly misrepresents it (see
  `development/Probleme.md`).

## Known limitations

- `CGMYProcess` simulation (`simulate`/`_increment`) requires `Y < 1` (the
  small-jump mean diverges for `Y >= 1`); `characteristic_function` remains
  valid for all `Y` in `(0, 2)`.
- `MultivariateHawkes` has no MLE fitting — simulation and the branching
  matrix only (documented scope limitation).
- `BatesModel`'s variance process is a full-truncation Euler scheme, not an
  exact CIR sampler — accurate for moderate `xi`/`kappa` but not unconditionally
  positivity-preserving at large steps.

Spec: vault `Modules/levy_processes.md` (private). Tests:
`tests/levy_processes/tests.py`.
