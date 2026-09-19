# stochpylib.advanced_mcmc

State-of-the-art sampling algorithms: standard and adaptive Metropolis/Gibbs samplers,
gradient-based samplers (HMC, NUTS, MALA, manifold MALA, Riemannian HMC, NeuTra HMC),
slice samplers (stepping-out, doubling, elliptical, Gibbsian polar), population/sequential
methods (replica exchange, parallel tempering, adaptive-tempering SMC, particle MCMC), and
trans-dimensional samplers (reversible-jump, Carlin-Chib product-space), convergence
diagnostics (R-hat, ESS, Gelman-Rubin/PSRF, Geweke, Raftery-Lewis, autocorrelation time),
and variational inference (mean-field, ADVI, black-box VI, planar normalizing flows, SVGD).
35 public names across six submodules, natively on numpy/scipy — no wrapper dependencies.

**Status:** implemented & tested (35/35 spec names).

## Files

- `_common.py` — private helpers: seeded RNG, dimension checks, systematic resampling,
  finite-difference gradient/Hessian, Cholesky-with-jitter, `_DualAveraging` (Hoffman-Gelman
  step-size adaptation), `_Welford` (running mean/covariance), `_Adam` optimizer.
- `_base.py` — `LogDensity` (wraps a `log_prob` + optional analytic gradient/Hessian with a
  finite-difference fallback; `.from_distribution()` adapts a library `Distribution`) and
  `MCMCSampler` (base class: chain bookkeeping, warmup/adaptation, `get_chains()`/
  `get_samples()`/`summary()`).
- `standard.py` — `MetropolisHastings` (random-walk or general/asymmetric proposal),
  `IndependenceSampler`, `GibbsSampler` (exact conditionals or Metropolis-within-Gibbs),
  `AdaptiveMetropolis` (Haario et al. 2001), `RobustAdaptiveMetropolis` (Vihola 2012).
- `gradient_based.py` — `HamiltonianMonteCarlo` (leapfrog, dual-averaging step size,
  windowed diagonal mass adaptation), `NoUTurnSampler` (multinomial trajectory sampling,
  classic position-momentum stopping criterion), `MALA`, `MMALA` (simplified manifold MALA,
  SoftAbs metric default), `RiemannianHMC` (generalized leapfrog via fixed-point iteration),
  `NeutraHMC` (fits a `NormalizingFlows` approximation, runs NUTS/HMC in its base space).
- `slice_sampling.py` — `Stepping`/`Doubling` (Neal 2003 interval strategies),
  `SliceSampling` (coordinate-wise), `EllipticalSliceSampling` (Murray et al. 2010, exact
  for a Gaussian prior), `Polar_Slice` (Gibbsian polar slice sampler, Schar-Habeck-Rudolf
  2023).
- `advanced.py` — `ReplicaExchange` (general engine), `ParallelTempering` (tempered-ladder
  specialization with pilot-phase ladder adaptation), `SequentialMonteCarlo`
  (adaptive-tempering SMC with evidence estimation), `ParticleMCMC` (PMMH via
  `timeseries.ParticleFilter`), `ReversibleJumpMCMC` (Green 1995, default Gaussian-auxiliary
  dimension-matching move), `TransdimensionalMCMC` (Carlin & Chib 1995 product-space
  sampler with fitted Gaussian pseudo-priors).
- `diagnostics.py` — `Rhat` (split/rank/classic), `ESS` (bulk/tail/mean/sd), `GelmanRubin`,
  `PSRF` (multivariate), `geweke_test`, `raftery_lewis` (+ `RafteryLewisResult`),
  `autocorr_time` (Sokal/Geyer), `TraceAnalysis` (bundles all of the above per parameter).
- `variational.py` — `MeanFieldVI`, `ADVI` (mean-field or full-rank, with unconstraining
  bound transforms), `BlackBoxVI` (score-function gradients, no target gradient needed),
  `NormalizingFlows` (planar flows with hand-derived reverse-mode gradients — no autodiff
  in the stack), `SteinVI` (Stein variational gradient descent).

## Conventions

- **Chain layout.** `chains_` is always `(n_chains, n_samples, dim)`; `get_samples()`
  flattens to `(n_chains * n_samples, dim)`. Diagnostic functions accept `(n,)`, `(n, dim)`
  or `(n_chains, n, dim)`.
- **Fluent sampling.** `sample(theta_init, random_state=None)` returns `self`; warmup draws
  adapt (step size, mass matrix, proposal covariance, ...) and are discarded, and
  adaptation is frozen once the kept draws begin.
- **Gradients are optional.** Every gradient-based sampler and variational method takes an
  optional `grad_log_prob=`; without it, `LogDensity` supplies a central finite-difference
  fallback (`2 * dim` extra density evaluations per gradient) — costlier but exact enough
  for the samplers' Metropolis corrections to remain valid.
- **Every stochastic method takes `random_state=`** (anything `np.random.default_rng`
  accepts, including a live `Generator` for streaming draws across multiple chains).
- **Trans-dimensional models need fully normalized log-posteriors.** Unlike single-model
  MCMC (where an additive constant cancels in the acceptance ratio), `ReversibleJumpMCMC`
  and `TransdimensionalMCMC` compare densities *across* models of different dimension, so
  every `log_posteriors[k]` must include *all* normalizing constants (likelihood **and**
  prior) — dropping even one changes the implied Bayes factor. This is the single most
  common way to misuse these two classes; both README examples spell out every constant.
- **`ESS` here is autocorrelation-based** (Vehtari et al. 2021's bulk/tail estimator),
  distinct from the importance-weight ESS reported in
  `montecarlo.importance_sampling`'s `extras['ess']`.
- **`NeutraHMC`'s correctness never depends on the flow's fit quality** — the flow is only
  a smooth reparameterization; the sampler runs exactly in the flow's base space regardless
  of how well the flow approximates the target, so a mediocre flow costs mixing efficiency,
  never correctness (verified: even an under-trained flow recovers the exact posterior
  given enough draws — see Known limitations).
- Library code never imports `scipy.stats`; `scipy.special` is the only numerical building
  block, and `scipy.stats`/`statsmodels` are the test suite's independent oracles only.

## Known limitations

- **Planar normalizing flows (`NormalizingFlows`) are hard to train tightly from scratch.**
  `u` is only constrained (for invertibility) through its dot product with `w`; the
  orthogonal component is otherwise unpenalized, and even plain small-step gradient ascent
  on the raw ELBO can drift there and diverge. Weight decay, per-sample and per-step
  gradient clipping, a hard cap on `||u||`, an invertibility safety margin (denominator
  bounded away from 0), and keep-best snapshotting (the returned flow is the best smoothed
  window seen during training, not just the last iterate) keep training numerically stable,
  but convergence to a *tight* fit is not guaranteed for a fixed iteration budget — this
  is a documented characteristic of planar flows, not unique to this implementation.
  `NeutraHMC` sidesteps the consequence (see above); users calling `NormalizingFlows`
  directly for its own samples/`log_prob` should check `elbo_history_` and increase
  `n_iter`/`n_mc` or reduce `n_layers` if the fit looks poor.
- Only the flow's *forward* direction is analytically evaluated; planar flows are not
  easily invertible, so `log_prob`/density values are available only for points produced
  by `sample(..., return_log_prob=True)`, not for arbitrary query points.
- `RiemannianHMC`/`MMALA`'s default SoftAbs metric costs an eigendecomposition (and
  `RiemannianHMC`'s generalized leapfrog a further `n_fixed_point` linear solves) per
  step — `O(dim**3)`, intended for modest dimension (roughly `dim <= 20`).
- `ESS`'s Geyer-paired estimator can, for genuinely independent draws, occasionally report
  values well above the number of draws (a known quirk of this estimator family, shared by
  Stan/ArviZ) — capped at `n * log10(n)` rather than reported as a tighter but less
  standard bound.
- The spec lists `ReplicaExchange`/`ParallelTempering` and `ReversibleJumpMCMC`/
  `TransdimensionalMCMC` as separate names without further definition; implemented here as
  a general population-MCMC engine plus its tempered-ladder specialization, and as Green's
  reversible jump versus Carlin & Chib's product-space alternative, respectively.
- `GelmanRubin`/`PSRF` need at least 2 chains; `raftery_lewis` takes a single univariate
  chain (multivariate diagnostics go through `TraceAnalysis`, one parameter at a time).
- No plotting: diagnostics return arrays/dataclasses to plot with any library.

Spec: vault `Modules/advanced_mcmc.md` (private). Tests:
`tests/advanced_mcmc/tests.py` (oracles) and `tests/advanced_mcmc/e2e.py` (API sweep).
