# stochpylib.bayesian

Bayesian inference framework: priors/likelihoods/posteriors with ten closed-form
conjugate families (predictives and evidence included), posterior approximations
(Laplace, expectation propagation, variational inference, importance sampling), model
selection (AIC/BIC/DIC/WAIC/PSIS-LOO/TIC, Bayes factors), and seven Bayesian models
(linear/logistic regression, naive Bayes, a hierarchical normal model, finite mixtures,
discrete Bayesian networks, Dirichlet-process mixtures). 25 public names across four
submodules, natively on numpy/scipy — MCMC, SMC and variational inference delegate to
`stochpylib.advanced_mcmc` rather than duplicating it (per the vault's design-scorecard
note that `bayesian.computation`'s VI was thin relative to `advanced_mcmc.variational`);
expectation propagation's moment matching delegates to `stochpylib.numerical_methods`.

**Status:** implemented & tested (25/25 spec names).

## Files

- `_common.py` — private helpers: seeded RNG, log-sum-exp, `_LocScaleT` (location-scale
  Student-t built on `distributions.Student_t`), `_NormalInverseGamma` (the conjugate
  joint prior for a normal likelihood with unknown mean and variance), `_as_log_density`
  (normalizes a callable/`LogDensity`/`(Prior, Likelihood)` tuple/`Posterior` into an
  `advanced_mcmc.LogDensity`), a Neal (2003) stepping-out-and-shrinkage slice update, and
  `_psis`/`_gpd_fit` (Pareto-smoothed importance sampling — Vehtari, Simpson, Gelman, Yao
  & Gabry — with the Zhang & Stephens 2009 profile-likelihood GPD estimator).
- `_result.py` — `Posterior` (dist-/sample-/grid-backed, with `credible_interval`
  equal-tailed or HPD), `PosteriorApproximation` (Gaussian or importance-weighted),
  `ICResult` (`float()`-coercible criterion value), `EmpiricalPredictive` (a
  posterior-predictive distribution from draws — Gaussian-KDE `pdf`, satisfies the full
  13-method distribution contract).
- `core.py` — `Prior`/`prior()`, `Likelihood`/`likelihood()` (ten built-in exponential
  families plus custom `loglik`), `ConjugateFamily`/`conjugate_prior()` (bernoulli,
  binomial, poisson, exponential, gamma-with-known-shape, normal-known-`sigma`,
  normal-variance-known-`mu`, normal-unknown-variance via NIG, categorical/Dirichlet,
  multivariate-normal-known-covariance), `posterior()` (`method="auto"`: conjugate when
  the pair matches, else a numerical grid for dim <= 2, else MCMC; also `"laplace"`,
  `"vi"`, `"importance"`, `"smc"`), `bayes_update()` (sequential == batch), `evidence()`,
  `posterior_predictive()` (closed form when conjugate, else a Monte Carlo
  `EmpiricalPredictive`).
- `computation.py` — `LaplacePosterior` (MAP + inverse-negative-Hessian covariance, via
  `advanced_mcmc.LogDensity`'s finite-difference fallback when no analytic gradient is
  given), `EP_Posterior` (Gaussian EP for `N(theta|m0,V0) * prod_i t_i(a_i^T theta)`,
  Gauss-Hermite moment matching per site — works for *any* analytically evaluable site,
  not only Gaussian-conjugate ones, e.g. the true logistic site), `MFVariational`
  (delegates to `advanced_mcmc.MeanFieldVI`/`ADVI`), `ImportanceSamplingPosterior`
  (self-normalized IS with a default Laplace-centred multivariate-t proposal and PSIS
  smoothing).
- `selection.py` — `AIC`/`BIC` (+ AICc), `DIC`, `WAIC`, `LOO_CV` (PSIS by default, plain
  IS via `method="is"`), `TICfit` (sandwich-covariance generalization of AIC), and
  `bayes_factor` (from two log-evidences, `(prior, likelihood)` tuples, or
  `method="bic"` on two fitted models), all returning `ICResult`.
- `models.py` — `BayesianLinear` (conjugate Normal-Inverse-Gamma regression),
  `BayesianLogistic` (`method="laplace"`/`"ep"`/`"mcmc"`/`"vi"`), `NaiveBayes`
  (gaussian/bernoulli/multinomial, posterior-mean parameters), `HierarchicalModel`
  (normal-normal hierarchy, Gibbs; `tau_prior` half-Cauchy/inverse-gamma/fixed; mode A
  known per-group `sigma`, mode B raw data + `groups=` + unknown shared `sigma`),
  `MixtureModel` (Bayesian finite mixture by collapsed Gibbs, gaussian NIG/NIW or
  1-D poisson-gamma base), `BayesianNetwork` (discrete, exact variable-elimination
  inference, BDeu structure score), `DirichletProcess` (stick-breaking, CRP, and a
  Neal-2000-Algorithm-3 collapsed Gibbs DP mixture).

## Conventions

- **Result objects.** `Posterior`/`PosteriorApproximation`/`ICResult` are this module's
  siblings of `statistics.EstimateResult`/`TestResult` — a point answer plus enough to
  quantify it (`credible_interval()`, `sample()`, `float(ICResult)`).
- **`.fit(X, y)` returns `self`**, fitted attributes end in `_`, every stochastic method
  takes `random_state=` — the same fluent convention as the rest of the library.
- **Delegation, not duplication.** MCMC posteriors run through `advanced_mcmc`'s
  samplers, `MFVariational` through `MeanFieldVI`/`ADVI`, SMC evidence through
  `SequentialMonteCarlo`, and EP's per-site moment matching through
  `numerical_methods.GaussHermite`.
- **`bayes_factor` returns `BF_12`** (`p(data|M1)/p(data|M2)`) with the natural-log value
  in `extras["log_bf"]` and a Jeffreys-scale label in `extras["jeffreys"]`.
- **Label switching** in `MixtureModel`/`DirichletProcess` is mitigated by sorting
  components by their mean each draw (`relabel="order"`, the default) — a documented
  heuristic, not a guarantee, and turned off with `relabel=None`.

## Known limitations

- **The `gamma` conjugate family (known shape, unknown rate) has no library predictive
  distribution** — the compound-gamma posterior predictive isn't one of the library's 47
  distributions; `ConjugateFamily("gamma").predictive(...)` returns `None`.
- **`EP_Posterior` covers Gaussian-prior, single linear-projection (`a_i^T theta`) sites**
  — a good fit for GLM-style models (as `BayesianLogistic(method="ep")` uses it) but not
  a general factor-graph EP.
- **`MixtureModel`/`DirichletProcess` cover gaussian (any dimension) and 1-D poisson
  families only.**
- **`BayesianNetwork` is discrete-only** (categorical CPTs); variable elimination uses a
  simple left-to-right elimination order, not a min-degree/min-fill heuristic, so very
  wide networks may be slow.
- **`HierarchicalModel` is a two-level normal-normal hierarchy only** — no
  non-normal likelihoods or additional levels.
- **`evidence(method="grid")`/`posterior(method="grid")` are limited to dim <= 2** (a
  dense tensor grid stops being practical beyond that); higher dimensions need
  `"laplace"`, `"vi"`, `"importance"`, `"smc"`, or `"mcmc"`.
- **VI's reported evidence is the ELBO**, a lower bound on the true log evidence, not an
  estimate of it (`extras["evidence_is_lower_bound"] = True`).
- **No autodiff.** Gradients/Hessians come from `advanced_mcmc.LogDensity`, which uses
  an analytic callable when given and otherwise falls back to finite differences.
- **`DirichletProcess.fit`/`MixtureModel.fit` are pure-Python Gibbs samplers** (an O(n)
  Python loop per sweep) — fine for the hundreds-of-points scale typical of these
  examples, slow much beyond a few thousand.

Spec: vault `Modules/bayesian.md` (private). Tests: `tests/bayesian/tests.py` (oracles)
and `tests/bayesian/e2e.py` (API sweep).
