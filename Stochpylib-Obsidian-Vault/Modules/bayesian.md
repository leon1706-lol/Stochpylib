# stochpylib.bayesian

*Bayesian inference framework*

**Design-completeness score:** 9/10 — Prior/posterior/model selection — needs more variational inference

Status: **planned** — not yet implemented.

## Submodules

### `bayesian.core`

- `prior()`
- `likelihood()`
- `posterior()`
- `bayes_update()`
- `conjugate_prior()`
- `posterior_predictive()`
- `evidence()`

### `bayesian.models`

- `BayesianLinear`
- `BayesianLogistic`
- `NaiveBayes`
- `HierarchicalModel`
- `MixtureModel`
- `BayesianNetwork`
- `DirichletProcess`

### `bayesian.selection`

- `bayes_factor()`
- `BIC()`
- `AIC()`
- `DIC()`
- `WAIC()`
- `LOO_CV()`
- `TICfit()`

### `bayesian.computation`

- `LaplacePosterior()`
- `EP_Posterior()`
- `MFVariational()`
- `ImportanceSamplingPosterior()`

---

[[Module-Map]] · [[Ratings]] · [[ARCHITECTURE]]
