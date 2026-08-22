# stochpylib.distributions — 80+ probability distributions

**Status: implemented and tested** (60/60 spec names).

Layout:

- `_base.py` — `Distribution` / `MultivariateDistribution` base classes providing the common
  interface (`.pdf()/.pmf()`, `.cdf()`, `.ppf()`, `.rvs()`, `.mean()`, `.var()`, `.skewness()`,
  `.kurtosis()`, `.entropy()`, `.mgf()`, `.cf()`, `.fit()`, `.ks_test()`) with generic numerical
  fallbacks; concrete subclasses override closed forms where they exist.
- `discrete.py` — Bernoulli, Binomial, Poisson, Geometric, NegBinomial, Hypergeometric,
  DiscreteUniform, Multinomial, Zipf, BetaBinomial, Conway–Maxwell–Poisson.
- `continuous.py` — 25 classics from Normal to Kumaraswamy.
- `multivariate.py` — MVN, Dirichlet, Wishart, InverseWishart, MultivariateT,
  MultivariatePareto.
- `heavy_tail.py` — stable family: exact Gaussian (α=2) and Cauchy (α=1, β=0) special cases,
  validated Chambers–Mallows–Leckie sampling for all α≠1, and a cached numerical-quantile
  sampler for the α=1, β≠0 corner; Lévy, SubGaussian, SubExponential.

Import from the module level per ARCHITECTURE.md:
`from stochpylib.distributions import Normal`.

Conventions worth knowing:

- No scipy.stats wrapping anywhere — scipy.special/optimize/integrate are used only as numerical
  building blocks. scipy.stats is used as the *test oracle* in `tests/distributions/tests.py`.
- Multivariate distributions raise `NotImplementedError` for `.ppf/.skewness/.kurtosis/.ks_test`
  by design.
- `VonMises.var()` is the circular variance.

Spec: vault `Modules/distributions.md` (private).
