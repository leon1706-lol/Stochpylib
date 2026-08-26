# stochpylib.distributions

47 probability distributions across discrete, continuous, multivariate and
heavy-tailed families behind one common interface — the load-bearing contract
of the library (`.pdf()/.pmf()`, `.cdf()`, `.ppf()`, `.rvs()`, `.mean()`,
`.var()`, `.skewness()`, `.kurtosis()`, `.entropy()`, `.mgf()`, `.cf()`,
`.fit()`, `.ks_test()` on every class), 60 spec names counting the shared methods.

**Status:** implemented & tested (60/60 spec names).

## Files

- `_base.py` — `Distribution` / `MultivariateDistribution` base classes providing
  the common interface with generic numerical fallbacks; concrete subclasses
  override closed forms where they exist.
- `discrete.py` — Bernoulli, Binomial, Poisson, Geometric, NegBinomial,
  Hypergeometric, DiscreteUniform, Multinomial, Zipf, BetaBinomial,
  Conway–Maxwell–Poisson.
- `continuous.py` — 25 classics from Normal to Kumaraswamy.
- `multivariate.py` — MVN, Dirichlet, Wishart, InverseWishart, MultivariateT,
  MultivariatePareto.
- `heavy_tail.py` — stable family with exact Gaussian (alpha=2) and Cauchy
  (alpha=1, beta=0) special cases, validated Chambers–Mallows–Leckie sampling for
  all alpha != 1, and a cached numerical-quantile sampler for the alpha=1, beta!=0
  corner; Levy, SubGaussian, SubExponential.

## Conventions

- Import from the module level per ARCHITECTURE.md:
  `from stochpylib.distributions import Normal`.
- No scipy.stats wrapping anywhere — scipy.special/optimize/integrate are raw
  numerical building blocks; scipy.stats is the *test oracle* in
  `tests/distributions/tests.py` (interface-contract matrix over all 13 methods
  x 47 classes).
- Every stochastic method takes `random_state=`; `.fit()` is maximum likelihood;
  `.ks_test()` returns `(statistic, p_value)` against data.

## Known limitations (documented deviations)

- The 7 multivariate classes expose `.pdf()` instead of `.pmf()` and omit the
  scalar-argument `.mgf()/.cf()` — the one sanctioned deviation from the common
  interface, asserted as such in the conformance tests.
- Multivariate distributions raise `NotImplementedError` for
  `.ppf/.skewness/.kurtosis/.ks_test` by design.
- `VonMises.var()` is the circular variance.
- Stable sampling for alpha=1, beta!=0 carries a ~5 s per-parameter-set quantile-
  table warmup (class-level cache); all other samplers are direct.

Spec: vault `Modules/distributions.md` (private). Tests:
`tests/distributions/tests.py`.
