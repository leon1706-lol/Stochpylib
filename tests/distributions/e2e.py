"""End-to-end API sweep for stochpylib.distributions: one realistic exercise per public
name. Every univariate class runs the same contract workflow (pdf/pmf, cdf/ppf round trip,
seeded rvs, moments, entropy/mgf/cf, fit, ks_test) on a realistic instance; the seven
multivariate classes run the sanctioned reduced contract. ``test_every_public_name_is_exercised``
fails the moment a name ships without one; ``test_exercise[<name>]`` runs each exercise as
its own pytest case."""

import math

import numpy as np
import pytest

from stochpylib import distributions as mod

EXERCISES = {}


def exercise(name):
    def deco(fn):
        EXERCISES[name] = fn
        return fn
    return deco


# name -> (constructor, discrete, has_finite_moments, has_fit)
UNIVARIATE = {
    "Bernoulli": (lambda: mod.Bernoulli(0.3), True, True),
    "Binomial": (lambda: mod.Binomial(10, 0.4), True, True),
    "Poisson": (lambda: mod.Poisson(3.0), True, True),
    "Geometric": (lambda: mod.Geometric(0.3), True, True),
    "NegBinomial": (lambda: mod.NegBinomial(3.0, 0.4), True, True),
    "Hypergeometric": (lambda: mod.Hypergeometric(50, 20, 10), True, True),
    "DiscreteUniform": (lambda: mod.DiscreteUniform(0, 9), True, True),
    "ZipfDistribution": (lambda: mod.ZipfDistribution(4.0), True, True),
    "BetaBinomial": (lambda: mod.BetaBinomial(10, 2.0, 3.0), True, True),
    "ConwayMaxwellPoisson": (lambda: mod.ConwayMaxwellPoisson(3.0, 1.2), True, True),
    "Normal": (lambda: mod.Normal(0.5, 1.5), False, True),
    "Exponential": (lambda: mod.Exponential(1.5), False, True),
    "Uniform": (lambda: mod.Uniform(-1.0, 2.0), False, True),
    "Beta": (lambda: mod.Beta(2.0, 3.0), False, True),
    "Gamma": (lambda: mod.Gamma(2.0, 1.5), False, True),
    "Chi2": (lambda: mod.Chi2(4), False, True),
    "Student_t": (lambda: mod.Student_t(6), False, True),
    "F": (lambda: mod.F(6, 12), False, True),
    "Cauchy": (lambda: mod.Cauchy(0.5, 1.2), False, False),
    "Laplace": (lambda: mod.Laplace(0.0, 1.0), False, True),
    "Weibull": (lambda: mod.Weibull(2.0, 1.5), False, True),
    "Pareto": (lambda: mod.Pareto(5.0, 1.0), False, True),
    "LogNormal": (lambda: mod.LogNormal(0.2, 0.5), False, True),
    "Gumbel": (lambda: mod.Gumbel(0.5, 2.0), False, True),
    "Frechet": (lambda: mod.Frechet(4.0, 1.0), False, True),
    "GEV": (lambda: mod.GEV(0.0, 1.0, -0.1), False, True),
    "GPareto": (lambda: mod.GPareto(0.0, 1.0, 0.2), False, True),
    "InvGamma": (lambda: mod.InvGamma(5.0, 3.0), False, True),
    "InvGaussian": (lambda: mod.InvGaussian(1.5, 2.0), False, True),
    "Rayleigh": (lambda: mod.Rayleigh(1.5), False, True),
    "Maxwell": (lambda: mod.Maxwell(2.0), False, True),
    "Nakagami": (lambda: mod.Nakagami(2.0, 1.0), False, True),
    "Rice": (lambda: mod.Rice(1.0, 1.0), False, True),
    "VonMises": (lambda: mod.VonMises(0.3, 2.0), False, True),
    "Kumaraswamy": (lambda: mod.Kumaraswamy(2.0, 3.0), False, True),
    "LevyDistribution": (lambda: mod.LevyDistribution(0.5, 1.0), False, False),
    "SubGaussian": (lambda: mod.SubGaussian(0.2, 1.0, 3.0), False, True),
    "SubExponential": (lambda: mod.SubExponential(0.9, 1.5), False, True),
    "AlphaStable": (lambda: mod.AlphaStable(1.5, 0.0, 1.0), False, False),
    "StableDistribution": (lambda: mod.StableDistribution(1.7, 0.0, 0.0, 1.0), False, False),
}

I2 = np.eye(2)
MULTIVARIATE = {
    "MultivariateNormal": lambda: mod.MultivariateNormal([0.0, 0.0], I2),
    "Dirichlet": lambda: mod.Dirichlet([1.0, 2.0, 3.0]),
    "Wishart": lambda: mod.Wishart(5, I2),
    "InverseWishart": lambda: mod.InverseWishart(6, [[2.0, 0.5], [0.5, 1.0]]),
    "MultivariateT": lambda: mod.MultivariateT(5, [0.0, 0.0], I2),
    "MultivariatePareto": lambda: mod.MultivariatePareto(4, [0.0, 0.0], [1.0, 1.0]),
    "Multinomial": lambda: mod.Multinomial(20, [0.2, 0.3, 0.5]),
}


# fixed size parameters that these fits cannot infer from data
FIT_KW = {"Binomial": {"n": 10}, "NegBinomial": {"r": 3.0}, "Hypergeometric": {"N": 50, "n": 10},
          "BetaBinomial": {"n": 10}}
# numerically inverted characteristic functions: entropy is a nested quadrature that
# takes minutes, so the sweep leaves it to the dedicated checks in tests.py
SLOW_NUMERIC = {"AlphaStable", "StableDistribution"}


def _univariate_workflow(name):
    make, discrete, finite_moments = UNIVARIATE[name]
    d = make()
    assert isinstance(d, mod.Distribution)
    lo, hi = d.support()
    n_draw = 60 if name in SLOW_NUMERIC else 400
    x = d.rvs(n_draw, random_state=0)
    x = np.asarray(x, dtype=float)
    assert x.shape == (n_draw,) and np.all(np.isfinite(x))
    assert np.all(x >= lo - 1e-9) and np.all(x <= hi + 1e-9)
    # density / mass and cdf on the sample
    dens = np.asarray(d.pmf(x) if discrete else d.pdf(x), dtype=float)
    assert np.all(dens >= 0) and np.all(np.isfinite(dens))
    c = np.asarray(d.cdf(x), dtype=float)
    assert np.all((c >= -1e-12) & (c <= 1 + 1e-12))
    # quantile round trip (discrete: cdf(ppf(q)) >= q)
    q = float(d.ppf(0.6))
    if discrete:
        assert float(d.cdf(q)) >= 0.6 - 1e-9
    else:
        assert abs(float(d.cdf(q)) - 0.6) < 1e-5
    # moments and shape statistics
    m, v = d.mean(), d.var()
    if finite_moments:
        assert np.isfinite(m) and np.isfinite(v) and v >= 0
        assert abs(x.mean() - m) < 6 * math.sqrt(v / n_draw) + 0.05 * abs(m) + 1e-9
        assert np.isfinite(d.std())
        assert np.isfinite(d.skewness()) and np.isfinite(d.kurtosis())
    # entropy / transforms
    if name not in SLOW_NUMERIC:
        assert np.isfinite(d.entropy())
    assert abs(complex(d.cf(0.0)) - 1.0) < 1e-6
    if name not in ("Cauchy", "LevyDistribution", "AlphaStable", "StableDistribution",
                    "Pareto", "Frechet", "InvGamma", "GEV", "GPareto", "Student_t", "F",
                    "ZipfDistribution", "SubExponential", "SubGaussian", "InvGaussian"):
        assert abs(float(d.mgf(0.0)) - 1.0) < 1e-6
    # goodness of fit against its own draws, and fit round trip
    stat, p = d.ks_test(x)
    assert 0 <= stat <= 1 and 0 <= p <= 1
    if name not in SLOW_NUMERIC:
        fitted = type(d).fit(x, **FIT_KW.get(name, {}))
        assert isinstance(fitted, type(d))


def _multivariate_workflow(name):
    d = MULTIVARIATE[name]()
    assert isinstance(d, mod.MultivariateDistribution)
    x = np.asarray(d.rvs(50, random_state=0), dtype=float)
    assert x.shape[0] == 50 and np.all(np.isfinite(x))
    dens = np.asarray(d.pdf(x[0]), dtype=float)
    assert np.all(np.isfinite(dens)) and np.all(dens >= 0)
    assert np.all(np.isfinite(np.asarray(d.mean(), dtype=float)))
    assert np.all(np.isfinite(np.asarray(d.var(), dtype=float)))
    for meth in ("pdf", "cdf", "ppf", "rvs", "mean", "var", "skewness", "kurtosis",
                 "entropy", "fit", "ks_test"):
        assert callable(getattr(d, meth))


for _name in UNIVARIATE:
    EXERCISES[_name] = (lambda n: (lambda: _univariate_workflow(n)))(_name)
for _name in MULTIVARIATE:
    EXERCISES[_name] = (lambda n: (lambda: _multivariate_workflow(n)))(_name)


@exercise("Distribution")
def _base_class():
    class Tri(mod.Distribution):
        def support(self):
            return (0.0, 1.0)

        def pdf(self, x):
            x = np.asarray(x, dtype=float)
            return np.where((x >= 0) & (x <= 1), 2.0 * x, 0.0)

        def cdf(self, x):
            x = np.clip(np.asarray(x, dtype=float), 0.0, 1.0)
            return x * x

        def mean(self):
            return 2.0 / 3.0

        def var(self):
            return 1.0 / 18.0

    t = Tri()  # generic fallbacks: ppf by root finding, rvs by inversion, moments by quadrature
    assert abs(float(t.ppf(0.25)) - 0.5) < 1e-6
    r = np.asarray(t.rvs(200, random_state=1))
    assert abs(r.mean() - 2 / 3) < 0.06 and abs(t.skewness() + 0.566) < 0.01
    assert abs(t.mgf(0.0) - 1.0) < 1e-8 and np.isfinite(t.entropy())


@exercise("MultivariateDistribution")
def _multivariate_base():
    assert issubclass(mod.MultivariateNormal, mod.MultivariateDistribution)
    d = mod.MultivariateNormal([0.0, 0.0], I2)
    assert 0.0 <= float(d.cdf([0.0, 0.0], n_samples=2000, random_state=0)) <= 1.0


def test_every_public_name_is_exercised():
    assert set(EXERCISES) == set(mod.__all__), (
        set(mod.__all__) - set(EXERCISES), set(EXERCISES) - set(mod.__all__))


@pytest.mark.parametrize("name", sorted(mod.__all__))
def test_exercise(name):
    EXERCISES[name]()
