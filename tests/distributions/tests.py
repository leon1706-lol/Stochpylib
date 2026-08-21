"""Tests for stochpylib.distributions.

Structure:
- export surface == spec (47 classes + 2 base classes)
- interface-contract matrix: every univariate class exposes the full method set
- scipy.stats cross-checks for every class with a closed-form reference mapping
- discrete ppf semantics, mass checks, fit() round-trips, ks_test sanity
- StableDistribution special-case delegation + CML sampler vs closed-form cf
- multivariate conventions (NotImplementedError set, shapes, MC cdf)
- CLI: spl --version / spl --test plumbing

Deterministic: all randomness via fixed seeds.
"""

import doctest
import io
import contextlib

import numpy as np
import pytest
from scipy import stats as st

import stochpylib
from stochpylib import cli as cli_mod
from stochpylib import selftest as selftest_mod
from stochpylib import distributions as D

# ---------------------------------------------------------------- export surface

SPEC_CLASSES = [
    # discrete
    "Bernoulli", "Binomial", "Poisson", "Geometric", "NegBinomial", "Hypergeometric",
    "DiscreteUniform", "Multinomial", "ZipfDistribution", "BetaBinomial",
    "ConwayMaxwellPoisson",
    # continuous
    "Normal", "Exponential", "Uniform", "Beta", "Gamma", "Chi2", "Student_t", "F",
    "Cauchy", "Laplace", "Weibull", "Pareto", "LogNormal", "Gumbel", "Frechet", "GEV",
    "GPareto", "InvGamma", "InvGaussian", "Rayleigh", "Maxwell", "Nakagami", "Rice",
    "VonMises", "Kumaraswamy",
    # multivariate
    "MultivariateNormal", "Dirichlet", "Wishart", "InverseWishart", "MultivariateT",
    "MultivariatePareto",
    # heavy tail
    "AlphaStable", "LevyDistribution", "StableDistribution", "SubGaussian",
    "SubExponential",
]


def test_module_exports_match_spec():
    for name in SPEC_CLASSES:
        assert hasattr(D, name), f"missing export: {name}"
    assert set(D.__all__) >= set(SPEC_CLASSES)
    assert stochpylib.distributions is D


# ---------------------------------------------------------------- instances

# name -> (factory, is_discrete, moments_finite)
UNIVARIATE = {
    "Bernoulli": (lambda: D.Bernoulli(0.3), True, True),
    "Binomial": (lambda: D.Binomial(10, 0.4), True, True),
    "Poisson": (lambda: D.Poisson(3.0), True, True),
    "Geometric": (lambda: D.Geometric(0.3), True, True),
    "NegBinomial": (lambda: D.NegBinomial(3.0, 0.4), True, True),
    "Hypergeometric": (lambda: D.Hypergeometric(50, 20, 10), True, True),
    "DiscreteUniform": (lambda: D.DiscreteUniform(0, 9), True, True),
    "ZipfDistribution": (lambda: D.ZipfDistribution(4.0), True, False),
    "BetaBinomial": (lambda: D.BetaBinomial(10, 2.0, 3.0), True, True),
    "ConwayMaxwellPoisson": (lambda: D.ConwayMaxwellPoisson(3.0, 1.2), True, True),
    "Normal": (lambda: D.Normal(0.5, 1.5), False, True),
    "Exponential": (lambda: D.Exponential(1.5), False, True),
    "Uniform": (lambda: D.Uniform(-1.0, 2.0), False, True),
    "Beta": (lambda: D.Beta(2.0, 3.0), False, True),
    "Gamma": (lambda: D.Gamma(2.0, 1.5), False, True),
    "Chi2": (lambda: D.Chi2(4), False, True),
    "Student_t": (lambda: D.Student_t(6), False, True),
    "F": (lambda: D.F(6, 12), False, True),
    "Cauchy": (lambda: D.Cauchy(0.5, 1.2), False, False),
    "Laplace": (lambda: D.Laplace(0.0, 1.0), False, True),
    "Weibull": (lambda: D.Weibull(2.0, 1.5), False, True),
    "Pareto": (lambda: D.Pareto(5.0, 1.0), False, True),
    "LogNormal": (lambda: D.LogNormal(0.2, 0.5), False, True),
    "Gumbel": (lambda: D.Gumbel(0.5, 2.0), False, True),
    "Frechet": (lambda: D.Frechet(4.0, 1.0), False, True),
    "GEV": (lambda: D.GEV(0.0, 1.0, -0.1), False, True),
    "GPareto": (lambda: D.GPareto(0.0, 1.0, 0.2), False, True),
    "InvGamma": (lambda: D.InvGamma(5.0, 3.0), False, True),
    "InvGaussian": (lambda: D.InvGaussian(1.5, 2.0), False, True),
    "Rayleigh": (lambda: D.Rayleigh(1.5), False, True),
    "Maxwell": (lambda: D.Maxwell(2.0), False, True),
    "Nakagami": (lambda: D.Nakagami(2.0, 1.0), False, True),
    "Rice": (lambda: D.Rice(1.0, 1.0), False, True),
    "VonMises": (lambda: D.VonMises(0.3, 2.0), False, True),
    "Kumaraswamy": (lambda: D.Kumaraswamy(2.0, 3.0), False, True),
    "LevyDistribution": (lambda: D.LevyDistribution(0.5, 1.0), False, False),
    "SubGaussian": (lambda: D.SubGaussian(0.2, 1.0, 3.0), False, True),
    "SubExponential": (lambda: D.SubExponential(0.9, 1.5), False, True),
}

INTERFACE_METHODS = [
    "pdf", "pmf", "cdf", "ppf", "rvs", "mean", "var", "std",
    "skewness", "kurtosis", "entropy", "mgf", "cf", "ks_test",
]


@pytest.mark.parametrize("name", sorted(UNIVARIATE))
def test_interface_contract(name):
    make, discrete, moments = UNIVARIATE[name]
    d = make()
    for m in INTERFACE_METHODS:
        assert callable(getattr(d, m)), f"{name}.{m} missing"
    lo, hi = d.support()
    assert lo < hi
    x = float(d.ppf(0.4))
    p = float(np.atleast_1d(np.asarray(d.pdf(x), dtype=float))[0])
    assert np.isfinite(p) and p >= 0.0
    c = float(d.cdf(x))
    assert 0.0 <= c <= 1.0
    if not discrete:
        assert abs(float(d.cdf(float(d.ppf(0.37)))) - 0.37) < 1e-4
    else:
        k = float(d.ppf(0.37))
        assert float(d.cdf(k)) >= 0.37
        below = float(d.cdf(k - 1)) if k > lo else 0.0
        assert below < 0.37
    smp = np.asarray(d.rvs(100, random_state=123), dtype=float)
    assert smp.shape == (100,)
    assert np.all(np.isfinite(smp) | (smp == np.inf))
    m_, v_ = d.mean(), d.var()
    std = d.std()
    if np.isfinite(v_):
        assert abs(std - np.sqrt(v_)) < 1e-10 * (1 + std)
    if moments:
        assert np.isfinite(m_) and v_ > 0
        assert abs(float(d.skewness())) < 50.0
        assert float(d.kurtosis()) > -7.0
        ent = float(d.entropy())
        assert np.isnan(ent) or ent < 100.0
    mg = float(d.mgf(0.05))
    assert np.isnan(mg) or mg >= 1.0 or mg == float("inf") or mg != mg  # must not raise
    cfv = complex(d.cf(0.5))
    assert abs(cfv) <= 1.0 + 1e-9
    data = np.asarray(d.rvs(200, random_state=7), dtype=float)
    stat, pval = d.ks_test(data)
    assert 0.0 <= pval <= 1.0


def test_multivariate_exports_and_conventions():
    I2 = [[1.0, 0.0], [0.0, 1.0]]
    mv = {
        "MultivariateNormal": lambda: D.MultivariateNormal([0, 0], I2),
        "Dirichlet": lambda: D.Dirichlet([1.0, 2.0, 3.0]),
        "Wishart": lambda: D.Wishart(5, I2),
        "InverseWishart": lambda: D.InverseWishart(6, [[2.0, 0.5], [0.5, 1.0]]),
        "MultivariateT": lambda: D.MultivariateT(5, [0, 0], I2),
        "MultivariatePareto": lambda: D.MultivariatePareto(4, [0, 0], [1, 1]),
        "Multinomial": lambda: D.Multinomial(20, [0.2, 0.3, 0.5]),
    }
    for name, make in mv.items():
        d = make()
        x = np.atleast_1d(d.mean()) * 1.0 + (np.zeros_like(np.atleast_1d(d.mean())))
        pdf_val = d.pdf(x if x.ndim > 0 else x.reshape(1))
        assert np.isfinite(pdf_val) and pdf_val >= 0, name
        cov = np.asarray(d.var())
        k = len(np.atleast_1d(d.mean()))
        assert cov.shape == (k, k), name
        smp = np.asarray(d.rvs(30, random_state=5))
        assert smp.shape[0] == 30, name
        with pytest.raises(NotImplementedError):
            d.ppf(0.5)
        with pytest.raises(NotImplementedError):
            d.skewness()
        with pytest.raises(NotImplementedError):
            d.kurtosis()
        with pytest.raises(NotImplementedError):
            d.ks_test(np.zeros((3, k)))


def test_mvn_pdf_matches_closed_form():
    mean = [1.0, -1.0]
    cov = [[2.0, 0.3], [0.3, 1.0]]
    d = D.MultivariateNormal(mean, cov)
    ref = st.multivariate_normal(mean, cov)
    for x in ([1.0, -1.0], [0.0, 0.0], [2.5, 1.0]):
        assert np.isclose(d.pdf(x), ref.pdf(x), rtol=1e-10)


# ---------------------------------------------------------------- scipy references

SCIPY_REF = {
    "Bernoulli": lambda: st.bernoulli(0.3),
    "Binomial": lambda: st.binom(10, 0.4),
    "Poisson": lambda: st.poisson(3.0),
    "Geometric": lambda: st.geom(0.3),
    "NegBinomial": lambda: st.nbinom(3.0, 0.4),
    "Hypergeometric": lambda: st.hypergeom(50, 20, 10),
    "DiscreteUniform": lambda: st.randint(0, 10),
    "ZipfDistribution": lambda: st.zipf(4.0),
    "BetaBinomial": lambda: st.betabinom(10, 2.0, 3.0),
    # ConwayMaxwellPoisson: exact nu==1 reduction to Poisson is covered by its dedicated test
    "Normal": lambda: st.norm(0.5, 1.5),
    "Exponential": lambda: st.expon(scale=1 / 1.5),
    "Uniform": lambda: st.uniform(-1.0, 3.0),
    "Beta": lambda: st.beta(2.0, 3.0),
    "Gamma": lambda: st.gamma(2.0, scale=1.5),
    "Chi2": lambda: st.chi2(4),
    "Student_t": lambda: st.t(6),
    "F": lambda: st.f(6, 12),
    "Cauchy": lambda: st.cauchy(0.5, 1.2),
    "Laplace": lambda: st.laplace(0.0, 1.0),
    "Weibull": lambda: st.weibull_min(2.0, scale=1.5),
    "Pareto": lambda: st.pareto(5.0, scale=1.0),
    "LogNormal": lambda: st.lognorm(0.5, scale=np.exp(0.2)),
    "Gumbel": lambda: st.gumbel_r(0.5, 2.0),
    "Frechet": lambda: st.invweibull(4.0),
    "GEV": lambda: st.genextreme(0.1),  # scipy negates the shape convention
    "GPareto": lambda: st.genpareto(0.2),
    "InvGamma": lambda: st.invgamma(5.0, scale=3.0),
    "InvGaussian": lambda: st.invgauss(1.5 / 2.0, scale=2.0),
    "Rayleigh": lambda: st.rayleigh(scale=1.5),
    "Maxwell": lambda: st.maxwell(scale=2.0),
    "Nakagami": lambda: st.nakagami(2.0, scale=1.0),
    "Rice": lambda: st.rice(1.0, scale=1.0),
    "VonMises": None,  # circular conventions differ; covered by dedicated checks
    "Kumaraswamy": None,
    "LevyDistribution": lambda: st.levy(loc=0.5, scale=1.0),
    "SubGaussian": lambda: st.gennorm(3.0, loc=0.2, scale=1.0),
    "SubExponential": lambda: st.weibull_min(0.9, scale=1.5),
}

REF_XS = {
    "Bernoulli": [0, 1], "Binomial": [2, 5], "Poisson": [1, 4], "Geometric": [1, 3],
    "NegBinomial": [2, 5], "Hypergeometric": [3, 5], "DiscreteUniform": [2, 7],
    "ZipfDistribution": [1, 4], "BetaBinomial": [3, 7], "ConwayMaxwellPoisson": [2, 6],
}


@pytest.mark.parametrize("name", sorted(SCIPY_REF))
def test_matches_scipy(name):
    factory = SCIPY_REF[name]
    if factory is None:
        pytest.skip(f"{name}: different convention, no direct scipy mapping")
    make, _, _ = UNIVARIATE[name]
    d = make()
    ref = factory()
    xs = REF_XS.get(name, [-0.8, 0.3, 1.1])
    ours_pmf = getattr(ref, "pmf" if getattr(d, "is_discrete", False) else "pdf")
    assert np.allclose(
        np.atleast_1d(np.asarray(d.pdf(xs), dtype=float)),
        np.atleast_1d(ours_pmf(xs)),
        rtol=1e-6, atol=1e-12,
    ), f"{name}: pdf mismatch"
    assert np.allclose(
        np.atleast_1d(np.asarray(d.cdf(xs), dtype=float)),
        np.atleast_1d(ref.cdf(xs)),
        rtol=1e-6, atol=1e-9,
    ), f"{name}: cdf mismatch"
    for q in (0.2, 0.55, 0.9):
        assert np.isclose(float(d.ppf(q)), float(ref.ppf(q)), rtol=1e-6, atol=1e-9), \
            f"{name}: ppf({q}) mismatch"
    rm, rv = ref.stats()
    om, ov = d.mean(), d.var()
    if np.isfinite(rm):
        assert np.isclose(om, rm, rtol=1e-8, atol=1e-10), f"{name}: mean {om} vs {rm}"
    if np.isfinite(rv):
        assert np.isclose(ov, rv, rtol=1e-8, atol=1e-10), f"{name}: var {ov} vs {rv}"


def test_conway_maxwell_poisson_nu_one_is_poisson():
    d = D.ConwayMaxwellPoisson(3.0, nu=1.0)
    ref = st.poisson(3.0)
    ks = np.arange(0, 25)
    assert np.allclose(d.pmf(ks), ref.pmf(ks), rtol=1e-8)
    assert np.isclose(d.mean(), ref.mean(), rtol=1e-8)


def test_kumaraswamy_mass_via_quadrature():
    d = D.Kumaraswamy(2.0, 3.0)
    xs = np.linspace(0, 1, 200001)
    mass = np.trapezoid(d.pdf(xs), xs)
    assert abs(mass - 1.0) < 1e-6


def test_von_mises_circular_variance_convention():
    from scipy import special

    d = D.VonMises(0.3, 2.0)
    expected = 1 - special.i1(2.0) / special.i0(2.0)
    assert np.isclose(d.var(), expected, rtol=1e-12)


def test_gpareto_support_mask():
    d = D.GPareto(0.0, 1.0, 0.3)
    assert float(d.pdf(-0.5)) == 0.0
    assert float(d.pdf(0.0)) > 0.0


# ---------------------------------------------------------------- masses & tails

CONTINUOUS_FOR_MASS = ["Normal", "Exponential", "Uniform", "Beta", "Gamma", "Chi2",
                       "Laplace", "Weibull", "Rayleigh", "Maxwell"]


@pytest.mark.parametrize("name", CONTINUOUS_FOR_MASS)
def test_pdf_integrates_to_one(name):
    from scipy import integrate as si

    d = UNIVARIATE[name][0]()
    lo, hi = d.support()
    a = lo if np.isfinite(lo) else float(d.ppf(1e-15))
    b = hi if np.isfinite(hi) else float(d.ppf(1 - 1e-15))
    mass, _ = si.quad(lambda t: float(d.pdf(t)), a, b, limit=400)
    assert abs(mass - 1.0) < 1e-4


@pytest.mark.parametrize("name", ["Bernoulli", "Binomial", "Poisson", "Geometric",
                                  "NegBinomial", "DiscreteUniform", "BetaBinomial"])
def test_discrete_pmf_sums_to_one(name):
    d = UNIVARIATE[name][0]()
    grid = d._support_grid(n=4000)
    assert abs(float(np.sum(d.pmf(grid))) - 1.0) < 1e-6


def test_tail_cdf_identities():
    cases = [
        (D.Cauchy(0, 1), 1e-6),
        (D.LevyDistribution(0.0, 1.0), 1e-6),
        (D.Pareto(5.0, 1.0), 1e-8),
    ]
    for d, tol in cases:
        q = 1e-9
        x_hi = float(d.ppf(1 - q))
        assert abs(float(d.cdf(x_hi)) - (1 - q)) < tol, type(d).__name__


# ---------------------------------------------------------------- fit round-trips

FIT_CASES = [
    (D.Normal, lambda n: np.random.default_rng(1).normal(2.0, 3.0, n),
     lambda d: np.isclose(d.mu, 2.0, atol=0.4) and np.isclose(d.sigma, 3.0, atol=0.4)),
    (D.Exponential, lambda n: np.random.default_rng(2).exponential(2.0, n),
     lambda d: np.isclose(d.rate, 0.5, atol=0.15)),
    (D.LogNormal, lambda n: np.random.default_rng(3).lognormal(0.5, 0.5, n),
     lambda d: np.isclose(d.mu, 0.5, atol=0.2) and np.isclose(d.sigma, 0.5, atol=0.15)),
    (D.Rayleigh, lambda n: np.random.default_rng(4).rayleigh(2.0, n),
     lambda d: np.isclose(d.scale, 2.0, atol=0.3)),
    (D.Poisson, lambda n: np.random.default_rng(5).poisson(4.0, n),
     lambda d: np.isclose(d.lam, 4.0, atol=0.5)),
    (D.Geometric, lambda n: np.random.default_rng(6).geometric(0.25, n),
     lambda d: np.isclose(d.p, 0.25, atol=0.05)),
    (D.Gamma, lambda n: np.random.default_rng(7).gamma(3.0, 2.0, n),
     lambda d: np.isclose(d.shape, 3.0, atol=0.8) and np.isclose(d.scale, 2.0, atol=0.5)),
]


@pytest.mark.parametrize("cls,gen,verify", FIT_CASES, ids=[c.__name__ for c, *_ in FIT_CASES])
def test_fit_recovers_parameters(cls, gen, verify):
    fitted = cls.fit(gen(500))
    assert verify(fitted), f"{cls.__name__}.fit params off: {vars(fitted)}"


def test_generic_fit_weibull_roundtrip():
    rng = np.random.default_rng(42)
    data = 2.0 * rng.weibull(2.5, 600)
    fitted = D.Weibull.fit(data)
    assert np.isclose(fitted.shape, 2.5, atol=0.35)
    assert np.isclose(fitted.scale, 2.0, atol=0.35)


def test_ks_test_on_fitted_normal():
    rng = np.random.default_rng(99)
    data = rng.normal(1.0, 2.0, 300)
    fitted = D.Normal.fit(data)
    _, pval = fitted.ks_test(rng.normal(1.0, 2.0, 300))
    assert pval > 0.01


# ---------------------------------------------------------------- stable family

def test_stable_alpha_two_delegates_to_gaussian():
    d = D.StableDistribution(2.0, 0.7, 1.0, 2.0)
    ref = st.norm(1.0, np.sqrt(8))
    for x in (-1.0, 0.0, 2.5):
        assert float(d.pdf(x)) == pytest.approx(float(ref.pdf(x)), rel=1e-12)
        assert float(d.cdf(x)) == pytest.approx(float(ref.cdf(x)), rel=1e-12)
    for q in (0.1, 0.5, 0.95):
        assert float(d.ppf(q)) == pytest.approx(float(ref.ppf(q)), rel=1e-12)
        assert d.mean() == pytest.approx(ref.mean())
        assert d.var() == pytest.approx(ref.var())
        t = 0.8
        expected_cf = np.exp(1j * ref.mean() * t - 0.5 * ref.var() * t**2)
        assert complex(d.cf(t)) == pytest.approx(complex(expected_cf), rel=1e-12)


def test_stable_cauchy_special_case():
    d = D.StableDistribution(1.0, 0.0, -0.5, 1.5)
    ref = st.cauchy(-0.5, 1.5)
    for x in (-2.0, 0.0, 3.0):
        assert float(d.pdf(x)) == pytest.approx(float(ref.pdf(x)), rel=1e-12)
        assert float(d.cdf(x)) == pytest.approx(float(ref.cdf(x)), rel=1e-12)


@pytest.mark.parametrize("alpha,beta", [(1.5, 0.0), (1.5, 0.6), (0.8, -0.4)])
def test_stable_cml_sampler_matches_characteristic_function(alpha, beta):
    d = D.StableDistribution(alpha, beta, 0.5, 2.0)
    x = np.asarray(d.rvs(150_000, random_state=11), dtype=float)
    ts = np.linspace(-2.0, 2.0, 13)
    emp = np.array([np.mean(np.exp(1j * t * x)) for t in ts])
    theo = np.array([d.cf(t) for t in ts])
    assert float(np.max(np.abs(emp - theo))) < 0.02


def test_stable_numeric_inversion_pdf_reasonable():
    # general case alpha=1.5 symmetric: numeric Gil-Pelaez pdf vs CML sample quantiles
    d = D.StableDistribution(1.5, 0.0, 0.0, 1.0)
    x = np.sort(np.asarray(d.rvs(40_000, random_state=5), dtype=float))
    probs = np.array([0.25, 0.5, 0.75])
    sample_q = np.interp(probs, (np.arange(1, len(x) + 1) - 0.5) / len(x), x)
    for p, sq in zip(probs, sample_q):
        assert float(d.cdf(sq)) == pytest.approx(float(p), abs=0.02)


def test_stable_alpha_one_beta_nonzero_rvs_slow_path_runs():
    d = D.StableDistribution(1.0, 0.5, 0.0, 1.0)
    vals = np.asarray(d.rvs(3, random_state=0), dtype=float)
    assert vals.shape == (3,)
    assert np.all(np.isfinite(vals))


# ---------------------------------------------------------------- CLI & selftest

def test_cli_version_flag(capsys):
    rc = cli_mod.main(["--version"])
    out = capsys.readouterr().out.strip()
    assert rc == 0
    assert out == stochpylib.__version__


def test_cli_help_shows_library_overview(capsys):
    with pytest.raises(SystemExit) as excinfo:
        cli_mod.main(["--help"])
    assert excinfo.value.code == 0
    out = capsys.readouterr().out
    # module overview with dynamic inventory
    for token in (
        "probability",
        "distributions",
        "47 distribution classes",
        "Normal",
        "StableDistribution",
        ".ks_test()",
        "Quick start",
        "--version",
        "--test",
    ):
        assert token in out, f"help output missing {token!r}"


def test_cli_selftest_flag_green(capsys):
    rc = cli_mod.main(["--test"])
    out = capsys.readouterr().out
    assert rc == 0
    assert "101 checks" in out or "checks, OK" in out


def test_selftest_run_returns_zero():
    assert selftest_mod.run() == 0


# ---------------------------------------------------------------- doctests

def test_doctests_pass():
    import stochpylib.distributions._base as _base
    import stochpylib.probability as probability_pkg

    total_failed = 0
    buf = io.StringIO()
    with contextlib.redirect_stdout(buf):
        for mod in (_base, probability_pkg.basics, probability_pkg.combinatorics,
                    probability_pkg.independence):
            total_failed += doctest.testmod(mod, verbose=False).failed
    assert total_failed == 0, buf.getvalue()
