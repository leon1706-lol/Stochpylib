"""End-to-end API sweep for stochpylib.bayesian: one realistic exercise per public
name. ``test_every_public_name_is_exercised`` fails the moment a name ships without
one; ``test_exercise[<name>]`` runs each exercise as its own pytest case."""

import numpy as np
import pytest

from stochpylib import bayesian as mod
from stochpylib.distributions import Beta, Normal

EXERCISES = {}


def exercise(name):
    def deco(fn):
        EXERCISES[name] = fn
        return fn
    return deco


def _finite(a):
    return bool(np.all(np.isfinite(np.asarray(a, dtype=float))))


_RNG = np.random.default_rng(0)
_COIN = _RNG.binomial(1, 0.6, 60)


# --------------------------------------------------------------------------- core

@exercise("Prior")
def _prior_cls():
    p = mod.Prior(Beta(2, 2))
    assert p.dim == 1 and _finite(p.logpdf(np.array([0.5])))


@exercise("prior")
def _prior_fn():
    p = mod.prior(Normal(0, 1))
    assert _finite(p.sample(3, random_state=0))


@exercise("Likelihood")
def _likelihood_cls():
    lik = mod.Likelihood("bernoulli", data=_COIN)
    assert _finite(lik.loglik(np.array([0.5])))


@exercise("likelihood")
def _likelihood_fn():
    lik = mod.likelihood("poisson", data=_RNG.poisson(3.0, 40))
    assert _finite(lik.pointwise(np.array([3.0])))


@exercise("ConjugateFamily")
def _conjugate_family():
    cf = mod.ConjugateFamily("bernoulli")
    post = cf.update(Beta(1, 1), _COIN)
    assert post.a > 1


@exercise("conjugate_prior")
def _conjugate_prior_fn():
    cf = mod.conjugate_prior("bernoulli", a=2.0, b=2.0)
    assert cf.prior_.a == 2.0


@exercise("posterior")
def _posterior_fn():
    post = mod.posterior(mod.prior(Beta(2, 2)), mod.likelihood("bernoulli", data=_COIN))
    assert post.mean()[0] == pytest.approx(0.6, abs=0.15)


@exercise("Posterior")
def _posterior_cls():
    post = mod.Posterior("conjugate", 1, dist=Beta(3, 3), mean_=[0.5], cov_=[[0.05]])
    assert post.sample(2, random_state=0).shape == (2, 1)


@exercise("bayes_update")
def _bayes_update_fn():
    post = mod.bayes_update(Beta(1, 1), _COIN, "bernoulli")
    assert post.a > 1


@exercise("posterior_predictive")
def _posterior_predictive_fn():
    post = mod.posterior(mod.prior(Beta(2, 2)), mod.likelihood("bernoulli", data=_COIN),
                          method="conjugate")
    pred = mod.posterior_predictive(post)
    assert 0 <= pred.mean() <= 1


@exercise("evidence")
def _evidence_fn():
    Z = mod.evidence(mod.prior(Beta(2, 2)), mod.likelihood("bernoulli", data=_COIN))
    assert np.isfinite(Z)


# -------------------------------------------------------------------- computation

@exercise("LaplacePosterior")
def _laplace():
    lap = mod.LaplacePosterior(lambda t: -0.5 * t[0] ** 2, np.array([1.0]))
    assert lap.mean_[0] == pytest.approx(0.0, abs=1e-3)


@exercise("EP_Posterior")
def _ep():
    rng = np.random.default_rng(1)
    A = rng.standard_normal((10, 1))
    y = A[:, 0] * 2.0 + rng.standard_normal(10) * 0.3
    ep = mod.EP_Posterior(np.zeros(1), np.eye(1) * 5,
                           A, lambda f, i: -0.5 * ((y[i] - f) / 0.3) ** 2)
    assert _finite(ep.mean_)


@exercise("MFVariational")
def _mfvi():
    vi = mod.MFVariational(lambda t: -0.5 * (t[0] - 1.0) ** 2, 1, n_iter=300, random_state=0)
    assert vi.mean_[0] == pytest.approx(1.0, abs=0.3)


@exercise("ImportanceSamplingPosterior")
def _isp():
    isp = mod.ImportanceSamplingPosterior(lambda t: -0.5 * t[0] ** 2, theta0=np.zeros(1), n=1000,
                                           random_state=0)
    assert isp.mean_[0] == pytest.approx(0.0, abs=0.3)


@exercise("PosteriorApproximation")
def _posterior_approx_cls():
    pa = mod.PosteriorApproximation("laplace", [0.0], [[1.0]])
    assert pa.sample(2, random_state=0).shape == (2, 1)


# ---------------------------------------------------------------------- selection

@exercise("AIC")
def _aic():
    assert mod.AIC(-100.0, k=3).value == pytest.approx(206.0)


@exercise("BIC")
def _bic():
    assert mod.BIC(-100.0, k=3, n=50).value == pytest.approx(211.74, abs=0.1)


@exercise("DIC")
def _dic():
    samples = np.random.default_rng(0).normal(0, 1, (200, 1))
    d = mod.DIC(lambda t: -0.5 * t[0] ** 2, samples)
    assert np.isfinite(d.value)


@exercise("WAIC")
def _waic():
    ll = np.random.default_rng(0).standard_normal((300, 10))
    assert np.isfinite(mod.WAIC(ll).value)


@exercise("LOO_CV")
def _loo():
    ll = np.random.default_rng(0).standard_normal((300, 10))
    assert np.isfinite(mod.LOO_CV(ll).value)


@exercise("TICfit")
def _tic():
    rng = np.random.default_rng(0)
    x = rng.normal(0, 1, 100)
    tic = mod.TICfit(lambda t: -0.5 * np.sum((x - t[0]) ** 2), np.array([x.mean()]),
                      pointwise=lambda t: -0.5 * (x - t[0]) ** 2)
    assert tic.p_eff > 0


@exercise("bayes_factor")
def _bf():
    bf = mod.bayes_factor(-100.0, -102.0)
    assert bf.value > 1


@exercise("ICResult")
def _ic_result():
    ic = mod.ICResult("AIC", 12.3)
    assert float(ic) == 12.3


# ------------------------------------------------------------------------- models

@exercise("BayesianLinear")
def _bayeslin():
    rng = np.random.default_rng(0)
    X = rng.standard_normal((60, 1))
    y = 2.0 * X[:, 0] + rng.standard_normal(60) * 0.3
    m = mod.BayesianLinear().fit(X, y)
    assert m.coef_[1] == pytest.approx(2.0, abs=0.3)


@exercise("BayesianLogistic")
def _bayeslogit():
    rng = np.random.default_rng(0)
    X = rng.standard_normal((150, 1))
    y = (rng.uniform(size=150) < 1 / (1 + np.exp(-2 * X[:, 0]))).astype(float)
    m = mod.BayesianLogistic().fit(X, y)
    assert m.coef_[1] > 0


@exercise("NaiveBayes")
def _naivebayes():
    rng = np.random.default_rng(0)
    X = np.vstack([rng.normal(0, 0.3, (30, 2)), rng.normal(4, 0.3, (30, 2))])
    y = np.array([0] * 30 + [1] * 30)
    nb = mod.NaiveBayes("gaussian").fit(X, y)
    assert nb.score(X, y) > 0.9


@exercise("HierarchicalModel")
def _hierarchical():
    y = np.array([28, 8, -3, 7, -1, 1, 18, 12], dtype=float)
    s = np.array([15, 10, 16, 11, 9, 11, 10, 18], dtype=float)
    hm = mod.HierarchicalModel(n_samples=800, n_warmup=300, random_state=0).fit(y, sigma=s)
    assert hm.theta_mean_.shape == (8,)


@exercise("MixtureModel")
def _mixture():
    rng = np.random.default_rng(0)
    x = np.concatenate([rng.normal(-4, 0.5, 60), rng.normal(4, 0.5, 60)])
    mm = mod.MixtureModel(2, n_samples=300, n_warmup=150, random_state=0).fit(x)
    assert mm.means_.shape == (2, 1)


@exercise("BayesianNetwork")
def _bayesnet():
    bn = mod.BayesianNetwork()
    bn.add_node("A", [0, 1])
    bn.add_node("B", [0, 1])
    bn.add_edge("A", "B")
    bn.set_cpt("A", [0.5, 0.5])
    bn.set_cpt("B", [[0.9, 0.1], [0.2, 0.8]])
    res = bn.query(["B"], {"A": 1})
    assert res[1] == pytest.approx(0.8)


@exercise("DirichletProcess")
def _dp():
    dp = mod.DirichletProcess(alpha=1.0)
    labels = dp.crp(30, random_state=0)
    assert len(labels) == 30


# -------------------------------------------------------------------------- results

@exercise("EmpiricalPredictive")
def _empirical_predictive():
    ep = mod.EmpiricalPredictive(np.random.default_rng(0).normal(0, 1, 500))
    assert _finite(ep.mean()) and _finite(ep.pdf(0.0))


def test_every_public_name_is_exercised():
    assert set(EXERCISES) == set(mod.__all__), (
        set(mod.__all__) - set(EXERCISES), set(EXERCISES) - set(mod.__all__))


@pytest.mark.parametrize("name", sorted(mod.__all__))
def test_exercise(name):
    EXERCISES[name]()
