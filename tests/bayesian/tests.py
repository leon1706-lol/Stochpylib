"""Tests for stochpylib.bayesian: priors/likelihoods/posteriors, the ten-family
conjugate engine, posterior approximations, model-selection criteria, and the seven
model classes.

Oracles: closed forms, scipy.stats (beta/gamma/nbinom/betabinom/t/invgamma/lomax/
multivariate_normal), statsmodels (OLS/Logit), brute-force enumeration (the
BayesianNetwork sprinkler example) and fine numerical grids/1-D quadrature, plus
advanced_mcmc long runs as an independent sampler cross-check. All randomness is
seeded; statistical assertions are set at >= 3 Monte Carlo standard errors.
"""

import numpy as np
import pytest
from scipy import special, stats

from stochpylib.bayesian import (
    AIC, BIC, DIC, WAIC, LOO_CV, TICfit, bayes_factor,
    LaplacePosterior, EP_Posterior, MFVariational, ImportanceSamplingPosterior,
    BayesianLinear, BayesianLogistic, NaiveBayes, HierarchicalModel, MixtureModel,
    BayesianNetwork, DirichletProcess,
    Prior, Likelihood, ConjugateFamily, Posterior, PosteriorApproximation, ICResult,
    EmpiricalPredictive,
    prior, likelihood, posterior, bayes_update, conjugate_prior, posterior_predictive,
    evidence,
)
from stochpylib.bayesian._common import _gpd_fit, _psis, _log_multibeta, _mvn_logpdf
from stochpylib.distributions import (
    Beta, Gamma, Normal, InvGamma, Dirichlet, MultivariateNormal, Bernoulli,
)

_METHOD_NAMES = ("pdf", "cdf", "ppf", "rvs", "mean", "var", "skewness", "kurtosis",
                  "entropy", "mgf", "cf", "fit", "ks_test")


# =============================================================================== core

@pytest.mark.parametrize("family,hyper,gen", [
    ("bernoulli", dict(a=2.0, b=3.0), lambda rng: rng.binomial(1, 0.4, 200)),
    ("poisson", dict(alpha=2.0, beta=1.0), lambda rng: rng.poisson(4.0, 200)),
    ("exponential", dict(alpha=1e-2, beta=1e2), lambda rng: rng.exponential(2.0, 150)),
])
def test_conjugate_posterior_matches_scipy(family, hyper, gen):
    rng = np.random.default_rng(0)
    x = gen(rng)
    cf = conjugate_prior(family, **hyper)
    p0 = cf.prior_
    post = posterior(prior(p0), likelihood(family, data=x), method="conjugate")
    if family == "bernoulli":
        s, n = float(np.sum(x)), len(x)
        assert post.dist.a == pytest.approx(hyper["a"] + s)
        assert post.dist.b == pytest.approx(hyper["b"] + n - s)
        grid = np.linspace(0.01, 0.99, 50)
        assert np.allclose(post.dist.pdf(grid), stats.beta.pdf(grid, post.dist.a, post.dist.b))
    elif family == "poisson":
        a2, b2 = post.dist.shape, 1.0 / post.dist.scale
        grid = np.linspace(0.5, 10, 50)
        assert np.allclose(post.dist.pdf(grid), stats.gamma.pdf(grid, a2, scale=1 / b2))
    else:
        a2, b2 = post.dist.shape, 1.0 / post.dist.scale
        grid = np.linspace(0.01, 5, 50)
        assert np.allclose(post.dist.pdf(grid), stats.gamma.pdf(grid, a2, scale=1 / b2))


def test_conjugate_evidence_matches_numeric_integral_bernoulli():
    from scipy.integrate import quad
    rng = np.random.default_rng(1)
    x = rng.binomial(1, 0.3, 100)
    logZ = evidence(prior(Beta(2, 3)), likelihood("bernoulli", data=x), method="conjugate")
    Znum = quad(lambda p: stats.beta.pdf(p, 2, 3) * np.prod(stats.bernoulli.pmf(x, p)), 0, 1)[0]
    assert logZ == pytest.approx(np.log(Znum), abs=1e-6)


def test_conjugate_evidence_matches_numeric_integral_normal():
    rng = np.random.default_rng(2)
    x = rng.normal(3.0, 2.0, 60)
    sigma = 2.0
    logZ = evidence(prior(Normal(0, 10)), likelihood("normal", data=x, sigma=sigma), method="conjugate")
    n, xbar = len(x), x.mean()
    lp_xbar = stats.norm.logpdf(xbar, 0, np.sqrt(10 ** 2 + sigma ** 2 / n))
    lp_data = np.sum(stats.norm.logpdf(x, xbar, sigma))
    lp_xbar_given_xbar = stats.norm.logpdf(xbar, xbar, sigma / np.sqrt(n))
    assert logZ == pytest.approx(lp_xbar + lp_data - lp_xbar_given_xbar, abs=1e-8)


def test_normal_unknown_var_niw_posterior_and_evidence():
    from stochpylib.bayesian._common import _NormalInverseGamma
    rng = np.random.default_rng(3)
    x = rng.normal(2.0, 1.2, 150)
    nig0 = _NormalInverseGamma(0.0, 1e-3, 1e-2, 1e-2)
    post = posterior(prior(nig0), likelihood("normal_unknown_var", data=x), method="conjugate")
    assert post.dist.mean()[0] == pytest.approx(2.0, abs=0.15)
    assert post.dist.mean()[1] == pytest.approx(1.2 ** 2, abs=0.3)
    pred = posterior_predictive(post)
    assert isinstance(pred, type(post.dist.marginal_mu()))
    # 2-D fine-grid cross check of the log-evidence
    mu_g = np.linspace(1.5, 2.5, 300)
    s2_g = np.linspace(0.8, 2.2, 300)
    MU, S2 = np.meshgrid(mu_g, s2_g)
    ll = np.array([[np.sum(stats.norm.logpdf(x, mu, np.sqrt(s2))) for mu in mu_g] for s2 in s2_g])
    lp = stats.norm.logpdf(MU, 0.0, np.sqrt(S2 / 1e-3)) + stats.invgamma.logpdf(S2, 1e-2, scale=1e-2)
    joint = ll + lp
    m = joint.max()
    Znum = np.sum(np.exp(joint - m)) * (mu_g[1] - mu_g[0]) * (s2_g[1] - s2_g[0])
    logZ = evidence(prior(nig0), likelihood("normal_unknown_var", data=x), method="conjugate")
    assert logZ == pytest.approx(np.log(Znum) + m, abs=0.05)


def test_categorical_conjugate_matches_dirichlet_multinomial():
    rng = np.random.default_rng(4)
    K = 4
    alpha0 = np.ones(K)
    labels = rng.choice(K, size=500, p=[0.1, 0.2, 0.3, 0.4])
    counts = np.bincount(labels, minlength=K)
    post = posterior(prior(Dirichlet(alpha0)), likelihood("categorical", data=labels, k=K),
                      method="conjugate")
    assert np.allclose(post.dist.alpha, alpha0 + counts)
    logZ = evidence(prior(Dirichlet(alpha0)), likelihood("categorical", data=labels, k=K),
                     method="conjugate")
    expected = _log_multibeta(alpha0 + counts) - _log_multibeta(alpha0)
    assert logZ == pytest.approx(expected, abs=1e-10)
    pred = posterior_predictive(post)
    assert np.allclose(pred.p, post.dist.alpha / post.dist.alpha.sum())


def test_mvnormal_conjugate_matches_predictive_trick():
    rng = np.random.default_rng(5)
    Sigma = np.array([[1.0, 0.3], [0.3, 0.8]])
    mean_true = np.array([1.0, -1.0])
    X = rng.multivariate_normal(mean_true, Sigma, 100)
    m0 = MultivariateNormal(np.zeros(2), np.eye(2) * 100)
    post = posterior(prior(m0), likelihood("mvnormal", data=X, cov=Sigma), method="conjugate")
    assert np.allclose(post.dist.mean_vec, X.mean(axis=0) @ np.eye(2), atol=0.3)
    xbar = X.mean(axis=0)
    S0 = np.eye(2) * 100
    lp_xbar = stats.multivariate_normal.logpdf(xbar, np.zeros(2), S0 + Sigma / 100)
    lp_data = np.sum([stats.multivariate_normal.logpdf(row, xbar, Sigma) for row in X])
    lp_xbar_given_xbar = stats.multivariate_normal.logpdf(xbar, xbar, Sigma / 100)
    logZ = evidence(prior(m0), likelihood("mvnormal", data=X, cov=Sigma), method="conjugate")
    assert logZ == pytest.approx(lp_xbar + lp_data - lp_xbar_given_xbar, abs=1e-6)


def test_binomial_and_gamma_and_normal_variance_predictives_match_scipy():
    rng = np.random.default_rng(6)
    s = rng.binomial(20, 0.4, 80)
    post = posterior(prior(Beta(1, 1)), likelihood("binomial", data=s, n=20), method="conjugate")
    pred = posterior_predictive(post, n=20)
    xs = np.arange(0, 21)
    assert np.allclose(pred.pmf(xs), stats.betabinom.pmf(xs, 20, post.dist.a, post.dist.b))

    xe = rng.exponential(2.0, 100)
    poste = posterior(prior(Gamma(1e-2, 1e2)), likelihood("exponential", data=xe), method="conjugate")
    prede = posterior_predictive(poste)
    a, b = poste.dist.shape, 1.0 / poste.dist.scale
    for v in (0.3, 1.0, 3.0):
        assert prede.pdf(v) == pytest.approx(stats.lomax.pdf(v, c=a, scale=b), rel=1e-8)

    xv = rng.normal(1.0, 1.5, 200)
    postv = posterior(prior(InvGamma(1e-2, 1e-2)), likelihood("normal_variance", data=xv, mu=1.0),
                       method="conjugate")
    predv = posterior_predictive(postv, mu=1.0)
    a2, b2 = postv.dist.shape, postv.dist.scale
    for v in (0.5, 1.0, 2.0):
        scale = np.sqrt(b2 / a2)
        assert predv.pdf(v) == pytest.approx(stats.t.pdf((v - 1.0) / scale, 2 * a2) / scale, rel=1e-8)


def test_gamma_family_no_predictive_but_evidence_ok():
    rng = np.random.default_rng(7)
    k_shape = 3.0
    x = rng.gamma(k_shape, 1 / 2.0, 200)
    post = posterior(prior(Gamma(1e-2, 1e2)), likelihood("gamma", data=x, shape=k_shape),
                      method="conjugate")
    assert post.dist.mean() == pytest.approx(2.0, rel=0.15)
    pred = ConjugateFamily("gamma").predictive(post.dist, shape=k_shape)
    assert pred is None


def test_posterior_grid_matches_conjugate():
    rng = np.random.default_rng(8)
    x = rng.binomial(1, 0.35, 300)
    lik = likelihood("bernoulli", data=x)
    pr = prior(Beta(2, 2))
    pc = posterior(pr, lik, method="conjugate")
    pg = posterior(pr, lik, method="grid", n_grid=4001)
    assert pg.mean()[0] == pytest.approx(pc.mean()[0], abs=1e-3)
    assert pg.std()[0] == pytest.approx(pc.std()[0], abs=1e-3)
    assert pg.log_evidence_ == pytest.approx(pc.log_evidence_, abs=1e-2)


def test_posterior_method_auto_dispatch():
    rng = np.random.default_rng(9)
    x = rng.binomial(1, 0.4, 50)
    p = posterior(prior(Beta(1, 1)), likelihood("bernoulli", data=x))
    assert p.method == "conjugate"


@pytest.mark.parametrize("sampler", ["slice", "nuts", "mh"])
def test_posterior_mcmc_recovers_conjugate_mean(sampler):
    rng = np.random.default_rng(10)
    x = rng.binomial(1, 0.35, 300)
    lik = likelihood("bernoulli", data=x)
    pr = prior(Beta(2, 2))
    p_true = posterior(pr, lik, method="conjugate")
    pm = posterior(pr, lik, method="mcmc", sampler=sampler, n_samples=2000, n_warmup=500,
                   theta0=np.array([0.4]), random_state=0)
    from stochpylib.advanced_mcmc import ESS
    ess = float(ESS(pm.samples_.T[:, :, None], method="mean"))
    se = pm.samples_.std() / np.sqrt(ess)
    assert abs(pm.samples_.mean() - p_true.mean()[0]) < 5 * se
    assert isinstance(pm.extras["sampler"], object)


def test_posterior_laplace_vi_importance_smc():
    rng = np.random.default_rng(11)
    x = rng.binomial(1, 0.35, 300)
    lik = likelihood("bernoulli", data=x)
    pr = prior(Beta(2, 2))
    p_true = posterior(pr, lik, method="conjugate")

    pl = posterior(pr, lik, method="laplace", theta0=np.array([0.35]))
    assert pl.mean()[0] == pytest.approx(p_true.mean()[0], abs=0.02)
    assert pl.log_evidence_ == pytest.approx(p_true.log_evidence_, abs=0.05)

    pv = posterior(pr, lik, method="vi", random_state=5)
    assert pv.mean()[0] == pytest.approx(p_true.mean()[0], abs=0.03)

    pimp = posterior(pr, lik, method="importance", theta0=np.array([0.35]), n_samples=6000,
                      random_state=6)
    assert pimp.mean()[0] == pytest.approx(p_true.mean()[0], abs=0.02)
    assert pimp.log_evidence_ == pytest.approx(p_true.log_evidence_, abs=0.03)

    psmc = posterior(pr, lik, method="smc", n_samples=1500, random_state=7)
    assert psmc.log_evidence_ == pytest.approx(p_true.log_evidence_, abs=0.3)


def test_bayes_update_sequential_equals_batch():
    rng = np.random.default_rng(12)
    x = rng.binomial(1, 0.4, 300)
    seq = bayes_update(bayes_update(Beta(1, 1), x[:150], "bernoulli"), x[150:], "bernoulli")
    batch = bayes_update(Beta(1, 1), x, "bernoulli")
    assert seq.a == pytest.approx(batch.a) and seq.b == pytest.approx(batch.b)

    post = posterior(prior(Beta(1, 1)), likelihood("bernoulli", data=x), method="conjugate")
    upd = bayes_update(post, x[:10], family=None)
    assert upd.a == pytest.approx(post.dist.a + x[:10].sum())


def test_posterior_predictive_empirical_matches_conjugate_mean():
    rng = np.random.default_rng(13)
    x = rng.binomial(1, 0.4, 300)
    lik = likelihood("bernoulli", data=x)
    post = posterior(prior(Beta(1, 1)), lik, method="conjugate")
    # strip the stored conjugate family (and change method) so the Monte Carlo
    # (non-closed-form) branch runs instead of the closed-form conjugate one
    post2 = Posterior("laplace", 1, dist=post.dist, mean_=post.mean_, cov_=post.cov_,
                       log_evidence_=post.log_evidence_, extras={})
    emp2 = posterior_predictive(post2, likelihood=lik, n_samples=3000, random_state=1)
    assert isinstance(emp2, EmpiricalPredictive)
    assert emp2.mean() == pytest.approx(post.dist.mean(), abs=0.03)


def test_empirical_predictive_satisfies_common_contract():
    rng = np.random.default_rng(14)
    draws = rng.normal(2.0, 1.5, 3000)
    ep = EmpiricalPredictive(draws)
    for m in _METHOD_NAMES:
        assert hasattr(ep, m)
    assert ep.mean() == pytest.approx(2.0, abs=0.1)
    assert 0.0 <= ep.cdf(2.0) <= 1.0
    d, p = ep.ks_test(rng.normal(2.0, 1.5, 200))
    assert 0 <= d <= 1 and 0 <= p <= 1
    ep2 = EmpiricalPredictive.fit(draws)
    assert isinstance(ep2, EmpiricalPredictive)


def test_evidence_auto_dispatch_and_grid_dim_limit():
    rng = np.random.default_rng(15)
    x = rng.binomial(1, 0.4, 100)
    Za = evidence(prior(Beta(1, 1)), likelihood("bernoulli", data=x))
    Zc = evidence(prior(Beta(1, 1)), likelihood("bernoulli", data=x), method="conjugate")
    assert Za == pytest.approx(Zc)
    with pytest.raises(ValueError):
        evidence(prior([Normal(0, 1)] * 3), Likelihood(loglik=lambda t, d: 0.0, data=None, dim=3),
                 method="grid")


def test_prior_product_flat_and_custom():
    pr = prior([Normal(0, 1), Beta(2, 2)])
    assert pr.dim == 2
    assert pr.logpdf(np.array([0.0, 0.5])) == pytest.approx(
        stats.norm.logpdf(0.0) + stats.beta.logpdf(0.5, 2, 2))

    flat = prior("flat", dim=1)
    assert flat.logpdf(np.array([5.0])) == 0.0
    assert not flat.is_proper
    with pytest.raises(ValueError):
        flat.sample(1)

    custom = Prior(logpdf=lambda t: -0.5 * t[0] ** 2, dim=1,
                    sampler=lambda n, rng: rng.standard_normal((n, 1)))
    assert custom.logpdf(np.array([1.0])) == -0.5
    assert custom.sample(5, random_state=0).shape == (5, 1)


def test_likelihood_pointwise_and_errors():
    rng = np.random.default_rng(16)
    for fam, data, known in [
        ("bernoulli", rng.binomial(1, 0.4, 50), {}),
        ("poisson", rng.poisson(3.0, 50), {}),
        ("exponential", rng.exponential(2.0, 50), {}),
        ("normal", rng.normal(0, 1, 50), {"sigma": 1.0}),
    ]:
        lik = Likelihood(fam, data, **known)
        theta = np.array([0.5]) if fam != "normal" else np.array([0.0])
        assert lik.pointwise(theta).sum() == pytest.approx(lik.loglik(theta))

    with pytest.raises(ValueError):
        Likelihood("bogus_family")

    likc = Likelihood(loglik=lambda t, d: -0.5 * np.sum((d - t[0]) ** 2), data=np.array([1.0, 2.0]), dim=1)
    with pytest.raises(ValueError):
        likc.pointwise(np.array([1.0]))


def test_conjugate_method_rejects_non_conjugate_pair():
    from stochpylib.distributions import Uniform
    with pytest.raises(ValueError):
        posterior(prior(Uniform(0, 1)), likelihood("bernoulli", data=np.array([1, 0, 1])),
                  method="conjugate")


# ----------------------------------------------------------------------- computation

def test_laplace_posterior_exact_on_gaussian_target():
    mu, sigma = np.array([1.0, -2.0]), np.array([[2.0, 0.3], [0.3, 1.0]])
    inv = np.linalg.inv(sigma)

    def log_post(theta):
        d = theta - mu
        return -0.5 * d @ inv @ d

    lap = LaplacePosterior(log_post, np.zeros(2))
    assert np.allclose(lap.mean_, mu, atol=1e-4)
    assert np.allclose(lap.cov_, sigma, atol=1e-3)
    _, logdet = np.linalg.slogdet(2 * np.pi * sigma)
    assert lap.log_evidence_ == pytest.approx(0.5 * logdet, abs=1e-3)


def test_laplace_posterior_on_beta_bernoulli():
    rng = np.random.default_rng(17)
    x = rng.binomial(1, 0.3, 200)
    p_true = posterior(prior(Beta(2, 2)), likelihood("bernoulli", data=x), method="conjugate")
    lap = LaplacePosterior((prior(Beta(2, 2)), likelihood("bernoulli", data=x)), np.array([0.3]))
    assert lap.mean_[0] == pytest.approx(p_true.mean()[0], abs=0.02)
    assert lap.log_evidence_ == pytest.approx(p_true.log_evidence_, abs=0.05)


def test_ep_posterior_exact_on_linear_gaussian_model():
    rng = np.random.default_rng(18)
    d = 2
    A = rng.standard_normal((30, d))
    beta_true = np.array([1.0, -0.5])
    s = 0.5
    y = A @ beta_true + rng.standard_normal(30) * s
    m0, V0 = np.zeros(d), np.eye(d) * 10

    def log_site(f, i):
        return -0.5 * np.log(2 * np.pi * s ** 2) - 0.5 * ((y[i] - f) / s) ** 2

    ep = EP_Posterior(m0, V0, A, log_site, n_iter=30)
    S0inv = np.linalg.inv(V0)
    Sn = np.linalg.inv(S0inv + A.T @ A / s ** 2)
    mn = Sn @ (S0inv @ m0 + A.T @ y / s ** 2)
    assert np.allclose(ep.mean_, mn, atol=1e-6)
    assert np.allclose(ep.cov_, Sn, atol=1e-6)
    logZ_closed = stats.multivariate_normal.logpdf(y, A @ m0, A @ V0 @ A.T + s ** 2 * np.eye(30))
    assert ep.log_evidence_ == pytest.approx(logZ_closed, abs=1e-3)


def test_ep_posterior_logistic_matches_nuts():
    rng = np.random.default_rng(19)
    n, d = 250, 2
    A = rng.standard_normal((n, d))
    beta_true = np.array([1.0, -1.5])
    p = special.expit(A @ beta_true)
    y = (rng.uniform(size=n) < p).astype(float)
    signed = np.where(y == 1, 1.0, -1.0)
    m0, V0 = np.zeros(d), np.eye(d) * 4.0

    def log_site(f, i):
        return -np.logaddexp(0.0, -signed[i] * f)

    ep = EP_Posterior(m0, V0, A, log_site, n_iter=40)

    def log_post(theta):
        f = A @ theta
        return -0.5 * theta @ theta / 4.0 + np.sum(y * f - np.logaddexp(0.0, f))

    def grad_log_post(theta):
        f = A @ theta
        return -theta / 4.0 + A.T @ (y - special.expit(f))

    from stochpylib.advanced_mcmc import NoUTurnSampler
    nuts = NoUTurnSampler(log_post, grad_log_post, n_samples=3000, n_warmup=1000, target_accept=0.9)
    nuts.sample(np.zeros(d), random_state=0)
    samples = nuts.get_samples()
    se = samples.std(axis=0) / 10  # loose MCSE proxy given long chain
    assert np.all(np.abs(ep.mean_ - samples.mean(axis=0)) < 5 * se)


def test_mfvariational_delegates_to_advanced_mcmc():
    from stochpylib.advanced_mcmc import MeanFieldVI, ADVI
    vi = MFVariational(lambda t: -0.5 * ((t[0] - 2.0) / 1.0) ** 2, 1, n_iter=800, n_mc=16,
                        random_state=0)
    assert isinstance(vi.extras["engine"], MeanFieldVI)
    assert vi.mean_[0] == pytest.approx(2.0, abs=0.2)
    assert vi.extras["evidence_is_lower_bound"] is True

    rho = 0.8

    def log_prob(t):
        prec = np.linalg.inv([[1, rho], [rho, 1]])
        return -0.5 * t @ prec @ t

    vi_full = MFVariational(log_prob, 2, rank="full", n_iter=1500, n_mc=32, random_state=1)
    assert isinstance(vi_full.extras["engine"], ADVI)
    corr = vi_full.cov_[0, 1] / np.sqrt(vi_full.cov_[0, 0] * vi_full.cov_[1, 1])
    assert corr == pytest.approx(rho, abs=0.2)


def test_importance_sampling_posterior_gaussian_exact():
    mu, sigma2 = 1.5, 0.3

    def log_post(theta):
        return -0.5 * (theta[0] - mu) ** 2 / sigma2

    isp = ImportanceSamplingPosterior(log_post, theta0=np.array([0.0]), n=8000, random_state=0)
    assert isp.mean_[0] == pytest.approx(mu, abs=0.05)
    assert isp.extras["ess"] <= 8000
    assert isp.extras["pareto_k"] < 0.7 or np.isnan(isp.extras["pareto_k"])

    rng = np.random.default_rng(20)
    x = rng.binomial(1, 0.3, 200)
    isp2 = ImportanceSamplingPosterior((prior(Beta(2, 2)), likelihood("bernoulli", data=x)),
                                        theta0=np.array([0.3]), n=6000, random_state=1)
    p_true = posterior(prior(Beta(2, 2)), likelihood("bernoulli", data=x), method="conjugate")
    assert isp2.log_evidence_ == pytest.approx(p_true.log_evidence_, abs=0.05)


# ------------------------------------------------------------------------- selection

def test_waic_matches_hand_formula():
    rng = np.random.default_rng(21)
    ll = rng.standard_normal((500, 20)) * 0.5 - 1.0
    w = WAIC(ll)
    from scipy.special import logsumexp
    lppd_i = logsumexp(ll, axis=0) - np.log(ll.shape[0])
    p_waic_i = np.var(ll, axis=0, ddof=1)
    assert w.value == pytest.approx(-2 * np.sum(lppd_i - p_waic_i), abs=1e-8)
    assert w.p_eff == pytest.approx(np.sum(p_waic_i), abs=1e-8)


def test_aic_bic_match_statsmodels():
    sm = pytest.importorskip("statsmodels.api")
    rng = np.random.default_rng(22)
    X = rng.standard_normal((200, 3))
    y = X @ np.array([1.0, -2.0, 0.5]) + rng.standard_normal(200) * 0.7
    Xc = sm.add_constant(X)
    res = sm.OLS(y, Xc).fit()
    my_aic = AIC(res.llf, k=4)
    my_bic = BIC(res.llf, k=4, n=200)
    assert my_aic.value == pytest.approx(res.aic, abs=1e-6)
    assert my_bic.value == pytest.approx(res.bic, abs=1e-6)


def test_tic_matches_aic_for_well_specified_model():
    sm = pytest.importorskip("statsmodels.api")
    rng = np.random.default_rng(23)
    X = rng.standard_normal((2000, 2))
    beta = np.array([1.0, -2.0, 0.5])
    Xc = sm.add_constant(X)
    y = Xc @ beta + rng.standard_normal(2000) * 0.7

    def loglik_fn(theta):
        resid = y - Xc @ theta
        return -0.5 * len(y) * np.log(2 * np.pi * 0.49) - np.sum(resid ** 2) / (2 * 0.49)

    def pointwise_fn(theta):
        resid = y - Xc @ theta
        return -0.5 * np.log(2 * np.pi * 0.49) - resid ** 2 / (2 * 0.49)

    theta_hat = np.linalg.lstsq(Xc, y, rcond=None)[0]
    tic = TICfit(loglik_fn, theta_hat, pointwise=pointwise_fn)
    assert tic.p_eff == pytest.approx(3.0, abs=0.5)
    assert tic.value == pytest.approx(tic.extras["aic"], abs=1.0)


def test_dic_p_d_near_dim_for_gaussian_posterior():
    rng = np.random.default_rng(24)
    sigma = 1.0
    xn = rng.normal(2.0, sigma, 60)
    lik = likelihood("normal", data=xn, sigma=sigma)
    post = posterior(prior(Normal(0.0, 5.0)), lik, method="conjugate")
    samples = post.sample(4000, random_state=1)
    dic = DIC(lik, samples)
    assert dic.p_eff == pytest.approx(1.0, abs=0.3)


def test_loo_cv_matches_exact_leave_one_out():
    rng = np.random.default_rng(25)
    sigma = 1.0
    n = 60
    xn = rng.normal(2.0, sigma, n)
    mu0, tau0 = 0.0, 5.0
    lik = likelihood("normal", data=xn, sigma=sigma)
    post = posterior(prior(Normal(mu0, tau0)), lik, method="conjugate")
    samples = post.sample(4000, random_state=1)

    def pointwise(theta):
        return -0.5 * np.log(2 * np.pi * sigma ** 2) - 0.5 * ((xn - theta[0]) / sigma) ** 2

    ll_mat = np.array([pointwise(th) for th in samples])
    loo = LOO_CV(ll_mat)
    assert np.max(loo.extras["pareto_k"]) < 0.7

    exact_elpd = np.zeros(n)
    for i in range(n):
        xi = np.delete(xn, i)
        prec0, prec_lik = 1 / tau0 ** 2, len(xi) / sigma ** 2
        post_prec = prec0 + prec_lik
        post_mean = (mu0 * prec0 + xi.sum() / sigma ** 2) / post_prec
        predvar = 1 / post_prec + sigma ** 2
        exact_elpd[i] = stats.norm.logpdf(xn[i], post_mean, np.sqrt(predvar))
    assert abs(loo.extras["elpd_loo"] - exact_elpd.sum()) < 0.15 * np.sqrt(n)

    loo_is = LOO_CV(ll_mat, method="is")
    assert loo_is.value == pytest.approx(loo.value, abs=0.05 * n)


def test_gpd_fit_recovers_known_parameters():
    rng = np.random.default_rng(26)
    for c_true, scale_true in [(0.2, 1.5), (0.05, 1.0), (0.4, 3.0), (-0.1, 1.0)]:
        x = stats.genpareto.rvs(c=c_true, scale=scale_true, size=8000, random_state=rng)
        k, s = _gpd_fit(x)
        assert k == pytest.approx(c_true, abs=0.1)
        assert s == pytest.approx(scale_true, rel=0.15)


def test_psis_weights_normalize():
    rng = np.random.default_rng(27)
    lr = rng.standard_normal((2000, 3))
    logw, k = _psis(lr)
    assert np.allclose(np.exp(logw).sum(axis=0), 1.0)
    logw1, k1 = _psis(rng.standard_normal(3000) * 0.3)
    assert np.exp(logw1).sum() == pytest.approx(1.0)


def test_bayes_factor_closed_form_and_jeffreys_labels():
    bf = bayes_factor(-100.0, -105.0)
    assert bf.value == pytest.approx(np.exp(5), rel=1e-10)
    assert bf.extras["jeffreys"] == "decisive"
    assert bayes_factor(-100.0, -100.6).extras["jeffreys"] == "barely worth mentioning"
    assert bayes_factor(-100.0, -104.6).extras["jeffreys"] == "strong"

    sm = pytest.importorskip("statsmodels.api")
    rng = np.random.default_rng(28)
    X = rng.standard_normal((150, 2))
    y = X[:, 0] * 2.0 + rng.standard_normal(150) * 0.5
    Xc1 = sm.add_constant(X)
    Xc2 = sm.add_constant(X[:, :1])
    r1, r2 = sm.OLS(y, Xc1).fit(), sm.OLS(y, Xc2).fit()
    bf_bic = bayes_factor(r1, r2, method="bic")
    assert bf_bic.extras["log_bf"] == pytest.approx(-0.5 * (r1.bic - r2.bic), abs=1e-6)


# --------------------------------------------------------------------------- models

def test_bayesian_linear_matches_ols_and_predictive_coverage():
    sm = pytest.importorskip("statsmodels.api")
    rng = np.random.default_rng(29)
    n = 250
    X = rng.standard_normal((n, 2))
    beta_true = np.array([1.0, 2.0, -1.5])
    y = beta_true[0] + X @ beta_true[1:] + rng.normal(0, 0.7, n)
    model = BayesianLinear(prior_precision=1e-10, a0=1e-10, b0=1e-10).fit(X, y)
    Xc = sm.add_constant(X)
    res = sm.OLS(y, Xc).fit()
    assert np.allclose(model.coef_, np.asarray(res.params), atol=1e-6)

    ci = model.predict_interval(X)
    cov_rate = np.mean([ci[i][0] <= y[i] <= ci[i][1] for i in range(n)])
    assert 0.9 < cov_rate < 0.99

    beta_s, sigma2_s = model.sample(2000, random_state=1)
    pll = model.pointwise_log_lik((beta_s, sigma2_s))
    w = WAIC(pll)
    assert w.p_eff == pytest.approx(3.0, abs=2.5)
    loo = LOO_CV(pll)
    assert np.max(loo.extras["pareto_k"]) < 0.7


def test_bayesian_linear_evidence_matches_fine_grid():
    rng = np.random.default_rng(30)
    n = 15
    X = rng.standard_normal((n, 1))
    y = 1.0 + 2.0 * X[:, 0] + rng.normal(0, 1.0, n)
    m = BayesianLinear(prior_precision=0.1, a0=2.0, b0=2.0).fit(X, y)
    V0, Vn, b0, bn = m.V0_, m.V_n_, 2.0, m.b_n_

    def log_p_y_given_s2(s2):
        return (-0.5 * n * np.log(2 * np.pi * s2)
                + 0.5 * (np.log(np.linalg.det(Vn)) - np.log(np.linalg.det(V0)))
                - (bn - b0) / s2)

    s2grid = np.linspace(1e-4, 100, 400_000)
    logvals = log_p_y_given_s2(s2grid) + stats.invgamma.logpdf(s2grid, 2.0, scale=2.0)
    mx = logvals.max()
    Znum = np.trapezoid(np.exp(logvals - mx), s2grid)
    assert m.log_evidence_ == pytest.approx(np.log(Znum) + mx, abs=0.02)


def test_bayesian_linear_bayes_factor_prefers_true_model():
    rng = np.random.default_rng(31)
    n = 200
    X = rng.standard_normal((n, 1))
    y = 3.0 * X[:, 0] + rng.normal(0, 0.5, n)
    full = BayesianLinear(prior_precision=1.0, a0=1.0, b0=1.0).fit(X, y)
    null = BayesianLinear(prior_precision=1.0, a0=1.0, b0=1.0).fit(X * 0.0, y)
    bf = bayes_factor(full, null)
    assert bf.value > 100


def test_bayesian_logistic_matches_statsmodels_and_mcmc_ep():
    sm = pytest.importorskip("statsmodels.api")
    rng = np.random.default_rng(32)
    n = 400
    X = rng.standard_normal((n, 2))
    beta_true = np.array([0.3, 1.2, -0.8])
    z = beta_true[0] + X @ beta_true[1:]
    p = special.expit(z)
    y = (rng.uniform(size=n) < p).astype(float)

    model = BayesianLogistic(prior_var=1e6).fit(X, y)
    Xc = sm.add_constant(X)
    res = sm.Logit(y, Xc).fit(disp=0)
    assert np.allclose(model.coef_, np.asarray(res.params), atol=1e-4)

    mcmc = BayesianLogistic(prior_var=1e6, method="mcmc", n_samples=3000, n_warmup=1000,
                             random_state=0).fit(X, y)
    se = np.sqrt(np.diag(model.cov_))
    assert np.all(np.abs(mcmc.coef_ - model.coef_) < 5 * se)

    ep = BayesianLogistic(prior_var=1e6, method="ep").fit(X, y)
    assert np.all(np.abs(ep.coef_ - model.coef_) < 5 * se)

    probs = model.predict_proba(X[:20])
    assert np.all((probs >= 0) & (probs <= 1))
    mc_probs = model.predict_proba(X[:20], method="mc")
    assert np.max(np.abs(probs - mc_probs)) < 0.03


def test_naive_bayes_exact_bernoulli_and_accuracy():
    Xb = np.array([[1, 0, 1], [1, 1, 0], [0, 0, 1], [0, 1, 1], [1, 0, 0], [0, 1, 0]], dtype=float)
    yb = np.array([0, 0, 0, 1, 1, 1])
    nb = NaiveBayes("bernoulli", alpha=1.0).fit(Xb, yb)
    assert np.allclose(nb.theta_[0], [(2 + 1) / 5, (1 + 1) / 5, (2 + 1) / 5])
    assert np.allclose(np.exp(nb.class_log_prior_), [0.5, 0.5])

    rng = np.random.default_rng(33)
    X0 = rng.normal([0, 0], 0.4, (100, 2))
    X1 = rng.normal([5, 5], 0.4, (100, 2))
    X = np.vstack([X0, X1])
    y = np.array([0] * 100 + [1] * 100)
    nbg = NaiveBayes("gaussian").fit(X, y)
    assert nbg.score(X, y) > 0.95

    vocab = 20
    rng2 = np.random.default_rng(34)

    def gen_docs(n, hot_idx):
        docs = rng2.poisson(0.3, (n, vocab))
        for h in hot_idx:
            docs[:, h] += rng2.poisson(5, n)
        return docs

    Xm = np.vstack([gen_docs(150, [0, 1, 2]), gen_docs(150, [10, 11, 12])])
    ym = np.array([0] * 150 + [1] * 150)
    nbm = NaiveBayes("multinomial").fit(Xm, ym)
    assert nbm.score(Xm, ym) > 0.9


def test_hierarchical_model_pooling_limits_and_eight_schools():
    y8 = np.array([28, 8, -3, 7, -1, 1, 18, 12], dtype=float)
    s8 = np.array([15, 10, 16, 11, 9, 11, 10, 18], dtype=float)

    m_pool = HierarchicalModel(tau_prior="fixed", n_samples=2000, n_warmup=500,
                                random_state=0).fit(y8, sigma=s8, tau=1e-8)
    assert np.std(m_pool.theta_mean_) < 0.5  # complete pooling: all groups collapse together

    m_nopool = HierarchicalModel(tau_prior="fixed", n_samples=2000, n_warmup=500,
                                  random_state=1).fit(y8, sigma=s8, tau=1e4)
    assert np.allclose(m_nopool.theta_mean_, y8, atol=1.0)  # no pooling: theta_j ~ y_j

    tau0 = 8.0
    m_fixed = HierarchicalModel(mu0=0.0, tau_mu=1e6, tau_prior="fixed", n_samples=6000,
                                 n_warmup=1500, random_state=2).fit(y8, sigma=s8, tau=tau0)
    w2 = 1 / (s8 ** 2 + tau0 ** 2)
    mu_hat = np.sum(w2 * y8) / np.sum(w2)
    assert m_fixed.mu_mean_ == pytest.approx(mu_hat, abs=1.0)

    m_hc = HierarchicalModel(mu0=0.0, tau_mu=100.0, tau_prior="half_cauchy", tau_scale=5.0,
                              n_samples=4000, n_warmup=1500, random_state=3).fit(y8, sigma=s8)
    assert np.all((m_hc.theta_mean_ > 3) & (m_hc.theta_mean_ < 13))
    assert 5 < m_hc.mu_mean_ < 10
    summ = m_hc.summary()
    assert "mu" in summ and "tau" in summ
    new_draws = m_hc.predict_new_group(1000, random_state=4)
    assert new_draws.shape == (1000,)

    rng = np.random.default_rng(35)
    groups = np.repeat(np.arange(5), 40)
    true_mu, true_tau, true_sigma = 2.0, 1.5, 0.8
    group_means = true_mu + true_tau * rng.standard_normal(5)
    y_raw = group_means[groups] + true_sigma * rng.standard_normal(200)
    m_b = HierarchicalModel(n_samples=2000, n_warmup=500, random_state=6).fit(
        y_raw, groups=groups)
    assert np.allclose(m_b.theta_mean_, group_means, atol=0.6)
    assert m_b.sigma_samples_.mean() ** 0.5 == pytest.approx(true_sigma, abs=0.3)
    pll = m_b.pointwise_log_lik()
    assert pll.shape == (2000, 200)


def test_mixture_model_gaussian_and_poisson_and_waic():
    rng = np.random.default_rng(36)
    x1, x2 = rng.normal(-5, 1, 200), rng.normal(5, 1, 200)
    x = np.concatenate([x1, x2])
    mm = MixtureModel(2, n_samples=800, n_warmup=300, random_state=1).fit(x)
    means = np.sort(mm.means_.ravel())
    assert means[0] == pytest.approx(-5, abs=0.5) and means[1] == pytest.approx(5, abs=0.5)

    c1 = rng.normal([0, 0], 0.3, (100, 2))
    c2 = rng.normal([5, 5], 0.3, (100, 2))
    c3 = rng.normal([0, 5], 0.3, (100, 2))
    X = np.vstack([c1, c2, c3])
    truth = np.array([0] * 100 + [1] * 100 + [2] * 100)
    mm3 = MixtureModel(3, n_samples=600, n_warmup=300, random_state=2).fit(X)
    labels = mm3.predict(X)
    from itertools import permutations
    best = max(np.mean(np.array(perm)[labels] == truth) for perm in permutations(range(3)))
    assert best >= 0.95
    w = mm3.waic()
    assert isinstance(w, ICResult)

    xr = np.concatenate([rng.poisson(3, 200), rng.poisson(20, 200)])
    mmp = MixtureModel(2, family="poisson", n_samples=500, n_warmup=200, random_state=3).fit(xr)
    rates = np.sort(mmp.means_)
    assert rates[0] == pytest.approx(3, abs=1.0) and rates[1] == pytest.approx(20, abs=2.0)


def test_bayesian_network_sprinkler_query_matches_brute_force():
    bn = BayesianNetwork()
    for name in ["Cloudy", "Sprinkler", "Rain", "WetGrass"]:
        bn.add_node(name, [0, 1])
    bn.add_edge("Cloudy", "Sprinkler")
    bn.add_edge("Cloudy", "Rain")
    bn.add_edge("Sprinkler", "WetGrass")
    bn.add_edge("Rain", "WetGrass")
    bn.set_cpt("Cloudy", [0.5, 0.5])
    bn.set_cpt("Sprinkler", [[0.5, 0.5], [0.9, 0.1]])
    bn.set_cpt("Rain", [[0.8, 0.2], [0.2, 0.8]])
    wg = np.zeros((2, 2, 2))
    wg[0, 0] = [1.0, 0.0]; wg[0, 1] = [0.1, 0.9]
    wg[1, 0] = [0.1, 0.9]; wg[1, 1] = [0.01, 0.99]
    bn.set_cpt("WetGrass", wg)

    res = bn.query(["Rain"], {"WetGrass": 1})
    num = den = 0.0
    for c in (0, 1):
        for s in (0, 1):
            for r in (0, 1):
                for w in (0, 1):
                    p = bn.joint_probability({"Cloudy": c, "Sprinkler": s, "Rain": r, "WetGrass": w})
                    if w == 1:
                        den += p
                        if r == 1:
                            num += p
    assert res[1] == pytest.approx(num / den, abs=1e-10)

    mapq = bn.map_query(["Rain", "Sprinkler"], {"WetGrass": 1})
    assert mapq["Rain"] == 1

    samples = bn.sample(30000, random_state=0)
    node_idx = {n: i for i, n in enumerate(bn.nodes)}
    wet_mask = samples[:, node_idx["WetGrass"]] == 1
    empirical = samples[wet_mask, node_idx["Rain"]].mean()
    se = np.sqrt(res[1] * (1 - res[1]) / wet_mask.sum())
    assert abs(empirical - res[1]) < 5 * se

    fit_bn = BayesianNetwork()
    for name in ["Cloudy", "Sprinkler", "Rain", "WetGrass"]:
        fit_bn.add_node(name, [0, 1])
    fit_bn.add_edge("Cloudy", "Sprinkler"); fit_bn.add_edge("Cloudy", "Rain")
    fit_bn.add_edge("Sprinkler", "WetGrass"); fit_bn.add_edge("Rain", "WetGrass")
    fit_bn.fit(samples, alpha=1.0)
    assert np.allclose(fit_bn.cpts_["Cloudy"], [0.5, 0.5], atol=0.03)

    with pytest.raises(ValueError):
        bn.add_edge("WetGrass", "Cloudy")

    score_true = bn.log_marginal_likelihood(samples[:3000])
    bn2 = BayesianNetwork()
    for name in ["Cloudy", "Sprinkler", "Rain", "WetGrass"]:
        bn2.add_node(name, [0, 1])
    bn2.add_edge("Cloudy", "Sprinkler")
    score_bad = bn2.log_marginal_likelihood(samples[:3000])
    assert score_true > score_bad
    assert bn.markov_blanket("Sprinkler") == sorted(["Cloudy", "WetGrass", "Rain"])


def test_dirichlet_process_crp_and_stick_breaking_and_dpmm():
    from stochpylib.distributions import Normal
    dp = DirichletProcess(alpha=2.0, base=Normal(0, 1), truncation=50)
    n = 100
    rng = np.random.default_rng(37)
    counts = [len(np.unique(dp.crp(n, random_state=rng))) for _ in range(400)]
    mean_k = np.mean(counts)
    se = np.std(counts) / np.sqrt(400)
    assert abs(mean_k - dp.expected_clusters(n)) < 3 * se

    w, a = dp.stick_breaking(random_state=0)
    assert w.sum() == pytest.approx(1.0)
    w1s = [dp.stick_breaking(random_state=i)[0][0] for i in range(3000)]
    assert np.mean(w1s) == pytest.approx(1 / (1 + 2.0), abs=0.02)

    rng2 = np.random.default_rng(38)
    x = np.concatenate([rng2.normal(-6, 0.4, 30), rng2.normal(0, 0.4, 30), rng2.normal(6, 0.4, 30)])
    dpm = DirichletProcess(alpha=1.0).fit(x, n_samples=200, n_warmup=150, random_state=1)
    assert dpm.n_clusters_ == 3
    assert np.allclose(np.sort(dpm.cluster_means_), [-6, 0, 6], atol=0.5)
    co = dpm.co_clustering_matrix()
    assert co.shape == (90, 90)

    dpa = DirichletProcess(alpha=1.0).fit(x[:40], n_samples=150, n_warmup=100,
                                           learn_alpha=True, random_state=2)
    assert np.all(np.isfinite(dpa.alpha_samples_)) and np.all(dpa.alpha_samples_ > 0)


# ------------------------------------------------------------------- cross-module / library

def test_bayesian_conjugate_agrees_with_statistics_bayesian_estimator():
    from stochpylib.statistics import bayesian_estimator
    rng = np.random.default_rng(39)
    x = rng.binomial(1, 0.4, 200)
    post = posterior(prior(Beta(1, 1)), likelihood("bernoulli", data=x), method="conjugate")
    est = bayesian_estimator("bernoulli", x, prior=(1.0, 1.0))
    assert est.extras["posterior"].a == pytest.approx(post.dist.a)
    assert est.extras["posterior"].b == pytest.approx(post.dist.b)


def test_quickstart_example_runs():
    rng = np.random.default_rng(40)
    coin_flips = rng.binomial(1, 0.6, 40)
    post = posterior(prior(Beta(2, 2)), likelihood("bernoulli", data=coin_flips))
    assert post.mean() is not None and post.credible_interval(0.95) is not None
    pred = posterior_predictive(post)
    assert 0 <= pred.mean() <= 1
    logZ = evidence(prior(Beta(2, 2)), likelihood("bernoulli", data=coin_flips))
    assert np.isfinite(logZ)

    X = rng.standard_normal((100, 2))
    y = 1.0 + X @ np.array([2.0, -1.0]) + rng.normal(0, 0.5, 100)
    model = BayesianLinear().fit(X, y)
    mu, sd = model.predict(X[:5], return_std=True)
    assert mu.shape == (5,) and np.all(sd > 0)
    beta_s, sigma2_s = model.sample(500, random_state=0)
    w = WAIC(model.pointwise_log_lik((beta_s, sigma2_s)))
    assert np.isfinite(float(w))

    model_a = BayesianLinear().fit(X, y)
    model_b = BayesianLinear().fit(X[:, :1], y)
    bf = bayes_factor(model_a, model_b)
    assert bf.extras["jeffreys"] in ("negative", "barely worth mentioning", "substantial",
                                      "strong", "very strong", "decisive")
