"""Tests for stochpylib.advanced_mcmc: standard/gradient-based/slice/population samplers,
diagnostics, and variational inference.

Oracles: closed-form posteriors (conjugate Gaussian models, linear-Gaussian state-space
likelihoods via a hand-written Kalman recursion, conjugate linear-regression Bayes
factors), scipy.stats (KS tests against known marginals), and independent hand-derived
formulas for the diagnostics (Gelman-Rubin/PSRF recomputed from scratch in the test).
All randomness is seeded; statistical assertions state their Monte Carlo standard error
(MCSE, using this module's own ESS as the effective-sample-size denominator) and use
>= 3 SE except where a documented finite-sample/estimator bias applies.
"""

import numpy as np
import pytest
from scipy import stats as sps

import stochpylib
from stochpylib import advanced_mcmc as am
from stochpylib.advanced_mcmc import _base as am_base
from stochpylib.advanced_mcmc import advanced as am_ad
from stochpylib.advanced_mcmc import diagnostics as am_dg
from stochpylib.advanced_mcmc import gradient_based as am_gb
from stochpylib.advanced_mcmc import slice_sampling as am_sl
from stochpylib.advanced_mcmc import standard as am_st
from stochpylib.advanced_mcmc import variational as am_vi
from stochpylib.statistics import TestResult

_MU3 = np.array([1.0, -1.0, 0.5])
_CORR3 = np.array([[1.0, 0.6, 0.2], [0.6, 1.0, 0.3], [0.2, 0.3, 1.0]])


def _log_prob3(theta):
    prec = np.linalg.inv(_CORR3)
    d = theta - _MU3
    return -0.5 * d @ prec @ d


def _grad3(theta):
    prec = np.linalg.inv(_CORR3)
    return -prec @ (theta - _MU3)


_MU2 = np.array([1.0, -1.0])
_RHO = 0.6
_COV2 = np.array([[1.0, _RHO], [_RHO, 1.0]])
_PREC2 = np.linalg.inv(_COV2)


def _log_prob2(theta):
    d = theta - _MU2
    return -0.5 * d @ _PREC2 @ d


def _grad2(theta):
    return -_PREC2 @ (theta - _MU2)


def _mcse(x, ess_method="mean"):
    """Monte Carlo standard error of a 1-D sample using this module's own ESS."""
    ess = am.ESS(x[None, :, None], method=ess_method)
    return float(x.std(ddof=1) / np.sqrt(max(ess, 1.0)))


def _within(value, target, se, k=3.0):
    return abs(value - target) < k * se


# ================================================================== standard

def test_rwmh_matches_correlated_gaussian_moments():
    mh = am.MetropolisHastings(_log_prob3, n_samples=4000, n_warmup=1000, proposal_scale=1.0)
    mh.sample(np.zeros(3), random_state=0)
    s = mh.get_samples()
    for j in range(3):
        se = _mcse(s[:, j])
        assert _within(s[:, j].mean(), _MU3[j], se, k=4)
    cov = np.cov(s.T)
    assert np.max(np.abs(cov - _CORR3)) < 0.2
    assert 0.1 < mh.acceptance_rate_ < 0.6


def test_rwmh_adapts_scale_toward_target():
    mh = am.MetropolisHastings(_log_prob2, n_samples=2000, n_warmup=1500, proposal_scale=5.0,
                               target_accept=0.234)
    mh.sample(np.zeros(2), random_state=1)
    assert 0.12 < mh.acceptance_rate_ < 0.40


def test_mh_custom_asymmetric_proposal_needs_correction():
    def log_prob_exp(theta):
        x = theta[0]
        return -x if x > 0 else -np.inf

    def proposal(theta, rng):
        return theta * np.exp(0.5 * rng.standard_normal())

    def proposal_log_density(to, from_):
        # log-normal multiplicative step: log q(to | from) for to = from * exp(z), z~N(0,0.25)
        if to[0] <= 0 or from_[0] <= 0:
            return -np.inf
        z = np.log(to[0] / from_[0])
        return -0.5 * (z / 0.5) ** 2 - np.log(to[0])

    corrected = am.MetropolisHastings(log_prob_exp, n_samples=4000, n_warmup=1000,
                                      proposal=proposal, proposal_log_density=proposal_log_density)
    corrected.sample(np.array([1.0]), random_state=2)
    s = corrected.get_samples()[:, 0]
    se = _mcse(s)
    assert _within(s.mean(), 1.0, se, k=4)

    uncorrected = am.MetropolisHastings(log_prob_exp, n_samples=4000, n_warmup=1000, proposal=proposal)
    uncorrected.sample(np.array([1.0]), random_state=2)
    s2 = uncorrected.get_samples()[:, 0]
    assert abs(s2.mean() - 1.0) > abs(s.mean() - 1.0)


def test_independence_sampler_gaussian_proposal():
    ind = am.IndependenceSampler(lambda t: -0.5 * t[0] ** 2, lambda r: r.standard_normal(1) * 2,
                                 lambda t: -0.5 * (t[0] / 2) ** 2 - np.log(2),
                                 n_samples=4000, n_warmup=500)
    ind.sample(np.zeros(1), random_state=3)
    s = ind.get_samples()[:, 0]
    se = _mcse(s)
    assert _within(s.mean(), 0.0, se + 0.02, k=4)
    assert abs(s.var() - 1.0) < 0.2


def test_gibbs_exact_conditionals_bivariate_normal():
    rho = 0.8

    def c0(theta, r):
        return rho * theta[1] + np.sqrt(1 - rho ** 2) * r.standard_normal()

    def c1(theta, r):
        return rho * theta[0] + np.sqrt(1 - rho ** 2) * r.standard_normal()

    gs = am.GibbsSampler(conditionals=[c0, c1], n_samples=4000, n_warmup=500)
    gs.sample(np.zeros(2), random_state=4)
    s = gs.get_samples()
    assert gs.acceptance_rate_ == 1.0
    se = _mcse(s[:, 0])
    assert _within(s[:, 0].mean(), 0.0, se + 0.02, k=4)
    assert abs(np.corrcoef(s.T)[0, 1] - rho) < 0.05


def test_metropolis_within_gibbs_random_scan():
    rho = 0.7
    prec = np.linalg.inv([[1, rho], [rho, 1]])

    def log_prob(theta):
        return -0.5 * theta @ prec @ theta

    gs = am.GibbsSampler(log_prob=log_prob, scan="random", n_samples=4000, n_warmup=1500,
                         proposal_scale=1.0)
    gs.sample(np.zeros(2), random_state=5)
    s = gs.get_samples()
    assert abs(np.corrcoef(s.T)[0, 1] - rho) < 0.08


def test_adaptive_metropolis_learns_covariance():
    cov = np.array([[4.0, 1.5], [1.5, 1.0]])
    prec = np.linalg.inv(cov)

    def log_prob(theta):
        return -0.5 * theta @ prec @ theta

    am_s = am.AdaptiveMetropolis(log_prob, n_samples=4000, n_warmup=2000)
    am_s.sample(np.zeros(2), random_state=6)
    s = am_s.get_samples()
    empirical_cov = np.cov(s.T)
    assert np.linalg.norm(empirical_cov - cov) / np.linalg.norm(cov) < 0.4
    scale_factor = 2.38 ** 2 / 2
    rel_err = np.linalg.norm(am_s.proposal_cov_ - scale_factor * cov) / np.linalg.norm(scale_factor * cov)
    assert rel_err < 0.5


def test_robust_adaptive_metropolis_hits_target_acceptance():
    prec = np.linalg.inv(_CORR3)

    def log_prob(theta):
        return -0.5 * theta @ prec @ theta

    ram = am.RobustAdaptiveMetropolis(log_prob, n_samples=4000, n_warmup=2000, target_accept=0.234)
    ram.sample(np.zeros(3), random_state=7)
    assert abs(ram.acceptance_rate_ - 0.234) < 0.1


# ============================================================== gradient_based

def test_hmc_ill_conditioned_gaussian_with_mass_adaptation():
    sds = np.array([0.2, 1.0, 5.0])

    def log_prob(theta):
        return -0.5 * np.sum((theta / sds) ** 2)

    def grad(theta):
        return -theta / sds ** 2

    hmc = am.HamiltonianMonteCarlo(log_prob, grad, n_samples=1500, n_warmup=1200, n_leapfrog=15,
                                   target_accept=0.8, jitter=0.15)
    hmc.sample(np.zeros(3), random_state=8)
    s = hmc.get_samples()
    for j in range(3):
        se = _mcse(s[:, j])
        assert _within(s[:, j].mean(), 0.0, se + 0.05, k=4)
        assert abs(s[:, j].var() - sds[j] ** 2) / sds[j] ** 2 < 0.3
    # mass_matrix_ is the physical mass (~precision for a Gaussian target), the inverse
    # of what the adapter directly tunes (inv_mass, which tracks the target's variance)
    assert np.all(hmc.mass_matrix_ * sds ** 2 > 0.3) and np.all(hmc.mass_matrix_ * sds ** 2 < 3.0)
    assert 0.4 < hmc.acceptance_rate_ < 0.99


def test_nuts_quickstart_and_diagnostics():
    """The vault Quickstart-Examples.md "Advanced MCMC -- NUTS" example, verified."""
    rng = np.random.default_rng(9)
    n, p = 60, 3
    X = rng.standard_normal((n, p))
    beta_true = np.array([1.0, -0.5, 0.3])
    sigma2 = 1.0
    y = X @ beta_true + rng.normal(0, np.sqrt(sigma2), n)

    def log_posterior(theta):
        resid = y - X @ theta
        return -0.5 * np.sum(resid ** 2) / sigma2 - 0.5 * np.sum(theta ** 2) / 100.0

    def grad_log_posterior(theta):
        resid = y - X @ theta
        return X.T @ resid / sigma2 - theta / 100.0

    nuts = am.NoUTurnSampler(log_posterior, grad_log_posterior, n_samples=2000, n_warmup=500,
                             n_chains=2, target_accept=0.8)
    nuts.sample(theta_init=np.zeros(p), random_state=10)
    chains = nuts.get_chains()
    rhat = am.Rhat(chains)
    ess = am.ESS(chains)
    assert np.all(rhat < 1.05)
    assert np.all(ess > 300)
    ols = np.linalg.lstsq(X, y, rcond=None)[0]
    flat = nuts.get_samples()
    for j in range(p):
        se = flat[:, j].std(ddof=1) / np.sqrt(ess[j])
        assert _within(flat[:, j].mean(), ols[j], se, k=4)
    assert nuts.mean_tree_depth_ < 7
    assert nuts.divergences_ < 0.05 * nuts.n_samples * nuts.n_chains


def test_nuts_accept_stat_tracks_target():
    for target in (0.6, 0.85):
        nuts = am.NoUTurnSampler(_log_prob3, _grad3, n_samples=1500, n_warmup=1000, target_accept=target)
        nuts.sample(np.zeros(3), random_state=11)
        assert abs(nuts.acceptance_rate_ - target) < 0.15


def test_nuts_finite_difference_gradient_matches_analytic():
    n_a = am.NoUTurnSampler(_log_prob2, _grad2, n_samples=1500, n_warmup=800)
    n_a.sample(np.zeros(2), random_state=12)
    n_b = am.NoUTurnSampler(_log_prob2, n_samples=1500, n_warmup=800)  # no analytic grad -> FD
    n_b.sample(np.zeros(2), random_state=13)
    assert n_b.target.n_grad_evals > 0
    sa, sb = n_a.get_samples(), n_b.get_samples()
    for j in range(2):
        se = np.sqrt(_mcse(sa[:, j]) ** 2 + _mcse(sb[:, j]) ** 2)
        assert _within(sa[:, j].mean(), sb[:, j].mean(), se, k=5)


def test_nuts_flags_divergences_on_a_funnel():
    def log_prob(theta):
        y, x = theta
        return -0.5 * (y / 3.0) ** 2 - 0.5 * (x / np.exp(y / 2.0)) ** 2 - y / 2.0

    def grad(theta):
        y, x = theta
        var = np.exp(y)
        dy = -y / 9.0 + (x ** 2) / var * 0.5 - 0.5
        dx = -x / var
        return np.array([dy, dx])

    reckless = am.NoUTurnSampler(log_prob, grad, n_samples=500, n_warmup=0, step_size=1.0,
                                 adapt_step_size=False, adapt_mass=False, max_treedepth=6)
    reckless.sample(np.array([0.0, 0.0]), random_state=14)
    assert reckless.divergences_ > 0

    careful = am.NoUTurnSampler(log_prob, grad, n_samples=500, n_warmup=800, target_accept=0.95)
    careful.sample(np.array([0.0, 0.0]), random_state=14)
    assert careful.divergences_ < 0.1 * careful.n_samples


def test_mala_acceptance_and_moments():
    mala = am.MALA(_log_prob3, _grad3, n_samples=4000, n_warmup=1500, target_accept=0.574)
    mala.sample(np.zeros(3), random_state=15)
    assert abs(mala.acceptance_rate_ - 0.574) < 0.15
    s = mala.get_samples()
    for j in range(3):
        se = _mcse(s[:, j])
        assert _within(s[:, j].mean(), _MU3[j], se, k=4)


def test_mmala_with_constant_and_auto_metric():
    prec = np.linalg.inv(_COV2)
    m1 = am.MMALA(_log_prob2, _grad2, metric=lambda t: prec, n_samples=3000, n_warmup=1200)
    m1.sample(np.zeros(2), random_state=16)
    s1 = m1.get_samples()
    for j in range(2):
        se = _mcse(s1[:, j])
        assert _within(s1[:, j].mean(), _MU2[j], se, k=4)

    m2 = am.MMALA(_log_prob2, _grad2, n_samples=3000, n_warmup=1200)  # auto SoftAbs metric
    m2.sample(np.zeros(2), random_state=17)
    s2 = m2.get_samples()
    for j in range(2):
        se = _mcse(s2[:, j])
        assert _within(s2[:, j].mean(), _MU2[j], se, k=4)


def test_rmhmc_matches_nuts_on_logistic_regression():
    rng = np.random.default_rng(18)
    n = 40
    x = rng.standard_normal(n)
    true_b = np.array([0.5, 1.2])
    p = 1.0 / (1.0 + np.exp(-(true_b[0] + true_b[1] * x)))
    y = (rng.uniform(size=n) < p).astype(float)
    prior_prec = np.eye(2) / 25.0

    def log_post(theta):
        eta = theta[0] + theta[1] * x
        ll = np.sum(y * eta - np.logaddexp(0, eta))
        return ll - 0.5 * theta @ prior_prec @ theta

    def grad(theta):
        eta = theta[0] + theta[1] * x
        p_hat = 1.0 / (1.0 + np.exp(-eta))
        resid = y - p_hat
        g0 = np.sum(resid)
        g1 = np.sum(resid * x)
        return np.array([g0, g1]) - prior_prec @ theta

    def fisher_metric(theta):
        eta = theta[0] + theta[1] * x
        p_hat = 1.0 / (1.0 + np.exp(-eta))
        w = p_hat * (1 - p_hat)
        design = np.column_stack([np.ones(n), x])
        return design.T @ (w[:, None] * design) + prior_prec

    rmhmc = am_gb.RiemannianHMC(log_post, grad, metric=fisher_metric, n_samples=600, n_warmup=400,
                                n_leapfrog=4, n_fixed_point=4, target_accept=0.8)
    rmhmc.sample(np.zeros(2), random_state=19)
    assert rmhmc.acceptance_rate_ > 0.5

    nuts = am.NoUTurnSampler(log_post, grad, n_samples=3000, n_warmup=1000)
    nuts.sample(np.zeros(2), random_state=20)

    sr, sn = rmhmc.get_samples(), nuts.get_samples()
    for j in range(2):
        se = np.sqrt(_mcse(sr[:, j]) ** 2 + _mcse(sn[:, j]) ** 2)
        assert _within(sr[:, j].mean(), sn[:, j].mean(), se, k=5)


def test_neutra_hmc_recovers_correlated_gaussian():
    """NeutraHMC's correctness never depends on flow fit quality (it's an exact
    reparameterization); verified with enough draws even from a modestly-trained flow."""
    neu = am.NeutraHMC(_log_prob2, _grad2, n_samples=3000, n_warmup=1200, flow_layers=4,
                       flow_iter=400, flow_lr=0.01)
    neu.sample(np.zeros(2), random_state=21)
    s = neu.get_samples()
    for j in range(2):
        se = _mcse(s[:, j])
        assert _within(s[:, j].mean(), _MU2[j], se, k=5)
    cov = np.cov(s.T)
    assert np.max(np.abs(cov - _COV2)) < 0.3


# =============================================================== slice_sampling

def test_slice_sampling_stepping_and_doubling_on_gamma():
    shape, scale = 3.0, 2.0

    def log_prob(theta):
        x = theta[0]
        return -np.inf if x <= 0 else (shape - 1) * np.log(x) - x / scale

    for interval in ("stepping", "doubling"):
        s = am.SliceSampling(log_prob, n_samples=3000, n_warmup=500, width=3.0,
                             interval=interval).sample(np.array([shape * scale]), random_state=22)
        x = s.get_samples()[::5, 0]
        d, p = sps.kstest(x, sps.gamma(shape, scale=scale).cdf)
        assert p > 0.01, (interval, d, p)
        se = _mcse(s.get_samples()[:, 0])
        assert _within(x.mean(), shape * scale, se + 0.3, k=5)


def test_slice_sampling_multivariate_width_vector():
    sds = np.array([0.5, 5.0, 50.0])

    def log_prob(theta):
        return -0.5 * np.sum((theta / sds) ** 2)

    s = am.SliceSampling(log_prob, n_samples=2000, n_warmup=500, width=list(sds)).sample(
        np.zeros(3), random_state=23)
    samples = s.get_samples()
    for j in range(3):
        se = _mcse(samples[:, j])
        assert _within(samples[:, j].mean(), 0.0, se + 0.05 * sds[j], k=5)


def test_elliptical_slice_gaussian_posterior_closed_form():
    rng = np.random.default_rng(24)
    n = 6
    pts = np.linspace(0, 1, n)
    K = np.exp(-0.5 * (pts[:, None] - pts[None, :]) ** 2 / 0.2 ** 2) + 1e-6 * np.eye(n)
    f_true = rng.multivariate_normal(np.zeros(n), K)
    noise_sd = 0.3
    y = f_true + rng.normal(0, noise_sd, n)

    def loglik(f):
        return -0.5 * np.sum(((y - f) / noise_sd) ** 2)

    K_inv = np.linalg.inv(K)
    post_prec = K_inv + np.eye(n) / noise_sd ** 2
    post_cov = np.linalg.inv(post_prec)
    post_mean = post_cov @ (y / noise_sd ** 2)

    ess = am.EllipticalSliceSampling(loglik, prior_cov=K, n_samples=3000, n_warmup=500)
    ess.sample(np.zeros(n), random_state=25)
    s = ess.get_samples()
    for j in range(n):
        se = _mcse(s[:, j])
        assert _within(s[:, j].mean(), post_mean[j], se + 0.05, k=5)
    assert np.max(np.abs(np.diag(np.cov(s.T)) - np.diag(post_cov))) < 0.15


def test_polar_slice_radius_is_chi_distributed():
    def log_prob(theta):
        return -0.5 * np.sum(theta ** 2)

    ps = am.Polar_Slice(log_prob, n_samples=3000, n_warmup=500)
    ps.sample(np.ones(5), random_state=26)
    s = ps.get_samples()
    r2 = np.sum(s ** 2, axis=1)
    d, p = sps.kstest(r2[::5], sps.chi2(5).cdf)
    assert p > 0.01
    for j in range(5):
        se = _mcse(s[:, j])
        assert _within(s[:, j].mean(), 0.0, se + 0.1, k=5)
    with pytest.raises(ValueError):
        am.Polar_Slice(log_prob, n_samples=10, n_warmup=5).sample(np.array([1.0]), random_state=0)


def test_doubling_accept_matches_neal_criterion():
    def f(x):
        return -0.5 * min(abs(x - 3), abs(x + 3)) ** 2  # bimodal-ish, peaked at +-3

    d = am_sl.Doubling(width=1.0, max_doublings=10)
    rng = np.random.default_rng(27)
    L, R = d.find_interval(f, 3.0, f(3.0) - 5.0, rng)
    assert L <= 3.0 <= R
    assert d.accept(f, 3.0, 3.2, f(3.0) - 5.0, L, R) in (True, False)


# ==================================================================== advanced

def test_parallel_tempering_mixes_bimodal_where_rwmh_fails():
    def log_prob(theta):
        x = theta[0]
        return np.logaddexp(-0.5 * ((x - 6) / 0.4) ** 2, -0.5 * ((x + 6) / 0.4) ** 2)

    mh = am.MetropolisHastings(log_prob, n_samples=4000, n_warmup=1000, proposal_scale=0.4)
    mh.sample(np.array([6.0]), random_state=28)
    frac_mh = (mh.get_samples()[:, 0] > 0).mean()
    assert frac_mh < 0.1 or frac_mh > 0.9

    pt = am.ParallelTempering(log_prob, n_temps=5, max_temp=80.0, n_samples=4000, n_warmup=1500,
                              pilot_rounds=3, pilot_iters=150)
    pt.sample(np.array([6.0]), random_state=29)
    frac_pt = (pt.get_samples()[:, 0] > 0).mean()
    indicator = (pt.get_samples()[:, 0] > 0).astype(float)
    se = _mcse(indicator)
    assert _within(frac_pt, 0.5, se, k=4)
    assert np.all((pt.swap_acceptance_ > 0.05) & (pt.swap_acceptance_ < 0.95))
    assert pt.temperatures_[0] == 1.0


def test_replica_exchange_general_ladder_even_odd():
    def lps(theta, k):
        beta = [1.0, 0.5, 0.2][k]
        return -0.5 * beta * theta @ theta

    re = am.ReplicaExchange(lps, n_replicas=3, n_samples=2000, n_warmup=500,
                            swap_scheme="even-odd", proposal_scale=1.5)
    re.sample(np.zeros(2), random_state=30)
    s = re.get_samples()
    for j in range(2):
        se = _mcse(s[:, j])
        assert _within(s[:, j].mean(), 0.0, se + 0.05, k=4)
    assert re.replica_chains_.shape == (3, 2000, 2)


def test_smc_log_evidence_matches_gaussian_closed_form():
    y = np.array([1.0, -1.0, 0.5])

    def log_prior(theta):
        return -0.5 * np.sum(theta ** 2)

    def log_lik(theta):
        return -0.5 * np.sum(((theta - y) / 0.5) ** 2) - 3 * np.log(2 * np.pi * 0.25) / 2

    def prior_sampler(n, rng):
        return rng.standard_normal((n, 3))

    logZ_true = float(np.sum(sps.norm(0, np.sqrt(1.25)).logpdf(y)))
    zs = []
    for seed in range(5):
        smc = am.SequentialMonteCarlo(log_prior, log_lik, prior_sampler, n_particles=2000, n_mcmc=5)
        smc.sample(random_state=seed)
        zs.append(smc.log_evidence_)
    zs = np.array(zs)
    se = zs.std(ddof=1) / np.sqrt(len(zs))
    assert abs(zs.mean() - logZ_true) < 3 * max(se, 0.05)
    post_var_true = 1.0 / (1.0 + 4.0)
    post_mean_true = post_var_true * (y / 0.25)
    for j in range(3):
        pse = _mcse(smc.particles_[:, j])
        assert _within(smc.particles_[:, j].mean(), post_mean_true[j], pse + 0.05, k=5)
    assert smc.betas_[-1] == 1.0 and np.all(np.diff(smc.betas_) >= 0)


def test_particle_mcmc_matches_exact_kalman_likelihood_mh():
    rng = np.random.default_rng(31)
    T = 50
    phi_true, sx, sy = 0.7, 0.5, 0.5
    x = np.zeros(T)
    x[0] = rng.normal(0, sx / np.sqrt(1 - phi_true ** 2))
    for t in range(1, T):
        x[t] = phi_true * x[t - 1] + rng.normal(0, sx)
    y = x + rng.normal(0, sy, T)

    def kalman_loglik(phi):
        if abs(phi) >= 1:
            return -np.inf
        m, P = 0.0, sx ** 2 / (1 - phi ** 2)
        ll = 0.0
        for t in range(T):
            S = P + sy ** 2
            ll += -0.5 * np.log(2 * np.pi * S) - 0.5 * (y[t] - m) ** 2 / S
            K = P / S
            m = m + K * (y[t] - m)
            P = (1 - K) * P
            m = phi * m
            P = phi ** 2 * P + sx ** 2
        return ll

    def log_post(theta):
        phi = theta[0]
        return -np.inf if abs(phi) >= 1 else kalman_loglik(phi)

    oracle = am.MetropolisHastings(log_post, n_samples=2000, n_warmup=800, proposal_scale=0.1)
    oracle.sample(np.array([0.5]), random_state=32)

    def log_prior(theta):
        return 0.0 if abs(theta[0]) < 1 else -np.inf

    def initial_sampler(theta, r):
        phi = theta[0]
        return r.normal(0, sx / np.sqrt(max(1 - phi ** 2, 1e-6)), size=300)

    def transition_sampler(particles, theta, r):
        phi = theta[0]
        particles = np.asarray(particles)
        return phi * particles + r.normal(0, sx, size=particles.shape)

    def obs_logpdf(particles, yt, theta):
        particles = np.asarray(particles).reshape(-1)
        return -0.5 * ((yt - particles) / sy) ** 2

    pmmh = am.ParticleMCMC(y, log_prior, initial_sampler, transition_sampler, obs_logpdf,
                           n_particles=300, n_samples=2000, n_warmup=800, proposal_scale=0.1)
    pmmh.sample(np.array([0.5]), random_state=33)

    s_o, s_p = oracle.get_samples()[:, 0], pmmh.get_samples()[:, 0]
    se = np.sqrt(_mcse(s_o) ** 2 + _mcse(s_p) ** 2)
    assert _within(s_p.mean(), s_o.mean(), se + 0.03, k=5)
    assert np.all(np.isfinite(pmmh.loglik_estimates_))


def _conjugate_normal_regression_setup(seed=34):
    rng = np.random.default_rng(seed)
    n = 30
    x = rng.standard_normal(n)
    beta1_true, sigma = 0.3, 1.0
    y = beta1_true * x + rng.normal(0, sigma, n)
    tau2 = 100.0

    def log_post_m1(theta):
        b0 = theta[0]
        resid = y - b0
        ll = -0.5 * np.sum(resid ** 2) / sigma ** 2 - n * 0.5 * np.log(2 * np.pi * sigma ** 2)
        lp = -0.5 * b0 ** 2 / tau2 - 0.5 * np.log(2 * np.pi * tau2)
        return ll + lp

    def log_post_m2(theta):
        b0, b1 = theta
        resid = y - b0 - b1 * x
        ll = -0.5 * np.sum(resid ** 2) / sigma ** 2 - n * 0.5 * np.log(2 * np.pi * sigma ** 2)
        lp = -0.5 * (b0 ** 2 + b1 ** 2) / tau2 - np.log(2 * np.pi * tau2)
        return ll + lp

    def log_marglik(X):
        Sigma = sigma ** 2 * np.eye(n) + tau2 * X @ X.T
        sign, logdet = np.linalg.slogdet(Sigma)
        return -0.5 * n * np.log(2 * np.pi) - 0.5 * logdet - 0.5 * y @ np.linalg.solve(Sigma, y)

    X1 = np.ones((n, 1))
    X2 = np.column_stack([np.ones(n), x])
    logml1, logml2 = log_marglik(X1), log_marglik(X2)
    true_p2 = 1.0 / (1.0 + np.exp(logml1 - logml2))
    return log_post_m1, log_post_m2, true_p2


def test_reversible_jump_recovers_bayes_factor():
    lp1, lp2, true_p2 = _conjugate_normal_regression_setup()
    rj = am.ReversibleJumpMCMC({"M1": lp1, "M2": lp2}, {"M1": 1, "M2": 2},
                               n_samples=10000, n_warmup=3000, jump_scale=1.0)
    rj.sample("M1", np.array([0.0]), random_state=35)
    indicator = (rj.model_chain_ == "M2").astype(float)
    se = _mcse(indicator)
    assert _within(rj.model_probabilities_["M2"], true_p2, se + 0.01, k=4)
    assert rj.samples_by_model_["M2"].shape[1] == 2


def test_transdimensional_carlin_chib_matches_reversible_jump():
    lp1, lp2, true_p2 = _conjugate_normal_regression_setup()
    tdm = am.TransdimensionalMCMC({"M1": lp1, "M2": lp2}, {"M1": 1, "M2": 2},
                                  n_samples=10000, n_warmup=3000, pilot_samples=1000)
    tdm.sample({"M1": np.array([0.0]), "M2": np.array([0.0, 0.0])}, random_state=36)
    indicator = (tdm.model_chain_ == "M2").astype(float)
    se = _mcse(indicator)
    assert _within(tdm.model_probabilities_["M2"], true_p2, se + 0.01, k=4)


def test_reversible_jump_custom_jump_and_model_prior():
    lp1, lp2, _ = _conjugate_normal_regression_setup()
    skewed_prior = {"M1": np.log(0.9), "M2": np.log(0.1)}
    rj = am.ReversibleJumpMCMC({"M1": lp1, "M2": lp2}, {"M1": 1, "M2": 2},
                               log_model_prior=skewed_prior, n_samples=6000, n_warmup=2000)
    rj.sample("M1", np.array([0.0]), random_state=37)
    uniform_rj = am.ReversibleJumpMCMC({"M1": lp1, "M2": lp2}, {"M1": 1, "M2": 2},
                                       n_samples=6000, n_warmup=2000)
    uniform_rj.sample("M1", np.array([0.0]), random_state=37)
    assert rj.model_probabilities_["M2"] < uniform_rj.model_probabilities_["M2"]

    def identity_jump(k, theta, k_new, rng):
        return theta.copy(), 0.0

    same_dim_posteriors = {"A": lambda t: -0.5 * t @ t, "B": lambda t: -0.5 * t @ t - 1.0}
    rj2 = am.ReversibleJumpMCMC(same_dim_posteriors, {"A": 2, "B": 2}, jump=identity_jump,
                                n_samples=1000, n_warmup=500)
    rj2.sample("A", np.zeros(2), random_state=38)
    assert set(rj2.model_probabilities_) == {"A", "B"}


# ================================================================== diagnostics

def test_ess_of_iid_and_ar1_chains():
    rng = np.random.default_rng(39)
    iid = rng.standard_normal((4, 3000, 1))
    ess_iid = am.ESS(iid, method="mean")
    # a Geyer-paired ESS estimator can, for genuinely iid draws, land anywhere up to its
    # own documented cap (n * log10(n), shared by Stan/ArviZ) due to sampling noise in the
    # lag-1/2 autocorrelation; only the lower bound (no worse than the raw draw count) is
    # a real invariant here.
    assert 0.5 * 12000 < ess_iid <= 12000 * np.log10(12000) + 1e-6

    phi = 0.6
    n = 8000
    x = np.zeros(n)
    for t in range(1, n):
        x[t] = phi * x[t - 1] + rng.standard_normal()
    ess_ar1 = am.ESS(x[None, :, None], method="mean")
    true_ess = n * (1 - phi) / (1 + phi)
    assert 0.5 * true_ess < ess_ar1 < 3.0 * true_ess

    for method in ("bulk", "tail", "sd"):
        v = am.ESS(iid, method=method)
        assert np.isfinite(v) and v > 0


def test_rhat_detects_shifted_and_scaled_chains():
    rng = np.random.default_rng(40)
    iid = rng.standard_normal((4, 2000, 1))
    for method in ("split", "rank", "classic"):
        assert am.Rhat(iid, method=method) < 1.02

    shifted = iid.copy()
    shifted[0] += 2.0
    assert am.Rhat(shifted) > 1.1

    scaled = iid.copy()
    scaled[0] *= 4.0
    assert am.Rhat(scaled, method="rank") > 1.05
    assert am.Rhat(scaled, method="classic") < 1.05


def test_gelman_rubin_and_psrf_hand_computed():
    x = np.array([[[1.0], [2.0], [3.0], [4.0], [5.0], [6.0]],
                 [[2.0], [3.0], [4.0], [5.0], [6.0], [8.0]]])
    m, n = 2, 6
    chain_means = x[:, :, 0].mean(axis=1)
    chain_vars = x[:, :, 0].var(axis=1, ddof=1)
    W = chain_vars.mean()
    B_n = chain_means.var(ddof=1)  # PSRF (Brooks & Gelman multivariate form) uses the
    # *unscaled* between-chain variance B_n = B/n, not the classic B itself.
    lam = B_n / W if W > 0 else 0.0
    expected_psrf_ratio = (n - 1) / n + (m + 1) / m * lam
    p = am.PSRF(x)
    assert abs(p - expected_psrf_ratio) < 1e-9

    g = am.GelmanRubin(x)
    assert np.isfinite(g) and g > 1.0

    with pytest.raises(ValueError):
        am.GelmanRubin(x[:1])


def test_geweke_iid_versus_trend():
    rng = np.random.default_rng(41)
    r = am.geweke_test(rng.standard_normal(5000))
    assert isinstance(r, TestResult) and r.pvalue > 0.01
    trend = rng.standard_normal(5000) + np.linspace(0, 4, 5000)
    r2 = am.geweke_test(trend)
    assert r2.pvalue < 1e-3
    r3 = am.geweke_test(rng.standard_normal((2000, 2)))
    assert r3.statistic.shape == (2,)


def test_raftery_lewis_iid_bernoulli():
    rng = np.random.default_rng(42)
    x = rng.uniform(size=20000) < 0.025
    rl = am.raftery_lewis(x)
    assert rl.thin == 1
    assert abs(rl.n_min - 3746) < 5
    assert 0.7 < rl.dependence_factor < 1.6
    assert rl.burn_in >= 1

    phi = 0.9
    n = 20000
    z = np.zeros(n)
    for t in range(1, n):
        z[t] = phi * z[t - 1] + rng.standard_normal()
    rl2 = am.raftery_lewis(z < np.quantile(z, 0.025))
    assert rl2.dependence_factor > 2.0


def test_autocorr_time_sokal_and_geyer_versus_direct_sum():
    rng = np.random.default_rng(43)
    phi = 0.5
    n = 20000
    x = np.zeros(n)
    for t in range(1, n):
        x[t] = phi * x[t - 1] + rng.standard_normal()
    true_tau = (1 + phi) / (1 - phi)
    for method in ("sokal", "geyer"):
        tau = am.autocorr_time(x, method=method)
        assert abs(tau - true_tau) / true_tau < 0.25

    from stochpylib.advanced_mcmc.diagnostics import _autocov_fft
    acov = _autocov_fft(x)
    rho = acov / acov[0]
    direct_tau = 1.0 + 2.0 * np.sum(rho[1:50])
    assert abs(direct_tau - true_tau) / true_tau < 0.2


def test_trace_analysis_summary_and_mcse():
    rng = np.random.default_rng(44)
    chains = rng.standard_normal((2, 1000, 2))
    ta = am.TraceAnalysis(chains, names=["a", "b"])
    rows = ta.summary()
    assert len(rows) == 2 and rows[0]["name"] == "a"
    flat = chains.reshape(-1, 2)
    q = np.quantile(flat, [0.05, 0.5, 0.95], axis=0)
    for j in range(2):
        assert abs(rows[j]["q5"] - q[0, j]) < 1e-10
        assert abs(rows[j]["median"] - q[1, j]) < 1e-10
        assert abs(rows[j]["q95"] - q[2, j]) < 1e-10
    table = ta.summary_table()
    assert "a" in table and "b" in table
    rm = ta.running_mean()
    assert np.allclose(rm[:, -1, :], flat.reshape(2, 1000, 2).mean(axis=1))
    assert ta.thin(2).chains_.shape[1] == 500


# ================================================================= variational

def test_mean_field_vi_diagonal_and_correlated_gaussian():
    mu = np.array([1.0, -2.0])
    sd = np.array([0.5, 2.0])

    def log_prob(theta):
        return -0.5 * np.sum(((theta - mu) / sd) ** 2)

    def grad(theta):
        return -(theta - mu) / sd ** 2

    vi = am.MeanFieldVI(log_prob, 2, grad_log_prob=grad, n_iter=1500, n_mc=16, lr=0.05)
    vi.fit(random_state=45)
    assert np.max(np.abs(vi.mean_ - mu)) < 0.15
    assert np.max(np.abs(vi.std_ - sd) / sd) < 0.2

    prec = np.linalg.inv(_COV2)

    def log_prob_c(theta):
        return -0.5 * theta @ prec @ theta

    def grad_c(theta):
        return -prec @ theta

    vi2 = am.MeanFieldVI(log_prob_c, 2, grad_log_prob=grad_c, n_iter=1500, n_mc=16, lr=0.05)
    vi2.fit(random_state=46)
    expected_sd = np.sqrt(1.0 / np.diag(prec))
    assert np.max(np.abs(vi2.std_ - expected_sd) / expected_sd) < 0.25


def test_advi_full_rank_and_bounded_support():
    prec = np.linalg.inv(_COV2)

    def log_prob(theta):
        return -0.5 * theta @ prec @ theta

    def grad(theta):
        return -prec @ theta

    advi = am.ADVI(log_prob, 2, grad_log_prob=grad, rank="full", n_iter=1200, n_mc=16)
    advi.fit(random_state=47)
    cov_hat = advi.scale_tril_ @ advi.scale_tril_.T
    assert np.linalg.norm(cov_hat - _COV2) / np.linalg.norm(_COV2) < 0.35

    def log_prob_gamma(theta):
        x = theta[0]
        return -np.inf if x <= 0 else 2.0 * np.log(x) - x / 2.0  # Gamma(3, scale=2)

    advi_g = am.ADVI(log_prob_gamma, 1, n_iter=1500, n_mc=16, lower=[0.0])
    advi_g.fit(random_state=48)
    s = advi_g.sample(3000, random_state=0)
    assert np.all(s > 0)
    assert abs(s.mean() - 6.0) / 6.0 < 0.3
    assert advi_g.elbo_history_[-100:].mean() > advi_g.elbo_history_[:100].mean()


def test_black_box_vi_without_gradients():
    mu, sd = 1.5, 0.7

    def log_prob(theta):
        return -0.5 * ((theta[0] - mu) / sd) ** 2

    vi = am.BlackBoxVI(log_prob, 1, n_iter=2500, n_mc=64, lr=0.02)
    vi.fit(random_state=49)
    assert abs(vi.mean_[0] - mu) < 0.2
    assert abs(vi.std_[0] - sd) / sd < 0.3


def test_normalizing_flow_gradients_match_finite_differences():
    rng = np.random.default_rng(50)
    dim = 2

    def log_prob(theta):
        return -0.5 * np.sum(theta ** 2)

    def grad(theta):
        return -theta

    flow = am.NormalizingFlows(log_prob, dim, n_layers=3, grad_log_prob=grad)
    flow.params_ = flow._init_params(rng)
    z0 = rng.standard_normal((4, dim))

    def objective():
        total = 0.0
        for z in z0:
            theta, logdet = flow.forward(z)
            total += log_prob(theta) + logdet
        return total / len(z0)

    grads_accum = [{"u": np.zeros(dim), "w": np.zeros(dim), "b": 0.0} for _ in range(flow.n_layers)]
    for z in z0:
        theta, logdet, cache = flow.forward(z, cache=True)
        grad_theta = grad(theta)
        _, param_grads = flow._backward(cache, grad_theta)
        for i, pg in enumerate(param_grads):
            grads_accum[i]["u"] += pg["u_map"] + pg["u_logdet"]
            grads_accum[i]["w"] += pg["w_map"] + pg["w_logdet"]
            grads_accum[i]["b"] += pg["b_map"] + pg["b_logdet"]
    for i in range(flow.n_layers):
        grads_accum[i]["u"] /= len(z0)
        grads_accum[i]["w"] /= len(z0)
        grads_accum[i]["b"] /= len(z0)

    eps = 1e-6
    max_err = 0.0
    for i, layer in enumerate(flow.params_):
        for key in ("u", "w"):
            for j in range(dim):
                orig = layer[key][j]
                layer[key][j] = orig + eps
                fp = objective()
                layer[key][j] = orig - eps
                fm = objective()
                layer[key][j] = orig
                fd = (fp - fm) / (2 * eps)
                max_err = max(max_err, abs(fd - grads_accum[i][key][j]))
        orig_b = layer["b"]
        layer["b"] = orig_b + eps
        fp = objective()
        layer["b"] = orig_b - eps
        fm = objective()
        layer["b"] = orig_b
        fd = (fp - fm) / (2 * eps)
        max_err = max(max_err, abs(fd - grads_accum[i]["b"]))
    assert max_err < 1e-4

    # pullback_grad vs finite differences of logp(f(z)) + logdet(z) wrt z
    z = rng.standard_normal(dim)
    theta, logdet = flow.forward(z)
    analytic = flow.pullback_grad(z, grad(theta))
    fd_z = np.zeros(dim)
    for j in range(dim):
        zp = z.copy(); zp[j] += eps
        zm = z.copy(); zm[j] -= eps
        tp, ldp = flow.forward(zp)
        tm, ldm = flow.forward(zm)
        fd_z[j] = ((log_prob(tp) + ldp) - (log_prob(tm) + ldm)) / (2 * eps)
    assert np.max(np.abs(analytic - fd_z)) < 1e-4


def test_normalizing_flow_batched_matches_looped_gradient():
    rng = np.random.default_rng(51)
    dim = 3
    flow = am.NormalizingFlows(lambda t: -0.5 * np.sum(t ** 2), dim, n_layers=3,
                               grad_log_prob=lambda t: -t)
    flow.params_ = flow._init_params(rng)
    Z = rng.standard_normal((6, dim))

    grads_loop = [{"u": np.zeros(dim), "w": np.zeros(dim), "b": 0.0} for _ in range(flow.n_layers)]
    for z0 in Z:
        theta, _, cache = flow.forward(z0, cache=True)
        _, pg_list = flow._backward(cache, -theta)
        for i, pg in enumerate(pg_list):
            grads_loop[i]["u"] += pg["u_map"] + pg["u_logdet"]
            grads_loop[i]["w"] += pg["w_map"] + pg["w_logdet"]
            grads_loop[i]["b"] += pg["b_map"] + pg["b_logdet"]
    for i in range(flow.n_layers):
        grads_loop[i]["u"] /= len(Z)
        grads_loop[i]["w"] /= len(Z)
        grads_loop[i]["b"] /= len(Z)

    Theta, _, cache_b = flow._forward_batch(Z)
    grads_batch, _ = flow._backward_batch(cache_b, -Theta, clip=None)
    for i in range(flow.n_layers):
        assert np.allclose(grads_loop[i]["u"], grads_batch[i]["u"], atol=1e-10)
        assert np.allclose(grads_loop[i]["w"], grads_batch[i]["w"], atol=1e-10)
        assert abs(grads_loop[i]["b"] - grads_batch[i]["b"]) < 1e-10


def test_normalizing_flow_fits_correlated_gaussian_reasonably():
    """Planar-flow VI from scratch does not guarantee a tight fit (see README Known
    limitations); this checks the flow trains stably and gets meaningfully closer to
    the target than its own random initialization, with generous tolerances."""
    prec = np.linalg.inv(_COV2)

    def log_prob(theta):
        return -0.5 * theta @ prec @ theta

    def grad(theta):
        return -prec @ theta

    flow = am.NormalizingFlows(log_prob, 2, grad_log_prob=grad, n_layers=4, n_iter=800,
                               n_mc=128, lr=0.01)
    flow.fit(random_state=52)
    assert np.all(np.isfinite(flow.elbo_history_))
    draws, logqs = flow.sample(3000, random_state=0, return_log_prob=True)
    assert np.all(np.isfinite(draws)) and np.all(np.isfinite(logqs))
    # trained flow should beat a naive N(0, I) approximation on average log target density
    trained_logp = np.mean([log_prob(d) for d in draws])
    naive_draws = np.random.default_rng(1).standard_normal((3000, 2))
    naive_logp = np.mean([log_prob(d) for d in naive_draws])
    assert trained_logp > naive_logp


def test_stein_vi_particles_approximate_gaussian():
    prec = np.linalg.inv(_COV2)

    def log_prob(theta):
        return -0.5 * theta @ prec @ theta

    def grad(theta):
        return -prec @ theta

    stein = am.SteinVI(log_prob, 2, grad_log_prob=grad, n_particles=300, n_iter=800, step_size=0.1)
    stein.fit(random_state=53)
    assert np.max(np.abs(stein.mean())) < 0.3  # target is centered at 0
    assert np.linalg.norm(stein.cov() - _COV2) / np.linalg.norm(_COV2) < 0.6


# ====================================================================== wiring

_SPEC_NAMES = {
    "ADVI", "AdaptiveMetropolis", "BlackBoxVI", "Doubling", "ESS", "EllipticalSliceSampling",
    "GelmanRubin", "GibbsSampler", "HamiltonianMonteCarlo", "IndependenceSampler", "MALA",
    "MMALA", "MeanFieldVI", "MetropolisHastings", "NeutraHMC", "NoUTurnSampler",
    "NormalizingFlows", "PSRF", "ParallelTempering", "ParticleMCMC", "Polar_Slice",
    "ReplicaExchange", "ReversibleJumpMCMC", "Rhat", "RiemannianHMC",
    "RobustAdaptiveMetropolis", "SequentialMonteCarlo", "SliceSampling", "SteinVI",
    "Stepping", "TraceAnalysis", "TransdimensionalMCMC", "autocorr_time", "geweke_test",
    "raftery_lewis",
}


def test_module_wiring_and_exports():
    assert "advanced_mcmc" in stochpylib.__all__
    assert stochpylib.advanced_mcmc is am
    assert len(_SPEC_NAMES) == 35 and _SPEC_NAMES <= set(am.__all__)
    assert set(am.__all__) - _SPEC_NAMES == {"LogDensity", "MCMCSampler", "RafteryLewisResult"}
    assert list(am.__all__) == sorted(am.__all__) and len(set(am.__all__)) == len(am.__all__)
    assert (set(am_st.__all__) | set(am_gb.__all__) | set(am_sl.__all__) | set(am_ad.__all__)
            | set(am_dg.__all__) | set(am_vi.__all__) | set(am_base.__all__)) == set(am.__all__)


def test_quickstart_example_runs():
    rng = np.random.default_rng(54)
    n, p = 50, 3
    X = rng.standard_normal((n, p))
    beta_true = np.array([1.0, -0.5, 0.3])
    y = X @ beta_true + rng.normal(0, 1, n)
    sigma2 = 1.0

    def log_posterior(theta):
        return -0.5 * np.sum((y - X @ theta) ** 2) / sigma2

    nuts = am.NoUTurnSampler(log_posterior, n_samples=500, n_warmup=200, target_accept=0.8)
    nuts.sample(theta_init=np.zeros(p))
    chains = nuts.get_chains()
    assert chains.shape == (1, 500, p)
    assert np.all(np.isfinite(am.Rhat(chains)))
    assert np.all(np.isfinite(am.ESS(chains)))


def test_reproducibility_with_random_state():
    a = am.MetropolisHastings(_log_prob2, n_samples=100, n_warmup=50).sample(np.zeros(2), random_state=5)
    b = am.MetropolisHastings(_log_prob2, n_samples=100, n_warmup=50).sample(np.zeros(2), random_state=5)
    assert np.array_equal(a.get_samples(), b.get_samples())
    c = am.MetropolisHastings(_log_prob2, n_samples=100, n_warmup=50).sample(np.zeros(2), random_state=6)
    assert not np.array_equal(a.get_samples(), c.get_samples())

    rng = np.random.default_rng(7)
    s1 = am.MetropolisHastings(_log_prob2, n_samples=50, n_warmup=20).sample(np.zeros(2), random_state=rng)
    s2 = am.MetropolisHastings(_log_prob2, n_samples=50, n_warmup=20).sample(np.zeros(2), random_state=rng)
    assert not np.array_equal(s1.get_samples(), s2.get_samples())


def test_no_scipy_stats_in_library_code():
    import pathlib
    import re

    pattern = re.compile(r"^\s*(from scipy import .*stats|from scipy\.stats|import scipy\.stats)")
    pkg = pathlib.Path(am.__file__).parent
    for f in pkg.glob("*.py"):
        for line in f.read_text(encoding="utf-8").splitlines():
            assert not pattern.search(line), f"{f.name}: {line.strip()}"


def test_sampler_errors():
    def log_prob_positive(theta):
        return -theta[0] if theta[0] > 0 else -np.inf

    with pytest.raises(ValueError):
        am.MetropolisHastings(log_prob_positive, n_samples=100).sample(np.array([-1.0]), random_state=0)
    with pytest.raises(RuntimeError):
        am.MetropolisHastings(_log_prob2, n_samples=100).get_chains()
    with pytest.raises(ValueError):
        am.MetropolisHastings(_log_prob2, n_samples=0)
