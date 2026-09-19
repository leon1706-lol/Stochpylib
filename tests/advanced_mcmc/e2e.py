"""End-to-end API sweep for stochpylib.advanced_mcmc: one realistic exercise per public
name. ``test_every_public_name_is_exercised`` fails the moment a name ships without one;
``test_exercise[<name>]`` runs each exercise as its own pytest case."""

import numpy as np
import pytest

from stochpylib import advanced_mcmc as mod
from stochpylib.statistics import TestResult

EXERCISES = {}


def exercise(name):
    def deco(fn):
        EXERCISES[name] = fn
        return fn
    return deco


def _finite(a):
    return bool(np.all(np.isfinite(np.asarray(a))))


_MU = np.array([1.0, -1.0])
_RHO = 0.6
_COV = np.array([[1.0, _RHO], [_RHO, 1.0]])
_PREC = np.linalg.inv(_COV)


def _log_prob(theta):
    d = theta - _MU
    return -0.5 * d @ _PREC @ d


def _grad_log_prob(theta):
    return -_PREC @ (theta - _MU)


def _iid_chains():
    return np.random.default_rng(0).standard_normal((4, 400, 2))


@exercise("MetropolisHastings")
def _mh():
    s = mod.MetropolisHastings(_log_prob, n_samples=500, n_warmup=200).sample(np.zeros(2), random_state=0)
    assert s.get_samples().shape == (500, 2) and 0.0 < s.acceptance_rate_ <= 1.0


@exercise("IndependenceSampler")
def _ind():
    s = mod.IndependenceSampler(_log_prob, lambda r: _MU + 2 * r.standard_normal(2),
                                lambda t: -0.5 * np.sum(((t - _MU) / 2) ** 2),
                                n_samples=500, n_warmup=200).sample(_MU, random_state=1)
    assert _finite(s.get_samples())


@exercise("GibbsSampler")
def _gibbs():
    def c0(theta, r):
        m = _MU[0] + _RHO * (theta[1] - _MU[1])
        return m + np.sqrt(1 - _RHO ** 2) * r.standard_normal()

    def c1(theta, r):
        m = _MU[1] + _RHO * (theta[0] - _MU[0])
        return m + np.sqrt(1 - _RHO ** 2) * r.standard_normal()

    s = mod.GibbsSampler(conditionals=[c0, c1], n_samples=500, n_warmup=200).sample(np.zeros(2), random_state=2)
    assert s.acceptance_rate_ == 1.0 and s.get_samples().shape == (500, 2)


@exercise("AdaptiveMetropolis")
def _am():
    s = mod.AdaptiveMetropolis(_log_prob, n_samples=500, n_warmup=300).sample(np.zeros(2), random_state=3)
    assert s.proposal_cov_.shape == (2, 2)


@exercise("RobustAdaptiveMetropolis")
def _ram():
    s = mod.RobustAdaptiveMetropolis(_log_prob, n_samples=500, n_warmup=300).sample(np.zeros(2), random_state=4)
    assert s.proposal_chol_.shape == (2, 2)


@exercise("HamiltonianMonteCarlo")
def _hmc():
    s = mod.HamiltonianMonteCarlo(_log_prob, _grad_log_prob, n_samples=300, n_warmup=300,
                                  n_leapfrog=8).sample(np.zeros(2), random_state=5)
    assert s.step_size_ > 0 and s.divergences_ >= 0


@exercise("NoUTurnSampler")
def _nuts():
    s = mod.NoUTurnSampler(_log_prob, _grad_log_prob, n_samples=300, n_warmup=300).sample(np.zeros(2), random_state=6)
    assert s.mean_tree_depth_ > 0 and s.get_chains().shape == (1, 300, 2)


@exercise("MALA")
def _mala():
    s = mod.MALA(_log_prob, _grad_log_prob, n_samples=400, n_warmup=300).sample(np.zeros(2), random_state=7)
    assert 0.0 < s.acceptance_rate_ <= 1.0


@exercise("MMALA")
def _mmala():
    s = mod.MMALA(_log_prob, _grad_log_prob, n_samples=300, n_warmup=200).sample(np.zeros(2), random_state=8)
    assert _finite(s.get_samples())


@exercise("RiemannianHMC")
def _rhmc():
    s = mod.RiemannianHMC(_log_prob, _grad_log_prob, metric=lambda t: _PREC, n_samples=150,
                          n_warmup=150, n_leapfrog=4).sample(np.zeros(2), random_state=9)
    assert _finite(s.get_samples())


@exercise("NeutraHMC")
def _neutra():
    s = mod.NeutraHMC(_log_prob, _grad_log_prob, n_samples=200, n_warmup=200,
                      flow_layers=3, flow_iter=150, flow_lr=0.02).sample(np.zeros(2), random_state=10)
    assert s.flow_ is not None and _finite(s.get_samples())


@exercise("Stepping")
def _stepping():
    st = mod.Stepping(width=1.0, max_steps=50)
    L, R = st.find_interval(lambda x: -0.5 * x ** 2, 0.0, -1.0, np.random.default_rng(0))
    assert L < 0 < R


@exercise("Doubling")
def _doubling():
    d = mod.Doubling(width=1.0, max_doublings=10)
    L, R = d.find_interval(lambda x: -0.5 * x ** 2, 0.0, -1.0, np.random.default_rng(0))
    assert L < 0 < R and d.accept(lambda x: -0.5 * x ** 2, 0.0, 0.1, -1.0, L, R) in (True, False)


@exercise("SliceSampling")
def _slice():
    s = mod.SliceSampling(_log_prob, n_samples=400, n_warmup=100).sample(np.zeros(2), random_state=11)
    assert s.get_samples().shape == (400, 2)


@exercise("EllipticalSliceSampling")
def _ess_slice():
    K = np.eye(2) * 2.0

    def loglik(f):
        return -0.5 * np.sum((f - 0.3) ** 2) / 0.09

    s = mod.EllipticalSliceSampling(loglik, prior_cov=K, n_samples=400, n_warmup=100).sample(np.zeros(2), random_state=12)
    assert _finite(s.get_samples())


@exercise("Polar_Slice")
def _polar():
    def lp(theta):
        return -0.5 * np.sum(theta ** 2)

    s = mod.Polar_Slice(lp, n_samples=300, n_warmup=100).sample(np.ones(3), random_state=13)
    assert s.get_samples().shape == (300, 3)
    with pytest.raises(ValueError):
        mod.Polar_Slice(lp, n_samples=10, n_warmup=5).sample(np.array([1.0]), random_state=0)


@exercise("ReplicaExchange")
def _replica():
    def lps(theta, k):
        beta = [1.0, 0.3][k]
        return -0.5 * beta * np.sum(theta ** 2)

    s = mod.ReplicaExchange(lps, n_replicas=2, n_samples=300, n_warmup=150).sample(np.zeros(1), random_state=14)
    assert s.swap_acceptance_.shape == (1,) and s.get_chains().shape == (1, 300, 1)


@exercise("ParallelTempering")
def _pt():
    def lp(theta):
        x = theta[0]
        return np.logaddexp(-0.5 * ((x - 3) / 0.3) ** 2, -0.5 * ((x + 3) / 0.3) ** 2)

    s = mod.ParallelTempering(lp, n_temps=3, max_temp=20.0, n_samples=300, n_warmup=200,
                              pilot_rounds=1, pilot_iters=50).sample(np.array([3.0]), random_state=15)
    assert s.temperatures_[0] == 1.0 and _finite(s.get_samples())


@exercise("SequentialMonteCarlo")
def _smc():
    y = np.array([1.0])

    def log_prior(theta):
        return -0.5 * np.sum(theta ** 2)

    def log_lik(theta):
        return -0.5 * np.sum(((theta - y) / 0.5) ** 2)

    def prior_sampler(n, rng):
        return rng.standard_normal((n, 1))

    s = mod.SequentialMonteCarlo(log_prior, log_lik, prior_sampler, n_particles=300, n_mcmc=2).sample(random_state=16)
    assert np.isfinite(s.log_evidence_) and s.betas_[-1] == 1.0


@exercise("ParticleMCMC")
def _pmcmc():
    rng = np.random.default_rng(0)
    T = 15
    y = rng.standard_normal(T)

    def log_prior(theta):
        return 0.0 if abs(theta[0]) < 1 else -np.inf

    def initial_sampler(theta, r):
        return r.standard_normal(50)

    def transition_sampler(particles, theta, r):
        particles = np.asarray(particles)
        return theta[0] * particles + r.standard_normal(particles.shape)

    def obs_logpdf(particles, yt, theta):
        particles = np.asarray(particles).reshape(-1)
        return -0.5 * (yt - particles) ** 2

    s = mod.ParticleMCMC(y, log_prior, initial_sampler, transition_sampler, obs_logpdf,
                         n_particles=50, n_samples=100, n_warmup=100).sample(np.array([0.3]), random_state=17)
    assert s.get_samples().shape == (100, 1)


@exercise("ReversibleJumpMCMC")
def _rj():
    n = 10
    rng = np.random.default_rng(0)
    x = rng.standard_normal(n)
    y = 0.2 * x + rng.standard_normal(n) * 0.5
    tau2 = 10.0

    def lp1(theta):
        b0 = theta[0]
        ll = -0.5 * np.sum((y - b0) ** 2) - n * 0.5 * np.log(2 * np.pi)
        lp = -0.5 * b0 ** 2 / tau2 - 0.5 * np.log(2 * np.pi * tau2)
        return ll + lp

    def lp2(theta):
        b0, b1 = theta
        ll = -0.5 * np.sum((y - b0 - b1 * x) ** 2) - n * 0.5 * np.log(2 * np.pi)
        lp = -0.5 * (b0 ** 2 + b1 ** 2) / tau2 - np.log(2 * np.pi * tau2)
        return ll + lp

    rj = mod.ReversibleJumpMCMC({"M1": lp1, "M2": lp2}, {"M1": 1, "M2": 2}, n_samples=500, n_warmup=200)
    rj.sample("M1", np.array([0.0]), random_state=18)
    assert set(rj.model_probabilities_) == {"M1", "M2"}
    assert abs(sum(rj.model_probabilities_.values()) - 1.0) < 1e-9


@exercise("TransdimensionalMCMC")
def _tdm():
    n = 10
    rng = np.random.default_rng(0)
    x = rng.standard_normal(n)
    y = 0.2 * x + rng.standard_normal(n) * 0.5
    tau2 = 10.0

    def lp1(theta):
        b0 = theta[0]
        ll = -0.5 * np.sum((y - b0) ** 2) - n * 0.5 * np.log(2 * np.pi)
        lp = -0.5 * b0 ** 2 / tau2 - 0.5 * np.log(2 * np.pi * tau2)
        return ll + lp

    def lp2(theta):
        b0, b1 = theta
        ll = -0.5 * np.sum((y - b0 - b1 * x) ** 2) - n * 0.5 * np.log(2 * np.pi)
        lp = -0.5 * (b0 ** 2 + b1 ** 2) / tau2 - np.log(2 * np.pi * tau2)
        return ll + lp

    tdm = mod.TransdimensionalMCMC({"M1": lp1, "M2": lp2}, {"M1": 1, "M2": 2},
                                   n_samples=400, n_warmup=200, pilot_samples=200)
    tdm.sample({"M1": np.array([0.0]), "M2": np.array([0.0, 0.0])}, random_state=19)
    assert abs(sum(tdm.model_probabilities_.values()) - 1.0) < 1e-9


@exercise("Rhat")
def _rhat():
    r = mod.Rhat(_iid_chains())
    assert r.shape == (2,) and np.all(np.abs(r - 1.0) < 0.05)


@exercise("ESS")
def _ess_diag():
    e = mod.ESS(_iid_chains(), method="mean")
    assert e.shape == (2,) and np.all(e > 0)


@exercise("GelmanRubin")
def _gr():
    g = mod.GelmanRubin(_iid_chains())
    assert g.shape == (2,) and np.all(np.abs(g - 1.0) < 0.1)


@exercise("PSRF")
def _psrf():
    p = mod.PSRF(_iid_chains())
    assert abs(p - 1.0) < 0.2


@exercise("geweke_test")
def _geweke():
    r = mod.geweke_test(np.random.default_rng(0).standard_normal(2000))
    assert isinstance(r, TestResult)


@exercise("raftery_lewis")
def _rl():
    rl = mod.raftery_lewis(np.random.default_rng(0).uniform(size=5000) < 0.025)
    assert isinstance(rl, mod.RafteryLewisResult) and rl.n_total > 0


@exercise("autocorr_time")
def _act():
    tau = mod.autocorr_time(np.random.default_rng(0).standard_normal(3000))
    assert tau > 0


@exercise("TraceAnalysis")
def _trace():
    ta = mod.TraceAnalysis(_iid_chains())
    rows = ta.summary()
    assert len(rows) == 2 and "rhat" in rows[0]


@exercise("MeanFieldVI")
def _mfvi():
    vi = mod.MeanFieldVI(_log_prob, 2, grad_log_prob=_grad_log_prob, n_iter=200, n_mc=16).fit(random_state=20)
    assert vi.mean_.shape == (2,) and _finite(vi.sample(5, random_state=0))


@exercise("ADVI")
def _advi():
    vi = mod.ADVI(_log_prob, 2, grad_log_prob=_grad_log_prob, n_iter=200, n_mc=16).fit(random_state=21)
    assert _finite(vi.sample(5, random_state=0))


@exercise("BlackBoxVI")
def _bbvi():
    vi = mod.BlackBoxVI(_log_prob, 2, n_iter=200, n_mc=32).fit(random_state=22)
    assert vi.mean_.shape == (2,)


@exercise("NormalizingFlows")
def _flows():
    flow = mod.NormalizingFlows(_log_prob, 2, grad_log_prob=_grad_log_prob, n_layers=2,
                                n_iter=150, n_mc=32).fit(random_state=23)
    theta, logdet = flow.forward(np.zeros(2))
    assert _finite(theta) and np.isfinite(logdet)
    assert _finite(flow.sample(5, random_state=0))


@exercise("SteinVI")
def _stein():
    vi = mod.SteinVI(_log_prob, 2, grad_log_prob=_grad_log_prob, n_particles=50, n_iter=100).fit(random_state=24)
    assert vi.particles_.shape == (50, 2) and vi.mean().shape == (2,)


@exercise("LogDensity")
def _logdensity():
    ld = mod.LogDensity(_log_prob, _grad_log_prob)
    assert np.isclose(ld(_MU), 0.0) and _finite(ld.grad(np.zeros(2)))
    from stochpylib.distributions import Normal
    ld2 = mod.LogDensity.from_distribution(Normal(0.0, 1.0))
    assert np.isfinite(ld2(np.array([0.5])))


@exercise("MCMCSampler")
def _mcmc_base():
    assert issubclass(mod.MetropolisHastings, mod.MCMCSampler)
    s = mod.MetropolisHastings(_log_prob, n_samples=50, n_warmup=50).sample(np.zeros(2), random_state=0)
    assert s.get_chains().shape == (1, 50, 2)


@exercise("RafteryLewisResult")
def _rlresult():
    rl = mod.raftery_lewis(np.random.default_rng(1).uniform(size=3000) < 0.025)
    assert isinstance(rl, mod.RafteryLewisResult) and rl.dependence_factor > 0


def test_every_public_name_is_exercised():
    assert set(EXERCISES) == set(mod.__all__), (
        set(mod.__all__) - set(EXERCISES), set(EXERCISES) - set(mod.__all__))


@pytest.mark.parametrize("name", sorted(mod.__all__))
def test_exercise(name):
    EXERCISES[name]()
