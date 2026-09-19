"""End-to-end API sweep for stochpylib.montecarlo: one realistic exercise per public
name. ``test_every_public_name_is_exercised`` fails the moment a name ships without one;
``test_exercise[<name>]`` runs each exercise as its own pytest case."""

import numpy as np
import pytest

from stochpylib import montecarlo as mod
from stochpylib.distributions import Normal, Weibull

EXERCISES = {}


def exercise(name):
    def deco(fn):
        EXERCISES[name] = fn
        return fn
    return deco


def _in_band(res, truth, k=4):
    return abs(res.estimate - truth) < k * res.std_error + 1e-12


@exercise("SobolSequence")
def _sobol():
    s = mod.SobolSequence(dim=3, random_state=0)
    pts = s.generate(256)
    assert pts.shape == (256, 3) and np.all((pts >= 0) & (pts < 1))
    s.reset()
    assert s.generate_block(4).shape == (16, 3)


@exercise("HaltonSequence")
def _halton():
    pts = mod.HaltonSequence(dim=2).generate(100)
    assert pts.shape == (100, 2) and abs(pts.mean() - 0.5) < 0.1


@exercise("FaureSequence")
def _faure():
    pts = mod.FaureSequence(dim=3).generate(81)
    assert pts.shape == (81, 3) and np.all((pts >= 0) & (pts < 1))


@exercise("NiederreiterSequence")
def _niederreiter():
    s = mod.NiederreiterSequence(dim=2)
    assert s.generate(64).shape == (64, 2)
    s.reset()
    assert s.generate_block(3).shape == (8, 2)


@exercise("DigitalNet")
def _digital_net():
    assert mod.DigitalNet(dim=2).generate_block(4).shape == (16, 2)


@exercise("DigitalNetBase2")
def _digital_net_base2():
    net = mod.DigitalNetBase2(dim=2)
    assert net.generate(10).shape == (10, 2)
    net.reset()


@exercise("LowDiscrepancy")
def _low_discrepancy():
    seq = mod.LowDiscrepancy("halton", dim=2)
    assert seq.generate(50).shape == (50, 2)


@exercise("simulate")
def _simulate():
    res = mod.simulate(statistic=lambda x: float(np.mean(x ** 2)),
                       sampler=lambda rng: rng.standard_normal(4), n_simulations=5000,
                       random_state=0)
    assert _in_band(res, 1.0)


@exercise("crude_mc")
def _crude():
    res = mod.crude_mc(lambda p: p[:, 0] ** 2, n=20000, random_state=1)
    assert _in_band(res, 1 / 3)


@exercise("importance_sampling")
def _importance():
    res = mod.importance_sampling(
        lambda p: (p[:, 0] > 3).astype(float), lambda p: Normal(0, 1).pdf(p[:, 0]),
        lambda n, rng: rng.normal(3.0, 1.0, size=(n, 1)), lambda p: Normal(3, 1).pdf(p[:, 0]),
        n=20000, random_state=2)
    assert abs(res.estimate - (1 - Normal(0, 1).cdf(3.0))) < 5 * res.std_error + 1e-4


@exercise("rejection_sampling")
def _rejection():
    samples, rate = mod.rejection_sampling(
        lambda p: Normal(0, 1).pdf(p[:, 0]), lambda n, rng: rng.uniform(-6, 6, size=(n, 1)),
        lambda p: np.full(p.shape[0], 1 / 12.0), n_samples=2000, k_bound=0.4 * 12 * 1.05,
        random_state=3)
    assert samples.shape == (2000, 1) and 0 < rate <= 1 and abs(samples.mean()) < 0.1


@exercise("stratified_sampling")
def _stratified():
    res = mod.stratified_sampling(lambda p: p[:, 0] ** 2, bounds=(0.0, 1.0), n_strata=50,
                                  n_per_stratum=20, random_state=4)
    assert abs(res.estimate - 1 / 3) < 0.01


@exercise("quasi_montecarlo")
def _qmc():
    res = mod.quasi_montecarlo(lambda p: p[:, 0] * p[:, 1], dim=2, n=4096, random_state=5)
    assert abs(res.estimate - 0.25) < 0.01


@exercise("AntitheticVariates")
def _antithetic():
    av = mod.AntitheticVariates(n_simulations=20000, random_state=6)
    call = av.price_european_call(S=100, K=100, T=1, r=0.05, sigma=0.2)
    put = av.price_european_put(S=100, K=100, T=1, r=0.05, sigma=0.2)
    assert abs((call.estimate - put.estimate) - (100 - 100 * np.exp(-0.05))) < 0.5
    assert _in_band(av.estimate(lambda u: u[:, 0] ** 2), 1 / 3)


@exercise("ControlVariates")
def _control():
    cv = mod.ControlVariates(n_simulations=20000, random_state=7)
    res = cv.estimate(lambda u: np.exp(u[:, 0]), lambda u: u[:, 0], 0.5)
    assert _in_band(res, np.e - 1)


@exercise("StratifiedSampling")
def _stratified_cls():
    res = mod.StratifiedSampling(n_strata=32, dim=1, n_per_stratum=8, random_state=8).estimate(
        lambda u: u[:, 0] ** 2)
    assert abs(res.estimate - 1 / 3) < 0.02


@exercise("LatinHypercubeSampling")
def _lhs():
    pts = mod.LatinHypercubeSampling(dim=3, n=64, random_state=9).generate()
    assert pts.shape == (64, 3)
    assert np.all(np.sort(np.floor(pts[:, 0] * 64)) == np.arange(64))


@exercise("OrthogonalSampling")
def _orthogonal():
    pts = mod.OrthogonalSampling(dim=2, n=64, random_state=10).generate()
    assert pts.shape[1] == 2 and np.all((pts >= 0) & (pts <= 1))


@exercise("ConditionedMC")
def _conditioned():
    res = mod.ConditionedMC(n_simulations=20000, random_state=11).estimate(
        cond_expectation=lambda y: y ** 2 + 1.0, y_sampler=lambda rng: rng.standard_normal())
    assert _in_band(res, 2.0)


@exercise("RejectionControl")
def _rejection_control():
    res = mod.RejectionControl(n_simulations=20000, random_state=12).estimate(
        lambda p: p[:, 0] ** 2, target_pdf=lambda p: Normal(0, 1).pdf(p[:, 0]),
        proposal_sampler=lambda n, rng: rng.normal(0, 1.5, size=(n, 1)),
        proposal_pdf=lambda p: Normal(0, 1.5).pdf(p[:, 0]))
    assert _in_band(res, 1.0, k=5)


@exercise("MonteCarloIntegration")
def _integration():
    mci = mod.MonteCarloIntegration(lambda p: np.sin(p[:, 0]), bounds=[(0.0, np.pi)],
                                    method="qmc", random_state=13)
    assert abs(mci.estimate(n=4096).estimate - 2.0) < 0.02


@exercise("pi_estimation")
def _pi():
    assert _in_band(mod.pi_estimation(n=50000, random_state=14), np.pi)


@exercise("option_pricing_mc")
def _option():
    res = mod.option_pricing_mc(n=20000, random_state=15)
    assert abs(res.estimate - 10.45) < 4 * res.std_error + 0.05


@exercise("risk_analysis")
def _risk():
    r = mod.risk_analysis(np.random.default_rng(16).normal(0, 1, 5000), alpha=0.95)
    assert 1.4 < r.estimate < 1.9 and r.expected_shortfall > r.estimate


@exercise("reliability_mc")
def _reliability():
    res = mod.reliability_mc(lambda X: X[:, 0], [Weibull(2.0, 10.0)], threshold=5.0,
                             n=20000, random_state=17)
    assert abs(res.estimate - (1 - np.exp(-0.25))) < 5 * res.std_error


@exercise("sensitivity_analysis")
def _sensitivity():
    out = mod.sensitivity_analysis(lambda X: 3 * X[:, 0] + X[:, 1], [Normal(0, 1), Normal(0, 1)],
                                   n=5000, random_state=18)
    assert out[0]["pearson"] > out[1]["pearson"] > 0


@exercise("MCResult")
def _mcresult():
    r = mod.MCResult(estimate=1.0, std_error=0.1, n_samples=100)
    lo, hi = r.confidence_interval(0.95)
    assert lo < 1.0 < hi and float(r) == 1.0


def test_every_public_name_is_exercised():
    assert set(EXERCISES) == set(mod.__all__), (
        set(mod.__all__) - set(EXERCISES), set(EXERCISES) - set(mod.__all__))


@pytest.mark.parametrize("name", sorted(mod.__all__))
def test_exercise(name):
    EXERCISES[name]()
