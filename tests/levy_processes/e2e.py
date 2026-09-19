"""End-to-end API sweep for stochpylib.levy_processes: one realistic exercise per public
name. ``test_every_public_name_is_exercised`` fails the moment a name ships without one;
``test_exercise[<name>]`` runs each exercise as its own pytest case."""

import numpy as np
import pytest

from stochpylib import levy_processes as mod
from stochpylib.distributions import Exponential

EXERCISES = {}


def exercise(name):
    def deco(fn):
        EXERCISES[name] = fn
        return fn
    return deco


def _paths_ok(paths, n_paths, n_steps):
    paths = np.asarray(paths, dtype=float)
    assert paths.shape == (n_paths, n_steps + 1) and np.all(np.isfinite(paths))
    return paths


def _gbm():
    return mod.SDE(drift=lambda t, x: 0.05 * x, diffusion=lambda t, x: 0.2 * x, x0=100.0)


@exercise("LevyProcess")
def _levy():
    lp = mod.LevyProcess(b=0.1, sigma=0.3, jump_rate=1.0,
                         jump_sampler=lambda n, rng: rng.normal(0, 0.1, n),
                         jump_cf=lambda u: np.exp(-0.5 * 0.01 * u ** 2))
    p = _paths_ok(lp.simulate(1.0, 50, n_paths=200, random_state=0), 200, 50)
    assert abs(complex(np.asarray(lp.characteristic_function(0.0, 1.0)).ravel()[0]) - 1) < 1e-9
    assert abs(p[:, -1].mean() - 0.1) < 0.1


@exercise("StableProcess")
def _stable():
    p = _paths_ok(mod.StableProcess(alpha=1.7).simulate(1.0, 20, n_paths=50, random_state=1), 50, 20)
    assert np.all(p[:, 0] == 0.0)


@exercise("AlphaStableDistribution")
def _alpha_stable():
    d = mod.AlphaStableDistribution(alpha=1.5)
    x = np.asarray(d.rvs(200, random_state=2))
    assert x.shape == (200,) and abs(complex(d.cf(0.0)) - 1.0) < 1e-9


@exercise("SpectrallyPositive")
def _spectrally_positive():
    p = _paths_ok(mod.SpectrallyPositive(alpha=0.5).simulate(1.0, 20, n_paths=50, random_state=3), 50, 20)
    assert np.all(np.diff(p, axis=1) >= -1e-12)


@exercise("SubordinatedProcess")
def _subordinated():
    sp = mod.SubordinatedProcess(base=mod.StableProcess(alpha=2.0),
                                 subordinator=mod.GammaSubordinator(rate=2.0, scale=0.5))
    _paths_ok(sp.simulate(1.0, 20, n_paths=100, random_state=4), 100, 20)
    assert np.asarray(sp.sample([0.5, 1.0], random_state=5)).shape == (2,)


@exercise("LevyKhintchine")
def _lk():
    lk = mod.LevyKhintchine(b=0.1, sigma=0.2, jump_rate=1.5, jump_cf=lambda u: np.exp(-0.5 * u ** 2))
    u = np.array([0.5, -0.3])
    manual = 1j * u * 0.1 - 0.5 * 0.04 * u ** 2 + 1.5 * (np.exp(-0.5 * u ** 2) - 1.0)
    assert np.allclose(lk.exponent(u), manual)
    assert np.allclose(lk.characteristic_function(u, 2.0), np.exp(2.0 * manual))


@exercise("JumpDiffusion")
def _jd():
    jd = mod.JumpDiffusion(b=0.0, sigma=0.2, jump_rate=1.0,
                           jump_sampler=lambda n, rng: rng.normal(-0.1, 0.2, n))
    _paths_ok(jd.simulate(1.0, 30, n_paths=50, random_state=6), 50, 30)
    S = _paths_ok(jd.simulate_prices(100.0, 1.0, 30, n_paths=50, random_state=6), 50, 30)
    assert np.all(S > 0)


@exercise("MertonJumpDiffusion")
def _merton():
    m = mod.MertonJumpDiffusion(mu=0.05, sigma=0.2, jump_rate=1.0, jump_mean=-0.1, jump_std=0.2)
    price = m.call_price(100.0, 100.0, 1.0, r=0.05)
    mc = m.call_price_mc(100.0, 100.0, 1.0, r=0.05, n_paths=20000, random_state=7)
    assert abs(mc.estimate - price) < 5 * mc.std_error + 0.2


@exercise("KouJumpDiffusion")
def _kou():
    k = mod.KouJumpDiffusion(mu=0.05, sigma=0.2, jump_rate=0.0)
    from stochpylib.levy_processes.jump_diffusion import _black_scholes_call
    assert abs(k.call_price(100.0, 100.0, 1.0, 0.05) - _black_scholes_call(100.0, 100.0, 1.0, 0.05, 0.2)) < 0.01
    assert abs(complex(np.asarray(k.log_price_cf(100.0, 1.0, 0.05)(0.0)).ravel()[0]) - 1) < 1e-9


@exercise("BatesModel")
def _bates():
    b = mod.BatesModel(S0=100, jump_rate=0.5)
    S, V = b.simulate(1.0, 20, n_paths=50, random_state=8)   # (prices, variances)
    _paths_ok(S, 50, 20)
    assert np.all(S > 0) and np.all(np.isfinite(V))   # full-truncation scheme: raw v may dip < 0
    mc = b.call_price_mc(100.0, 1.0, n_paths=2000, n_steps=20, random_state=9)
    assert mc.estimate > 0


@exercise("VarianceGammaProcess")
def _vg():
    vg = mod.VarianceGammaProcess(sigma=0.2, nu=0.5, theta=0.0)
    p = _paths_ok(vg.simulate(1.0, 20, n_paths=200, random_state=10), 200, 20)
    assert abs(p[:, -1].mean()) < 0.1 and abs(complex(np.asarray(vg.characteristic_function(0.0, 1.0)).ravel()[0]) - 1) < 1e-9


@exercise("CGMYProcess")
def _cgmy():
    c = mod.CGMYProcess(C=1.0, G=5.0, M=5.0, Y=0.5)
    _paths_ok(c.simulate(1.0, 10, n_paths=20, random_state=11), 20, 10)
    assert np.isfinite(complex(np.asarray(c.characteristic_exponent(1.0)).ravel()[0]))


@exercise("NormalInverseGaussianProcess")
def _nig():
    n = mod.NormalInverseGaussianProcess(alpha=5.0, beta=-1.0, delta=1.0)
    _paths_ok(n.simulate(1.0, 20, n_paths=50, random_state=12), 50, 20)
    assert abs(complex(np.asarray(n.characteristic_function(0.0, 1.0)).ravel()[0]) - 1) < 1e-9


def _subordinator_workflow(sub, mean_rate=None):
    inc = np.asarray(sub.increments(0.1, 500, random_state=13), dtype=float)
    assert inc.shape == (500,) and np.all(inc >= 0)
    path = np.asarray(sub.sample([0.25, 0.5, 1.0], random_state=14), dtype=float)
    assert np.all(np.diff(path) >= 0)
    p = _paths_ok(sub.simulate(1.0, 10, n_paths=20, random_state=15), 20, 10)
    assert np.all(np.diff(p, axis=1) >= -1e-12)
    if mean_rate is not None:
        assert abs(inc.mean() / 0.1 - mean_rate) < 0.25 * mean_rate


@exercise("Subordinator")
def _subordinator():
    assert issubclass(mod.GammaSubordinator, mod.Subordinator)


@exercise("GammaSubordinator")
def _gamma_sub():
    _subordinator_workflow(mod.GammaSubordinator(rate=2.0, scale=0.5), mean_rate=1.0)


@exercise("InverseGaussianSubordinator")
def _ig_sub():
    _subordinator_workflow(mod.InverseGaussianSubordinator(lam=1.0), mean_rate=1.0)


@exercise("StableSubordinator")
def _stable_sub():
    _subordinator_workflow(mod.StableSubordinator(alpha=0.5))


@exercise("TemperingSubordinator")
def _tempering_sub():
    ts = mod.TemperingSubordinator(C=1.0, lam=5.0, alpha=0.5)
    _subordinator_workflow(ts, mean_rate=ts.mean_rate())
    assert ts.truncation_mass() > 0    # retained jump intensity above the floor


@exercise("SemiMarkovProcess")
def _semi_markov():
    sm = mod.SemiMarkovProcess([[0, 1], [1, 0]], [Exponential(2.0), Exponential(4.0)])
    times, states = sm.simulate(50.0, random_state=16)
    assert np.all(np.diff(times) >= 0) and set(np.unique(states)) <= {0, 1}


@exercise("RenewalProcess")
def _renewal():
    rp = mod.RenewalProcess(Exponential(2.0))
    ev = np.asarray(rp.simulate(20.0, random_state=17))
    assert np.all(np.diff(ev) > 0)
    m, se = rp.renewal_function(10.0, n_paths=500, random_state=18)
    assert abs(m - 20.0) < 4 * se + 0.5


@exercise("BranchingProcess")
def _branching():
    bp = mod.BranchingProcess(lambda n, rng: rng.poisson(0.7, n))
    z = np.asarray(bp.simulate(generations=10, random_state=19))
    assert z[0] == 1 and z.shape == (11,)
    assert bp.extinction_probability(generations=20, n_paths=500, random_state=20) > 0.8


@exercise("HawkesProcess")
def _hawkes():
    hp = mod.HawkesProcess(mu=0.5, alpha=0.3, beta=1.0)
    ev = np.asarray(hp.simulate(200.0, random_state=21))
    assert ev.size > 50 and abs(hp.branching_ratio() - 0.3) < 1e-9
    fit = mod.HawkesProcess().fit(ev, T=200.0)
    assert 0 < fit.branching_ratio() < 1
    assert 0 <= fit.ks_residuals()[1] <= 1 if isinstance(fit.ks_residuals(), tuple) else True
    assert float(np.asarray(hp.intensity(10.0, ev[ev < 10.0])).ravel()[0]) >= 0.5


@exercise("MultivariateHawkes")
def _mv_hawkes():
    mh = mod.MultivariateHawkes(mu=[0.3, 0.3], alpha_matrix=[[0.1, 0.05], [0.05, 0.1]], beta_vec=[1.0, 1.0])
    times, dims = mh.simulate(50.0, random_state=22)
    assert times.size == dims.size and np.allclose(mh.branching_matrix(), [[0.1, 0.05], [0.05, 0.1]])


@exercise("CoxProcess")
def _cox():
    ev, lam_T = mod.CoxProcess(lambda t: 3.0).simulate(20.0, random_state=23)
    assert 30 < np.asarray(ev).size < 90 and lam_T == pytest.approx(60.0, rel=0.2)


@exercise("GaussianRandomField")
def _grf():
    f = mod.GaussianRandomField(spectrum=lambda k: 1.0 / (1.0 + k ** 2), shape=(64,)).sample(random_state=24)
    assert f.shape == (64,) and np.all(np.isfinite(f))


@exercise("RandomMeasure")
def _random_measure():
    rm = mod.RandomMeasure(kind="gamma", shape_per_unit=2.0, scale=1.5)
    assert rm.sample((0, 1), random_state=25) > 0 and rm.sample((0, 0)) == 0.0


@exercise("SDE")
def _sde():
    sde = _gbm()
    assert sde.x0 == 100.0 and sde.a(0.0, 1.0) == pytest.approx(0.05) and sde.b(0.0, 1.0) == pytest.approx(0.2)


def _solver(fn, order_hint=None):
    p = _paths_ok(fn(_gbm(), T=1.0, n_steps=50, n_paths=4000, random_state=26), 4000, 50)
    se = p[:, -1].std(ddof=1) / np.sqrt(4000)
    assert abs(p[:, -1].mean() - 100.0 * np.exp(0.05)) < 5 * se + 0.5


@exercise("Euler_Maruyama")
def _em():
    _solver(mod.Euler_Maruyama)


@exercise("Milstein")
def _milstein():
    _solver(mod.Milstein)


@exercise("Runge_Kutta_SDE")
def _rk():
    _solver(mod.Runge_Kutta_SDE)


@exercise("StochasticTaylor")
def _taylor():
    _solver(mod.StochasticTaylor)


@exercise("WeakApproximation")
def _weak():
    xT = np.asarray(mod.WeakApproximation(_gbm(), T=1.0, n_steps=20, n_paths=20000, random_state=27))
    se = xT.std(ddof=1) / np.sqrt(xT.size)
    assert abs(xT.mean() - 100.0 * np.exp(0.05)) < 6 * se + 0.3


@exercise("StrongApproximation")
def _strong():
    def exact(sde, T, n_steps, n_paths, rng):
        dt = T / n_steps
        W = np.cumsum(rng.standard_normal((n_paths, n_steps)) * np.sqrt(dt), axis=1)
        t = np.linspace(dt, T, n_steps)
        out = np.empty((n_paths, n_steps + 1))
        out[:, 0] = 100.0
        out[:, 1:] = 100.0 * np.exp((0.05 - 0.02) * t + 0.2 * W)
        return out

    res = mod.StrongApproximation(_gbm(), exact, {"EM": mod.Euler_Maruyama, "Milstein": mod.Milstein},
                                  step_sizes=[8, 16, 32], n_paths=2000, random_state=28)
    assert res["Milstein"]["error"][-1] < res["EM"]["error"][-1]


def test_every_public_name_is_exercised():
    assert set(EXERCISES) == set(mod.__all__), (
        set(mod.__all__) - set(EXERCISES), set(EXERCISES) - set(mod.__all__))


@pytest.mark.parametrize("name", sorted(mod.__all__))
def test_exercise(name):
    EXERCISES[name]()
