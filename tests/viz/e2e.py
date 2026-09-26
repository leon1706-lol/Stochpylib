"""End-to-end exercise sweep: one realistic use per public name in
``stochpylib.viz.__all__``. Kept fast (small n, cheap constructors) -- exhaustive oracle
checks live in ``tests.py``. Every exercise renders a real model built from another
stochpylib module, not mock data.
"""

import xml.etree.ElementTree as ET

import numpy as np
import pytest

from stochpylib import viz as mod

EXERCISES = {}


def exercise(name):
    def deco(fn):
        EXERCISES[name] = fn
        return fn
    return deco


_RNG = np.random.default_rng(0)


def _svg(fig):
    ET.fromstring(fig.to_svg())
    return fig


# ------------------------------------------------------------------------- scene graph

@exercise("Figure")
def _figure():
    fig = mod.Figure(nrows=1, ncols=2, title="demo")
    fig.axes[0].line([0, 1, 2], [0, 1, 0])
    assert len(fig.axes) == 2
    _svg(fig)


@exercise("Axes")
def _axes():
    fig = mod.Figure()
    art = fig.ax.scatter([0, 1], [1, 0], label="pts")
    assert art.kind == "scatter"
    _svg(fig)


# ------------------------------------------------------------------------- distributions

@exercise("plot_pdf")
def _plot_pdf():
    from stochpylib.distributions import Gamma

    _svg(mod.plot_pdf(Gamma(shape=3.0, scale=2.0)))


@exercise("plot_pmf")
def _plot_pmf():
    from stochpylib.distributions import Poisson

    _svg(mod.plot_pmf(Poisson(4.0)))


@exercise("plot_cdf")
def _plot_cdf():
    data = _RNG.standard_normal(200)
    _svg(mod.plot_cdf(data, confidence=0.95))


@exercise("plot_survival")
def _plot_survival():
    from stochpylib.distributions import Weibull

    _svg(mod.plot_survival(Weibull(shape=1.5, scale=2.0)))


@exercise("plot_hazard")
def _plot_hazard():
    from stochpylib.distributions import Weibull

    _svg(mod.plot_hazard(Weibull(shape=1.5, scale=2.0)))


@exercise("plot_qqplot")
def _plot_qqplot():
    from stochpylib.distributions import Gamma

    g = Gamma(shape=3.0, scale=2.0)
    fig = mod.plot_qqplot(g.rvs(300, random_state=0), dist=g)
    assert fig.data["r"] > 0.9
    _svg(fig)


@exercise("plot_ppplot")
def _plot_ppplot():
    from stochpylib.distributions import Normal

    _svg(mod.plot_ppplot(_RNG.standard_normal(200), Normal(0, 1)))


@exercise("plot_histogram")
def _plot_histogram():
    from stochpylib.distributions import Gamma

    g = Gamma(shape=3.0, scale=2.0)
    _svg(mod.plot_histogram(g.rvs(400, random_state=1), dist=g, kde=True))


@exercise("plot_kde")
def _plot_kde():
    _svg(mod.plot_kde(_RNG.standard_normal(300)))


# ------------------------------------------------------------------------- processes

@exercise("plot_process")
def _plot_process():
    paths = _RNG.standard_normal((20, 60)).cumsum(axis=1)
    _svg(mod.plot_process(paths))


def _ar1_series(n=300, phi=0.7, seed=0):
    rng = np.random.default_rng(seed)
    x = np.zeros(n)
    for i in range(1, n):
        x[i] = phi * x[i - 1] + rng.standard_normal()
    return x


@exercise("plot_acf")
def _plot_acf():
    _svg(mod.plot_acf(_ar1_series()))


@exercise("plot_pacf")
def _plot_pacf():
    _svg(mod.plot_pacf(_ar1_series()))


@exercise("plot_periodogram")
def _plot_periodogram():
    t = np.arange(300)
    sig = np.sin(2 * np.pi * 5 * t / 300) + 0.2 * _RNG.standard_normal(300)
    _svg(mod.plot_periodogram(sig, fs=300))


@exercise("plot_spectrogram")
def _plot_spectrogram():
    t = np.arange(300)
    sig = np.sin(2 * np.pi * 5 * t / 300) + 0.2 * _RNG.standard_normal(300)
    _svg(mod.plot_spectrogram(sig, window_len=64, hop=16))


@exercise("plot_wavelet")
def _plot_wavelet():
    t = np.arange(256)
    sig = np.sin(2 * np.pi * t / 32)
    _svg(mod.plot_wavelet(sig))


@exercise("plot_trajectory")
def _plot_trajectory():
    _svg(mod.plot_trajectory(_ar1_series(), lag=1))


# ------------------------------------------------------------------------- diagnostics

def _hmc_sampler():
    from stochpylib.advanced_mcmc import HamiltonianMonteCarlo

    def logp(theta):
        return -0.5 * np.sum(theta ** 2)

    sampler = HamiltonianMonteCarlo(logp, n_samples=200, n_warmup=100, n_chains=2,
                                    step_size=0.5, n_leapfrog=8)
    sampler.sample(np.zeros(2), random_state=0)
    return sampler


@exercise("trace_plot")
def _trace_plot():
    _svg(mod.trace_plot(_hmc_sampler()))


@exercise("posterior_plot")
def _posterior_plot():
    _svg(mod.posterior_plot(_hmc_sampler(), hdi_prob=0.9))


@exercise("pair_plot")
def _pair_plot():
    _svg(mod.pair_plot(_hmc_sampler()))


def _ols_fixture():
    rng = np.random.default_rng(2)
    X = rng.standard_normal((50, 2))
    y = 1 + 2 * X[:, 0] - X[:, 1] + rng.standard_normal(50) * 0.5
    from stochpylib.statistics import linear_regression

    return X, linear_regression(X, y)


@exercise("residual_plot")
def _residual_plot():
    _, res = _ols_fixture()
    _svg(mod.residual_plot(res))


@exercise("leverage_plot")
def _leverage_plot():
    X, res = _ols_fixture()
    _svg(mod.leverage_plot(X, res))


@exercise("influence_plot")
def _influence_plot():
    X, res = _ols_fixture()
    _svg(mod.influence_plot(X, res))


@exercise("funnel_plot")
def _funnel_plot():
    effects = _RNG.normal(0.5, 0.2, 12)
    se = np.abs(_RNG.normal(0.3, 0.1, 12)) + 0.05
    fig = mod.funnel_plot(effects, se)
    assert "egger_pvalue" in fig.data
    _svg(fig)


# ------------------------------------------------------------------------- multivariate

@exercise("plot_heatmap")
def _plot_heatmap():
    _svg(mod.plot_heatmap(_RNG.random((5, 6))))


@exercise("plot_correlation")
def _plot_correlation():
    _svg(mod.plot_correlation(_RNG.standard_normal((80, 4))))


@exercise("plot_copula")
def _plot_copula():
    from stochpylib.copulas import ClaytonCopula

    _svg(mod.plot_copula(ClaytonCopula(theta=3.0), n=300, random_state=0))


@exercise("plot_scatter_matrix")
def _plot_scatter_matrix():
    _svg(mod.plot_scatter_matrix(_RNG.standard_normal((60, 3))))


@exercise("plot_biplot")
def _plot_biplot():
    _svg(mod.plot_biplot(_RNG.standard_normal((80, 4))))


@exercise("plot_dendrogram")
def _plot_dendrogram():
    X = np.vstack([_RNG.standard_normal((5, 2)), _RNG.standard_normal((5, 2)) + 8])
    _svg(mod.plot_dendrogram(X))


# ------------------------------------------------------------------------- special

@exercise("plot_markov_chain")
def _plot_markov_chain():
    P = np.array([[0.9, 0.1], [0.3, 0.7]])
    _svg(mod.plot_markov_chain(P, states=["A", "B"]))


@exercise("plot_brownian")
def _plot_brownian():
    _svg(mod.plot_brownian(n_paths=10, n_steps=50, random_state=0))


@exercise("plot_gp")
def _plot_gp():
    from stochpylib.gaussian_processes import GPRegression, RBFKernel

    X = _RNG.uniform(0, 10, 12)[:, None]
    y = np.sin(X[:, 0]) + _RNG.standard_normal(12) * 0.1
    gp = GPRegression(kernel=RBFKernel(length_scale=1.5), noise=0.05).fit(X, y)
    _svg(mod.plot_gp(gp))


@exercise("plot_survival_km")
def _plot_survival_km():
    durations = _RNG.exponential(10, 40)
    events = (_RNG.random(40) > 0.3).astype(float)
    _svg(mod.plot_survival_km(durations, events))


@exercise("plot_variogram")
def _plot_variogram():
    coords = _RNG.uniform(0, 10, size=(80, 2))
    values = _RNG.standard_normal(80)
    _svg(mod.plot_variogram(None, coords=coords, values=values, bins=8))


@exercise("plot_eigenvalues")
def _plot_eigenvalues():
    from stochpylib.random_matrix import GOE

    _svg(mod.plot_eigenvalues(GOE(n=80), random_state=0, bins=20))


def test_every_public_name_is_exercised():
    assert set(EXERCISES) == set(mod.__all__), (
        set(mod.__all__) - set(EXERCISES), set(EXERCISES) - set(mod.__all__))


@pytest.mark.parametrize("name", sorted(mod.__all__))
def test_exercise(name):
    EXERCISES[name]()
