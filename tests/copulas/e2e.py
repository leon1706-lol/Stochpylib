"""End-to-end API sweep for stochpylib.copulas: one realistic exercise per public name.
``test_every_public_name_is_exercised`` fails the moment a name ships without one;
``test_exercise[<name>]`` runs each exercise as its own pytest case."""

import functools

import numpy as np
import pytest

from stochpylib import copulas as mod

EXERCISES = {}


def exercise(name):
    def deco(fn):
        EXERCISES[name] = fn
        return fn
    return deco


@functools.lru_cache(maxsize=None)
def _gauss_data(n=800, rho=0.6):
    rng = np.random.default_rng(0)
    z = rng.multivariate_normal([0, 0], [[1, rho], [rho, 1]], n)
    from scipy.special import ndtr
    return ndtr(z)


@functools.lru_cache(maxsize=None)
def _clayton_data():
    return mod.ClaytonCopula(theta=2.0).sample(800, random_state=1)


@functools.lru_cache(maxsize=None)
def _data_4d():
    rng = np.random.default_rng(2)
    R = np.array([[1, .5, .3, .2], [.5, 1, .4, .3], [.3, .4, 1, .5], [.2, .3, .5, 1]])
    from scipy.special import ndtr
    return ndtr(rng.multivariate_normal(np.zeros(4), R, 600))


def _bivariate_workflow(cop, data, expect_tau=None):
    fitted = cop.fit(data)
    u = fitted.sample(300, random_state=3)
    assert u.shape == (300, 2) and np.all((u > 0) & (u < 1))
    c = np.asarray(fitted.cdf(u[:20]), dtype=float)
    assert np.all((c >= -1e-9) & (c <= 1 + 1e-9))
    try:                                    # empirical families expose no closed-form density
        d = np.asarray(fitted.density(u[:20]), dtype=float)
        assert np.all(d >= 0) and np.isfinite(fitted.aic(u))
    except NotImplementedError:
        pass
    tau = fitted.kendall_tau()              # None when the family has no closed form
    if tau is not None:
        assert -1 <= tau <= 1
        if expect_tau is not None:
            assert abs(tau - expect_tau) < 0.12
    td = fitted.tail_dependence()
    assert set(td) >= {"lower", "upper"}
    return fitted


@exercise("BaseCopula")
def _base():
    assert issubclass(mod.GaussianCopula, mod.BaseCopula)


@exercise("GaussianCopula")
def _gaussian():
    f = _bivariate_workflow(mod.GaussianCopula(), _gauss_data(), expect_tau=2 / np.pi * np.arcsin(0.6))
    assert abs(f.correlation_[0, 1] - 0.6) < 0.1 and -1 <= f.spearman_rho() <= 1


@exercise("StudentTCopula")
def _student():
    f = _bivariate_workflow(mod.StudentTCopula(), _gauss_data())
    assert f.df_ > 1


@exercise("ClaytonCopula")
def _clayton():
    f = _bivariate_workflow(mod.ClaytonCopula(), _clayton_data(), expect_tau=0.5)
    assert f.tail_dependence()["lower"] > 0.3


@exercise("FrankCopula")
def _frank():
    _bivariate_workflow(mod.FrankCopula(), _clayton_data())


@exercise("GumbelCopula")
def _gumbel():
    f = _bivariate_workflow(mod.GumbelCopula(), _clayton_data())
    assert f.tail_dependence()["upper"] > 0


@exercise("JoeCopula")
def _joe():
    _bivariate_workflow(mod.JoeCopula(), _clayton_data())


@exercise("AliMikhailHaqCopula")
def _amh():
    weak = mod.FrankCopula(theta=1.0).sample(600, random_state=4)
    _bivariate_workflow(mod.AliMikhailHaqCopula(), weak)


@exercise("PlackettCopula")
def _plackett():
    _bivariate_workflow(mod.PlackettCopula(), _gauss_data())


@exercise("BB1Copula")
def _bb1():
    _bivariate_workflow(mod.BB1Copula(), _clayton_data())


@exercise("BB7Copula")
def _bb7():
    _bivariate_workflow(mod.BB7Copula(), _clayton_data())


@exercise("EmpiricalCopula")
def _empirical():
    _bivariate_workflow(mod.EmpiricalCopula(), _gauss_data())


@exercise("CheckerboardCopula")
def _checkerboard():
    f = _bivariate_workflow(mod.CheckerboardCopula(n_bins=8), _gauss_data())
    assert abs(f.cell_mass_.sum() - 1.0) < 1e-9


@exercise("BetaCopula")
def _beta():
    _bivariate_workflow(mod.BetaCopula(n_bins=10), _gauss_data())


@exercise("PairCopulaConstruction")
def _pcc():
    pc = mod.PairCopulaConstruction(mod.ClaytonCopula(theta=2.0), rotation=0)
    h = pc.h(np.array([0.3, 0.6]), np.array([0.5, 0.5]))
    back = pc.h_inv(h, np.array([0.5, 0.5]))
    assert np.allclose(back, [0.3, 0.6], atol=1e-6) and isinstance(pc.describe(), str)


def _vine_workflow(vine):
    vine.fit(_data_4d())
    s = vine.sample(200, random_state=5)
    assert s.shape == (200, 4) and np.all((s > 0) & (s < 1))
    assert np.isfinite(vine.loglik()) and np.isfinite(vine.aic())
    assert np.isfinite(vine.loglik(s, raw=False))
    assert isinstance(vine.summary(), str)
    tau = np.asarray(vine.kendall_tau(), dtype=float)     # pairwise tau matrix
    assert tau.shape == (4, 4) and np.all(np.abs(tau) <= 1)


@exercise("CVine")
def _cvine():
    _vine_workflow(mod.CVine(families=("gaussian", "clayton")))


@exercise("DVine")
def _dvine():
    _vine_workflow(mod.DVine(families=("gaussian", "clayton")))


@exercise("RVine")
def _rvine():
    _vine_workflow(mod.RVine(families=("gaussian", "clayton")))


@exercise("VineCopula")
def _vine_copula():
    _vine_workflow(mod.VineCopula(type="DVine", families=("gaussian", "frank")))


@exercise("VineStructureSelect")
def _vine_select():
    best = mod.VineStructureSelect(_data_4d(), types=("CVine", "DVine"), families=("gaussian",))
    assert np.isfinite(best.aic())


@exercise("CopulaFit")
def _copula_fit():
    fit = mod.CopulaFit(families=("clayton", "gaussian", "frank")).fit(_clayton_data())
    assert fit.best_name_ == "clayton" and fit.table_[0][0] == "clayton"


@exercise("CopulaSample")
def _copula_sample():
    cs = mod.CopulaSample(mod.GumbelCopula(theta=2.0))
    s = cs.sample(300, random_state=6)
    assert s.shape == (300, 2) and 0 <= float(cs.cdf([[0.5, 0.5]])[0]) <= 1


@exercise("kendall_tau")
def _kendall():
    u = _clayton_data()
    assert 0.3 < mod.kendall_tau(u[:, 0], u[:, 1]) < 0.7


@exercise("spearman_rho")
def _spearman():
    u = _gauss_data()
    assert 0.4 < mod.spearman_rho(u[:, 0], u[:, 1]) < 0.8


@exercise("tail_dependence")
def _tail():
    c = mod.ClaytonCopula(theta=2.0)
    out = mod.tail_dependence(c, data=_clayton_data())
    assert abs(out["lower"] - 2 ** (-0.5)) < 1e-9 and "lower_emp" in out


@exercise("copula_density")
def _density():
    d = mod.copula_density(mod.GaussianCopula().fit(_gauss_data()), np.array([[0.5, 0.5], [0.2, 0.8]]))
    assert np.asarray(d).shape == (2,) and np.all(np.asarray(d) >= 0)


@exercise("conditional_copula")
def _conditional():
    c = mod.GumbelCopula(theta=2.5)
    c.dimension = 2
    out = mod.conditional_copula(c, [0.0, 0.5, 1.0], [0.5, 0.5, 0.5])
    assert np.allclose([out[0], out[2]], [0.0, 1.0]) and 0 < out[1] < 1


def test_every_public_name_is_exercised():
    assert set(EXERCISES) == set(mod.__all__), (
        set(mod.__all__) - set(EXERCISES), set(EXERCISES) - set(mod.__all__))


@pytest.mark.parametrize("name", sorted(mod.__all__))
def test_exercise(name):
    EXERCISES[name]()
