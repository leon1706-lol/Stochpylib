"""End-to-end API sweep for stochpylib.survival: one realistic exercise per public name.
``test_every_public_name_is_exercised`` fails the moment a name ships without one;
``test_exercise[<name>]`` runs each exercise as its own pytest case."""

import numpy as np
import pytest

from stochpylib import survival as mod
from stochpylib.distributions import Weibull

EXERCISES = {}


def exercise(name):
    def deco(fn):
        EXERCISES[name] = fn
        return fn
    return deco


_RNG = np.random.default_rng(0)
_N = 600
_X = _RNG.integers(0, 2, _N).astype(float)          # treatment indicator
_T_TRUE = _RNG.exponential(np.where(_X == 1, 4.0, 2.0))
_C = _RNG.uniform(1.0, 10.0, _N)
T = np.minimum(_T_TRUE, _C)
E = (_T_TRUE <= _C).astype(int)
G = np.where(_X == 1, "B", "A")
CAUSE = np.where(E == 1, _RNG.integers(1, 3, _N), 0)  # competing risks: 0 censored, 1 / 2 causes
_TIMES = np.array([1.0, 2.0, 4.0])


def _monotone_survival(fitter):
    s = np.asarray(fitter.predict(_TIMES), dtype=float)
    assert np.all((s >= 0) & (s <= 1)) and np.all(np.diff(s) <= 1e-12)
    return s


@exercise("SurvivalFitter")
def _fitter():
    assert issubclass(mod.KaplanMeier, mod.SurvivalFitter)


@exercise("KaplanMeier")
def _km():
    km = mod.KaplanMeier().fit(T, E)
    s = _monotone_survival(km)
    assert 0.3 < s[0] < 0.9 and np.isfinite(km.median_survival_time_)


@exercise("NelsonAalen")
def _na():
    na = mod.NelsonAalen().fit(T, E)
    h = np.asarray(na.predict(_TIMES), dtype=float)
    assert np.all(h >= 0) and np.all(np.diff(h) >= 0)


@exercise("LifeTable")
def _life_table():
    lt = mod.LifeTable().fit(T, E, width=1.0)
    _monotone_survival(lt)


@exercise("EmpiricalSurvival")
def _empirical():
    es = mod.EmpiricalSurvival().fit(_T_TRUE)
    s = _monotone_survival(es)
    assert abs(s[1] - np.mean(_T_TRUE > 2.0)) < 1e-9


@exercise("BreslowEstimator")
def _breslow():
    be = mod.BreslowEstimator().fit(T, E, np.exp(-0.5 * _X))
    s0 = np.asarray(be.baseline_survival_(_TIMES), dtype=float)
    h0 = np.asarray(be.predict(_TIMES), dtype=float)     # baseline cumulative hazard
    assert np.all((s0 >= 0) & (s0 <= 1)) and np.all(np.diff(h0) >= -1e-12)
    assert np.allclose(np.exp(-h0), s0)


@exercise("SurvivalFunction")
def _survival_function():
    sf = mod.SurvivalFunction(durations=T, events=E)
    sfd = mod.SurvivalFunction(source=Weibull(1.5, 10.0))
    assert 0 < float(sf.predict([2.0])[0]) < 1 and abs(float(sfd.predict([10.0])[0]) - np.exp(-1)) < 1e-9


@exercise("HazardFunction")
def _hazard_function():
    hf = mod.HazardFunction(source=Weibull(1.0, 2.0))
    assert abs(float(hf.predict([1.0])[0]) - 0.5) < 1e-6


@exercise("CumulativeHazard")
def _cumulative_hazard():
    ch = mod.CumulativeHazard(durations=T, events=E)
    h = np.asarray(ch.predict(_TIMES), dtype=float)
    assert np.all(np.diff(h) >= -1e-12) and h[0] > 0


@exercise("ResidualLifetime")
def _residual():
    rl = mod.ResidualLifetime(source=Weibull(1.0, 2.0))
    assert abs(rl.value(1.0) - 2.0) < 0.05   # memoryless: mean residual life = scale


@exercise("MeanResidualLife")
def _mrl():
    mrl = mod.MeanResidualLife(durations=T, events=E)
    c = np.asarray(mrl.curve([0.5, 1.0]), dtype=float)
    assert c.shape == (2,) and np.all(np.isfinite(c))


def _parametric(cls, key):
    m = cls().fit(T, E)
    assert np.isfinite(m.loglik()) and np.isfinite(m.aic())
    s = np.asarray(m.survival_(_TIMES), dtype=float)
    h = np.asarray(m.hazard_(_TIMES), dtype=float)
    assert np.all((s >= 0) & (s <= 1)) and np.all(h >= 0)
    assert key in m.params_ and np.allclose(m.predict(_TIMES), s)


@exercise("WeibullSurvival")
def _weibull():
    _parametric(mod.WeibullSurvival, "shape")


@exercise("ExponentialSurvival")
def _exponential():
    _parametric(mod.ExponentialSurvival, "rate")


@exercise("LogNormalSurvival")
def _lognormal():
    _parametric(mod.LogNormalSurvival, "mu")


@exercise("LogLogisticSurvival")
def _loglogistic():
    m = mod.LogLogisticSurvival().fit(T, E)
    assert np.all(np.isfinite(list(m.params_.values()))) and np.isfinite(m.aic())


@exercise("GompertzSurvival")
def _gompertz():
    m = mod.GompertzSurvival().fit(T, E)
    assert np.all(np.isfinite(list(m.params_.values()))) and np.isfinite(m.loglik())


@exercise("CoxProportionalHazards")
def _cox():
    cph = mod.CoxProportionalHazards().fit(T, E, _X[:, None])
    assert cph.coefficients_[0] < 0 and cph.concordance_index_ > 0.5
    ph = np.asarray(cph.predict_partial_hazard(np.array([[0.0], [1.0]])), dtype=float)
    assert ph[1] < ph[0] and isinstance(cph.summary(), (str, dict, list))
    assert np.all(np.diff(np.asarray(cph.baseline_cumulative_hazard_(_TIMES))) >= -1e-12)


@exercise("StratifiedCox")
def _stratified_cox():
    strata = (T > np.median(T)).astype(int)
    sc = mod.StratifiedCox().fit(T, E, _X[:, None], strata)
    assert np.isfinite(sc.coefficients_[0])


@exercise("AcceleratedFailureTime")
def _aft():
    aft = mod.AcceleratedFailureTime().fit(T, E, _X[:, None])
    med = np.asarray(aft.predict_median(np.array([[0.0], [1.0]])), dtype=float)
    assert med[1] > med[0]


@exercise("AalenAdditiveModel")
def _aalen():
    am = mod.AalenAdditiveModel().fit(T, E, np.column_stack([np.ones(_N), _X]))
    h0 = float(am.predict([1.0, 0.0], [2.0])[0])
    h1 = float(am.predict([1.0, 1.0], [2.0])[0])
    assert h0 > h1


@exercise("FineGrayModel")
def _fine_gray():
    fg = mod.FineGrayModel().fit(T, CAUSE, _X[:, None], cause_of_interest=1)
    assert np.isfinite(fg.coefficients_[0]) and fg.standard_errors_[0] > 0


def _two_sample(cls):
    r = cls().fit(T, E, G)
    assert r.p_value_ < 0.01 and np.isfinite(r.test_statistic_)


@exercise("LogRankTest")
def _logrank():
    _two_sample(mod.LogRankTest)


@exercise("WilcoxonSurvival")
def _wilcoxon():
    _two_sample(mod.WilcoxonSurvival)


@exercise("TaroneWareTest")
def _tarone():
    _two_sample(mod.TaroneWareTest)


@exercise("PetoTest")
def _peto():
    _two_sample(mod.PetoTest)


@exercise("FlemingHarrington")
def _fleming():
    r = mod.FlemingHarrington(rho=1.0, gamma=1.0).fit(T, E, G)
    assert 0 <= r.p_value_ <= 1


@exercise("CauseSpecificHazard")
def _csh():
    csh = mod.CauseSpecificHazard().fit(T, CAUSE, cause_of_interest=2)
    h = np.asarray(csh.predict(_TIMES), dtype=float)
    assert np.all(h >= 0) and np.all(np.diff(h) >= -1e-12)


@exercise("CumulativeIncidenceFunction")
def _cif():
    cif = mod.CumulativeIncidenceFunction().fit(T, CAUSE, cause_of_interest=1)
    f = np.asarray(cif.predict(_TIMES), dtype=float)
    assert np.all((f >= 0) & (f <= 1)) and np.all(np.diff(f) >= -1e-12)


@exercise("CompetingRisksModel")
def _crm():
    crm = mod.CompetingRisksModel().fit(T, CAUSE)
    assert crm.check_identity() < 1e-8 and set(crm.causes_) == {1, 2}


def test_every_public_name_is_exercised():
    assert set(EXERCISES) == set(mod.__all__), (
        set(mod.__all__) - set(EXERCISES), set(EXERCISES) - set(mod.__all__))


@pytest.mark.parametrize("name", sorted(mod.__all__))
def test_exercise(name):
    EXERCISES[name]()
