"""End-to-end API sweep for stochpylib.nonparametric: one realistic exercise
per public name. ``test_every_public_name_is_exercised`` fails the moment a
name ships without one; ``test_exercise[<name>]`` runs each exercise as its
own pytest case."""

import numpy as np
import pytest

from stochpylib import nonparametric as mod

EXERCISES = {}


def exercise(name):
    def deco(fn):
        EXERCISES[name] = fn
        return fn
    return deco


_RNG = np.random.default_rng(0)
_X = _RNG.normal(0, 1, 200)
_T = np.sort(_RNG.uniform(0, 10, 100))
_Z = np.sin(_T) + _RNG.normal(0, 0.2, 100)
_A = _RNG.normal(0, 1, 80)
_B = _RNG.normal(0.4, 1, 80)


def _finite(a):
    return bool(np.all(np.isfinite(np.asarray(a, dtype=float))))


# --------------------------------------------------------------- base classes

@exercise("NonparametricDensity")
def _nonparametric_density():
    kde = mod.KernelDensityEstimate().fit(_X)
    assert isinstance(kde, mod.NonparametricDensity)
    assert _finite(kde.pdf(0.0))


@exercise("NonparametricTest")
def _nonparametric_test():
    t = mod.SignTest().fit(_X)
    assert isinstance(t, mod.NonparametricTest)
    assert t.to_result() is t.result_


@exercise("DependenceMeasure")
def _dependence_measure():
    d = mod.SpearmanCorrelation().fit(_A, _B)
    assert isinstance(d, mod.DependenceMeasure)
    assert isinstance(float(d), float)


@exercise("NonparametricRegressor")
def _nonparametric_regressor():
    r = mod.IsotonicRegression().fit(np.arange(30), np.sort(_RNG.normal(size=30)))
    assert isinstance(r, mod.NonparametricRegressor)
    assert _finite(r.predict(np.array([5.0, 15.0])))


# --------------------------------------------------------------------- density

@exercise("KernelDensityEstimate")
def _kde():
    k = mod.KernelDensityEstimate(bandwidth="silverman").fit(_X)
    assert _finite(k.pdf(0.0)) and _finite(k.cdf(0.0))


@exercise("AdaptiveKDE")
def _adaptive_kde():
    k = mod.AdaptiveKDE(alpha=0.5).fit(_X)
    assert _finite(k.pdf(0.0))


@exercise("NearestNeighborDensity")
def _nn_density():
    k = mod.NearestNeighborDensity(k=10).fit(_X)
    assert _finite(k.pdf(0.0))


@exercise("OrthogonalSeriesDensity")
def _osd():
    k = mod.OrthogonalSeriesDensity(n_terms=10).fit(_X)
    assert _finite(k.pdf(0.0))


@exercise("LogsplineEstimator")
def _logspline():
    k = mod.LogsplineEstimator(n_knots=6).fit(_X)
    assert _finite(k.pdf(0.0))


# ------------------------------------------------------------------- empirical

@exercise("EmpiricalDistribution")
def _empirical_distribution():
    e = mod.EmpiricalDistribution().fit(_X)
    assert 0 <= e.cdf(0.0) <= 1


@exercise("EmpiricalCDF")
def _empirical_cdf():
    e = mod.EmpiricalCDF().fit(_X)
    assert 0 <= e.evaluate(0.0) <= 1


@exercise("EmpiricalCharFn")
def _empirical_charfn():
    e = mod.EmpiricalCharFn().fit(_X)
    assert _finite(abs(e.evaluate(0.5)))


@exercise("GlivenkoCantelli")
def _glivenko_cantelli():
    from stochpylib.distributions import Normal
    g = mod.GlivenkoCantelli().fit(_X, Normal(0, 1))
    assert g.distance_ >= 0


@exercise("EmpiricalLikelihood")
def _empirical_likelihood():
    e = mod.EmpiricalLikelihood().fit(_X)
    res = e.test_mean(0.0)
    assert res.pvalue is not None


# ---------------------------------------------------------------------- tests

@exercise("PermutationTest")
def _permutation_test():
    t = mod.PermutationTest(n_resamples=500, random_state=0).fit(_A, _B)
    assert t.pvalue_ is not None


@exercise("BootstrapTest")
def _bootstrap_test():
    t = mod.BootstrapTest(n_boot=500, random_state=0).fit(_A, _B)
    assert t.pvalue_ is not None


@exercise("MoodTest")
def _mood_test():
    t = mod.MoodTest().fit(_A, _B)
    assert t.pvalue_ is not None


@exercise("KruskalWallis")
def _kruskal_wallis():
    t = mod.KruskalWallis().fit(_A, _B, _RNG.normal(1, 1, 80))
    assert t.pvalue_ is not None


@exercise("FriedmanTest")
def _friedman_test():
    data = _RNG.normal(size=(20, 3))
    t = mod.FriedmanTest().fit(data[:, 0], data[:, 1], data[:, 2])
    assert t.pvalue_ is not None


@exercise("SignTest")
def _sign_test():
    t = mod.SignTest().fit(_A)
    assert t.pvalue_ is not None


@exercise("RunsTest")
def _runs_test():
    t = mod.RunsTest().fit(_A)
    assert t.pvalue_ is not None


@exercise("WaldWolfowitz")
def _wald_wolfowitz():
    t = mod.WaldWolfowitz().fit(_A, _B)
    assert t.pvalue_ is not None


@exercise("AndersenDarling")
def _andersen_darling():
    t = mod.AndersenDarling(dist="norm").fit(_A)
    assert t.pvalue_ is not None


@exercise("AndersonDarling")
def _anderson_darling_alias():
    assert mod.AndersonDarling is mod.AndersenDarling


@exercise("CramerVonMises")
def _cramer_von_mises():
    t = mod.CramerVonMises().fit(_A, _B)
    assert t.pvalue_ is not None


# ----------------------------------------------------------------- correlation

@exercise("SpearmanCorrelation")
def _spearman():
    d = mod.SpearmanCorrelation().fit(_A, _B)
    assert -1 <= d.estimate_ <= 1


@exercise("KendallTau")
def _kendall():
    d = mod.KendallTau().fit(_A, _B)
    assert -1 <= d.estimate_ <= 1


@exercise("RankCorrelation")
def _rank_correlation():
    d = mod.RankCorrelation(method="kendall").fit(_A, _B)
    assert -1 <= d.estimate_ <= 1


@exercise("DistanceCorrelation")
def _distance_correlation():
    d = mod.DistanceCorrelation(n_resamples=50, random_state=0).fit(_A, _B)
    assert 0 <= d.estimate_ <= 1


@exercise("BrownianCorrelation")
def _brownian_correlation():
    d = mod.BrownianCorrelation(n_resamples=50, random_state=0).fit(_A, _B)
    assert 0 <= d.estimate_ <= 1


@exercise("HoeffdingD")
def _hoeffding_d():
    d = mod.HoeffdingD(n_resamples=50, random_state=0).fit(_A, _B)
    assert _finite(d.estimate_)


# ----------------------------------------------------------------- regression

@exercise("LocalPolynomialReg")
def _local_polynomial():
    r = mod.LocalPolynomialReg(bandwidth=0.8).fit(_T, _Z)
    assert _finite(r.predict(np.array([5.0])))


@exercise("IsotonicRegression")
def _isotonic():
    y = np.sort(_RNG.normal(size=40))
    r = mod.IsotonicRegression().fit(np.arange(40), y)
    assert np.all(np.diff(r.fitted_) >= -1e-10)


@exercise("SplineRegression")
def _spline_regression():
    r = mod.SplineRegression(method="pspline", n_knots=8).fit(_T, _Z)
    assert _finite(r.predict(np.array([5.0])))


@exercise("GPR_Nonparametric")
def _gpr_nonparametric():
    r = mod.GPR_Nonparametric(optimize=False).fit(_T, _Z)
    mu, std = r.predict(np.array([5.0])[:, None], return_std=True)
    assert _finite(mu) and _finite(std)


@exercise("QuantileRegression")
def _quantile_regression():
    r = mod.QuantileRegression(q=0.5, bandwidth=2.0).fit(_T, _Z)
    assert _finite(r.predict(np.array([5.0])))


def test_every_public_name_is_exercised():
    assert set(EXERCISES) == set(mod.__all__), (
        set(mod.__all__) - set(EXERCISES), set(EXERCISES) - set(mod.__all__))


@pytest.mark.parametrize("name", sorted(mod.__all__))
def test_exercise(name):
    EXERCISES[name]()
