"""Library-level conformance and cross-module integration tests.

This suite is intentionally NOT one package module: it pins the public
surface of every implemented module against the design spec (so names can
never silently disappear again — see development/Probleme.md [21] for the
motivating incident) and exercises end-to-end workflows that cross module
boundaries.

The spec-name lists are generated from
development/Implementation-Checklist.md (the project's own tracking source)
via tests/library/_spec_names.json; regenerate with
tests/library/_extract_spec_names.py at the repo root if the checklist
changes.
"""

import json
import os

import numpy as np
import pytest
from scipy import stats

import stochpylib
from stochpylib import (
    copulas,
    distributions,
    gaussian_processes,
    montecarlo,
    probability,
    survival,
    timeseries,
)

_SPEC = json.load(open(os.path.join(os.path.dirname(__file__),
                                    "_spec_names.json"), encoding="utf-8"))

_MODULES = {
    "probability": probability,
    "distributions": distributions,
    "montecarlo": montecarlo,
    "timeseries": timeseries,
    "gaussian_processes": gaussian_processes,
    "copulas": copulas,
    "survival": survival,
}

# Documented public extras beyond the 229 spec names (utilities & result
# objects introduced during implementation). Pinned so they cannot silently
# disappear either.
_EXTRAS = {
    "distributions": {"Distribution", "MultivariateDistribution"},
    "montecarlo": {"MCResult", "DigitalNetBase2"},
    "timeseries": {"ForecastResult", "TestResult", "ChangePointResult",
                   "BOCPDResult"},
    "gaussian_processes": {"BaseKernel", "StationaryKernel",
                           "NonStationaryKernel", "StationaryKernelOp",
                           "NonStationaryKernelOp", "cholesky_with_jitter"},
    "copulas": {"BaseCopula"},
    "survival": set(),
    "probability": set(),
}

DISTRIBUTION_METHODS_DOC = (".pdf()/.pmf()", ".cdf()", ".ppf()", ".rvs()",
                            ".mean()", ".var()", ".skewness()", ".kurtosis()",
                            ".entropy()", ".mgf()", ".cf()", ".fit()",
                            ".ks_test()")
_METHOD_NAMES = ("pdf", "cdf", "ppf", "rvs", "mean", "var", "skewness",
                 "kurtosis", "entropy", "mgf", "cf", "fit", "ks_test")


# ---------------------------------------------------------------- conformance

@pytest.mark.parametrize("name", sorted(_MODULES))
def test_spec_names_present(name):
    mod = _MODULES[name]
    missing = [n for n in _SPEC[name] if not hasattr(mod, n)]
    assert not missing, f"{name}: spec names missing from exports: {missing}"


@pytest.mark.parametrize("name", sorted(_EXTRAS))
def test_documented_extras_present(name):
    mod = _MODULES[name]
    missing = [n for n in _EXTRAS[name] if not hasattr(mod, n)]
    assert not missing, f"{name}: documented extras missing: {missing}"


def test_total_spec_name_count_is_257():
    implemented = ("probability", "montecarlo", "timeseries",
                   "gaussian_processes", "copulas", "survival")
    total = sum(len(_SPEC[k]) for k in implemented) + 60  # +60 distributions
    assert total == 257


# Multivariate distributions legitimately deviate from the scalar-method
# contract: they expose pdf (not pmf) and cannot offer scalar-argument
# mgf/cf (the multinomial/wishart transforms are vector-valued).
_MULTIVARIATE = {"Multinomial", "Dirichlet", "InverseWishart",
                 "MultivariateNormal", "MultivariatePareto", "MultivariateT",
                 "Wishart"}


def test_every_distribution_class_exposes_common_interface():
    classes = [n for n in distributions.__all__
               if n not in ("Distribution", "MultivariateDistribution")]
    assert len(classes) == 47
    for name in classes:
        cls = getattr(distributions, name)
        required = set(_METHOD_NAMES) | {"pmf"}
        if name in _MULTIVARIATE:
            # multivariate: pdf instead of pmf; scalar-argument mgf/cf are
            # mathematically inapplicable (vector-valued transforms) — the
            # documented contract deviation for these 7 classes
            required -= {"pmf", "mgf", "cf"}
            required |= {"pdf"}
        for meth in sorted(required):
            assert hasattr(cls, meth), f"{name}.{meth} missing"


def test_top_level_package_wiring():
    assert set(stochpylib.__all__) == {
        "copulas", "distributions", "gaussian_processes", "montecarlo",
        "probability", "queueing", "survival", "timeseries"}
    assert stochpylib.__version__ == "0.5.1"


# ---------------------------------------------------------------- integration

def test_montecarlo_reliability_driven_by_library_weibull():
    """E2E: reliability_mc consumes library distribution objects and matches
    the closed-form failure probability of a stress-strength problem."""
    from stochpylib.distributions import Weibull
    # X ~ Weibull(k=2, scale=10): P(X <= 5) = 1 - exp(-(5/10)^2) ~= 0.2212
    p_true = 1 - np.exp(-0.25)
    res = montecarlo.reliability_mc(
        lambda X: X[:, 0], [Weibull(2.0, 10.0)], threshold=5.0, n=100_000,
        random_state=7)
    assert abs(res.estimate - p_true) < 4 * np.sqrt(
        p_true * (1 - p_true) / 100_000)


def test_t_copula_margins_follow_library_student_t():
    """E2E: copulas -> distributions. Mapping the t-copula sample through the
    library Student_t quantile function must produce t(4)-distributed margins
    (KS against scipy.stats.t as oracle), i.e. dependence + margins compose."""
    from stochpylib.copulas.elliptical import StudentTCopula
    rng = np.random.default_rng(31)
    R = np.array([[1.0, .5], [.5, 1.0]])
    z = rng.standard_normal((4000, 2)) @ np.linalg.cholesky(R).T
    w = rng.chisquare(4, 4000)
    t_draws = z * np.sqrt(4 / w)[:, None]
    data = stats.t.cdf(t_draws, 4)              # copula-scale (uniform)

    fitted = StudentTCopula().fit(data)
    assert 3.0 < fitted.df_ < 6.5
    lib = distributions.Student_t(4)
    x_t = np.asarray(lib.ppf(data[:, 0]), dtype=float)
    ks = stats.kstest(x_t, lambda q: stats.t.cdf(q, 4)).statistic
    assert ks < 0.05


def test_arima_and_gp_forecasts_agree_on_smooth_series():
    """E2E: timeseries <-> gaussian_processes. On a smooth low-noise series
    both forecasters must track the truth at short horizons (AR(2) and an RBF
    GP both revert to the mean beyond ~one period, so 10 steps is the fair
    horizon)."""
    rng = np.random.default_rng(91)
    t = np.arange(300)
    y = np.sin(2 * np.pi * t / 50) + 0.03 * rng.standard_normal(300)
    arima_fc = timeseries.ARIMA(2, 0, 0).fit(y[:260]).forecast(horizon=20)
    gp_fc = gaussian_processes.GPTimeSeriesModel(
        length_scale=12.0, noise=0.05).fit(y[:260]).forecast(horizon=20)
    truth = y[260:280]
    err_a = float(np.max(np.abs(np.asarray(arima_fc.mean)[:10] - truth[:10])))
    err_g = float(np.max(np.abs(np.asarray(gp_fc.mean)[:10] - truth[:10])))
    assert err_a < 0.35, err_a
    assert err_g < 0.25, err_g


def test_copulafit_sample_refit_round_trip():
    """E2E: copulas self-contained fit -> sample -> refit consistency."""
    data = copulas.ClaytonCopula(theta=3.0).sample(2500, random_state=5)
    fit1 = copulas.CopulaFit(families=("clayton", "gumbel", "gaussian")).fit(
        data)
    assert fit1.best_name_ == "clayton"
    s = fit1.best_.sample(8000, random_state=6)
    fit2 = copulas.ClaytonCopula().fit(s)
    tau_model = fit1.best_.kendall_tau()
    tau_refit = copulas.ClaytonCopula(theta=fit2.theta_).kendall_tau()
    assert abs(tau_model - tau_refit) < 0.04


def test_qmc_integral_agrees_between_apis():
    """E2E: montecarlo internal consistency — Sobol-driven QMC integral
    matches the crude estimator and dense quadrature on a smooth integrand."""
    f = lambda pts: (np.sin(2 * np.pi * pts[:, 0]) *
                     np.cos(np.pi * pts[:, 1]))
    qmc_res = montecarlo.MonteCarloIntegration(
        f, bounds=[(0, 1), (0, 1)], method="qmc", sequence="sobol",
        random_state=3).estimate(n=16384)
    qmc = float(qmc_res.estimate)
    crude = montecarlo.crude_mc(f, n=16384, dim=2, random_state=4)
    # ground truth by dense quadrature
    g = np.linspace(0, 1, 801)
    uu, vv = np.meshgrid(g, g)
    exact = float(np.trapezoid(
        np.trapezoid(f(np.column_stack([uu.ravel(), vv.ravel()])
                       ).reshape(801, 801), g, axis=1), g))
    assert abs(qmc - exact) < 5e-3
    assert abs(crude.estimate - exact) < 4 * crude.std_error


def test_survival_uses_library_weibull_for_parametric_fit():
    """E2E: survival -> distributions. WeibullSurvival recovers parameters
    from data generated by the library's own Weibull distribution."""
    from stochpylib.distributions import Weibull as LibWeibull
    rng = np.random.default_rng(50)
    dist = LibWeibull(1.5, 10.0)
    t_potential = np.asarray(dist.rvs(3000, random_state=51), dtype=float)
    c = rng.uniform(2, 30, 3000)
    dur = np.minimum(t_potential, c)
    ev = (t_potential <= c).astype(int)
    ws = survival.WeibullSurvival().fit(dur, ev)
    assert abs(ws.params_["shape"] - 1.5) < .15
    assert abs(ws.params_["scale"] - 10) < 1.2


def test_montecarlo_reliability_with_survival_km_cross_check():
    """E2E: montecarlo <-> survival. reliability MC failure probability is
    consistent with the Kaplan-Meier estimate at the same threshold."""
    rng = np.random.default_rng(52)
    t_true = rng.exponential(2, 5000)
    c = rng.uniform(.2, 8, 5000)
    dur = np.minimum(t_true, c)
    ev = (t_true <= c).astype(int)

    km = survival.KaplanMeier().fit(dur, ev)
    km_s_at_2 = float(km.predict([2.0])[0])

    from stochpylib.distributions import Exponential as DExp
    res = montecarlo.reliability_mc(
        lambda X: X[:, 0], [DExp(0.5)], threshold=2.0, n=100_000,
        random_state=53)
    mc_fail = res.estimate

    # KM S(2) = P(T>2), so failure prob P(T<=2)=1-S(2); MC uses uncensored
    # exp(0.5) draws so they should be consistent within MC noise
    assert abs((1 - km_s_at_2) - mc_fail) < .06
