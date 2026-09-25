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
    advanced_mcmc,
    bayesian,
    copulas,
    distributions,
    experimental_design,
    financial_stochastics,
    gaussian_processes,
    information_theory,
    levy_processes,
    montecarlo,
    nonparametric,
    numerical_methods,
    optimization,
    probability,
    queueing,
    random_matrix,
    robust_statistics,
    statistics,
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
    "levy_processes": levy_processes,
    "financial_stochastics": financial_stochastics,
    "statistics": statistics,
    "random_matrix": random_matrix,
    "queueing": queueing,
    "information_theory": information_theory,
    "advanced_mcmc": advanced_mcmc,
    "numerical_methods": numerical_methods,
    "bayesian": bayesian,
    "robust_statistics": robust_statistics,
    "nonparametric": nonparametric,
    "optimization": optimization,
    "experimental_design": experimental_design,
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
    "survival": {"SurvivalFitter"},
    "queueing": {"BaseQueue", "QueueResult", "erlang_b_formula", "erlang_c_formula",
                 "engset_formula"},
    "information_theory": set(),
    "probability": set(),
    "levy_processes": set(),
    "financial_stochastics": set(),
    "statistics": {"TestResult", "EstimateResult", "DescribeResult", "RegressionResult",
                   "PCAResult", "FactorResult", "CanonicalCorrelationResult",
                   "DiscriminantResult", "ClusterResult", "MDSResult"},
    "random_matrix": {"MatrixEnsemble", "CircularLaw"},
    "advanced_mcmc": {"LogDensity", "MCMCSampler", "RafteryLewisResult"},
    "numerical_methods": {"QuadratureResult", "RootResult", "ODESolution",
                          "SDESolution", "Mesh"},
    "bayesian": {"Prior", "Likelihood", "ConjugateFamily", "Posterior",
                "PosteriorApproximation", "ICResult", "EmpiricalPredictive"},
    "robust_statistics": {"RobustEstimator", "RobustRegressor", "RobustCovarianceEstimator",
                          "Resampler", "SiegelRegression"},
    "nonparametric": {"NonparametricDensity", "NonparametricTest", "DependenceMeasure",
                      "NonparametricRegressor", "AndersonDarling"},
    "optimization": {"Objective", "Optimizer", "PopulationOptimizer",
                     "ConstrainedOptimizer", "OptimizeResult"},
    "experimental_design": {"Design", "DesignGenerator", "OptimalDesign"},
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


def test_total_spec_name_count():
    implemented = ("probability", "montecarlo", "timeseries",
                   "gaussian_processes", "copulas", "survival", "queueing",
                   "information_theory", "levy_processes",
                   "financial_stochastics", "statistics", "random_matrix",
                   "advanced_mcmc", "numerical_methods", "bayesian",
                   "robust_statistics", "nonparametric", "optimization",
                   "experimental_design")
    total = sum(len(_SPEC[k]) for k in implemented) + 60  # +60 distributions
    assert total == 689  # 689/794 across the twenty implemented modules


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
        "advanced_mcmc", "bayesian", "copulas", "distributions", "experimental_design",
        "financial_stochastics", "gaussian_processes", "information_theory", "levy_processes",
        "montecarlo", "nonparametric", "numerical_methods", "optimization",
        "probability", "queueing", "random_matrix", "robust_statistics", "statistics",
        "survival", "timeseries"}
    # version consistency, never a literal: the installed metadata and the
    # in-code __version__ must agree (a hardcoded literal here broke CI on
    # every version bump — development/Probleme.md [39])
    from importlib.metadata import version, PackageNotFoundError
    try:
        assert version("stochpylib") == stochpylib.__version__
    except PackageNotFoundError:  # running from source without install
        pass


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


def test_random_matrix_wishart_spectrum_flows_into_statistics_test_result():
    """E2E: random_matrix -> distributions -> statistics. A Wishart ensemble sampled
    through the library Wishart distribution has a Marchenko-Pastur bulk, and the
    comparison comes back as the shared statistics.TestResult object."""
    from stochpylib.statistics import TestResult
    W = random_matrix.WishartMatrix(p=150, n=600)
    res = W.limit_law().compare(W.normalized_eigenvalues(random_state=21))
    assert isinstance(res, TestResult) and not res.reject(0.01)
    assert res.extras["n"] == 150


def test_random_matrix_wigner_universality_with_library_entry_distribution():
    """E2E: random_matrix <- distributions. A Wigner matrix whose entries are drawn from
    the library Exponential distribution still has a semicircular bulk, and the limit
    law behaves as a full library distribution (ks_test contract)."""
    from stochpylib.distributions import Exponential
    W = random_matrix.WignerMatrix(600, entries=Exponential(2.0))
    e = W.normalized_eigenvalues(random_state=22)
    d, p = random_matrix.WignerSemicircle(2.0).ks_test(e)
    assert d < 0.04 and p > 0.01


def test_mcmc_samples_a_library_distribution_and_diagnostics_return_test_result():
    """E2E: advanced_mcmc -> distributions/statistics. A slice sampler driven by a
    library Gamma log-density reproduces its mean, and geweke_test returns a shared
    statistics.TestResult."""
    from stochpylib.advanced_mcmc import LogDensity, SliceSampling, geweke_test
    from stochpylib.distributions import Gamma
    from stochpylib.statistics import TestResult

    g = Gamma(3.0, 2.0)
    target = LogDensity.from_distribution(g)
    s = SliceSampling(target, n_samples=3000, n_warmup=500)
    s.sample(np.array([g.mean()]), random_state=5)
    x = s.get_samples()[:, 0]
    se = g.std() / np.sqrt(advanced_mcmc.ESS(x[None, :, None], method="mean"))
    assert abs(x.mean() - g.mean()) < 5 * se
    assert isinstance(geweke_test(x), TestResult)


def test_numerical_methods_pde_and_quadrature_agree_with_black_scholes():
    """E2E: numerical_methods -> financial_stochastics. A Crank-Nicolson finite-
    difference solve of the Black-Scholes PDE and a Gauss-Hermite risk-neutral
    expectation of the discounted payoff both match the closed-form price."""
    S, K, T, r, sigma = 100.0, 100.0, 1.0, 0.05, 0.2
    bs = financial_stochastics.BlackScholes(S=S, K=K, T=T, r=r, sigma=sigma)
    fd = numerical_methods.FiniteDifference()
    pde_price = fd.black_scholes(K, T, r, sigma, kind="call").price(S)
    assert abs(pde_price - bs.call_price()) < 1e-2

    gh = numerical_methods.GaussHermite(300, kind="probabilists")

    def discounted_payoff(z):
        ST = S * np.exp((r - 0.5 * sigma ** 2) * T + sigma * np.sqrt(T) * z)
        return np.maximum(ST - K, 0.0) * np.exp(-r * T)

    gh_price = gh.expectation(discounted_payoff, vectorized=True).value
    assert abs(gh_price - bs.call_price()) < 1e-2


def test_robust_statistics_agrees_with_statistics_copulas_and_distributions():
    """E2E: robust_statistics -> statistics/copulas/distributions. Huber regression on
    clean library-Normal errors agrees with statistics.linear_regression; MM regression
    stays close to the truth under y-outliers where OLS does not; robust_statistics'
    Kendall correlation matches copulas.kendall_tau exactly; MCD recovers a library
    MultivariateNormal's covariance; and RobustBootstrap returns the shared
    statistics.EstimateResult."""
    rng = np.random.default_rng(300)
    x = rng.standard_normal(200)
    err = np.asarray(distributions.Normal(0, 0.3).rvs(200, random_state=1), dtype=float)
    y = 1.0 + 2.0 * x + err
    hr = robust_statistics.HuberRegression().fit(x, y)
    ols = statistics.linear_regression(x, y)
    assert np.max(np.abs(hr.coef_ - ols.coef_)) < 3 * np.max(ols.std_errors_)

    y_out = y.copy()
    y_out[:60] += 15.0
    mm = robust_statistics.MMRegression(random_state=0).fit(x, y_out)
    ols_out = statistics.linear_regression(x, y_out)
    assert abs(mm.coef_[1] - 2.0) < 0.2
    assert abs(ols_out.coef_[1] - 2.0) > 0.3

    U = rng.uniform(size=(400, 2))
    rc = robust_statistics.RobustCorrelation(method="kendall").fit(U)
    assert rc.correlation_[0, 1] == pytest.approx(float(copulas.kendall_tau(U)), abs=1e-12)

    mu = np.array([1.0, -2.0])
    Sigma = np.array([[2.0, 0.6], [0.6, 1.0]])
    mvn = distributions.MultivariateNormal(mu, Sigma)
    X = np.asarray(mvn.rvs(1000, random_state=2), dtype=float)
    mcd = robust_statistics.MCD(random_state=0).fit(X)
    assert np.linalg.norm(mcd.covariance_ - Sigma) / np.linalg.norm(Sigma) < 0.15

    boot = robust_statistics.RobustBootstrap("median", n_boot=500, random_state=0).fit(x)
    from stochpylib.statistics import EstimateResult
    assert isinstance(boot.to_result(), EstimateResult)


def test_bayesian_conjugate_agrees_with_statistics_estimator_and_mcmc():
    """E2E: bayesian -> statistics/advanced_mcmc. The conjugate beta-bernoulli posterior
    exactly matches statistics.bayesian_estimator's own conjugate path, and an MCMC
    posterior (delegated to advanced_mcmc) recovers the same mean within its MCSE."""
    rng = np.random.default_rng(200)
    x = rng.binomial(1, 0.4, 300)
    post = bayesian.posterior(bayesian.prior(distributions.Beta(1, 1)),
                               bayesian.likelihood("bernoulli", data=x), method="conjugate")
    est = statistics.bayesian_estimator("bernoulli", x, prior=(1.0, 1.0))
    assert post.dist.a == pytest.approx(est.extras["posterior"].a)
    assert post.dist.b == pytest.approx(est.extras["posterior"].b)

    pm = bayesian.posterior(bayesian.prior(distributions.Beta(1, 1)),
                             bayesian.likelihood("bernoulli", data=x), method="mcmc",
                             sampler="slice", n_samples=2000, n_warmup=500,
                             theta0=np.array([0.4]), random_state=1)
    from stochpylib.advanced_mcmc import ESS, MCMCSampler
    assert isinstance(pm.extras["sampler"], MCMCSampler)
    ess = float(ESS(pm.samples_.T[:, :, None], method="mean"))
    se = pm.samples_.std() / np.sqrt(ess)
    assert abs(pm.samples_.mean() - post.mean()[0]) < 5 * se


def test_nonparametric_agrees_with_statistics_copulas_and_gaussian_processes():
    """E2E: nonparametric -> statistics/copulas/gaussian_processes. KendallTau matches
    copulas.kendall_tau exactly; KruskalWallis returns the shared statistics.TestResult;
    GPR_Nonparametric (optimize=False) reproduces gaussian_processes.GPRegression exactly
    on the same kernel; KernelDensityEstimate's ks_test recovers a library Normal sample."""
    rng = np.random.default_rng(400)
    x = rng.standard_normal(150)
    y = x + rng.normal(0, 0.5, 150)
    kt = nonparametric.KendallTau().fit(x, y)
    assert kt.estimate_ == pytest.approx(float(copulas.kendall_tau(np.column_stack([x, y]))),
                                          abs=1e-10)

    from stochpylib.statistics import TestResult
    g1, g2, g3 = rng.normal(0, 1, 30), rng.normal(0, 1, 30), rng.normal(2, 1, 30)
    kw = nonparametric.KruskalWallis().fit(g1, g2, g3)
    assert isinstance(kw.to_result(), TestResult)

    from stochpylib.gaussian_processes import GPRegression
    from stochpylib.gaussian_processes.kernels import RBFKernel, WhiteNoiseKernel
    t = np.sort(rng.uniform(0, 10, 60))
    z = np.sin(t) + rng.normal(0, 0.1, 60)
    kernel = RBFKernel(length_scale=float(np.std(t)), variance=float(np.var(z))) + \
        WhiteNoiseKernel(0.1)
    gpr_np = nonparametric.GPR_Nonparametric(kernel=kernel, noise=0.1, optimize=False).fit(t, z)
    gpr_ref = GPRegression(kernel=kernel, noise=0.1).fit(t[:, None], z)
    q = np.array([2.0, 5.0, 8.0])[:, None]
    assert np.allclose(gpr_np.predict(q), gpr_ref.predict(q)[0])

    x_lib = np.asarray(distributions.Normal(3, 1).rvs(500, random_state=1), dtype=float)
    kde = nonparametric.KernelDensityEstimate().fit(x_lib)
    d, p = kde.ks_test(x_lib)
    assert p > 0.01


def test_optimization_agrees_with_statistics_distributions_and_gaussian_processes():
    """E2E: optimization -> statistics/distributions/gaussian_processes/montecarlo. BFGS
    maximizes the logistic likelihood to statistics.logistic_regression's coefficients and
    its inverse-Hessian reproduces that fit's standard errors; NewtonMethod reproduces
    distributions.Gamma.fit; BayesianOptimization drives a gaussian_processes.GPRegression
    surrogate; SAA reports its optimality gap as a montecarlo.MCResult."""
    from scipy import special

    rng = np.random.default_rng(410)
    n, p = 400, 3
    X = rng.normal(size=(n, p))
    Xd = np.column_stack([np.ones(n), X])
    y = (rng.random(n) < 1.0 / (1.0 + np.exp(-Xd @ np.array([0.4, -1.0, 0.7, 0.2])))
         ).astype(float)
    nll = lambda b: float(np.sum(np.logaddexp(0.0, Xd @ b) - y * (Xd @ b)))
    nll_grad = lambda b: Xd.T @ (1.0 / (1.0 + np.exp(-Xd @ b)) - y)
    fit = optimization.BFGS().minimize(nll, np.zeros(p + 1), grad=nll_grad)
    ref = statistics.logistic_regression(X, y)
    assert np.max(np.abs(fit.x_ - ref.coef_)) < 1e-5
    se = np.sqrt(np.diag(fit.result_.hess_inv))
    assert np.max(np.abs(se - ref.std_errors_) / ref.std_errors_) < 0.05

    data = np.asarray(distributions.Gamma(3.0, 2.0).rvs(1500, random_state=41), dtype=float)
    gamma_nll = lambda th: float(-np.sum(
        (np.exp(th[0]) - 1) * np.log(data) - data / np.exp(th[1])
        - special.gammaln(np.exp(th[0])) - np.exp(th[0]) * th[1]))
    mle = optimization.NewtonMethod().minimize(gamma_nll, np.log([1.0, 1.0]))
    fitted = distributions.Gamma(1.0, 1.0).fit(data)
    assert float(np.exp(mle.x_[0])) == pytest.approx(fitted.shape, rel=1e-5)
    assert float(np.exp(mle.x_[1])) == pytest.approx(fitted.scale, rel=1e-5)

    bo = optimization.BayesianOptimization(n_init=6, n_iter=8, n_candidates=80,
                                           bounds=[(-2.0, 2.0)] * 2,
                                           random_state=0).minimize(
        lambda z: float(np.sum(z ** 2)), [1.5, 1.5])
    assert bo.result_.extras["X_observed"].shape == (14, 2)
    assert bo.fun_ <= float(np.sum(np.array([1.5, 1.5]) ** 2))

    cost = lambda q, d: float((q[0] - d) ** 2)
    saa = optimization.SAA(n_samples=2000, n_batches=4, batch_size=200,
                           random_state=0).minimize(
        cost, [0.0], sampler=lambda size, r: r.normal(4.0, 1.0, size=size))
    assert isinstance(saa.gap_, montecarlo.MCResult)
    lo, hi = saa.gap_.confidence_interval()
    assert lo <= float(saa.gap_) <= hi
    assert abs(float(saa.x_[0]) - 4.0) < 4 / np.sqrt(2000)


def test_experimental_design_agrees_with_statistics_gaussian_processes_montecarlo_and_optimization():
    """E2E: experimental_design -> statistics/gaussian_processes/montecarlo/optimization.
    ResponseSurface is statistics.linear_regression on the model matrix; KrigingSurrogate
    without a trend is gaussian_processes.GPRegression; LatinHypercubeDesign is
    montecarlo.LatinHypercubeSampling; Sobol indices are montecarlo.MCResults;
    ResponseSurface.optimize runs optimization.DifferentialEvolution to the grid optimum;
    and the DOE ANOVA of a balanced two-way layout is statistics.ANOVA's table."""
    rng = np.random.default_rng(420)
    ccd = experimental_design.CCD(2, center=5).generate()
    X = ccd.points
    y = 5 + X[:, 0] - 2 * X[:, 1] - X[:, 0] ** 2 - 0.5 * X[:, 1] ** 2 + rng.normal(0, 0.1,
                                                                                   len(X))
    rs = experimental_design.ResponseSurface(2).fit(X, y)
    F, _ = ccd.model_matrix("quadratic")
    ref = statistics.linear_regression(F[:, 1:], y)
    assert np.allclose(rs.coef_, ref.coef_, atol=1e-12)
    assert np.allclose(rs.regression_.std_errors_, ref.std_errors_)

    best = rs.optimize(bounds=[(-1, 1), (-1, 1)], maximize=True, random_state=0)
    g1, g2 = np.meshgrid(np.linspace(-1, 1, 401), np.linspace(-1, 1, 401))
    grid = np.column_stack([g1.ravel(), g2.ravel()])
    assert best["value"] == pytest.approx(float(np.max(rs.predict(grid))), abs=1e-4)

    Xk = experimental_design.LatinHypercubeDesign(15, 2, random_state=3).generate().points
    lhs = montecarlo.LatinHypercubeSampling(dim=2, n=15,
                                            random_state=np.random.default_rng(3)).generate()
    assert np.array_equal(Xk, lhs)
    yk = np.sin(4 * Xk[:, 0]) + Xk[:, 1]
    kern = gaussian_processes.MaternKernel(nu=2.5, length_scale=0.4)
    kr = experimental_design.KrigingSurrogate(kernel=kern, trend="none", optimize=False,
                                              normalize=False, noise=1e-6).fit(Xk, yk)
    gp = gaussian_processes.GPRegression(gaussian_processes.MaternKernel(nu=2.5,
                                                                         length_scale=0.4),
                                         noise=1e-6).fit(Xk, yk)
    Xt = rng.random((30, 2))
    assert np.allclose(kr.predict(Xt), gp.predict(Xt, return_std=False), atol=1e-10)

    so = experimental_design.SobolIndex(n_samples=1024, n_bootstrap=20, random_state=0).analyze(
        lambda Z: Z[:, 0] + 2 * Z[:, 1], bounds=[(0, 1), (0, 1)])
    assert all(isinstance(r, montecarlo.MCResult) for r in so.first_order_ + so.total_order_)
    assert np.allclose(so.S1_, [0.2, 0.8], atol=0.03)

    a = np.repeat([0.0, 1.0, 2.0], 8)
    b = np.tile([0.0, 1.0], 12)
    yy = a + 0.5 * b + rng.normal(0, 1, 24)
    mine = {r["source"]: r["ss"] for r in experimental_design.ANOVA_DOE(
        model="interaction").fit(np.column_stack([a, b]), yy).table_.table}
    theirs = {r["source"]: r["ss"] for r in statistics.ANOVA(yy, factors=[a, b]).table}
    assert mine["A"] == pytest.approx(theirs["A"]) and mine["AB"] == pytest.approx(theirs["A:B"])
