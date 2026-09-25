"""Embedded self-check suite for stochpylib.

Ships inside the installed package so ``spl --test`` can verify any installation
(pip-installed copies included) without pytest or a source checkout. This is a fast
smoke test — the full test suite lives in ``tests/`` at the repository root and runs
via ``pytest tests/ -v``.

Run programmatically via :func:`run` (returns the number of failing checks).
"""

import numpy as np

import stochpylib
from stochpylib.distributions import (
    AlphaStable,
    Bernoulli,
    Beta,
    Binomial,
    Cauchy,
    Chi2,
    Dirichlet,
    Exponential,
    Gamma,
    Geometric,
    LevyDistribution,
    LogNormal,
    MultivariateNormal,
    Normal,
    Poisson,
    Rayleigh,
    StableDistribution,
    Student_t,
    Uniform,
    Weibull,
)

_TOL = 1e-6


class _SelfTest:
    def __init__(self):
        self.failures = []
        self.count = 0

    def check(self, name, cond):
        self.count += 1
        if not cond:
            self.failures.append(name)
            print(f"  FAIL {name}")
        return bool(cond)


def _univariate_checks(st, name, d, discrete=False, moments=True):
    lo, hi = d.support()
    x = float(d.ppf(0.4)) if np.isfinite(lo) or np.isfinite(hi) else 0.3
    p = float(np.atleast_1d(np.asarray(d.pdf(x), dtype=float))[0])
    st.check(f"{name}: pdf>=0", np.isfinite(p) and p >= 0.0)
    c = float(d.cdf(x))
    st.check(f"{name}: cdf in [0,1]", 0.0 <= c <= 1.0)
    q = 0.5
    xp = float(d.ppf(q))
    cp = float(d.cdf(xp))
    if discrete:
        st.check(f"{name}: ppf semantics", cp >= q)
    else:
        st.check(f"{name}: cdf(ppf(q))==q", abs(cp - q) < 1e-4)
    smp = np.asarray(d.rvs(50, random_state=0), dtype=float)
    st.check(f"{name}: rvs finite", bool(np.all(np.isfinite(smp))))
    if moments:
        m, v = d.mean(), d.var()
        if np.isfinite(m) and np.isfinite(v):
            st.check(f"{name}: var>0", v >= 0.0)


def _stats_t_cdf(x, df):
    from scipy import special
    x = np.asarray(x, dtype=float)
    r = df / (df + x * x)
    body = 0.5 * special.betainc(0.5 * df, 0.5, r)
    return np.where(x >= 0, 1.0 - body, body)


def run(verbose=False):
    """Run all self-checks. Returns the number of failing checks (0 == success)."""
    st = _SelfTest()
    if verbose:
        print(f"stochpylib selftest - version {stochpylib.__version__}")

    # package sanity
    st.check("package: version string", isinstance(stochpylib.__version__, str))
    st.check("package: probability import", hasattr(getattr(stochpylib, "probability"), "bayes_theorem"))
    bt = stochpylib.probability.bayes_theorem(0.01, 0.99, 0.0197)
    st.check("probability: bayes sanity", abs(bt - 0.502539) < 0.001)

    # univariate distributions (one small instance each; closed-form spot values where cheap)
    n = Normal(0.0, 1.0)
    st.check("Normal: pdf(0)", abs(n.pdf(0.0) - 0.3989422804014327) < _TOL)
    st.check("Normal: mean/var", abs(n.mean()) < _TOL and abs(n.var() - 1.0) < _TOL)
    _univariate_checks(st, "Normal", n)

    e = Exponential(2.0)
    st.check("Exponential: mean", abs(e.mean() - 0.5) < _TOL)
    _univariate_checks(st, "Exponential", e)

    b = Bernoulli(0.3)
    st.check("Bernoulli: pmf(1)", abs(float(b.pmf(1)) - 0.3) < _TOL)
    _univariate_checks(st, "Bernoulli", b, discrete=True)

    bi = Binomial(10, 0.4)
    st.check("Binomial: mean", abs(bi.mean() - 4.0) < _TOL)
    _univariate_checks(st, "Binomial", bi, discrete=True)

    po = Poisson(3.0)
    st.check("Poisson: pmf sums", abs(float(np.sum(po.pmf(np.arange(0, 60))))) > 1.0 - 1e-8)
    _univariate_checks(st, "Poisson", po, discrete=True)

    g = Geometric(0.25)
    st.check("Geometric: mean", abs(g.mean() - 4.0) < _TOL)
    _univariate_checks(st, "Geometric", g, discrete=True)

    u = Uniform(0.0, 1.0)
    st.check("Uniform: pdf height", abs(u.pdf(0.5) - 1.0) < _TOL)
    _univariate_checks(st, "Uniform", u)

    be = Beta(2.0, 3.0)
    st.check("Beta: mean", abs(be.mean() - 0.4) < _TOL)
    _univariate_checks(st, "Beta", be)

    ga = Gamma(2.0, 1.0)
    st.check("Gamma: mean", abs(ga.mean() - 2.0) < _TOL)
    _univariate_checks(st, "Gamma", ga)

    c2 = Chi2(3)
    st.check("Chi2: mean", abs(c2.mean() - 3.0) < _TOL)
    _univariate_checks(st, "Chi2", c2)

    t5 = Student_t(5)
    st.check("Student_t: var", abs(t5.var() - 5.0 / 3.0) < _TOL)
    _univariate_checks(st, "Student_t", t5)

    ca = Cauchy(0.0, 1.0)
    st.check("Cauchy: median", abs(ca.ppf(0.5)) < _TOL)
    _univariate_checks(st, "Cauchy", ca, moments=False)

    we = Weibull(2.0, 1.0)
    st.check("Weibull: cdf(1)", abs(we.cdf(1.0) - (1 - np.exp(-1.0))) < _TOL)
    _univariate_checks(st, "Weibull", we)

    ln = LogNormal(0.0, 0.5)
    st.check("LogNormal: mean", abs(ln.mean() - np.exp(0.125)) < _TOL)
    _univariate_checks(st, "LogNormal", ln)

    ra = Rayleigh(1.0)
    st.check("Rayleigh: mean", abs(ra.mean() - np.sqrt(np.pi / 2)) < _TOL)
    _univariate_checks(st, "Rayleigh", ra)

    # stable family: exact special-case delegation + fast CML sampling
    s2 = StableDistribution(2.0, 0.0, 0.0, 1.0)
    st.check(
        "Stable alpha=2 == Gaussian",
        abs(float(s2.pdf(0.7)) - float(Normal(0.0, np.sqrt(2)).pdf(0.7))) < _TOL,
    )
    sc = StableDistribution(1.0, 0.0, 0.0, 1.0)
    st.check("Stable alpha=1,beta=0 == Cauchy", abs(float(sc.pdf(1.3)) - float(Cauchy(0, 1).pdf(1.3))) < _TOL)
    sa = AlphaStable(1.5, 0.0, 1.0)
    xs = np.asarray(sa.rvs(20000, random_state=1))
    st.check("AlphaStable CML rvs: symmetric median~loc", abs(float(np.median(xs)) - 0.0) < 0.05)
    lv = LevyDistribution(0.0, 1.0)
    st.check("Levy: support", lv.support()[0] == 0.0)
    _univariate_checks(st, "Levy", lv, moments=False)

    # montecarlo quick checks
    from stochpylib.montecarlo import (
        AntitheticVariates,
        HaltonSequence,
        SobolSequence,
        crude_mc,
        pi_estimation,
    )
    from stochpylib.montecarlo.variance_reduction import _black_scholes_price

    h = HaltonSequence(2).generate(2)
    st.check("MC: halton first point", abs(h[0, 0] - 0.5) < 1e-12 and abs(h[0, 1] - 1 / 3) < 1e-12)
    vdc3 = np.array([0.5, 0.25, 0.75])
    s1 = SobolSequence(1).generate(3)[:, 0]
    st.check("MC: sobol d1 == van der Corput", np.all(s1 == vdc3))
    rmc = crude_mc(lambda p: p[:, 0] ** 2, n=50_000, random_state=7)
    st.check("MC: integral x^2 ~ 1/3", abs(rmc.estimate - 1 / 3) < 4 * rmc.std_error)
    bs = _black_scholes_price(100, 100, 1.0, 0.05, 0.2)
    pr = AntitheticVariates(n_simulations=40_000, random_state=8).price_european_call()
    st.check("MC: antithetic call ~ Black-Scholes", abs(pr.estimate - bs) < 4 * pr.std_error)
    rpi = pi_estimation(n=100_000, random_state=9)
    st.check("MC: pi in confidence band", abs(rpi.estimate - np.pi) < 4 * rpi.std_error)

    # timeseries quick checks
    from stochpylib.timeseries import ARIMA, GARCH, KalmanFilter, SpectralAnalysis
    from stochpylib.timeseries import adf_test as ts_adf

    rng_ts = np.random.default_rng(31)
    y_walk = np.cumsum(rng_ts.standard_normal(800))
    arima = ARIMA(1, 1, 0).fit(y_walk)
    fc_ts = arima.forecast(10)
    st.check("TS: ARIMA level forecast finite", bool(np.all(np.isfinite(fc_ts.mean))))
    g_ts = GARCH(1, 1).fit(0.01 * rng_ts.standard_normal(1200))
    st.check("TS: GARCH persistence < 1", g_ts.persistence_ < 1.0)
    kf_ts = KalmanFilter(F=[[1.0]], H=[[1.0]], Q=[[0.01]], R=[[1.0]])
    kf_ts.fit(np.cumsum(rng_ts.standard_normal(300)))
    st.check("TS: Kalman loglik finite", bool(np.isfinite(kf_ts.loglik_)))
    t_arr = np.arange(1000) / 50.0
    sine_sig = np.sin(2 * np.pi * 8.0 * t_arr)
    st.check("TS: dominant frequency ~8 Hz",
             abs(SpectralAnalysis(sine_sig, fs=50.0).dominant_frequency() - 8.0) < 0.5)
    adf_res = ts_adf(np.cumsum(rng_ts.standard_normal(400)))
    st.check("TS: ADF walk fails to reject", adf_res.statistic > -3.41)

    # gaussian_processes quick checks
    from stochpylib.gaussian_processes import (
        GPClassification,
        GPRegression,
        RBFKernel,
        SparseGaussianProcess,
    )

    rng_gp = np.random.default_rng(41)
    X_tr = np.linspace(0.0, 1.0, 24)[:, None]
    y_tr = np.sin(2 * np.pi * X_tr[:, 0]) + 0.05 * rng_gp.standard_normal(24)
    gp_reg = GPRegression(kernel=RBFKernel(length_scale=0.2), noise=0.01).fit(X_tr, y_tr)
    mu_gp, sd_gp = gp_reg.predict(X_tr, return_std=True)
    st.check("GP: regression tracks sine", float(np.max(np.abs(mu_gp - y_tr))) < 0.15)
    st.check("GP: predictive std positive/finite",
             bool(np.all(sd_gp > 0)) and bool(np.all(np.isfinite(sd_gp))))
    st.check("GP: log-marginal-likelihood finite",
             bool(np.isfinite(gp_reg.log_marginal_likelihood_)))
    y_cl = (X_tr[:, 0] > 0.5).astype(float)
    gpc = GPClassification(kernel=RBFKernel(length_scale=0.2)).fit(X_tr, y_cl)
    probs = gpc.predict_proba(X_tr)
    st.check("GP: classification probs in [0,1]",
             bool(np.all((probs >= 0.0) & (probs <= 1.0))))
    st.check("GP: classifier separates halves",
             float(probs[y_cl == 1].mean()) > 0.7
             and float(probs[y_cl == 0].mean()) < 0.3)
    sgp = SparseGaussianProcess(
        kernel=RBFKernel(length_scale=0.3),
        inducing_points=np.linspace(0.0, 1.0, 8)[:, None],
        noise=0.01,
    ).fit(X_tr, y_tr)
    mu_s = sgp.predict(X_tr, return_std=False)
    st.check("GP: sparse approximates exact", float(np.max(np.abs(mu_s - mu_gp))) < 0.2)

    # copulas quick checks
    from scipy.special import ndtr as _norm_cdf

    from stochpylib.copulas import (
        CheckerboardCopula,
        ClaytonCopula,
        CopulaFit,
        GaussianCopula,
        VineCopula,
    )

    rng_c = np.random.default_rng(61)
    zc = rng_c.multivariate_normal([0.0, 0.0], [[1.0, .6], [.6, 1.0]], 1200)
    gc_fit = GaussianCopula().fit(_norm_cdf(zc))
    st.check("COP: gaussian rho recovery",
             abs(float(gc_fit.correlation_[0, 1]) - 0.6) < 0.06)
    cl_data = ClaytonCopula(theta=3.0).sample(1500, random_state=62)
    cl_fit = ClaytonCopula().fit(cl_data)
    st.check("COP: clayton tail dependence",
             abs(cl_fit.tail_dependence()["lower"] - 2 ** (-1.0 / 3.0)) < 0.08)
    cb_fit = CheckerboardCopula(n_bins=10).fit(cl_data)
    st.check("COP: checkerboard mass", abs(cb_fit.cell_mass_.sum() - 1.0) < 1e-9)
    vine_fit = VineCopula(type="DVine").fit(_norm_cdf(zc))
    vs = vine_fit.sample(800, random_state=63)
    st.check("COP: vine margins", bool(np.all(np.abs(vs.mean(axis=0) - .5) < .06)))
    best = CopulaFit(families=("clayton", "gaussian", "frank")).fit(cl_data)
    st.check("COP: CopulaFit ranking", best.best_name_ == "clayton")

    # survival quick checks
    from stochpylib.survival import (
        KaplanMeier,
        CoxProportionalHazards,
        LogRankTest,
    )
    rng_s = np.random.default_rng(81)
    ts_exp = rng_s.exponential(2.0, 3000)
    cs = rng_s.uniform(.2, 8, 3000)
    s_km = KaplanMeier().fit(np.minimum(ts_exp, cs), (ts_exp <= cs).astype(int))
    st.check("SURV: KM exp S(2)", abs(float(s_km.predict([2.0])[0]) -
                                     np.exp(-1.0)) < .04)
    t_cox_a = rng_s.exponential(2, 800)
    c_cox_a = rng_s.uniform(.5, 8, 800)
    t_cox_b = rng_s.exponential(.5, 800)
    c_cox_b = rng_s.uniform(.5, 8, 800)
    t_cox = np.r_[np.minimum(t_cox_a, c_cox_a),
                  np.minimum(t_cox_b, c_cox_b)]
    e_cox = np.r_[(t_cox_a <= c_cox_a).astype(int),
                  (t_cox_b <= c_cox_b).astype(int)]
    x_cox = np.r_[np.zeros(800), np.ones(800)]
    cph = CoxProportionalHazards().fit(t_cox, e_cox, x_cox)
    st.check("SURV: Cox coef sign", cph.coefficients_[0] > .3)
    lr = LogRankTest().fit(t_cox, e_cox, np.repeat(['A', 'B'], 800))
    st.check("SURV: logrank separates", lr.p_value_ < 1e-10)

    # queueing quick checks
    from stochpylib.queueing import MM1Queue, erlang_b_formula, JacksonNetwork
    q_mm1 = MM1Queue().fit(.5, 1.0)
    st.check("QUEUE: M/M/1 L=1", abs(q_mm1.L - 1.0) < 1e-9)
    b_val = erlang_b_formula(10, 7.0)
    st.check("QUEUE: ErlangB(10,7)", .05 < b_val < .12)
    jn = JacksonNetwork([1., 0.], [3., 3.],
                        routing_matrix=[[0., 1.], [0., 0.]])
    jn.fit()
    st.check("QUEUE: Jackson traffic eq",
             np.allclose(jn.lam, [1., 1.], atol=1e-8))

    # library conformance + cross-module checks (mirrors tests/library)
    spec_counts = {
        "probability": (21, ["sample_space", "P", "bayes_theorem",
                             "derangement"]),
        "montecarlo": (25, ["SobolSequence", "crude_mc",
                            "AntitheticVariates", "pi_estimation"]),
        "timeseries": (61, ["ARIMA", "GARCH", "KalmanFilter", "PELT",
                            "adf_test", "forecast"]),
        "gaussian_processes": (36, ["GPRegression", "GPClassification",
                                    "RBFKernel", "RVine" if False else
                                    "optimize_hyperparams"]),
        "copulas": (26, ["GaussianCopula", "ClaytonCopula", "VineCopula",
                         "CopulaFit"]),
        "levy_processes": (33, ["LevyProcess", "TemperingSubordinator",
                                "KouJumpDiffusion", "StochasticTaylor"]),
        "financial_stochastics": (50, ["BlackScholes", "HestonModel",
                                       "ValueAtRisk", "RiskParity"]),
        "statistics": (48, ["t_test", "linear_regression", "PCA", "bootstrap_ci"]),
        "random_matrix": (23, ["GOE", "MarchenkoPastur", "HaarMeasure",
                               "EigenvalueSpacing"]),
        "advanced_mcmc": (35, ["MetropolisHastings", "NoUTurnSampler", "Rhat", "ESS"]),
        "numerical_methods": (38, ["GaussLegendre", "DormandPrince", "Brent",
                                   "MatrixExponential", "FiniteDifference"]),
        "bayesian": (25, ["posterior", "conjugate_prior", "BayesianLinear", "WAIC",
                          "LaplacePosterior"]),
        "robust_statistics": (28, ["Median", "HuberRegression", "MCD", "TheilSenRegression",
                                   "BlockBootstrap"]),
        "nonparametric": (31, ["KernelDensityEstimate", "EmpiricalCDF", "KruskalWallis",
                               "KendallTau", "LocalPolynomialReg"]),
        "optimization": (32, ["AdamOptimizer", "BFGS", "ParticleSwarmOptimization",
                              "SPSA", "AugmentedLagrangian"]),
        "experimental_design": (29, ["FullFactorial", "FractionalFactorial",
                                     "D_OptimalDesign", "LatinHypercubeDesign",
                                     "ResponseSurface", "SobolIndex"]),
    }
    for mod_name, (count, spot) in spec_counts.items():
        mod = getattr(stochpylib, mod_name, None)
        ok_mod = mod is not None and all(hasattr(mod, n) for n in spot)
        st.check(f"CONFORM: {mod_name} exports ({count})", ok_mod)
    dist_ok = hasattr(stochpylib.distributions, "Normal") and \
        all(hasattr(stochpylib.distributions.Normal, m)
            for m in ("pdf", "cdf", "ppf", "rvs", "fit", "ks_test"))
    st.check("CONFORM: distributions contract spot", dist_ok)

    # cross-module: reliability_mc driven by a library Weibull
    # (Weibull is imported at module level above)
    import stochpylib.montecarlo as _mc_mod
    rel = _mc_mod.reliability_mc(lambda X: X[:, 0], [Weibull(2.0, 10.0)],
                                    threshold=5.0, n=30000,
                                    random_state=71)
    p_true = 1 - np.exp(-0.25)
    st.check("XMOD: reliability vs closed form",
             abs(rel.estimate - p_true) < 4 * np.sqrt(
                 p_true * (1 - p_true) / 30000))
    # cross-module: copula margins through the library Student_t
    from scipy.special import ndtr as _ndtr
    zc2 = rng_c.multivariate_normal([0.0, 0.0], [[1.0, .5], [.5, 1.0]], 1500)
    w2 = rng_c.chisquare(4, 1500)
    tc_data = _stats_t_cdf(zc2 * np.sqrt(4 / w2)[:, None], 4)
    tfit = __import__("stochpylib.copulas.elliptical",
                      fromlist=["StudentTCopula"]).StudentTCopula().fit(
        tc_data)
    st.check("XMOD: t-copula df recovery", 3.0 < tfit.df_ < 6.5)

    # levy_processes quick checks
    from stochpylib.levy_processes import (
        GammaSubordinator,
        HawkesProcess,
        KouJumpDiffusion,
        TemperingSubordinator,
    )
    from stochpylib.levy_processes.jump_diffusion import _black_scholes_call
    from stochpylib.levy_processes.sde import SDE, Euler_Maruyama

    rng_lp = np.random.default_rng(101)
    ts_lp = TemperingSubordinator(C=1.0, lam=5.0, alpha=0.5)
    ts_inc = np.array([ts_lp._increment(0.1, rng_lp) for _ in range(20000)])
    st.check("LEVY: TemperingSubordinator mean matches analytic",
             abs(ts_inc.mean() - ts_lp.mean_rate() * 0.1) < 0.05 * ts_lp.mean_rate() * 0.1)
    gs_lp = GammaSubordinator(rate=2.0, scale=0.5)
    gs_inc = np.array([gs_lp._increment(1.0, rng_lp) for _ in range(20000)])
    st.check("LEVY: GammaSubordinator mean", abs(gs_inc.mean() - 1.0) < 0.05)
    kou0 = KouJumpDiffusion(mu=0.05, sigma=0.2, jump_rate=0.0)
    st.check("LEVY: Kou zero-jump call_price == Black-Scholes",
             abs(kou0.call_price(100.0, 100.0, 1.0, 0.05)
                 - _black_scholes_call(100.0, 100.0, 1.0, 0.05, 0.2)) < 0.01)
    hp_lp = HawkesProcess(mu=0.5, alpha=0.3, beta=1.0)
    st.check("LEVY: Hawkes branching_ratio", abs(hp_lp.branching_ratio() - 0.3) < _TOL)
    sde_lp = SDE(drift=lambda t, x: 0.05 * x, diffusion=lambda t, x: 0.2 * x, x0=100.0)
    em_paths = Euler_Maruyama(sde_lp, T=1.0, n_steps=100, n_paths=20000, random_state=102)
    st.check("LEVY: Euler-Maruyama GBM terminal mean",
             abs(em_paths[:, -1].mean() - 100.0 * np.exp(0.05)) < 3.0)

    # financial_stochastics quick checks
    from stochpylib.financial_stochastics import BlackScholes, HestonModel, RiskParity, VasicekModel
    from stochpylib.financial_stochastics.option_pricing import BinomialTree as _FinBinomialTree

    bs_fs = BlackScholes(S=100, K=100, T=1, r=0.05, sigma=0.2)
    st.check("FIN: BlackScholes reference value",
             abs(bs_fs.call_price() - 10.4506) < 1e-3)
    st.check("FIN: BlackScholes put-call parity",
             abs((bs_fs.call_price() - bs_fs.put_price())
                 - (100.0 - 100.0 * np.exp(-0.05))) < 1e-9)
    bt_fs = _FinBinomialTree(100, 100, 1, 0.05, 0.2, n_steps=500)
    st.check("FIN: Binomial converges toward Black-Scholes",
             abs(bt_fs.call_price() - bs_fs.call_price()) < 0.05)
    heston_fs = HestonModel(S0=100, v0=0.04, kappa=2, theta=0.04, xi=1e-4, rho=-0.7, r=0.05)
    bs_match_fs = BlackScholes(S=100, K=100, T=1, r=0.05, sigma=0.2)
    st.check("FIN: Heston xi->0 matches Black-Scholes",
             abs(heston_fs.call_price(100, 1) - bs_match_fs.call_price()) < 5e-3)
    vas_fs = VasicekModel(r0=0.03, kappa=0.5, theta=0.04, sigma=0.01)
    hw_fs = __import__("stochpylib.financial_stochastics.rate_models",
                       fromlist=["HullWhiteModel"]).HullWhiteModel(
        kappa=0.5, sigma=0.01, r0=0.03, theta=0.02)
    vas_match_fs = VasicekModel(0.03, 0.5, 0.02 / 0.5, 0.01)
    st.check("FIN: HullWhite const-theta matches Vasicek ZCB",
             abs(hw_fs.zcb_price(5) - vas_match_fs.zcb_price(5)) < 1e-9)
    rp_fs = RiskParity(np.array([[0.04, 0.015, 0.0], [0.015, 0.03, 0.005], [0.0, 0.005, 0.02]]))
    w_fs = rp_fs.optimize()
    rc_fs = rp_fs.risk_contributions(w_fs)
    st.check("FIN: RiskParity equal risk contributions",
             np.allclose(rc_fs, rc_fs[0], atol=1e-6))

    # statistics quick checks
    from stochpylib.statistics import (
        PCA as _StatPCA,
        ANOVA as _StatANOVA,
        bonferroni as _stat_bonferroni,
        bootstrap_ci as _stat_bootstrap_ci,
        linear_regression as _stat_linear_regression,
        t_test as _stat_t_test,
    )
    from stochpylib.statistics._common import _ptukey as _stat_ptukey

    rng_st = np.random.default_rng(103)
    x_st = rng_st.normal(5.0, 2.0, 200)
    t_st = _stat_t_test(x_st, mu0=5.0)
    st.check("STAT: t_test null-true type-I not rejected", t_st.pvalue > 0.001)

    Xr = rng_st.normal(size=(200, 2))
    yr = 1.0 + Xr @ np.array([2.0, -1.0]) + rng_st.normal(0, 0.5, 200)
    ols_st = _stat_linear_regression(Xr, yr)
    st.check("STAT: OLS recovers known slope",
             abs(ols_st.coef_[1] - 2.0) < 5 * ols_st.std_errors_[1])

    g1_st, g2_st, g3_st = (rng_st.normal(0, 1, 40), rng_st.normal(1.5, 1, 40),
                            rng_st.normal(3.0, 1, 40))
    aov_st = _StatANOVA(g1_st, g2_st, g3_st)
    st.check("STAT: ANOVA detects group differences", aov_st.pvalue < 1e-6)

    pca_st = _StatPCA(rng_st.normal(size=(100, 4)) @ np.diag([4.0, 1.0, 0.1, 0.01]))
    st.check("STAT: PCA explained variance is decreasing",
             np.all(np.diff(pca_st.explained_variance_) <= 0))

    boot_st = _stat_bootstrap_ci(x_st, lambda a: np.mean(a, axis=-1), n_boot=500,
                                  method="percentile", random_state=1)
    lo_st, hi_st = boot_st.extras["conf_int"]
    st.check("STAT: bootstrap CI covers the sample mean", lo_st < np.mean(x_st) < hi_st)

    adj_st = _stat_bonferroni([0.001, 0.2, 0.5], method="holm")
    st.check("STAT: Holm adjustment is monotone non-decreasing with rank",
             adj_st.table[0]["adjusted_pvalue"] <= adj_st.table[1]["adjusted_pvalue"])

    st.check("STAT: studentized range CDF at q=0 is 0",
             abs(_stat_ptukey(0.0, 3, 20)) < 1e-8)

    # random_matrix quick checks
    from stochpylib.random_matrix import (
        GOE as _RmGOE,
        EigenvalueSpacing as _RmSpacing,
        HaarMeasure as _RmHaar,
        MarchenkoPastur as _RmMP,
        MuresanMatrix as _RmGinibre,
        TracyWidomDistribution as _RmTW,
        WignerSemicircle as _RmSemicircle,
    )

    semi_rm = _RmSemicircle(2.0)
    st.check("RMT: semicircle cdf spans [0, 1] and var = R^2/4",
             abs(semi_rm.cdf(2.0) - 1.0) < 1e-12 and abs(semi_rm.cdf(-2.0)) < 1e-12
             and abs(semi_rm.var() - 1.0) < 1e-12)
    goe_rm = _RmGOE(300)
    eig_rm = goe_rm.eigenvalues(random_state=104)
    st.check("RMT: GOE(300) bulk matches the semicircle",
             semi_rm.compare(goe_rm.normalize(eig_rm)).pvalue > 1e-3)
    st.check("RMT: GOE mean gap ratio ~ 0.53 (GOE reference)",
             abs(_RmSpacing(eig_rm).mean_ratio() - 0.5307) < 0.05)
    q_rm = _RmHaar("O", 6).sample(random_state=105)
    st.check("RMT: Haar orthogonal matrix satisfies Q^T Q = I",
             bool(np.allclose(q_rm.T @ q_rm, np.eye(6), atol=1e-10)))
    mp_rm = _RmMP(0.5, sigma=1.3)
    st.check("RMT: Marchenko-Pastur mean = sigma^2", abs(mp_rm.mean() - 1.69) < 1e-12)
    st.check("RMT: Tracy-Widom beta=2 mean ~ -1.771",
             abs(_RmTW(2).mean() + 1.7711) < 1e-3)
    z_rm = _RmGinibre(200).normalized_eigenvalues(random_state=106)
    st.check("RMT: Ginibre-type circular law E|z|^2 ~ 1/2",
             abs(float(np.mean(np.abs(z_rm) ** 2)) - 0.5) < 0.08)

    # advanced_mcmc quick checks
    from stochpylib.advanced_mcmc import ESS as _AmESS
    from stochpylib.advanced_mcmc import MetropolisHastings as _AmMH
    from stochpylib.advanced_mcmc import NoUTurnSampler as _AmNUTS
    from stochpylib.advanced_mcmc import Rhat as _AmRhat
    from stochpylib.advanced_mcmc import SequentialMonteCarlo as _AmSMC
    from stochpylib.advanced_mcmc import SliceSampling as _AmSlice
    from stochpylib.advanced_mcmc import MeanFieldVI as _AmMFVI
    from stochpylib.advanced_mcmc import autocorr_time as _am_autocorr_time

    mh_rm = _AmMH(lambda t: -0.5 * ((t[0] - 1.0) / 0.5) ** 2, n_samples=1500, n_warmup=500)
    mh_rm.sample(np.zeros(1), random_state=107)
    st.check("MCMC: random-walk MH recovers N(1, 0.5^2) mean",
             abs(mh_rm.get_samples().mean() - 1.0) < 0.1)

    nuts_rm = _AmNUTS(lambda t: -0.5 * t @ np.linalg.inv([[1.0, 0.8], [0.8, 1.0]]) @ t,
                      n_samples=800, n_warmup=400)
    nuts_rm.sample(np.zeros(2), random_state=108)
    chains_rm = nuts_rm.get_chains()
    st.check("MCMC: NUTS on a correlated 2-D Gaussian mixes well",
             0.5 < nuts_rm.acceptance_rate_ < 0.98
             and abs(np.cov(nuts_rm.get_samples().T)[0, 1] - 0.8) < 0.25)

    iid_rm = np.random.default_rng(109).standard_normal((4, 500, 1))
    st.check("MCMC: R-hat of four iid chains ~ 1", abs(_AmRhat(iid_rm) - 1.0) < 0.03)
    st.check("MCMC: ESS of iid draws ~ n", _AmESS(iid_rm, method="mean") > 0.6 * 2000)

    sl_rm = _AmSlice(lambda t: -t[0] if t[0] > 0 else -np.inf, n_samples=1500, n_warmup=300)
    sl_rm.sample(np.array([1.0]), random_state=110)
    st.check("MCMC: slice sampler on Exp(1) has mean ~ 1",
             abs(sl_rm.get_samples().mean() - 1.0) < 0.1)

    y_rm = np.array([1.0])
    smc_rm = _AmSMC(lambda t: -0.5 * np.sum(t ** 2), lambda t: -0.5 * np.sum(((t - y_rm) / 0.5) ** 2),
                    lambda n, r: r.standard_normal((n, 1)), n_particles=1500, n_mcmc=3)
    smc_rm.sample(random_state=111)
    logZ_true_rm = -0.5 * np.log(2 * np.pi * 1.25) - 0.5 * y_rm[0] ** 2 / 1.25
    st.check("MCMC: SMC log evidence matches the Gaussian closed form",
             abs(smc_rm.log_evidence_ - logZ_true_rm) < 0.3)

    mfvi_rm = _AmMFVI(lambda t: -0.5 * ((t[0] - 2.0) / 1.0) ** 2, 1, n_iter=800, n_mc=16)
    mfvi_rm.fit(random_state=112)
    st.check("MCMC: mean-field VI locates N(2, 1)", abs(mfvi_rm.mean_[0] - 2.0) < 0.2)

    rng_ar1 = np.random.default_rng(113)
    ar1_rm = np.zeros(20000)
    for t in range(1, 20000):
        ar1_rm[t] = 0.5 * ar1_rm[t - 1] + rng_ar1.standard_normal()
    tau_rm = _am_autocorr_time(ar1_rm)
    st.check("MCMC: autocorr_time of AR(1) phi=0.5 ~ 3", 2.0 < tau_rm < 4.5)

    # numerical_methods quick checks
    from stochpylib.numerical_methods import (
        BDF as _NmBDF,
        Brent as _NmBrent,
        DormandPrince as _NmDP,
        FiniteDifference as _NmFD,
        GaussLegendre as _NmGL,
        MatrixExponential as _NmExpm,
        SplineInterpolation as _NmSpline,
    )
    from stochpylib.numerical_methods.pde import SpectralMethod as _NmSpectral

    gl_nm = _NmGL(5)
    st.check("NUM: Gauss-Legendre exact for x^8 (n=5)",
             abs(gl_nm.integrate(lambda x: x ** 8, -1, 1).value - 2 / 9) < 1e-10)

    def _npdf(x):
        return np.exp(-0.5 * x * x) / np.sqrt(2 * np.pi)

    from stochpylib.numerical_methods import AdaptiveQuadrature as _NmAdaptive
    aq_nm = _NmAdaptive(_npdf, -np.inf, np.inf).integrate()
    st.check("NUM: adaptive quadrature of N(0,1) pdf over R = 1", abs(aq_nm.value - 1.0) < 1e-8)

    dp_nm = _NmDP(rtol=1e-8).solve(lambda t, y: y, (0, 1), [1.0])
    st.check("NUM: Dormand-Prince y'=y gives e", abs(dp_nm.y[-1, 0] - np.e) < 1e-6)

    bdf_nm = _NmBDF(order=2).solve(lambda t, y: np.array([-1000 * y[0]]), (0, 1), [1.0], h=0.01)
    st.check("NUM: BDF stays bounded on a stiff y'=-1000y",
             bool(np.all(np.isfinite(bdf_nm.y))))

    brent_nm = _NmBrent(lambda x: np.cos(x) - x, 0, 1)
    st.check("NUM: Brent solves cos(x)=x", abs(brent_nm.root - 0.7390851332151607) < 1e-10)

    xc_nm = np.linspace(0, 5, 7)
    yc_nm = 2 * xc_nm ** 3 - xc_nm ** 2 + 3 * xc_nm - 1
    sp_nm = _NmSpline(xc_nm, yc_nm, bc="not-a-knot")
    st.check("NUM: not-a-knot spline reproduces a cubic exactly",
             abs(float(sp_nm(2.5)) - (2 * 2.5 ** 3 - 2.5 ** 2 + 3 * 2.5 - 1)) < 1e-8)

    rot_nm = np.array([[0.0, -1.0], [1.0, 0.0]])
    expm_nm = _NmExpm(rot_nm).at(np.pi / 2)
    st.check("NUM: expm of a 2-D rotation generator is a 90-degree rotation",
             bool(np.allclose(expm_nm, [[0.0, -1.0], [1.0, 0.0]], atol=1e-8)))

    fd_nm = _NmFD()
    bs_price_nm = fd_nm.black_scholes(100.0, 1.0, 0.05, 0.2, kind="call").price(100.0)
    from stochpylib.financial_stochastics import BlackScholes as _NmBS
    bs_ref_nm = _NmBS(S=100.0, K=100.0, T=1.0, r=0.05, sigma=0.2).call_price()
    st.check("NUM: Crank-Nicolson Black-Scholes PDE matches closed form",
             abs(bs_price_nm - bs_ref_nm) < 2e-2)

    sm_nm = _NmSpectral()
    x_nm = np.linspace(0, 2 * np.pi, 64, endpoint=False)
    du_nm = sm_nm.fourier_derivative(np.sin(2 * x_nm), 2 * np.pi)
    st.check("NUM: Fourier spectral derivative of sin(2x) matches 2cos(2x)",
             bool(np.max(np.abs(du_nm - 2 * np.cos(2 * x_nm))) < 1e-10))

    # bayesian quick checks
    from stochpylib.bayesian import (
        prior as _bay_prior,
        likelihood as _bay_lik,
        posterior as _bay_post,
        posterior_predictive as _bay_pred,
        evidence as _bay_ev,
        BayesianLinear as _BayLinear,
        BayesianNetwork as _BayNet,
        LaplacePosterior as _BayLaplace,
        WAIC as _BayWAIC,
        DirichletProcess as _BayDP,
        NaiveBayes as _BayNB,
    )
    from stochpylib.distributions import Beta as _BayBeta

    rng_bay = np.random.default_rng(120)
    x_bay = rng_bay.binomial(1, 0.4, 300)
    post_bay = _bay_post(_bay_prior(_BayBeta(2, 3)), _bay_lik("bernoulli", data=x_bay),
                          method="conjugate")
    s_bay, n_bay = float(x_bay.sum()), len(x_bay)
    st.check("BAYES: conjugate beta posterior params",
             abs(post_bay.dist.a - (2 + s_bay)) < _TOL and abs(post_bay.dist.b - (3 + n_bay - s_bay)) < _TOL)
    post_grid_bay = _bay_post(_bay_prior(_BayBeta(2, 3)), _bay_lik("bernoulli", data=x_bay),
                               method="grid", n_grid=2001)
    st.check("BAYES: grid posterior mean matches conjugate",
             abs(post_grid_bay.mean()[0] - post_bay.mean()[0]) < 1e-3)
    ev_conj_bay = _bay_ev(_bay_prior(_BayBeta(2, 3)), _bay_lik("bernoulli", data=x_bay), method="conjugate")
    ev_grid_bay = _bay_ev(_bay_prior(_BayBeta(2, 3)), _bay_lik("bernoulli", data=x_bay), method="grid",
                           n_grid=2001)
    st.check("BAYES: conjugate evidence matches grid evidence", abs(ev_conj_bay - ev_grid_bay) < 1e-3)
    pred_bay = _bay_pred(post_bay)
    st.check("BAYES: beta-binomial predictive mean matches posterior mean",
             abs(pred_bay.mean() - post_bay.mean()[0]) < _TOL)

    X_bay = rng_bay.standard_normal((100, 2))
    beta_true_bay = np.array([1.0, 2.0, -1.0])
    y_bay = beta_true_bay[0] + X_bay @ beta_true_bay[1:] + rng_bay.normal(0, 0.1, 100)
    lin_bay = _BayLinear(prior_precision=1e-10, a0=1e-8, b0=1e-8).fit(X_bay, y_bay)
    ols_bay = np.linalg.lstsq(np.column_stack([np.ones(100), X_bay]), y_bay, rcond=None)[0]
    st.check("BAYES: BayesianLinear matches OLS with a near-flat prior",
             bool(np.max(np.abs(lin_bay.coef_ - ols_bay)) < 1e-6))

    lap_bay = _BayLaplace(lambda t: -0.5 * t[0] ** 2, np.array([1.0]))
    st.check("BAYES: LaplacePosterior exact on a standard Gaussian target",
             abs(lap_bay.mean_[0]) < 1e-3 and abs(lap_bay.cov_[0, 0] - 1.0) < 1e-3)

    ll_bay = np.array([[0.0, -1.0], [-0.5, -0.5], [-1.0, 0.0]])
    waic_bay = _BayWAIC(ll_bay)
    from scipy.special import logsumexp as _lse_bay
    lppd_bay = _lse_bay(ll_bay, axis=0) - np.log(3)
    pwaic_bay = np.var(ll_bay, axis=0, ddof=1)
    st.check("BAYES: WAIC matches its own hand formula",
             abs(waic_bay.value - (-2 * np.sum(lppd_bay - pwaic_bay))) < 1e-8)

    bn_bay = _BayNet()
    for name_bay in ("Cloudy", "Sprinkler", "Rain", "WetGrass"):
        bn_bay.add_node(name_bay, [0, 1])
    bn_bay.add_edge("Cloudy", "Sprinkler")
    bn_bay.add_edge("Cloudy", "Rain")
    bn_bay.add_edge("Sprinkler", "WetGrass")
    bn_bay.add_edge("Rain", "WetGrass")
    bn_bay.set_cpt("Cloudy", [0.5, 0.5])
    bn_bay.set_cpt("Sprinkler", [[0.5, 0.5], [0.9, 0.1]])
    bn_bay.set_cpt("Rain", [[0.8, 0.2], [0.2, 0.8]])
    wg_bay = np.zeros((2, 2, 2))
    wg_bay[0, 0] = [1.0, 0.0]; wg_bay[0, 1] = [0.1, 0.9]
    wg_bay[1, 0] = [0.1, 0.9]; wg_bay[1, 1] = [0.01, 0.99]
    bn_bay.set_cpt("WetGrass", wg_bay)
    res_bay = bn_bay.query(["Rain"], {"WetGrass": 1})
    st.check("BAYES: sprinkler network P(Rain=1|WetGrass=1)", abs(res_bay[1] - 0.70792768) < 1e-3)

    dp_bay = _BayDP(alpha=2.0)
    st.check("BAYES: DirichletProcess expected_clusters closed form",
             abs(dp_bay.expected_clusters(10) - sum(2.0 / (2.0 + i) for i in range(10))) < 1e-10)

    Xnb_bay = np.array([[0.0, 0.0], [0.1, -0.1], [5.0, 5.0], [5.1, 4.9]])
    ynb_bay = np.array([0, 0, 1, 1])
    nb_bay = _BayNB("gaussian").fit(Xnb_bay, ynb_bay)
    st.check("BAYES: NaiveBayes separates two well-separated blobs",
             bool(np.all(nb_bay.predict(Xnb_bay) == ynb_bay)))

    from stochpylib.bayesian import bayes_factor as _bay_bf
    bf_bay = _bay_bf(post_bay.log_evidence_, -1e9)
    st.check("BAYES: bayes_factor > 1 for the far-better model", bf_bay.value > 1.0)

    # robust_statistics quick checks
    from stochpylib.robust_statistics import (
        HodgesLehmann as _RsHL,
        HuberRegression as _RsHuberReg,
        LTS_Regression as _RsLTS,
        MCD as _RsMCD,
        MMRegression as _RsMM,
        Median as _RsMedian,
        MedianAbsoluteDeviation as _RsMAD,
        Qn_Estimator as _RsQn,
        TheilSenRegression as _RsTheilSen,
        TrimmedMean as _RsTrimMean,
        BlockBootstrap as _RsBlockBoot,
        CovShrinkage as _RsShrink,
    )

    trim_rs = _RsTrimMean(0.1).fit(np.arange(1.0, 11.0))
    st.check("ROBUST: trimmed mean of 1..10 (p=0.1) matches hand formula",
             abs(trim_rs.estimate_ - np.mean(np.arange(2.0, 10.0))) < 1e-10)

    mad_rs = _RsMAD(n_boot=0).fit(np.array([1.0, 2.0, 3.0, 4.0, 5.0, 100.0]))
    st.check("ROBUST: MAD resists a single gross outlier",
             abs(mad_rs.estimate_ - 1.4826022185056018 * 1.5) < 1e-8)

    qn_rs = _RsQn(n_boot=0).fit(np.arange(1.0, 11.0))
    st.check("ROBUST: Qn of 1..10 matches the stored reference value",
             abs(qn_rs.estimate_ - 4.438288931970152) < 1e-8)

    x_rs = np.array([1.0, 2.0, 3.0, 4.0, 100.0])
    hl_rs = _RsHL().fit(x_rs)
    st.check("ROBUST: Hodges-Lehmann on a known small sample",
             abs(hl_rs.estimate_ - 3.0) < 1e-8)

    med_rs = _RsMedian().fit(x_rs)
    st.check("ROBUST: Median resists a gross outlier", abs(med_rs.estimate_ - 3.0) < 1e-10)

    rng_rs = np.random.default_rng(200)
    t_rs = rng_rs.uniform(0, 10, 60)
    y_rs = 1.0 + 2.0 * t_rs
    ts_rs = _RsTheilSen().fit(t_rs, y_rs)
    st.check("ROBUST: Theil-Sen recovers an exact line", abs(ts_rs.coef_[1] - 2.0) < 1e-8)

    y_out_rs = y_rs + rng_rs.normal(0, 0.05, 60)
    y_out_rs[:18] += 20.0
    hub_rs = _RsHuberReg().fit(t_rs, y_out_rs)
    mm_rs = _RsMM(random_state=0).fit(t_rs, y_out_rs)
    lts_rs = _RsLTS(random_state=0).fit(t_rs, y_out_rs)
    st.check("ROBUST: MM slope recovers truth under leverage-free y-outliers",
             abs(mm_rs.coef_[1] - 2.0) < 0.15)
    st.check("ROBUST: LTS slope recovers truth under leverage-free y-outliers",
             abs(lts_rs.coef_[1] - 2.0) < 0.15)
    st.check("ROBUST: Huber regression runs and returns finite coefficients",
             bool(np.all(np.isfinite(hub_rs.coef_))))

    rng_mcd = np.random.default_rng(201)
    X_mcd = rng_mcd.standard_normal((300, 2))
    out_mcd = rng_mcd.choice(300, 45, replace=False)
    X_mcd[out_mcd] += 15.0
    mcd_rs = _RsMCD(random_state=0).fit(X_mcd)
    frac_flagged_rs = float(np.mean(mcd_rs.outliers()[out_mcd]))
    st.check("ROBUST: MCD flags most planted outliers", frac_flagged_rs > 0.8)

    shrink_rs = _RsShrink().fit(rng_mcd.standard_normal((50, 3)))
    st.check("ROBUST: Ledoit-Wolf shrinkage intensity in [0, 1]",
             0.0 <= shrink_rs.shrinkage_ <= 1.0)

    x_bb_rs = rng_mcd.standard_normal(200)
    bb_rs = _RsBlockBoot(np.mean, random_state=1).fit(x_bb_rs)
    bb_rs2 = _RsBlockBoot(np.mean, random_state=1).fit(x_bb_rs)
    st.check("ROBUST: block bootstrap SE finite and reproducible",
             np.isfinite(bb_rs.std_error_) and bb_rs.std_error_ == bb_rs2.std_error_)

    # nonparametric quick checks
    from stochpylib.nonparametric import (
        DistanceCorrelation as _NpDCor,
        EmpiricalCDF as _NpECDF,
        EmpiricalLikelihood as _NpEL,
        IsotonicRegression as _NpIso,
        KendallTau as _NpKendall,
        KernelDensityEstimate as _NpKDE,
        KruskalWallis as _NpKW,
        RunsTest as _NpRuns,
        SignTest as _NpSign,
    )

    rng_np = np.random.default_rng(300)
    x_np = rng_np.normal(0, 2, 300)
    kde_np = _NpKDE(bandwidth="silverman").fit(x_np)
    st.check("NONPAR: KDE pdf integrates to ~1",
             abs(np.trapezoid(kde_np.pdf(np.linspace(-10, 10, 2000)),
                               np.linspace(-10, 10, 2000)) - 1.0) < 0.02)

    ecdf_np = _NpECDF().fit(x_np)
    st.check("NONPAR: EmpiricalCDF matches the sample fraction",
             abs(ecdf_np.evaluate(0.0) - np.mean(x_np <= 0.0)) < 1e-10)

    el_np = _NpEL().fit(x_np)
    st.check("NONPAR: EmpiricalLikelihood rejects a clearly wrong mean",
             el_np.test_mean(50.0).pvalue < 0.001)

    g1_np, g2_np, g3_np = (rng_np.normal(0, 1, 30), rng_np.normal(0, 1, 30),
                            rng_np.normal(3, 1, 30))
    kw_np = _NpKW().fit(g1_np, g2_np, g3_np)
    st.check("NONPAR: Kruskal-Wallis detects the shifted third group",
             kw_np.pvalue_ < 0.001)

    xk_np = rng_np.normal(size=60)
    yk_np = xk_np + rng_np.normal(0, 0.3, 60)
    kt_np = _NpKendall().fit(xk_np, yk_np)
    from stochpylib.copulas import kendall_tau as _cop_kendall_tau
    st.check("NONPAR: KendallTau matches copulas.kendall_tau",
             abs(kt_np.estimate_ - float(_cop_kendall_tau(np.column_stack([xk_np, yk_np]))))
             < 1e-10)

    dc_np = _NpDCor(n_resamples=200, random_state=1).fit(
        rng_np.uniform(-2, 2, 100), rng_np.normal(size=100))
    st.check("NONPAR: DistanceCorrelation is in [0, 1]", 0.0 <= dc_np.estimate_ <= 1.0)

    iso_np = _NpIso().fit(np.arange(20), np.sort(rng_np.normal(size=20)))
    st.check("NONPAR: IsotonicRegression fit is non-decreasing",
             bool(np.all(np.diff(iso_np.fitted_) >= -1e-10)))

    sign_np = _NpSign(mu0=0.0).fit(rng_np.normal(2.0, 1, 40))
    st.check("NONPAR: SignTest rejects a clearly nonzero median", sign_np.pvalue_ < 0.01)

    runs_np = _NpRuns().fit(np.tile([1.0, -1.0], 20))
    st.check("NONPAR: RunsTest flags a perfectly alternating sequence",
             runs_np.reject(0.01))

    # optimization quick checks
    from stochpylib.optimization import (
        AugmentedLagrangian as _OptAugLag, BFGS as _OptBFGS, CMA_ES as _OptCMA,
        LevenbergMarquardt as _OptLM, Objective as _OptObjective,
        ParticleSwarmOptimization as _OptPSO, SPSA as _OptSPSA,
        SimulatedAnnealing as _OptSA,
    )
    A_opt = np.array([[3.0, 1.0], [1.0, 2.0]])
    b_opt = np.array([1.0, -1.0])
    star_opt = np.linalg.solve(A_opt, b_opt)
    f_opt = lambda z: 0.5 * float(z @ A_opt @ z) - float(b_opt @ z)
    g_opt = lambda z: A_opt @ z - b_opt
    sphere_opt = lambda z: float(np.sum(np.asarray(z, dtype=float) ** 2))

    bfgs_opt = _OptBFGS().minimize(f_opt, [5.0, 5.0], grad=g_opt)
    st.check("OPTIM: BFGS solves a quadratic to its closed-form minimizer",
             float(np.max(np.abs(bfgs_opt.x_ - star_opt))) < 1e-8)
    st.check("OPTIM: BFGS inverse-Hessian approximates inv(A)",
             float(np.max(np.abs(bfgs_opt.result_.hess_inv - np.linalg.inv(A_opt)))) < 0.01)
    st.check("OPTIM: minimize() returns self and exposes a converged OptimizeResult",
             bfgs_opt.result_ is bfgs_opt.to_result() and bfgs_opt.result_.converged)

    obj_opt = _OptObjective(f_opt)
    st.check("OPTIM: finite-difference gradient matches the analytic one",
             float(np.max(np.abs(obj_opt.grad(np.array([1.0, 2.0]))
                                 - g_opt(np.array([1.0, 2.0]))))) < 1e-6)
    st.check("OPTIM: Objective maps invalid values to +inf",
             _OptObjective(lambda z: float("nan"))(np.zeros(2)) == float("inf"))

    pso_opt = _OptPSO(bounds=(-5.0, 5.0), n_iter=60, n_particles=20,
                      random_state=0).minimize(sphere_opt, np.full(3, 3.0))
    st.check("OPTIM: particle swarm finds the sphere minimum", pso_opt.fun_ < 1e-6)
    pso_again = _OptPSO(bounds=(-5.0, 5.0), n_iter=60, n_particles=20,
                        random_state=0).minimize(sphere_opt, np.full(3, 3.0))
    st.check("OPTIM: the same random_state reproduces a run exactly",
             bool(np.array_equal(pso_opt.x_, pso_again.x_)))

    cma_opt = _OptCMA(sigma0=1.0, n_iter=150, random_state=0).minimize(
        sphere_opt, np.full(3, 2.0))
    st.check("OPTIM: CMA-ES converges and learns a covariance",
             cma_opt.fun_ < 1e-6 and cma_opt.result_.extras["C"].shape == (3, 3))

    sa_opt = _OptSA(n_iter=2000, bounds=(-5.0, 5.0), random_state=0).minimize(
        sphere_opt, np.full(3, 3.0))
    st.check("OPTIM: simulated annealing descends from a poor start", sa_opt.fun_ < 1.0)

    spsa_opt = _OptSPSA(n_iter=100, random_state=0).minimize(sphere_opt, np.ones(6))
    st.check("OPTIM: SPSA costs 2 objective evaluations per iteration at any dimension",
             spsa_opt.result_.extras["direction_evals"] == 200)

    t_opt = np.linspace(0.0, 2.0, 20)
    truth_opt = 2.0 * np.exp(-0.8 * t_opt)
    lm_opt = _OptLM().minimize(lambda q: q[0] * np.exp(-q[1] * t_opt) - truth_opt,
                               [1.0, 1.0])
    st.check("OPTIM: Levenberg-Marquardt recovers exact least-squares parameters",
             bool(np.allclose(lm_opt.x_, [2.0, 0.8], atol=1e-5)))

    al_opt = _OptAugLag(
        constraints=[{"type": "eq", "fun": lambda z: np.array([z[0] + z[1] - 1.0])}]
    ).minimize(sphere_opt, [2.0, -1.0])
    st.check("OPTIM: augmented Lagrangian solves min x^2+y^2 s.t. x+y=1",
             bool(np.allclose(al_opt.x_, [0.5, 0.5], atol=1e-5))
             and al_opt.result_.extras["feasible"])

    # experimental_design quick checks
    from stochpylib.experimental_design import (
        BoxBehnken as _DoeBB, CCD as _DoeCCD, D_OptimalDesign as _DoeD,
        FractionalFactorial as _DoeFF, FullFactorial as _DoeFull, GraecoLatin as _DoeGL,
        MainEffects as _DoeME, MaximinLHD as _DoeMM, NormalPlot as _DoeNP,
        Plackett_Burman as _DoePB, PolynomialChaos as _DoePCE,
        ResponseSurface as _DoeRS, SobolIndex as _DoeSobol, UniformDesign as _DoeUD,
    )

    full_doe = _DoeFull(2, 3).generate()
    st.check("DOE: 2^3 full factorial has orthogonal columns",
             full_doe.n_runs == 8 and bool(np.allclose(full_doe.points.T @ full_doe.points,
                                                       8 * np.eye(3))))
    st.check("DOE: searched 2^(5-1) fraction reaches resolution V",
             _DoeFF(5, p=1).generate().properties["resolution"] == 5)
    pb_doe = _DoePB(11).generate()
    H_doe = np.column_stack([np.ones(12), pb_doe.points])
    st.check("DOE: Plackett-Burman(11) is a 12-run Hadamard design",
             pb_doe.n_runs == 12 and bool(np.allclose(H_doe.T @ H_doe, 12 * np.eye(12))))
    st.check("DOE: rotatable CCD alpha = F^(1/4)",
             abs(_DoeCCD(3).generate().properties["alpha"] - 8 ** 0.25) < 1e-12)
    F_bb, _ = _DoeBB(3).generate().model_matrix("quadratic")
    st.check("DOE: Box-Behnken(3) supports the full quadratic model",
             int(np.linalg.matrix_rank(F_bb)) == 10)
    gl_doe = _DoeGL(4, random_state=0)
    gl_doe.generate()
    st.check("DOE: Graeco-Latin square of order 4 is orthogonal",
             len({(a, b) for a, b in zip(gl_doe.latin_.ravel(), gl_doe.greek_.ravel())}) == 16)
    d_doe = _DoeD(6, 1, model="quadratic", random_state=0).generate()
    st.check("DOE: D-optimal quadratic design puts two runs at each of -1, 0, 1",
             sorted(np.round(d_doe.points[:, 0], 9).tolist()) == [-1, -1, 0, 0, 1, 1])
    U_doe = np.array([[0.1, 0.2], [0.6, 0.9], [0.8, 0.4]])
    a_doe = np.abs(U_doe - 0.5)
    D_doe = np.abs(U_doe[:, None, :] - U_doe[None, :, :])
    cd_doe = ((13 / 12) ** 2 - 2 / 3 * np.prod(1 + a_doe / 2 - a_doe ** 2 / 2, 1).sum()
              + np.prod(1 + a_doe[:, None] / 2 + a_doe[None] / 2 - D_doe / 2, 2).sum() / 9)
    st.check("DOE: centred L2 discrepancy matches its closed form",
             abs(_DoeUD.discrepancy(U_doe, "CD") - cd_doe) < 1e-14)
    mm_doe = _DoeMM(8, 2, n_iter=300, random_state=0).generate()
    st.check("DOE: maximin LHD stays a Latin hypercube",
             all(sorted(np.floor(mm_doe.points[:, j] * 8).astype(int).tolist())
                 == list(range(8)) for j in range(2)))
    ccd_doe = _DoeCCD(2, center=3).generate().points
    rs_doe = _DoeRS(2).fit(ccd_doe, 5 - (ccd_doe[:, 0] - 0.3) ** 2 - (ccd_doe[:, 1] + 0.2) ** 2)
    st.check("DOE: response surface recovers an exact stationary point",
             bool(np.allclose(rs_doe.stationary_point(), [0.3, -0.2], atol=1e-10)))
    y_doe = np.array([45, 71, 48, 65, 68, 60, 80, 65, 43, 100, 45, 104, 75, 86, 70, 96],
                     dtype=float)
    d4_doe = _DoeFull(2, 4).generate()
    st.check("DOE: main effect A of Montgomery's 2^4 filtration data is 21.625",
             abs(_DoeME().fit(d4_doe, y_doe).effects_["A"] - 21.625) < 1e-12)
    st.check("DOE: Lenth's method flags A, C, D, AC, AD as active",
             set(_DoeNP().fit(d4_doe, y_doe).active_) == {"A", "C", "D", "AC", "AD"})
    so_doe = _DoeSobol(n_samples=4096, n_bootstrap=20, random_state=0).analyze(
        lambda X: np.sin(X[:, 0]) + 7 * np.sin(X[:, 1]) ** 2
        + 0.1 * X[:, 2] ** 4 * np.sin(X[:, 0]), bounds=[(-np.pi, np.pi)] * 3)
    st.check("DOE: Sobol indices of the Ishigami function match the analytic values",
             bool(np.allclose(so_doe.S1_, [0.3139, 0.4424, 0.0], atol=0.05)))
    pce_doe = _DoePCE(2, bounds=[(-1.0, 1.0)]).fit_function(lambda X: X[:, 0] ** 2)
    st.check("DOE: polynomial chaos gives the exact mean and variance of x^2",
             abs(pce_doe.mean_ - 1 / 3) < 1e-12 and abs(pce_doe.var_ - 4 / 45) < 1e-12)

    # CLI helpers: pure offline logic behind spl --version / spl update
    from stochpylib.cli_pypi import install_mode, update_available, version_key
    st.check("CLI: version_key numeric ordering",
             version_key("0.10.2") > version_key("0.9.9")
             and version_key("0.6.3") == (0, 6, 3))
    st.check("CLI: update_available statuses",
             update_available("0.6.3", {"latest": "0.7.0", "releases": []}) == "update"
             and update_available("0.7.0", {"latest": "0.7.0", "releases": []}) == "current"
             and update_available("0.8.0", {"latest": "0.7.0", "releases": []}) == "newer"
             and update_available("0.6.3", None) == "unknown")
    st.check("CLI: install_mode classifies", install_mode() in
             ("editable", "local", "wheel", "source"))

    if verbose:
        status = "OK" if not st.failures else f"FAILED ({len(st.failures)})"
        print(f"selftest: {st.count} checks, {status}")

    return len(st.failures)


if __name__ == "__main__":
    raise SystemExit(1 if run(verbose=True) else 0)
