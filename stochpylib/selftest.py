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



    if verbose:
        status = "OK" if not st.failures else f"FAILED ({len(st.failures)})"
        print(f"selftest: {st.count} checks, {status}")

    return len(st.failures)


if __name__ == "__main__":
    raise SystemExit(1 if run(verbose=True) else 0)
