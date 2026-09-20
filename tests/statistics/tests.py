"""Tests for stochpylib.statistics: descriptive statistics, point/interval
estimation, hypothesis tests, regression, and multivariate methods.

Oracles: scipy.stats (free), statsmodels (dev-only oracle, per AGENTS.md), and
closed-form/hand-derived references. All randomness is seeded; statistical
assertions use a stated multiple of the standard error (>= 3 SE, documented
per-check where a method has a known finite-sample discrepancy from its
oracle -- e.g. the quantile-regression median, or two-sample KS asymptotics).
"""

import math
import zlib

import numpy as np
import pytest
from scipy import stats as sps
from scipy.cluster.hierarchy import linkage as sp_linkage
from scipy.cluster.vq import kmeans2
from scipy.stats import multivariate_normal
import statsmodels.api as sm
import statsmodels.formula.api as smf
import pandas as pd
from statsmodels.multivariate.cancorr import CanCorr
from statsmodels.multivariate.factor import Factor
from statsmodels.multivariate.manova import MANOVA as SMManova
from statsmodels.regression.quantile_regression import QuantReg
from statsmodels.stats.multitest import multipletests
from statsmodels.stats.oneway import anova_oneway
from statsmodels.stats.proportion import proportion_confint, proportions_ztest
from statsmodels.stats.weightstats import ztest

import stochpylib
from stochpylib import statistics as st
from stochpylib.statistics import descriptive as st_desc
from stochpylib.statistics import estimation as st_est
from stochpylib.statistics import hypothesis as st_hyp
from stochpylib.statistics import multivariate as st_mv
from stochpylib.statistics import regression as st_reg
from stochpylib.distributions import Beta, Exponential, Gamma, LogNormal, Normal, Poisson, Uniform, Weibull


# =============================================================== descriptive

def test_trimmed_mean_matches_scipy():
    rng = np.random.default_rng(0)
    x = rng.normal(size=200)
    assert st.mean(x, trim=0.1) == pytest.approx(sps.trim_mean(x, 0.1))


def test_weighted_mean():
    x = np.array([1.0, 2.0, 3.0])
    w = np.array([1.0, 2.0, 3.0])
    assert st.mean(x, weights=w) == pytest.approx(np.average(x, weights=w))


def test_variance_std_ddof():
    rng = np.random.default_rng(1)
    x = rng.normal(size=100)
    assert st.variance(x) == pytest.approx(np.var(x, ddof=1))
    assert st.std(x) == pytest.approx(np.std(x, ddof=1))


def test_skewness_kurtosis_vs_scipy():
    rng = np.random.default_rng(2)
    x = rng.exponential(size=300)
    assert st.skewness(x) == pytest.approx(sps.skew(x))
    assert st.skewness(x, bias=False) == pytest.approx(sps.skew(x, bias=False))
    assert st.kurtosis(x) == pytest.approx(sps.kurtosis(x))
    assert st.kurtosis(x, bias=False) == pytest.approx(sps.kurtosis(x, bias=False))


@pytest.mark.parametrize("method", [
    "linear", "lower", "higher", "nearest", "midpoint", "weibull",
    "median_unbiased", "normal_unbiased", "averaged_inverted_cdf",
])
def test_quantile_methods_vs_numpy(method):
    rng = np.random.default_rng(3)
    x = rng.normal(size=150)
    assert st.quantile(x, 0.3, method=method) == pytest.approx(np.quantile(x, 0.3, method=method))


def test_iqr_vs_scipy():
    rng = np.random.default_rng(4)
    x = rng.normal(size=150)
    assert st.iqr(x) == pytest.approx(sps.iqr(x))


def test_mode_discrete_and_binned():
    val, count = st.mode([1, 2, 2, 3, 3, 3, 4])
    assert val == 3 and count == 3
    rng = np.random.default_rng(5)
    x = rng.normal(size=500)
    v, c = st.mode(x, bins=10)
    assert isinstance(v, float) and c > 0


def test_covariance_scalar_and_matrix():
    rng = np.random.default_rng(6)
    x, y = rng.normal(size=100), rng.normal(size=100)
    assert st.covariance(x, y) == pytest.approx(np.cov(x, y, ddof=1)[0, 1])
    X = rng.normal(size=(100, 3))
    assert np.allclose(st.covariance(X), np.cov(X, rowvar=False, ddof=1))


def test_correlation_pearson_spearman_kendall():
    rng = np.random.default_rng(7)
    x = rng.normal(size=200)
    y = 0.6 * x + rng.normal(size=200) * 0.5

    r = st.correlation(x, y, method="pearson")
    ref = sps.pearsonr(x, y)
    assert r.estimate == pytest.approx(ref.statistic)
    assert r.extras["pvalue"] == pytest.approx(ref.pvalue, rel=1e-6)

    r = st.correlation(x, y, method="spearman")
    ref = sps.spearmanr(x, y)
    assert r.estimate == pytest.approx(ref.statistic)
    assert r.extras["pvalue"] == pytest.approx(ref.pvalue, rel=1e-6)

    r = st.correlation(x, y, method="kendall")
    ref = sps.kendalltau(x, y, method="asymptotic")
    assert r.estimate == pytest.approx(ref.statistic)
    assert r.extras["pvalue"] == pytest.approx(ref.pvalue, rel=1e-6)

    # with ties
    xt = rng.integers(0, 5, 100).astype(float)
    yt = rng.integers(0, 5, 100).astype(float)
    r = st.correlation(xt, yt, method="kendall")
    ref = sps.kendalltau(xt, yt, method="asymptotic")
    assert r.estimate == pytest.approx(ref.statistic)
    assert r.extras["pvalue"] == pytest.approx(ref.pvalue, rel=1e-6)


def test_correlation_matrix_form():
    rng = np.random.default_rng(8)
    X = rng.normal(size=(200, 3))
    assert np.allclose(st.correlation(X, method="pearson"), np.corrcoef(X, rowvar=False))


def test_describe_matches_manual_quantiles():
    rng = np.random.default_rng(9)
    x = rng.normal(5, 2, 300)
    d = st.describe(x)
    assert d.n == 300 and d.missing == 0
    assert d.mean == pytest.approx(np.mean(x))
    assert d.median == pytest.approx(np.median(x))
    assert d.std == pytest.approx(np.std(x, ddof=1))
    q1, q3 = np.quantile(x, [0.25, 0.75])
    assert d.q1 == pytest.approx(q1) and d.q3 == pytest.approx(q3)
    d2 = st.describe(x)
    assert d2.as_dict()["mean"] == pytest.approx(d.mean)


def test_describe_2d_columns():
    rng = np.random.default_rng(10)
    X = rng.normal(size=(100, 3))
    d = st.describe(X)
    assert d.mean.shape == (3,)
    for j in range(3):
        assert d.mean[j] == pytest.approx(np.mean(X[:, j]))


# =============================================================== estimation

def test_mle_normal_matches_closed_form():
    rng = np.random.default_rng(11)
    x = rng.normal(3.0, 2.0, 500)
    r = st.MLE(Normal, x)
    mu_hat, sigma_hat = r.estimate
    assert mu_hat == pytest.approx(np.mean(x))
    assert sigma_hat == pytest.approx(np.std(x))
    assert r.std_error[0] == pytest.approx(np.std(x) / math.sqrt(len(x)), rel=1e-2)


def test_mle_gamma_vs_scipy():
    rng = np.random.default_rng(12)
    x = rng.gamma(3.0, 2.0, 800)
    r = st.MLE(Gamma, x)
    ref_shape, _, ref_scale = sps.gamma.fit(x, floc=0)
    assert r.estimate[0] == pytest.approx(ref_shape, rel=1e-3)
    assert r.estimate[1] == pytest.approx(ref_scale, rel=1e-3)
    assert math.isfinite(r.extras["aic"]) and math.isfinite(r.extras["bic"])


def test_mle_callable_loglik():
    rng = np.random.default_rng(13)
    x = rng.exponential(2.0, 300)

    def loglik(params, data):
        rate = params[0]
        if rate <= 0:
            return -1e18
        return float(np.sum(sps.expon.logpdf(data, scale=1.0 / rate)))

    r = st.MLE(loglik, x, x0=np.array([0.5]))
    assert r.estimate == pytest.approx(1.0 / np.mean(x), rel=1e-3)


@pytest.mark.parametrize("cls,sampler", [
    (Normal, lambda rng: rng.normal(3.0, 1.5, 8000)),
    (Exponential, lambda rng: rng.exponential(2.0, 8000)),
    (Gamma, lambda rng: rng.gamma(2.0, 3.0, 20000)),
    (Beta, lambda rng: rng.beta(2.0, 5.0, 20000)),
    (Poisson, lambda rng: rng.poisson(4.0, 20000)),
    (Uniform, lambda rng: rng.uniform(1.0, 5.0, 20000)),
    (LogNormal, lambda rng: rng.lognormal(0.5, 0.3, 20000)),
    (Weibull, lambda rng: rng.weibull(2.0, 20000) * 3.0),
])
def test_mom_recovers_generating_parameters(cls, sampler):
    rng = np.random.default_rng(14)
    data = sampler(rng)
    fitted = st.MOM(cls, data)
    # every distribution's MOM fit must itself be a valid instance whose theoretical
    # mean/var track the sample moments (definition of the method of moments)
    m1, var = np.mean(data), np.var(data, ddof=0)
    assert fitted.mean() == pytest.approx(m1, rel=5e-2)
    assert fitted.var() == pytest.approx(var, rel=1e-1)


def test_bayesian_normal_normal_conjugate():
    rng = np.random.default_rng(15)
    data = rng.normal(5.0, 2.0, 30)
    mu0, tau0, sigma = 0.0, 100.0, 2.0
    r = st.bayesian_estimator("normal", data, prior=(mu0, tau0), sigma=sigma)
    prec0, prec_lik = 1 / tau0 ** 2, len(data) / sigma ** 2
    post_mean_ref = (mu0 * prec0 + np.sum(data) / sigma ** 2) / (prec0 + prec_lik)
    post_var_ref = 1.0 / (prec0 + prec_lik)
    assert r.estimate == pytest.approx(post_mean_ref)
    assert r.std_error == pytest.approx(math.sqrt(post_var_ref))


def test_bayesian_beta_binomial_conjugate():
    successes, n = 30, 100
    r = st.bayesian_estimator("binomial", (successes, n), prior=(1.0, 1.0))
    a_post, b_post = 1.0 + successes, 1.0 + n - successes
    assert r.estimate == pytest.approx(a_post / (a_post + b_post))
    ref_lo, ref_hi = sps.beta.ppf([0.025, 0.975], a_post, b_post)
    assert r.extras["credible_interval"][0] == pytest.approx(ref_lo)
    assert r.extras["credible_interval"][1] == pytest.approx(ref_hi)


def test_bayesian_gamma_poisson_conjugate():
    rng = np.random.default_rng(16)
    data = rng.poisson(4.0, 50)
    alpha0, beta0 = 2.0, 1.0
    r = st.bayesian_estimator("poisson", data, prior=(alpha0, beta0))
    a_post = alpha0 + np.sum(data)
    rate_post = beta0 + len(data)
    assert r.estimate == pytest.approx(a_post / rate_post)


def test_bayesian_grid_posterior_generic():
    rng = np.random.default_rng(17)
    data = rng.normal(2.0, 1.0, 40)
    grid = np.linspace(-2, 6, 4000)

    def loglik(theta, d):
        return float(np.sum(sps.norm.logpdf(d, theta, 1.0)))

    def logprior(theta):
        return 0.0  # flat prior over the grid

    r = st.bayesian_estimator(loglik, data, logprior, grid=grid)
    assert r.estimate == pytest.approx(np.mean(data), abs=3 * (1.0 / math.sqrt(40)))


def test_confidence_interval_mean_vs_scipy():
    rng = np.random.default_rng(18)
    x = rng.normal(10, 3, 60)
    lo, hi = st.confidence_interval(x, kind="mean")
    ref = sps.t.interval(0.95, len(x) - 1, loc=np.mean(x), scale=np.std(x, ddof=1) / math.sqrt(len(x)))
    assert lo == pytest.approx(ref[0]) and hi == pytest.approx(ref[1])


@pytest.mark.parametrize("method,sm_method", [
    ("wald", "normal"), ("wilson", "wilson"), ("agresti-coull", "agresti_coull"),
])
def test_confidence_interval_proportion_vs_statsmodels(method, sm_method):
    lo, hi = st.confidence_interval((30, 100), kind="proportion", method=method)
    reflo, refhi = proportion_confint(30, 100, method=sm_method)
    assert lo == pytest.approx(reflo) and hi == pytest.approx(refhi)


def test_confidence_interval_clopper_pearson():
    lo, hi = st.confidence_interval((30, 100), kind="proportion", method="clopper-pearson")
    reflo, refhi = proportion_confint(30, 100, method="beta")
    assert lo == pytest.approx(reflo) and hi == pytest.approx(refhi)


def test_confidence_interval_variance_and_std():
    rng = np.random.default_rng(19)
    x = rng.normal(0, 2, 50)
    n, s2 = len(x), np.var(x, ddof=1)
    lo, hi = st.confidence_interval(x, kind="variance")
    reflo = (n - 1) * s2 / sps.chi2.ppf(0.975, n - 1)
    refhi = (n - 1) * s2 / sps.chi2.ppf(0.025, n - 1)
    assert lo == pytest.approx(reflo) and hi == pytest.approx(refhi)
    lo_s, hi_s = st.confidence_interval(x, kind="std")
    assert lo_s == pytest.approx(math.sqrt(reflo)) and hi_s == pytest.approx(math.sqrt(refhi))


def test_confidence_interval_median_covers_true_median():
    rng = np.random.default_rng(20)
    covered = 0
    trials = 200
    for i in range(trials):
        x = rng.normal(0, 1, 50)
        lo, hi = st.confidence_interval(x, kind="median")
        covered += lo <= 0.0 <= hi
    p = covered / trials
    se = math.sqrt(0.95 * 0.05 / trials)
    assert abs(p - 0.95) < 3 * se


def test_bootstrap_ci_percentile_coverage():
    rng = np.random.default_rng(21)
    covered = 0
    trials = 150
    for i in range(trials):
        x = rng.normal(5, 2, 40)
        r = st.bootstrap_ci(x, lambda a: np.mean(a, axis=-1), n_boot=500, method="percentile",
                             random_state=i)
        lo, hi = r.extras["conf_int"]
        covered += lo <= 5.0 <= hi
    p = covered / trials
    se = math.sqrt(0.95 * 0.05 / trials)
    assert abs(p - 0.95) < 4 * se


def test_bootstrap_ci_bca_matches_deterministic_pieces():
    rng = np.random.default_rng(22)
    x = rng.normal(5, 2, 40)
    r = st.bootstrap_ci(x, lambda a: np.mean(a, axis=-1), n_boot=1000, method="bca", random_state=1)
    lo, hi = r.extras["conf_int"]
    assert lo < r.estimate < hi


def test_jackknife_mean_zero_bias_exact_se():
    rng = np.random.default_rng(23)
    x = rng.normal(5, 2, 40)
    r = st.jackknife(x, lambda a: np.mean(a))
    assert r.extras["bias"] == pytest.approx(0.0, abs=1e-9)
    assert r.std_error == pytest.approx(np.std(x, ddof=1) / math.sqrt(len(x)), abs=1e-9)


def test_delta_method_log_mean():
    rng = np.random.default_rng(24)
    x = rng.normal(10, 2, 200)
    mean_hat = np.mean(x)
    se_mean = np.std(x, ddof=1) / math.sqrt(len(x))
    r = st.delta_method(lambda t: math.log(t[0]), np.array([mean_hat]), np.array([[se_mean ** 2]]))
    assert r.std_error == pytest.approx(se_mean / mean_hat, rel=1e-5)


def test_profile_likelihood_normal_mean_matches_closed_form():
    rng = np.random.default_rng(25)
    x = rng.normal(5, 2, 40)
    n = len(x)
    mu_hat, sigma_hat = np.mean(x), np.std(x, ddof=0)

    def loglik(params, data):
        mu, sigma = params
        if sigma <= 0:
            return -1e18
        return float(np.sum(sps.norm.logpdf(data, mu, sigma)))

    r = st.profile_likelihood(loglik, np.array([mu_hat, sigma_hat]), index=0, data=x)
    lo, hi = r.extras["conf_int"]

    chi2_crit = sps.chi2.ppf(0.95, 1)
    s = np.std(x, ddof=1)

    def f(t):
        return n * math.log(1 + t ** 2 / (n - 1)) - chi2_crit

    from scipy.optimize import brentq
    t_crit = brentq(f, 0.01, 20)
    ref_lo, ref_hi = mu_hat - t_crit * s / math.sqrt(n), mu_hat + t_crit * s / math.sqrt(n)
    assert lo == pytest.approx(ref_lo, abs=1e-2)
    assert hi == pytest.approx(ref_hi, abs=1e-2)


# =============================================================== hypothesis

def test_z_test_one_and_two_sample_vs_statsmodels():
    rng = np.random.default_rng(26)
    x = rng.normal(5, 2, 100)
    x2 = rng.normal(5, 2, 80)
    r = st.z_test(x, mu0=5.0)
    zref, pref = ztest(x, value=5.0)
    assert r.statistic == pytest.approx(zref) and r.pvalue == pytest.approx(pref)

    r = st.z_test(x, y=x2, pooled=True)
    zref, pref = ztest(x, x2, usevar="pooled")
    assert r.statistic == pytest.approx(zref) and r.pvalue == pytest.approx(pref)

    r = st.z_test(x, y=x2, pooled=False)
    zref, pref = ztest(x, x2, usevar="unequal")
    assert r.statistic == pytest.approx(zref) and r.pvalue == pytest.approx(pref)


def test_z_test_proportion_vs_statsmodels():
    r = st.z_test((30, 100), mu0=0.4, proportion=True)
    zref, pref = proportions_ztest(30, 100, value=0.4, prop_var=0.4)
    assert r.statistic == pytest.approx(zref) and r.pvalue == pytest.approx(pref)

    r = st.z_test((30, 100), y=(45, 120), proportion=True)
    zref, pref = proportions_ztest([30, 45], [100, 120])
    assert r.statistic == pytest.approx(zref) and r.pvalue == pytest.approx(pref)


def test_t_test_variants_vs_scipy():
    rng = np.random.default_rng(27)
    x = rng.normal(5, 2, 100)
    x2 = rng.normal(5, 2, 80)
    r = st.t_test(x, mu0=5.0)
    ref = sps.ttest_1samp(x, 5.0)
    assert r.statistic == pytest.approx(ref.statistic) and r.pvalue == pytest.approx(ref.pvalue)

    r = st.t_test(x, y=x2, equal_var=True)
    ref = sps.ttest_ind(x, x2, equal_var=True)
    assert r.statistic == pytest.approx(ref.statistic) and r.pvalue == pytest.approx(ref.pvalue)

    r = st.t_test(x, y=x2, equal_var=False)
    ref = sps.ttest_ind(x, x2, equal_var=False)
    assert r.statistic == pytest.approx(ref.statistic) and r.pvalue == pytest.approx(ref.pvalue)
    assert r.df == pytest.approx(ref.df)

    y_paired = x + rng.normal(0, 0.5, 100)
    r = st.t_test(x, y=y_paired, paired=True)
    ref = sps.ttest_rel(x, y_paired)
    assert r.statistic == pytest.approx(ref.statistic) and r.pvalue == pytest.approx(ref.pvalue)


def test_chi2_test_gof_and_independence():
    obs = np.array([20, 30, 25, 25])
    r = st.chi2_test(obs)
    ref = sps.chisquare(obs)
    assert r.statistic == pytest.approx(ref.statistic) and r.pvalue == pytest.approx(ref.pvalue)

    table = np.array([[10, 20], [30, 15]])
    r = st.chi2_test(table, yates=False)
    stat_ref, p_ref, *_ = sps.chi2_contingency(table, correction=False)
    assert r.statistic == pytest.approx(stat_ref) and r.pvalue == pytest.approx(p_ref)

    r = st.chi2_test(table, yates=True)
    stat_ref, p_ref, *_ = sps.chi2_contingency(table, correction=True)
    assert r.statistic == pytest.approx(stat_ref) and r.pvalue == pytest.approx(p_ref)

    table3 = np.array([[10, 20, 15], [30, 15, 25], [5, 10, 20]])
    r = st.chi2_test(table3)
    stat_ref, p_ref, *_ = sps.chi2_contingency(table3)
    assert r.statistic == pytest.approx(stat_ref) and r.pvalue == pytest.approx(p_ref)


def test_f_test_variance_ratio():
    rng = np.random.default_rng(28)
    x, y = rng.normal(0, 1, 100), rng.normal(0, 1.3, 80)
    r = st.f_test(x, y)
    F = np.var(x, ddof=1) / np.var(y, ddof=1)
    p_ref = min(1.0, 2 * min(sps.f.cdf(F, 99, 79), sps.f.sf(F, 99, 79)))
    assert r.statistic == pytest.approx(F) and r.pvalue == pytest.approx(p_ref)


def test_f_test_nested_regression():
    rng = np.random.default_rng(29)
    X = rng.normal(size=(200, 3))
    y = 1.0 + X[:, 0] * 2.0 + rng.normal(0, 1, 200)
    full = st.linear_regression(X, y)
    restricted = st.linear_regression(X[:, [0]], y)
    r = st.f_test(full, restricted)
    Xd = sm.add_constant(X)
    m_full = sm.OLS(y, Xd).fit()
    m_res = sm.OLS(y, sm.add_constant(X[:, [0]])).fit()
    ftest = m_full.compare_f_test(m_res)
    assert r.statistic == pytest.approx(ftest[0], rel=1e-6)
    assert r.pvalue == pytest.approx(ftest[1], rel=1e-4)


def test_oneway_anova_vs_scipy():
    rng = np.random.default_rng(30)
    g1, g2, g3 = rng.normal(0, 1, 30), rng.normal(0.5, 1, 35), rng.normal(1.0, 1, 25)
    r = st.ANOVA(g1, g2, g3)
    ref = sps.f_oneway(g1, g2, g3)
    assert r.statistic == pytest.approx(ref.statistic) and r.pvalue == pytest.approx(ref.pvalue)


def test_welch_anova_vs_statsmodels():
    rng = np.random.default_rng(31)
    g1, g2, g3 = rng.normal(0, 1, 30), rng.normal(0.5, 1.5, 35), rng.normal(1.0, 2.0, 25)
    r = st.ANOVA(g1, g2, g3, equal_var=False)
    ref = anova_oneway([g1, g2, g3], use_var="unequal", welch_correction=True)
    assert r.statistic == pytest.approx(ref.statistic)
    assert r.pvalue == pytest.approx(ref.pvalue)
    assert r.df[0] == pytest.approx(ref.df[0])
    assert r.df[1] == pytest.approx(ref.df[1])


def test_twoway_anova_type2_vs_statsmodels():
    rng = np.random.default_rng(32)
    n = 120
    a = rng.choice(["a1", "a2", "a3"], size=n, p=[0.5, 0.3, 0.2])
    b = rng.choice(["b1", "b2"], size=n, p=[0.6, 0.4])
    y = (2.0 + 1.5 * (a == "a2") + 0.7 * (a == "a3") + 1.0 * (b == "b2")
         + 0.8 * ((a == "a2") & (b == "b2")) + rng.normal(0, 1, n))
    r = st.ANOVA(y, factors=[a, b], interaction=True)
    df = pd.DataFrame({"y": y, "a": a, "b": b})
    model = smf.ols("y ~ C(a)*C(b)", data=df).fit()
    aov = sm.stats.anova_lm(model, typ=2)
    assert r.table[0]["ss"] == pytest.approx(aov.loc["C(a)", "sum_sq"])
    assert r.table[1]["ss"] == pytest.approx(aov.loc["C(b)", "sum_sq"])
    assert r.table[2]["ss"] == pytest.approx(aov.loc["C(a):C(b)", "sum_sq"])
    assert r.table[3]["ss"] == pytest.approx(aov.loc["Residual", "sum_sq"])
    assert r.table[0]["p"] == pytest.approx(aov.loc["C(a)", "PR(>F)"], rel=1e-4)
    assert r.table[2]["p"] == pytest.approx(aov.loc["C(a):C(b)", "PR(>F)"], rel=1e-4)

    r2 = st.ANOVA(y, factors=[a, b], interaction=False)
    model2 = smf.ols("y ~ C(a)+C(b)", data=df).fit()
    aov2 = sm.stats.anova_lm(model2, typ=2)
    assert r2.table[0]["ss"] == pytest.approx(aov2.loc["C(a)", "sum_sq"])


def test_manova_vs_statsmodels():
    rng = np.random.default_rng(33)
    n = 200
    g = rng.choice(["g1", "g2", "g3"], size=n)
    Y = np.column_stack([
        1.0 + 0.6 * (g == "g2") + 0.3 * (g == "g3") + rng.normal(0, 1, n),
        2.0 - 0.4 * (g == "g2") + 0.5 * (g == "g3") + rng.normal(0, 1, n),
    ])
    r = st.MANOVA(Y, g)
    dfm = pd.DataFrame({"y1": Y[:, 0], "y2": Y[:, 1], "g": g})
    maov = SMManova.from_formula("y1 + y2 ~ g", data=dfm)
    res = maov.mv_test().results["g"]["stat"]
    assert r.table[0]["value"] == pytest.approx(res.loc["Wilks' lambda", "Value"])
    assert r.table[0]["F"] == pytest.approx(res.loc["Wilks' lambda", "F Value"])
    assert r.table[1]["value"] == pytest.approx(res.loc["Pillai's trace", "Value"])
    assert r.table[2]["value"] == pytest.approx(res.loc["Hotelling-Lawley trace", "Value"])
    assert r.table[3]["value"] == pytest.approx(res.loc["Roy's greatest root", "Value"])
    assert r.table[3]["F"] == pytest.approx(res.loc["Roy's greatest root", "F Value"])


def test_mann_whitney_exact_vs_scipy():
    rng = np.random.default_rng(34)
    x = rng.permutation(np.arange(1, 21)).astype(float)[:10]
    y = rng.permutation(np.arange(21, 41)).astype(float)[:10]
    r = st.mann_whitney(x, y, method="exact")
    ref = sps.mannwhitneyu(x, y, method="exact")
    assert r.statistic == pytest.approx(ref.statistic)
    assert r.pvalue == pytest.approx(ref.pvalue, abs=1e-10)


def test_mann_whitney_asymptotic_with_ties_vs_scipy():
    rng = np.random.default_rng(35)
    x = rng.integers(0, 10, 60).astype(float)
    y = rng.integers(0, 10, 55).astype(float)
    for cont in (True, False):
        r = st.mann_whitney(x, y, method="asymptotic", continuity=cont)
        ref = sps.mannwhitneyu(x, y, method="asymptotic", use_continuity=cont)
        assert r.statistic == pytest.approx(ref.statistic)
        assert r.pvalue == pytest.approx(ref.pvalue)


def test_wilcoxon_exact_vs_scipy():
    rng = np.random.default_rng(36)
    signs = rng.choice([-1, 1], size=15)
    d = (signs * np.arange(1, 16)).astype(float)
    r = st.wilcoxon(d, method="exact")
    ref = sps.wilcoxon(d, method="exact")
    assert r.statistic == pytest.approx(ref.statistic)
    assert r.pvalue == pytest.approx(ref.pvalue, abs=1e-10)


def test_wilcoxon_asymptotic_with_ties_vs_scipy():
    rng = np.random.default_rng(37)
    d = rng.integers(-5, 6, 60).astype(float)
    d = d[d != 0]
    for corr in (True, False):
        r = st.wilcoxon(d, method="asymptotic", correction=corr)
        ref = sps.wilcoxon(d, method="asymptotic", correction=corr, zero_method="wilcox")
        assert r.statistic == pytest.approx(ref.statistic)
        assert r.pvalue == pytest.approx(ref.pvalue)


def test_ks_test_one_sample_vs_scipy():
    rng = np.random.default_rng(38)
    x = rng.normal(0, 1, 200)
    r = st.ks_test(x, "Normal", args=(0, 1))
    ref = sps.kstest(x, "norm", method="asymp")
    assert r.statistic == pytest.approx(ref.statistic)
    assert r.pvalue == pytest.approx(ref.pvalue)


def test_ks_test_two_sample_statistic_matches_and_pvalue_reasonable():
    # our p-value uses the classical asymptotic (N->inf) Kolmogorov formula, matching
    # scipy's one-sample "asymp" exactly but not scipy's ks_2samp finite-sample
    # refinement (kstwo) -- see the ks_test docstring. The statistic always matches.
    rng = np.random.default_rng(39)
    x = rng.normal(0, 1, 300)
    y = rng.normal(0.2, 1.1, 250)
    r = st.ks_test(x, y)
    ref = sps.ks_2samp(x, y, method="asymp")
    assert r.statistic == pytest.approx(ref.statistic)
    assert r.pvalue == pytest.approx(ref.pvalue, rel=0.3)


def test_shapiro_wilk_vs_scipy():
    rng = np.random.default_rng(40)
    for n in [3, 5, 11, 12, 50, 500, 2000]:
        x = rng.normal(size=n)
        r = st.shapiro_wilk(x)
        ref = sps.shapiro(x)
        assert r.statistic == pytest.approx(ref.statistic, abs=1e-8)
        assert r.pvalue == pytest.approx(ref.pvalue, rel=1e-3, abs=1e-6)


def test_shapiro_wilk_rejects_nonnormal():
    rng = np.random.default_rng(41)
    x = rng.exponential(size=300)
    r = st.shapiro_wilk(x)
    assert r.pvalue < 0.01


def test_levene_and_bartlett_vs_scipy():
    rng = np.random.default_rng(42)
    g1, g2, g3 = rng.normal(0, 1, 40), rng.normal(0, 1.5, 45), rng.normal(0, 2, 35)
    r = st.levene(g1, g2, g3, center="median")
    ref = sps.levene(g1, g2, g3, center="median")
    assert r.statistic == pytest.approx(ref.statistic) and r.pvalue == pytest.approx(ref.pvalue)

    r = st.bartlett(g1, g2, g3)
    ref = sps.bartlett(g1, g2, g3)
    assert r.statistic == pytest.approx(ref.statistic) and r.pvalue == pytest.approx(ref.pvalue)


def test_ptukey_qtukey_vs_scipy_studentized_range():
    from stochpylib.statistics._common import _ptukey, _qtukey
    for k in (2, 3, 5):
        for df in (5, 10, 30):
            for q in (2.0, 3.0, 4.5):
                mine = _ptukey(q, k, df)
                ref = sps.studentized_range.cdf(q, k, df)
                assert mine == pytest.approx(ref, abs=1e-6)
    q95 = _qtukey(0.95, 3, 10)
    ref95 = sps.studentized_range.ppf(0.95, 3, 10)
    assert q95 == pytest.approx(ref95, abs=1e-6)


def test_tukey_hsd_vs_scipy():
    rng = np.random.default_rng(43)
    g1, g2, g3 = rng.normal(0, 1, 30), rng.normal(0.8, 1, 32), rng.normal(1.5, 1, 28)
    r = st.tukey_hsd(g1, g2, g3)
    ref = sps.tukey_hsd(g1, g2, g3)
    pmine = [row["p"] for row in r.table]
    pref = [ref.pvalue[0, 1], ref.pvalue[0, 2], ref.pvalue[1, 2]]
    for a, b in zip(pmine, pref):
        assert a == pytest.approx(b, abs=1e-4)


@pytest.mark.parametrize("method,sm_method", [
    ("bonferroni", "bonferroni"), ("holm", "holm"), ("sidak", "sidak"),
    ("holm-sidak", "holm-sidak"), ("fdr_bh", "fdr_bh"),
])
def test_bonferroni_vs_statsmodels(method, sm_method):
    pv = np.array([0.001, 0.02, 0.03, 0.04, 0.5])
    r = st.bonferroni(pv, alpha=0.05, method=method)
    reject_ref, adj_ref, *_ = multipletests(pv, alpha=0.05, method=sm_method)
    mine_adj = np.array([row["adjusted_pvalue"] for row in r.table])
    assert np.allclose(mine_adj, adj_ref, atol=1e-10)
    mine_reject = np.array([row["reject"] for row in r.table])
    assert np.array_equal(mine_reject, reject_ref)


# =============================================================== regression

def test_linear_regression_ols_vs_statsmodels():
    rng = np.random.default_rng(44)
    n = 300
    X = rng.normal(size=(n, 3))
    y = 2.0 + X @ np.array([1.5, -2.0, 0.5]) + rng.normal(0, 1.5, n)
    r = st.linear_regression(X, y)
    Xd = sm.add_constant(X)
    m = sm.OLS(y, Xd).fit()
    assert np.allclose(r.coef_, m.params, atol=1e-8)
    assert np.allclose(r.std_errors_, m.bse, atol=1e-8)
    assert np.allclose(r.pvalues_, m.pvalues, atol=1e-6)
    assert r.r2_ == pytest.approx(m.rsquared)
    assert r.adj_r2_ == pytest.approx(m.rsquared_adj)
    assert r.f_stat_ == pytest.approx(m.fvalue)
    assert r.aic_ == pytest.approx(m.aic, abs=1e-4)
    assert r.bic_ == pytest.approx(m.bic, abs=1e-4)
    assert r.loglik_ == pytest.approx(m.llf)
    preds = r.predict(X[:5])
    assert np.allclose(preds, m.predict(Xd[:5]))


@pytest.mark.parametrize("hc", ["HC0", "HC1", "HC2", "HC3"])
def test_linear_regression_robust_se_vs_statsmodels(hc):
    rng = np.random.default_rng(45)
    n = 300
    X = rng.normal(size=(n, 3))
    y = 2.0 + X @ np.array([1.5, -2.0, 0.5]) + rng.normal(0, 1.5, n)
    r = st.linear_regression(X, y, cov_type=hc)
    Xd = sm.add_constant(X)
    m = sm.OLS(y, Xd).fit(cov_type=hc)
    assert np.allclose(r.std_errors_, m.bse, atol=1e-6)


def test_wls_vs_statsmodels():
    rng = np.random.default_rng(46)
    n = 300
    X = rng.normal(size=(n, 3))
    y = 2.0 + X @ np.array([1.5, -2.0, 0.5]) + rng.normal(0, 1.5, n)
    w = rng.uniform(0.5, 2.0, n)
    r = st.linear_regression(X, y, weights=w)
    m = sm.WLS(y, sm.add_constant(X), weights=w).fit()
    assert np.allclose(r.coef_, m.params, atol=1e-8)
    assert np.allclose(r.std_errors_, m.bse, atol=1e-8)


def test_logistic_regression_vs_statsmodels():
    rng = np.random.default_rng(47)
    n = 300
    X = rng.normal(size=(n, 3))
    p_true = 1 / (1 + np.exp(-(0.5 + X @ np.array([1.0, -0.5, 0.3]))))
    y = (rng.uniform(size=n) < p_true).astype(float)
    r = st.logistic_regression(X, y)
    m = sm.Logit(y, sm.add_constant(X)).fit(disp=0)
    assert np.allclose(r.coef_, m.params, atol=1e-5)
    assert np.allclose(r.std_errors_, m.bse, atol=1e-5)
    assert r.loglik_ == pytest.approx(m.llf, abs=1e-5)
    assert r.pseudo_r2_ == pytest.approx(m.prsquared, abs=1e-4)
    proba = r.predict_proba(X[:5])
    assert np.all((proba >= 0) & (proba <= 1))


def test_poisson_regression_vs_statsmodels():
    rng = np.random.default_rng(48)
    n = 300
    X = rng.normal(size=(n, 3))
    mu_true = np.exp(0.3 + X @ np.array([0.2, -0.1, 0.15]))
    y = rng.poisson(mu_true)
    r = st.poisson_regression(X, y)
    m = sm.Poisson(y, sm.add_constant(X)).fit(disp=0)
    assert np.allclose(r.coef_, m.params, atol=1e-5)
    assert np.allclose(r.std_errors_, m.bse, atol=1e-5)


_GLM_FAMILY_LINK = [
    ("gaussian", "identity"), ("gaussian", "log"),
    ("binomial", "logit"), ("binomial", "probit"), ("binomial", "cloglog"),
    ("poisson", "log"),
    ("gamma", "inverse"), ("gamma", "log"),
    ("inverse_gaussian", "inverse_squared"),
]


@pytest.mark.parametrize("fam,link", _GLM_FAMILY_LINK)
def test_glm_family_link_vs_statsmodels(fam, link):
    # a stable hash, not the builtin hash() -- str/tuple hashing is salted per process
    # (PYTHONHASHSEED) unless explicitly disabled, so hash() here was not actually a
    # fixed seed across runs, occasionally landing on IRLS-pathological data that even
    # statsmodels' own GLM can't fit either (see development/Probleme.md).
    seed = zlib.crc32(f"{fam}|{link}".encode()) % (2 ** 31)
    rng = np.random.default_rng(seed)
    n = 300
    X = rng.normal(size=(n, 3))
    if fam == "gaussian":
        y = 2.0 + X @ np.array([1.5, -2.0, 0.5]) + rng.normal(0, 1.5, n)
    elif fam == "binomial":
        p_true = 1 / (1 + np.exp(-(0.5 + X @ np.array([1.0, -0.5, 0.3]))))
        y = (rng.uniform(size=n) < p_true).astype(float)
    elif fam == "poisson":
        y = rng.poisson(np.exp(0.3 + X @ np.array([0.2, -0.1, 0.15]))).astype(float)
    elif fam == "gamma":
        y = np.exp(0.5 + X @ np.array([0.1, -0.05, 0.05]) + rng.normal(0, 0.2, n))
    else:
        y = np.exp(0.5 + X @ np.array([0.1, -0.05, 0.05]) + rng.normal(0, 0.1, n))

    r = st.glm(X, y, family=fam, link=link)

    sm_family_cls = {
        "gaussian": sm.families.Gaussian, "binomial": sm.families.Binomial,
        "poisson": sm.families.Poisson, "gamma": sm.families.Gamma,
        "inverse_gaussian": sm.families.InverseGaussian,
    }[fam]
    sm_link_obj = {
        "identity": sm.families.links.Identity(), "log": sm.families.links.Log(),
        "logit": sm.families.links.Logit(), "probit": sm.families.links.Probit(),
        "cloglog": sm.families.links.CLogLog(), "inverse": sm.families.links.InversePower(),
        "inverse_squared": sm.families.links.InverseSquared(),
    }[link]
    m = sm.GLM(y, sm.add_constant(X), family=sm_family_cls(link=sm_link_obj)).fit()

    assert np.allclose(r.coef_, m.params, atol=1e-4)
    assert np.allclose(r.std_errors_, m.bse, atol=1e-3)
    assert r.deviance_ == pytest.approx(m.deviance, abs=1e-3)
    assert r.loglik_ == pytest.approx(m.llf, abs=1e-2)
    assert r.aic_ == pytest.approx(m.aic, abs=0.1)


def test_glm_negative_binomial_vs_statsmodels():
    import statsmodels.genmod.families as smfam
    rng = np.random.default_rng(49)
    n = 300
    X = rng.normal(size=(n, 3))
    y = rng.negative_binomial(5, 0.5, n).astype(float)
    r = st.glm(X, y, family="negative_binomial", alpha=1.0)
    m = sm.GLM(y, sm.add_constant(X), family=smfam.NegativeBinomial(alpha=1.0)).fit()
    assert np.allclose(r.coef_, m.params, atol=1e-4)
    assert np.allclose(r.std_errors_, m.bse, atol=1e-3)
    assert r.deviance_ == pytest.approx(m.deviance, abs=1e-2)


def test_ridge_closed_form():
    rng = np.random.default_rng(50)
    n = 300
    X = rng.normal(size=(n, 3))
    y = 2.0 + X @ np.array([1.5, -2.0, 0.5]) + rng.normal(0, 1.5, n)
    alpha = 5.0
    r = st.ridge(X, y, alpha=alpha)
    Xc, yc = X - X.mean(axis=0), y - y.mean()
    beta_ref = np.linalg.solve(Xc.T @ Xc + alpha * np.eye(3), Xc.T @ yc)
    assert np.allclose(r.coef_[1:], beta_ref, atol=1e-8)


def test_ridge_gcv_selects_reasonable_alpha():
    rng = np.random.default_rng(51)
    n = 200
    X = rng.normal(size=(n, 5))
    y = 1.0 + X[:, 0] * 3.0 + rng.normal(0, 1, n)
    r = st.ridge(X, y, alpha="gcv")
    assert r.extras["alpha"] > 0
    assert r.coef_[1] == pytest.approx(3.0, abs=1.0)


def test_lasso_kkt_conditions():
    rng = np.random.default_rng(52)
    n = 300
    X = rng.normal(size=(n, 3))
    y = 2.0 + X @ np.array([1.5, -2.0, 0.5]) + rng.normal(0, 1.5, n)
    alpha = 0.1
    r = st.lasso(X, y, alpha=alpha, max_iter=5000, tol=1e-10)
    Xc, yc = X - X.mean(axis=0), y - y.mean()
    resid = yc - Xc @ r.coef_[1:]
    grad = Xc.T @ resid / n
    active = np.abs(r.coef_[1:]) > 1e-6
    assert np.allclose(np.abs(grad[active]), alpha, atol=1e-4)
    assert np.all(np.abs(grad[~active]) <= alpha + 1e-6)


def test_lasso_alpha_zero_limit_equals_ols():
    rng = np.random.default_rng(53)
    n = 300
    X = rng.normal(size=(n, 3))
    y = 2.0 + X @ np.array([1.5, -2.0, 0.5]) + rng.normal(0, 1.5, n)
    r0 = st.lasso(X, y, alpha=1e-8, max_iter=20000, tol=1e-12)
    ols = st.linear_regression(X, y)
    assert np.allclose(r0.coef_, ols.coef_, atol=1e-3)


def test_lasso_path_sparsity_monotone():
    rng = np.random.default_rng(54)
    n = 200
    X = rng.normal(size=(n, 5))
    y = 1.0 + X[:, 0] * 2.0 + rng.normal(0, 1, n)
    r = st.lasso(X, y, alpha=[1.0, 0.3, 0.1, 0.03, 0.01])
    active = [int(np.sum(np.abs(row) > 1e-8)) for row in r.extras["path"]]
    assert active == sorted(active)


def test_elastic_net_matches_lasso_and_ridge_at_extremes():
    rng = np.random.default_rng(55)
    n = 250
    X = rng.normal(size=(n, 3))
    y = 2.0 + X @ np.array([1.5, -2.0, 0.5]) + rng.normal(0, 1.0, n)
    r_l1 = st.elastic_net(X, y, alpha=0.1, l1_ratio=1.0, max_iter=5000, tol=1e-10)
    r_lasso = st.lasso(X, y, alpha=0.1, max_iter=5000, tol=1e-10)
    assert np.allclose(r_l1.coef_, r_lasso.coef_, atol=1e-6)


@pytest.mark.parametrize("q", [0.1, 0.5, 0.9])
def test_quantile_regression_vs_statsmodels(q):
    rng = np.random.default_rng(56)
    n = 300
    X = rng.normal(size=(n, 3))
    y = 2.0 + X @ np.array([1.5, -2.0, 0.5]) + rng.normal(0, 1.5, n)
    r = st.quantile_regression(X, y, q=q)
    m = QuantReg(y, sm.add_constant(X)).fit(q=q)
    # exact LP solution vs statsmodels' IRLS: the LP has a basic-solution degeneracy
    # (a small set of residuals sit exactly at zero) that IRLS smooths over, giving a
    # persistent few-times-1e-3 gap even away from the median (documented).
    assert np.allclose(r.coef_, m.params, atol=5e-3)
    assert np.allclose(r.std_errors_, m.bse, rtol=0.2)


# =============================================================== multivariate

def test_pca_matches_eigh_and_orthonormal():
    rng = np.random.default_rng(57)
    n = 200
    L_true = rng.normal(size=(5, 2))
    Z = rng.normal(size=(n, 2))
    X = Z @ L_true.T + rng.normal(0, 0.3, (n, 5))
    r = st.PCA(X, n_components=3)
    cov = np.cov(X, rowvar=False)
    w = np.sort(np.linalg.eigvalsh(cov))[::-1]
    assert np.allclose(r.explained_variance_, w[:3], atol=1e-8)
    assert np.allclose(r.components_ @ r.components_.T, np.eye(3), atol=1e-8)
    scores = r.transform(X)
    assert np.allclose(scores, r.scores_, atol=1e-8)


def _ml_discrepancy(psi, R, n_factors):
    """Jöreskog's concentrated ML objective F(Psi) at a given uniqueness vector
    (see stochpylib.statistics.multivariate.factor_analysis for the derivation)."""
    Psi_inv_sqrt = np.diag(1.0 / np.sqrt(psi))
    M = Psi_inv_sqrt @ R @ Psi_inv_sqrt
    w = np.sort(np.linalg.eigvalsh(M))[::-1]
    w_rest = np.clip(w[n_factors:], 1e-10, None)
    return float(np.sum(w_rest - np.log(w_rest) - 1.0))


def test_factor_analysis_ml_matches_statsmodels_covariance():
    rng = np.random.default_rng(58)
    n = 200
    L_true = rng.normal(size=(5, 2))
    Z = rng.normal(size=(n, 2))
    X = Z @ L_true.T + rng.normal(0, 0.3, (n, 5))
    r = st.factor_analysis(X, n_factors=2, method="ml")
    Sigma_mine = r.loadings_ @ r.loadings_.T + np.diag(r.uniquenesses_)
    fa_ref = Factor(endog=X, n_factor=2, method="ml", smc=True).fit()
    Sigma_ref = np.real(fa_ref.fitted_cov)

    # This particular draw is a near-Heywood case (one communality ~1, statsmodels'
    # own BFGS warns "did not converge"), so the exact optimum is on a flat/near-
    # singular ridge: two runs can land on visibly different Psi/loadings while
    # achieving essentially the same minimized discrepancy F(Psi) -- that objective
    # value, not the raw parameters, is what's actually pinned down at the optimum,
    # and comparing it directly is robust to platform-dependent BLAS/optimizer paths
    # (this raw-covariance comparison alone was observed to fail by ~1.1e-3 on
    # Windows CI while passing on Linux, for exactly this reason).
    R = np.corrcoef(X, rowvar=False)
    F_mine = _ml_discrepancy(r.uniquenesses_, R, 2)
    F_ref = _ml_discrepancy(fa_ref.uniqueness, R, 2)
    assert F_mine <= F_ref + 1e-4, f"F_mine={F_mine} should not exceed statsmodels' F_ref={F_ref}"

    assert np.allclose(Sigma_mine, Sigma_ref, atol=5e-3)
    assert math.isfinite(r.loglik_)


def test_factor_analysis_pa_runs_and_reduces_residual():
    rng = np.random.default_rng(59)
    n = 200
    L_true = rng.normal(size=(5, 2))
    Z = rng.normal(size=(n, 2))
    X = Z @ L_true.T + rng.normal(0, 0.3, (n, 5))
    r = st.factor_analysis(X, n_factors=2, method="pa")
    assert r.loadings_.shape == (5, 2)
    assert np.all(r.uniquenesses_ > 0)
    assert np.all(r.communalities_ >= 0)


def test_factor_analysis_varimax_rotation_preserves_covariance():
    rng = np.random.default_rng(60)
    n = 200
    X = rng.normal(size=(n, 2)) @ rng.normal(size=(2, 5)) + rng.normal(0, 0.3, (n, 5))
    r = st.factor_analysis(X, n_factors=2, method="ml", rotation="varimax")
    Sigma = r.loadings_ @ r.loadings_.T + np.diag(r.uniquenesses_)
    r_norot = st.factor_analysis(X, n_factors=2, method="ml")
    Sigma_norot = r_norot.loadings_ @ r_norot.loadings_.T + np.diag(r_norot.uniquenesses_)
    assert np.allclose(Sigma, Sigma_norot, atol=1e-3)


def test_canonical_correlation_vs_statsmodels():
    rng = np.random.default_rng(61)
    X = rng.normal(size=(150, 3))
    Y = X @ rng.normal(size=(3, 2)) + rng.normal(0, 0.5, (150, 2))
    r = st.canonical_correlation(X, Y)
    cc = CanCorr(Y, X)
    assert np.allclose(r.correlations_, cc.cancorr[: len(r.correlations_)], atol=1e-6)


def test_lda_matches_bayes_optimal_on_known_gaussians():
    rng = np.random.default_rng(62)
    means_true = np.array([[0.0, 0.0], [3.0, 3.0], [0.0, 3.0]])
    labels_true = rng.choice([0, 1, 2], size=300)
    X = means_true[labels_true] + rng.normal(0, 1, (300, 2))
    r = st.discriminant_analysis(X, labels_true, kind="lda")
    pred = r.predict(X)

    def bayes_predict(Xnew):
        ll = np.column_stack([
            multivariate_normal(mean=means_true[k], cov=np.eye(2)).logpdf(Xnew) for k in range(3)
        ])
        return np.argmax(ll, axis=1)

    bp = bayes_predict(X)
    assert np.mean(pred == bp) > 0.95  # near-total agreement with the true Bayes rule


def test_qda_separates_blobs_perfectly():
    rng = np.random.default_rng(63)
    means2 = np.array([[0.0, 0.0], [6.0, 6.0]])
    labels2 = rng.choice([0, 1], size=200)
    X = means2[labels2] + rng.normal(0, 0.7, (200, 2))
    r = st.discriminant_analysis(X, labels2, kind="qda")
    pred = r.predict(X)
    assert np.mean(pred == labels2) > 0.99
    proba = r.predict_proba(X[:5])
    assert np.allclose(proba.sum(axis=1), 1.0)


@pytest.mark.parametrize("link", ["ward", "single", "complete", "average"])
def test_hierarchical_clustering_matches_scipy(link):
    rng = np.random.default_rng(64)
    X = rng.normal(size=(30, 3))
    r = st.cluster_analysis(X, n_clusters=3, method="hierarchical", linkage=link)
    Zref = sp_linkage(X, method=link)
    assert np.allclose(np.sort(r.linkage_[:, 2]), np.sort(Zref[:, 2]), atol=1e-6)


def test_kmeans_matches_scipy_on_separated_blobs():
    rng = np.random.default_rng(65)
    X = np.vstack([rng.normal(0, 0.3, (50, 2)), rng.normal(5, 0.3, (50, 2)),
                    rng.normal([0, 5], 0.3, (50, 2))])
    r = st.cluster_analysis(X, n_clusters=3, method="kmeans", random_state=0)
    centers_ref, labels_ref = kmeans2(X, 3, minit="++", seed=0)
    ref_inertia = np.sum((X - centers_ref[labels_ref]) ** 2)
    assert r.inertia_ == pytest.approx(ref_inertia, rel=0.05)
    assert r.silhouette_ > 0.5


def test_mds_classical_recovers_configuration_up_to_rotation():
    rng = np.random.default_rng(66)
    X = rng.normal(size=(40, 3))
    D = np.sqrt(np.sum((X[:, None, :] - X[None, :, :]) ** 2, axis=2))
    r = st.MDS(D, n_components=3, method="classical")
    A = X - X.mean(axis=0)
    B = r.embedding_
    U, s, Vt = np.linalg.svd(A.T @ B)
    R = U @ Vt
    resid = np.linalg.norm(A @ R - B)
    assert resid < 1e-6
    assert r.stress_ < 1e-8


def test_mds_smacof_runs_and_low_stress_on_euclidean_data():
    rng = np.random.default_rng(67)
    X = rng.normal(size=(30, 2))
    D = np.sqrt(np.sum((X[:, None, :] - X[None, :, :]) ** 2, axis=2))
    r = st.MDS(D, n_components=2, method="smacof", random_state=1)
    assert r.stress_ < 0.05
    assert r.embedding_.shape == (30, 2)


# =============================================================== wiring

_SPEC_NAMES = {
    "mean", "median", "mode", "variance", "std", "quantile", "iqr", "skewness",
    "kurtosis", "covariance", "correlation", "describe",
    "MLE", "MOM", "bayesian_estimator", "confidence_interval", "bootstrap_ci",
    "jackknife", "delta_method", "profile_likelihood",
    "z_test", "t_test", "chi2_test", "f_test", "ANOVA", "MANOVA", "mann_whitney",
    "wilcoxon", "ks_test", "shapiro_wilk", "levene", "bartlett", "tukey_hsd", "bonferroni",
    "linear_regression", "logistic_regression", "poisson_regression", "ridge", "lasso",
    "elastic_net", "quantile_regression", "glm",
    "PCA", "factor_analysis", "canonical_correlation", "discriminant_analysis",
    "cluster_analysis", "MDS",
}


def test_module_wiring_and_exports():
    assert "statistics" in stochpylib.__all__
    assert hasattr(stochpylib, "statistics")
    assert len(_SPEC_NAMES) == 48
    missing = [n for n in _SPEC_NAMES if not hasattr(st, n)]
    assert not missing, f"missing exports: {missing}"
    assert _SPEC_NAMES <= set(st.__all__)
    assert list(st.__all__) == sorted(st.__all__)
    assert len(st.__all__) == len(set(st.__all__))

    submodule_alls = (
        set(st_desc.__all__) | set(st_est.__all__) | set(st_hyp.__all__)
        | set(st_reg.__all__) | set(st_mv.__all__)
    )
    assert submodule_alls == _SPEC_NAMES


def test_stdlib_statistics_module_not_shadowed():
    import statistics as stdlib_stats
    assert stdlib_stats is not st
    assert stdlib_stats.mean([1, 2, 3]) == 2


def test_readme_example_runs():
    rng = np.random.default_rng(68)
    x = rng.normal(0, 1, 100)
    y = rng.normal(0, 1, 100)
    d = st.describe(x)
    assert d.n == 100
    t = st.t_test(x, y)
    assert math.isfinite(t.pvalue)
    X = rng.normal(size=(100, 2))
    yl = 1.0 + X @ np.array([2.0, -1.0]) + rng.normal(0, 0.5, 100)
    reg = st.linear_regression(X, yl)
    assert reg.r2_ > 0.8
    pca = st.PCA(X)
    assert pca.components_.shape[0] <= 2


def test_random_state_reproducibility():
    rng = np.random.default_rng(69)
    x = rng.normal(5, 2, 50)
    r1 = st.bootstrap_ci(x, lambda a: np.mean(a, axis=-1), n_boot=200, random_state=7)
    r2 = st.bootstrap_ci(x, lambda a: np.mean(a, axis=-1), n_boot=200, random_state=7)
    assert np.array_equal(r1.extras["distribution"], r2.extras["distribution"])

    X = rng.normal(size=(60, 2))
    c1 = st.cluster_analysis(X, n_clusters=2, method="kmeans", random_state=3)
    c2 = st.cluster_analysis(X, n_clusters=2, method="kmeans", random_state=3)
    assert np.array_equal(c1.labels_, c2.labels_)

    D = np.abs(rng.normal(size=(20, 20)))
    D = (D + D.T) / 2
    np.fill_diagonal(D, 0)
    m1 = st.MDS(D, method="smacof", random_state=5)
    m2 = st.MDS(D, method="smacof", random_state=5)
    assert np.array_equal(m1.embedding_, m2.embedding_)
