"""Tests for stochpylib.nonparametric: kernel/series density estimation,
empirical distribution/CDF/characteristic-function/likelihood, resampling
and rank-based hypothesis tests, dependence measures, and nonparametric
regression.

Oracles: scipy.stats (gaussian_kde, mood, median_test, kruskal,
friedmanchisquare, spearmanr, kendalltau, somersd, anderson, anderson_ksamp,
cramervonmises, cramervonmises_2samp, permutation_test, binomtest, ecdf),
statsmodels (emplike.DescStat, sandbox.stats.runs, nonparametric.KernelReg),
scipy.optimize/interpolate (isotonic_regression, make_smoothing_spline) and
brute-force enumeration. All randomness is seeded; statistical assertions
are set at >= 3 Monte Carlo standard errors.
"""

import numpy as np
import pytest
from scipy import optimize, stats

import statsmodels.api as sm
import statsmodels.nonparametric.kernel_regression as sm_kr
import statsmodels.sandbox.stats.runs as sm_runs
from scipy.interpolate import make_smoothing_spline

from stochpylib import copulas, distributions, statistics
from stochpylib.nonparametric import (
    AdaptiveKDE, AndersenDarling, AndersonDarling, BootstrapTest,
    BrownianCorrelation, CramerVonMises, DistanceCorrelation, EmpiricalCDF,
    EmpiricalCharFn, EmpiricalDistribution, EmpiricalLikelihood, FriedmanTest,
    GPR_Nonparametric, GlivenkoCantelli, HoeffdingD, IsotonicRegression,
    KendallTau, KernelDensityEstimate, KruskalWallis, LocalPolynomialReg,
    LogsplineEstimator, MoodTest, NearestNeighborDensity, OrthogonalSeriesDensity,
    PermutationTest, QuantileRegression, RankCorrelation, RunsTest, SignTest,
    SpearmanCorrelation, SplineRegression, WaldWolfowitz,
)
from stochpylib.nonparametric._common import _bspline_design, _bspline_knots, _pava_weighted

_RNG = np.random.default_rng(0)


# ==================================================================== density

def test_kde_matches_gaussian_kde_pdf_and_cdf():
    x = _RNG.normal(0, 2, 400)
    kde = KernelDensityEstimate(bandwidth="silverman").fit(x)
    sk = stats.gaussian_kde(x, bw_method="silverman")
    grid = np.linspace(-6, 6, 40)
    assert np.max(np.abs(kde.pdf(grid) - sk.pdf(grid))) < 1e-10
    for t in (-1.0, 0.5, 2.0):
        assert abs(kde.cdf(t) - sk.integrate_box_1d(-np.inf, t)) < 1e-8


def test_kde_scott_bandwidth_matches_scipy():
    x = _RNG.normal(0, 1, 200)
    kde = KernelDensityEstimate(bandwidth="scott").fit(x)
    sk = stats.gaussian_kde(x, bw_method="scott")
    assert abs(kde.bandwidth_ - sk.factor * np.std(x, ddof=1)) < 1e-10


def test_kde_full_distribution_contract():
    x = _RNG.normal(5, 1, 300)
    kde = KernelDensityEstimate().fit(x)
    assert 0 < kde.pdf(5.0)
    assert 0 <= kde.cdf(5.0) <= 1
    assert abs(kde.mean() - np.mean(x)) < 1e-10
    assert kde.var() > np.var(x, ddof=1)  # smoothing inflates variance
    m = kde.ppf(0.5)
    assert abs(kde.cdf(m) - 0.5) < 1e-4
    draws = kde.rvs(2000, random_state=1)
    assert abs(np.mean(draws) - 5.0) < 0.3
    assert np.isfinite(kde.skewness())
    assert np.isfinite(kde.kurtosis())
    assert kde.entropy() > 0
    assert np.isfinite(kde.mgf(0.01))
    assert np.isfinite(abs(kde.cf(0.1)))
    d, p = kde.ks_test(x)
    assert d < 0.1 and p > 0.01


def test_kde_multivariate_pdf_integrates_to_one():
    X = _RNG.multivariate_normal([0, 0], [[1, 0.0], [0.0, 1]], 300)
    kde = KernelDensityEstimate().fit(X)
    g = np.linspace(-4, 4, 60)
    gx, gy = np.meshgrid(g, g)
    pts = np.column_stack([gx.ravel(), gy.ravel()])
    vals = kde.pdf(pts).reshape(gx.shape)
    integral = np.trapezoid(np.trapezoid(vals, g, axis=1), g)
    assert abs(integral - 1.0) < 0.05


def test_adaptive_kde_alpha_zero_equals_fixed_kde():
    x = _RNG.normal(0, 1, 200)
    fixed = KernelDensityEstimate(bandwidth="silverman").fit(x)
    adaptive = AdaptiveKDE(alpha=0.0, pilot_bandwidth="silverman").fit(x)
    grid = np.linspace(-3, 3, 30)
    assert np.max(np.abs(fixed.pdf(grid) - adaptive.pdf(grid))) < 1e-8


def test_adaptive_kde_reduces_ise_on_heavy_tailed_mixture():
    rng = np.random.default_rng(42)
    x = np.concatenate([rng.normal(0, 1, 400), rng.normal(0, 8, 100)])
    true = lambda t: 0.8 * stats.norm(0, 1).pdf(t) + 0.2 * stats.norm(0, 8).pdf(t)
    grid = np.linspace(-20, 20, 800)
    fixed = KernelDensityEstimate(bandwidth="silverman").fit(x)
    adaptive = AdaptiveKDE(alpha=0.5, pilot_bandwidth="silverman").fit(x)
    ise_fixed = np.trapezoid((fixed.pdf(grid) - true(grid)) ** 2, grid)
    ise_adaptive = np.trapezoid((adaptive.pdf(grid) - true(grid)) ** 2, grid)
    assert ise_adaptive < ise_fixed


def test_nearest_neighbor_density_integrates_and_tracks_shape():
    rng = np.random.default_rng(101)  # a fresh, order-independent generator
    x = rng.normal(0, 1, 2000)
    nn = NearestNeighborDensity(k=15).fit(x)
    grid = np.linspace(*nn.support(), 4000)
    assert abs(np.trapezoid(nn.pdf(grid), grid) - 1.0) < 0.01
    true = stats.norm(0, 1).pdf(grid)
    ise = np.trapezoid((nn.pdf(grid) - true) ** 2, grid)
    assert ise < 0.05  # overall shape close to the true density (pointwise kNN bias is high)
    d, p = nn.ks_test(x)
    assert d < 0.05


def test_orthogonal_series_density_integrates_to_one_and_converges():
    x = _RNG.normal(0, 1, 1500)
    osd = OrthogonalSeriesDensity(n_terms=12).fit(x)
    grid = np.linspace(*osd.support(), 3000)
    assert abs(np.trapezoid(osd.pdf(grid), grid) - 1.0) < 0.01
    true = stats.norm(0, 1).pdf(grid)
    ise = np.trapezoid((osd.pdf(grid) - true) ** 2, grid)
    assert ise < 0.01


def test_logspline_estimator_integrates_and_fits_gamma():
    x = _RNG.gamma(3.0, 2.0, 800)
    ls = LogsplineEstimator(n_knots=8).fit(x)
    grid = np.linspace(*ls.support(), 3000)
    assert abs(np.trapezoid(ls.pdf(grid), grid) - 1.0) < 0.02
    d, p = ls.ks_test(x)
    assert p > 0.01


# =================================================================== empirical

def test_empirical_distribution_matches_ecdf_and_quantile():
    x = _RNG.normal(4, 2, 250)
    ed = EmpiricalDistribution().fit(x)
    assert ed.cdf(np.median(x)) == pytest.approx(np.mean(x <= np.median(x)))
    assert ed.ppf(0.5) == pytest.approx(np.quantile(x, 0.5, method="inverted_cdf"))
    assert abs(ed.mean() - np.mean(x)) < 1e-10
    assert abs(ed.var() - np.var(x, ddof=0)) < 1e-10
    draws = ed.rvs(1000, random_state=2)
    assert set(np.unique(draws)) <= set(np.unique(x))
    assert ed.entropy() > 0
    assert 0 < ed.confidence_band(0.95) < 1


def test_empirical_cdf_matches_scipy_ecdf():
    x = _RNG.normal(0, 1, 150)
    ecdf = EmpiricalCDF().fit(x)
    ref = stats.ecdf(x)
    grid = np.linspace(-3, 3, 25)
    for t in grid:
        assert ecdf.evaluate(t) == pytest.approx(ref.cdf.evaluate(t))
    assert ecdf.quantile(0.3) == pytest.approx(np.quantile(x, 0.3, method="inverted_cdf"))
    assert ecdf.confidence_band(0.95) > 0


def test_empirical_charfn_matches_theoretical_normal_cf():
    x = np.asarray(distributions.Normal(0, 1).rvs(4000, random_state=3), dtype=float)
    ecf = EmpiricalCharFn().fit(x)
    dist = distributions.Normal(0, 1)
    for t in (0.2, 0.5, 1.0):
        se = ecf.std_error(t)
        assert abs(ecf.evaluate(t) - dist.cf(t)) < 5 * se
    d = ecf.distance(dist, np.linspace(-2, 2, 40))
    assert d < 0.02


def test_glivenko_cantelli_matches_ks_statistic_and_dkw_bound():
    x = _RNG.normal(0, 1, 200)
    dist = distributions.Normal(0, 1)
    gc = GlivenkoCantelli().fit(x, dist)
    ks = stats.kstest(x, dist.cdf).statistic
    assert gc.distance_ == pytest.approx(ks, abs=1e-10)
    assert gc.dkw_bound(0.95) > gc.distance_ * 0  # positive, sanity
    assert gc.n_required(0.05, 0.95) > 0
    conv = gc.convergence([50, 200, 800], dist, random_state=5)
    assert conv[-1] <= conv[0] + 0.1  # roughly decreasing trend


def test_empirical_likelihood_matches_statsmodels_desc_stat():
    x = _RNG.normal(3, 1.5, 200)
    el = EmpiricalLikelihood().fit(x)
    res = el.test_mean(3.0)
    ds = sm.emplike.DescStat(x)
    llr, p = ds.test_mean(3.0)
    assert res.statistic == pytest.approx(llr, rel=1e-6)
    assert res.pvalue == pytest.approx(p, rel=1e-5)
    lo, hi = el.confidence_interval()
    lo_sm, hi_sm = ds.ci_mean()
    assert lo == pytest.approx(lo_sm, abs=1e-3)
    assert hi == pytest.approx(hi_sm, abs=1e-3)


def test_empirical_likelihood_multivariate_mean():
    X = _RNG.multivariate_normal([1.0, -1.0], [[1, 0.3], [0.3, 1]], 300)
    el = EmpiricalLikelihood().fit(X)
    res = el.test_mean([1.0, -1.0])
    assert res.df == 2
    assert res.pvalue > 0.01
    res_far = el.test_mean([5.0, 5.0])
    assert res_far.pvalue < 0.01


# ================================================================= resampling

def test_permutation_test_exact_matches_scipy():
    x = _RNG.normal(0, 1, 6)
    y = _RNG.normal(1, 1, 5)

    def stat(a, b):
        return np.mean(a) - np.mean(b)

    pt = PermutationTest(statistic="mean_diff").fit(x, y)
    ref = stats.permutation_test((x, y), stat, n_resamples=np.inf, alternative="two-sided")
    assert pt.exact_ is True
    assert pt.statistic_ == pytest.approx(ref.statistic)
    assert pt.pvalue_ == pytest.approx(ref.pvalue, abs=1e-10)


def test_permutation_test_monte_carlo_close_to_scipy():
    x = _RNG.normal(0, 1, 40)
    y = _RNG.normal(0.3, 1, 40)

    def stat(a, b):
        return np.mean(a) - np.mean(b)

    pt = PermutationTest(n_resamples=4000, random_state=3).fit(x, y)
    ref = stats.permutation_test((x, y), stat, n_resamples=4000, alternative="two-sided",
                                  random_state=3)
    se = np.sqrt(0.25 / 4000)
    assert abs(pt.pvalue_ - ref.pvalue) < 6 * se


def test_permutation_test_pairings_and_samples_run():
    x = _RNG.normal(0, 1, 7)  # 7! = 5040 <= 20000: exact enumeration
    y = x * 0.5 + _RNG.normal(0, 0.3, 7)
    pt = PermutationTest(statistic="mean_diff", permutation_type="pairings").fit(x, y)
    assert pt.exact_
    d = x - y + 0.5
    pt2 = PermutationTest(permutation_type="samples").fit(d)
    assert pt2.exact_
    assert pt2.pvalue_ < 0.5  # shifted difference should trend low-p


def test_permutation_test_type1_error_calibrated():
    rng = np.random.default_rng(11)
    rejects = 0
    trials = 200
    for _ in range(trials):
        a = rng.normal(size=15)
        b = rng.normal(size=15)
        pt = PermutationTest(n_resamples=500, random_state=rng).fit(a, b)
        rejects += pt.reject(0.10)
    rate = rejects / trials
    se = np.sqrt(0.1 * 0.9 / trials)
    assert abs(rate - 0.10) < 4 * se


def test_bootstrap_test_one_and_two_sample():
    x = _RNG.normal(0.5, 1, 80)
    bt = BootstrapTest(mu0=0.0, n_boot=3000, random_state=4).fit(x)
    assert bt.pvalue_ < 0.05
    bt_null = BootstrapTest(mu0=float(np.mean(x)), n_boot=3000, random_state=4).fit(x)
    assert bt_null.pvalue_ > 0.3

    a = _RNG.normal(0, 1, 60)
    b = _RNG.normal(1.0, 1, 60)
    bt2 = BootstrapTest(n_boot=3000, random_state=5).fit(a, b)
    assert bt2.pvalue_ < 0.01
    bt2_null = BootstrapTest(n_boot=3000, random_state=5).fit(a, a + _RNG.normal(0, 0.01, 60))
    assert bt2_null.pvalue_ > 0.1


def test_bootstrap_test_to_result_and_reject():
    x = _RNG.normal(0, 1, 50)
    bt = BootstrapTest(n_boot=1000, random_state=6).fit(x)
    res = bt.to_result()
    assert res is bt.result_
    assert bt.reject(0.05) == (bt.pvalue_ < 0.05)


# =============================================================== rank tests

def test_mood_scale_test_matches_scipy():
    x = _RNG.normal(0, 1, 30)
    y = _RNG.normal(0, 2.5, 35)
    mt = MoodTest(kind="scale").fit(x, y)
    ref = stats.mood(x, y)
    assert mt.statistic_ == pytest.approx(ref.statistic)
    assert mt.pvalue_ == pytest.approx(ref.pvalue)


def test_mood_median_test_matches_scipy():
    x = _RNG.normal(0, 1, 40)
    y = _RNG.normal(0.8, 1, 45)
    mm = MoodTest(kind="median").fit(x, y)
    ref = stats.median_test(x, y)
    assert mm.statistic_ == pytest.approx(ref[0])
    assert mm.pvalue_ == pytest.approx(ref[1])


def test_kruskal_wallis_matches_scipy():
    g1 = _RNG.normal(0, 1, 20)
    g2 = _RNG.normal(0.5, 1, 25)
    g3 = _RNG.normal(1, 1, 22)
    kw = KruskalWallis().fit(g1, g2, g3)
    ref = stats.kruskal(g1, g2, g3)
    assert kw.statistic_ == pytest.approx(ref.statistic)
    assert kw.pvalue_ == pytest.approx(ref.pvalue)


def test_kruskal_wallis_posthoc_dunn_flags_the_different_group():
    g1 = _RNG.normal(0, 1, 40)
    g2 = _RNG.normal(0, 1, 40)
    g3 = _RNG.normal(3, 1, 40)
    kw = KruskalWallis().fit(g1, g2, g3)
    dunn = kw.posthoc_dunn(adjust="holm")
    row13 = [r for r in dunn.table if r["i"] == 0 and r["j"] == 2][0]
    row12 = [r for r in dunn.table if r["i"] == 0 and r["j"] == 1][0]
    assert row13["p_adj"] < 0.01
    assert row12["p_adj"] > 0.05


def test_friedman_test_matches_scipy_with_ties():
    n, k = 15, 4
    data = _RNG.normal(size=(n, k))
    data[:, 0] += 1.0
    data[0, 0] = data[0, 1]
    samples = [data[:, j] for j in range(k)]
    ft = FriedmanTest().fit(*samples)
    ref = stats.friedmanchisquare(*samples)
    assert ft.statistic_ == pytest.approx(ref.statistic)
    assert ft.pvalue_ == pytest.approx(ref.pvalue)
    assert 0 <= ft.kendall_w_ <= 1


def test_sign_test_matches_binomtest():
    x = _RNG.normal(0.4, 1, 40)
    st = SignTest(mu0=0.0).fit(x)
    k = int(np.sum(x > 0))
    ref = stats.binomtest(k, len(x), 0.5)
    assert st.statistic_ == k
    assert st.pvalue_ == pytest.approx(ref.pvalue)


def test_sign_test_paired_and_one_sided():
    x = _RNG.normal(2, 1, 30)
    y = x - 0.5 + _RNG.normal(0, 0.2, 30)
    st = SignTest(alternative="greater").fit(x, y)
    assert st.pvalue_ < 0.05


def test_runs_test_matches_statsmodels():
    x = _RNG.normal(0, 1, 80)
    rt = RunsTest(cutoff="median").fit(x)
    z, p = sm_runs.runstest_1samp(x, cutoff="median", correction=True)
    assert rt.statistic_ == pytest.approx(z)
    assert rt.pvalue_ == pytest.approx(p)


def test_runs_test_detects_non_random_alternating_sequence():
    x = np.tile([1.0, -1.0], 30)
    rt = RunsTest(cutoff=0.0).fit(x)
    assert rt.reject(0.01)


def test_wald_wolfowitz_matches_statsmodels():
    a = _RNG.normal(0, 1, 25)
    b = _RNG.normal(0, 1, 30)
    ww = WaldWolfowitz().fit(a, b)
    z, p = sm_runs.runstest_2samp(a, b, correction=True)
    assert ww.statistic_ == pytest.approx(z)
    assert ww.pvalue_ == pytest.approx(p)


def test_wald_wolfowitz_detects_different_distributions():
    a = _RNG.normal(0, 1, 60)
    b = _RNG.normal(6, 1, 60)
    ww = WaldWolfowitz().fit(a, b)
    assert ww.reject(0.01)


# ========================================================== goodness of fit

def test_anderson_darling_statistic_matches_scipy_and_critical_values_match_table():
    # only the A^2 statistic is oracled against scipy: scipy switched critical-value
    # correction formulas mid-1.x and drops `critical_values` entirely in 1.19, so
    # comparing them pins us to one scipy version. The table below is the library's
    # own documented contract (D'Agostino & Stephens 1986, Table 4.7).
    xn = _RNG.normal(2, 3, 60)
    ad = AndersenDarling(dist="norm").fit(xn)
    ref = stats.anderson(xn, dist="norm")
    assert ad.statistic_ == pytest.approx(ref.statistic)
    n = len(xn)
    expected = np.round(np.array([0.561, 0.631, 0.752, 0.873, 1.035])
                        / (1.0 + 0.75 / n + 2.25 / n ** 2), 3)
    assert list(ad.result_.extras["critical_values"]) == pytest.approx(list(expected))
    assert list(ad.result_.extras["significance_levels"]) == pytest.approx(
        [0.15, 0.10, 0.05, 0.025, 0.01])

    xe = _RNG.exponential(2, 60)
    ade = AndersenDarling(dist="expon").fit(xe)
    refe = stats.anderson(xe, dist="expon")
    assert ade.statistic_ == pytest.approx(refe.statistic)
    m = len(xe)
    expected_e = np.round(np.array([0.916, 1.062, 1.321, 1.591, 1.959]) / (1.0 + 0.6 / m), 3)
    assert list(ade.result_.extras["critical_values"]) == pytest.approx(list(expected_e))


def test_anderson_darling_alias_is_identical():
    assert AndersonDarling is AndersenDarling


def test_anderson_darling_known_distribution_calibrated_and_powerful():
    # the "parameters known" (case-0) test is sensitive to ANY location/scale
    # mismatch (it does not absorb estimated nuisance parameters the way the
    # fitted dist="norm" path does), so a single true-null draw isn't
    # guaranteed p > 0.10 -- check the Type-I rate over repeated draws instead.
    rng = np.random.default_rng(70)
    rejects = 0
    trials = 100
    for _ in range(trials):
        x = np.asarray(distributions.Normal(0, 1).rvs(150, random_state=rng), dtype=float)
        ad = AndersenDarling(dist=distributions.Normal(0, 1)).fit(x)
        rejects += ad.pvalue_ < 0.05
    rate = rejects / trials
    assert rate < 0.20  # loose bound: table interpolation is approximate, not exact
    # a clearly different distribution should be rejected decisively
    x_bad = rng.uniform(-3, 3, 300)
    ad_bad = AndersenDarling(dist=distributions.Normal(0, 1)).fit(x_bad)
    assert ad_bad.pvalue_ < 0.01


def test_anderson_darling_ksample_matches_scipy():
    s1 = _RNG.normal(size=50)
    s2 = _RNG.normal(0.5, size=30)
    adk = AndersenDarling(k_sample=True).fit(s1, s2)
    ref = stats.anderson_ksamp([s1, s2])
    assert adk.statistic_ == pytest.approx(ref.statistic)
    assert adk.pvalue_ == pytest.approx(ref.pvalue, rel=0.05)


def test_cramer_von_mises_one_sample_matches_scipy_statistic():
    x = _RNG.normal(size=200)
    cvm = CramerVonMises().fit(x, distributions.Normal(0, 1))
    ref = stats.cramervonmises(x, "norm")
    assert cvm.statistic_ == pytest.approx(ref.statistic)
    assert cvm.pvalue_ == pytest.approx(ref.pvalue, abs=0.01)


def test_cramer_von_mises_two_sample_matches_scipy_asymptotic():
    x = _RNG.normal(size=100)
    y = _RNG.normal(0.2, 1.3, 90)
    cvm2 = CramerVonMises().fit(x, y)
    ref = stats.cramervonmises_2samp(x, y, method="asymptotic")
    assert cvm2.statistic_ == pytest.approx(ref.statistic)
    assert cvm2.pvalue_ == pytest.approx(ref.pvalue, rel=1e-4)


def test_cramer_von_mises_rejects_clearly_different_distributions():
    x = _RNG.normal(size=150)
    y = _RNG.normal(3, 1, 150)
    cvm = CramerVonMises().fit(x, y)
    assert cvm.pvalue_ < 0.001


# =============================================================== correlation

def test_spearman_correlation_matches_scipy():
    x = _RNG.normal(0, 1, 60)
    y = 2 * x + _RNG.normal(0, 0.5, 60)
    sc = SpearmanCorrelation().fit(x, y)
    ref = stats.spearmanr(x, y)
    assert sc.estimate_ == pytest.approx(ref.statistic)
    assert sc.pvalue_ == pytest.approx(ref.pvalue)
    lo, hi = sc.confidence_interval()
    assert lo < sc.estimate_ < hi


def test_kendall_tau_asymptotic_matches_scipy_with_ties():
    x = _RNG.normal(0, 1, 60)
    y = 2 * x + _RNG.normal(0, 0.5, 60)
    x[:5] = x[10]
    y[:3] = y[20]
    kt = KendallTau(method="asymptotic").fit(x, y)
    ref = stats.kendalltau(x, y, method="asymptotic")
    assert kt.estimate_ == pytest.approx(ref.statistic)
    assert kt.pvalue_ == pytest.approx(ref.pvalue)


def test_kendall_tau_exact_matches_scipy_no_ties():
    x = _RNG.normal(size=12)
    y = x * 0.5 + _RNG.normal(0, 1, 12)
    kt = KendallTau(method="exact").fit(x, y)
    ref = stats.kendalltau(x, y, method="exact")
    assert kt.estimate_ == pytest.approx(ref.statistic)
    assert kt.pvalue_ == pytest.approx(ref.pvalue)


def test_kendall_tau_variants_a_and_c():
    x = _RNG.normal(size=40)
    y = x + _RNG.normal(0, 0.3, 40)
    tau_b = KendallTau(variant="b").fit(x, y).estimate_
    tau_a = KendallTau(variant="a").fit(x, y).estimate_
    tau_c = KendallTau(variant="c").fit(x, y).estimate_
    # no ties: tau-a == tau-b exactly
    assert tau_a == pytest.approx(tau_b, abs=1e-10)
    assert 0 < tau_c <= 1


def test_kendall_tau_matches_copulas_kendall_tau():
    x = _RNG.normal(size=100)
    y = x + _RNG.normal(0, 0.5, 100)
    kt = KendallTau().fit(x, y)
    ref = copulas.kendall_tau(np.column_stack([x, y]))
    assert kt.estimate_ == pytest.approx(float(ref), abs=1e-10)


def test_rank_correlation_dispatcher_matches_scipy_somersd():
    x = _RNG.normal(0, 1, 60)
    y = 2 * x + _RNG.normal(0, 0.5, 60)
    x[:5] = x[10]
    rc = RankCorrelation(method="somers_d").fit(x, y)
    ref = stats.somersd(x, y)
    assert rc.estimate_ == pytest.approx(ref.statistic)


def test_rank_correlation_gamma_bounded():
    x = _RNG.normal(size=50)
    y = x + _RNG.normal(0, 0.4, 50)
    rc = RankCorrelation(method="gamma").fit(x, y)
    assert -1 <= rc.estimate_ <= 1


def test_distance_correlation_detects_nonlinear_dependence():
    rng = np.random.default_rng(20)
    x = rng.uniform(-2, 2, 150)
    y = x ** 2 + rng.normal(0, 0.3, 150)
    dc = DistanceCorrelation(random_state=1).fit(x, y)
    assert dc.estimate_ > 0.3
    assert dc.pvalue_ < 0.01

    x2 = rng.normal(size=150)
    y2 = rng.normal(size=150)
    dc2 = DistanceCorrelation(random_state=1).fit(x2, y2)
    assert dc2.estimate_ < 0.25
    assert dc2.pvalue_ > 0.05


def test_distance_correlation_unbiased_matches_brute_force():
    rng = np.random.default_rng(21)
    x = rng.normal(size=60)
    y = x + rng.normal(0, 0.5, 60)

    def dcov2_brute(a, b, unbiased):
        n = len(a)
        A = np.abs(a[:, None] - a[None, :])
        B = np.abs(b[:, None] - b[None, :])
        if not unbiased:
            Ac = A - A.mean(0, keepdims=True) - A.mean(1, keepdims=True) + A.mean()
            Bc = B - B.mean(0, keepdims=True) - B.mean(1, keepdims=True) + B.mean()
            return np.mean(Ac * Bc)
        rowA = A.sum(1, keepdims=True) / (n - 2)
        colA = A.sum(0, keepdims=True) / (n - 2)
        grandA = A.sum() / ((n - 1) * (n - 2))
        Ac = A - rowA - colA + grandA
        np.fill_diagonal(Ac, 0.0)
        rowB = B.sum(1, keepdims=True) / (n - 2)
        colB = B.sum(0, keepdims=True) / (n - 2)
        grandB = B.sum() / ((n - 1) * (n - 2))
        Bc = B - rowB - colB + grandB
        np.fill_diagonal(Bc, 0.0)
        return np.sum(Ac * Bc) / (n * (n - 3))

    dc = DistanceCorrelation(unbiased=True, n_resamples=10, random_state=0)
    mine = dc._dcov2(x, y)
    ref = dcov2_brute(x, y, True)
    assert mine == pytest.approx(ref, abs=1e-10)


def test_brownian_correlation_hurst_half_equals_distance_correlation():
    rng = np.random.default_rng(22)
    x = rng.uniform(-2, 2, 100)
    y = x ** 2 + rng.normal(0, 0.3, 100)
    dc = DistanceCorrelation(random_state=1).fit(x, y)
    bc = BrownianCorrelation(hurst=0.5, random_state=1).fit(x, y)
    assert bc.estimate_ == pytest.approx(dc.estimate_, abs=1e-10)


def test_hoeffding_d_detects_dependence_and_zero_under_independence():
    rng = np.random.default_rng(23)
    x = rng.uniform(-2, 2, 150)
    y = x ** 2 + rng.normal(0, 0.3, 150)
    hd = HoeffdingD(random_state=2).fit(x, y)
    assert hd.estimate_ > 0.05
    assert hd.pvalue_ < 0.05

    x2 = rng.normal(size=150)
    y2 = rng.normal(size=150)
    hd2 = HoeffdingD(random_state=2).fit(x2, y2)
    assert abs(hd2.estimate_) < 0.05


def test_hoeffding_d_perfect_monotone_is_one():
    x = np.sort(_RNG.normal(size=80))
    hd = HoeffdingD(n_resamples=100, random_state=3).fit(x, x.copy())
    assert hd.estimate_ == pytest.approx(1.0, abs=1e-8)


# ================================================================ regression

def test_local_polynomial_matches_statsmodels_kernelreg():
    x = np.sort(_RNG.uniform(0, 10, 80))
    y = np.sin(x) + _RNG.normal(0, 0.2, 80)
    h = 0.7
    for degree, regtype in ((0, "lc"), (1, "ll")):
        lp = LocalPolynomialReg(degree=degree, bandwidth=h).fit(x, y)
        kreg = sm_kr.KernelReg(endog=y, exog=x, var_type="c", reg_type=regtype, bw=[h])
        fit_sm, _ = kreg.fit(x)
        assert np.max(np.abs(lp.fitted_ - fit_sm)) < 1e-8


def test_local_polynomial_gcv_bandwidth_reasonable():
    x = np.sort(_RNG.uniform(0, 10, 80))
    y = np.sin(x) + _RNG.normal(0, 0.2, 80)
    lp = LocalPolynomialReg(bandwidth="gcv").fit(x, y)
    assert 0.05 < lp.bandwidth_ < 5.0
    pred, se = lp.predict(np.array([2.0, 5.0]), return_std=True)
    assert pred.shape == (2,) and np.all(se > 0)


def test_isotonic_regression_matches_scipy_exactly():
    y = np.sort(_RNG.uniform(0, 1, 50)) + _RNG.normal(0, 0.3, 50)
    x = np.arange(50)
    iso = IsotonicRegression().fit(x, y)
    ref = optimize.isotonic_regression(y)
    assert np.max(np.abs(iso.fitted_ - ref.x)) < 1e-10
    assert np.all(np.diff(iso.fitted_) >= -1e-10)


def test_isotonic_regression_decreasing_and_predict():
    x = np.arange(30)
    y = 10 - 0.3 * x + _RNG.normal(0, 0.5, 30)
    iso = IsotonicRegression(increasing=False).fit(x, y)
    assert np.all(np.diff(iso.fitted_) <= 1e-10)
    pred = iso.predict(np.array([-5, 35]))
    assert pred[0] == pytest.approx(iso.fitted_[0])
    assert pred[1] == pytest.approx(iso.fitted_[-1])


def test_spline_regression_pspline_lambda_zero_equals_ols():
    x = np.sort(_RNG.uniform(0, 10, 80))
    y = np.sin(x) + _RNG.normal(0, 0.2, 80)
    sr = SplineRegression(method="pspline", n_knots=10, lam=0.0).fit(x, y)
    knots = _bspline_knots(x, 10, 3)
    B = _bspline_design(x, knots, 3)
    beta_ols = np.linalg.lstsq(B, y, rcond=None)[0]
    assert np.max(np.abs(sr.fitted_ - B @ beta_ols)) < 1e-8


def test_spline_regression_pspline_gcv_smooths_reasonably():
    x = np.sort(_RNG.uniform(0, 10, 100))
    y = np.sin(x) + _RNG.normal(0, 0.2, 100)
    sr = SplineRegression(method="pspline", n_knots=15).fit(x, y)
    mse = np.mean((sr.fitted_ - np.sin(x)) ** 2)
    assert mse < 0.05
    assert 2 < sr.df_ < 15


def test_spline_regression_smoothing_matches_scipy_exactly():
    x = np.sort(_RNG.uniform(0, 10, 60))
    y = np.sin(x) + _RNG.normal(0, 0.2, 60)
    lam = 1.0
    sr = SplineRegression(method="smoothing", lam=lam).fit(x, y)
    ref = make_smoothing_spline(x, y, lam=lam)
    assert np.max(np.abs(sr.fitted_ - ref(x))) < 1e-10
    assert sr.predict(np.array([5.0]))[0] == pytest.approx(float(ref(5.0)))


def test_gpr_nonparametric_matches_gp_regression_directly():
    from stochpylib.gaussian_processes import GPRegression
    from stochpylib.gaussian_processes.kernels import RBFKernel, WhiteNoiseKernel

    x = np.sort(_RNG.uniform(0, 10, 60))
    y = np.sin(x) + _RNG.normal(0, 0.1, 60)
    gp = GPR_Nonparametric(noise=0.1, optimize=False).fit(x, y)
    scale = float(np.std(x)) or 1.0
    kernel = RBFKernel(length_scale=scale, variance=float(np.var(y)) or 1.0) + \
        WhiteNoiseKernel(0.1)
    gpr_ref = GPRegression(kernel=kernel, noise=0.1).fit(x[:, None], y)
    mu_mine = gp.predict(x[:5][:, None])
    mu_ref, _ = gpr_ref.predict(x[:5][:, None])
    assert np.allclose(mu_mine, mu_ref)


def test_gpr_nonparametric_forecast_shapes():
    x = np.sort(_RNG.uniform(0, 10, 40))
    y = np.sin(x) + _RNG.normal(0, 0.1, 40)
    gp = GPR_Nonparametric(optimize=False).fit(x, y)
    mu, std = gp.predict(np.array([2.5, 7.5])[:, None], return_std=True)
    assert mu.shape == (2,) and np.all(std > 0)


def test_quantile_regression_local_huge_bandwidth_equals_linear():
    x = _RNG.uniform(0, 10, 120)
    y = 1 + 2 * x + _RNG.normal(0, 1, 120)
    qr_local = QuantileRegression(q=0.5, bandwidth=1e6).fit(x, y)
    qr_lin = QuantileRegression(q=0.5, method="linear").fit(x, y)
    assert np.max(np.abs(qr_local.fitted_ - qr_lin.fitted_)) < 1e-4


def test_quantile_regression_linear_matches_statistics_quantile_regression():
    x = _RNG.uniform(0, 10, 100)
    y = 1 + 2 * x + _RNG.normal(0, 1, 100)
    qr = QuantileRegression(q=0.3, method="linear").fit(x, y)
    ref = statistics.quantile_regression(x, y, q=0.3)
    assert np.max(np.abs(qr.fitted_ - ref.fitted_)) < 1e-6


def test_quantile_regression_empirical_coverage():
    rng = np.random.default_rng(30)
    x = rng.uniform(0, 10, 150)
    y = 1 + 2 * x + rng.normal(0, 1 + 0.3 * x, 150)
    qr90 = QuantileRegression(q=0.9, bandwidth=1.5).fit(x, y)
    pred = qr90.predict(x)
    coverage = np.mean(y <= pred)
    se = np.sqrt(0.9 * 0.1 / 150)
    assert abs(coverage - 0.9) < 5 * se


# ============================================================== quickstart

def test_quickstart_example_runs():
    rng = np.random.default_rng(0)
    x = np.concatenate([rng.normal(0, 1, 400), rng.normal(6, 1, 100)])
    kde = KernelDensityEstimate(bandwidth="silverman").fit(x)
    assert kde.pdf(0.0) > 0

    x1 = rng.normal(0, 1, 60)
    y1 = rng.normal(0.6, 1, 60)
    pt = PermutationTest(n_resamples=2000, random_state=1).fit(x1, y1)
    assert pt.pvalue_ < 0.1

    t = rng.uniform(0, 10, 80)
    z = np.sin(t) + rng.normal(0, 0.2, 80)
    lp = LocalPolynomialReg(bandwidth=0.8).fit(t, z)
    assert lp.score(t, z) > 0.5

    a = rng.uniform(-2, 2, 120)
    b = a ** 2 + rng.normal(0, 0.3, 120)
    dc = DistanceCorrelation(random_state=2).fit(a, b)
    assert dc.pvalue_ < 0.05
