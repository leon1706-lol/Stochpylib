"""Tests for stochpylib.robust_statistics: robust location/scale estimators,
high-breakdown regression, robust covariance, and resampling.

Oracles: scipy.stats (trim_mean, mstats.winsorize, median_abs_deviation, iqr,
theilslopes, siegelslopes, spearmanr, kendalltau), statsmodels (RLM, qn_scale,
Huber's proposal 2), brute-force enumeration (LTS/MCD/MVE exact subsets, Sn),
scipy.integrate (tau-scale consistency constant), and a cross-module check
against stochpylib.financial_stochastics.CovarianceEstimation's Ledoit-Wolf
shrinkage. All randomness is seeded; statistical assertions are set at
>= 3 Monte Carlo standard errors.
"""

import itertools
import math

import numpy as np
import pytest
from scipy import integrate, stats

import statsmodels.api as sm
import statsmodels.robust.scale as sms

from stochpylib.robust_statistics import (
    BlockBootstrap, CovShrinkage, HodgesLehmann, HuberRegression, IQR_Scale,
    LTS_Regression, L_Estimator, MCD, MMRegression, MVE, M_Estimator, Median,
    MedianAbsoluteDeviation, OGK, Qn_Estimator, RANSACRegression, R_Estimator,
    Resampler, RobustBootstrap, RobustCorrelation, RobustCovariance,
    RobustCovarianceEstimator, RobustEstimator, RobustRegressor, RobustStd,
    SiegalRegression, SiegelRegression, Sn_Estimator, StationaryBootstrap,
    TheilSenRegression, TrimmedMean, WildBootstrap, WinsorizedMean,
)
from stochpylib.robust_statistics._common import (
    _huber_proposal2, _kth_cross_diff, _kth_pairwise, _mad, _tau_consistency, _tau_scale,
)
from stochpylib.statistics import EstimateResult, RegressionResult, linear_regression

_RNG = np.random.default_rng(0)


def _contaminated(n_clean=160, n_out=40, loc=0.0, scale=1.0, out_val=1e6, seed=0):
    rng = np.random.default_rng(seed)
    return np.concatenate([rng.normal(loc, scale, n_clean), np.full(n_out, out_val)])


# ============================================================== _common helpers

def test_kth_pairwise_bisection_matches_explicit():
    rng = np.random.default_rng(1)
    x = rng.standard_normal(1500)
    for kind, max_k in [("diff", 1500 * 1499 // 2), ("walsh", 1500 * 1501 // 2)]:
        for k in (1, 30, 500, max_k // 2, max_k):
            v_explicit = _kth_pairwise(x, k, kind=kind, max_explicit=100_000)
            v_bisect = _kth_pairwise(x, k, kind=kind, max_explicit=0)
            assert abs(v_explicit - v_bisect) < 1e-8


def test_kth_cross_diff_bisection_matches_explicit():
    rng = np.random.default_rng(2)
    x = rng.standard_normal(200)
    y = rng.standard_normal(300)
    mn = len(x) * len(y)
    for k in (1, 100, mn // 2, mn):
        v_explicit = _kth_cross_diff(x, y, k, max_explicit=1_000_000)
        v_bisect = _kth_cross_diff(x, y, k, max_explicit=0)
        assert abs(v_explicit - v_bisect) < 1e-8


def test_huber_proposal2_matches_statsmodels():
    rng = np.random.default_rng(3)
    for seed in range(5):
        x = np.concatenate([rng.normal(5, 2, 190), rng.normal(60, 5, 10)])
        mu, s = _huber_proposal2(x)
        mu2, s2 = sms.Huber()(x)
        assert abs(mu - mu2) < 1e-6
        assert abs(s - s2) < 1e-6


def test_tau_consistency_matches_quadrature():
    for c in (2.0, 3.0, 4.5):
        val = _tau_consistency(c)
        exact, _ = integrate.quad(
            lambda z: min(z ** 2, c ** 2) * math.exp(-0.5 * z * z) / math.sqrt(2 * math.pi),
            -60, 60)
        assert abs(val - exact) < 1e-8


def test_tau_scale_consistent_at_gaussian():
    rng = np.random.default_rng(4)
    z = rng.standard_normal(20000)
    loc, scale = _tau_scale(z)
    assert abs(loc) < 0.05
    assert abs(scale - 1.0) < 0.05


# ==================================================================== location

@pytest.mark.parametrize("p", [0.0, 0.1, 0.2, 0.25])
def test_trimmed_mean_matches_scipy(p):
    x = _RNG.normal(0, 1, 137)
    tm = TrimmedMean(p).fit(x)
    assert tm.estimate_ == pytest.approx(stats.trim_mean(x, p), abs=1e-10)


def test_trimmed_mean_se_and_df():
    x = _RNG.normal(0, 1, 100)
    tm = TrimmedMean(0.1).fit(x)
    g = int(np.floor(0.1 * 100))
    xs = np.sort(x)
    win = np.concatenate([np.full(g, xs[g]), xs[g:100 - g], np.full(g, xs[100 - g - 1])])
    s_w = np.std(win, ddof=1)
    se_expected = s_w / (0.8 * math.sqrt(100))
    assert tm.std_error_ == pytest.approx(se_expected, rel=1e-10)
    assert tm.extras_["df"] == 100 - 2 * g - 1


def test_winsorized_mean_matches_scipy():
    from scipy.stats.mstats import winsorize
    x = _RNG.normal(0, 1, 141)
    wm = WinsorizedMean(0.1).fit(x)
    expected = float(np.mean(winsorize(x, (0.1, 0.1))))
    assert wm.estimate_ == pytest.approx(expected, abs=1e-10)


def test_median_matches_numpy_and_mj_se():
    x = _RNG.normal(5, 2, 101)
    med = Median().fit(x)
    assert med.estimate_ == pytest.approx(float(np.median(x)))
    from scipy import special
    xs = np.sort(x)
    n = len(xs)
    m = (n + 1) // 2
    i = np.arange(1, n + 1)
    W = special.betainc(m, n - m + 1, i / n) - special.betainc(m, n - m + 1, (i - 1) / n)
    c1 = np.sum(W * xs)
    c2 = np.sum(W * xs ** 2)
    se_expected = math.sqrt(max(c2 - c1 ** 2, 0.0))
    assert med.std_error_ == pytest.approx(se_expected, rel=1e-10)


def test_median_order_statistic_ci_coverage():
    rng = np.random.default_rng(5)
    n_cover = 0
    trials = 400
    for _ in range(trials):
        x = rng.normal(0, 1, 25)
        lo, hi = Median().fit(x).confidence_interval(0.95)
        n_cover += lo <= 0.0 <= hi
    rate = n_cover / trials
    se = math.sqrt(0.95 * 0.05 / trials)
    assert rate >= 0.93 - 3 * se


def test_median_bootstrap_se_close_to_mj():
    x = _RNG.normal(0, 1, 300)
    mj = Median(se_method="maritz-jarrett").fit(x)
    boot = Median(se_method="bootstrap", n_boot=1000, random_state=0).fit(x)
    assert boot.std_error_ == pytest.approx(mj.std_error_, rel=0.35)


def test_hodges_lehmann_one_sample_matches_brute_force():
    for n in (20, 21, 41, 42):
        x = _RNG.normal(0, 1, n)
        hl = HodgesLehmann().fit(x)
        walsh = np.array([(x[i] + x[j]) / 2.0 for i in range(n) for j in range(i, n)])
        assert hl.estimate_ == pytest.approx(float(np.median(walsh)), abs=1e-8)


def test_hodges_lehmann_two_sample_matches_brute_force():
    x = _RNG.normal(0, 1, 18)
    y = _RNG.normal(1, 1, 23)
    hl = HodgesLehmann().fit(x, y)
    diffs = (x[:, None] - y[None, :]).ravel()
    assert hl.estimate_ == pytest.approx(float(np.median(diffs)), abs=1e-8)


def test_hodges_lehmann_ci_coverage_and_shrinks():
    rng = np.random.default_rng(6)
    n_cover, trials = 0, 300
    widths = {}
    for n in (20, 80):
        w = []
        for _ in range(trials // (2 if n == 80 else 1)):
            x = rng.normal(0, 1, n)
            lo, hi = HodgesLehmann().fit(x).confidence_interval()
            w.append(hi - lo)
            if n == 20:
                n_cover += lo <= 0.0 <= hi
        widths[n] = np.mean(w)
    rate = n_cover / trials
    se = math.sqrt(0.95 * 0.05 / trials)
    assert rate >= 0.93 - 3 * se
    assert widths[80] < widths[20]


def test_l_estimator_trimean_and_alias_behaviors():
    x = _RNG.normal(0, 1, 91)
    xs = np.sort(x)
    q1, q3 = np.quantile(xs, [0.25, 0.75])
    med = np.median(xs)
    le = L_Estimator("trimean", n_boot=50, random_state=0).fit(x)
    assert le.estimate_ == pytest.approx(0.25 * q1 + 0.5 * med + 0.25 * q3, abs=1e-8)

    le_trim = L_Estimator("trimmed", proportion=0.1, n_boot=50, random_state=0).fit(x)
    tm = TrimmedMean(0.1).fit(x)
    assert le_trim.estimate_ == pytest.approx(tm.estimate_, abs=1e-8)

    le_const = L_Estimator(lambda u: np.ones_like(u), n_boot=50, random_state=0).fit(x)
    assert le_const.estimate_ == pytest.approx(float(np.mean(x)), abs=1e-8)

    w = np.zeros(91)
    w[45] = 1.0
    le_expl = L_Estimator(w, n_boot=50, random_state=0).fit(x)
    assert le_expl.estimate_ == pytest.approx(float(xs[45]), abs=1e-8)


@pytest.mark.parametrize("psi", ["huber", "tukey"])
def test_m_estimator_matches_statsmodels_rlm(psi):
    m_class = {"huber": sm.robust.norms.HuberT(), "tukey": sm.robust.norms.TukeyBiweight()}[psi]
    for seed in range(4):
        rng = np.random.default_rng(seed)
        x = np.concatenate([rng.normal(3, 1, 90), rng.normal(-8, 1, 10)])
        me = M_Estimator(psi, scale="mad").fit(x)
        rlm = sm.RLM(x, np.ones(len(x)), M=m_class).fit(scale_est="mad", cov="H1")
        assert me.estimate_ == pytest.approx(rlm.params[0], abs=1e-5)
        assert me.std_error_ == pytest.approx(rlm.bse[0], rel=1e-4)
        assert me.extras_["scale_"] == pytest.approx(rlm.scale, abs=1e-5)


def test_m_estimator_huber_scale_matches_huber_proposal2():
    x = _RNG.normal(0, 1, 150)
    me = M_Estimator("huber", scale="huber").fit(x)
    mu, s = sms.Huber()(x)
    assert me.estimate_ == pytest.approx(mu, abs=1e-4)
    assert me.extras_["scale_"] == pytest.approx(s, abs=1e-8)


@pytest.mark.parametrize("psi", sorted(["huber", "tukey", "biweight", "hampel", "andrews",
                                        "cauchy", "welsch", "fair", "logistic"]))
def test_m_estimator_every_psi_finite_and_robust(psi):
    x = _contaminated(180, 20, seed=int(hash(psi) % 1000))
    me = M_Estimator(psi).fit(x)
    assert np.isfinite(me.estimate_)
    assert abs(me.estimate_) < abs(float(np.mean(x)))


def test_r_estimator_wilcoxon_equals_hodges_lehmann():
    x = _RNG.normal(0, 1, 51)
    y = _RNG.normal(1, 1, 44)
    assert R_Estimator("wilcoxon").fit(x).estimate_ == pytest.approx(
        HodgesLehmann().fit(x).estimate_, abs=1e-9)
    assert R_Estimator("wilcoxon").fit(x, y).estimate_ == pytest.approx(
        HodgesLehmann().fit(x, y).estimate_, abs=1e-9)


def test_r_estimator_sign_equals_median():
    x = _RNG.normal(0, 1, 61)
    y = _RNG.normal(2, 1, 47)
    assert R_Estimator("sign").fit(x).estimate_ == pytest.approx(float(np.median(x)), abs=1e-9)
    assert R_Estimator("sign").fit(x, y).estimate_ == pytest.approx(
        float(np.median(x) - np.median(y)), abs=1e-9)


def test_r_estimator_normal_scores_close_to_truth():
    rng = np.random.default_rng(7)
    x = rng.normal(3.0, 1.0, 200)
    est = R_Estimator("normal").fit(x)
    assert abs(est.estimate_ - 3.0) < 5 * est.std_error_
    y = rng.normal(-1.0, 1.0, 150)
    est2 = R_Estimator("normal").fit(x, y)
    assert abs(est2.estimate_ - 4.0) < 5 * est2.std_error_


@pytest.mark.parametrize("cls,kwargs", [
    (TrimmedMean, dict(proportion=0.25)), (WinsorizedMean, dict(proportion=0.25)),
    (Median, {}), (HodgesLehmann, {}), (L_Estimator, dict(n_boot=50)),
    (M_Estimator, dict(psi="huber")), (R_Estimator, dict(score="wilcoxon")),
])
def test_location_breakdown_under_20pct_contamination(cls, kwargs):
    # HodgesLehmann/Wilcoxon-R/Huber-M have breakdown points below 50% (~29% for HL,
    # and a fixed-scale Huber M-estimator can wander further under 20% contamination
    # depending on the draw) -- 2.0 still separates "resists" from the raw mean, which
    # blows up to ~2e5 on this same data.
    x = _contaminated(160, 40, seed=8)
    est = cls(**kwargs).fit(x)
    assert abs(est.estimate_) < 2.0


# ======================================================================= scale

def test_mad_matches_scipy():
    x = _RNG.normal(0, 2, 150)
    assert MedianAbsoluteDeviation(n_boot=0).fit(x).estimate_ == pytest.approx(
        float(stats.median_abs_deviation(x, scale="normal")), abs=1e-10)


def test_iqr_scale_matches_scipy():
    x = _RNG.normal(0, 2, 150)
    assert IQR_Scale(n_boot=0).fit(x).estimate_ == pytest.approx(
        float(stats.iqr(x, scale="normal")), abs=1e-10)


@pytest.mark.parametrize("n", [5, 8, 9, 10, 11, 20, 51, 100, 501, 1500])
def test_qn_matches_statsmodels(n):
    x = np.random.default_rng(9).normal(size=n)
    got = Qn_Estimator(n_boot=0).fit(x).estimate_
    expected = float(sms.qn_scale(x))
    assert got == pytest.approx(expected, abs=1e-8)


def _sn_bruteforce(x):
    x = np.asarray(x, dtype=float)
    n = len(x)
    himeds = np.array([np.sort(np.abs(x - x[i]))[n // 2] for i in range(n)])
    return 1.1926 * float(np.sort(himeds)[(n - 1) // 2])


@pytest.mark.parametrize("n", [5, 8, 9, 10, 25, 200])
def test_sn_matches_brute_force(n):
    x = np.random.default_rng(10).normal(size=n)
    got = Sn_Estimator(n_boot=0).fit(x).estimate_
    assert got == pytest.approx(_sn_bruteforce(x), abs=1e-8)


def test_scale_estimators_consistent_at_gaussian():
    x = np.random.default_rng(11).normal(0, 2, 6000)
    for cls in (MedianAbsoluteDeviation, IQR_Scale, Qn_Estimator):
        est = cls(n_boot=0).fit(x).estimate_
        assert abs(est - 2.0) < 0.06
    sn = Sn_Estimator(n_boot=0).fit(x).estimate_
    assert abs(sn - 2.0) < 0.08
    for method in ("biweight", "tau", "huber"):
        est = RobustStd(method, n_boot=0).fit(x).estimate_
        assert abs(est - 2.0) < 0.08


@pytest.mark.parametrize("cls", [MedianAbsoluteDeviation, Qn_Estimator, Sn_Estimator])
def test_scale_breakdown_under_20pct_contamination(cls):
    x = _contaminated(160, 40, scale=1.0, seed=12)
    est = cls(n_boot=0).fit(x).estimate_
    assert 0.5 < est < 2.0


def test_scale_bootstrap_se_finite_and_reproducible():
    x = np.random.default_rng(13).normal(0, 1, 200)
    a = MedianAbsoluteDeviation(n_boot=200, random_state=1).fit(x)
    b = MedianAbsoluteDeviation(n_boot=200, random_state=1).fit(x)
    assert np.isfinite(a.std_error_)
    assert a.std_error_ == b.std_error_


def test_robust_std_location_extras_for_huber_and_tau():
    x = np.random.default_rng(14).normal(5, 1, 300)
    rh = RobustStd("huber", n_boot=0).fit(x)
    rt = RobustStd("tau", n_boot=0).fit(x)
    assert abs(rh.extras_["location_"] - 5.0) < 0.3
    assert abs(rt.extras_["location_"] - 5.0) < 0.3


def test_robust_std_invalid_method_raises():
    with pytest.raises(ValueError):
        RobustStd("bogus")


# =================================================================== regression

@pytest.mark.parametrize("method", ["separate", "joint"])
def test_theilsen_matches_scipy_simple(method):
    x = _RNG.uniform(0, 10, 64)
    y = 2.0 + 3.0 * x + _RNG.normal(0, 1, 64)
    y[:5] += 25
    ts = TheilSenRegression(intercept_method=method).fit(x, y)
    sp = stats.theilslopes(y, x, alpha=0.95, method=method)
    assert ts.coef_[1] == pytest.approx(sp.slope, abs=1e-9)
    assert ts.coef_[0] == pytest.approx(sp.intercept, abs=1e-9)
    lo, hi = ts.extras_["confidence_interval"]
    assert lo == pytest.approx(sp.low_slope, abs=1e-9)
    assert hi == pytest.approx(sp.high_slope, abs=1e-9)


def test_theilsen_matches_scipy_with_ties():
    x = np.round(_RNG.uniform(0, 10, 60))
    y = 1 + 2 * x + _RNG.normal(0, 1, 60)
    ts = TheilSenRegression().fit(x, y)
    sp = stats.theilslopes(y, x)
    assert ts.coef_[1] == pytest.approx(sp.slope, abs=1e-9)
    assert ts.coef_[0] == pytest.approx(sp.intercept, abs=1e-9)


def test_theilsen_multi_predictor_recovers_truth():
    # small-p subsets need (1-contamination)^(p+1) > 0.5 for a comfortable majority
    # of clean subsets to dominate the spatial median -- n=60 at 15% keeps that margin
    # (n=15 at 25%+ makes the estimator's own breakdown boundary the binding constraint).
    rng = np.random.default_rng(15)
    n = 60
    X = rng.uniform(0, 10, (n, 2))
    beta_true = np.array([1.0, 2.0, -1.0])
    y = beta_true[0] + X @ beta_true[1:] + rng.normal(0, 0.3, n)
    out_idx = rng.choice(n, 9, replace=False)
    y[out_idx] += 20
    ts = TheilSenRegression(max_subpopulation=5000, random_state=0).fit(X, y)
    assert np.max(np.abs(ts.coef_ - beta_true)) < 0.3


@pytest.mark.parametrize("method", ["hierarchical", "separate"])
def test_siegel_matches_scipy(method):
    x = _RNG.uniform(0, 10, 55)
    y = 1 + 2 * x + _RNG.normal(0, 1, 55)
    y[:6] += 20
    sg = SiegalRegression(method=method).fit(x, y)
    ss = stats.siegelslopes(y, x, method=method)
    assert sg.coef_[1] == pytest.approx(ss.slope, abs=1e-9)
    assert sg.coef_[0] == pytest.approx(ss.intercept, abs=1e-9)


def test_siegel_multi_predictor_raises():
    X = _RNG.standard_normal((20, 2))
    y = _RNG.standard_normal(20)
    with pytest.raises(ValueError):
        SiegalRegression().fit(X, y)


def test_siegel_regression_is_alias():
    assert SiegelRegression is SiegalRegression


@pytest.mark.parametrize("epsilon", [1.345])
def test_huber_regression_matches_statsmodels(epsilon):
    for seed in range(4):
        rng = np.random.default_rng(seed)
        X = rng.standard_normal((150, 3))
        beta_true = np.array([1.0, 2.0, -1.0, 0.5])
        y = beta_true[0] + X @ beta_true[1:] + rng.standard_t(3, 150) * 0.5
        hr = HuberRegression(epsilon=epsilon).fit(X, y)
        Xc = sm.add_constant(X)
        rlm = sm.RLM(y, Xc, M=sm.robust.norms.HuberT(t=epsilon)).fit(scale_est="mad", cov="H1")
        assert np.max(np.abs(hr.coef_ - rlm.params)) < 1e-4
        assert np.max(np.abs(hr.std_errors_ - rlm.bse) / np.abs(rlm.bse)) < 1e-2
        assert abs(hr.scale_ - rlm.scale) < 1e-4


def test_ransac_recovers_line_under_40pct_outliers():
    rng = np.random.default_rng(16)
    n = 200
    x = rng.uniform(0, 10, n)
    y = 1 + 2 * x + rng.normal(0, 0.5, n)
    n_out = 80
    out_idx = rng.choice(n, n_out, replace=False)
    y[out_idx] += rng.uniform(-50, 50, n_out)
    rs = RANSACRegression(random_state=0, stop_probability=0.999).fit(x, y)
    assert np.max(np.abs(rs.coef_ - np.array([1.0, 2.0]))) < 0.15
    detected = ~rs.inlier_mask_
    assert np.mean(detected[out_idx]) >= 0.9
    assert rs.n_trials_ <= rs.max_trials


def test_ransac_min_samples_too_large_raises():
    x = _RNG.uniform(0, 1, 10)
    y = _RNG.uniform(0, 1, 10)
    with pytest.raises(ValueError):
        RANSACRegression(min_samples=500).fit(x, y)


def test_lts_exact_enumeration_matches_fast():
    rng = np.random.default_rng(17)
    x = rng.uniform(0, 10, 14)
    y = 1 + 2 * x + rng.normal(0, 0.3, 14)
    y[:3] += 15
    lts_exact = LTS_Regression(random_state=0).fit(x, y)
    assert lts_exact.exhaustive_

    # brute-force cross-check of the raw objective
    n, h = 14, lts_exact.h_ if hasattr(lts_exact, "h_") else None
    p = 2
    from stochpylib.robust_statistics.regression import _lts_h
    h = _lts_h(n, p, 0.5)
    best_obj = np.inf
    Xd = np.column_stack([np.ones(n), x])
    for rows in itertools.combinations(range(n), h):
        rows = np.array(rows)
        beta, *_ = np.linalg.lstsq(Xd[rows], y[rows], rcond=None)
        resid2 = (y - Xd @ beta) ** 2
        obj = float(np.sum(np.sort(resid2)[:h]))
        best_obj = min(best_obj, obj)
    assert lts_exact.objective_ == pytest.approx(best_obj, rel=1e-6)


def test_lts_reweight_false_equals_raw():
    x = _RNG.uniform(0, 10, 100)
    y = 1 + 2 * x + _RNG.normal(0, 0.3, 100)
    lts = LTS_Regression(reweight=False, random_state=0).fit(x, y)
    assert np.allclose(lts.coef_, lts.raw_coef_)


def test_lts_clean_data_matches_ols_within_3se():
    rng = np.random.default_rng(18)
    x = rng.standard_normal(500)
    y = 1 + 2 * x + rng.normal(0, 1, 500)
    lts = LTS_Regression(random_state=0).fit(x, y)
    ols = linear_regression(x, y)
    assert np.all(np.abs(lts.coef_ - ols.coef_) < 3 * ols.std_errors_)
    assert abs(lts.scale_ - 1.0) < 0.15


def test_mm_regression_clean_efficiency():
    rng = np.random.default_rng(19)
    n = 300
    x = rng.standard_normal(n)
    y = 1 + 2 * x + rng.normal(0, 1, n)
    mm = MMRegression(random_state=0).fit(x, y)
    ols = linear_regression(x, y)
    assert np.all(np.abs(mm.coef_ - ols.coef_) < 3 * ols.std_errors_)
    ratio = mm.std_errors_ / ols.std_errors_
    assert np.all((ratio > 0.85) & (ratio < 1.4))


def test_mm_and_lts_survive_bad_leverage_where_huber_and_ols_fail():
    rng = np.random.default_rng(20)
    n = 200
    x = rng.uniform(-5, 5, n)
    y = 1 + 2 * x + rng.normal(0, 0.5, n)
    n_lev = 80
    lev_idx = rng.choice(n, n_lev, replace=False)
    x[lev_idx] = rng.uniform(20, 30, n_lev)
    y[lev_idx] = rng.uniform(-10, 10, n_lev)

    mm = MMRegression(random_state=1).fit(x, y)
    lts = LTS_Regression(random_state=1).fit(x, y)
    hub = HuberRegression().fit(x, y)
    ols = linear_regression(x, y)

    assert abs(mm.coef_[1] - 2.0) < 0.1
    assert abs(lts.coef_[1] - 2.0) < 0.1
    assert abs(hub.coef_[1] - 2.0) > 0.5
    assert abs(ols.coef_[1] - 2.0) > 0.5


@pytest.mark.parametrize("cls,kwargs", [
    (TheilSenRegression, {}), (SiegalRegression, {}), (HuberRegression, {}),
    (RANSACRegression, dict(random_state=0)), (LTS_Regression, dict(random_state=0)),
    (MMRegression, dict(random_state=0)),
])
def test_regressor_predict_score_and_result(cls, kwargs):
    x = _RNG.uniform(0, 10, 60)
    y = 1 + 2 * x + _RNG.normal(0, 0.5, 60)
    m = cls(**kwargs).fit(x, y)
    pred = m.predict(np.array([0.0, 5.0]))
    assert pred.shape == (2,)
    score = m.score(x, y)
    assert score > 0.5
    res = m.to_result()
    assert isinstance(res, RegressionResult)
    if not np.all(np.isnan(m.std_errors_)):
        lo, hi = m.conf_int()
        lo, hi = np.asarray(lo, dtype=float), np.asarray(hi, dtype=float)
        finite = np.isfinite(lo) & np.isfinite(hi)
        assert np.all(lo[finite] <= hi[finite])


# =================================================================== covariance

def test_mcd_exact_enumeration_matches_fast():
    rng = np.random.default_rng(21)
    X = rng.standard_normal((12, 2))
    exact = MCD(random_state=0).fit(X)
    fast = MCD(n_trials=250, random_state=1).fit(X)
    assert exact.exhaustive_
    assert set(np.where(exact.raw_support_mask_)[0]) == set(np.where(fast.raw_support_mask_)[0])


def test_mcd_recovers_gaussian_parameters():
    rng = np.random.default_rng(22)
    mu = np.array([1.0, 2.0, 3.0])
    Sigma = np.array([[2, 0.5, 0.1], [0.5, 1.5, 0.2], [0.1, 0.2, 1.0]])
    L = np.linalg.cholesky(Sigma)
    X = mu + rng.standard_normal((1000, 3)) @ L.T
    mcd = MCD(random_state=0).fit(X)
    se = np.sqrt(np.diag(Sigma) / 1000)
    assert np.all(np.abs(mcd.location_ - mu) < 4 * se)
    rel = np.linalg.norm(mcd.covariance_ - Sigma) / np.linalg.norm(Sigma)
    assert rel < 0.15


def test_mcd_flags_planted_outliers():
    rng = np.random.default_rng(23)
    mu = np.zeros(3)
    Sigma = np.eye(3)
    X = mu + rng.standard_normal((1000, 3))
    n_out = 150
    out_idx = rng.choice(1000, n_out, replace=False)
    X[out_idx] += rng.normal(0, 1, (n_out, 3)) * 8 + 20
    mcd = MCD(random_state=1).fit(X)
    mask = mcd.outliers()
    true_out = np.zeros(1000, dtype=bool)
    true_out[out_idx] = True
    assert np.mean(mask[out_idx]) >= 0.95
    assert np.mean(mask[~true_out]) <= 0.05


def test_mve_exact_enumeration_close_to_resampling():
    rng = np.random.default_rng(24)
    X = rng.standard_normal((12, 2))
    exact = MVE(random_state=0).fit(X)
    fast = MVE(n_trials=250, random_state=1).fit(X)
    assert exact.exhaustive_
    assert np.max(np.abs(exact.location_ - fast.location_)) < 1.0


def test_mve_robust_to_20pct_outliers():
    rng = np.random.default_rng(25)
    mu = np.zeros(2)
    X = rng.standard_normal((300, 2))
    n_out = 60
    out_idx = rng.choice(300, n_out, replace=False)
    X[out_idx] += 15
    mve = MVE(n_trials=1500, random_state=2).fit(X)
    assert np.max(np.abs(mve.location_ - mu)) < 1.0


def test_ogk_clean_and_contaminated_recovery():
    rng = np.random.default_rng(26)
    mu = np.array([1.0, 2.0, 3.0])
    Sigma = np.array([[2, 0.5, 0.1], [0.5, 1.5, 0.2], [0.1, 0.2, 1.0]])
    L = np.linalg.cholesky(Sigma)
    X = mu + rng.standard_normal((1000, 3)) @ L.T
    ogk = OGK().fit(X)
    rel = np.linalg.norm(ogk.covariance_ - Sigma) / np.linalg.norm(Sigma)
    assert rel < 0.15

    n_out = 150
    out_idx = rng.choice(1000, n_out, replace=False)
    X2 = X.copy()
    X2[out_idx] += rng.normal(0, 1, (n_out, 3)) * 8 + 20
    ogk2 = OGK().fit(X2)
    rel2 = np.linalg.norm(ogk2.covariance_ - Sigma) / np.linalg.norm(Sigma)
    assert rel2 < 0.30
    for scale in ("qn", "mad"):
        OGK(scale=scale).fit(X)  # runs without error


def test_robust_covariance_dispatcher():
    X = _RNG.standard_normal((200, 3))
    sample = RobustCovariance(method="sample").fit(X)
    assert np.allclose(sample.covariance_, np.cov(X, rowvar=False, ddof=1))
    assert np.max(np.abs(sample.precision_ @ sample.covariance_ - np.eye(3))) < 1e-8
    for method in ("mcd", "mve", "ogk", "huber"):
        rc = RobustCovariance(method=method).fit(X)
        assert rc.covariance_.shape == (3, 3)
        d2 = rc.mahalanobis()
        assert np.all(np.isfinite(d2))


def test_robust_correlation_matches_scipy():
    U = _RNG.uniform(size=(500, 2))
    sp = RobustCorrelation(method="spearman").fit(U)
    kt = RobustCorrelation(method="kendall").fit(U)
    assert sp.correlation_[0, 1] == pytest.approx(float(stats.spearmanr(U[:, 0], U[:, 1]).statistic), abs=1e-9)
    assert kt.correlation_[0, 1] == pytest.approx(float(stats.kendalltau(U[:, 0], U[:, 1]).statistic), abs=1e-9)


def test_robust_correlation_gaussian_rank_and_quadrant():
    rng = np.random.default_rng(27)
    rho = 0.6
    Sigma = np.array([[1.0, rho], [rho, 1.0]])
    L = np.linalg.cholesky(Sigma)
    X = rng.standard_normal((2000, 2)) @ L.T
    gr = RobustCorrelation(method="gaussian_rank").fit(X)
    q = RobustCorrelation(method="quadrant").fit(X)
    mcd_c = RobustCorrelation(method="mcd").fit(X)
    assert abs(gr.correlation_[0, 1] - rho) < 0.05
    assert abs(q.correlation_[0, 1] - rho) < 0.08
    assert abs(mcd_c.correlation_[0, 1] - rho) < 0.1


def test_cov_shrinkage_ledoit_wolf_matches_hand_formula_and_financial_stochastics():
    from stochpylib.financial_stochastics.portfolio import CovarianceEstimation
    rng = np.random.default_rng(28)
    R = rng.standard_normal((100, 4)) @ np.linalg.cholesky(np.eye(4) * 0.5 + 0.5)

    S = np.cov(R, rowvar=False, ddof=1)
    p = 4
    mu_bar = np.trace(S) / p
    target = mu_bar * np.eye(p)
    Xc = R - R.mean(axis=0)
    delta2 = np.sum((S - target) ** 2)
    pi_hat = sum(np.sum((np.outer(Xc[t], Xc[t]) - S) ** 2) for t in range(100)) / 100 ** 2
    beta2 = min(pi_hat, delta2)
    delta_expected = float(np.clip(beta2 / delta2 if delta2 > 0 else 0.0, 0.0, 1.0))

    cs = CovShrinkage(method="ledoit_wolf", target="identity", ddof=1).fit(R)
    assert cs.shrinkage_ == pytest.approx(delta_expected, abs=1e-10)

    ce = CovarianceEstimation(method="ledoit_wolf").fit(R)
    assert cs.shrinkage_ == pytest.approx(ce.shrinkage_, abs=1e-10)
    assert np.max(np.abs(cs.covariance_ - ce.covariance_)) < 1e-10


def test_cov_shrinkage_oas_and_fixed_and_targets():
    rng = np.random.default_rng(29)
    R = rng.standard_normal((60, 5))
    oas = CovShrinkage(method="oas").fit(R)
    assert 0.0 <= oas.shrinkage_ <= 1.0
    fixed = CovShrinkage(method="fixed", shrinkage=0.3).fit(R)
    S = np.cov(R, rowvar=False, ddof=0)
    F = (np.trace(S) / 5) * np.eye(5)
    assert np.allclose(fixed.covariance_, 0.7 * S + 0.3 * F)
    diag = CovShrinkage(method="ledoit_wolf", target="diagonal").fit(R)
    assert np.allclose(diag.target_, np.diag(np.diag(diag.sample_covariance_)))
    cc = CovShrinkage(method="ledoit_wolf", target="constant_correlation").fit(R)
    d = np.sqrt(np.diag(cc.target_))
    off_corr = cc.target_ / np.outer(d, d)
    np.fill_diagonal(off_corr, 0)
    vals = off_corr[off_corr != 0]
    assert np.ptp(vals) < 1e-8


def test_cov_shrinkage_large_n_shrinks_little():
    # shrinkage intensity depends on how far the TRUE covariance is from the target,
    # not on n alone -- a true covariance equal to the identity target correctly keeps
    # delta near 1 even as n grows (full shrinkage is then asymptotically lossless), so
    # this needs off-target correlation to see delta -> 0 with more data.
    rng = np.random.default_rng(30)
    Sigma = np.array([[1.0, 0.5, 0.2], [0.5, 1.0, 0.3], [0.2, 0.3, 1.0]])
    L = np.linalg.cholesky(Sigma)
    R = rng.standard_normal((5000, 3)) @ L.T
    cs = CovShrinkage(method="ledoit_wolf").fit(R)
    assert cs.shrinkage_ < 0.05


# =================================================================== bootstrap

def test_robust_bootstrap_median_se_and_ci_methods():
    x = np.random.default_rng(31).normal(0, 1, 200)
    rb = RobustBootstrap("median", n_boot=3000, random_state=0).fit(x)
    mj = Median().fit(x)
    assert rb.std_error_ == pytest.approx(mj.std_error_, rel=0.3)
    for method in ("percentile", "basic", "normal", "bca"):
        lo, hi = rb.confidence_interval(method=method)
        assert lo < hi


def test_robust_bootstrap_statistic_forms_agree():
    x = np.random.default_rng(32).normal(0, 1, 150)
    a = RobustBootstrap(lambda arr: float(np.median(arr)), n_boot=200, random_state=0).fit(x)
    b = RobustBootstrap(Median(), n_boot=200, random_state=0).fit(x)
    c = RobustBootstrap("median", n_boot=200, random_state=0).fit(x)
    assert a.estimate_ == pytest.approx(b.estimate_) == pytest.approx(c.estimate_)


def test_robust_bootstrap_reproducible():
    x = np.random.default_rng(33).normal(0, 1, 100)
    a = RobustBootstrap("median", n_boot=300, random_state=42).fit(x)
    b = RobustBootstrap("median", n_boot=300, random_state=42).fit(x)
    assert a.std_error_ == b.std_error_

    result = a.to_result()
    assert isinstance(result, EstimateResult)


def test_wild_bootstrap_rademacher_close_to_hc0():
    rng = np.random.default_rng(34)
    n = 400
    x = rng.uniform(0, 10, n)
    sigma_x = 0.2 + 0.3 * x
    y = 1 + 2 * x + rng.normal(0, 1, n) * sigma_x
    wb = WildBootstrap(n_boot=2500, weights="rademacher", random_state=0).fit(x, y)
    hc0 = linear_regression(x, y, cov_type="HC0")
    assert np.all(np.abs(wb.std_errors_ - hc0.std_errors_) / hc0.std_errors_ < 0.15)


@pytest.mark.parametrize("weights", ["rademacher", "mammen", "normal", "webb"])
def test_wild_weight_schemes_mean_zero_var_one(weights):
    from stochpylib.robust_statistics.bootstrap import _WILD_WEIGHTS
    rng = np.random.default_rng(35)
    v = _WILD_WEIGHTS[weights](rng, 500_000)
    assert abs(np.mean(v)) < 0.01
    assert abs(np.var(v) - 1.0) < 0.02


def test_wild_bootstrap_with_huber_model_runs():
    rng = np.random.default_rng(36)
    x = rng.uniform(0, 10, 150)
    y = 1 + 2 * x + rng.normal(0, 1, 150)
    wb = WildBootstrap(model=HuberRegression(), n_boot=200, random_state=0).fit(x, y)
    assert np.all(np.isfinite(wb.std_errors_))
    from statsmodels.api import RLM
    Xc = sm.add_constant(x)
    rlm = sm.RLM(y, Xc, M=sm.robust.norms.HuberT()).fit(scale_est="mad")
    assert np.all(np.abs(wb.std_errors_ - rlm.bse) / rlm.bse < 0.4)


@pytest.mark.parametrize("kind", ["moving", "circular", "nonoverlapping"])
def test_block_bootstrap_ar1_discriminates_from_iid(kind):
    rng = np.random.default_rng(37)
    phi, sigma, n = 0.5, 1.0, 2000
    ar = np.empty(n)
    ar[0] = rng.normal(0, sigma / math.sqrt(1 - phi ** 2))
    for i in range(1, n):
        ar[i] = phi * ar[i - 1] + rng.normal(0, sigma)
    asym_se = math.sqrt(sigma ** 2 * (1 + phi) / ((1 - phi) * n))

    bb = BlockBootstrap(np.mean, kind=kind, n_boot=800, random_state=0).fit(ar)
    ratio = bb.std_error_ / asym_se
    assert 0.75 < ratio < 1.3

    iid = RobustBootstrap(lambda a: float(np.mean(a)), n_boot=800, random_state=0).fit(ar)
    assert iid.std_error_ / asym_se < 0.75


def test_stationary_bootstrap_ar1_ratio():
    rng = np.random.default_rng(38)
    phi, sigma, n = 0.5, 1.0, 2000
    ar = np.empty(n)
    ar[0] = rng.normal(0, sigma / math.sqrt(1 - phi ** 2))
    for i in range(1, n):
        ar[i] = phi * ar[i - 1] + rng.normal(0, sigma)
    asym_se = math.sqrt(sigma ** 2 * (1 + phi) / ((1 - phi) * n))
    sb = StationaryBootstrap(np.mean, n_boot=800, random_state=0).fit(ar)
    assert 0.75 < sb.std_error_ / asym_se < 1.3


def test_block_bootstrap_indices_shapes_and_default_length():
    rng = np.random.default_rng(39)
    x = rng.normal(size=100)
    bb = BlockBootstrap(np.mean, block_length=10, kind="moving", random_state=0).fit(x)
    idx = bb.indices(rng)
    assert idx.shape == (100,)
    assert idx.min() >= 0 and idx.max() < 100

    bb2 = BlockBootstrap(np.mean, random_state=0).fit(np.random.default_rng(1).normal(size=2000))
    assert bb2.block_length_ == max(round(2000 ** (1 / 3)), 2)

    sb = StationaryBootstrap(np.mean, random_state=0).fit(x)
    idx2 = sb.indices(rng, 100)
    assert idx2.shape == (100,)


def test_bootstrap_deterministic_with_seed():
    x = np.random.default_rng(40).normal(size=150)
    for cls, kw in [(BlockBootstrap, dict(random_state=1)), (StationaryBootstrap, dict(random_state=1))]:
        a = cls(np.mean, **kw).fit(x)
        b = cls(np.mean, **kw).fit(x)
        assert a.std_error_ == b.std_error_


# ============================================================== base-class checks

def test_base_classes_present_and_subclassed():
    assert issubclass(TrimmedMean, RobustEstimator)
    assert issubclass(HuberRegression, RobustRegressor)
    assert issubclass(MCD, RobustCovarianceEstimator)
    assert issubclass(BlockBootstrap, Resampler)
    assert issubclass(WildBootstrap, Resampler)


def test_quickstart_example_runs():
    """The vault's Quickstart-Examples.md "Robust Statistics" snippet, verified verbatim."""
    rng = np.random.default_rng(0)
    x = np.concatenate([rng.normal(10, 1, 90), rng.normal(60, 5, 10)])
    assert abs(float(Median().fit(x)) - 10.0) < 1.0
    assert abs(float(HodgesLehmann().fit(x)) - 10.0) < 1.0
    assert abs(float(M_Estimator("huber").fit(x)) - 10.0) < 1.0
    assert 0.5 < float(Qn_Estimator().fit(x)) < 3.0

    t = rng.uniform(0, 10, 100)
    y = 1 + 2 * t + rng.normal(0, 1, 100)
    y[:30] += 40
    ts = TheilSenRegression().fit(t, y)
    mm = MMRegression(random_state=0).fit(t, y)
    assert abs(ts.coef_[1] - 2.0) < 0.3
    assert np.max(np.abs(mm.coef_ - np.array([1.0, 2.0]))) < 0.3

    X = rng.multivariate_normal([0, 0], [[1, .8], [.8, 1]], 200)
    X[:20] += 8
    n_flagged = MCD(random_state=0).fit(X).outliers().sum()
    assert 10 <= n_flagged <= 30

    ar = np.zeros(500)
    e = rng.normal(size=500)
    for i in range(1, 500):
        ar[i] = 0.5 * ar[i - 1] + e[i]
    se = BlockBootstrap(np.mean, random_state=0).fit(ar).std_error_
    assert 0.0 < se < 1.0


def test_estimate_result_and_regression_result_reused():
    x = _RNG.normal(size=100)
    assert isinstance(TrimmedMean().fit(x).to_result(), EstimateResult)
    y = 1 + 2 * x + _RNG.normal(0, 0.1, 100)
    assert isinstance(HuberRegression().fit(x, y).to_result(), RegressionResult)
