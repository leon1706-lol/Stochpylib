"""Tests for stochpylib.timeseries.

Verification strategy:
- linear models: DGP parameter recovery + statsmodels oracle comparisons (exact for
  shared OLS estimators, generous for CSS-vs-ML differences)
- statistical tests: exact statistic equality where formulas match; behavioral checks
  in both directions (unit root / stationarity)
- volatility: GARCH parameter recovery on a known DGP, long-run convergence of the
  variance forecast, leverage detection, multivariate shapes
- state space: exact single-step loglik identity, tracking correlations,
  particle/EKF/UKF nonlinear tracking
- latent/changepoint/spectral: synthetic regime recovery, planted change points,
  Parseval identity, perfect DWT reconstruction, Hilbert quadrature

All randomness is seeded.
"""

import math

import numpy as np
import pytest
from scipy import stats as spt

from stochpylib import timeseries as ts
from stochpylib.timeseries import (
    AR, ARMA, ARIMA, SARIMA, ARFIMA, VAR, VARMA, VECM,
    GARCH, IGARCH, TGARCH, GJRGARCH, EGARCH, APARCH, FIGARCH, MGARCH, DCC_GARCH,
    StateSpaceModel, KalmanFilter, KalmanSmoother, ExtendedKalmanFilter,
    UnscentedKalmanFilter, ParticleFilter, RaoBlackwellFilter,
    HiddenMarkovModel, SwitchingRegression, RegimeSwitching, MixtureAutoregressive,
    ChangePointDetection, BayesianChangePoint, BinarySegmentation, BottomUp, PELT,
    SpectralAnalysis, Periodogram, PowerSpectrum, WaveletTransform,
    CWTTransform, DWTTransform, STFT,
    SeasonalDecomposition, STLDecomposition, X11Decomposition, TrendFilter, HPFilter,
    ForecastResult, TestResult,
    forecast, predict, confidence_bands, backtesting, cross_validation_ts,
    adf_test, kpss_test, pp_test, ljung_box, durbin_watson, arch_test,
    granger_causality, johansen_test,
)
from stochpylib.timeseries._utils import frac_diff_weights
from stochpylib.timeseries.spectral import Hilbert, IDWTTransform
from stochpylib.timeseries.decomposition import (
    SeasonalDecomposition as _SD, STLDecomposition as _STL,
    X11Decomposition as _X11, TrendFilter as _TF, HPFilter as _HPF,
)
from stochpylib.timeseries.spectral import (
    Periodogram as _PG, PowerSpectrum as _PS, STFT as _STFT,
)
from stochpylib.gaussian_processes.kernels import RBFKernel
from stochpylib.gaussian_processes.inference import ExpectationPropagation


def _sm():
    """Import and return statsmodels with the subpackages our oracles need."""
    statsmodels = pytest.importorskip("statsmodels")
    import statsmodels.stats.diagnostic  # noqa: F401
    import statsmodels.tsa.api  # noqa: F401
    import statsmodels.tsa.ar_model  # noqa: F401
    import statsmodels.tsa.arima.model  # noqa: F401
    import statsmodels.tsa.statespace.structural  # noqa: F401
    import statsmodels.tsa.stattools  # noqa: F401
    return statsmodels


# ================================================================ export surface


def test_spec_names_present():
    spec = {
        "AR", "MA", "ARMA", "ARIMA", "SARIMA", "ARFIMA", "VARMA", "VAR", "VECM",
        "ARCH", "GARCH", "IGARCH", "TGARCH", "GJRGARCH", "EGARCH", "APARCH",
        "FIGARCH", "MGARCH", "DCC_GARCH",
        "StateSpaceModel", "KalmanFilter", "KalmanSmoother", "ExtendedKalmanFilter",
        "UnscentedKalmanFilter", "ParticleFilter", "RaoBlackwellFilter",
        "HiddenMarkovModel", "SwitchingRegression", "RegimeSwitching",
        "MixtureAutoregressive",
        "ChangePointDetection", "BayesianChangePoint", "BinarySegmentation",
        "BottomUp", "PELT",
        "SpectralAnalysis", "Periodogram", "PowerSpectrum", "WaveletTransform",
        "CWTTransform", "DWTTransform", "STFT", "Hilbert",
        "SeasonalDecomposition", "STLDecomposition", "X11Decomposition",
        "TrendFilter", "HPFilter",
        "adf_test", "kpss_test", "pp_test", "ljung_box", "durbin_watson",
        "arch_test", "granger_causality", "johansen_test",
        "forecast", "predict", "confidence_bands", "backtesting",
        "cross_validation_ts",
    }
    missing = spec - set(ts.__all__)
    assert not missing, f"missing exports: {missing}"
    import stochpylib
    assert stochpylib.timeseries is ts


# ================================================================ linear models


def test_ar_recovery_and_statsmodels_exact():
    sm = _sm()
    rng = np.random.default_rng(42)
    y = 1.0 + 0.6 * rng.standard_normal(4000)
    for t in range(2, len(y)):
        y[t] += 0.7 * y[t - 1] - 0.25 * y[t - 2]
    ar = ts.AR(2).fit(y)
    assert np.allclose(ar.ar_coefs_, [0.7, -0.25], atol=0.06)
    assert abs(ar.intercept_ - 1.0) < 0.15
    ref = sm.tsa.ar_model.AutoReg(y, lags=2, old_names=False).fit()
    assert np.allclose(ref.params[1:], ar.ar_coefs_, rtol=1e-8)


def test_ar_forecast_converges_to_unconditional_mean():
    rng = np.random.default_rng(43)
    y = np.zeros(1500)
    for t in range(1, len(y)):
        y[t] = 1.5 + 0.5 * y[t - 1] + rng.standard_normal()
    ar = ts.AR(1).fit(y)
    fc = ar.forecast(60)
    uncond = ar.intercept_ / (1 - ar.ar_coefs_[0])
    assert abs(fc.mean[-1] - uncond) < 0.05
    assert abs(fc.std[0] - math.sqrt(ar.sigma2_)) < 1e-12
    stat_sigma = math.sqrt(ar.sigma2_ / (1 - ar.ar_coefs_[0] ** 2))
    assert fc.std[-1] < stat_sigma + 0.05


def test_ma1_recovery():
    rng = np.random.default_rng(44)
    theta = 0.65
    e = rng.standard_normal(6000)
    y = 2.0 + e[1:] + theta * e[:-1]
    ma = ts.MA(1).fit(y)
    assert abs(ma.ma_coefs_[0] - theta) < 0.08
    assert abs(ma.intercept_ - 2.0) < 0.08


def test_arma11_recovery_and_loose_oracle():
    sm = _sm()
    rng = np.random.default_rng(45)
    phi, th = 0.6, 0.4
    e = rng.standard_normal(8500)
    y = np.zeros(len(e))
    for t in range(1, len(e)):
        y[t] = 5.0 + phi * y[t - 1] + e[t] + th * e[t - 1]
    y = y[500:]
    arma = ts.ARMA(1, 1).fit(y)
    assert abs(arma.ar_coefs_[0] - phi) < 0.12
    assert abs(arma.ma_coefs_[0] - th) < 0.12
    sm_fit = sm.tsa.arima.model.ARIMA(y, order=(1, 0, 1), trend="c").fit()
    assert abs(sm_fit.params[1] - arma.ar_coefs_[0]) < 0.1
    assert abs(abs(sm_fit.params[2]) - abs(arma.ma_coefs_[0])) < 0.1


def test_arima110_continues_linear_trend():
    rng = np.random.default_rng(46)
    trend = np.arange(1, 1500) * 0.5 + 3
    yi = trend + 0.8 * rng.standard_normal(len(trend))
    arima = ts.ARIMA(1, 1, 0).fit(yi)
    fc = arima.forecast(10)
    slope = float(np.mean(np.diff(fc.mean)))
    assert abs(slope - 0.5) < 0.05
    assert abs(fc.mean[0] - yi[-1]) < 1.5


def test_sarima_seasonal_recovery():
    rng = np.random.default_rng(47)
    Phi = 0.7
    e = rng.standard_normal(6004)
    ys = np.zeros(len(e))
    for t in range(4, len(e)):
        ys[t] = e[t] + 0.5 * e[t - 1] + Phi * ys[t - 4]
    ys = ys[4:]
    sar = ts.SARIMA(0, 0, 1, 1, 0, 0, 4).fit(ys)
    assert abs(sar.ar_coefs_seasonal_[0] - Phi) < 0.12
    assert abs(sar.ma_coefs_[0] - 0.5) < 0.12


def test_arfima_roundtrip_whitens_and_forecasts():
    d = 0.35
    g_inv = frac_diff_weights(-d)
    eps = np.random.default_rng(48).standard_normal(30000)
    frac_series = np.convolve(eps, g_inv)[:30000]
    filtered = np.convolve(frac_series, frac_diff_weights(d), mode="valid")
    lag1 = np.corrcoef(filtered[:-1], filtered[1:])[0, 1]
    assert abs(lag1) < 0.08
    fit = ts.ARFIMA(1, d, 0).fit(frac_series[:20000])
    fc = fit.forecast(5)
    assert len(np.atleast_1d(fc.mean)) == 5


def test_var1_recovery_and_statsmodels_exact():
    sm = _sm()
    rng = np.random.default_rng(49)
    A1 = np.array([[0.55, 0.1], [0.0, 0.4]])
    inn = rng.multivariate_normal([0, 0], [[1.0, 0.3], [0.3, 0.8]], size=5000)
    yv = np.zeros((5000, 2))
    for t in range(1, 5000):
        yv[t] = A1 @ yv[t - 1] + inn[t]
    varm = ts.VAR(1).fit(yv)
    assert np.allclose(varm.coef_matrices_[0], A1, atol=0.05)
    from statsmodels.tsa.vector_ar.var_model import VAR as SM_VAR

    smv = SM_VAR(yv).fit(maxlags=1)
    assert np.allclose(smv.coefs[0], varm.coef_matrices_[0], rtol=1e-8)


def test_varma_smoke():
    rng = np.random.default_rng(50)
    inn = rng.multivariate_normal([0, 0], [[1.0, 0.3], [0.3, 0.8]], size=1200)
    yv = np.zeros((1200, 2))
    for t in range(1, 1200):
        yv[t] = 0.5 * yv[t - 1] + inn[t]
    vm = ts.VARMA(1, 1).fit(yv)
    assert vm.coef_matrices_[0].shape == (2, 2)


def test_vecm_cointegration_recovery():
    rng = np.random.default_rng(51)
    Pi_true = np.array([[-0.15], [0.12]]) @ np.array([[1.0, -0.8]])
    yn = np.zeros((6000, 2))
    inn = rng.multivariate_normal(np.zeros(2), [[0.5, 0.1], [0.1, 0.4]], size=6000)
    for t in range(1, 6000):
        yn[t] = yn[t - 1] + Pi_true @ yn[t - 1] + inn[t]
    vecm = ts.VECM(rank=1, p=1).fit(yn)
    rel_err = np.linalg.norm(vecm.Pi_mat_ - Pi_true) / np.linalg.norm(Pi_true)
    assert rel_err < 0.25
    beta_dir = vecm.beta_[:, 0] / vecm.beta_[0, 0]
    assert abs(beta_dir[1] - (-0.8)) < 0.1


# ================================================================ tests submodule


def test_adf_stat_matches_statsmodels_fixed_lag():
    sm = _sm()
    rng = np.random.default_rng(52)
    wn = rng.standard_normal(500)
    ours = ts.adf_test(wn, max_lag=2, regression="c")
    ref = sm.tsa.stattools.adfuller(wn, maxlag=2, regression="c", autolag=None)
    assert abs(ours.statistic - ref[0]) < 1e-8


@pytest.mark.parametrize("kind,expect_rejects", [("noise", True), ("walk", False)])
def test_adf_behavioral(kind, expect_rejects):
    rng = np.random.default_rng(53)
    data = rng.standard_normal(500) if kind == "noise" else \
        np.cumsum(rng.standard_normal(500))
    res = ts.adf_test(data)
    rejects = res.pvalue is not None and res.pvalue < 0.05
    rejects = rejects or res.statistic < res.critical_values["5%"]
    assert rejects == expect_rejects


def test_kpss_behavioral_both_directions():
    rng = np.random.default_rng(54)
    wn = rng.standard_normal(500)
    rw = np.cumsum(rng.standard_normal(500))
    assert kpss_test(wn).pvalue >= 0.10 - 1e-9
    assert kpss_test(rw).pvalue <= 0.01 + 1e-9


def test_pp_test_on_random_walk_fails_to_reject():
    rng = np.random.default_rng(55)
    rw = np.cumsum(rng.standard_normal(500))
    assert pp_test(rw).pvalue > 0.05


def test_ljung_box_matches_statsmodels():
    sm = _sm()
    rng = np.random.default_rng(56)
    wn = rng.standard_normal(400)
    lb = ts.ljung_box(wn, lags=10)
    ref = sm.stats.diagnostic.acorr_ljungbox(wn, lags=[10], return_df=True)
    assert abs(lb.statistics[-1] - float(ref["lb_stat"].iloc[0])) < 1e-8


def test_durbin_watson_bounds():
    rng = np.random.default_rng(57)
    assert abs(durbin_watson(rng.standard_normal(1000)) - 2.0) < 0.25
    assert durbin_watson(np.arange(800, dtype=float)) < 0.1


def test_arch_test_detects_and_accepts():
    rng = np.random.default_rng(58)
    e = np.zeros(3000)
    v = np.ones(3000)
    for t in range(1, 3000):
        v[t] = 0.2 + 0.75 * e[t - 1] ** 2 + 0.15 * v[t - 1]
        e[t] = np.sqrt(v[t]) * rng.standard_normal()
    assert arch_test(e, lags=12).pvalue < 0.01
    assert arch_test(rng.standard_normal(3000), lags=12).pvalue > 0.05


def test_granger_causality_directionality():
    rng = np.random.default_rng(59)
    xg = rng.standard_normal(2000)
    yg = np.zeros(2000)
    for t in range(1, 2000):
        yg[t] = 0.5 * yg[t - 1] + 0.6 * xg[t - 1] + 0.8 * rng.standard_normal()
    fwd = granger_causality(xg, yg, max_lag=2)
    bwd = granger_causality(yg, xg, max_lag=2)
    assert fwd[1].pvalue < 0.001
    assert bwd[1].pvalue > 0.01


def test_johansen_rank_decisions_on_cointegrated_pair():
    rng = np.random.default_rng(60)
    Pi_true = np.array([[-0.15], [0.12]]) @ np.array([[1.0, -0.8]])
    yci = np.zeros((800, 2))
    inn = rng.multivariate_normal(np.zeros(2), [[0.5, 0.1], [0.1, 0.4]], size=800)
    for t in range(1, 800):
        yci[t] = yci[t - 1] + Pi_true @ yci[t - 1] + inn[t]
    joh = ts.johansen_test(yci, p=1)
    r0, r1 = joh["trace"][0], joh["trace"][1]
    assert r0.statistic > r0.critical_values["95%"]
    assert r1.statistic <= r1.critical_values["95%"] * 1.3


# ================================================================ volatility


def test_garch_recovery_forecast_convergence_and_simulation():
    rng = np.random.default_rng(61)
    omega_t, alpha_t, beta_t = 0.05e-4, 0.12, 0.82
    n = 3000
    s2 = np.empty(n)
    eps = np.zeros(n)
    s2[0] = omega_t / (1 - alpha_t - beta_t)
    for t in range(1, n):
        eps[t] = np.sqrt(s2[t - 1]) * rng.standard_normal()
        s2[t] = omega_t + alpha_t * eps[t - 1] ** 2 + beta_t * s2[t - 1]
    rets = eps[100:]

    g = ts.GARCH(1, 1).fit(rets)
    assert abs(g.persistence_) < 1.0
    assert abs(g.params_["alpha"][0] - alpha_t) < 0.06
    assert abs(g.params_["beta"][0] - beta_t) < 0.10

    vol = g.conditional_volatility()
    assert np.all(vol > 0) and np.all(np.isfinite(vol))

    fc = g.forecast(10)
    lr = g.params_["omega"] / (1.0 - g.persistence_)
    assert abs(fc.mean[-1] - lr) / lr < 0.02

    sim = g.simulate(2000, random_state=5)
    assert len(sim) == 2000 and np.all(np.isfinite(sim))


def test_leverage_gamma_positive_on_asymmetric_data():
    rng = np.random.default_rng(62)
    lev = np.zeros(2500)
    v = 0.3e-4
    for t in range(1, 2500):
        shock = rng.standard_normal()
        lev[t] = np.sqrt(v) * shock
        v = 0.05e-4 + 0.06 * lev[t - 1] ** 2 + 0.90 * v + \
            (0.25e-4 if lev[t - 1] < 0 else 0.0)
    gjr = ts.GJRGARCH(1, 1).fit(lev)
    assert gjr.params_["gamma"][0] > 0.005
    tg = ts.TGARCH(0, 1).fit(lev)
    assert np.all(np.isfinite(tg.params_["gamma"]))


def test_igarch_enforces_unit_persistence():
    rng = np.random.default_rng(63)
    rets = 0.01 * rng.standard_normal(1500)
    ig = ts.IGARCH(1, 1).fit(rets)
    assert abs(ig.persistence_ - 1.0) < 1e-9


@pytest.mark.parametrize("name", ["EGARCH", "APARCH", "FIGARCH"])
def test_exotic_volatility_models_fit_smoke(name):
    rng = np.random.default_rng(64)
    rets = 0.01 * rng.standard_normal(1200)
    if name == "EGARCH":
        model = ts.EGARCH().fit(rets)
    elif name == "APARCH":
        model = ts.APARCH(1, 1).fit(rets)
    else:
        model = ts.FIGARCH(d=0.45).fit(rets)
    assert np.all(model.sigma2_ > 0)
    fc = model.forecast(5)
    assert len(np.atleast_1d(fc.mean)) == 5


def test_mgarch_and_dcc_shapes_and_bounds():
    rng = np.random.default_rng(65)
    base = 0.01 * rng.standard_normal(1500)
    Y = np.column_stack([base, base * 0.8 + 0.3 * rng.standard_normal(1500)])
    mg = ts.MGARCH().fit(Y)
    cov_path = mg.conditional_covariance()
    assert cov_path.shape == (1500, 2, 2)
    dcc = ts.DCC_GARCH().fit(Y)
    rho = dcc.dynamic_correlations_
    assert np.all(np.abs(rho[:, 0, 1]) <= 1.0)
    fc_covs = mg.forecast(5)["covariance_forecast"]
    assert fc_covs.shape == (5, 2, 2)


# ================================================================ state space


def test_kalman_single_step_loglik_exact_identity():
    kf1 = KalmanFilter(F=[[1.0]], H=[[1.0]], Q=[[0.01]], R=[[1.0]],
                       x0=[0.0], P0=[[1.0]])
    kf1.fit(np.array([0.5]))
    expected = -0.5 * (math.log(2 * math.pi * 2.0) + 0.25 / 2.0)
    assert abs(kf1.loglik_ - expected) < 1e-12


def test_kalman_tracks_local_level_and_smooth_improves():
    rng = np.random.default_rng(66)
    latent = np.zeros(500)
    obs = np.empty(500)
    for t in range(1, 500):
        latent[t] = latent[t - 1] + 0.1 * rng.standard_normal()
        obs[t] = latent[t] + rng.standard_normal()
    kf = KalmanFilter(F=[[1.0]], H=[[1.0]], Q=[[0.01]], R=[[1.0]]).fit(obs)
    corr_filter = np.corrcoef(kf.filtered_means_.ravel(), latent)[0, 1]
    assert corr_filter > 0.85
    smoothed = kf.smooth()
    corr_smooth = np.corrcoef(smoothed.smoothed_means.ravel(), latent)[0, 1]
    assert corr_smooth >= corr_filter - 1e-9
    fc = kf.forecast(5)
    assert np.allclose(fc.mean[1:], fc.mean[:-1])

    # statsmodels sanity: their MLE refits variances -> allow a few loglik units
    sm = _sm()
    ll_ref = sm.tsa.statespace.structural.UnobservedComponents(
        obs, level="local level"
    ).fit(disp=False).llf
    assert abs(kf.loglik_ - ll_ref) < 5.0


def test_ekf_ukf_track_nonlinear_observation():
    rng = np.random.default_rng(67)
    true_x = np.abs(np.cumsum(0.05 * rng.standard_normal(400))) + 0.5
    y_nl = true_x**2 + 0.25 * rng.standard_normal(400)
    common = dict(Q=0.0025 * np.eye(1), R=0.25 * np.eye(1),
                  x0=np.array([1.0]), P0=1.0 * np.eye(1))
    ekf = ExtendedKalmanFilter(f=lambda x: x.copy(),
                               h=lambda x: np.array([x[0] ** 2]), **common).fit(y_nl)
    ukf = UnscentedKalmanFilter(f=lambda x: x.copy(),
                                h=lambda x: np.array([x[0] ** 2]), **common).fit(y_nl)
    for est in (ekf, ukf):
        corr = np.corrcoef(est.filtered_means_.ravel(), true_x)[0, 1]
        assert corr > 0.8


def test_particle_filter_tracks_local_level():
    rng = np.random.default_rng(68)
    latent = np.cumsum(0.1 * rng.standard_normal(300))
    obs = latent + rng.standard_normal(300)
    pf = ParticleFilter(
        n_particles=3000,
        transition_sampler=lambda p, r: p + 0.1 * r.standard_normal(p.shape),
        observation_logpdf=lambda p, y: -0.5 * ((p.ravel() - y) ** 2),
        initial_sampler=lambda r: r.standard_normal(3000),
        random_state=17,
    ).fit(obs)
    corr = np.corrcoef(pf.filtered_means_[:, 0], latent)[0, 1]
    assert corr > 0.85
    assert np.isfinite(pf.loglik_)
    assert pf.effective_sample_sizes_.min() > 0


def test_rao_blackwell_weights_are_valid_probabilities():
    rng = np.random.default_rng(69)
    rb = RaoBlackwellFilter(modes=[
        {"F": [[1.0]], "H": [[1.0]], "Q": [[0.02]], "R": [[0.5]]},
        {"F": [[1.02]], "H": [[1.0]], "Q": [[0.08]], "R": [[2.0]]},
    ])
    obs_rb = []
    lv = 0.0
    state = 0
    for t in range(600):
        if t > 0 and rng.random() < 0.03:
            state = 1 - state
        lv = (1.0 if state == 0 else 1.02) * lv \
            + math.sqrt([0.02, 0.08][state]) * rng.standard_normal()
        obs_rb.append(lv + math.sqrt([0.5, 2.0][state]) * rng.standard_normal())
    rb.fit(np.asarray(obs_rb))
    w = rb.mode_weights_
    assert w.shape == (600, 2)
    assert np.all((w >= 0) & (w <= 1))
    assert np.allclose(w.sum(axis=1), 1.0)


# ================================================================ latent


def test_hmm_decode_agreement_and_means():
    rng = np.random.default_rng(70)
    states = np.zeros(3000, dtype=int)
    yh = np.zeros(3000)
    for t in range(3000):
        if t > 0:
            states[t] = states[t - 1] if rng.random() < 0.95 else 1 - states[t - 1]
        yh[t] = [-1.0, 2.0][states[t]] + 0.7 * rng.standard_normal()
    hmm = HiddenMarkovModel(n_states=2).fit(yh)
    dec = hmm.decode()
    agree = max(float(np.mean(dec == states)), float(np.mean((1 - dec) == states)))
    assert agree > 0.90
    recovered = sorted(hmm.means_)
    assert abs(recovered[0] - (-1.0)) < 0.3 and abs(recovered[1] - 2.0) < 0.3
    assert np.isfinite(hmm.score())


def test_switching_regression_slope_recovery():
    rng = np.random.default_rng(71)
    ns = 2000
    Xr = rng.standard_normal((ns, 1))
    sr = np.zeros(ns, dtype=int)
    yr = np.zeros(ns)
    for t in range(1, ns):
        sr[t] = sr[t - 1] if rng.random() < 0.97 else 1 - sr[t - 1]
        slope = 1.0 if sr[t] == 0 else -2.0
        yr[t] = slope * Xr[t, 0] + 0.5 * rng.standard_normal()
    sw = SwitchingRegression(n_regimes=2).fit(Xr, yr)
    slopes = sorted(np.array(sw.coefficients_)[:, 1].ravel())
    assert abs(slopes[0] - (-2.0)) < 0.35 and abs(slopes[1] - 1.0) < 0.35
    assert sw.predict(Xr[:50]).shape == (50,)


def test_regime_switching_ar_and_mixture_ar_run():
    rng = np.random.default_rng(72)
    ns = 2000
    Xr = rng.standard_normal((ns, 1))
    sr = np.zeros(ns, dtype=int)
    yr = np.zeros(ns)
    for t in range(1, ns):
        sr[t] = sr[t - 1] if rng.random() < 0.97 else 1 - sr[t - 1]
        yr[t] = (-2.0 if sr[t] else 1.0) * Xr[t, 0] + 0.5 * rng.standard_normal()
    rs = RegimeSwitching(p=1, n_states=2).fit(yr)
    assert len(rs.ar_coefficients_) == 2
    mix = MixtureAutoregressive(k=2, p=1).fit(yr)
    assert abs(mix.weights_.sum() - 1.0) < 1e-6


# ================================================================ changepoint


def test_planted_change_points_found_by_all_detectors():
    rng = np.random.default_rng(73)
    truth = [200, 400]
    ycp = np.concatenate([
        rng.standard_normal(200),
        rng.standard_normal(200) + 3.0,
        rng.standard_normal(200) - 1.5,
    ])
    pelt_res = ChangePointDetection(ycp, method="pelt")
    pts = sorted(pelt_res.points)[:2]
    assert all(abs(p - t) <= 10 for p, t in zip(pts, truth))

    bs_res = ChangePointDetection(ycp, method="binseg")
    assert len(bs_res.points) == 2 and all(abs(p - t) <= 10 for p, t in zip(bs_res.points, truth))

    bu_res = BottomUp(ycp)
    assert len(bu_res.points) == 2 and all(abs(p - t) <= 15 for p, t in zip(bu_res.points, truth))


def test_bocpd_detects_at_least_one_change():
    rng = np.random.default_rng(74)
    ycp = np.concatenate([
        rng.standard_normal(200),
        rng.standard_normal(200) + 3.0,
        rng.standard_normal(200) - 1.5,
    ])
    bocpd = ts.BayesianChangePoint(hazard_rate=0.02, threshold=0.3).fit(ycp)
    b_pts = bocpd.result_.points
    assert len(b_pts) >= 1 and any(
        abs(p - 200) <= 15 or abs(p - 400) <= 15 for p in b_pts
    )


# ================================================================ spectral


def test_periodogram_finds_both_tones():
    rng = np.random.default_rng(75)
    fs = 100.0
    t = np.arange(2000) / fs
    sig = np.sin(2 * np.pi * 10.0 * t) + 0.5 * np.sin(2 * np.pi * 25.0 * t) \
        + 0.3 * rng.standard_normal(len(t))
    freqs, power = PowerSpectrum(sig[:100], fs=fs, nperseg=64)
    peaks = freqs[np.argsort(power)[-4:]]
    assert min(abs(peaks - 10)) < 1.0 and min(abs(peaks - 25)) < 1.0


def test_parseval_identity_and_dominant_frequency():
    rng = np.random.default_rng(76)
    fs = 100.0
    t = np.arange(2000) / fs
    sig = np.sin(2 * np.pi * 10.0 * t) + 0.3 * rng.standard_normal(len(t))
    sa = SpectralAnalysis(sig, fs=fs)
    tp = sa.total_power()
    var_direct = float(((sig - sig.mean()) ** 2).mean())
    assert abs(tp["time_domain"] - tp["frequency_domain"]) / max(var_direct, 1e-12) < 0.05
    assert abs(sa.dominant_frequency() - 10.0) < 1.0


def test_cwt_ridge_near_known_frequency():
    fs = 100.0
    t = np.arange(1500) / fs
    sig = np.sin(2 * np.pi * 10.0 * t)
    scales = np.geomspace(1.0, 40.0, 30)
    scales_c, coeffs = WaveletTransform(sig, scales=scales, fs=fs, kind="cwt")
    best_scale = scales_c[int(np.argmax(np.abs(coeffs).mean(axis=1)))]
    est_freq = 6.0 * fs / (2 * np.pi * best_scale)
    assert abs(est_freq - 10.0) / 10.0 < 0.35


@pytest.mark.parametrize("wavelet", ["haar", "db2"])
def test_dwt_idwt_perfect_reconstruction(wavelet):
    rng = np.random.default_rng(77)
    x_test = rng.standard_normal(1024)
    d = DWTTransform(x_test, wavelet=wavelet, level=4)
    rec = IDWTTransform(d)
    assert np.allclose(rec[: len(x_test)], x_test, atol=1e-10)


def test_stft_shape_and_frame_dominance():
    fs = 100.0
    t = np.arange(1000) / fs
    sig = np.sin(2 * np.pi * 10.0 * t)
    sf, stimes, spec = STFT(sig, fs=fs, window_len=128, hop=64)
    assert spec.shape[0] == len(sf) and spec.shape[1] > 5
    dom_bin = int(np.argmax(spec[:, 3]))
    assert abs(sf[dom_bin] - 10.0) <= fs / 128 * 1.5


def test_hilbert_quadrature_exact_on_integer_cycles():
    fs = 100.0
    n = 700  # 7 Hz * 700 / 100 = 49 integer cycles
    x_cos = np.cos(2 * np.pi * 7.0 * np.arange(n) / fs)
    z = Hilbert(x_cos)
    sine = np.sin(2 * np.pi * 7.0 * np.arange(n) / fs)
    assert np.corrcoef(z.imag, sine)[0, 1] > 0.99
    assert np.allclose(np.abs(z), 1.0, atol=1e-6)


# ================================================================ forecasting


def test_forecast_dispatch_alias_and_bands():
    rng = np.random.default_rng(78)
    data = rng.standard_normal(600)
    fitted = ts.AR(2).fit(data)
    fc1 = forecast(fitted, 10)
    fc2 = predict(fitted, 10)
    assert np.array_equal(fc1.mean, fc2.mean)
    lo, hi = confidence_bands(fc1, level=0.95)
    assert np.all(lo <= fc1.mean) and np.all(hi >= fc1.mean)


def test_backtesting_rmse_matches_sigma_and_grows_with_horizon():
    rng = np.random.default_rng(79)
    y = np.zeros(1500)
    for t in range(1, 1500):
        y[t] = 0.6 * y[t - 1] + rng.standard_normal()
    bt = backtesting(y, lambda: ts.AR(1), horizon=3, min_train=800, step=4)
    assert abs(bt.rmse[0] - 1.0) < 0.2  # innovation sigma = 1
    assert bt.rmse[-1] > bt.rmse[0]


def test_cross_validation_ts_fold_structure():
    rng = np.random.default_rng(80)
    y = rng.standard_normal(1200)
    folds = cross_validation_ts(y, lambda: ts.AR(1), n_splits=4, horizon=1, min_train=700)
    assert len(folds) == 4
    assert all(f["test_points"] > 0 for f in folds)


# ================================================================ misc


def test_determinism_same_seed_bitwise():
    rng = np.random.default_rng(81)
    y = rng.standard_normal(500)
    m1 = ts.ARMA(1, 1).fit(y)
    m2 = ts.ARMA(1, 1).fit(y)
    assert m1.ar_coefs_[0] == m2.ar_coefs_[0]
    f1 = forecast(m1, 5)
    f2 = forecast(m2, 5)
    assert np.array_equal(f1.mean, f2.mean)


def test_forecast_result_contract():
    rng = np.random.default_rng(82)
    res = ts.AR(1).fit(rng.standard_normal(400)).forecast(5)
    assert isinstance(res, ForecastResult)
    lo, hi = res.confidence_interval(0.99)
    assert np.all(lo <= res.mean) and np.all(hi >= res.mean)


def test_selftest_suite_includes_timeseries_checks():
    from stochpylib import selftest as selftest_mod

    assert selftest_mod.run() == 0

