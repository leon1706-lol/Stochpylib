"""End-to-end API sweep for stochpylib.timeseries: one realistic exercise per public name.
``test_every_public_name_is_exercised`` fails the moment a name ships without one;
``test_exercise[<name>]`` runs each exercise as its own pytest case."""

import functools

import numpy as np
import pytest

from stochpylib import timeseries as mod

EXERCISES = {}


def exercise(name):
    def deco(fn):
        EXERCISES[name] = fn
        return fn
    return deco


@functools.lru_cache(maxsize=None)
def _ar1(n=600, phi=0.6, seed=0):
    rng = np.random.default_rng(seed)
    y = np.zeros(n)
    for t in range(1, n):
        y[t] = phi * y[t - 1] + rng.standard_normal()
    return y


@functools.lru_cache(maxsize=None)
def _garch_returns(n=1200, seed=1):
    rng = np.random.default_rng(seed)
    r = np.zeros(n)
    s2 = 0.0004
    for t in range(1, n):
        s2 = 1e-5 + 0.1 * r[t - 1] ** 2 + 0.85 * s2
        r[t] = np.sqrt(s2) * rng.standard_normal()
    return r


@functools.lru_cache(maxsize=None)
def _bivariate(n=500, seed=2):
    rng = np.random.default_rng(seed)
    Y = np.zeros((n, 2))
    A = np.array([[0.5, 0.1], [0.0, 0.4]])
    for t in range(1, n):
        Y[t] = A @ Y[t - 1] + rng.standard_normal(2)
    return Y


def _forecast_ok(fc, horizon):
    m, s = np.asarray(fc.mean, dtype=float), np.asarray(fc.std, dtype=float)
    assert m.shape[0] == horizon and np.all(np.isfinite(m)) and np.all(s >= 0)
    lo, hi = fc.confidence_interval(0.9)
    assert np.all(np.asarray(lo) <= np.asarray(hi))


def _linear_workflow(model, y, horizon=5):
    model.fit(y)
    r = np.asarray(model.residuals(), dtype=float)
    assert np.all(np.isfinite(r))
    _forecast_ok(model.forecast(horizon), horizon)
    return model


@exercise("AR")
def _ar():
    m = _linear_workflow(mod.AR(1), _ar1())
    assert abs(float(np.atleast_1d(m.ar_coefs_)[0]) - 0.6) < 0.1
    assert np.asarray(m.simulate(n=50, random_state=0)).shape == (50,)


@exercise("MA")
def _ma():
    _linear_workflow(mod.MA(1), _ar1())


@exercise("ARMA")
def _arma():
    _linear_workflow(mod.ARMA(1, 1), _ar1())


@exercise("ARIMA")
def _arima():
    m = _linear_workflow(mod.ARIMA(1, 1, 0), np.cumsum(_ar1()))
    assert np.asarray(m.simulate(n=30, random_state=1)).shape == (30,)


@exercise("SARIMA")
def _sarima():
    y = _ar1() + np.tile([1.0, -1.0, 0.5, -0.5], 150)
    _linear_workflow(mod.SARIMA(1, 0, 0, 1, 0, 0, 4), y)


@exercise("ARFIMA")
def _arfima():
    _linear_workflow(mod.ARFIMA(1, 0.3, 0), _ar1())


@exercise("VAR")
def _var():
    m = mod.VAR(1).fit(_bivariate())
    _forecast_ok(m.forecast(3), 3)
    assert np.asarray(m.residuals()).shape[1] == 2 and np.asarray(m.simulate(n=20, random_state=2)).shape == (20, 2)


@exercise("VARMA")
def _varma():
    m = mod.VARMA(1, 1).fit(_bivariate()[:300])
    _forecast_ok(m.forecast(3), 3)


@exercise("VECM")
def _vecm():
    rng = np.random.default_rng(3)
    common = np.cumsum(rng.standard_normal(400))
    Y = np.column_stack([common + rng.standard_normal(400) * 0.3, 0.8 * common + rng.standard_normal(400) * 0.3])
    m = mod.VECM(rank=1, p=1).fit(Y)
    _forecast_ok(m.forecast(3), 3)
    assert np.asarray(m.residuals()).shape[1] == 2


def _garch_workflow(model):
    model.fit(_garch_returns())
    vol = np.asarray(model.conditional_volatility(), dtype=float)
    assert np.all(vol > 0) and np.all(np.isfinite(model.standardized_residuals()))
    _forecast_ok(model.forecast(5), 5)
    assert np.asarray(model.simulate(n=40, random_state=4)).shape == (40,)
    return model


@exercise("ARCH")
def _arch():
    _garch_workflow(mod.ARCH(1))


@exercise("GARCH")
def _garch():
    m = _garch_workflow(mod.GARCH(1, 1))
    assert 0.5 < m.persistence_ < 1.0


for _name, _ctor in (("IGARCH", lambda: mod.IGARCH(1, 1)), ("TGARCH", lambda: mod.TGARCH(1, 1)),
                     ("GJRGARCH", lambda: mod.GJRGARCH(1, 1)), ("EGARCH", lambda: mod.EGARCH()),
                     ("APARCH", lambda: mod.APARCH(1, 1)), ("FIGARCH", lambda: mod.FIGARCH())):
    EXERCISES[_name] = (lambda c: (lambda: _garch_workflow(c())))(_ctor)


def _mgarch_workflow(model):
    R = _bivariate()[:400] * 0.01
    model.fit(R)
    H = np.asarray(model.conditional_covariance(), dtype=float)
    assert H.shape[1:] == (2, 2) and np.all(np.isfinite(H))
    fc = model.forecast(3)
    assert np.asarray(fc["covariance_forecast"]).shape == (3, 2, 2)


@exercise("MGARCH")
def _mgarch():
    _mgarch_workflow(mod.MGARCH())


@exercise("DCC_GARCH")
def _dcc():
    _mgarch_workflow(mod.DCC_GARCH())


@exercise("StateSpaceModel")
def _ssm():
    ssm = mod.StateSpaceModel(F=[[1.0]], H=[[1.0]], Q=[[0.01]], R=[[1.0]])
    assert ssm.F.shape == (1, 1)


@exercise("KalmanFilter")
def _kf():
    kf = mod.KalmanFilter(F=[[1.0]], H=[[1.0]], Q=[[0.01]], R=[[1.0]]).fit(np.cumsum(np.random.default_rng(5).standard_normal(200)))
    assert np.isfinite(kf.loglik_) and kf.filtered_means_.shape == (200, 1)
    sm = kf.smooth()
    assert sm.smoothed_means.shape == (200, 1)
    _forecast_ok(kf.forecast(3), 3)
    assert np.all(np.isfinite(kf.residuals()))


@exercise("KalmanSmoother")
def _ks():
    kf = mod.KalmanFilter(F=[[1.0]], H=[[1.0]], Q=[[0.01]], R=[[1.0]]).fit(np.cumsum(np.random.default_rng(6).standard_normal(100)))
    path = mod.KalmanSmoother().apply(kf)
    assert path.smoothed_means.shape == (100, 1)


def _nonlinear_filter(cls):
    rng = np.random.default_rng(7)
    x = np.cumsum(0.05 * rng.standard_normal(300)) + 1.5
    y = x + 0.2 * x ** 3 + 0.25 * rng.standard_normal(300)          # monotone nonlinearity
    est = cls(f=lambda s: s.copy(), h=lambda s: np.array([s[0] + 0.2 * s[0] ** 3]),
              Q=0.0025 * np.eye(1), R=0.0625 * np.eye(1), x0=np.array([1.5]), P0=np.eye(1)).fit(y)
    mse_f = np.mean((est.filtered_means_.ravel() - x) ** 2)
    sm = est.smooth()
    assert sm.smoothed_means.shape == (300, 1) and mse_f < np.var(x)   # tracks the latent state
    assert np.mean((sm.smoothed_means.ravel() - x) ** 2) < mse_f      # smoother beats filter


@exercise("ExtendedKalmanFilter")
def _ekf():
    _nonlinear_filter(mod.ExtendedKalmanFilter)


@exercise("UnscentedKalmanFilter")
def _ukf():
    _nonlinear_filter(mod.UnscentedKalmanFilter)


@exercise("ParticleFilter")
def _pf():
    rng = np.random.default_rng(8)
    latent = np.cumsum(0.1 * rng.standard_normal(150))
    obs = latent + rng.standard_normal(150)
    pf = mod.ParticleFilter(n_particles=1000,
                            transition_sampler=lambda p, r: p + 0.1 * r.standard_normal(p.shape),
                            observation_logpdf=lambda p, y: -0.5 * ((p.ravel() - y) ** 2),
                            initial_sampler=lambda r: r.standard_normal(1000), random_state=9).fit(obs)
    # the filter must track better than the raw observations do
    assert np.mean((pf.filtered_means_[:, 0] - latent) ** 2) < np.mean((obs - latent) ** 2)
    assert np.isfinite(pf.loglik_) and pf.effective_sample_sizes_.min() > 0


@exercise("RaoBlackwellFilter")
def _rb():
    rb = mod.RaoBlackwellFilter(modes=[{"F": [[1.0]], "H": [[1.0]], "Q": [[0.02]], "R": [[0.5]]},
                                       {"F": [[1.02]], "H": [[1.0]], "Q": [[0.08]], "R": [[2.0]]}])
    rb.fit(np.cumsum(np.random.default_rng(10).standard_normal(200)))
    w = np.asarray(rb.mode_probabilities_ if hasattr(rb, "mode_probabilities_") else rb.filtered_means_)
    assert np.all(np.isfinite(w))


@exercise("HiddenMarkovModel")
def _hmm():
    rng = np.random.default_rng(11)
    states = np.repeat([0, 1, 0, 1], 100)
    y = np.where(states == 0, rng.normal(0, 1, 400), rng.normal(4, 1, 400))
    hmm = mod.HiddenMarkovModel(n_states=2, random_state=0).fit(y)
    dec = np.asarray(hmm.decode())
    assert dec.shape == (400,) and np.isfinite(hmm.score())
    assert np.asarray(hmm.predict_proba()).shape == (400, 2)


@exercise("SwitchingRegression")
def _switching():
    rng = np.random.default_rng(12)
    X = rng.standard_normal((400, 1))
    y = np.where(np.arange(400) < 200, 2.0 * X[:, 0], -2.0 * X[:, 0]) + 0.3 * rng.standard_normal(400)
    sw = mod.SwitchingRegression(n_regimes=2, random_state=0).fit(X, y)
    assert np.asarray(sw.predict(X[:10])).shape == (10,)


@exercise("RegimeSwitching")
def _regime():
    rs = mod.RegimeSwitching(p=1, n_states=2, random_state=0).fit(_ar1(400))
    assert len(rs.ar_coefficients_) == 2 and all(a.shape == (1,) for a in rs.ar_coefficients_)
    assert np.asarray(rs.predict(_ar1(400)[-5:][:, None])).shape == (5,)


@exercise("MixtureAutoregressive")
def _mixture_ar():
    mix = mod.MixtureAutoregressive(k=2, p=1, random_state=0).fit(_ar1(400))
    assert abs(mix.weights_.sum() - 1.0) < 1e-6 and all(a.shape == (1,) for a in mix.ar_coefficients_)
    assert np.asarray(mix.predict(_ar1(400)[-5:][:, None])).shape == (5,)


@functools.lru_cache(maxsize=None)
def _changepoint_series():
    rng = np.random.default_rng(13)
    return np.concatenate([rng.standard_normal(150), rng.standard_normal(150) + 3.0])


@exercise("ChangePointDetection")
def _cpd():
    res = mod.ChangePointDetection(_changepoint_series(), method="pelt")
    assert any(abs(p - 150) <= 10 for p in res.points)


@exercise("PELT")
def _pelt():
    assert any(abs(p - 150) <= 10 for p in mod.PELT(_changepoint_series()).points)


@exercise("BinarySegmentation")
def _binseg():
    assert any(abs(p - 150) <= 10 for p in mod.BinarySegmentation(_changepoint_series()).points)


@exercise("BottomUp")
def _bottom_up():
    assert any(abs(p - 150) <= 15 for p in mod.BottomUp(_changepoint_series()).points)


@exercise("BayesianChangePoint")
def _bocpd():
    b = mod.BayesianChangePoint(hazard_rate=0.02, threshold=0.3).fit(_changepoint_series())
    assert b.result_.probability_of_change.shape == (300,)


@exercise("ChangePointResult")
def _cp_result():
    assert isinstance(mod.PELT(_changepoint_series()), mod.ChangePointResult)


@exercise("BOCPDResult")
def _bocpd_result():
    b = mod.BayesianChangePoint(hazard_rate=0.02).fit(_changepoint_series())
    assert isinstance(b.result_, mod.BOCPDResult)


@functools.lru_cache(maxsize=None)
def _signal():
    t = np.arange(1024) / 100.0
    return np.sin(2 * np.pi * 10.0 * t) + 0.2 * np.random.default_rng(14).standard_normal(1024)


@exercise("SpectralAnalysis")
def _spectral():
    sa = mod.SpectralAnalysis(_signal(), fs=100.0)
    assert abs(sa.dominant_frequency() - 10.0) < 1.0
    f, p = sa.periodogram()
    assert f.shape == p.shape and "time_domain" in sa.total_power()


@exercise("Periodogram")
def _periodogram():
    f, p = mod.Periodogram(_signal(), fs=100.0)
    assert abs(f[np.argmax(p)] - 10.0) < 1.0


@exercise("PowerSpectrum")
def _power_spectrum():
    f, p = mod.PowerSpectrum(_signal(), fs=100.0, nperseg=128)
    assert abs(f[np.argmax(p)] - 10.0) < 2.0


@exercise("WaveletTransform")
def _wavelet():
    scales, coeffs = mod.WaveletTransform(_signal()[:256], scales=np.arange(1, 16), fs=100.0, kind="cwt")
    assert coeffs.shape == (15, 256)


@exercise("CWTTransform")
def _cwt():
    scales, coeffs = mod.CWTTransform(_signal()[:256], scales=np.arange(1, 8), fs=100.0)
    assert coeffs.shape == (7, 256)


@exercise("DWTTransform")
def _dwt():
    d = mod.DWTTransform(_signal()[:256], wavelet="haar", level=3)
    assert isinstance(d, dict) and len(d) > 0


@exercise("STFT")
def _stft():
    f, t, spec = mod.STFT(_signal(), fs=100.0, window_len=128, hop=64)
    assert spec.shape == (f.size, t.size)


@exercise("Hilbert")
def _hilbert():
    z = mod.Hilbert(np.cos(2 * np.pi * np.arange(256) / 32.0))
    assert np.iscomplexobj(z) and np.allclose(np.abs(z)[20:-20], 1.0, atol=0.05)


@functools.lru_cache(maxsize=None)
def _seasonal():
    t = np.arange(240)
    return 0.05 * t + 2.0 * np.sin(2 * np.pi * t / 12) + 0.3 * np.random.default_rng(15).standard_normal(240)


def _decomp_ok(res, n=240):
    for part in (res.trend, res.seasonal, res.resid):
        assert np.asarray(part).shape == (n,)
    return res


@exercise("SeasonalDecomposition")
def _seasonal_decomp():
    _decomp_ok(mod.SeasonalDecomposition(_seasonal(), period=12))


@exercise("STLDecomposition")
def _stl():
    _decomp_ok(mod.STLDecomposition(_seasonal(), period=12))


@exercise("X11Decomposition")
def _x11():
    _decomp_ok(mod.X11Decomposition(_seasonal() + 10.0, period=12))   # X11 needs positive data


@exercise("TrendFilter")
def _trend_filter():
    r = _decomp_ok(mod.TrendFilter(_seasonal(), lam=10.0))
    assert np.corrcoef(r.trend, np.arange(240))[0, 1] > 0.9


@exercise("HPFilter")
def _hp():
    _decomp_ok(mod.HPFilter(_seasonal(), lamb=1600.0))


@exercise("adf_test")
def _adf():
    assert mod.adf_test(_ar1()).pvalue < 0.05 and mod.adf_test(np.cumsum(_ar1())).pvalue > 0.05


@exercise("kpss_test")
def _kpss():
    assert mod.kpss_test(np.cumsum(_ar1())).pvalue <= 0.05


@exercise("pp_test")
def _pp():
    assert mod.pp_test(_ar1()).pvalue < 0.05


@exercise("ljung_box")
def _ljung_box():
    assert mod.ljung_box(_ar1(), lags=10).pvalues[-1] < 1e-6


@exercise("durbin_watson")
def _dw():
    assert abs(mod.durbin_watson(np.random.default_rng(16).standard_normal(500)) - 2.0) < 0.3


@exercise("arch_test")
def _arch_test():
    assert mod.arch_test(_garch_returns(), lags=5).pvalue < 0.01


@exercise("granger_causality")
def _granger():
    Y = _bivariate()
    fwd = mod.granger_causality(Y[:, 1], Y[:, 0], max_lag=2)   # does y1 cause y0? (A[0,1]=0.1)
    assert set(fwd) == {1, 2} and all(0 <= r.pvalue <= 1 for r in fwd.values())


@exercise("johansen_test")
def _johansen():
    rng = np.random.default_rng(17)
    c = np.cumsum(rng.standard_normal(300))
    Y = np.column_stack([c + 0.2 * rng.standard_normal(300), c + 0.2 * rng.standard_normal(300)])
    res = mod.johansen_test(Y, p=1)
    r0 = res["trace"][0]
    assert r0.statistic > r0.critical_values["95%"]          # one cointegrating relation


@exercise("forecast")
def _forecast():
    _forecast_ok(mod.forecast(mod.AR(1).fit(_ar1()), horizon=4), 4)


@exercise("predict")
def _predict():
    _forecast_ok(mod.predict(mod.AR(1).fit(_ar1()), horizon=4), 4)


@exercise("confidence_bands")
def _bands():
    lo, hi = mod.confidence_bands(mod.AR(1).fit(_ar1()).forecast(4), level=0.95)
    assert np.all(np.asarray(lo) < np.asarray(hi))


@exercise("backtesting")
def _backtesting():
    bt = mod.backtesting(_ar1(), lambda: mod.AR(1), horizon=2, min_train=400, step=20)
    assert len(bt.rmse) == 2 and np.all(np.isfinite(bt.rmse))


@exercise("cross_validation_ts")
def _cv():
    folds = mod.cross_validation_ts(_ar1(), lambda: mod.AR(1), n_splits=3, horizon=1, min_train=300)
    assert len(folds) == 3


@exercise("ForecastResult")
def _forecast_result():
    fc = mod.ForecastResult(mean=np.array([1.0, 2.0]), std=np.array([0.1, 0.2]))
    lo, hi = fc.confidence_interval(0.95)
    assert np.all(lo < fc.mean) and np.all(fc.mean < hi)


@exercise("TestResult")
def _test_result():
    r = mod.adf_test(_ar1())
    assert isinstance(r, mod.TestResult) and r.null


def test_every_public_name_is_exercised():
    assert set(EXERCISES) == set(mod.__all__), (
        set(mod.__all__) - set(EXERCISES), set(EXERCISES) - set(mod.__all__))


@pytest.mark.parametrize("name", sorted(mod.__all__))
def test_exercise(name):
    EXERCISES[name]()
