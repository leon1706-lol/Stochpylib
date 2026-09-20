"""End-to-end API sweep for stochpylib.robust_statistics: one realistic
exercise per public name. ``test_every_public_name_is_exercised`` fails the
moment a name ships without one; ``test_exercise[<name>]`` runs each exercise
as its own pytest case."""

import numpy as np
import pytest

from stochpylib import robust_statistics as mod

EXERCISES = {}


def exercise(name):
    def deco(fn):
        EXERCISES[name] = fn
        return fn
    return deco


def _finite(a):
    return bool(np.all(np.isfinite(np.asarray(a, dtype=float))))


_RNG = np.random.default_rng(0)
_X_CLEAN = np.concatenate([_RNG.normal(10, 1, 90), _RNG.uniform(-20, 40, 10)])
_T = _RNG.uniform(0, 10, 80)
_Y = 1.0 + 2.0 * _T + _RNG.normal(0, 0.5, 80)
_Y_OUT = _Y.copy()
_Y_OUT[:15] += 30


# --------------------------------------------------------------------- base

@exercise("RobustEstimator")
def _robust_estimator():
    class _Mean(mod.RobustEstimator):
        method = "mean"

        def _fit(self, x):
            return float(np.mean(x)), float(np.std(x, ddof=1) / np.sqrt(len(x))), {}

    m = _Mean().fit(_X_CLEAN)
    assert _finite(m.estimate_) and isinstance(m.to_result().estimate, float)


@exercise("RobustRegressor")
def _robust_regressor():
    hr = mod.HuberRegression().fit(_T, _Y)
    pred = hr.predict(np.array([0.0, 5.0]))
    assert pred.shape == (2,)


@exercise("RobustCovarianceEstimator")
def _robust_covariance_estimator():
    X = _RNG.standard_normal((100, 3))
    mcd = mod.MCD(random_state=0).fit(X)
    mask = mcd.outliers()
    assert mask.shape == (100,)


@exercise("Resampler")
def _resampler():
    bb = mod.BlockBootstrap(np.mean, random_state=0).fit(_X_CLEAN)
    idx = bb.indices(np.random.default_rng(1))
    assert idx.shape == (len(_X_CLEAN),)


# ----------------------------------------------------------------- location

@exercise("TrimmedMean")
def _trimmed_mean():
    tm = mod.TrimmedMean(0.1).fit(_X_CLEAN)
    assert abs(tm.estimate_ - 10.0) < 1.0


@exercise("WinsorizedMean")
def _winsorized_mean():
    wm = mod.WinsorizedMean(0.1).fit(_X_CLEAN)
    assert abs(wm.estimate_ - 10.0) < 1.0


@exercise("Median")
def _median():
    med = mod.Median().fit(_X_CLEAN)
    lo, hi = med.confidence_interval()
    assert lo <= med.estimate_ <= hi


@exercise("HodgesLehmann")
def _hodges_lehmann():
    hl = mod.HodgesLehmann().fit(_X_CLEAN)
    assert abs(hl.estimate_ - 10.0) < 1.0


@exercise("L_Estimator")
def _l_estimator():
    le = mod.L_Estimator("trimean").fit(_X_CLEAN)
    assert _finite(le.estimate_)


@exercise("M_Estimator")
def _m_estimator():
    me = mod.M_Estimator("huber").fit(_X_CLEAN)
    assert abs(me.estimate_ - 10.0) < 1.0


@exercise("R_Estimator")
def _r_estimator():
    re_ = mod.R_Estimator("wilcoxon").fit(_X_CLEAN)
    assert abs(re_.estimate_ - 10.0) < 1.0


# -------------------------------------------------------------------- scale

@exercise("MedianAbsoluteDeviation")
def _mad_est():
    m = mod.MedianAbsoluteDeviation().fit(_X_CLEAN)
    assert m.estimate_ > 0


@exercise("Qn_Estimator")
def _qn():
    q = mod.Qn_Estimator().fit(_X_CLEAN)
    assert q.estimate_ > 0


@exercise("Sn_Estimator")
def _sn():
    s = mod.Sn_Estimator().fit(_X_CLEAN)
    assert s.estimate_ > 0


@exercise("IQR_Scale")
def _iqr():
    i = mod.IQR_Scale().fit(_X_CLEAN)
    assert i.estimate_ > 0


@exercise("RobustStd")
def _robust_std():
    r = mod.RobustStd("tau").fit(_X_CLEAN)
    assert r.estimate_ > 0


# -------------------------------------------------------------- regression

@exercise("TheilSenRegression")
def _theilsen():
    ts = mod.TheilSenRegression().fit(_T, _Y_OUT)
    assert abs(ts.coef_[1] - 2.0) < 0.3


@exercise("SiegalRegression")
def _siegal():
    sg = mod.SiegalRegression().fit(_T, _Y_OUT)
    assert abs(sg.coef_[1] - 2.0) < 0.3


@exercise("SiegelRegression")
def _siegel_alias():
    assert mod.SiegelRegression is mod.SiegalRegression
    sg = mod.SiegelRegression().fit(_T, _Y_OUT)
    assert abs(sg.coef_[1] - 2.0) < 0.3


@exercise("RANSACRegression")
def _ransac():
    rs = mod.RANSACRegression(random_state=0).fit(_T, _Y_OUT)
    assert abs(rs.coef_[1] - 2.0) < 0.3


@exercise("LTS_Regression")
def _lts():
    lt = mod.LTS_Regression(random_state=0).fit(_T, _Y_OUT)
    assert abs(lt.coef_[1] - 2.0) < 0.3


@exercise("MMRegression")
def _mm():
    mm = mod.MMRegression(random_state=0).fit(_T, _Y_OUT)
    assert abs(mm.coef_[1] - 2.0) < 0.3


@exercise("HuberRegression")
def _huber_reg():
    hr = mod.HuberRegression().fit(_T, _Y)
    assert abs(hr.coef_[1] - 2.0) < 0.3
    res = hr.to_result()
    assert hasattr(res, "coef_")


# -------------------------------------------------------------- covariance

@exercise("RobustCovariance")
def _robust_covariance():
    X = _RNG.standard_normal((150, 3))
    rc = mod.RobustCovariance(method="mcd").fit(X)
    assert rc.covariance_.shape == (3, 3)


@exercise("MCD")
def _mcd():
    X = _RNG.standard_normal((150, 3))
    m = mod.MCD(random_state=0).fit(X)
    assert m.covariance_.shape == (3, 3)


@exercise("MVE")
def _mve():
    X = _RNG.standard_normal((60, 2))
    m = mod.MVE(random_state=0, n_trials=500).fit(X)
    assert m.covariance_.shape == (2, 2)


@exercise("OGK")
def _ogk():
    X = _RNG.standard_normal((150, 3))
    m = mod.OGK().fit(X)
    assert m.covariance_.shape == (3, 3)


@exercise("RobustCorrelation")
def _robust_correlation():
    U = _RNG.uniform(size=(200, 2))
    rc = mod.RobustCorrelation(method="spearman").fit(U)
    assert -1 <= rc.correlation_[0, 1] <= 1


@exercise("CovShrinkage")
def _cov_shrinkage():
    R = _RNG.standard_normal((80, 4))
    cs = mod.CovShrinkage().fit(R)
    assert 0 <= cs.shrinkage_ <= 1


# --------------------------------------------------------------- bootstrap

@exercise("RobustBootstrap")
def _robust_bootstrap():
    rb = mod.RobustBootstrap("median", n_boot=300, random_state=0).fit(_X_CLEAN)
    lo, hi = rb.confidence_interval()
    assert lo <= hi


@exercise("WildBootstrap")
def _wild_bootstrap():
    wb = mod.WildBootstrap(n_boot=300, random_state=0).fit(_T, _Y)
    assert _finite(wb.std_errors_)


@exercise("BlockBootstrap")
def _block_bootstrap():
    bb = mod.BlockBootstrap(np.mean, n_boot=300, random_state=0).fit(_X_CLEAN)
    assert bb.std_error_ > 0


@exercise("StationaryBootstrap")
def _stationary_bootstrap():
    sb = mod.StationaryBootstrap(np.mean, n_boot=300, random_state=0).fit(_X_CLEAN)
    assert sb.std_error_ > 0


def test_every_public_name_is_exercised():
    assert set(EXERCISES) == set(mod.__all__), (
        set(mod.__all__) - set(EXERCISES), set(EXERCISES) - set(mod.__all__))


@pytest.mark.parametrize("name", sorted(mod.__all__))
def test_exercise(name):
    EXERCISES[name]()
