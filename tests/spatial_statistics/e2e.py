"""End-to-end exercise sweep: one realistic use per public name in
``stochpylib.spatial_statistics.__all__``. Kept fast (small n, cheap constructors) --
exhaustive oracle checks live in ``tests.py``.
"""

import numpy as np
import pytest

from stochpylib import spatial_statistics as mod

EXERCISES = {}


def exercise(name):
    def deco(fn):
        EXERCISES[name] = fn
        return fn
    return deco


_RNG = np.random.default_rng(0)
_COORDS = _RNG.uniform(0, 10, size=(60, 2))
_VALUES = _RNG.normal(size=60) + 0.05 * _COORDS[:, 0]
_WINDOW = ((0.0, 10.0), (0.0, 10.0))


# ------------------------------------------------------------------------- variogram

@exercise("Variogram")
def _variogram_base():
    v1 = mod.Semivariogram(model="spherical", nugget=0.1, sill=1.0, range=2.0)
    v2 = mod.Semivariogram(model="nugget", nugget=0.2)
    nested = v1 + v2
    assert isinstance(nested, mod.Variogram)
    assert abs(nested(0.0)) < 1e-12


@exercise("Semivariogram")
def _semivariogram():
    v = mod.Semivariogram(model="exponential", nugget=0.0, sill=1.0, range=1.0)
    assert v(0.0) == 0.0 and abs(v.sill - 1.0) < 1e-9


@exercise("SpatialCovariance")
def _spatial_covariance():
    v = mod.Semivariogram(model="exponential", nugget=0.0, sill=1.0, range=3.0)
    cov = mod.SpatialCovariance(v)
    kernel = cov.to_kernel()
    back = mod.SpatialCovariance.from_kernel(kernel)
    assert abs(back(1.0) - cov(1.0)) < 1e-8


@exercise("ExperimentalVariogram")
def _experimental_variogram():
    ev = mod.ExperimentalVariogram(bins=10).fit(_COORDS, _VALUES)
    assert len(ev.lags_) > 0 and len(ev.lags_) == len(ev.gamma_)


@exercise("VariogramFitting")
def _variogram_fitting():
    ev = mod.ExperimentalVariogram(bins=10).fit(_COORDS, _VALUES)
    vf = mod.VariogramFitting(model=["spherical", "exponential"]).fit(ev)
    assert isinstance(vf.variogram_, mod.Semivariogram) and hasattr(vf, "table_")


@exercise("Nugget")
def _nugget():
    ev = mod.ExperimentalVariogram(bins=10).fit(_COORDS, _VALUES)
    n = mod.Nugget().fit(ev)
    assert float(n) >= 0.0


@exercise("Sill")
def _sill():
    ev = mod.ExperimentalVariogram(bins=10).fit(_COORDS, _VALUES)
    s = mod.Sill().fit(ev)
    assert float(s) > 0.0


@exercise("Range")
def _range():
    ev = mod.ExperimentalVariogram(bins=10).fit(_COORDS, _VALUES)
    r = mod.Range().fit(ev)
    assert float(r) > 0.0


# --------------------------------------------------------------------------- kriging

_V = None  # populated lazily on first use, shared across kriging exercises


def _shared_variogram():
    global _V
    if _V is None:
        ev = mod.ExperimentalVariogram(bins=10).fit(_COORDS, _VALUES)
        _V = mod.VariogramFitting(model="exponential").fit(ev).variogram_
    return _V


@exercise("Kriging")
def _kriging():
    k = mod.Kriging(variogram=_shared_variogram()).fit(_COORDS, _VALUES)
    mean = k.predict(_COORDS[:3])
    assert mean.shape == (3,)


@exercise("OrdinaryKriging")
def _ordinary_kriging():
    ok = mod.OrdinaryKriging(variogram=_shared_variogram()).fit(_COORDS, _VALUES)
    mean, std = ok.predict(_COORDS[:1], return_std=True)
    assert abs(mean[0] - _VALUES[0]) < 1e-6 and std[0] < 1e-6


@exercise("UniversalKriging")
def _universal_kriging():
    uk = mod.UniversalKriging(drift="linear", variogram=_shared_variogram()).fit(_COORDS, _VALUES)
    r = uk.predict_result(_COORDS[:5])
    lo, hi = r.confidence_interval(0.9)
    assert lo.shape == (5,)


@exercise("SimpleKriging")
def _simple_kriging():
    sk = mod.SimpleKriging(variogram=_shared_variogram(), mean=0.0).fit(_COORDS, _VALUES)
    w = sk.weights(_COORDS[0])
    assert w.shape == (60,)


@exercise("CoKriging")
def _cokriging():
    z2 = _VALUES + _RNG.normal(scale=0.5, size=len(_VALUES))
    ck = mod.CoKriging(variogram=_shared_variogram()).fit(_COORDS, _VALUES, _COORDS, z2)
    mean = ck.predict(_COORDS[:4])
    assert mean.shape == (4,)


@exercise("IndicatorKriging")
def _indicator_kriging():
    ik = mod.IndicatorKriging(thresholds=np.quantile(_VALUES, [0.33, 0.66])).fit(_COORDS, _VALUES)
    cdf = ik.predict_cdf(_COORDS[:5])
    assert np.all(np.diff(cdf, axis=1) >= -1e-9)


@exercise("DisjunctiveKriging")
def _disjunctive_kriging():
    dk = mod.DisjunctiveKriging(n_hermite=10, variogram=_shared_variogram()).fit(_COORDS, _VALUES)
    p = dk.predict_proba(_COORDS[:3], threshold=float(np.median(_VALUES)))
    assert np.all((p >= 0) & (p <= 1))


# ---------------------------------------------------------------------- random fields

@exercise("GaussianRandomField")
def _gaussian_random_field():
    grf = mod.GaussianRandomField(covariance=_shared_variogram())
    s = grf.sample(_COORDS[:20], random_state=1)
    assert s.shape == (20,)


@exercise("MaternField")
def _matern_field():
    mf = mod.MaternField(nu=1.5, length_scale=1.5)
    s = mf.sample(_COORDS[:20], n_samples=3, random_state=2)
    assert s.shape == (3, 20)


@exercise("OrnsteinUhlenbeckField")
def _ou_field():
    ou = mod.OrnsteinUhlenbeckField(length_scale=1.0)
    g = ou.sample_grid((10, 10), spacing=0.5, random_state=3)
    assert g.shape == (10, 10)


@exercise("BrownianSheet")
def _brownian_sheet():
    bs = mod.BrownianSheet(extent=(1.0, 1.0))
    g = bs.sample_grid((8, 8), random_state=4)
    assert g.shape == (8, 8) and g[0, 0] == 0.0


@exercise("FractionalBrownianSheet")
def _fbm_sheet():
    fbs = mod.FractionalBrownianSheet(hurst=(0.7, 0.3))
    g = fbs.sample_grid((8, 8), random_state=5)
    assert g.shape == (8, 8)


# ------------------------------------------------------------------- point processes

@exercise("SpatialPointProcess")
def _spatial_point_process():
    pp = mod.PoissonPointProcess(intensity=0.3, window=_WINDOW)
    assert isinstance(pp, mod.SpatialPointProcess)


@exercise("PoissonPointProcess")
def _poisson_pp():
    pp = mod.PoissonPointProcess(intensity=0.3, window=_WINDOW)
    pts = pp.sample(random_state=6)
    assert pts.ndim == 2 and pts.shape[1] == 2


@exercise("InhomogeneousPoisson")
def _inhomogeneous_poisson():
    ip = mod.InhomogeneousPoisson(lambda X: 0.1 + 0.02 * X[:, 0], window=_WINDOW)
    pts = ip.sample(random_state=7)
    assert pts.ndim == 2


@exercise("ThomasProcess")
def _thomas_process():
    tp = mod.ThomasProcess(kappa=0.05, mu=5, sigma=0.3, window=_WINDOW)
    pts = tp.sample(random_state=8)
    assert pts.ndim == 2
    assert tp.K(0.5) > 0


@exercise("MaternCluster")
def _matern_cluster():
    mc = mod.MaternCluster(kappa=0.05, mu=5, radius=0.4, window=_WINDOW)
    pts = mc.sample(random_state=9)
    assert pts.ndim == 2


@exercise("LogGaussianCox")
def _log_gaussian_cox():
    lgcp = mod.LogGaussianCox(mean=np.log(0.3), covariance=_shared_variogram(),
                               window=_WINDOW, grid_shape=(24, 24))
    pts = lgcp.sample(random_state=10)
    assert pts.ndim == 2


@exercise("RipleyK")
def _ripley_k():
    k = mod.RipleyK(_COORDS, _WINDOW, correction="translation")
    assert isinstance(k, mod.SpatialFunction)


@exercise("PairCorrelation")
def _pair_correlation():
    g = mod.PairCorrelation(_COORDS, _WINDOW)
    assert isinstance(g, mod.SpatialFunction)


# -------------------------------------------------------------------------------- tests

@exercise("MoransI")
def _morans_i():
    W = mod.SpatialWeights.knn(_COORDS, k=6)
    r = mod.MoransI(_VALUES, W)
    assert r.pvalue is not None


@exercise("GearyC")
def _geary_c():
    W = mod.SpatialWeights.knn(_COORDS, k=6)
    r = mod.GearyC(_VALUES, W)
    assert r.pvalue is not None


@exercise("SpatialAutocorrelation")
def _spatial_autocorrelation():
    W = mod.SpatialWeights.knn(_COORDS, k=6)
    r = mod.SpatialAutocorrelation(_VALUES, W, statistic="moran", local=True)
    assert r.statistic.shape == (60,)


@exercise("NNDistanceTest")
def _nn_distance_test():
    r = mod.NNDistanceTest(_COORDS, _WINDOW)
    assert r.pvalue is not None


# ------------------------------------------------------------------------------- extras

@exercise("SpatialWeights")
def _spatial_weights():
    W = mod.SpatialWeights.lattice((5, 5), rule="queen")
    assert W.n == 25 and W.S0 > 0


@exercise("SpatialFunction")
def _spatial_function():
    k = mod.RipleyK(_COORDS, _WINDOW)
    L = k.L()
    assert isinstance(L, mod.SpatialFunction)


@exercise("SARModel")
def _sar_model():
    W = mod.SpatialWeights.lattice((5, 5)).row_standardize()
    n = W.n
    y = _RNG.normal(size=n)
    sar = mod.SARModel(W).fit(y)
    assert hasattr(sar, "rho_")


@exercise("CARModel")
def _car_model():
    W = mod.SpatialWeights.lattice((5, 5))
    n = W.n
    y = _RNG.normal(size=n)
    car = mod.CARModel(W).fit(y)
    assert hasattr(car, "rho_")


def test_every_public_name_is_exercised():
    assert set(EXERCISES) == set(mod.__all__), (
        set(mod.__all__) - set(EXERCISES), set(EXERCISES) - set(mod.__all__))


@pytest.mark.parametrize("name", sorted(mod.__all__))
def test_exercise(name):
    EXERCISES[name]()
