"""End-to-end API sweep for stochpylib.gaussian_processes: one realistic exercise per
public name. ``test_every_public_name_is_exercised`` fails the moment a name ships without
one; ``test_exercise[<name>]`` runs each exercise as its own pytest case."""

import numpy as np
import pytest

from stochpylib import gaussian_processes as mod

EXERCISES = {}


def exercise(name):
    def deco(fn):
        EXERCISES[name] = fn
        return fn
    return deco


_RNG = np.random.default_rng(0)
X = np.linspace(-2.0, 2.0, 60)[:, None]
Y = np.sin(2.0 * X[:, 0]) + 0.05 * _RNG.standard_normal(60)
X2 = _RNG.uniform(-1, 1, (40, 2))
Y_CLS = (X[:, 0] > 0).astype(float)
XT = np.array([[0.3], [1.1], [-1.7]])


def _kernel_workflow(k):
    K = mod.kernel_matrix(k, X)
    assert K.shape == (60, 60) and np.allclose(K, K.T, atol=1e-10)
    assert np.all(np.linalg.eigvalsh(K + 1e-8 * np.eye(60)) > -1e-6)
    assert np.allclose(np.asarray(k.diag(X)), np.diag(K), atol=1e-8)
    params = k.get_params()
    assert isinstance(params, dict)
    k.set_params(params)
    combo = k + mod.WhiteNoiseKernel(0.1)
    assert mod.kernel_matrix(combo, XT).shape == (3, 3)
    return K


for _name in ("RBFKernel", "MaternKernel", "PeriodicKernel", "LinearKernel", "PolynomialKernel",
              "RationalQuadraticKernel", "WhiteNoiseKernel", "NeuralNetworkKernel",
              "ArcCosineKernel"):
    EXERCISES[_name] = (lambda n: (lambda: _kernel_workflow(getattr(mod, n)())))(_name)


@exercise("SpectralMixtureKernel")
def _sm():
    _kernel_workflow(mod.SpectralMixtureKernel(q=2, dimension=1))


@exercise("BaseKernel")
def _base_kernel():
    assert issubclass(mod.RBFKernel, mod.BaseKernel)


@exercise("StationaryKernel")
def _stationary():
    assert issubclass(mod.MaternKernel, mod.StationaryKernel)


@exercise("NonStationaryKernel")
def _nonstationary():
    assert issubclass(mod.LinearKernel, mod.NonStationaryKernel)


@exercise("StationaryKernelOp")
def _stationary_op():
    assert mod.StationaryKernelOp is mod.StationaryKernel


@exercise("NonStationaryKernelOp")
def _nonstationary_op():
    assert mod.NonStationaryKernelOp is mod.NonStationaryKernel


@exercise("KernelSum")
def _sum():
    k = mod.RBFKernel(1.0) + mod.LinearKernel()
    assert isinstance(k, mod.KernelSum)
    assert np.allclose(mod.kernel_matrix(k, XT), mod.kernel_matrix(mod.RBFKernel(1.0), XT)
                       + mod.kernel_matrix(mod.LinearKernel(), XT))


@exercise("KernelProduct")
def _product():
    k = mod.RBFKernel(1.0) * mod.PeriodicKernel(period=2.0)
    assert isinstance(k, mod.KernelProduct) and "part0__length_scale" in k.get_params()


@exercise("KernelPower")
def _power():
    k = mod.RBFKernel(1.0) ** 2
    assert isinstance(k, mod.KernelPower)
    assert np.allclose(mod.kernel_matrix(k, XT), mod.kernel_matrix(mod.RBFKernel(1.0), XT) ** 2)


@exercise("KernelComposition")
def _composition():
    k = mod.KernelComposition([mod.RBFKernel(1.0), mod.LinearKernel()], weights=[0.5, 2.0])
    expected = 0.5 * mod.kernel_matrix(mod.RBFKernel(1.0), XT) + 2.0 * mod.kernel_matrix(mod.LinearKernel(), XT)
    assert np.allclose(mod.kernel_matrix(k, XT), expected)
    k.set_params({"part0__part0__length_scale": 2.0})
    assert k.get_params()["part0__part0__length_scale"] == 2.0


@exercise("kernel_matrix")
def _kernel_matrix():
    assert mod.kernel_matrix(mod.RBFKernel(0.5), X, XT).shape == (60, 3)


@exercise("kernel_grad")
def _kernel_grad():
    g = mod.kernel_grad(mod.RBFKernel(1.0), XT, "length_scale")
    assert g.shape == (3, 3) and np.all(np.isfinite(g))


@exercise("cholesky_with_jitter")
def _chol():
    K = mod.kernel_matrix(mod.RBFKernel(1.0), X)
    L, jitter = mod.cholesky_with_jitter(K)
    assert np.allclose(L @ L.T, K, atol=1e-5) and jitter >= 0


def _regression_workflow(model):
    model.fit(X, Y)
    mu, sd = model.predict(XT, return_std=True)
    assert np.allclose(mu, np.sin(2.0 * XT[:, 0]), atol=0.3) and np.all(sd > 0)
    assert np.isfinite(model.log_marginal_likelihood())
    return model


@exercise("GaussianProcess")
def _gp():
    _regression_workflow(mod.GaussianProcess(kernel=mod.RBFKernel(0.5), noise=0.01))


@exercise("GPRegression")
def _gpr():
    m = _regression_workflow(mod.GPRegression(kernel=mod.RBFKernel(0.5), noise=0.01))
    assert m.predict(XT, full_cov=True)[1].shape == (3, 3)


@exercise("ExactInference")
def _exact():
    _regression_workflow(mod.ExactInference(kernel=mod.RBFKernel(0.5), noise=0.01))


@exercise("GPTimeSeriesModel")
def _gpts():
    y = np.sin(np.arange(120) / 8.0) + 0.05 * _RNG.standard_normal(120)
    fc = mod.GPTimeSeriesModel(length_scale=10.0, noise=0.05).fit(y).forecast(horizon=5)
    assert np.asarray(fc.mean).shape == (5,) and np.all(np.asarray(fc.std) > 0)


@exercise("GPClassification")
def _gpc():
    clf = mod.GPClassification(kernel=mod.RBFKernel(0.5)).fit(X, Y_CLS)
    p = clf.predict_proba(X)
    assert np.all((p >= 0) & (p <= 1)) and np.mean(clf.predict(X) == Y_CLS) > 0.9


@exercise("LaplacePropagation")
def _laplace():
    lap = mod.LaplacePropagation(kernel=mod.RBFKernel(0.5)).fit(X, Y_CLS)
    p = np.asarray(lap.predict(XT))
    assert p.shape == (3,) and np.all((p >= 0) & (p <= 1))


@exercise("ExpectationPropagation")
def _ep():
    ep = mod.ExpectationPropagation(kernel=mod.RBFKernel(0.5), max_iter=10).fit(X, Y_CLS)
    p = np.asarray(ep.predict_proba(XT))
    assert p.shape == (3,) and np.all(np.isfinite(p))


@exercise("VariationalInference")
def _vi():
    vi = mod.VariationalInference(kernel=mod.RBFKernel(0.5)).fit(X, Y_CLS)
    p = np.asarray(vi.predict_proba(XT))
    assert p.shape == (3,) and np.all((p >= 0) & (p <= 1))


def _sparse_workflow(cls):
    m = cls(kernel=mod.RBFKernel(0.5), inducing_points=X[::5], noise=0.01).fit(X, Y)
    mu, sd = m.predict(XT, return_std=True)
    assert np.allclose(mu, np.sin(2.0 * XT[:, 0]), atol=0.35) and np.all(sd > 0)
    assert np.isfinite(m.log_marginal_likelihood())


for _name in ("FITC", "VFE", "SparseVFE", "InducingPointGP", "SparseGaussianProcess"):
    EXERCISES[_name] = (lambda n: (lambda: _sparse_workflow(getattr(mod, n))))(_name)


@exercise("DeepGP")
def _deep():
    deep = mod.DeepGP(kernel_latent=mod.RBFKernel(1.0), kernel_observed=mod.RBFKernel(0.8),
                      n_inducing=12, noise_obs=0.05, noise_latent=0.01, random_state=0).fit(X, Y)
    mu, sd = deep.predict(XT, return_std=True)
    assert mu.shape == (3,) and np.all(sd > 0)


@exercise("ARD")
def _ard():
    assert np.allclose(mod.ARD(3), 1.0)


@exercise("MarginalLikelihood")
def _ml():
    inf = mod.ExactInference(kernel=mod.RBFKernel(0.5), noise=0.01).fit(X, Y)
    ml = mod.MarginalLikelihood(inf)
    assert np.isfinite(ml.value({"length_scale": 0.6, "variance": 1.0}))


@exercise("optimize_hyperparams")
def _opt():
    inf = mod.ExactInference(kernel=mod.RBFKernel(0.3), noise=0.01).fit(X, Y)
    before = inf.log_marginal_likelihood_
    res = mod.optimize_hyperparams(inf, maxiter=30)
    assert res["log_marginal_likelihood"] >= before - 1e-6


@exercise("cross_validate_gp")
def _cv():
    cv = mod.cross_validate_gp(X, Y, lambda: mod.GPRegression(kernel=mod.RBFKernel(0.5), noise=0.05), k=3)
    assert len(cv["folds"]) == 3 and np.isfinite(cv["mean_rmse"])


def test_every_public_name_is_exercised():
    assert set(EXERCISES) == set(mod.__all__), (
        set(mod.__all__) - set(EXERCISES), set(EXERCISES) - set(mod.__all__))


@pytest.mark.parametrize("name", sorted(mod.__all__))
def test_exercise(name):
    EXERCISES[name]()
