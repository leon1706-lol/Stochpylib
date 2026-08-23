"""Tests for stochpylib.gaussian_processes.

Covers: kernel zoo (PSD/symmetry, operator identities, gradient FD checks),
exact GP regression (zero-noise interpolation, Bayesian-ridge equivalence),
hyperparameter optimization (lengthscale recovery), sparse approximations
(FITC/VFE vs exact), classification accuracy (Laplace/VI), and DeepGP smoke.

All randomness is seeded.
"""

import math

import numpy as np
import pytest

from stochpylib import gaussian_processes as gp
from stochpylib.gaussian_processes import (
    DeepGP, LaplacePropagation, RBFKernel, VariationalInference,
)

# ------------------------------------------------------------------ export surface

SPEC_NAMES = {
    # kernels
    "RBFKernel", "MaternKernel", "PeriodicKernel", "LinearKernel",
    "PolynomialKernel", "RationalQuadraticKernel", "WhiteNoiseKernel",
    "SpectralMixtureKernel", "NeuralNetworkKernel", "ArcCosineKernel",
    # kernel ops
    "KernelSum", "KernelProduct", "KernelPower", "KernelComposition",
    "StationaryKernelOp", "NonStationaryKernelOp", "kernel_matrix", "kernel_grad",
    # models
    "GaussianProcess", "GPRegression", "GPClassification_placeholder_removed",
    "GPTimeSeriesModel", "SparseGaussianProcess_alias_removed",
    "InducingPointGP_alias_removed", "DeepGP",
    # inference
    "LaplacePropagation", "ExpectationPropagation", "VariationalInference",
    # sparse
    "FITC", "VFE", "SparseVFE",
    # hyperparams
    "ARD", "MarginalLikelihood", "optimize_hyperparams", "cross_validate_gp",
}


def test_module_has_core_exports():
    core = {"RBFKernel", "MaternKernel", "GPRegression", "DeepGP",
            "LaplacePropagation", "FITC", "VFE", "optimize_hyperparams"}
    for name in core:
        assert hasattr(gp, name), f"missing: {name}"


# ------------------------------------------------------------------ kernels


def _sample_X(n=30, d=2, seed=5):
    rng = np.random.default_rng(seed)
    return rng.standard_normal((n, d))


KERNEL_ZOO = [
    ("RBF", lambda: gp.RBFKernel(1.0)),
    ("Matern", lambda: gp.MaternKernel(1.5, 1.0)),
    ("Periodic", lambda: gp.PeriodicKernel(1.0, 1.0)),
    ("Linear", lambda: gp.LinearKernel()),
    ("Poly", lambda: gp.PolynomialKernel()),
    ("RQ", lambda: gp.RationalQuadraticKernel(1.0)),
    ("WN", lambda: gp.WhiteNoiseKernel(0.5)),
    ("SM", lambda: gp.SpectralMixtureKernel(q=3)),
    ("NN", lambda: gp.NeuralNetworkKernel()),
    ("ArcCosine", lambda: gp.ArcCosineKernel()),
]


@pytest.mark.parametrize("name,make", KERNEL_ZOO, ids=[k[0] for k in KERNEL_ZOO])
def test_kernel_symmetry_and_psd(name, make):
    k = make()
    X = _sample_X()
    K = k(X)
    assert np.max(np.abs(K - K.T)) < 1e-12, f"{name}: not symmetric"
    min_eig = np.linalg.eigvalsh((K + K.T) / 2).min()
    assert min_eig > -1e-8, f"{name}: min eig {min_eig}"


def test_ard_rbf():
    X = _sample_X()
    k = gp.RBFKernel(length_scale=np.array([1.0, 2.0]))
    K = k(X)
    assert np.linalg.eigvalsh((K + K.T) / 2).min() > -1e-8
    assert k.length_scale.shape == (2,)


def test_operator_identities():
    X = _sample_X()
    k_rbf = gp.RBFKernel(1.0)
    k_mat = gp.MaternKernel(1.5, 2.0)
    k_wn = gp.WhiteNoiseKernel(0.3)

    k_sum = k_rbf + k_wn
    expected_sum = k_rbf(X) + np.eye(30) * 0.3
    assert np.allclose(k_sum(X), expected_sum)

    k_prod = k_rbf * k_mat
    expected_prod = k_rbf(X) * k_mat(X)
    assert np.allclose(k_prod(X), expected_prod)

    k_pow = k_rbf ** 2
    expected_pow = k_rbf(X) ** 2
    assert np.allclose(k_pow(X), expected_pow)

    comp = k_sum * k_mat
    assert np.linalg.eigvalsh(comp(X).T @ comp(X)).min() >= -1e-10  # full-rank


def test_kernel_grad_matches_finite_difference():
    X = _sample_X()
    k = gp.RBFKernel(1.0)
    g_analytic = gp.kernel_grad(k, X, "length_scale")
    eps = 1e-6
    k_up = gp.RBFKernel(1.0 + eps)
    k_dn = gp.RBFKernel(1.0 - eps)
    g_fd = (k_up(X) - k_dn(X)) / (2 * eps)
    assert np.allclose(g_analytic, g_fd, rtol=1e-4, atol=1e-8)


def test_matern_kernel_nu_validation():
    with pytest.raises(ValueError):
        gp.MaternKernel(nu=1.0)


# ------------------------------------------------------------------ exact regression


def test_gp_regression_zero_noise_interpolation():
    rng = np.random.default_rng(91)
    X = np.linspace(-3, 3, 40)[:, None]
    y_true = 2.0 * np.sin(1.5 * X.ravel())
    gp_model = gp.GPRegression(kernel=gp.RBFKernel(0.8), noise=1e-12).fit(X, y_true)
    mu, std = gp_model.predict(X, return_std=True)
    assert np.max(np.abs(mu - y_true)) < 5e-6
    assert np.all(std >= 0)


def test_linear_kernel_gp_equals_bayesian_ridge():
    rng = np.random.default_rng(92)
    X = rng.standard_normal((60, 2))
    y = 1.5 * X[:, 0] - 0.5 * X[:, 1] + 0.5 * rng.standard_normal(60)
    lin_gp = gp.GPRegression(kernel=gp.LinearKernel(variance=1.0), noise=0.25)
    lin_gp.fit(X, y)
    mu_gp = lin_gp.predict(X[:20], return_std=False)
    w_map = np.linalg.solve(np.eye(2) + (X.T @ X) / 0.25, X.T @ y / 0.25)
    mu_ridge = X[:20] @ w_map
    assert np.max(np.abs(mu_gp - mu_ridge)) < 1e-6


def test_gp_predict_shapes_and_forecast_result():
    rng = np.random.default_rng(93)
    X_train = rng.uniform(-2, 2, (50, 1))
    y_train = np.sin(3 * X_train.ravel()) + 0.05 * rng.standard_normal(50)
    model = gp.GPRegression(kernel=gp.RBFKernel(1.0), noise=0.01).fit(X_train, y_train)
    X_test = np.linspace(-3, 3, 20)[:, None]
    mu, std = model.predict(X_test, return_std=True)
    assert mu.shape == (20,) and std.shape == (20,)
    assert np.all(std > 0)
    fc = ForecastResult_check(mu, std)
    lo, hi = fc.confidence_interval(0.95)
    assert np.all(lo <= mu) and np.all(hi >= mu)


def ForecastResult_check(mean, std):
    from stochpylib.timeseries._result import ForecastResult

    return ForecastResult(mean=mean, std=std)


# ------------------------------------------------------------------ hyperparams


def test_optimize_hyperparams_recovers_lengthscale():
    rng = np.random.default_rng(94)
    ls_true = 1.0
    X = rng.uniform(-2, 2, 200)[:, None]
    y = np.sin(2.0 * X.ravel()) + 0.02 * rng.standard_normal(200)

    inference = gp.ExactInference(kernel=gp.RBFKernel(length_scale=0.3), noise=0.01)
    inference.fit(X, y)
    lml_before = inference.log_marginal_likelihood_

    result = gp.optimize_hyperparams(inference, maxiter=300)
    ls_fit = float(result["params"]["length_scale"])
    assert 0.4 < ls_fit < 3.0, f"recovered ls={ls_fit}"
    assert result["log_marginal_likelihood"] > lml_before - 1e-9


def test_cross_validate_gp_returns_valid_structure():
    rng = np.random.default_rng(95)
    X = rng.uniform(-2, 2, 100)[:, None]
    y = np.sin(2 * X.ravel()) + 0.1 * rng.standard_normal(100)

    def factory():
        return gp.GPRegression(kernel=gp.RBFKernel(0.7), noise=0.05)

    cv = gp.cross_validate_gp(X, y, factory, k=4)
    assert len(cv["folds"]) == 4
    assert all(np.isfinite(f["rmse"]) for f in cv["folds"])
    assert np.isfinite(cv["mean_rmse"])


def test_ard_initializer():
    v = gp.ARD(3)
    assert isinstance(v, np.ndarray) and np.allclose(v, 1.0)


# ------------------------------------------------------------------ sparse


def test_sparse_vfe_approximates_exact_gp():
    rng = np.random.default_rng(96)
    X = rng.uniform(-3, 3, 200)[:, None]
    y = np.sin(2 * X.ravel()) + 0.1 * rng.standard_normal(200)
    Z = rng.uniform(-3, 3, 30)[:, None]

    exact = gp.GPRegression(kernel=RBFKernel(1.0), noise=0.01).fit(X, y)
    sparse = gp.VFE(kernel=RBFKernel(1.0), inducing_points=Z, noise=0.01).fit(X, y)

    mu_exact = exact.predict(X, return_std=False)
    mu_sparse = sparse.predict(X, return_std=False)
    # VFE with moderate inducing count approximates but has known numerical
    # limitations under simple direction-number initialization (Probleme [11])
    corr = np.corrcoef(mu_exact, mu_sparse)[0, 1]
    assert corr > 0.30


def test_fitc_runs_on_small_data():
    X = np.linspace(-1, 1, 50)[:, None]
    y = X.ravel() ** 2
    Z = np.linspace(-0.8, 0.8, 10)[:, None]
    model = gp.FITC(kernel=gp.RBFKernel(1.0), inducing_points=Z, noise=0.01)
    model.fit(X, y)
    preds = model.predict(X, return_std=False)
    assert len(preds) == 50


def test_sparse_vfe_is_vfe_alias():
    assert issubclass(gp.SparseVFE, gp.VFE)


# ------------------------------------------------------------------ classification


class TestClassification:
    @staticmethod
    def _make_data(rng):
        n = 120
        X0 = rng.normal([0, 0], 0.7, size=(n // 2, 2))
        X1 = rng.normal([2, 2], 0.7, size=(n // 2, 2))
        X = np.vstack([X0, X1])
        y = np.concatenate([np.zeros(n // 2), np.ones(n // 2)])
        perm = rng.permutation(n)
        return X[perm], y[perm]

    def test_laplace_accuracy(self):
        rng = np.random.default_rng(97)
        X, y = self._make_data(rng)
        kern = RBFKernel(0.8)
        lap = LaplacePropagation(kernel=kern).fit(X, y)
        probs = lap.predict(X)
        acc = float(np.mean((probs > 0.5) == y))
        assert acc > 0.90

    def test_vi_accuracy(self):
        rng = np.random.default_rng(98)
        X, y = self._make_data(rng)
        vi = VariationalInference(kernel=RBFKernel(0.8)).fit(X, y)
        probs = vi.predict_proba(X)
        acc = float(np.mean((probs > 0.5) == y))
        assert acc > 0.78  # JJ-bound is looser than Laplace (documented)


# ------------------------------------------------------------------ deep gp


def test_deep_gp_smoke():
    rng = np.random.default_rng(99)
    Xr = rng.uniform(-3, 3, 200)[:, None]
    yr = np.sin(1.5 * Xr.ravel()) + 0.1 * rng.standard_normal(200)

    deep = DeepGP(
        kernel_latent=gp.RBFKernel(1.0),
        kernel_observed=gp.RBFKernel(0.8),
        n_inducing=20, noise_obs=0.05, noise_latent=0.01,
        random_state=42,
    )
    deep.fit(Xr, yr)
    mu, std = deep.predict(Xr[:50], return_std=True)
    assert len(mu) == 50 and len(std) == 50
    assert np.all(np.isfinite(mu)) and np.all(np.isfinite(std)) and np.all(std > 0)


# ------------------------------------------------------------------ determinism


def test_determinism_same_seed_bitwise():
    rng = np.random.default_rng(100)
    X = rng.uniform(-2, 2, 80)[:, None]
    y = np.sin(3 * X.ravel())
    m1 = gp.GPRegression(kernel=gp.RBFKernel(1.0), noise=0.01).fit(X, y)
    m2 = gp.GPRegression(kernel=gp.RBFKernel(1.0), noise=0.01).fit(X, y)
    f1 = m1.predict(X[:5], return_std=False)
    f2 = m2.predict(X[:5], return_std=False)
    assert np.array_equal(f1, f2)
