"""Tests for stochpylib.gaussian_processes.

Covers: kernel zoo (PSD/symmetry, operator identities, gradient FD checks),
exact GP regression (zero-noise interpolation, Bayesian-ridge equivalence),
hyperparameter optimization (lengthscale recovery), sparse approximations
(FITC/VFE vs exact incl. the M=T identity), classification via the engines and
the spec-facing ``GPClassification`` facade, and DeepGP smoke.

All randomness is seeded.
"""

import math

import numpy as np
import pytest

import stochpylib
from stochpylib import gaussian_processes as gp
from stochpylib.gaussian_processes import (
    DeepGP, GPClassification, LaplacePropagation, RBFKernel, VariationalInference,
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
    "GaussianProcess", "GPRegression", "GPClassification",
    "GPTimeSeriesModel", "SparseGaussianProcess",
    "InducingPointGP", "DeepGP",
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


def test_all_spec_names_exported():
    for name in SPEC_NAMES:
        assert hasattr(gp, name), f"spec name missing from package: {name}"


def test_gaussian_processes_wired_into_top_level_package():
    assert "gaussian_processes" in stochpylib.__all__
    assert hasattr(stochpylib, "gaussian_processes")
    assert "GPClassification" in stochpylib.gaussian_processes.__all__


def test_inference_module_is_classification_only():
    # the broken duplicate FITC/VFE copy was removed from inference.py (Probleme [21])
    import stochpylib.gaussian_processes.inference as inf

    for name in ("FITC", "VFE", "SparseVFE"):
        assert not hasattr(inf, name), f"{name} must live in sparse.py only"
    for name in ("LaplacePropagation", "ExpectationPropagation", "VariationalInference"):
        assert hasattr(inf, name)


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


def test_diag_matches_full_matrix_all_kernels():
    # Regression lock for the broken single-arg BaseKernel.diag (Probleme [24]):
    # every kernel must expose a working diag() equal to diag(k(X, X)).
    X = np.linspace(-1.5, 1.5, 21)[:, None]
    makes = dict(KERNEL_ZOO)
    makes["RBF_ARD"] = lambda: gp.RBFKernel(length_scale=np.array([1.0]))
    for name, make in makes.items():
        k = make()
        d = k.diag(X)
        K = k(X)
        assert d.shape == (len(X),), name
        assert np.allclose(d, np.diag(K)), f"{name}: diag != diag(K)"


def test_composite_diag_and_predict_with_product_kernel():
    # KernelProduct/KernelPower used to inherit the broken diag() and crashed any
    # exact-GP prediction built on a composed kernel (Probleme [24]).
    rng = np.random.default_rng(965)
    X = np.linspace(0, 4, 100)[:, None]
    y = np.sin(2 * np.pi * X[:, 0]) + 0.05 * rng.standard_normal(100)
    kern = gp.RBFKernel(0.6) * gp.PeriodicKernel(1.0, length_scale=3.0)
    model = gp.GPRegression(kernel=kern, noise=0.05).fit(X, y)
    mu, sd = model.predict(X, return_std=True)
    assert mu.shape == (100,) and sd.shape == (100,)
    assert np.all(sd > 0) and np.all(np.isfinite(mu))
    assert np.max(np.abs(mu - y)) < 0.2

    kpow = gp.RBFKernel(0.8) ** 2
    assert np.allclose(kpow.diag(X), np.diag(kpow(X)))
    k_prod = gp.RBFKernel(0.8) * gp.WhiteNoiseKernel(0.2)
    assert np.allclose(k_prod.diag(X), np.diag(k_prod(X)))
    assert np.allclose(k_prod.diag(X), gp.RBFKernel(0.8).diag(X) * 0.2)


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
    # With the whitened (numerically stable) posterior the pseudo-point
    # approximation tracks the exact GP closely at M=30 for T=200.
    corr = np.corrcoef(mu_exact, mu_sparse)[0, 1]
    assert corr > 0.999
    assert np.max(np.abs(mu_exact - mu_sparse)) < 0.05


def test_sparse_equals_exact_gp_when_inducing_points_are_training_points():
    # M = T: the low-rank prior spans the full covariance, so VFE/FITC must
    # reproduce the exact GP posterior to machine precision.
    rng = np.random.default_rng(961)
    X = np.linspace(-2, 2, 40)[:, None]
    y = np.sin(2 * X.ravel()) + 0.05 * rng.standard_normal(40)
    kern = RBFKernel(1.0)
    exact = gp.GPRegression(kernel=kern, noise=0.05).fit(X, y)
    mu_e, sd_e = exact.predict(X, return_std=True)

    for cls in (gp.VFE, gp.FITC):
        sparse = cls(kernel=kern, inducing_points=X.copy(), noise=0.05).fit(X, y)
        mu_s, sd_s = sparse.predict(X, return_std=True)
        assert np.max(np.abs(mu_s - mu_e)) < 1e-8, cls.__name__
        assert np.max(np.abs(sd_s - sd_e)) < 1e-7, cls.__name__


def test_sparse_stable_for_many_inducing_points():
    # Regression lock for the raw-inverse instability (Probleme [23]): with the
    # whitened engine, growing M must converge to the exact posterior instead of
    # blowing up once Kuu becomes near-singular.
    rng = np.random.default_rng(962)
    X = np.linspace(-3, 3, 120)[:, None]
    y = np.sin(2 * X.ravel()) + 0.05 * rng.standard_normal(120)
    kern = RBFKernel(1.5)
    mu_e = gp.GPRegression(kernel=kern, noise=0.05).fit(X, y).predict(
        X, return_std=False)
    devs = []
    for m in (20, 60, 120):
        Z = np.linspace(-3, 3, m)[:, None]
        sgp = gp.SparseGaussianProcess(kernel=kern, inducing_points=Z,
                                       noise=0.05).fit(X, y)
        devs.append(float(np.max(np.abs(sgp.predict(X, return_std=False) - mu_e))))
    assert devs[0] < 0.1 and devs[-1] < 1e-6


def test_sparse_lml_method_and_finite_values():
    X = np.linspace(-1, 1, 50)[:, None]
    y = X.ravel() ** 2 + 0.01 * np.sin(9 * X.ravel())
    Z = np.linspace(-0.9, 0.9, 12)[:, None]
    model = gp.VFE(kernel=RBFKernel(0.8), inducing_points=Z, noise=0.02).fit(X, y)
    assert callable(model.log_marginal_likelihood)
    assert np.isfinite(model.log_marginal_likelihood())
    assert np.isfinite(model.log_marginal_likelihood_)


def test_fitc_runs_on_small_data():
    X = np.linspace(-1, 1, 50)[:, None]
    y = X.ravel() ** 2
    Z = np.linspace(-0.8, 0.8, 10)[:, None]
    model = gp.FITC(kernel=gp.RBFKernel(1.0), inducing_points=Z, noise=0.01)
    model.fit(X, y)
    preds = model.predict(X, return_std=False)
    assert len(preds) == 50


def test_spec_aliases_subclass_engines():
    assert issubclass(gp.SparseVFE, gp.VFE)
    assert issubclass(gp.SparseGaussianProcess, gp.VFE)
    assert issubclass(gp.InducingPointGP, gp.FITC)


def test_inducing_point_gp_tracks_data():
    rng = np.random.default_rng(963)
    X = np.linspace(0, 4, 80)[:, None]
    y = np.sin(X.ravel()) + 0.03 * rng.standard_normal(80)
    Z = np.linspace(0, 4, 20)[:, None]
    model = gp.InducingPointGP(kernel=RBFKernel(1.0), inducing_points=Z,
                               noise=0.03).fit(X, y)
    mu = model.predict(X, return_std=False)
    assert np.max(np.abs(mu - y)) < 0.15


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


class TestGPClassificationFacade:
    @staticmethod
    def _make_data(rng):
        n = 120
        X0 = rng.normal([0, 0], 0.7, size=(n // 2, 2))
        X1 = rng.normal([2, 2], 0.7, size=(n // 2, 2))
        X = np.vstack([X0, X1])
        y = np.concatenate([np.zeros(n // 2), np.ones(n // 2)])
        perm = rng.permutation(n)
        return X[perm], y[perm]

    def test_laplace_default_accuracy_and_labels(self):
        rng = np.random.default_rng(971)
        X, y = self._make_data(rng)
        clf = GPClassification(kernel=RBFKernel(0.8)).fit(X, y)
        probs = clf.predict_proba(X)
        assert np.all((probs >= 0.0) & (probs <= 1.0))
        acc = float(np.mean((probs > 0.5) == y))
        assert acc > 0.90
        labels = clf.predict(X)
        assert set(np.unique(labels)) <= {0, 1}
        assert np.array_equal(labels, (probs >= 0.5).astype(int))
        assert np.isfinite(clf.log_marginal_likelihood_)

    def test_vi_method_selection(self):
        rng = np.random.default_rng(972)
        X, y = self._make_data(rng)
        clf = GPClassification(kernel=RBFKernel(0.8), method="vi").fit(X, y)
        probs = clf.predict_proba(X)
        acc = float(np.mean((probs > 0.5) == y))
        assert acc > 0.75

    def test_ep_method_runs(self):
        rng = np.random.default_rng(973)
        X, y = self._make_data(rng)
        clf = GPClassification(kernel=RBFKernel(0.8), method="ep").fit(X, y)
        probs = clf.predict_proba(X)
        assert probs.shape == (len(y),)

    def test_unknown_method_raises(self):
        with pytest.raises(ValueError):
            GPClassification(kernel=RBFKernel(0.8), method="nope")

    def test_predict_proba_before_fit_raises(self):
        clf = GPClassification(kernel=RBFKernel(0.8))
        with pytest.raises(RuntimeError):
            clf.predict_proba(np.zeros((3, 2)))


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
