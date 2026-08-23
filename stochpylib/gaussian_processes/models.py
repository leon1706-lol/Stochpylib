"""Gaussian-process models and the exact inference engine."""

import numpy as np

from stochpylib.gaussian_processes._utils import _as_2d, cholesky_with_jitter
from stochpylib.gaussian_processes.inference import (
    ExpectationPropagation,
    LaplacePropagation,
    VariationalInference,
)
from stochpylib.timeseries._result import ForecastResult

__all__ = [
    "ExactInference", "GaussianProcess", "GPRegression", "GPTimeSeriesModel",
    "GPClassification",
]


def _as_series(y):
    arr = np.asarray(y, dtype=float).ravel()
    if not np.all(np.isfinite(arr)):
        raise ValueError("series contains non-finite values")
    return arr


class ExactInference:
    """Exact Gaussian inference for regression: Cholesky solve + analytic LML.

    ``fit(X, y)`` factors ``kern(X) + noise * I``; ``predict`` returns the posterior
    mean and either pointwise standard deviations (``full_cov=False``) or the full
    predictive covariance matrix (``full_cov=True``).
    """

    def __init__(self, kernel, noise=1e-6):
        self.kernel = kernel
        self.noise = float(noise)
        self.X_train = None
        self.y_train = None
        self.L_ = None
        self.alpha_ = None
        self.log_marginal_likelihood_ = None
        self.jitter_used_ = 0.0

    def fit(self, X, y):
        X = _as_2d(X)
        y = np.asarray(y, dtype=float).ravel()
        if len(y) != len(X):
            raise ValueError("X and y must have equal length")
        K = self.kernel(X) + self.noise * np.eye(len(X))
        L, jitter = cholesky_with_jitter(K)
        alpha = np.linalg.solve(L.T, np.linalg.solve(L, y))
        self.X_train, self.y_train = X, y
        self.L_, self.alpha_, self.jitter_used_ = L, alpha, jitter
        n = len(y)
        self.log_marginal_likelihood_ = float(
            -0.5 * float(y @ alpha)
            - float(np.log(np.diag(L)).sum())
            - 0.5 * n * np.log(2 * np.pi)
        )
        return self

    def predict(self, X_test, return_std=True, full_cov=False):
        """Posterior mean plus predictive std (or full covariance when requested)."""
        X_test = _as_2d(X_test)
        K_star = self.kernel(X_test, self.X_train)
        mean = K_star @ self.alpha_
        v = np.linalg.solve(self.L_, K_star.T)                     # (n_train, n_test)
        k_ss_diag = self.kernel.diag(X_test) - np.sum(v**2, axis=0)
        if full_cov:
            cov = self.kernel(X_test, X_test) - v.T @ v
            cov = 0.5 * (cov + cov.T)
            second = cov
        else:
            second = np.sqrt(np.clip(k_ss_diag, 0.0, None))
        if not (return_std or full_cov):
            return mean
        return mean, second

    def log_marginal_likelihood(self):
        if self.log_marginal_likelihood_ is None:
            raise RuntimeError("fit() must be called first")
        return self.log_marginal_likelihood_


class GaussianProcess(ExactInference):
    """User-facing exact GP regressor (spec name; thin alias of ExactInference)."""


class GPRegression(GaussianProcess):
    """Exact GP regression per the spec quickstart::

        gp = GPRegression(kernel=k, noise=0.01).fit(X_train, y_train)
        mu, sigma = gp.predict(X_test, return_std=True)
    """


class GPTimeSeriesModel(GaussianProcess):
    """Convenience GP over evenly spaced time indices (RBF trend + white noise)."""

    def __init__(self, length_scale=10.0, noise=0.01, variance=1.0):
        from stochpylib.gaussian_processes.kernels import RBFKernel, WhiteNoiseKernel

        kernel = RBFKernel(length_scale=length_scale, variance=variance) \
            + WhiteNoiseKernel(noise)
        super().__init__(kernel=kernel, noise=noise)

    def fit(self, y_or_X, y=None):
        """Fit either on a bare series (evenly spaced indices) or on (X, y)."""
        if y is None:
            yy = _as_series(y_or_X)
            X = np.arange(len(yy))[:, None]
        else:
            X = _as_2d(y_or_X)
            yy = _as_series(y)
        return super().fit(X, yy)

    def forecast(self, horizon=10):
        """Continue the index axis beyond the training range."""
        if self.X_train is None:
            raise RuntimeError("fit() must be called first")
        start = int(self.X_train[-1, 0]) + 1
        X_future = np.arange(start, start + horizon)[:, None]
        mean, std = self.predict(X_future, return_std=True)
        return ForecastResult(mean, std)


class GPClassification:
    """Spec-facing binary Gaussian-process classifier (targets {0, 1}).

    Thin facade over the approximate classification engines — ``method`` selects

    - ``"laplace"`` (default): Laplace approximation, RW Algorithm 3.1; supports
      ``link="logit"`` (with the kappa predictive correction) and ``"probit"``.
    - ``"ep"``: damped expectation propagation (**experimental**, may not converge).
    - ``"vi"``: Jaakkola-Jordan variational bound (logit link only).

    Extra keyword arguments are forwarded to the engine constructor::

        gp = GPClassification(kernel=k).fit(X_train, y_train)
        probs = gp.predict_proba(X_test)          # P(y=1 | x)
        labels = gp.predict(X_test)               # 0/1 decisions at 0.5
    """

    _engines = {
        "laplace": LaplacePropagation,
        "ep": ExpectationPropagation,
        "vi": VariationalInference,
    }

    def __init__(self, kernel, method="laplace", **engine_kwargs):
        if method not in self._engines:
            raise ValueError(
                f"unknown method {method!r}; expected one of {sorted(self._engines)}")
        self.kernel = kernel
        self.method = str(method)
        self.engine_kwargs = dict(engine_kwargs)
        self.engine_ = None
        self.X_train = None
        self.y_train = None
        self.log_marginal_likelihood_ = None

    def fit(self, X, y):
        engine = self._engines[self.method](self.kernel, **self.engine_kwargs)
        self.engine_ = engine.fit(X, y)
        self.X_train = self.engine_.X_train
        self.y_train = self.engine_.y_train
        self.log_marginal_likelihood_ = getattr(
            engine, "log_marginal_likelihood_", None)
        return self

    def predict_proba(self, X_test):
        """Posterior P(y=1 | x) for each test point."""
        if self.engine_ is None:
            raise RuntimeError("fit() must be called first")
        if hasattr(self.engine_, "predict_proba"):
            probs = self.engine_.predict_proba(X_test)
        else:  # LaplacePropagation exposes .predict for class probabilities
            probs = self.engine_.predict(X_test)
        return np.asarray(probs, dtype=float)

    def predict(self, X_test):
        """0/1 class labels at the 0.5 probability threshold."""
        return (self.predict_proba(X_test) >= 0.5).astype(int)
