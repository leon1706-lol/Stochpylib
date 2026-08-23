"""The kernel zoo — ten covariance functions, all natively implemented.

Every kernel is callable (``k(X)`` / ``k(X, Y)``), supports ARD through vector length
scales where meaningful, and composes via ``+`` / ``*`` / ``**`` (see ``kernel_ops``).
"""

import numpy as np
from scipy import special

from stochpylib.gaussian_processes._utils import _as_2d, _sqdist
from stochpylib.gaussian_processes.kernels._base import (
    BaseKernel, NonStationaryKernel, StationaryKernel,
)

__all__ = [
    "RBFKernel", "MaternKernel", "PeriodicKernel", "LinearKernel",
    "PolynomialKernel", "RationalQuadraticKernel", "WhiteNoiseKernel",
    "SpectralMixtureKernel", "NeuralNetworkKernel", "ArcCosineKernel",
]


class RBFKernel(StationaryKernel):
    """Squared-exponential kernel: k(r) = variance * exp(-r^2 / 2).

    Infinitely mean-square differentiable. Supports ARD via a vector length scale.
    """

    def _shape_fn(self, r):
        return np.exp(-0.5 * r**2)

    def diag(self, X):
        return np.full(len(_as_2d(X)), self.variance)


class MaternKernel(StationaryKernel):
    """Matérn kernel with nu in {0.5, 1.5, 2.5} (closed-form Bessel limits)."""

    def __init__(self, nu=1.5, length_scale=1.0, variance=1.0):
        super().__init__(length_scale=length_scale, variance=variance)
        if nu not in (0.5, 1.5, 2.5):
            raise ValueError("nu must be one of 0.5, 1.5, 2.5 (closed forms)")
        self.nu = float(nu)

    def get_params(self):
        params = super().get_params()
        params["nu"] = self.nu
        return params

    def set_params(self, params):
        nu = params.pop("nu", None)
        if nu is not None and float(nu) not in (0.5, 1.5, 2.5):
            raise ValueError("nu must be one of 0.5, 1.5, 2.5")
        self.nu = float(nu) if nu is not None else self.nu
        return super().set_params(params)

    def _shape_fn(self, r):
        if self.nu == 0.5:
            return np.exp(-r)
        if self.nu == 1.5:
            s = np.sqrt(3.0) * r
            return (1.0 + s) * np.exp(-s)
        s = np.sqrt(5.0) * r
        return (1.0 + s + s**2 / 3.0) * np.exp(-s)


class PeriodicKernel(StationaryKernel):
    """Exponentiated-sine-squared kernel for periodic structure."""

    def __init__(self, period=1.0, length_scale=1.0, variance=1.0):
        super().__init__(length_scale=length_scale, variance=variance)
        self.period = float(period)

    def get_params(self):
        params = super().get_params()
        params["period"] = self.period
        return params

    def set_params(self, params):
        p = params.pop("period", None)
        if p is not None:
            self.period = float(p)
        return super().set_params(params)

    def _matrix(self, X, Y):
        X = _as_2d(X)
        Y = X if Y is None else _as_2d(Y)
        # per-dimension exponentiated sine — the Euclidean-distance variant is NOT
        # positive semidefinite in more than one dimension
        diff = (X[:, None, :] - Y[None, :, :]) / max(self.period, 1e-12)
        ls = np.atleast_1d(self.length_scale)
        if ls.size == 1:
            ls = np.full(diff.shape[2], float(ls[0]))
        expo = np.sum(np.sin(np.pi * diff) ** 2 / (ls[None, None, :] ** 2), axis=2)
        return self.variance * np.exp(-2.0 * expo)


class LinearKernel(NonStationaryKernel):
    """Linear kernel: k(x, y) = variance * (x . y + c)."""

    _param_names = ("variance", "constant")

    def __init__(self, variance=1.0, constant=0.0):
        self.variance = float(variance)
        self.constant = float(constant)

    def _matrix(self, X, Y):
        Y = X if Y is None else Y
        return self.variance * (X @ Y.T + self.constant)

    def diag(self, X):
        return self.variance * np.einsum("ij,ij->i", X, X) + \
            self.variance * self.constant


class PolynomialKernel(NonStationaryKernel):
    """Polynomial kernel: k(x, y) = variance * (x . y + c)^degree."""

    _param_names = ("variance", "constant", "degree")

    def __init__(self, degree=2.0, constant=1.0, variance=1.0):
        self.variance = float(variance)
        self.constant = float(constant)
        self.degree = float(degree)
        if self.degree <= 0:
            raise ValueError("degree must be > 0")

    def _matrix(self, X, Y):
        Y = X if Y is None else Y
        return self.variance * (X @ Y.T + self.constant) ** self.degree


class RationalQuadraticKernel(StationaryKernel):
    """Rational quadratic: scale-mixture of RBF kernels with gamma-distributed scales."""

    def __init__(self, alpha=1.0, length_scale=1.0, variance=1.0):
        super().__init__(length_scale=length_scale, variance=variance)
        self.alpha = float(alpha)
        if self.alpha <= 0:
            raise ValueError("alpha must be > 0")

    def get_params(self):
        params = super().get_params()
        params["alpha"] = self.alpha
        return params

    def set_params(self, params):
        a = params.pop("alpha", None)
        if a is not None:
            self.alpha = float(a)
        return super().set_params(params)

    def _shape_fn(self, r):
        return (1.0 + r**2 / (2.0 * self.alpha)) ** (-self.alpha)


class WhiteNoiseKernel(BaseKernel):
    """White-noise kernel: variance on the diagonal only."""

    _param_names = ("variance",)

    def __init__(self, variance=1.0):
        self.variance = float(variance)

    def _matrix(self, X, Y):
        X = _as_2d(X)
        if Y is None:
            return self.variance * np.eye(len(X))
        Y = _as_2d(Y)
        # identical rows share noise; distinct inputs are uncorrelated
        same = np.all(X[:, None, :] == Y[None, :, :], axis=-1)
        return self.variance * same.astype(float)

    def diag(self, X):
        return np.full(len(_as_2d(X)), self.variance)


class SpectralMixtureKernel(NonStationaryKernel):
    """Spectral mixture kernel (Wilson & Adams, 2013), isotropic per component.

    k(x, y) = sum_q w_q * prod_d exp(-2 pi^2 s_q^2 tau_d^2) * cos(2 pi m_q tau_d),
    with tau = x - y. ``q`` components each carry weight/mean/scale.
    """

    _param_names = ("weights", "means", "scales")

    def __init__(self, q=3, means=None, scales=None, weights=None, dimension=None):
        q = int(q)
        self.weights = np.asarray(
            weights if weights is not None else np.ones(q) / q, dtype=float
        )
        self.means = np.asarray(
            means if means is not None else rng_default_means(q), dtype=float
        )
        self.scales = np.asarray(
            scales if scales is not None else np.abs(rng_default_scales(q)), dtype=float
        )
        if dimension is not None:
            d = int(dimension)
            for name in ("means", "scales"):
                vec = getattr(self, name)
                setattr(self, name, np.full(d, vec.mean()))
        if not (len(self.weights) == len(self.means) == len(self.scales)):
            raise ValueError("weights, means and scales must have equal length")
        if np.any(self.weights < 0) or abs(self.weights.sum() - 1.0) > 1e-9:
            wsum = self.weights.sum()
            self.weights = self.weights / wsum if wsum > 0 else np.ones_like(self.weights) / len(self.weights)

    def _matrix(self, X, Y):
        X = _as_2d(X)
        Y = X if Y is None else _as_2d(Y)
        tau = X[:, None, :] - Y[None, :, :]                      # (n, m, d)
        total = np.zeros((len(X), len(Y)))
        for w_q, m_q, s_q in zip(self.weights, self.means, self.scales):
            per_dim = np.exp(-2.0 * np.pi**2 * s_q**2 * tau**2) * \
                np.cos(2.0 * np.pi * m_q * tau)
            total += w_q * per_dim.prod(axis=2)
        return total


def rng_default_means(q):
    return np.linspace(0.05, 0.45, q)


def rng_default_scales(q):
    return np.geomspace(0.05, 0.5, q)


class NeuralNetworkKernel(NonStationaryKernel):
    """Neural-network covariance (Rasmussen & Williams eq. 4.29-4.31, depth 1).

    Uses the bias-augmented inputs u = [x, sqrt(bias_variance)] and the exact
    arc-sine integral — positive semidefinite by construction.
    """

    _param_names = ("variance", "bias_variance")

    def __init__(self, variance=1.0, bias_variance=1.0):
        self.variance = float(variance)
        self.bias_variance = float(bias_variance)

    def _matrix(self, X, Y):
        X = _as_2d(X)
        Y = X if Y is None else _as_2d(Y)
        b = np.sqrt(self.bias_variance)
        Xa = np.hstack([X, np.full((len(X), 1), b)])
        Ya = np.hstack([Y, np.full((len(Y), 1), b)])
        num = 2.0 * (Xa @ Ya.T)
        denom = np.sqrt((1.0 + 2.0 * np.sum(Xa**2, axis=1))[:, None]
                        * (1.0 + 2.0 * np.sum(Ya**2, axis=1))[None, :])
        ratio = np.clip(num / denom, -1.0, 1.0)
        return self.variance * (2.0 / np.pi) * np.arcsin(ratio)


class ArcCosineKernel(NonStationaryKernel):
    """Arc-cosine covariance of the ReLU unit (Cho & Saul, depth 1, degree 0):

    k(x, y) = sigma^2 * |x| |y| * J(theta),  J = (sin theta + (pi - theta) cos theta)/pi.
    """

    _param_names = ("variance", "weight_variance")

    def __init__(self, variance=1.0, weight_variance=1.0):
        self.variance = float(variance)
        self.weight_variance = float(weight_variance)

    def _matrix(self, X, Y):
        X = _as_2d(X)
        Y = X if Y is None else _as_2d(Y)
        sx2 = self.weight_variance * np.sum(X**2, axis=1)
        sy2 = self.weight_variance * np.sum(Y**2, axis=1)
        cos_theta = np.clip((X @ Y.T) * self.weight_variance /
                            np.sqrt(sx2[:, None] * sy2[None, :] + 1e-300), -1.0, 1.0)
        theta = np.arccos(cos_theta)
        J = (np.sin(theta) + (np.pi - theta) * cos_theta) / np.pi
        return self.variance * np.sqrt(sx2[:, None]) * np.sqrt(sy2[None, :]) * J
