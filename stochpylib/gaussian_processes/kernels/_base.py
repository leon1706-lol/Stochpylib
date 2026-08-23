"""Kernel base classes and the composability operators.

Every kernel is a callable: ``k(X)`` or ``k(X, Y)`` returns the covariance matrix.
Hyperparameters live as plain attributes and are exposed to optimizers through
:meth:`BaseKernel.get_params` / :meth:`set_params` (values are optimized in log-space
by :func:`~stochpylib.gaussian_processes.hyperparams.optimize_hyperparams`).

Operator overloading implements the load-bearing composability convention from
ARCHITECTURE.md::

    k = RBFKernel(length_scale=1.0) + MaternKernel(nu=2.5)
    k2 = WhiteNoiseKernel(0.01) * RBFKernel(1.0) ** 2
"""

import numpy as np

from stochpylib.gaussian_processes._utils import _as_2d, _sqdist


class BaseKernel:
    """Abstract kernel with operator overloading and a parameter interface."""

    #: subclasses list their optimizable attribute names here
    _param_names = ()

    def __call__(self, X, Y=None):
        return self._matrix(_as_2d(X), None if Y is None else _as_2d(Y))

    def diag(self, X):
        """Diagonal of k(X, X) — overridden where cheaper than the full matrix."""
        return np.diag(self._matrix(_as_2d(X), None))

    def _matrix(self, X, Y):
        raise NotImplementedError

    # parameter interface ----------------------------------------------------
    def get_params(self):
        return {name: getattr(self, name) for name in self._param_names}

    def set_params(self, params):
        for name, value in params.items():
            if not hasattr(self, name):
                raise KeyError(f"{type(self).__name__} has no parameter {name!r}")
            setattr(self, name, np.asarray(value, dtype=float).item()
                    if np.ndim(value) == 0 else np.asarray(value, dtype=float))
        return self

    # operator overloading -----------------------------------------------------
    def __add__(self, other):
        from stochpylib.gaussian_processes.kernel_ops import KernelSum

        return KernelSum([self, other])

    __radd__ = __add__

    def __mul__(self, other):
        from stochpylib.gaussian_processes.kernel_ops import KernelProduct

        return KernelProduct([self, other])

    __rmul__ = __mul__

    def __pow__(self, exponent):
        from stochpylib.gaussian_processes.kernel_ops import KernelPower

        return KernelPower(self, float(exponent))


class StationaryKernel(BaseKernel):
    """Base for isotropic kernels: k(x, y) = variance * f(||x - y|| / length_scale)."""

    _param_names = ("length_scale", "variance")

    def __init__(self, length_scale=1.0, variance=1.0):
        self.length_scale = length_scale
        self.variance = float(variance)

    @property
    def is_ard(self):
        return np.size(self.length_scale) > 1

    def _matrix(self, X, Y):
        r2 = _sqdist(X, Y, self.length_scale)
        r = np.sqrt(r2)
        return self.variance * self._shape_fn(r)

    def _shape_fn(self, r):
        raise NotImplementedError


class NonStationaryKernel(BaseKernel):
    """Base for input-dependent kernels (linear-family, neural, arc-cosine)."""


class ConstantBasis:
    """Trivial helper treating scalars as 1-D inputs in kernel calls."""
