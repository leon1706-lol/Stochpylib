"""Kernel composition operators and generic matrix/gradient helpers.

``KernelSum``, ``KernelProduct``, ``KernelPower`` and ``KernelComposition`` make the
kernel language algebraic — ``k1 + k2 * k3 ** 2`` builds a valid covariance operator.
``kernel_matrix`` evaluates any kernel (atomic or composite) and ``kernel_grad``
differentiates a kernel matrix with respect to a named hyperparameter.
"""

import numpy as np

from stochpylib.gaussian_processes._utils import _as_2d
from stochpylib.gaussian_processes.kernels._base import BaseKernel

__all__ = [
    "KernelSum", "KernelProduct", "KernelPower", "KernelComposition",
    "StationaryKernelOp", "NonStationaryKernelOp",
    "kernel_matrix", "kernel_grad",
]


class _CompositeBase(BaseKernel):
    """Composite kernels expose flattened parameters of their parts as
    ``part<i>__<name>`` so optimizers can walk the whole tree uniformly."""

    def __init__(self, parts):
        parts = list(parts)
        if len(parts) < 2:
            raise ValueError("composites need at least two parts")
        for i, k in enumerate(parts):
            if not isinstance(k, BaseKernel):
                raise TypeError(f"part {i} is not a kernel")
        self.parts = parts

    def get_params(self):
        out = {}
        for i, part in enumerate(self.parts):
            for name, value in part.get_params().items():
                out[f"part{i}__{name}"] = value
        return out

    def set_params(self, params):
        per_part = [dict() for _ in self.parts]
        for key, value in params.items():
            if "__" not in key:
                raise KeyError(f"composite parameter keys must look like 'part0__{key}'")
            idx, name = key.split("__", 1)
            if not idx.startswith("part"):
                raise KeyError(key)
            per_part[int(idx[4:])][name] = value
        for part, sub in zip(self.parts, per_part):
            if sub:
                part.set_params(sub)
        return self


class KernelSum(_CompositeBase):
    """Elementwise sum of kernel matrices."""

    def _matrix(self, X, Y):
        out = None
        for part in self.parts:
            term = part(X, Y)
            out = term.copy() if out is None else out + term
        return out

    def diag(self, X):
        return sum(part.diag(X) for part in self.parts)


class KernelProduct(_CompositeBase):
    """Elementwise product of kernel matrices; scalar factors allowed."""

    def _matrix(self, X, Y):
        Y2d = None if Y is None else _as_2d(Y)
        out = np.ones((len(_as_2d(X)), len(Y2d) if Y2d is not None else len(X)))
        for part in self.parts:
            if isinstance(part, (int, float)):
                term = np.full(out.shape, float(part))
            else:
                term = part(X, Y)
            out = out * term
        return out

    def get_params(self):
        out = {}
        idx_kernel = 0
        for part in self.parts:
            if isinstance(part, (int, float)):
                continue
            for name, value in part.get_params().items():
                out[f"part{idx_kernel}__{name}"] = value
            idx_kernel += 1
        return out


# grouping aliases matching the spec naming (stationary / non-stationary families)
from stochpylib.gaussian_processes.kernels._base import (
    NonStationaryKernel, StationaryKernel,
)

StationaryKernelOp = StationaryKernel
NonStationaryKernelOp = NonStationaryKernel


class KernelPower(BaseKernel):
    """Elementwise power of another kernel's matrix."""

    def __init__(self, kernel, exponent):
        if not isinstance(kernel, BaseKernel):
            raise TypeError("KernelPower needs a kernel")
        self.kernel = kernel
        self.exponent = float(exponent)

    def _matrix(self, X, Y):
        base = self.kernel(X, Y)
        return base ** self.exponent

    def get_params(self):
        inner = {f"k__{k}": v for k, v in self.kernel.get_params().items()}
        inner["exponent"] = self.exponent
        return inner

    def set_params(self, params):
        exponent = params.pop("exponent", None)
        if exponent is not None:
            self.exponent = float(exponent)
        self.kernel.set_params({
            k[len("k__"):]: v for k, v in params.items() if k.startswith("k__")
        })
        return self


class KernelComposition(KernelSum):
    """Weighted sum: k = sum_i w_i * k_i (weights are plain attributes)."""

    def __init__(self, kernels, weights=None):
        kernels = list(kernels)
        if weights is None:
            weights = np.ones(len(kernels))
        weights = np.asarray(weights, dtype=float)
        if len(weights) != len(kernels):
            raise ValueError("need one weight per kernel")
        wrapped = []
        for w, k in zip(weights, kernels):
            wrapped.append(k if w == 1.0 else k * float(w))
        super().__init__(wrapped)



def kernel_matrix(kernel, X, Y=None):
    """Evaluate ``kernel`` on ``(X, Y)`` (or X vs X when Y is None)."""
    return kernel(_as_2d(X), None if Y is None else _as_2d(Y))


def kernel_grad(kernel, X, param_name, epsilon=1e-6, Y=None):
    """Central finite-difference gradient of k(X, Y) w.r.t. one hyperparameter.

    Vector parameters (ARD length scales) are addressed element-wise by index,
    e.g. ``"length_scale[0]"``.
    """
    saved = dict(kernel.get_params())
    base_name = param_name.split("[", 1)[0]
    if base_name not in saved:
        raise KeyError(f"{type(kernel).__name__} has no parameter {param_name!r}")
    value = np.atleast_1d(np.asarray(saved[base_name], dtype=float))
    is_indexed = "[" in param_name
    elem = int(param_name.split("[", 1)[1][:-1]) if is_indexed else None

    def eval_at(delta):
        if is_indexed:
            vec = np.asarray(saved[base_name], dtype=float).copy()
            vec[elem] += delta
            full = dict(saved)
            full[base_name] = vec
            kernel.set_params(full)
        else:
            kernel.set_params({base_name: float(value[0]) + delta})
        return kernel(_as_2d(X), None if Y is None else _as_2d(Y)).copy()

    K_plus = eval_at(+epsilon)
    K_minus = eval_at(-epsilon)
    kernel.set_params(saved)
    return (K_plus - K_minus) / (2.0 * epsilon)
