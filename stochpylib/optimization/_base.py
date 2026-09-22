"""Objective wrapper and optimizer base classes for :mod:`stochpylib.optimization`.

Every optimizer subclasses :class:`Optimizer`, which owns iteration bookkeeping (effort
counters, history, convergence bookkeeping, result construction) and delegates the actual
algorithm to a ``_minimize`` hook. Every objective is wrapped in :class:`Objective`, which
supplies a finite-difference gradient/Hessian fallback when no analytic one is given -- the
library has no autodiff, so second-order methods work either way.
"""

import numpy as np

from stochpylib.optimization._common import (
    _clip_to_bounds,
    _normalize_bounds,
    _numeric_grad,
    _numeric_hessian,
    _rng,
)
from stochpylib.optimization._result import OptimizeResult

__all__ = ["Objective", "Optimizer", "PopulationOptimizer", "ConstrainedOptimizer"]


class Objective:
    """Callable objective with an optional analytic gradient/Hessian and evaluation counters.

    ``fun(x) -> float``. ``grad(x) -> (dim,)`` uses ``grad`` when given, otherwise a central
    finite difference of ``fun`` with step ``fd_eps``. ``hess(x)`` uses an analytic ``hess``
    callable when given, otherwise a finite difference of ``grad`` (cheaper than differencing
    ``fun`` twice) if a gradient is available, else a finite difference of ``fun`` itself.
    Values that are NaN or ``-inf`` are mapped to ``+inf`` so optimizers can treat any invalid
    point as infinitely bad without special-casing -- the mirror image of
    :class:`stochpylib.advanced_mcmc.LogDensity`'s ``-inf`` convention.
    ``n_evals``/``n_grad_evals`` count calls.
    """

    def __init__(self, fun, grad=None, hess=None, fd_eps=1e-6, args=()):
        if isinstance(fun, Objective):
            other = fun
            fun = other.fun
            grad = other._grad_fn if grad is None else grad
            hess = other._hess_fn if hess is None else hess
            fd_eps = other.fd_eps
            args = other.args
        self.fun = fun
        self._grad_fn = grad
        self._hess_fn = hess
        self.fd_eps = float(fd_eps)
        self.args = tuple(args)
        self.n_evals = 0
        self.n_grad_evals = 0

    def __call__(self, x):
        self.n_evals += 1
        x = np.atleast_1d(np.asarray(x, dtype=float))
        # a diverging iterate legitimately overflows the objective; this wrapper's job is
        # to turn that into +inf, so numpy's warning about it is noise, not information
        try:
            with np.errstate(all="ignore"):
                v = float(self.fun(x, *self.args))
        except (ValueError, FloatingPointError, ArithmeticError):
            return np.inf
        if v != v or v == -np.inf:
            return np.inf
        return v

    @property
    def has_gradient(self):
        return self._grad_fn is not None

    @property
    def has_hessian(self):
        return self._hess_fn is not None

    def grad(self, x):
        x = np.atleast_1d(np.asarray(x, dtype=float))
        self.n_grad_evals += 1
        with np.errstate(all="ignore"):
            if self._grad_fn is not None:
                return np.atleast_1d(np.asarray(self._grad_fn(x, *self.args), dtype=float))
            return _numeric_grad(self.__call__, x, self.fd_eps)

    def value_and_grad(self, x):
        return self(x), self.grad(x)

    def hess(self, x):
        x = np.atleast_1d(np.asarray(x, dtype=float))
        if self._hess_fn is not None:
            return np.asarray(self._hess_fn(x, *self.args), dtype=float)
        if self._grad_fn is not None:
            return _hessian_of_grad(self.grad, x, self.fd_eps * 10)
        return _numeric_hessian(self.__call__, x, max(self.fd_eps * 100, 1e-4))

    def negated(self):
        """The objective with its sign flipped, for turning a maximization into a minimization."""
        g = None if self._grad_fn is None else (lambda x, *a: -np.asarray(self._grad_fn(x, *a)))
        h = None if self._hess_fn is None else (lambda x, *a: -np.asarray(self._hess_fn(x, *a)))
        return Objective(lambda x, *a: -float(self.fun(x, *a)), g, h, self.fd_eps, self.args)

    def __repr__(self):
        kind = "analytic" if self.has_gradient else "finite-difference"
        return f"Objective({kind} gradient, n_evals={self.n_evals})"


def _hessian_of_grad(grad, x, eps):
    """Symmetrized central difference of an analytic gradient."""
    x = np.asarray(x, dtype=float)
    n = len(x)
    H = np.empty((n, n))
    steps = eps * np.maximum(1.0, np.abs(x))
    for i in range(n):
        ei = np.zeros(n)
        ei[i] = steps[i]
        H[:, i] = (grad(x + ei) - grad(x - ei)) / (2 * steps[i])
    return 0.5 * (H + H.T)


class Optimizer:
    """Base for iterate-based minimizers.

    ``minimize(fun, x0, ...)`` wraps the objective, resolves the generator, calls the
    subclass ``_minimize(obj, x0, rng)`` hook and stores the returned
    :class:`~stochpylib.optimization._result.OptimizeResult` on ``self.result_`` (with
    ``self.x_``/``self.fun_`` as shortcuts), then returns ``self`` -- the same
    ``fit``-returns-self shape the rest of the library uses. A run that exhausts
    ``max_iter`` sets ``converged=False`` rather than raising, so the best point found is
    always inspectable.
    """

    method = "optimizer"

    def minimize(self, fun, x0=None, grad=None, hess=None, args=(), random_state=None):
        obj = Objective(fun, grad, hess, args=args)
        x0 = self._initial_point(x0)
        seed = getattr(self, "random_state", None) if random_state is None else random_state
        rng = _rng(seed)
        result = self._minimize(obj, x0, rng)
        result.nfev = result.nfev or obj.n_evals
        result.njev = result.njev or obj.n_grad_evals
        if not result.method:
            result.method = self.method
        self.result_ = result
        self.x_ = result.x
        self.fun_ = result.fun
        return self

    def maximize(self, fun, x0=None, grad=None, hess=None, args=(), random_state=None):
        """Maximize by minimizing the negated objective; ``result_.fun`` is the true maximum."""
        obj = Objective(fun, grad, hess, args=args).negated()
        self.minimize(obj, x0, args=(), random_state=random_state)
        self.result_.fun = -self.result_.fun
        self.result_.history = [-h for h in self.result_.history]
        if self.result_.jac is not None:
            self.result_.jac = -self.result_.jac
        self.fun_ = self.result_.fun
        return self

    def _initial_point(self, x0):
        if x0 is None:
            raise ValueError(f"{type(self).__name__} requires an initial point x0")
        return np.atleast_1d(np.asarray(x0, dtype=float)).copy()

    def _minimize(self, obj, x0, rng):
        raise NotImplementedError

    def to_result(self):
        return self.result_

    def __repr__(self):
        if not hasattr(self, "result_"):
            return f"{type(self).__name__}(unfitted)"
        return (f"{type(self).__name__}(fun={self.fun_:.6g}, nit={self.result_.nit}, "
                f"converged={self.result_.converged})")


class PopulationOptimizer(Optimizer):
    """Base for population-based (derivative-free) minimizers.

    Adds ``bounds`` normalization and an initial-design helper, and exposes the final
    ``population_``/``population_fun_`` after a run. ``x0`` is optional: when it is omitted
    the population is drawn from ``bounds``, and when both are omitted a symmetric box
    around the origin is used.
    """

    method = "population"

    def _initial_point(self, x0):
        if x0 is not None:
            return np.atleast_1d(np.asarray(x0, dtype=float)).copy()
        # dimension is only recoverable from per-coordinate bounds, not from a scalar (lo, hi)
        bounds = getattr(self, "bounds", None)
        arr = None if bounds is None else np.asarray(bounds, dtype=float)
        if arr is None or arr.ndim != 2 or arr.shape[1] != 2:
            raise ValueError(f"{type(self).__name__} requires x0, or per-coordinate bounds")
        lo, hi = _normalize_bounds(bounds, arr.shape[0])
        return 0.5 * (lo + hi)

    def _resolve_bounds(self, x0):
        return _normalize_bounds(getattr(self, "bounds", None), len(x0))

    def _store_population(self, population, values):
        self.population_ = np.asarray(population, dtype=float)
        self.population_fun_ = np.asarray(values, dtype=float)


class ConstrainedOptimizer(Optimizer):
    """Base for constrained minimizers.

    ``constraints`` is a dict or a sequence of dicts in the canonical
    ``{"type": "eq"|"ineq", "fun": callable, "jac": callable or None}`` form; ``"ineq"``
    follows the convention ``fun(x) >= 0`` is feasible (the same sign convention scipy's
    SLSQP uses). Box ``bounds`` are kept separate from general constraints because every
    method here handles them by projection rather than as extra multipliers.
    """

    method = "constrained"

    def _normalized_constraints(self):
        raw = getattr(self, "constraints", None) or ()
        if isinstance(raw, dict):
            raw = (raw,)
        out = []
        for c in raw:
            kind = c.get("type", "ineq")
            if kind not in ("eq", "ineq"):
                raise ValueError(f"constraint type must be 'eq' or 'ineq', got {kind!r}")
            out.append({"type": kind, "fun": c["fun"], "jac": c.get("jac")})
        return out

    @staticmethod
    def _constraint_values(constraints, x):
        eq, ineq = [], []
        for c in constraints:
            v = np.atleast_1d(np.asarray(c["fun"](x), dtype=float))
            (eq if c["type"] == "eq" else ineq).append(v)
        eq = np.concatenate(eq) if eq else np.zeros(0)
        ineq = np.concatenate(ineq) if ineq else np.zeros(0)
        return eq, ineq

    @classmethod
    def _violation(cls, constraints, x):
        """Max absolute equality residual and max inequality shortfall, combined."""
        eq, ineq = cls._constraint_values(constraints, x)
        v = 0.0
        if eq.size:
            v = max(v, float(np.max(np.abs(eq))))
        if ineq.size:
            v = max(v, float(np.max(np.maximum(0.0, -ineq))))
        return v

    def _project(self, x):
        bounds = _normalize_bounds(getattr(self, "bounds", None), len(x))
        return _clip_to_bounds(x, bounds)

    def _finish(self, result, constraints, x):
        result.extras["violation"] = self._violation(constraints, x)
        result.extras["feasible"] = result.extras["violation"] <= getattr(self, "ctol", 1e-6)
        return result


def _result(x, obj, nit, converged, method, message="", **kw):
    """Build an OptimizeResult without re-evaluating the objective at the solution."""
    fun = kw.pop("fun", None)
    if fun is None:
        fun = obj(x)
    return OptimizeResult(np.atleast_1d(np.asarray(x, dtype=float)), fun, nit=nit,
                          converged=converged, method=method, message=message, **kw)
