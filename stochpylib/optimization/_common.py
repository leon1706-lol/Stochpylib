"""Shared numerical machinery for stochpylib.optimization: line searches, bound
handling, convergence tests and the space-filling initial designs used by the
population-based and Bayesian optimizers.

Finite-difference derivatives and RNG plumbing are reused from
``stochpylib.advanced_mcmc._common``, and the Steihaug/Newton inner solve from
``stochpylib.numerical_methods._common``, rather than duplicated -- library code never
wraps ``scipy.optimize`` either (that module is the test suite's independent oracle only,
per AGENTS.md).
"""

import numpy as np

from stochpylib.advanced_mcmc._common import (  # noqa: F401  (re-exported for siblings)
    _numeric_grad, _numeric_hessian, _rng, _spawn_rngs,
)
from stochpylib.numerical_methods._common import (  # noqa: F401  (re-exported for siblings)
    _conjugate_gradient, _numeric_jacobian,
)

__all__ = []


# ------------------------------------------------------------------------ bounds

def _normalize_bounds(bounds, dim):
    """``None``, ``(lo, hi)`` or a per-coordinate sequence -> ``(lo, hi)`` arrays of length ``dim``."""
    if bounds is None:
        return None
    arr = np.asarray(bounds, dtype=float)
    if arr.shape == (2,):
        lo = np.full(dim, arr[0])
        hi = np.full(dim, arr[1])
    elif arr.ndim == 2 and arr.shape[1] == 2:
        if arr.shape[0] == 1:
            arr = np.repeat(arr, dim, axis=0)
        if arr.shape[0] != dim:
            raise ValueError(f"bounds has {arr.shape[0]} rows, expected {dim}")
        lo, hi = arr[:, 0].copy(), arr[:, 1].copy()
    else:
        raise ValueError("bounds must be (lo, hi) or a sequence of (lo, hi) pairs")
    if np.any(hi < lo):
        raise ValueError("every bound must satisfy lo <= hi")
    return lo, hi


def _clip_to_bounds(x, bounds):
    return x if bounds is None else np.clip(x, bounds[0], bounds[1])


def _bounds_or_box(bounds, x0, width=10.0):
    """Fall back to a symmetric box around ``x0`` when a population method got no bounds."""
    if bounds is not None:
        return bounds
    x0 = np.atleast_1d(np.asarray(x0, dtype=float))
    half = 0.5 * width * np.maximum(1.0, np.abs(x0))
    return x0 - half, x0 + half


# ------------------------------------------------------------------ initial designs

def _latin_hypercube(n, bounds, rng):
    """Latin-hypercube design over a box, via ``montecarlo.LatinHypercubeSampling``."""
    from stochpylib.montecarlo import LatinHypercubeSampling

    lo, hi = bounds
    u = LatinHypercubeSampling(dim=len(lo), n=int(n), random_state=rng).generate()
    return lo + np.asarray(u, dtype=float).reshape(int(n), len(lo)) * (hi - lo)


def _uniform_population(n, bounds, rng):
    lo, hi = bounds
    return lo + rng.random((int(n), len(lo))) * (hi - lo)


# --------------------------------------------------------------------- convergence

def _converged(x_new, x_old, f_new, f_old, g, xtol, ftol, gtol):
    """The single stopping predicate: flat gradient, small step, or small objective change."""
    if g is not None and gtol is not None and np.max(np.abs(g)) <= gtol:
        return True, "gradient below gtol"
    if xtol is not None:
        scale = np.maximum(1.0, np.abs(np.asarray(x_old, dtype=float)))
        if np.max(np.abs(np.asarray(x_new, dtype=float) - x_old) / scale) <= xtol:
            return True, "step below xtol"
    if ftol is not None and np.isfinite(f_old):
        if abs(f_new - f_old) <= ftol * max(1.0, abs(f_old)):
            return True, "objective change below ftol"
    return False, ""


# -------------------------------------------------------------------- line searches

def _backtracking_armijo(obj, x, d, f0, g0, alpha0=1.0, c1=1e-4, rho=0.5, max_ls=60,
                         bounds=None):
    """Armijo backtracking along ``d``; returns ``(alpha, x_new, f_new)``.

    ``alpha`` is 0 when no acceptable step was found -- callers treat that as a stall
    rather than an error, since a descent direction can legitimately be exhausted at a
    numerically flat point.
    """
    d = np.asarray(d, dtype=float)
    slope = float(np.dot(g0, d))
    if slope >= 0:  # not a descent direction; fall back to steepest descent
        d = -np.asarray(g0, dtype=float)
        slope = float(np.dot(g0, d))
        if slope >= 0:
            return 0.0, x, f0
    alpha = float(alpha0)
    for _ in range(int(max_ls)):
        x_new = _clip_to_bounds(x + alpha * d, bounds)
        f_new = obj(x_new)
        if np.isfinite(f_new) and f_new <= f0 + c1 * alpha * slope:
            return alpha, x_new, f_new
        alpha *= rho
    return 0.0, x, f0


def _strong_wolfe(obj, x, d, f0, g0, alpha0=1.0, c1=1e-4, c2=0.9, max_ls=40, amax=50.0):
    """Nocedal & Wright Alg. 3.5/3.6 (bracket + zoom); returns ``(alpha, x_new, f_new, g_new)``.

    The curvature condition is what keeps the BFGS/L-BFGS Hessian approximation positive
    definite, so those methods use this rather than plain Armijo.
    """
    d = np.asarray(d, dtype=float)
    phi0 = float(f0)
    dphi0 = float(np.dot(g0, d))
    if dphi0 >= 0:
        d = -np.asarray(g0, dtype=float)
        dphi0 = float(np.dot(g0, d))
        if dphi0 >= 0:
            return 0.0, x, f0, np.asarray(g0, dtype=float)

    a_prev, f_prev = 0.0, phi0
    a_i = min(float(alpha0), amax)
    for i in range(1, int(max_ls) + 1):
        x_i = x + a_i * d
        f_i = obj(x_i)
        if not np.isfinite(f_i) or f_i > phi0 + c1 * a_i * dphi0 or (i > 1 and f_i >= f_prev):
            return _wolfe_zoom(obj, x, d, phi0, dphi0, a_prev, a_i, c1, c2, max_ls)
        g_i = obj.grad(x_i)
        dphi_i = float(np.dot(g_i, d))
        if abs(dphi_i) <= -c2 * dphi0:
            return a_i, x_i, f_i, g_i
        if dphi_i >= 0:
            return _wolfe_zoom(obj, x, d, phi0, dphi0, a_i, a_prev, c1, c2, max_ls)
        a_prev, f_prev = a_i, f_i
        a_i = min(2.0 * a_i, amax)
    # degrade to Armijo rather than stalling outright on a hard line search
    alpha, x_new, f_new = _backtracking_armijo(obj, x, d, phi0, g0, alpha0=alpha0, c1=c1)
    return alpha, x_new, f_new, obj.grad(x_new)


def _wolfe_zoom(obj, x, d, phi0, dphi0, a_lo, a_hi, c1, c2, max_ls):
    f_lo = obj(x + a_lo * d)
    for _ in range(int(max_ls)):
        a_j = 0.5 * (a_lo + a_hi)
        x_j = x + a_j * d
        f_j = obj(x_j)
        if not np.isfinite(f_j) or f_j > phi0 + c1 * a_j * dphi0 or f_j >= f_lo:
            a_hi = a_j
            continue
        g_j = obj.grad(x_j)
        dphi_j = float(np.dot(g_j, d))
        if abs(dphi_j) <= -c2 * dphi0:
            return a_j, x_j, f_j, g_j
        if dphi_j * (a_hi - a_lo) >= 0:
            a_hi = a_lo
        a_lo, f_lo = a_j, f_j
    a_j = 0.5 * (a_lo + a_hi)
    x_j = x + a_j * d
    return a_j, x_j, obj(x_j), obj.grad(x_j)


# -------------------------------------------------------------------- linear algebra

def _modified_cholesky_solve(H, g, beta=1e-3):
    """Solve ``H p = -g`` with ``H`` made positive definite by ridge inflation.

    Newton's direction only descends when the Hessian is positive definite; at a saddle or
    in a negative-curvature region the raw solve points uphill, so the diagonal is inflated
    until the factorization succeeds.
    """
    H = np.asarray(H, dtype=float)
    n = H.shape[0]
    diag = np.diag(H)
    tau = 0.0 if np.min(diag) > 0 else -np.min(diag) + beta
    identity = np.eye(n)
    for _ in range(60):
        try:
            L = np.linalg.cholesky(H + tau * identity)
        except np.linalg.LinAlgError:
            tau = max(2.0 * tau, beta)
            continue
        y = np.linalg.solve(L, -np.asarray(g, dtype=float))
        return np.linalg.solve(L.T, y)
    return -np.asarray(g, dtype=float)
