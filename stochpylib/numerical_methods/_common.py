"""Private helpers shared across the numerical_methods submodules.

Not part of the public API (no ``__all__``). Numpy-only except where noted; the module's
own algorithms are native -- ``scipy.integrate/optimize/interpolate/linalg`` are test
oracles only, never imported here. ``numpy.linalg`` (eigh/eig/qr/solve/svd) is the array
backend and is used as an explicit fast path alongside a native implementation in the
linear-algebra classes.
"""

import numpy as np


def _rng(random_state):
    return np.random.default_rng(random_state)


def _as_scalar_fn(f, vectorized=False):
    """Wrap ``f`` so it can be called on a numpy array and return an array.

    If ``vectorized`` is True, ``f`` already accepts/returns arrays and is used as-is.
    Otherwise ``f`` is treated as a plain scalar function and vectorized with
    ``np.vectorize`` (correct, not fast -- fine for the modest evaluation counts these
    solvers use).
    """
    if vectorized:
        def g(x):
            return np.asarray(f(np.asarray(x, dtype=float)), dtype=float)
        return g
    vf = np.vectorize(f, otypes=[float])

    def g(x):
        x = np.asarray(x, dtype=float)
        if x.ndim == 0:
            return np.asarray(float(f(float(x))))
        return vf(x)
    return g


def _check_interval(a, b):
    a, b = float(a), float(b)
    if not (a < b):
        raise ValueError(f"require a < b, got a={a}, b={b}")
    return a, b


def _transform_infinite(f, a, b):
    """Map an (possibly) infinite integration range to a finite one.

    Returns ``(g, lo, hi)`` such that ``integral(f, a, b) == integral(g, lo, hi)``.
    Both infinite: ``x = t / (1 - t^2)`` on ``(-1, 1)``. Half-infinite ``[a, inf)``:
    ``x = a + t / (1 - t)`` on ``[0, 1)``. Half-infinite ``(-inf, b]`` mirrors that.
    """
    a_inf = np.isinf(a)
    b_inf = np.isinf(b)
    if not a_inf and not b_inf:
        return f, a, b
    if a_inf and b_inf:
        def g(t):
            t = np.asarray(t, dtype=float)
            denom = 1.0 - t * t
            denom = np.where(np.abs(denom) < 1e-300, 1e-300, denom)
            x = t / denom
            jac = (1.0 + t * t) / (denom * denom)
            return np.asarray(f(x), dtype=float) * jac
        return g, -1.0 + 1e-10, 1.0 - 1e-10
    if b_inf:
        def g(t):
            t = np.asarray(t, dtype=float)
            denom = 1.0 - t
            denom = np.where(np.abs(denom) < 1e-300, 1e-300, denom)
            x = a + t / denom
            jac = 1.0 / (denom * denom)
            return np.asarray(f(x), dtype=float) * jac
        return g, 0.0, 1.0 - 1e-10
    # a_inf only: (-inf, b]
    def g(t):
        t = np.asarray(t, dtype=float)
        denom = 1.0 - t
        denom = np.where(np.abs(denom) < 1e-300, 1e-300, denom)
        x = b - t / denom
        jac = 1.0 / (denom * denom)
        return np.asarray(f(x), dtype=float) * jac
    return g, 0.0, 1.0 - 1e-10


def _thomas(lower, diag, upper, rhs):
    """Tridiagonal (Thomas algorithm) solve.

    ``lower`` (n-1,), ``diag`` (n,), ``upper`` (n-1,), ``rhs`` (n,) or (n, k).
    """
    diag = np.array(diag, dtype=float, copy=True)
    rhs = np.array(rhs, dtype=float, copy=True)
    lower = np.asarray(lower, dtype=float)
    upper = np.asarray(upper, dtype=float)
    n = len(diag)
    vector = rhs.ndim == 1
    if vector:
        rhs = rhs[:, None]
    for i in range(1, n):
        w = lower[i - 1] / diag[i - 1]
        diag[i] -= w * upper[i - 1]
        rhs[i] -= w * rhs[i - 1]
    x = np.empty_like(rhs)
    x[-1] = rhs[-1] / diag[-1]
    for i in range(n - 2, -1, -1):
        x[i] = (rhs[i] - upper[i] * x[i + 1]) / diag[i]
    return x[:, 0] if vector else x


def _numeric_derivative(f, x, eps=1e-6):
    """Central-difference derivative of a scalar function at scalar/array x."""
    x = np.asarray(x, dtype=float)
    h = eps * np.maximum(1.0, np.abs(x))
    return (np.asarray(f(x + h), dtype=float) - np.asarray(f(x - h), dtype=float)) / (2 * h)


def _numeric_jacobian(f, x, eps=1e-7):
    """Central-difference Jacobian of a vector function f: R^n -> R^m at x (n,)."""
    x = np.asarray(x, dtype=float)
    f0 = np.asarray(f(x), dtype=float)
    n = len(x)
    m = len(np.atleast_1d(f0))
    J = np.empty((m, n))
    for j in range(n):
        h = eps * max(1.0, abs(x[j]))
        xp = x.copy()
        xp[j] += h
        xm = x.copy()
        xm[j] -= h
        J[:, j] = (np.asarray(f(xp), dtype=float).ravel() - np.asarray(f(xm), dtype=float).ravel()) / (2 * h)
    return J


def _conjugate_gradient(matvec, b, x0=None, tol=1e-10, max_iter=None, precond=None):
    """Matrix-free conjugate gradient for a symmetric positive-definite system.

    ``matvec(x) -> A @ x``. Returns ``(x, n_iter, converged)``.
    """
    b = np.asarray(b, dtype=float)
    n = len(b)
    if max_iter is None:
        max_iter = 10 * n
    x = np.zeros(n) if x0 is None else np.array(x0, dtype=float, copy=True)
    r = b - matvec(x)
    z = r if precond is None else precond(r)
    p = z.copy()
    rz_old = float(r @ z)
    b_norm = float(np.linalg.norm(b)) or 1.0
    if float(np.linalg.norm(r)) / b_norm < tol:
        return x, 0, True
    for k in range(1, max_iter + 1):
        Ap = matvec(p)
        denom = float(p @ Ap)
        if abs(denom) < 1e-300:
            return x, k, False
        alpha = rz_old / denom
        x = x + alpha * p
        r = r - alpha * Ap
        if float(np.linalg.norm(r)) / b_norm < tol:
            return x, k, True
        z = r if precond is None else precond(r)
        rz_new = float(r @ z)
        beta = rz_new / rz_old
        p = z + beta * p
        rz_old = rz_new
    return x, max_iter, False


def _householder_vector(x):
    """Householder vector v (unnormalized, v[0]=1) and beta s.t. H = I - beta v v^T
    zeros x[1:] and maps x -> (-sign(x0)*||x||, 0, ..., 0)."""
    x = np.asarray(x, dtype=float)
    sigma = float(x[1:] @ x[1:]) if len(x) > 1 else 0.0
    v = x.copy()
    v[0] = 1.0
    if sigma == 0.0 and x[0] >= 0:
        return v, 0.0
    mu = np.sqrt(x[0] ** 2 + sigma)
    if x[0] <= 0:
        v0 = x[0] - mu
    else:
        v0 = -sigma / (x[0] + mu)
    beta = 2 * v0 ** 2 / (sigma + v0 ** 2)
    v = x / v0
    v[0] = 1.0
    return v, beta


def _hessenberg(A):
    """Reduce A to upper Hessenberg form H = Q^T A Q via Householder reflections.

    Returns (H, Q).
    """
    A = np.array(A, dtype=float, copy=True)
    n = A.shape[0]
    Q = np.eye(n)
    for k in range(n - 2):
        x = A[k + 1:, k]
        if np.allclose(x[1:], 0.0) and len(x) <= 1:
            continue
        v, beta = _householder_vector(x)
        if beta == 0.0:
            continue
        Hk = np.eye(len(x)) - beta * np.outer(v, v)
        A[k + 1:, k:] = Hk @ A[k + 1:, k:]
        A[:, k + 1:] = A[:, k + 1:] @ Hk
        Qk = np.eye(n)
        Qk[k + 1:, k + 1:] = Hk
        Q = Q @ Qk
    # clean sub-sub-diagonal numerical noise
    for i in range(2, n):
        A[i, :i - 1] = 0.0
    return A, Q


# ---- Gauss-Kronrod 7-15 tables (QUADPACK, published, used only as building blocks) ----
# _GK15_NODES[i] / _GK15_WEIGHTS[i] (i=0..6) are the positive Kronrod abscissae/weights;
# index 7 is the center (x=0). The embedded 7-point Gauss rule reuses Kronrod nodes at
# positions 1, 3, 5 (positive side) plus the center, with its own weights _GK7_WEIGHTS.

_GK15_NODES = np.array([
    0.991455371120813, 0.949107912342759, 0.864864423359769,
    0.741531185599394, 0.586087235467691, 0.405845151377397,
    0.207784955007898, 0.0,
])
_GK15_WEIGHTS = np.array([
    0.022935322010529, 0.063092092629979, 0.104790010322250,
    0.140653259715525, 0.169004726639267, 0.190350578064785,
    0.204432940075298, 0.209482141084728,
])
_GK7_WEIGHTS = np.array([
    0.129484966168870, 0.279705391489277, 0.381830050505119, 0.417959183673469,
])


def _gauss_kronrod_15(f, a, b):
    """Single-panel Gauss-Kronrod 7-15 estimate on [a, b].

    Returns (k15, error_estimate, g7).
    """
    c = 0.5 * (a + b)
    h = 0.5 * (b - a)
    pos = _GK15_NODES[:-1]                       # 7 positive nonzero nodes
    nodes = c + h * np.concatenate([pos, [0.0], -pos])
    fv = np.asarray(f(nodes), dtype=float)
    w15 = np.concatenate([_GK15_WEIGHTS[:-1], [_GK15_WEIGHTS[-1]], _GK15_WEIGHTS[:-1]])
    k15 = h * float(np.sum(w15 * fv))
    # embedded Gauss-7 reuses the same 15 evaluations: fv[1,3,5] are +node, fv[9,11,13]
    # are the mirrored -node (pos block is fv[0:7], center fv[7], mirror block fv[8:15]),
    # fv[7] is the center.
    f_pos = fv[[1, 3, 5]]
    f_neg = fv[[9, 11, 13]]
    f_center = fv[7]
    g7 = h * float(np.sum(_GK7_WEIGHTS[:3] * (f_pos + f_neg)) + _GK7_WEIGHTS[3] * f_center)
    err = abs(k15 - g7)
    scaled = (200.0 * err) ** 1.5
    err_est = min(err, scaled)
    return k15, err_est, g7
