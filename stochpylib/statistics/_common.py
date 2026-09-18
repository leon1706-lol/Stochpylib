"""Private helpers shared across the statistics submodules.

Not part of the public API (no ``__all__``). Every distributional building block
here comes from ``scipy.special``/``scipy.optimize`` directly -- library code
never wraps ``scipy.stats`` (that module is the test suite's independent
oracle only, per AGENTS.md).
"""

import numpy as np
from scipy import optimize, special


def _rng(random_state):
    return np.random.default_rng(random_state)


def _as_1d(x, name="x"):
    arr = np.asarray(x, dtype=float).ravel()
    if arr.size == 0:
        raise ValueError(f"{name} must be non-empty")
    if not np.all(np.isfinite(arr)):
        raise ValueError(f"{name} contains non-finite values")
    return arr


def _as_2d(X, name="X"):
    arr = np.asarray(X, dtype=float)
    if arr.ndim == 1:
        arr = arr[:, None]
    if arr.ndim != 2 or arr.shape[0] == 0:
        raise ValueError(f"{name} must be a (n, p) array")
    if not np.all(np.isfinite(arr)):
        raise ValueError(f"{name} contains non-finite values")
    return arr


def _design(X, fit_intercept=True):
    """Build an OLS/GLM design matrix, prepending an intercept column if requested."""
    X = _as_2d(X, "X")
    if fit_intercept:
        return np.column_stack([np.ones(X.shape[0]), X])
    return X


# --------------------------------------------------------------------- distributions

def _norm_cdf(x):
    return special.ndtr(np.asarray(x, dtype=float))


def _norm_pdf(x):
    x = np.asarray(x, dtype=float)
    return np.exp(-0.5 * x * x) / np.sqrt(2.0 * np.pi)


def _norm_ppf(p):
    return special.ndtri(np.asarray(p, dtype=float))


def _t_cdf(x, df):
    x = np.asarray(x, dtype=float)
    return special.stdtr(df, x)


def _t_ppf(p, df):
    return special.stdtrit(df, np.asarray(p, dtype=float))


def _t_sf(x, df):
    """Student-t survival function P(T > x), by symmetry (avoids 1-cdf cancellation)."""
    return special.stdtr(df, -np.asarray(x, dtype=float))


def _chi2_cdf(x, df):
    return special.chdtr(df, np.asarray(x, dtype=float))


def _chi2_sf(x, df):
    return special.chdtrc(df, np.asarray(x, dtype=float))


def _chi2_ppf(p, df):
    return special.chdtri(df, 1.0 - np.asarray(p, dtype=float))


def _f_cdf(x, dfn, dfd):
    return special.fdtr(dfn, dfd, np.asarray(x, dtype=float))


def _f_sf(x, dfn, dfd):
    return special.fdtrc(dfn, dfd, np.asarray(x, dtype=float))


def _f_ppf(p, dfn, dfd):
    return special.fdtri(dfn, dfd, np.asarray(p, dtype=float))


def _kolmogorov_sf(lam):
    """Asymptotic Kolmogorov distribution survival function (``scipy.special.kolmogorov``)."""
    return special.kolmogorov(np.asarray(lam, dtype=float))


def _pvalue_from_z(z, alternative="two-sided"):
    if alternative == "two-sided":
        return float(2.0 * _norm_cdf(-abs(z)))
    if alternative == "greater":
        return float(_norm_cdf(-z))
    if alternative == "less":
        return float(_norm_cdf(z))
    raise ValueError("alternative must be 'two-sided', 'greater', or 'less'")


def _pvalue_from_t(t, df, alternative="two-sided"):
    if alternative == "two-sided":
        return float(2.0 * _t_sf(abs(t), df))
    if alternative == "greater":
        return float(_t_sf(t, df))
    if alternative == "less":
        return float(_t_cdf(t, df))
    raise ValueError("alternative must be 'two-sided', 'greater', or 'less'")


# --------------------------------------------------------------------- studentized range
#
# q ~ range of k iid N(0,1) draws, scaled by an independent sqrt(chi2_df/df) estimate of
# scale. P(q < x) = integral over s>0 of f_df(s) * [ k * integral phi(z)[Phi(z)-Phi(z-x*s)]^(k-1) dz ] ds
# where f_df is the density of s = sqrt(chi2_df/df) (so s has mean ~1). Used by tukey_hsd.

_GL96_X, _GL96_W = np.polynomial.legendre.leggauss(96)
_GL128_X, _GL128_W = np.polynomial.legendre.leggauss(128)


def _inner_range_integral(x, k):
    """k * integral_{-inf}^{inf} phi(z) [Phi(z) - Phi(z - x)]^(k-1) dz, vectorized over x."""
    x = np.atleast_1d(np.asarray(x, dtype=float))
    lo, hi = -8.5, 8.5
    z = 0.5 * (hi - lo) * _GL96_X + 0.5 * (hi + lo)
    w = 0.5 * (hi - lo) * _GL96_W
    # z: (96,), x: (m,) -> broadcast to (m, 96)
    zz = z[None, :]
    xx = x[:, None]
    diff = _norm_cdf(zz) - _norm_cdf(zz - xx)
    diff = np.clip(diff, 0.0, None)
    integrand = _norm_pdf(zz) * diff ** (k - 1)
    val = k * np.sum(integrand * w[None, :], axis=1)
    return np.clip(val, 0.0, 1.0)


def _ptukey(q, k, df):
    """CDF of the studentized range statistic at q, for k means and df error d.o.f.

    ``df = np.inf`` uses the standard-normal scale (no outer integral).
    """
    q = np.atleast_1d(np.asarray(q, dtype=float))
    out = np.empty_like(q)
    if np.isinf(df):
        out = _inner_range_integral(q, k)
        return out if out.size > 1 else float(out[0])

    # s has density of sqrt(chi2_df/df); peak near s=1, spread ~ 1/sqrt(2df).
    lo = np.sqrt(max(_chi2_ppf(1e-14, df), 1e-8) / df)
    hi = np.sqrt(_chi2_ppf(1.0 - 1e-14, df) / df)
    s = 0.5 * (hi - lo) * _GL128_X + 0.5 * (hi + lo)
    w = 0.5 * (hi - lo) * _GL128_W
    log_f = (
        0.5 * df * np.log(df) - special.gammaln(0.5 * df) - (0.5 * df - 1) * np.log(2.0)
        + (df - 1) * np.log(s) - 0.5 * df * s * s
    )
    f_s = np.exp(log_f)

    # For each q, integrate inner(q * s, k) * f_s(s) ds
    qs = q[:, None] * s[None, :]  # (m, 128)
    inner = _inner_range_integral(qs.ravel(), k).reshape(qs.shape)
    out = np.sum(inner * (f_s * w)[None, :], axis=1)
    out = np.clip(out, 0.0, 1.0)
    return out if out.size > 1 else float(out[0])


def _qtukey(p, k, df, lo=0.1, hi=30.0):
    """Inverse of _ptukey: studentized range critical value at cumulative prob p."""
    def f(q):
        return _ptukey(q, k, df) - p

    flo, fhi = f(lo), f(hi)
    if flo > 0:
        return lo
    if fhi < 0:
        return hi
    return float(optimize.brentq(f, lo, hi, xtol=1e-10, rtol=1e-12))


# --------------------------------------------------------------------- calculus

def _numeric_grad(f, x, eps=1e-6):
    """Central-difference gradient of a scalar function f(x) -> float."""
    x = np.asarray(x, dtype=float)
    g = np.empty_like(x)
    for i in range(len(x)):
        dx = np.zeros_like(x)
        dx[i] = eps * max(1.0, abs(x[i]))
        g[i] = (f(x + dx) - f(x - dx)) / (2 * dx[i])
    return g


def _numeric_jacobian(f, x, eps=1e-6):
    """Central-difference Jacobian of a vector function f(x) -> (m,)."""
    x = np.asarray(x, dtype=float)
    f0 = np.atleast_1d(f(x))
    J = np.empty((len(f0), len(x)))
    for i in range(len(x)):
        dx = np.zeros_like(x)
        dx[i] = eps * max(1.0, abs(x[i]))
        J[:, i] = (np.atleast_1d(f(x + dx)) - np.atleast_1d(f(x - dx))) / (2 * dx[i])
    return J


def _numeric_hessian(f, x, eps=1e-4):
    """Central-difference Hessian of a scalar function f(x) -> float."""
    x = np.asarray(x, dtype=float)
    n = len(x)
    H = np.empty((n, n))
    steps = eps * np.maximum(1.0, np.abs(x))
    f0 = f(x)
    for i in range(n):
        ei = np.zeros(n)
        ei[i] = steps[i]
        for j in range(i, n):
            if i == j:
                H[i, i] = (f(x + ei) - 2 * f0 + f(x - ei)) / steps[i] ** 2
            else:
                ej = np.zeros(n)
                ej[j] = steps[j]
                fpp = f(x + ei + ej)
                fpm = f(x + ei - ej)
                fmp = f(x - ei + ej)
                fmm = f(x - ei - ej)
                H[i, j] = H[j, i] = (fpp - fpm - fmp + fmm) / (4 * steps[i] * steps[j])
    return H


def _cholesky_psd(A, jitter=1e-10):
    """Cholesky factor of A, adding diagonal jitter if not exactly PSD."""
    A = 0.5 * (np.asarray(A, dtype=float) + np.asarray(A, dtype=float).T)
    n = A.shape[0]
    eps = jitter * (np.trace(A) / n if np.trace(A) > 0 else 1.0)
    for _ in range(7):
        try:
            return np.linalg.cholesky(A)
        except np.linalg.LinAlgError:
            A = A + eps * np.eye(n)
            eps *= 2.0
    raise np.linalg.LinAlgError("matrix not positive semi-definite after jittering")
