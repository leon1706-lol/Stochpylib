"""Private helpers shared across the financial_stochastics submodules.

Not part of the public API (no ``__all__``). Black-Scholes/Black-76 pricing is
centralized here so option_pricing, stochastic_vol and rate_models all price
off the same kernel; scipy.special/optimize only, never scipy.stats.
"""

import numpy as np
from scipy import optimize, special

from stochpylib.montecarlo._result import MCResult


def _norm_cdf(x):
    return special.ndtr(np.asarray(x, dtype=float))


def _norm_pdf(x):
    x = np.asarray(x, dtype=float)
    return np.exp(-0.5 * x * x) / np.sqrt(2.0 * np.pi)


def _norm_ppf(p):
    return special.ndtri(np.asarray(p, dtype=float))


def _check_kind(kind):
    if kind not in ("call", "put"):
        raise ValueError("kind must be 'call' or 'put'")


def _rng(random_state):
    return np.random.default_rng(random_state)


def _black_price(F, K, T, sigma, df=1.0, kind="call"):
    """Black-76 forward price: ``df * [F N(d1) - K N(d2)]`` (call).

    Vectorized over any combination of array/scalar ``F, K, T, sigma, df``.
    Degenerates to the correct discounted intrinsic value when ``T <= 0`` or
    ``sigma <= 0`` (no singular d1/d2 evaluated in those cells).
    """
    _check_kind(kind)
    F = np.asarray(F, dtype=float)
    K = np.asarray(K, dtype=float)
    T = np.asarray(T, dtype=float)
    sigma = np.asarray(sigma, dtype=float)
    df = np.asarray(df, dtype=float)

    degenerate = (T <= 0) | (sigma <= 0)
    T_safe = np.where(degenerate, 1.0, T)
    sigma_safe = np.where(degenerate, 1.0, sigma)

    sqrtT = np.sqrt(T_safe)
    d1 = (np.log(F / K) + 0.5 * sigma_safe**2 * T_safe) / (sigma_safe * sqrtT)
    d2 = d1 - sigma_safe * sqrtT

    if kind == "call":
        price = df * (F * _norm_cdf(d1) - K * _norm_cdf(d2))
        intrinsic = df * np.maximum(F - K, 0.0)
    else:
        price = df * (K * _norm_cdf(-d2) - F * _norm_cdf(-d1))
        intrinsic = df * np.maximum(K - F, 0.0)

    price = np.where(degenerate, intrinsic, price)
    return price if price.ndim else float(price)


def _bs_price(S, K, T, r, sigma, q=0.0, kind="call"):
    """Spot-quoted Black-Scholes-Merton with continuous dividend yield ``q``."""
    S = np.asarray(S, dtype=float)
    T = np.asarray(T, dtype=float)
    r = np.asarray(r, dtype=float)
    q = np.asarray(q, dtype=float)
    F = S * np.exp((r - q) * T)
    df = np.exp(-r * T)
    return _black_price(F, K, T, sigma, df=df, kind=kind)


def _bs_d1d2(S, K, T, r, sigma, q=0.0):
    S = np.asarray(S, dtype=float)
    K = np.asarray(K, dtype=float)
    T = np.asarray(T, dtype=float)
    r = np.asarray(r, dtype=float)
    sigma = np.asarray(sigma, dtype=float)
    q = np.asarray(q, dtype=float)
    T_safe = np.where(T <= 0, 1e-12, T)
    sigma_safe = np.where(sigma <= 0, 1e-12, sigma)
    sqrtT = np.sqrt(T_safe)
    d1 = (np.log(S / K) + (r - q + 0.5 * sigma_safe**2) * T_safe) / (sigma_safe * sqrtT)
    d2 = d1 - sigma_safe * sqrtT
    return d1, d2


def _implied_vol(price, S, K, T, r, q=0.0, kind="call", lo=1e-6, hi=5.0):
    """Invert ``_bs_price`` for volatility via bisection (robust, monotone)."""
    price = float(price)
    S, K, T, r, q = float(S), float(K), float(T), float(r), float(q)
    _check_kind(kind)
    df = np.exp(-r * T)
    if kind == "call":
        intrinsic = max(S * np.exp(-q * T) - K * df, 0.0)
        upper_bound = S * np.exp(-q * T)
    else:
        intrinsic = max(K * df - S * np.exp(-q * T), 0.0)
        upper_bound = K * df
    if price < intrinsic - 1e-10 or price > upper_bound + 1e-10:
        raise ValueError("price outside no-arbitrage bounds for implied vol")

    def f(sigma):
        return _bs_price(S, K, T, r, sigma, q=q, kind=kind) - price

    flo, fhi = f(lo), f(hi)
    if flo > 0:
        return lo
    if fhi < 0:
        return hi
    return float(optimize.brentq(f, lo, hi, xtol=1e-12, rtol=1e-12))


def _cholesky_psd(A, jitter=1e-10):
    """Cholesky factor of ``A``, adding diagonal jitter if not exactly PSD."""
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


def _solve_tridiagonal(lower, diag, upper, rhs):
    """Thomas algorithm for a tridiagonal system ``A x = rhs``.

    ``lower[1:]``, ``diag``, ``upper[:-1]`` are the three diagonals (all
    length ``n``; ``lower[0]`` and ``upper[-1]`` are ignored).
    """
    n = len(diag)
    c = np.empty(n)
    d = np.empty(n)
    c[0] = upper[0] / diag[0]
    d[0] = rhs[0] / diag[0]
    for i in range(1, n):
        m = diag[i] - lower[i] * c[i - 1]
        c[i] = upper[i] / m if i < n - 1 else 0.0
        d[i] = (rhs[i] - lower[i] * d[i - 1]) / m
    x = np.empty(n)
    x[-1] = d[-1]
    for i in range(n - 2, -1, -1):
        x[i] = d[i] - c[i] * x[i + 1]
    return x


def _mc_result(sample, method, extras=None, discount=1.0):
    """Build an ``MCResult`` from raw (already-discounted, unless ``discount``
    is given) Monte Carlo payoffs/statistics."""
    sample = np.asarray(sample, dtype=float) * discount
    n = sample.size
    mean = float(np.mean(sample))
    se = float(np.std(sample, ddof=1) / np.sqrt(n)) if n > 1 else float("nan")
    return MCResult(mean, se, n, method, extras or {})


def _matrix_log(P):
    """Real matrix logarithm via eigendecomposition (numpy only, no scipy.linalg)."""
    P = np.asarray(P, dtype=float)
    w, V = np.linalg.eig(P)
    logw = np.log(w.astype(complex))
    L = V @ np.diag(logw) @ np.linalg.inv(V)
    return np.real(L)


def _matrix_exp(G):
    """Real matrix exponential via eigendecomposition (numpy only)."""
    G = np.asarray(G, dtype=float)
    w, V = np.linalg.eig(G)
    ew = np.exp(w.astype(complex))
    E = V @ np.diag(ew) @ np.linalg.inv(V)
    return np.real(E)
