"""Shared numerical plumbing for the timeseries module."""

import numpy as np


def as_1d(x, name="series"):
    """Validate and coerce input to a 1-D float array."""
    arr = np.asarray(x, dtype=float).ravel()
    if arr.size == 0:
        raise ValueError(f"{name} must be non-empty")
    if not np.all(np.isfinite(arr)):
        raise ValueError(f"{name} contains non-finite values")
    return arr


def as_2d(x, name="series"):
    """Validate and coerce input to a 2-D (T, k) float array."""
    arr = np.asarray(x, dtype=float)
    if arr.ndim == 1:
        arr = arr[:, None]
    if arr.ndim != 2 or arr.shape[0] == 0:
        raise ValueError(f"{name} must be a (T, k) array")
    if not np.all(np.isfinite(arr)):
        raise ValueError(f"{name} contains non-finite values")
    return arr


def lag_matrix(y, p):
    """OLS design ``[1, y_{t-1}, ..., y_{t-p}]`` with targets ``y_t`` for t >= p.

    Returns ``(X, target)`` of shapes ``(T-p, p+1)`` and ``(T-p,)``.
    """
    y = as_1d(y)
    if p < 1:
        raise ValueError("p must be >= 1")
    T = len(y)
    X = np.ones((T - p, p + 1))
    for j in range(1, p + 1):
        X[:, j] = y[p - j : T - j]
    return X, y[p:]


def difference(x, d=1):
    x = np.asarray(x, dtype=float)
    for _ in range(int(d)):
        x = np.diff(x)
    return x


def seasonal_difference(x, s, D=1):
    x = np.asarray(x, dtype=float)
    s = int(s)
    if s < 1:
        raise ValueError("seasonal period s must be >= 1")
    for _ in range(int(D)):
        x = x[s:] - x[:-s]
    return x


def integrate_levels(diff_path, last_values):
    """Undo differencing: rebuild level forecasts from a differenced forecast path.

    ``last_values`` holds the final ``d`` observed levels (oldest first); the newest is
    integrated repeatedly so a d-times differenced forecast becomes a level forecast.
    """
    levels = list(np.asarray(last_values, dtype=float).ravel()[-int(len(last_values)) :])
    out = np.asarray(diff_path, dtype=float)
    d = len(levels)
    # iteratively accumulate from the most recent level backwards
    history = levels[::-1]  # [y_T, y_{T-1}, ...]
    for _ in range(d):
        prev = history[0]
        acc = []
        for step in out:
            prev = prev + step
            acc.append(prev)
        out = np.array(acc)
        history = [history[0]] + history  # last integrated value becomes the new anchor
    return out


def frac_diff_weights(d, tol=1e-6, max_terms=2000):
    """Truncated binomial weights ``(1-B)^d`` up to the point where terms fall below tol.

    Returns weights w with w[0]=1; convolving ``w[::-1]`` against the series yields the
    fractionally differenced input (fixed-width window variant).
    """
    if abs(d) >= 1:
        raise ValueError("|d| must be < 1 for fractional differencing")
    w = [1.0]
    for k in range(1, max_terms):
        nxt = -w[-1] * (d - k + 1) / k
        if abs(nxt) < tol and k > int(abs(d)):
            break
        w.append(nxt)
    return np.array(w)


def psi_weights_ar(phi, h):
    """MA(infinity) coefficients psi_1..psi_h of an AR process (psi_0 = 1 implied)."""
    phi = np.asarray(phi, dtype=float)
    psi = np.zeros(h)
    psi[0] = 1.0
    for i in range(1, h):
        total = 0.0
        for j in range(1, min(i, len(phi)) + 1):
            total += phi[j - 1] * psi[i - j]
        psi[i] = total
    return psi


def aic_bic(sigma2, n, n_params):
    loglik = -0.5 * n * (np.log(2 * np.pi * sigma2) + 1.0)
    aic = -2 * loglik + 2 * n_params
    bic = -2 * loglik + np.log(n) * n_params
    return float(aic), float(bic)


def nw_lags(n):
    """Newey-West automatic lag: floor(4 (n/100)^{2/9})."""
    return int(np.floor(4.0 * (n / 100.0) ** (2.0 / 9.0)))
