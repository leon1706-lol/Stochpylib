"""Private helpers shared across the nonparametric submodules.

Not part of the public API. Kernel functions, bandwidth-selection rules,
weighted PAVA, and B-spline machinery are implemented natively; rank/tie
handling and Kendall/Spearman estimators are reused from
``stochpylib.copulas._utils`` and ``stochpylib.statistics._common`` rather
than duplicated -- library code never wraps ``scipy.stats`` (that module is
the test suite's independent oracle only, per AGENTS.md).
"""

import numpy as np
from scipy import special

from stochpylib.statistics._common import (  # noqa: F401  (re-exported for siblings)
    _as_1d, _as_2d, _chi2_cdf, _chi2_ppf, _chi2_sf, _norm_cdf, _norm_pdf,
    _norm_ppf, _rng, _t_ppf, _t_sf,
)
from stochpylib.copulas._utils import _rank_ties  # noqa: F401


# --------------------------------------------------------------------- kernels
#
# Each kernel is K(u); R(K) = integral K(u)^2 du and mu2(K) = integral u^2 K(u) du
# are used by bandwidth rules and asymptotic-variance formulas in density/regression.

def _kernel_gaussian(u):
    return np.exp(-0.5 * u * u) / np.sqrt(2.0 * np.pi)


def _kernel_epanechnikov(u):
    out = 0.75 * (1.0 - u * u)
    return np.where(np.abs(u) <= 1.0, out, 0.0)


def _kernel_biweight(u):
    out = 15.0 / 16.0 * (1.0 - u * u) ** 2
    return np.where(np.abs(u) <= 1.0, out, 0.0)


def _kernel_triangular(u):
    out = 1.0 - np.abs(u)
    return np.where(np.abs(u) <= 1.0, out, 0.0)


def _kernel_uniform(u):
    return np.where(np.abs(u) <= 1.0, 0.5, 0.0)


def _kernel_cosine(u):
    out = (np.pi / 4.0) * np.cos(np.pi * u / 2.0)
    return np.where(np.abs(u) <= 1.0, out, 0.0)


_KERNELS = {
    "gaussian": _kernel_gaussian,
    "epanechnikov": _kernel_epanechnikov,
    "biweight": _kernel_biweight,
    "triangular": _kernel_triangular,
    "uniform": _kernel_uniform,
    "cosine": _kernel_cosine,
}

_KERNEL_R = {
    "gaussian": 1.0 / (2.0 * np.sqrt(np.pi)),
    "epanechnikov": 3.0 / 5.0,
    "biweight": 5.0 / 7.0,
    "triangular": 2.0 / 3.0,
    "uniform": 0.5,
    "cosine": np.pi ** 2 / 16.0,
}
_KERNEL_MU2 = {
    "gaussian": 1.0,
    "epanechnikov": 1.0 / 5.0,
    "biweight": 1.0 / 7.0,
    "triangular": 1.0 / 6.0,
    "uniform": 1.0 / 3.0,
    "cosine": 1.0 - 8.0 / np.pi ** 2,
}


def _kernel_fn(kernel):
    if kernel not in _KERNELS:
        raise ValueError(f"unknown kernel {kernel!r}; choose from {sorted(_KERNELS)}")
    return _KERNELS[kernel]


# --------------------------------------------------------------------- bandwidths

def _bw_scott(x):
    """Scott's rule, matching scipy.stats.gaussian_kde's 1-D ``scotts_factor``:
    h = sigma * n^(-1/5)."""
    x = np.asarray(x, dtype=float)
    n = len(x)
    sigma = np.std(x, ddof=1)
    return float(sigma * n ** (-1.0 / 5.0))


def _bw_silverman(x):
    """Silverman's rule of thumb (1986), matching scipy.stats.gaussian_kde's
    1-D ``silverman_factor``: h = (4/3)^(1/5) * sigma * n^(-1/5)."""
    x = np.asarray(x, dtype=float)
    n = len(x)
    sigma = np.std(x, ddof=1)
    return float((4.0 / 3.0) ** (1.0 / 5.0) * sigma * n ** (-1.0 / 5.0))


def _bw_normal_reference(x, kernel="gaussian"):
    """statsmodels' ``bw="normal_reference"``: h = const * A * n^(-1/5) with
    A = min(std, IQR/1.349), per-kernel constant from
    statsmodels.nonparametric.bandwidths (gaussian case = Silverman's 1.06 rule)."""
    x = np.asarray(x, dtype=float)
    n = len(x)
    std = np.std(x, ddof=1)
    q75, q25 = np.percentile(x, [75, 25])
    iqr = q75 - q25
    a = min(std, iqr / 1.349) if iqr > 0 else std
    const = {"gaussian": 1.0592, "epanechnikov": 2.34, "biweight": 2.78,
             "triangular": 2.43, "uniform": 1.84, "cosine": 2.41}.get(kernel, 1.0592)
    return float(const * a * n ** (-1.0 / 5.0))


def _bw_lscv(x, kernel="gaussian", h_grid=None):
    """Least-squares cross-validation bandwidth: grid-minimizes the leave-one-out
    LSCV score, anchored on Silverman's rule."""
    x = np.asarray(x, dtype=float)
    n = len(x)
    h_silverman = _bw_silverman(x)
    if h_grid is None:
        h_grid = np.geomspace(0.1 * h_silverman, 3.0 * h_silverman, 40)
    K = _kernel_fn(kernel)
    diffs = x[:, None] - x[None, :]
    best_h, best_score = float(h_grid[0]), np.inf
    for h in h_grid:
        u = diffs / h
        Kmat = K(u)
        if kernel == "gaussian":
            conv = np.exp(-0.25 * u * u) / np.sqrt(4.0 * np.pi)
        else:
            grid = np.linspace(-3, 3, 400)
            Kg = K(grid)
            conv_vals = np.convolve(Kg, Kg, mode="same") * (grid[1] - grid[0])
            conv = np.interp(np.abs(u), np.abs(grid), conv_vals)
        term1 = np.sum(conv) / (n * n * h)
        np.fill_diagonal(Kmat, 0.0)
        term2 = 2.0 * np.sum(Kmat) / (n * (n - 1) * h)
        score = term1 - term2
        if score < best_score:
            best_score, best_h = score, float(h)
    return best_h


def _select_bandwidth(x, bandwidth, kernel="gaussian"):
    if isinstance(bandwidth, (int, float)):
        return float(bandwidth)
    if bandwidth == "scott":
        return _bw_scott(x)
    if bandwidth == "silverman":
        return _bw_silverman(x)
    if bandwidth == "normal_reference":
        return _bw_normal_reference(x, kernel)
    if bandwidth == "lscv":
        return _bw_lscv(x, kernel)
    raise ValueError(f"unknown bandwidth rule {bandwidth!r}")


# --------------------------------------------------------------------- isotonic (weighted PAVA)

def _pava_weighted(y, w=None, increasing=True):
    """Weighted pool-adjacent-violators isotonic regression.

    Returns the fitted values in the original order. ``increasing=False``
    fits a non-increasing sequence by negating, fitting, and negating back.
    """
    y = np.asarray(y, dtype=float)
    n = len(y)
    w = np.ones(n) if w is None else np.asarray(w, dtype=float)
    if not increasing:
        return -_pava_weighted(-y, w, increasing=True)
    val, wt, size = [], [], []
    for i in range(n):
        val.append(y[i])
        wt.append(w[i])
        size.append(1)
        while len(val) > 1 and val[-2] > val[-1]:
            v2, w2, s2 = val.pop(), wt.pop(), size.pop()
            v1, w1, s1 = val.pop(), wt.pop(), size.pop()
            nv = (v1 * w1 + v2 * w2) / (w1 + w2)
            val.append(nv)
            wt.append(w1 + w2)
            size.append(s1 + s2)
    out = np.empty(n)
    pos = 0
    for v, s in zip(val, size):
        out[pos:pos + s] = v
        pos += s
    return out


# --------------------------------------------------------------------- B-splines

def _bspline_knots(x, n_knots, degree):
    """Clamped uniform knot vector spanning ``x``'s range with ``n_knots`` interior
    control intervals."""
    lo, hi = float(np.min(x)), float(np.max(x))
    interior = np.linspace(lo, hi, n_knots + 1)[1:-1]
    knots = np.concatenate([
        np.full(degree + 1, lo), interior, np.full(degree + 1, hi),
    ])
    return knots


def _bspline_design(x, knots, degree):
    """B-spline design matrix via scipy.interpolate.BSpline -- raw numerical
    machinery (like numpy.linalg elsewhere), not a statistical oracle."""
    from scipy.interpolate import BSpline
    x = np.asarray(x, dtype=float)
    n_basis = len(knots) - degree - 1
    B = np.zeros((len(x), n_basis))
    for j in range(n_basis):
        c = np.zeros(n_basis)
        c[j] = 1.0
        spline = BSpline(knots, c, degree, extrapolate=True)
        B[:, j] = spline(x)
    return B


# --------------------------------------------------------------------- Owen's empirical likelihood

def _el_logstar(z, eps):
    """Owen's pseudo-log: log(z) for z >= eps, a quadratic extrapolation below eps
    that keeps the function twice differentiable (Owen 2001, Sec 3.14)."""
    z = np.asarray(z, dtype=float)
    out = np.empty_like(z)
    hi = z >= eps
    out[hi] = np.log(z[hi])
    lo = ~hi
    zl = z[lo]
    out[lo] = np.log(eps) - 1.5 + 2.0 * zl / eps - 0.5 * (zl / eps) ** 2
    return out


def _el_logstar1(z, eps):
    z = np.asarray(z, dtype=float)
    out = np.empty_like(z)
    hi = z >= eps
    out[hi] = 1.0 / z[hi]
    lo = ~hi
    out[lo] = 2.0 / eps - z[lo] / eps ** 2
    return out


def _el_logstar2(z, eps):
    z = np.asarray(z, dtype=float)
    out = np.empty_like(z)
    hi = z >= eps
    out[hi] = -1.0 / z[hi] ** 2
    lo = ~hi
    out[lo] = -1.0 / eps ** 2
    return out


# --------------------------------------------------------------------- DKW / distance helpers

def _cdf_grid_from_pdf(grid, pdf_vals):
    """Cumulative-trapezoid CDF grid from a pdf sampled on ``grid``, normalized
    to end exactly at 1 (mops up the grid's own numerical integration error)."""
    from scipy.integrate import cumulative_trapezoid
    cum = np.concatenate([[0.0], cumulative_trapezoid(pdf_vals, grid)])
    total = cum[-1]
    return cum / total if total > 0 else cum


def _dkw_band(n, level):
    """Dvoretzky-Kiefer-Wolfowitz simultaneous confidence band half-width."""
    alpha = 1.0 - level
    return float(np.sqrt(np.log(2.0 / alpha) / (2.0 * n)))


def _pairwise_dist(x):
    x = np.asarray(x, dtype=float).ravel()
    return np.abs(x[:, None] - x[None, :])


def _double_center(D):
    D = np.asarray(D, dtype=float)
    row = D.mean(axis=1, keepdims=True)
    col = D.mean(axis=0, keepdims=True)
    grand = D.mean()
    return D - row - col + grand


def _u_center(D):
    """U-centering for the bias-corrected distance covariance estimator
    (Szekely & Rizzo 2014)."""
    D = np.asarray(D, dtype=float)
    n = D.shape[0]
    row = D.sum(axis=1, keepdims=True) / (n - 2)
    col = D.sum(axis=0, keepdims=True) / (n - 2)
    grand = D.sum() / ((n - 1) * (n - 2))
    out = D - row - col + grand
    np.fill_diagonal(out, 0.0)
    return out
