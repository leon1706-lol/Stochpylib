"""Descriptive statistics: location, spread, shape, and association.

All functions accept array-likes and coerce to float via ``_common._as_1d``/``_as_2d``.
No randomness, so none take ``random_state``.
"""

import numpy as np
from scipy import special

from stochpylib.statistics._common import _as_1d, _norm_cdf, _t_sf
from stochpylib.statistics._result import DescribeResult, EstimateResult

__all__ = [
    "correlation", "covariance", "describe", "iqr", "kurtosis", "mean", "median",
    "mode", "quantile", "skewness", "std", "variance",
]


# ---------------------------------------------------------------- location / spread

def mean(x, weights=None, trim=0.0, axis=None):
    """Arithmetic mean, optionally weighted or symmetrically trimmed.

    ``weights`` and ``trim`` are mutually exclusive. ``trim`` is the fraction removed
    from *each* tail (matches ``scipy.stats.trim_mean``'s ``proportiontocut``).
    """
    x = np.asarray(x, dtype=float)
    if weights is not None and trim:
        raise ValueError("weights and trim cannot both be given")
    if weights is not None:
        weights = np.asarray(weights, dtype=float)
        return float(np.average(x, weights=weights, axis=axis)) if axis is None else np.average(x, weights=weights, axis=axis)
    if trim:
        if not (0 <= trim < 0.5):
            raise ValueError("trim must be in [0, 0.5)")
        xs = np.sort(x, axis=axis if axis is not None else -1)
        n = xs.shape[axis] if axis is not None else xs.shape[-1]
        k = int(np.floor(n * trim))
        if axis is None:
            xs = xs.ravel()
            return float(np.mean(xs[k: n - k] if k else xs))
        sl = [slice(None)] * xs.ndim
        sl[axis] = slice(k, n - k) if k else slice(None)
        return np.mean(xs[tuple(sl)], axis=axis)
    return float(np.mean(x)) if axis is None else np.mean(x, axis=axis)


def median(x, axis=None):
    """Sample median."""
    x = np.asarray(x, dtype=float)
    return float(np.median(x)) if axis is None else np.median(x, axis=axis)


def mode(x, bins=None):
    """Most frequent value(s) (or, with ``bins``, the modal histogram bin center).

    Returns ``(mode_value, count)``; ``mode_value`` is the smallest value achieving
    the maximum count when there are ties (matches ``scipy.stats.mode``).
    """
    x = _as_1d(x)
    if bins is not None:
        counts, edges = np.histogram(x, bins=bins)
        i = int(np.argmax(counts))
        center = 0.5 * (edges[i] + edges[i + 1])
        return float(center), int(counts[i])
    vals, counts = np.unique(x, return_counts=True)
    best = counts == counts.max()
    return float(vals[best][0]), int(counts.max())


def variance(x, ddof=1, weights=None):
    """Sample variance; a reliability (frequency) weight formula is used when
    ``weights`` is given: ``sum(w (x - xbar)^2) / (sum(w) - ddof * sum(w)/n)``."""
    x = np.asarray(x, dtype=float)
    if weights is None:
        return float(np.var(x, ddof=ddof))
    w = np.asarray(weights, dtype=float)
    n = len(x)
    xbar = np.average(x, weights=w)
    num = np.sum(w * (x - xbar) ** 2)
    denom = np.sum(w) - ddof * np.sum(w) / n
    return float(num / denom)


def std(x, ddof=1, weights=None):
    """Sample standard deviation (see :func:`variance` for the weighted formula)."""
    return float(np.sqrt(variance(x, ddof=ddof, weights=weights)))


def quantile(x, q, method="linear"):
    """Sample quantile(s); ``method`` accepts any ``numpy.quantile`` interpolation name
    (the nine Hyndman-Fan types: linear, lower, higher, nearest, midpoint,
    weibull, median_unbiased, normal_unbiased, averaged_inverted_cdf)."""
    x = _as_1d(x)
    return np.quantile(x, q, method=method)


def iqr(x, scale=1.0, method="linear"):
    """Interquartile range Q3 - Q1, optionally rescaled (``scale=1.349`` for a
    normal-consistent robust std estimate)."""
    x = _as_1d(x)
    q1, q3 = np.quantile(x, [0.25, 0.75], method=method)
    return float((q3 - q1) / scale)


def skewness(x, bias=True):
    """Fisher-Pearson skewness g1 (``bias=True``) or the bias-adjusted sample
    skewness G1 (``bias=False``)."""
    x = _as_1d(x)
    n = len(x)
    m = x - np.mean(x)
    m2 = np.mean(m ** 2)
    m3 = np.mean(m ** 3)
    g1 = m3 / m2 ** 1.5 if m2 > 0 else 0.0
    if bias:
        return float(g1)
    return float(np.sqrt(n * (n - 1)) / (n - 2) * g1)


def kurtosis(x, excess=True, bias=True):
    """Kurtosis g2 (``bias=True``) or bias-adjusted G2 (``bias=False``); ``excess=True``
    (default) subtracts 3 so a normal distribution has kurtosis 0."""
    x = _as_1d(x)
    n = len(x)
    m = x - np.mean(x)
    m2 = np.mean(m ** 2)
    m4 = np.mean(m ** 4)
    g2 = m4 / m2 ** 2 - 3.0 if m2 > 0 else -3.0
    if not bias:
        g2 = ((n - 1) / ((n - 2) * (n - 3))) * ((n + 1) * g2 + 6.0)
    return float(g2 if excess else g2 + 3.0)


def covariance(x, y=None, ddof=1):
    """Scalar covariance of ``x``,``y`` or the covariance matrix of a (n, p) array."""
    if y is not None:
        x = _as_1d(x)
        y = _as_1d(y)
        return float(np.cov(x, y, ddof=ddof)[0, 1])
    x = np.asarray(x, dtype=float)
    return np.cov(x, rowvar=False, ddof=ddof)


def correlation(x, y=None, method="pearson"):
    """Correlation coefficient (pearson/spearman/kendall). With ``y=None`` and a
    (n, p) matrix ``x``, returns the full correlation matrix (no p-values); with two
    1-D arrays, returns an :class:`EstimateResult` (``estimate`` = coefficient,
    ``extras['pvalue']`` = two-sided significance test p-value)."""
    if method not in ("pearson", "spearman", "kendall"):
        raise ValueError("method must be 'pearson', 'spearman', or 'kendall'")
    if y is None:
        X = np.asarray(x, dtype=float)
        if X.ndim != 2:
            raise ValueError("x must be a (n, p) array when y is None")
        if method == "pearson":
            return np.corrcoef(X, rowvar=False)
        if method == "spearman":
            R = np.apply_along_axis(_rankdata, 0, X)
            return np.corrcoef(R, rowvar=False)
        p = X.shape[1]
        out = np.eye(p)
        for i in range(p):
            for j in range(i + 1, p):
                out[i, j] = out[j, i] = _kendall_tau(X[:, i], X[:, j])
        return out

    x = _as_1d(x)
    y = _as_1d(y)
    n = len(x)
    if method == "pearson":
        r = float(np.corrcoef(x, y)[0, 1])
        df = n - 2
        if df <= 0 or abs(r) >= 1.0:
            pvalue = 0.0 if abs(r) >= 1.0 else 1.0
        else:
            t = r * np.sqrt(df / (1.0 - r * r))
            pvalue = 2.0 * float(_t_sf(abs(t), df))
        se = float(np.sqrt((1.0 - r * r) / df)) if df > 0 else float("nan")
        return EstimateResult(r, se, n, "pearson", {"pvalue": pvalue, "df": df})
    if method == "spearman":
        rx, ry = _rankdata(x), _rankdata(y)
        r = float(np.corrcoef(rx, ry)[0, 1])
        df = n - 2
        if df <= 0 or abs(r) >= 1.0:
            pvalue = 0.0 if abs(r) >= 1.0 else 1.0
        else:
            t = r * np.sqrt(df / (1.0 - r * r))
            pvalue = 2.0 * float(_t_sf(abs(t), df))
        se = float(np.sqrt((1.0 - r * r) / df)) if df > 0 else float("nan")
        return EstimateResult(r, se, n, "spearman", {"pvalue": pvalue, "df": df})

    tau, pvalue = _kendall_tau(x, y, return_pvalue=True)
    se = float(np.sqrt(2.0 * (2 * n + 5) / (9.0 * n * (n - 1)))) if n > 1 else float("nan")
    return EstimateResult(tau, se, n, "kendall", {"pvalue": pvalue})


def _rankdata(x):
    """Average ranks (1-based), ties resolved by averaging."""
    x = np.asarray(x, dtype=float)
    order = np.argsort(x, kind="mergesort")
    ranks = np.empty(len(x))
    sorted_x = x[order]
    i = 0
    n = len(x)
    while i < n:
        j = i
        while j + 1 < n and sorted_x[j + 1] == sorted_x[i]:
            j += 1
        avg_rank = 0.5 * (i + j) + 1.0
        ranks[order[i: j + 1]] = avg_rank
        i = j + 1
    return ranks


def _kendall_tau(x, y, return_pvalue=False):
    """Tau-b (tie-corrected) via the O(n log n) concordance count, with the standard
    large-sample normal-approximation p-value (matches
    ``scipy.stats.kendalltau(method='asymptotic')``)."""
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    n = len(x)
    order = np.argsort(x, kind="mergesort")
    x, y = x[order], y[order]

    def count_ties(a):
        _, counts = np.unique(a, return_counts=True)
        return counts

    tx = count_ties(x)
    ty = count_ties(y)
    n0 = n * (n - 1) / 2.0
    n1 = np.sum(tx * (tx - 1)) / 2.0
    n2 = np.sum(ty * (ty - 1)) / 2.0

    # concordant - discordant via merge-sort inversion counting on y (grouped by equal x
    # handled through explicit pairwise correction below for simplicity/n<=2000 use case).
    conc = disc = 0
    for i in range(n):
        dx = x[i + 1:] - x[i]
        dy = y[i + 1:] - y[i]
        sgn = np.sign(dx) * np.sign(dy)
        conc += int(np.sum(sgn > 0))
        disc += int(np.sum(sgn < 0))
    tau = (conc - disc) / np.sqrt((n0 - n1) * (n0 - n2)) if (n0 - n1) * (n0 - n2) > 0 else 0.0

    if not return_pvalue:
        return float(tau)

    v0 = n * (n - 1) * (2 * n + 5)
    vt = np.sum(tx * (tx - 1) * (2 * tx + 5))
    vu = np.sum(ty * (ty - 1) * (2 * ty + 5))
    v1 = np.sum(tx * (tx - 1)) * np.sum(ty * (ty - 1)) / (2.0 * n * (n - 1)) if n > 1 else 0.0
    v2 = (np.sum(tx * (tx - 1) * (tx - 2)) * np.sum(ty * (ty - 1) * (ty - 2))
          / (9.0 * n * (n - 1) * (n - 2))) if n > 2 else 0.0
    var_s = (v0 - vt - vu) / 18.0 + v1 + v2
    s = conc - disc
    if var_s <= 0:
        return float(tau), 1.0
    z = s / np.sqrt(var_s)
    pvalue = 2.0 * float(1.0 - _norm_cdf(abs(z)))
    return float(tau), pvalue


def describe(x):
    """Full descriptive summary: n, missing, mean, std, min, quartiles, max, skew,
    kurtosis. For a 2-D array, computed column-wise (missing values are NaNs, silently
    excluded from every other statistic via ``nan``-aware reductions)."""
    arr = np.asarray(x, dtype=float)
    axis = 0 if arr.ndim == 2 else None
    missing = np.sum(np.isnan(arr), axis=axis)
    valid_n = arr.shape[0] if axis is not None else arr.size

    def _clean(col):
        return col[~np.isnan(col)]

    if axis is None:
        clean = _clean(arr)
        n = len(clean)
        q1, med, q3 = np.quantile(clean, [0.25, 0.5, 0.75])
        return DescribeResult(
            n=n, missing=int(missing), mean=float(np.mean(clean)), std=float(np.std(clean, ddof=1)),
            min=float(np.min(clean)), q1=float(q1), median=float(med), q3=float(q3),
            max=float(np.max(clean)), skewness=skewness(clean), kurtosis=kurtosis(clean),
        )

    p = arr.shape[1]
    means = np.empty(p); stds = np.empty(p); mins = np.empty(p); q1s = np.empty(p)
    meds = np.empty(p); q3s = np.empty(p); maxs = np.empty(p); sks = np.empty(p); kus = np.empty(p)
    for j in range(p):
        c = _clean(arr[:, j])
        means[j] = np.mean(c); stds[j] = np.std(c, ddof=1)
        mins[j], maxs[j] = np.min(c), np.max(c)
        q1s[j], meds[j], q3s[j] = np.quantile(c, [0.25, 0.5, 0.75])
        sks[j] = skewness(c); kus[j] = kurtosis(c)
    return DescribeResult(
        n=valid_n, missing=missing.astype(int), mean=means, std=stds, min=mins, q1=q1s,
        median=meds, q3=q3s, max=maxs, skewness=sks, kurtosis=kus,
    )
