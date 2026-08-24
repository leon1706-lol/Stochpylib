"""Shared plumbing for the copulas module: pseudo-observations, validation,
dependence-measure estimators, and the native Student-t quantile function."""

import numpy as np
from scipy import optimize, special

__all__ = [
    "pseudo_obs", "as_u_matrix", "kendall_tau_estimate", "spearman_rho_estimate",
    "student_t_ppf",
]


def pseudo_obs(data):
    """Rank-based pseudo-observations ``U_ij = R_ij / (n + 1)`` in (0, 1)^d.

    The ``(n + 1)`` scaling keeps every point strictly inside the open unit cube so
    generator/log evaluations never hit singular boundary values.
    """
    arr = np.asarray(data, dtype=float)
    if arr.ndim != 2 or arr.shape[0] < 2:
        raise ValueError("data must be a (n, d) array with n >= 2")
    if not np.all(np.isfinite(arr)):
        raise ValueError("data contains non-finite values")
    n = arr.shape[0]
    return np.column_stack([
        _rank_ties(arr[:, j]) / (n + 1.0) for j in range(arr.shape[1])
    ])


def _rank_ties(col):
    """Average ranks (1-based), ties get their mean — matches Kendall's tau usage."""
    col = np.asarray(col, dtype=float)
    order = np.argsort(col, kind="mergesort")
    ranks = np.empty(len(col))
    sorted_col = col[order]
    i = 0
    while i < len(col):
        j = i
        while j + 1 < len(col) and sorted_col[j + 1] == sorted_col[i]:
            j += 1
        avg = 0.5 * (i + j) + 1.0          # mean of positions i+1 .. j+1
        ranks[order[i:j + 1]] = avg
        i = j + 1
    return ranks


def as_u_matrix(u, name="u"):
    """Coerce to a finite (n, d) matrix and clip into the closed unit cube."""
    arr = np.asarray(u, dtype=float)
    if arr.ndim == 1:
        arr = arr[None, :]
    if arr.ndim != 2 or arr.shape[0] == 0:
        raise ValueError(f"{name} must be a non-empty (n, d) array")
    if not np.all(np.isfinite(arr)):
        raise ValueError(f"{name} contains non-finite values")
    if np.any(arr < -1e-12) or np.any(arr > 1 + 1e-12):
        raise ValueError(f"{name} must lie in the unit cube [0, 1]^d")
    return np.clip(arr, 0.0, 1.0)


def kendall_tau_estimate(x, y):
    """Kendall's tau-b between two equally long samples (ties handled).

    O(n log n): sort by ``x``, count discordant pairs as strict inversions of
    the ``y``-ranks through a Fenwick tree, apply the standard tie corrections::

        tau_b = (P - Q) / sqrt((N - Tx) (N - Ty)),
        P - Q = N - Tx - Ty + Txy - 2 Q.
    """
    x = np.asarray(x, dtype=float).ravel()
    y = np.asarray(y, dtype=float).ravel()
    if len(x) != len(y) or len(x) < 2:
        raise ValueError("x and y must be equal-length arrays with n >= 2")
    n = len(x)
    rx = _rank_ties(x)
    ry = _rank_ties(y)

    N = n * (n - 1) / 2.0

    def _group_ties(ranks):
        """Sum of C(m, 2) over equal-value groups."""
        _, counts = np.unique(ranks, return_counts=True)
        return float(np.sum(counts * (counts - 1) / 2.0))

    tx = _group_ties(rx)
    ty = _group_ties(ry)

    # pairs tied in BOTH coordinates
    txy = 0.0
    order = np.argsort(rx, kind="mergesort")
    s = ry[order]
    boundaries = np.flatnonzero(np.diff(rx[order]) != 0) + 1
    start = 0
    for b in list(boundaries) + [n]:
        grp = s[start:b]
        if len(grp) > 1:
            _, cts = np.unique(grp, return_counts=True)
            txy += float(np.sum(cts * (cts - 1) / 2.0))
        start = b

    # strict inversions of s via Fenwick tree over compressed values;
    # pairs inside an equal-x group contribute neither P nor Q, so each whole
    # group queries against previous groups before any of its members inserts
    keys = np.unique(s)
    comp = np.searchsorted(keys, s) + 1               # 1..K
    size = len(keys)
    tree = [0] * (size + 1)

    def update(v):
        i = v
        while i <= size:
            tree[i] += 1
            i += i & (-i)

    def prefix(i):
        out = 0
        while i > 0:
            out += tree[i]
            i -= i & (-i)
        return out

    q = 0
    inserted = 0
    start = 0
    for b in list(boundaries) + [n]:
        grp = comp[start:b]
        for v in grp:
            q += inserted - prefix(v)                 # strictly greater
        for v in grp:
            update(v)
        inserted += len(grp)
        start = b

    pq = N - tx - ty + txy - 2.0 * q
    denom = np.sqrt(N - tx) * np.sqrt(N - ty)
    return float(pq / denom)


def spearman_rho_estimate(x, y):
    """Spearman's rho via average ranks and Pearson correlation of those ranks."""
    rx = _rank_ties(np.asarray(x, dtype=float).ravel())
    ry = _rank_ties(np.asarray(y, dtype=float).ravel())
    rc = np.corrcoef(rx, ry)[0, 1]
    return float(rc)


def student_t_ppf(q, df):
    """Student-t quantile function built on the regularized incomplete beta inverse.

    Uses ``I_x(a,b)`` from :func:`scipy.special.betaincinv` with the identity
    ``P(T <= t) = 0.5 + 0.5 sign(t) I_{df/(df+t^2)}(df/2, 1/2)`` — no
    ``scipy.stats`` involved.
    """
    q = np.asarray(q, dtype=float)
    if np.any((q <= 0.0) | (q >= 1.0)):
        raise ValueError("quantiles must lie strictly inside (0, 1)")
    df = float(df)
    half = special.betaincinv(0.5 * df, 0.5, 2.0 * np.minimum(q, 1.0 - q))
    x2 = df * (1.0 - half) / half
    out = np.sqrt(x2)
    return np.where(q > 0.5, out, -out)


def brentq_on_bracket(fn, lo, hi, **kwargs):
    """brentq wrapper that widens a failing bracket a little before giving up."""
    try:
        return optimize.brentq(fn, lo, hi, **kwargs)
    except ValueError:
        for grow in (4.0, 16.0, 256.0):
            try:
                return optimize.brentq(fn, lo / grow, hi * grow, **kwargs)
            except ValueError:
                continue
        raise ValueError("could not bracket the root") from None
