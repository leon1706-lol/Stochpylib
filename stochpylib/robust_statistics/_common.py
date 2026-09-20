"""Private helpers shared across the robust_statistics submodules.

Not part of the public API. Distributional building blocks come from
``scipy.special`` directly; the generic array/design helpers are reused from
``stochpylib.statistics._common`` rather than duplicated -- library code never
wraps ``scipy.stats`` (that module is the test suite's independent oracle only,
per AGENTS.md).
"""

import itertools

import numpy as np
from scipy import special

from stochpylib.statistics._common import (  # noqa: F401  (re-exported for siblings)
    _as_1d, _as_2d, _chi2_cdf, _chi2_ppf, _cholesky_psd, _design, _norm_cdf,
    _norm_pdf, _norm_ppf, _rng, _t_ppf,
)

_MAD_CONST = 1.0 / special.ndtri(0.75)          # ~= 1.4826022185056018
_IQR_CONST = 2.0 * special.ndtri(0.75)          # ~= 1.3489795003921635
_QN_CONST = 1.0 / (np.sqrt(2.0) * special.ndtri(5.0 / 8.0))   # ~= 2.2219
_SN_CONST = 1.1926                              # Rousseeuw & Croux (1993)


def _mad(x, center=None, normal=True):
    """Median absolute deviation. ``center=None`` uses the sample median;
    ``center=0.0`` is the regression convention (residuals already centered)."""
    x = np.asarray(x, dtype=float)
    c = np.median(x) if center is None else float(center)
    m = float(np.median(np.abs(x - c)))
    return m * _MAD_CONST if normal else m


def _lomed(a):
    """Low median: for even n, the lower of the two middle order statistics."""
    a = np.sort(np.asarray(a, dtype=float))
    n = len(a)
    return float(a[(n - 1) // 2])


def _himed(a):
    """High median: for even n, the upper of the two middle order statistics."""
    a = np.sort(np.asarray(a, dtype=float))
    n = len(a)
    return float(a[n // 2])


def _kth_pairwise(x, k, kind="diff", max_explicit=1000):
    """k-th smallest (1-based) of the implicit pairwise multiset.

    ``kind="diff"``: {x_j - x_i : i < j} (n(n-1)/2 values, used by Qn/Sn).
    ``kind="walsh"``: {(x_i + x_j)/2 : i <= j} (n(n+1)/2 values, Hodges-Lehmann).
    Exact for n <= max_explicit (materialize the multiset); a monotone
    bisection on the pair value above that (O(n log n) per probe via a
    sorted two-pointer count).
    """
    x = np.sort(np.asarray(x, dtype=float))
    n = len(x)
    if n <= max_explicit:
        if kind == "diff":
            iu = np.triu_indices(n, k=1)
            vals = x[iu[1]] - x[iu[0]]
        elif kind == "walsh":
            iu = np.triu_indices(n, k=0)
            vals = (x[iu[0]] + x[iu[1]]) / 2.0
        else:
            raise ValueError("kind must be 'diff' or 'walsh'")
        return float(np.partition(vals, k - 1)[k - 1])

    if kind == "diff":
        lo, hi = 0.0, float(x[-1] - x[0])

        def count_le(t):
            # #{(i<j): x_j - x_i <= t}, two-pointer over sorted x
            j = 0
            c = 0
            for i in range(n):
                j = max(j, i + 1)
                while j < n and x[j] - x[i] <= t:
                    j += 1
                c += j - i - 1
            return c
    elif kind == "walsh":
        lo, hi = float(x[0]), float(x[-1])

        def count_le(t):
            # #{(i<=j): (x_i+x_j)/2 <= t} = #{(i<=j): x_i+x_j <= 2t}
            two_t = 2.0 * t
            j = n - 1
            c = 0
            for i in range(n):
                if j < i:
                    j = i
                while j >= i and x[i] + x[j] > two_t:
                    j -= 1
                c += max(j - i + 1, 0)
            return c
    else:
        raise ValueError("kind must be 'diff' or 'walsh'")

    for _ in range(200):
        mid = 0.5 * (lo + hi)
        if count_le(mid) >= k:
            hi = mid
        else:
            lo = mid
        if hi - lo <= 1e-13 * max(1.0, abs(hi)):
            break
    return hi


# --------------------------------------------------------------- M-estimation

def _huber_psi(u, c=1.345):
    return np.clip(u, -c, c)


def _huber_psi_prime(u, c=1.345):
    return (np.abs(u) <= c).astype(float)


def _huber_weight(u, c=1.345):
    u = np.asarray(u, dtype=float)
    au = np.abs(u)
    w = np.ones_like(u)
    mask = au > c
    with np.errstate(divide="ignore", invalid="ignore"):
        w[mask] = c / au[mask]
    return w


def _huber_rho(u, c=1.345):
    u = np.asarray(u, dtype=float)
    au = np.abs(u)
    return np.where(au <= c, 0.5 * u * u, c * au - 0.5 * c * c)


def _tukey_weight(u, c=4.685061):
    u = np.asarray(u, dtype=float)
    t = np.clip(1.0 - (u / c) ** 2, 0.0, None)
    return np.where(np.abs(u) <= c, t * t, 0.0)


def _tukey_psi(u, c=4.685061):
    return u * _tukey_weight(u, c)


def _tukey_psi_prime(u, c=4.685061):
    u = np.asarray(u, dtype=float)
    z = (u / c) ** 2
    val = (1.0 - z) * (1.0 - 5.0 * z)
    return np.where(np.abs(u) <= c, val, 0.0)


def _tukey_rho(u, c=4.685061):
    u = np.asarray(u, dtype=float)
    z = np.clip(1.0 - (u / c) ** 2, 0.0, None)
    return np.where(np.abs(u) <= c, (c * c / 6.0) * (1.0 - z ** 3), c * c / 6.0)


def _hampel_psi(u, a=2.0, b=4.0, c=8.0):
    u = np.asarray(u, dtype=float)
    au = np.abs(u)
    s = np.sign(u)
    out = np.where(
        au <= a, u,
        np.where(au <= b, a * s,
                 np.where(au <= c, a * s * (c - au) / (c - b), 0.0)))
    return out


def _hampel_weight(u, a=2.0, b=4.0, c=8.0):
    u = np.asarray(u, dtype=float)
    au = np.abs(u)
    psi = _hampel_psi(u, a, b, c)
    w = np.ones_like(u)
    nz = au > 1e-12
    w[nz] = psi[nz] / u[nz]
    w[~nz] = 1.0
    return w


def _andrews_psi(u, c=1.339):
    u = np.asarray(u, dtype=float)
    return np.where(np.abs(u) <= c * np.pi, c * np.sin(u / c), 0.0)


def _andrews_weight(u, c=1.339):
    u = np.asarray(u, dtype=float)
    psi = _andrews_psi(u, c)
    w = np.ones_like(u)
    nz = np.abs(u) > 1e-12
    w[nz] = psi[nz] / u[nz]
    return w


def _cauchy_psi(u, c=2.385):
    u = np.asarray(u, dtype=float)
    return u / (1.0 + (u / c) ** 2)


def _cauchy_weight(u, c=2.385):
    u = np.asarray(u, dtype=float)
    return 1.0 / (1.0 + (u / c) ** 2)


def _welsch_psi(u, c=2.985):
    u = np.asarray(u, dtype=float)
    return u * np.exp(-((u / c) ** 2))


def _welsch_weight(u, c=2.985):
    u = np.asarray(u, dtype=float)
    return np.exp(-((u / c) ** 2))


def _fair_psi(u, c=1.4):
    u = np.asarray(u, dtype=float)
    return u / (1.0 + np.abs(u) / c)


def _fair_weight(u, c=1.4):
    u = np.asarray(u, dtype=float)
    return 1.0 / (1.0 + np.abs(u) / c)


def _logistic_psi(u, c=1.205):
    u = np.asarray(u, dtype=float)
    return c * np.tanh(u / c)


def _logistic_weight(u, c=1.205):
    u = np.asarray(u, dtype=float)
    w = np.ones_like(u)
    nz = np.abs(u) > 1e-12
    w[nz] = c * np.tanh(u[nz] / c) / u[nz]
    return w


def _numeric_psi_prime(psi_fn, u, c, eps=1e-6):
    u = np.asarray(u, dtype=float)
    h = eps * np.maximum(1.0, np.abs(u))
    return (psi_fn(u + h, c) - psi_fn(u - h, c)) / (2.0 * h)


_PSI = {
    "huber": dict(rho=_huber_rho, psi=_huber_psi, weight=_huber_weight,
                  psi_prime=_huber_psi_prime, c=1.345),
    "tukey": dict(rho=_tukey_rho, psi=_tukey_psi, weight=_tukey_weight,
                  psi_prime=_tukey_psi_prime, c=4.685061),
    "biweight": dict(rho=_tukey_rho, psi=_tukey_psi, weight=_tukey_weight,
                     psi_prime=_tukey_psi_prime, c=4.685061),
    "hampel": dict(rho=None, psi=_hampel_psi, weight=_hampel_weight,
                   psi_prime=None, c=(2.0, 4.0, 8.0)),
    "andrews": dict(rho=None, psi=_andrews_psi, weight=_andrews_weight,
                    psi_prime=None, c=1.339),
    "cauchy": dict(rho=None, psi=_cauchy_psi, weight=_cauchy_weight,
                   psi_prime=None, c=2.385),
    "welsch": dict(rho=None, psi=_welsch_psi, weight=_welsch_weight,
                   psi_prime=None, c=2.985),
    "fair": dict(rho=None, psi=_fair_psi, weight=_fair_weight,
                 psi_prime=None, c=1.4),
    "logistic": dict(rho=None, psi=_logistic_psi, weight=_logistic_weight,
                     psi_prime=None, c=1.205),
}


def _psi_prime_of(name, u, c):
    spec = _PSI[name]
    if spec["psi_prime"] is not None:
        if name == "hampel":
            return spec["psi_prime"](u, *c)
        return spec["psi_prime"](u, c)
    psi_fn = spec["psi"]
    if name == "hampel":
        return _numeric_psi_prime(lambda uu, cc: psi_fn(uu, *cc), u, c)
    return _numeric_psi_prime(psi_fn, u, c)


def _psi_weight(name, u, c=None):
    spec = _PSI[name]
    c = spec["c"] if c is None else c
    if name == "hampel":
        return spec["weight"](u, *c) if isinstance(c, tuple) else spec["weight"](u, *spec["c"])
    return spec["weight"](u, c)


def _psi_value(name, u, c=None):
    spec = _PSI[name]
    c = spec["c"] if c is None else c
    if name == "hampel":
        return spec["psi"](u, *c) if isinstance(c, tuple) else spec["psi"](u, *spec["c"])
    return spec["psi"](u, c)


# --------------------------------------------------------------- S/M-scale

def _tukey_rho_norm(u, c=1.547645):
    """Tukey biweight rho normalized so rho(inf) = 1 -- used for the S-scale."""
    u = np.asarray(u, dtype=float)
    z = np.clip(1.0 - (u / c) ** 2, 0.0, None)
    return np.where(np.abs(u) <= c, 1.0 - z ** 3, 1.0)


def _m_scale(r, c=1.547645, b=0.5, s0=None, tol=1e-9, max_iter=200):
    """S-scale: solves mean(rho_norm(r/s)) = b by fixed point."""
    r = np.asarray(r, dtype=float)
    s = _mad(r, center=0.0) if s0 is None else float(s0)
    if s <= 0.0:
        s = float(np.std(r)) if np.std(r) > 0 else 1.0
    if s <= 0.0:
        return 0.0
    for _ in range(max_iter):
        m = float(np.mean(_tukey_rho_norm(r / s, c)))
        if m <= 0.0:
            break
        s_new = s * np.sqrt(m / b)
        if abs(s_new - s) <= tol * max(1.0, abs(s)):
            s = s_new
            break
        s = s_new
    return float(s)


def _huber_proposal2(x, c=1.5, tol=1.0e-8, max_iter=30):
    """Huber (1981) proposal 2: joint M-estimate of location and scale,
    transcribed from ``statsmodels.robust.scale.Huber`` to match it exactly."""
    x = np.asarray(x, dtype=float)
    n = x.shape[0] - 1
    gamma_c = 2.0 * float(_norm_cdf(c)) - 1.0
    gamma = gamma_c + c ** 2 * (1.0 - gamma_c) - 2.0 * c * float(_norm_pdf(c))

    mu = float(np.median(x))
    scale = _mad(x, normal=True)
    for _ in range(max_iter):
        nmu = float(np.clip(x, mu - c * scale, mu + c * scale).sum() / x.shape[0])
        subset = np.abs((x - mu) / scale) <= c
        card = int(subset.sum())
        scale_num = float(np.sum(subset * (x - nmu) ** 2))
        scale_denom = n * gamma - (x.shape[0] - card) * c ** 2
        nscale = np.sqrt(scale_num / scale_denom) if scale_denom > 0 else scale
        test1 = abs(scale - nscale) <= nscale * tol
        test2 = abs(mu - nmu) <= nscale * tol
        if test1 and test2:
            return float(nmu), float(nscale)
        mu, scale = nmu, nscale
    return float(mu), float(scale)


def _biweight_midvariance(x, c=9.0):
    """Biweight midvariance (Mosteller & Tukey), a scale M-estimator."""
    x = np.asarray(x, dtype=float)
    n = len(x)
    med = float(np.median(x))
    mad_raw = _mad(x, center=med, normal=False)
    if mad_raw == 0.0:
        return 0.0
    u = (x - med) / (c * mad_raw)
    a = (np.abs(u) < 1.0).astype(float)
    num = n * np.sum(a * (x - med) ** 2 * (1.0 - u ** 2) ** 4)
    den = np.sum(a * (1.0 - u ** 2) * (1.0 - 5.0 * u ** 2))
    if den == 0.0:
        return 0.0
    return float(np.sqrt(num) / abs(den))


def _tau_scale(x, c1=4.5, c2=3.0):
    """Yohai-Zamar tau location/scale (robustbase's ``scaleTau2``)."""
    x = np.asarray(x, dtype=float)
    med = float(np.median(x))
    s0 = _mad(x, center=med, normal=True)
    if s0 == 0.0:
        return med, 0.0
    u = (x - med) / (c1 * s0)
    w = (1.0 - u ** 2) ** 2
    w = np.where(np.abs(u) <= 1.0, w, 0.0)
    sw = np.sum(w)
    mu = float(np.sum(w * x) / sw) if sw > 0 else med
    z = (x - mu) / s0
    rho_c2 = np.minimum(z ** 2, c2 ** 2)
    ez = _tau_consistency(c2)
    scale = s0 * np.sqrt(np.mean(rho_c2) / ez)
    return mu, float(scale)


def _tau_consistency(c2):
    """E[min(Z^2, c2^2)] under Z ~ N(0,1), closed form via integration by parts:
    2*int_0^c z^2 phi(z) dz + c^2*P(|Z|>=c) = (2*Phi(c)-1) - 2*c*phi(c) + 2*c^2*(1-Phi(c))."""
    phi_c = float(_norm_pdf(c2))
    Phi_c = float(_norm_cdf(c2))
    return (2.0 * Phi_c - 1.0) - 2.0 * c2 * phi_c + 2.0 * c2 ** 2 * (1.0 - Phi_c)


# --------------------------------------------------------------- misc

def _chi2_consistency(alpha, p):
    """MCD/MVE raw-scatter consistency factor: alpha / chi2_cdf(chi2_ppf(alpha,p), p+2)."""
    q = _chi2_ppf(alpha, p)
    denom = _chi2_cdf(q, p + 2)
    return alpha / denom if denom > 0 else 1.0


def _mahalanobis(X, loc, cov):
    X = np.asarray(X, dtype=float)
    loc = np.asarray(loc, dtype=float)
    L = _cholesky_psd(np.asarray(cov, dtype=float))
    diff = (X - loc).T
    y = np.linalg.solve(L, diff)
    return np.sum(y * y, axis=0)


def _subsets(n, k, max_count, rng):
    """All C(n,k) index tuples when small enough, else max_count random draws.

    Returns (array (m, k) int, exhaustive: bool).
    """
    total = special.comb(n, k, exact=True) if n >= k >= 0 else 0
    if 0 < total <= max_count:
        idx = np.array(list(itertools.combinations(range(n), k)), dtype=int)
        return idx, True
    m = max_count
    out = np.empty((m, k), dtype=int)
    for i in range(m):
        out[i] = np.sort(rng.choice(n, size=k, replace=False))
    return out, False


def _weiszfeld(P, tol=1e-8, max_iter=300):
    """Spatial (geometric) median of the rows of P via Weiszfeld's algorithm."""
    P = np.asarray(P, dtype=float)
    x = np.median(P, axis=0)
    for _ in range(max_iter):
        d = np.linalg.norm(P - x, axis=1)
        zero = d < 1e-12
        if np.any(zero):
            return P[zero][0]
        w = 1.0 / d
        x_new = np.sum(P * w[:, None], axis=0) / np.sum(w)
        if np.linalg.norm(x_new - x) <= tol * max(1.0, np.linalg.norm(x)):
            x = x_new
            break
        x = x_new
    return x


def _ols(Xd, y, w=None):
    """Plain (optionally weighted) least squares -> coef array."""
    if w is None:
        beta, *_ = np.linalg.lstsq(Xd, y, rcond=None)
        return beta
    sw = np.sqrt(np.clip(w, 0.0, None))
    beta, *_ = np.linalg.lstsq(Xd * sw[:, None], y * sw, rcond=None)
    return beta


def _kth_cross_diff(x, y, k, max_explicit=2_000_000):
    """k-th smallest (1-based) of {x_i - y_j : all i, j} (size m*n).

    Used by the two-sample Hodges-Lehmann/R-estimator confidence intervals.
    Exact for m*n <= max_explicit; a monotone bisection via vectorized
    ``searchsorted`` above that.
    """
    x = np.sort(np.asarray(x, dtype=float))
    y = np.sort(np.asarray(y, dtype=float))
    m, n = len(x), len(y)
    if m * n <= max_explicit:
        vals = (x[:, None] - y[None, :]).ravel()
        return float(np.partition(vals, k - 1)[k - 1])

    lo, hi = float(x[0] - y[-1]), float(x[-1] - y[0])

    def count_le(t):
        # #{(i,j): x_i - y_j <= t} = sum_i #{j: y_j >= x_i - t}
        thresh = x - t
        return int(np.sum(n - np.searchsorted(y, thresh, side="left")))

    for _ in range(200):
        mid = 0.5 * (lo + hi)
        if count_le(mid) >= k:
            hi = mid
        else:
            lo = mid
        if hi - lo <= 1e-13 * max(1.0, abs(hi)):
            break
    return hi


def _rankdata_ties(x):
    """Per-element tie-group size (length n), used by the Sen (1968) CI."""
    x = np.asarray(x, dtype=float)
    _, inverse, counts = np.unique(x, return_inverse=True, return_counts=True)
    return counts[inverse].astype(float)
