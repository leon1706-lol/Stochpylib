"""Time-series decomposition: classical, STL-style, X11-style, and trend filters.

All decompositions return a :class:`DecompositionResult` with ``trend``, ``seasonal``
and ``resid`` arrays such that (additive case) ``x = trend + seasonal + resid``.

Documented simplifications: STLDecomposition is a native Cleveland-style LOESS
implementation without robustness re-weighting; X11Decomposition is a simplified
ratio-to-moving-average iteration, not Census X-11.
"""

from dataclasses import dataclass

import numpy as np

from stochpylib.timeseries._utils import as_1d


@dataclass
class DecompositionResult:
    trend: np.ndarray
    seasonal: np.ndarray
    resid: np.ndarray
    model: str = "additive"

    def __repr__(self):
        return (
            f"DecompositionResult(n={len(self.trend)}, model={self.model!r}, "
            f"seasonal_range=[{self.seasonal.min():.4g}, {self.seasonal.max():.4g}])"
        )


# ---------------------------------------------------------------------------
# smoothing primitives


def _centered_ma(x, window):
    """Centered moving average of exact length ``len(x)``.

    Odd windows are straightforward; even windows use the classic 2xW average so the
    result is symmetric around each point.
    """
    window = int(window)
    if window < 1:
        raise ValueError("window must be >= 1")
    kernel = np.ones(window) / window
    half_left = window // 2
    half_right = window - 1 - half_left
    padded = np.concatenate([np.full(half_left, x[0]), x, np.full(half_right, x[-1])])
    out = np.convolve(padded, kernel, mode="valid")
    if window % 2 == 0 and len(out) == len(x) + 1:
        # even window: average consecutive alignments -> symmetric end weights
        out = 0.5 * (out[:-1] + out[1:])
    return np.asarray(out[: len(x)], dtype=float)


def _loess_smooth(y, fraction=0.5):
    """Local-linear (LOESS-style) smoothing at every point with tricube weights."""
    y = np.asarray(y, dtype=float)
    n = len(y)
    span = max(int(np.ceil(fraction * n)), 3)
    half = span // 2
    out = np.empty(n)
    for i in range(n):
        lo = max(0, i - half)
        hi = min(n, i + half + 1)
        xs = np.arange(lo, hi, dtype=float)
        ys = y[lo:hi]
        d = np.abs(xs - i)
        maxd = max(d.max(), 1e-12)
        w = np.where(d < maxd, (1.0 - (d / maxd) ** 3) ** 3, 0.0)
        if w.sum() <= 0:
            out[i] = ys.mean()
            continue
        xm, ym = np.average(xs, weights=w), np.average(ys, weights=w)
        var_x = np.average((xs - xm) ** 2, weights=w)
        cov_xy = np.average((xs - xm) * (ys - ym), weights=w)
        slope = cov_xy / var_x if var_x > 1e-14 else 0.0
        out[i] = ym + slope * (i - xm)
    return out


# ---------------------------------------------------------------------------
# classical


def _seasonal_profile(detrended, period, center="mean"):
    """Per-position profile from the detrended series, centered to zero mean."""
    profile = np.zeros(period)
    counts = np.zeros(period)
    for i, v in enumerate(detrended):
        pos = i % period
        if np.isfinite(v):
            profile[pos] += v
            counts[pos] += 1
    counts[counts == 0] = 1
    profile /= counts
    return profile


def SeasonalDecomposition(x, period, model="additive"):
    """Classical moving-average decomposition into trend / seasonal / remainder."""
    x = as_1d(x)
    period = int(period)
    if period < 2:
        raise ValueError("period must be >= 2")
    if model not in ("additive", "multiplicative"):
        raise ValueError("model must be 'additive' or 'multiplicative'")
    if model == "multiplicative" and np.any(x <= 0):
        raise ValueError("multiplicative decomposition requires strictly positive data")

    trend = _centered_ma(x, period)
    detrended = x / trend if model == "multiplicative" else x - trend

    profile = _seasonal_profile(detrended, period)
    if model == "additive":
        profile -= profile.mean()
    else:
        profile /= profile.mean()

    seasonal = np.tile(profile, len(x) // period + 1)[: len(x)]
    if model == "multiplicative":
        resid = x / (trend * seasonal)
    else:
        resid = x - trend - seasonal
    return DecompositionResult(trend=trend, seasonal=seasonal, resid=resid, model=model)


# ---------------------------------------------------------------------------
# STL (Cleveland-style, simplified)


def STLDecomposition(x, period, inner_iterations=3, loess_fraction=0.45):
    """Cleveland-style STL: subseries-LOESS seasonal extraction with a low-pass pass.

    Simplified relative to the reference implementation (no robustness re-weighting);
    the structure follows the original: cycle-subseries smoothing -> low-pass filter of
    the raw seasonal -> deseasonalized trend smoothing, repeated ``inner_iterations``
    times.
    """
    x = as_1d(x)
    period = int(period)
    if period < 2 or len(x) < 2 * period:
        raise ValueError("period must be >= 2 and the series needs two full periods")
    T = len(x)

    trend = _centered_ma(x, period)
    seasonal = np.zeros(T)

    for _ in range(int(inner_iterations)):
        # 1. subseries smoothing of the detrended series per position
        detrended = x - trend
        raw_seasonal = np.zeros(T)
        for pos in range(period):
            sub_idx = np.arange(pos, T, period)
            smoothed_sub = _loess_smooth(detrended[sub_idx], loess_fraction)
            raw_seasonal[sub_idx] = smoothed_sub

        # 2. low-pass filter the raw seasonal (MA chain then short LOESS)
        lowpass = _centered_ma(raw_seasonal, period)
        lowpass = _centered_ma(lowpass, period)
        lowpass = _loess_smooth(lowpass, min(1.0, 3.0 * period / T))

        # 3. seasonal component and refreshed trend
        seasonal = raw_seasonal - lowpass
        for pos in range(period):  # re-center each position's seasonal level
            idx = np.arange(pos, T, period)
            seasonal[idx] -= seasonal[idx].mean()
        trend = _loess_smooth(x - seasonal, loess_fraction)

    resid = x - trend - seasonal
    return DecompositionResult(trend=trend, seasonal=seasonal, resid=resid)


# ---------------------------------------------------------------------------
# X11-style


def X11Decomposition(x, period, iterations=2):
    """Simplified ratio-to-moving-average X11-style seasonal adjustment (documented).

    Multiplicative by construction (like classical X-11 on positive series); iterates
    the ratio-to-moving-average step a few times before a final pass.
    """
    x = as_1d(x)
    if np.any(x <= 0):
        raise ValueError("X11-style decomposition requires strictly positive data")
    iterations = max(int(iterations), 1)

    work = x.copy()
    seasonal_final = None
    for _ in range(iterations):
        res = SeasonalDecomposition(work, period, model="multiplicative")
        work = work / res.seasonal
        seasonal_final = res.seasonal

    trend = _centered_ma(work, period)
    resid = x / (trend * seasonal_final)
    return DecompositionResult(trend=trend, seasonal=seasonal_final, resid=resid,
                               model="multiplicative")


# --------------------------------------------------------------------------- filters


def TrendFilter(x, lam=10.0, order=2):
    """Whittaker-Henderson smoother: argmin ||y-tau||^2 + lam * ||D^order tau||^2."""
    from scipy import sparse
    from scipy.sparse.linalg import spsolve

    x = as_1d(x)
    T = len(x)
    D = sparse.eye(T, format="csc")
    for _ in range(int(order)):
        rows = D.shape[0] - 1
        if rows < 1:
            raise ValueError("order too large for this sample size")
        Dm = sparse.diags([-np.ones(rows), np.ones(rows)], [0, 1],
                          shape=(rows, rows + 1), format="csc")
        D = Dm @ D
    A = sparse.eye(T, format="csc") + float(lam) * (D.T @ D)
    tau = spsolve(A.tocsc(), x)
    tau = np.asarray(tau, dtype=float)
    return DecompositionResult(trend=tau, seasonal=np.zeros(T),
                               resid=x - tau)


def HPFilter(x, lamb=1600.0):
    """Hodrick-Prescott filter (TrendFilter with the conventional quarterly lambda)."""
    return TrendFilter(x, lam=float(lamb), order=2)


__all__ = [
    "SeasonalDecomposition",
    "STLDecomposition",
    "X11Decomposition",
    "TrendFilter",
    "HPFilter",
    "DecompositionResult",
]
