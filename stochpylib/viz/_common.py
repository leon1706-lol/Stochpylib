"""Private helpers shared across the ``viz`` submodules: input validation, RNG, the
distribution plotting grid, plotting positions, MCMC sample extraction, and regression
diagnostics (hat/Cook's/studentized residuals) shared by the diagnostics plots.

Native numpy/scipy.special only -- scipy.stats is the test suite's oracle, never a
runtime import here (AGENTS.md).
"""

import numpy as np

from stochpylib.viz._figure import Figure

__all__ = []  # private module


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


def _get_fig_ax(ax, nrows=1, ncols=1, **fig_kwargs):
    """Composition contract: draw into a passed ``Axes``, else make a fresh ``Figure``."""
    if ax is not None:
        return ax.figure, ax
    fig = Figure(nrows=nrows, ncols=ncols, **fig_kwargs)
    return fig, fig.ax


# --------------------------------------------------------------------- distributions


def _dist_grid(dist, n_points=200, lo_q=0.001, hi_q=0.999):
    """A sensible plotting grid for a distribution: ``ppf([lo_q, hi_q])`` clipped to
    the distribution's own support."""
    lo = float(np.asarray(dist.ppf(lo_q)))
    hi = float(np.asarray(dist.ppf(hi_q)))
    if hasattr(dist, "support"):
        s_lo, s_hi = dist.support()
        if np.isfinite(s_lo):
            lo = max(lo, float(s_lo))
        if np.isfinite(s_hi):
            hi = min(hi, float(s_hi))
    if not (np.isfinite(lo) and np.isfinite(hi)) or lo >= hi:
        raise ValueError("could not build a finite plotting grid for this distribution")
    return np.linspace(lo, hi, int(n_points))


def _dist_int_grid(dist, lo_q=0.001, hi_q=0.999):
    """Integer support grid for a discrete distribution's pmf/cdf."""
    lo = float(np.asarray(dist.ppf(lo_q)))
    hi = float(np.asarray(dist.ppf(hi_q)))
    if hasattr(dist, "support"):
        s_lo, s_hi = dist.support()
        if np.isfinite(s_lo):
            lo = max(lo, float(s_lo))
        if np.isfinite(s_hi):
            hi = min(hi, float(s_hi))
    lo, hi = int(np.floor(lo)), int(np.ceil(hi))
    if hi < lo:
        hi = lo
    return np.arange(lo, hi + 1)


def _filliben_pp(n):
    """Filliben (1975) plotting positions, matching ``scipy.stats.probplot``'s default."""
    v = np.empty(n)
    if n == 1:
        v[0] = 0.5
        return v
    v[-1] = 0.5 ** (1.0 / n)
    v[0] = 1.0 - v[-1]
    i = np.arange(2, n)
    v[1:-1] = (i - 0.3175) / (n + 0.365)
    return v


# --------------------------------------------------------------------- time series


def _durbin_levinson(acf):
    """Partial autocorrelations from an autocorrelation sequence ``acf`` (``acf[0]==1``).

    Returns ``pacf`` with ``pacf[0] = 1``, ``pacf[k]`` the k-th partial autocorrelation.
    """
    K = len(acf) - 1
    pacf = np.zeros(K + 1)
    pacf[0] = 1.0
    if K == 0:
        return pacf
    phi_prev = np.zeros(K + 1)
    phi_prev[1] = acf[1]
    pacf[1] = acf[1]
    for k in range(2, K + 1):
        num = acf[k] - np.sum(phi_prev[1:k] * acf[k - 1:0:-1])
        den = 1.0 - np.sum(phi_prev[1:k] * acf[1:k])
        phi_kk = num / den if den != 0 else 0.0
        phi_new = phi_prev.copy()
        phi_new[1:k] = phi_prev[1:k] - phi_kk * phi_prev[k - 1:0:-1]
        phi_new[k] = phi_kk
        phi_prev = phi_new
        pacf[k] = phi_kk
    return pacf


# --------------------------------------------------------------------- MCMC samples


def _extract_chains(obj):
    """Normalize a sampler or array to a ``(n_chains, n_samples, dim)`` chains array."""
    if hasattr(obj, "get_chains"):
        arr = np.asarray(obj.get_chains(), dtype=float)
    else:
        from stochpylib.advanced_mcmc.diagnostics import _chains_3d

        arr = _chains_3d(obj)
    if arr.ndim != 3 or arr.size == 0:
        raise ValueError("chains must be non-empty with shape (n_chains, n_samples, dim)")
    return arr


def _extract_samples(obj):
    """Normalize a sampler or a flat samples table to a ``(n_samples, dim)`` array.

    A plain ndarray is treated as an already-flattened samples table (one row per draw,
    one column per parameter) -- distinct from :func:`_extract_chains`, which treats a
    2-D array as ``(n_chains, n_samples)`` for a single scalar parameter.
    """
    if hasattr(obj, "get_samples"):
        arr = np.asarray(obj.get_samples(), dtype=float)
    elif hasattr(obj, "get_chains"):
        chains = np.asarray(obj.get_chains(), dtype=float)
        arr = chains.reshape(-1, chains.shape[-1])
    else:
        arr = np.asarray(obj, dtype=float)
        if arr.ndim == 1:
            arr = arr[:, None]
        elif arr.ndim == 3:
            arr = arr.reshape(-1, arr.shape[-1])
    if arr.ndim != 2 or arr.size == 0:
        raise ValueError("samples must be non-empty with shape (n_samples, n_params)")
    if not np.all(np.isfinite(arr)):
        raise ValueError("samples contains non-finite values")
    return arr


# --------------------------------------------------------------------- regression diagnostics


def _scatter_grid(arr, labels, diagonal="hist", title=None):
    """Shared ``dim``-by-``dim`` layout for :func:`pair_plot`/:func:`plot_scatter_matrix`:
    a density on the diagonal, scatter below it, and correlation text above it."""
    from stochpylib.viz._figure import Figure
    from stochpylib.nonparametric.density import KernelDensityEstimate

    n, dim = arr.shape
    corr = np.corrcoef(arr, rowvar=False)
    fig = Figure(nrows=dim, ncols=dim, width=180 * dim, height=180 * dim, title=title)
    for i in range(dim):
        for j in range(dim):
            panel = fig.axes[i * dim + j]
            if i == j:
                if diagonal == "kde":
                    kde = KernelDensityEstimate().fit(arr[:, i])
                    h = float(np.ravel(kde.bandwidth_)[0])
                    grid = np.linspace(arr[:, i].min() - 3 * h, arr[:, i].max() + 3 * h, 100)
                    panel.line(grid, np.asarray(kde.pdf(grid), dtype=float))
                else:
                    heights, edges = np.histogram(arr[:, i], bins="auto", density=True)
                    centers = 0.5 * (edges[:-1] + edges[1:])
                    panel.bar(centers, heights, width=np.diff(edges))
            elif i > j:
                panel.scatter(arr[:, j], arr[:, i], size=2.5, alpha=0.5)
            else:
                panel.text(0.5, 0.5, f"r={corr[i, j]:.2f}", anchor="middle")
                panel.grid = False
            if i == dim - 1:
                panel.xlabel = labels[j]
            if j == 0:
                panel.ylabel = labels[i]
    return fig, corr


def _leverage_stats(X, residuals, fit_intercept=True):
    """Hat diagonal, internally/externally studentized residuals and Cook's distance.

    Matches ``statsmodels.stats.outliers_influence.OLSInfluence``'s definitions.
    """
    X = _as_2d(X, "X")
    resid = np.asarray(residuals, dtype=float).ravel()
    if resid.shape[0] != X.shape[0]:
        raise ValueError("X and residuals must have the same number of rows")
    design = np.column_stack([np.ones(X.shape[0]), X]) if fit_intercept else X
    n, p = design.shape
    Q, _ = np.linalg.qr(design)
    hat = np.sum(Q ** 2, axis=1)
    hat = np.clip(hat, 0.0, 1.0 - 1e-12)
    rss = float(resid @ resid)
    dof = max(n - p, 1)
    mse = rss / dof
    resid_int = resid / np.sqrt(mse * (1.0 - hat))
    cooks = resid_int ** 2 * hat / ((1.0 - hat) * p)
    with np.errstate(invalid="ignore", divide="ignore"):
        resid_ext = resid_int * np.sqrt(np.clip((n - p - 1.0) / (n - p - resid_int ** 2),
                                                 0.0, None))
    return {"hat": hat, "resid_internal": resid_int, "resid_external": resid_ext,
            "cooks": cooks, "n": n, "p": p}
