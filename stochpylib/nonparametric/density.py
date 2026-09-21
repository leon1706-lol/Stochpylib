"""Kernel and series density estimators.

Every class is a full :class:`~stochpylib.nonparametric._base.NonparametricDensity`
(and therefore a full :class:`~stochpylib.distributions._base.Distribution`):
``pdf/cdf/ppf/rvs/mean/var/skewness/kurtosis/entropy/mgf/cf/fit/ks_test`` all
work, with the generic numerical fallbacks in ``Distribution`` covering
anything not given a closed form here.
"""

import numpy as np
from scipy import special

from stochpylib.nonparametric._base import NonparametricDensity
from stochpylib.nonparametric._common import (
    _kernel_fn, _norm_cdf, _select_bandwidth,
)

__all__ = ["KernelDensityEstimate", "NearestNeighborDensity", "AdaptiveKDE",
           "OrthogonalSeriesDensity", "LogsplineEstimator"]


class KernelDensityEstimate(NonparametricDensity):
    """Univariate or multivariate kernel density estimate.

    1-D data gets exact closed-form ``cdf`` (integrated kernel), ``mean``/``var``
    (sample mean / sample variance + h^2 * mu2(K) for the smoothing inflation),
    and a proper ``rvs`` (resample + kernel noise). d > 1 data uses a diagonal
    product-kernel ``pdf``/``rvs``; ``cdf`` and moments fall back to the generic
    Monte Carlo / grid machinery in ``Distribution``.
    """

    def __init__(self, kernel="gaussian", bandwidth="silverman"):
        self.kernel = kernel
        self.bandwidth = bandwidth

    def _fit(self, x):
        h = _select_bandwidth(x, self.bandwidth, self.kernel)
        self.bandwidth_ = h
        self.is_discrete = False
        return {"h": h}

    def _fit_2d(self, X):
        # used only when constructed directly on (n, d) data via fit()
        pass

    def fit(self, x):
        arr = np.asarray(x, dtype=float)
        if arr.ndim == 2 and arr.shape[1] > 1:
            self.data_ = arr
            self.n_ = arr.shape[0]
            d = arr.shape[1]
            h = np.array([_select_bandwidth(arr[:, j], self.bandwidth, self.kernel)
                          for j in range(d)])
            self.bandwidth_ = h
            self.is_discrete = False
            self.params_ = {"h": h}
            self._d = d
            return self
        self._d = 1
        return super().fit(arr.ravel())

    def pdf(self, x):
        K = _kernel_fn(self.kernel)
        if getattr(self, "_d", 1) == 1:
            x = np.asarray(x, dtype=float)
            scalar = x.ndim == 0
            x = np.atleast_1d(x)
            h = self.bandwidth_
            u = (x[:, None] - self.data_[None, :]) / h
            out = np.mean(K(u), axis=1) / h
            return float(out[0]) if scalar else out
        X = np.atleast_2d(np.asarray(x, dtype=float))
        h = self.bandwidth_
        n, d = self.data_.shape
        m = X.shape[0]
        out = np.empty(m)
        for i in range(m):
            u = (X[i][None, :] - self.data_) / h[None, :]
            dens = np.prod(K(u), axis=1) / np.prod(h)
            out[i] = np.mean(dens)
        return out[0] if np.asarray(x).ndim == 1 and out.size == 1 else out

    def cdf(self, x):
        if getattr(self, "_d", 1) != 1:
            return super().cdf(x)
        x = np.asarray(x, dtype=float)
        scalar = x.ndim == 0
        x = np.atleast_1d(x)
        h = self.bandwidth_
        u = (x[:, None] - self.data_[None, :]) / h
        if self.kernel == "gaussian":
            out = np.mean(_norm_cdf(u), axis=1)
        elif self.kernel == "uniform":
            out = np.mean(np.clip((u + 1.0) / 2.0, 0.0, 1.0), axis=1)
        elif self.kernel == "epanechnikov":
            uc = np.clip(u, -1.0, 1.0)
            out = np.mean(0.5 + 0.75 * uc - 0.25 * uc ** 3, axis=1)
        elif self.kernel == "triangular":
            uc = np.clip(u, -1.0, 1.0)
            pos = uc >= 0
            F = np.where(pos, 0.5 + uc - 0.5 * uc ** 2, 0.5 * (1 + uc) ** 2)
            out = np.mean(F, axis=1)
        else:
            # numerically integrate the kernel CDF via fine quadrature (biweight/cosine)
            K = _kernel_fn(self.kernel)
            grid = np.linspace(-1.0, 1.0, 2001)
            cum = np.concatenate([[0.0], np.cumsum((K(grid[1:]) + K(grid[:-1])) / 2.0
                                                     * np.diff(grid))])
            uc = np.clip(u, -1.0, 1.0)
            F = np.interp(uc, grid, cum)
            out = np.mean(F, axis=1)
        return float(out[0]) if scalar else out

    def mean(self):
        if getattr(self, "_d", 1) == 1:
            return float(np.mean(self.data_))
        return np.mean(self.data_, axis=0)

    def var(self):
        from stochpylib.nonparametric._common import _KERNEL_MU2
        if getattr(self, "_d", 1) == 1:
            h = self.bandwidth_
            mu2 = _KERNEL_MU2[self.kernel]
            return float(np.var(self.data_, ddof=1) + h ** 2 * mu2)
        return np.cov(self.data_, rowvar=False)

    def rvs(self, size=1, random_state=None):
        rng = np.random.default_rng(random_state)
        idx = rng.integers(0, self.n_, size=size)
        if getattr(self, "_d", 1) == 1:
            base = self.data_[idx]
            h = self.bandwidth_
            if self.kernel == "gaussian":
                noise = rng.standard_normal(size) * h
            elif self.kernel in ("epanechnikov", "biweight", "triangular", "uniform", "cosine"):
                # rejection sample from the kernel shape on [-1, 1], scaled by h
                K = _kernel_fn(self.kernel)
                peak = K(np.array([0.0]))[0] if self.kernel != "triangular" else 1.0
                noise = np.empty(size)
                filled = 0
                while filled < size:
                    cand = rng.uniform(-1.0, 1.0, size * 2)
                    acc = rng.uniform(0.0, peak, size * 2) < K(cand)
                    take = cand[acc]
                    k = min(len(take), size - filled)
                    noise[filled:filled + k] = take[:k]
                    filled += k
                noise = noise * h
            else:
                noise = rng.standard_normal(size) * h
            return base + noise
        base = self.data_[idx]
        h = self.bandwidth_
        noise = rng.standard_normal((size, self._d)) * h[None, :]
        return base + noise


class AdaptiveKDE(NonparametricDensity):
    """Abramson (1982) / Silverman two-stage adaptive-bandwidth KDE.

    Stage 1: a pilot fixed-bandwidth KDE gives a density estimate at each
    point. Stage 2: local bandwidths ``h_i = h * (f_pilot(x_i) / g)^(-alpha)``
    (``g`` the geometric mean of the pilot density) widen the kernel in low
    density regions. ``alpha=0`` reduces exactly to :class:`KernelDensityEstimate`.
    """

    def __init__(self, alpha=0.5, kernel="gaussian", pilot_bandwidth="silverman"):
        self.alpha = alpha
        self.kernel = kernel
        self.pilot_bandwidth = pilot_bandwidth

    def _fit(self, x):
        self.is_discrete = False
        pilot = KernelDensityEstimate(self.kernel, self.pilot_bandwidth).fit(x)
        h0 = pilot.bandwidth_
        f_pilot = np.clip(pilot.pdf(x), 1e-300, None)
        log_g = np.mean(np.log(f_pilot))
        g = np.exp(log_g)
        local_h = h0 * (f_pilot / g) ** (-self.alpha)
        self.bandwidth_ = h0
        self.local_bandwidth_ = local_h
        return {"h0": h0, "local_h": local_h}

    def pdf(self, x):
        K = _kernel_fn(self.kernel)
        x = np.asarray(x, dtype=float)
        scalar = x.ndim == 0
        x = np.atleast_1d(x)
        lh = self.local_bandwidth_
        u = (x[:, None] - self.data_[None, :]) / lh[None, :]
        out = np.mean(K(u) / lh[None, :], axis=1)
        return float(out[0]) if scalar else out

    def rvs(self, size=1, random_state=None):
        rng = np.random.default_rng(random_state)
        idx = rng.integers(0, self.n_, size=size)
        base = self.data_[idx]
        h = self.local_bandwidth_[idx]
        noise = rng.standard_normal(size) * h if self.kernel == "gaussian" else \
            rng.standard_normal(size) * h
        return base + noise


class NearestNeighborDensity(NonparametricDensity):
    """k-th nearest-neighbor density estimate: ``f(x) = k / (n * 2 * R_k(x))``
    in 1-D (the ball volume ``V_1 = 2 * R``).

    The raw kNN estimate does not integrate to 1 over an unbounded support, so
    ``cdf``/moments/``entropy`` operate on a truncated-and-renormalized version
    (support padded 3 bandwidths beyond the data range) -- see the module
    README's Known limitations.
    """

    def __init__(self, k=5):
        self.k = int(k)

    def _fit(self, x):
        from stochpylib.nonparametric._common import _cdf_grid_from_pdf
        self.is_discrete = False
        if self.k >= len(x):
            raise ValueError("k must be smaller than the sample size")
        lo, hi = self.support()
        grid = np.linspace(lo, hi, 4000)
        raw_vals = self._raw_pdf(grid)
        self._norm_const = float(np.trapezoid(raw_vals, grid))
        self._grid = grid
        self._grid_vals = raw_vals / self._norm_const
        self._cdf_grid = _cdf_grid_from_pdf(grid, self._grid_vals)
        return {}

    def _raw_pdf(self, x):
        x = np.atleast_1d(np.asarray(x, dtype=float))
        dist = np.abs(x[:, None] - self.data_[None, :])
        r_k = np.partition(dist, self.k - 1, axis=1)[:, self.k - 1]
        r_k = np.maximum(r_k, 1e-12)
        return self.k / (self.n_ * 2.0 * r_k)

    def pdf(self, x):
        scalar = np.asarray(x).ndim == 0
        raw = self._raw_pdf(x)
        out = raw / self._norm_const
        return float(out[0]) if scalar else out

    def cdf(self, x):
        x = np.asarray(x, dtype=float)
        scalar = x.ndim == 0
        x = np.atleast_1d(x)
        out = np.interp(x, self._grid, self._cdf_grid, left=0.0, right=1.0)
        return float(out[0]) if scalar else out


class OrthogonalSeriesDensity(NonparametricDensity):
    """Orthogonal-series density estimate on a rescaled-to-[0,1] cosine basis
    (Chentsov / Efromovich): ``f(x) = 1 + sum_j 2 a_j cos(j pi t)`` with
    ``t = (x - lo)/(hi - lo)``, ``a_j`` the sample mean of ``cos(j pi t)``.
    Kronmal-Tarter term selection (``n_terms="auto"``) truncates once a term's
    ``|a_j|`` falls below its Monte Carlo noise level. Clipped to
    non-negative and renormalized.
    """

    def __init__(self, basis="cosine", n_terms=10, bounds=None):
        self.basis = basis
        self.n_terms = n_terms
        self.bounds = bounds

    def _fit(self, x):
        self.is_discrete = False
        if self.basis != "cosine":
            raise ValueError("only basis='cosine' is implemented")
        lo, hi = self.bounds if self.bounds is not None else (
            float(np.min(x)) - 1e-6, float(np.max(x)) + 1e-6)
        self._lo, self._hi = lo, hi
        t = (x - lo) / (hi - lo)
        max_j = self.n_terms if isinstance(self.n_terms, int) else 40
        coefs = []
        n = len(x)
        for j in range(1, max_j + 1):
            a_j = float(np.mean(np.sqrt(2.0) * np.cos(j * np.pi * t)))
            coefs.append(a_j)
            if self.n_terms == "auto" and j >= 3:
                var_j = float(np.var(np.sqrt(2.0) * np.cos(j * np.pi * t), ddof=1) / n)
                if a_j ** 2 < 2.0 * var_j:
                    break
        self.coefs_ = np.array(coefs)
        # precompute a fine grid for the clip-and-renormalize step
        grid = np.linspace(lo, hi, 4000)
        tg = (grid - lo) / (hi - lo)
        vals = np.ones_like(tg)
        for j, a_j in enumerate(self.coefs_, start=1):
            vals += a_j * np.sqrt(2.0) * np.cos(j * np.pi * tg)
        vals = np.clip(vals, 0.0, None) / (hi - lo)
        norm = float(np.trapezoid(vals, grid))
        self._grid, self._grid_vals = grid, vals / norm
        from stochpylib.nonparametric._common import _cdf_grid_from_pdf
        self._cdf_grid = _cdf_grid_from_pdf(self._grid, self._grid_vals)
        return {"lo": lo, "hi": hi}

    def support(self):
        return (self._lo, self._hi)

    def pdf(self, x):
        x = np.asarray(x, dtype=float)
        scalar = x.ndim == 0
        x = np.atleast_1d(x)
        out = np.interp(x, self._grid, self._grid_vals, left=0.0, right=0.0)
        return float(out[0]) if scalar else out

    def cdf(self, x):
        x = np.asarray(x, dtype=float)
        scalar = x.ndim == 0
        x = np.atleast_1d(x)
        out = np.interp(x, self._grid, self._cdf_grid, left=0.0, right=1.0)
        return float(out[0]) if scalar else out


class LogsplineEstimator(NonparametricDensity):
    """Kooperberg-Stone log-spline density estimate: ``log f(x) = B(x).theta - c(theta)``
    for a cubic B-spline basis ``B``, fit by maximum likelihood (Newton's
    method) with the normalizing constant ``c(theta)`` computed by fixed-grid
    quadrature. No knot-deletion search (a fixed, evenly-spaced knot count) --
    see the module README's Known limitations.
    """

    def __init__(self, n_knots=8, degree=3, bounds=None):
        self.n_knots = n_knots
        self.degree = degree
        self.bounds = bounds

    def _fit(self, x):
        from stochpylib.nonparametric._common import _bspline_design, _bspline_knots
        self.is_discrete = False
        lo, hi = self.bounds if self.bounds is not None else (
            float(np.min(x)) - 0.5 * np.std(x), float(np.max(x)) + 0.5 * np.std(x))
        self._lo, self._hi = lo, hi
        knots = _bspline_knots(np.array([lo, hi]), self.n_knots, self.degree)
        self._knots = knots
        B_data = _bspline_design(x, knots, self.degree)
        p = B_data.shape[1]
        grid = np.linspace(lo, hi, 800)
        B_grid = _bspline_design(grid, knots, self.degree)
        dgrid = grid[1] - grid[0]

        theta = np.zeros(p)
        for _ in range(100):
            eta_grid = B_grid @ theta
            m = np.max(eta_grid)
            w = np.exp(eta_grid - m)
            Z = np.trapezoid(w, grid)
            log_c = m + np.log(Z)
            # gradient: mean(B(x_i)) - E_theta[B(X)]
            p_grid = w / Z
            E_B = np.trapezoid(B_grid * p_grid[:, None], grid, axis=0)
            grad = np.mean(B_data, axis=0) - E_B
            # Hessian: -Cov_theta[B(X)]
            EBB = np.trapezoid(
                B_grid[:, :, None] * B_grid[:, None, :] * p_grid[:, None, None],
                grid, axis=0)
            cov = EBB - np.outer(E_B, E_B)
            try:
                step = np.linalg.solve(cov + 1e-8 * np.eye(p), grad)
            except np.linalg.LinAlgError:
                step = grad
            theta = theta + step
            if np.max(np.abs(step)) < 1e-8:
                break
        self.theta_ = theta
        eta_grid = B_grid @ theta
        m = np.max(eta_grid)
        Z = np.trapezoid(np.exp(eta_grid - m), grid)
        self.log_c_ = m + np.log(Z)
        self._grid = grid
        self._grid_vals = np.exp(eta_grid - self.log_c_)
        from stochpylib.nonparametric._common import _cdf_grid_from_pdf
        self._cdf_grid = _cdf_grid_from_pdf(self._grid, self._grid_vals)
        return {"theta": theta}

    def support(self):
        return (self._lo, self._hi)

    def pdf(self, x):
        from stochpylib.nonparametric._common import _bspline_design
        x = np.asarray(x, dtype=float)
        scalar = x.ndim == 0
        x = np.atleast_1d(x)
        inside = (x >= self._lo) & (x <= self._hi)
        out = np.zeros_like(x)
        if np.any(inside):
            B = _bspline_design(x[inside], self._knots, self.degree)
            out[inside] = np.exp(B @ self.theta_ - self.log_c_)
        return float(out[0]) if scalar else out

    def cdf(self, x):
        x = np.asarray(x, dtype=float)
        scalar = x.ndim == 0
        x = np.atleast_1d(x)
        out = np.interp(x, self._grid, self._cdf_grid, left=0.0, right=1.0)
        return float(out[0]) if scalar else out
