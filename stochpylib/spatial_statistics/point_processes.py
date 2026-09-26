"""Spatial point processes and their second-order summary functions.

``RipleyK``/``PairCorrelation`` are plain functions (per the spec's ``()`` convention)
returning a ``SpatialFunction``. Edge correction weights (translation, and a
quadrature-evaluated isotropic/Ripley correction for 2-D rectangles) are shared between the
two. ``SpatialPointProcess`` subclasses fit by minimum contrast against an empirical
``RipleyK`` (Diggle's ``K**(1/4)`` criterion), except the two Poisson processes (closed-form
MLE).
"""

import numpy as np
from scipy import integrate, optimize

from stochpylib.spatial_statistics._common import _ball_volume, _box_window, _in_window, _rng
from stochpylib.spatial_statistics._result import SpatialFunction
from stochpylib.spatial_statistics.variogram import SpatialCovariance, Variogram

__all__ = [
    "SpatialPointProcess", "PoissonPointProcess", "InhomogeneousPoisson", "ThomasProcess",
    "MaternCluster", "LogGaussianCox", "RipleyK", "PairCorrelation",
]


def _as_pts(X):
    arr = np.asarray(X, dtype=float)
    if arr.ndim == 1:
        arr = arr[:, None]
    return arr


def _pw(A, B=None):
    B = A if B is None else B
    return np.sqrt(np.maximum(((A[:, None, :] - B[None, :, :]) ** 2).sum(-1), 0.0))


# --------------------------------------------------------------- edge corrections (shared)

def _ripley_frac_inside(X, D, W, n_theta=180):
    """Fraction of each pairwise-distance circle (centered at i, radius d_ij) inside W.

    Evaluated by quadrature over the circle (n_theta angular samples) rather than the
    closed-form corner geometry -- exact in the n_theta -> infinity limit, accurate to
    ~1/n_theta in practice, and far simpler to get right.
    """
    xlo, xhi = W[0]
    ylo, yhi = W[1]
    theta = np.linspace(0.0, 2.0 * np.pi, n_theta, endpoint=False)
    cos_t, sin_t = np.cos(theta), np.sin(theta)
    n = len(X)
    frac = np.ones((n, n))
    for i in range(n):
        d_row = D[i]
        finite = np.isfinite(d_row) & (d_row > 0)
        if not np.any(finite):
            continue
        d_safe = np.where(finite, d_row, 0.0)  # avoid inf * 0 -> nan on the discarded rows
        px, py = X[i]
        xs = px + d_safe[:, None] * cos_t[None, :]
        ys = py + d_safe[:, None] * sin_t[None, :]
        inside = (xs >= xlo) & (xs <= xhi) & (ys >= ylo) & (ys <= yhi)
        frac[i] = np.where(finite, inside.mean(axis=1), 1.0)
    return frac


def _edge_weights(X, D, W, correction):
    if correction == "none":
        return np.ones_like(D)
    if correction == "translation":
        L = W[:, 1] - W[:, 0]
        diff = np.abs(X[:, None, :] - X[None, :, :])
        frac = np.prod(np.clip(L[None, None, :] - diff, 1e-300, None) / L[None, None, :], axis=2)
        return np.where((frac > 0) & np.isfinite(frac), 1.0 / frac, 0.0)
    if correction == "ripley":
        if W.shape[0] != 2:
            raise ValueError("correction='ripley' supports 2-D rectangular windows only")
        frac = _ripley_frac_inside(X, D, W)
        return 1.0 / np.clip(frac, 1e-6, 1.0)
    raise ValueError("correction must be 'none', 'translation', 'ripley' or 'border'")


def _ripley_k_estimate(X, W, r, correction, lam):
    n = len(X)
    if n < 2:
        return np.zeros_like(r)
    D = _pw(X)
    np.fill_diagonal(D, np.inf)
    if correction == "border":
        out = []
        for ri in r:
            inside = np.all((X - W[:, 0] >= ri) & (W[:, 1] - X >= ri), axis=1)
            ni = int(inside.sum())
            if ni == 0:
                out.append(np.nan)
                continue
            counts = np.sum(D[inside] <= ri, axis=1)
            out.append(float(np.mean(counts)) / lam)
        return np.array(out)
    w = _edge_weights(X, D, W, correction)
    return np.array([np.sum(w[D <= ri]) for ri in r]) / (n * lam)


def _epanechnikov(x, b):
    u = np.asarray(x, dtype=float) / b
    return np.where(np.abs(u) < 1.0, 0.75 * (1.0 - u ** 2) / b, 0.0)


def _pcf_estimate(X, W, r, b, correction, lam):
    n = len(X)
    if n < 2:
        return np.zeros_like(r)
    D = _pw(X)
    np.fill_diagonal(D, np.inf)
    w = _edge_weights(X, D, W, "translation" if correction == "border" else correction)
    finite = np.isfinite(D)
    out = np.empty(len(r))
    for i, ri in enumerate(r):
        if ri <= 0:
            out[i] = np.nan
            continue
        k = np.where(finite, _epanechnikov(ri - D, b), 0.0)
        out[i] = np.sum(w * k) / (2.0 * np.pi * ri * lam * n)
    return out


def RipleyK(points, window, r=None, correction="ripley", n_simulations=0, random_state=None):
    """Ripley's K function, edge-corrected, with an optional Monte Carlo CSR envelope."""
    X = _as_pts(points)
    W, vol = _box_window(window)
    d = W.shape[0]
    n = len(X)
    if r is None:
        rmax = 0.25 * float(np.min(W[:, 1] - W[:, 0]))
        r = np.linspace(rmax / 20.0, rmax, 15)
    r = np.asarray(r, dtype=float)
    lam = n / vol
    estimate = _ripley_k_estimate(X, W, r, correction, lam)
    theoretical = _ball_volume(d, r)
    lower = upper = pvalue = None
    extras = {"dim": d, "n": n, "lambda_hat": lam}
    if n_simulations > 0:
        rng = _rng(random_state)
        sims = np.empty((int(n_simulations), len(r)))
        for s in range(int(n_simulations)):
            pts = rng.uniform(W[:, 0], W[:, 1], size=(n, d))
            sims[s] = _ripley_k_estimate(pts, W, r, correction, lam)
        lower, upper = np.nanmin(sims, axis=0), np.nanmax(sims, axis=0)
        dev = np.nanmax(np.abs(sims - theoretical[None, :]), axis=1)
        obs_dev = np.nanmax(np.abs(estimate - theoretical))
        pvalue = float((1 + np.sum(dev >= obs_dev)) / (1 + n_simulations))
    return SpatialFunction(r, estimate, theoretical, name="K", correction=correction,
                            lower=lower, upper=upper, pvalue=pvalue, extras=extras)


def PairCorrelation(points, window, r=None, bandwidth=None, correction="translation",
                     n_simulations=0, random_state=None):
    """Kernel-smoothed pair correlation function g(r), Epanechnikov kernel."""
    X = _as_pts(points)
    W, vol = _box_window(window)
    d = W.shape[0]
    n = len(X)
    lam = n / vol
    if r is None:
        rmax = 0.25 * float(np.min(W[:, 1] - W[:, 0]))
        r = np.linspace(rmax / 20.0, rmax, 15)
    r = np.asarray(r, dtype=float)
    if bandwidth is None:
        bandwidth = 0.15 / np.sqrt(max(lam, 1e-12))
    estimate = _pcf_estimate(X, W, r, bandwidth, correction, lam)
    theoretical = np.ones_like(r)
    lower = upper = pvalue = None
    extras = {"dim": d, "n": n, "lambda_hat": lam, "bandwidth": bandwidth}
    if n_simulations > 0:
        rng = _rng(random_state)
        sims = np.empty((int(n_simulations), len(r)))
        for s in range(int(n_simulations)):
            pts = rng.uniform(W[:, 0], W[:, 1], size=(n, d))
            sims[s] = _pcf_estimate(pts, W, r, bandwidth, correction, lam)
        lower, upper = np.nanmin(sims, axis=0), np.nanmax(sims, axis=0)
        dev = np.nanmax(np.abs(sims - 1.0), axis=1)
        obs_dev = np.nanmax(np.abs(estimate - 1.0))
        pvalue = float((1 + np.sum(dev >= obs_dev)) / (1 + n_simulations))
    return SpatialFunction(r, estimate, theoretical, name="pcf", correction=correction,
                            lower=lower, upper=upper, pvalue=pvalue, extras=extras)


# ------------------------------------------------------------------------- process classes

class SpatialPointProcess:
    """Base for point process models: ``sample``, ``intensity``, ``K``/``pcf``, ``fit``."""

    def __init__(self, window=((0.0, 1.0), (0.0, 1.0))):
        self.window = window

    def _window_arr(self):
        return _box_window(self.window)

    def sample(self, random_state=None):
        raise NotImplementedError

    def intensity(self, x):
        raise NotImplementedError

    def expected_count(self):
        W, vol = self._window_arr()
        pts = np.mean(W, axis=1)[None, :]
        return float(self.intensity(pts)[0]) * vol

    def K(self, r):
        raise NotImplementedError

    def pcf(self, r):
        raise NotImplementedError

    def fit(self, points, window=None):
        if window is not None:
            self.window = window
        self.points_ = _as_pts(points)
        W, vol = self._window_arr()
        self.n_ = len(self.points_)
        self.intensity_hat_ = self.n_ / vol
        self._fit(self.points_, W, vol)
        return self

    def _fit(self, points, W, vol):
        raise NotImplementedError

    def _minimum_contrast(self, points, W, x0, build):
        """Diggle's K**(1/4) minimum-contrast fit against an empirical RipleyK.

        ``build`` takes only the *clustering-shape* parameters (e.g. kappa, sigma) --
        never mu/the child count -- because K(r) for a Poisson-cluster process does not
        depend on it (mu only scales the first-order intensity); the caller derives
        mu_hat = (n / |W|) / kappa_hat separately from the fitted kappa.
        """
        rmax = 0.25 * float(np.min(W[:, 1] - W[:, 0]))
        # geometric spacing resolves small-r clustering structure much better than linear
        r = np.geomspace(rmax / 50.0, rmax, 10)
        emp = RipleyK(points, self.window, r=r, correction="translation")

        def obj(theta):
            model = build(np.exp(theta))
            pred = model.K(r)
            return float(np.sum((np.asarray(pred) ** 0.25 - emp.estimate ** 0.25) ** 2))

        log_x0 = np.log(np.maximum(np.asarray(x0, dtype=float), 1e-6))
        # multi-start (deterministic seed): the K**(1/4) surface has local optima away from
        # a poor initial guess, and each start stays within a wide-but-finite band of its own
        # starting point, so a flat/near-unidentifiable direction can't run to a degenerate extreme
        rng = np.random.default_rng(0)
        starts = [log_x0] + [log_x0 + rng.normal(scale=1.5, size=log_x0.shape) for _ in range(3)]
        best = None
        for s in starts:
            bounds = [(v - 6.0, v + 6.0) for v in s]
            res = optimize.minimize(obj, s, method="Nelder-Mead", bounds=bounds,
                                     options={"maxiter": 600, "xatol": 1e-7, "fatol": 1e-10})
            if best is None or res.fun < best.fun:
                best = res
        return np.exp(best.x)

    def __repr__(self):
        return f"{type(self).__name__}(window={self.window!r})"


class PoissonPointProcess(SpatialPointProcess):
    """Homogeneous Poisson (complete spatial randomness): N ~ Poisson(lambda * |W|)."""

    def __init__(self, intensity=1.0, window=((0.0, 1.0), (0.0, 1.0))):
        super().__init__(window)
        self.lam = float(intensity)

    def intensity(self, x):
        return np.full(len(_as_pts(x)), self.lam)

    def expected_count(self):
        return self.lam * self._window_arr()[1]

    def K(self, r):
        d = self._window_arr()[0].shape[0]
        return _ball_volume(d, np.asarray(r, dtype=float))

    def pcf(self, r):
        return np.ones_like(np.asarray(r, dtype=float))

    def sample(self, random_state=None):
        rng = _rng(random_state)
        W, vol = self._window_arr()
        n = rng.poisson(self.lam * vol)
        return rng.uniform(W[:, 0], W[:, 1], size=(n, W.shape[0]))

    def _fit(self, points, W, vol):
        n = len(points)
        self.lam = n / vol
        self.intensity_ = self.lam
        self.std_error_ = np.sqrt(n) / vol


class InhomogeneousPoisson(SpatialPointProcess):
    """Poisson process with a spatially varying intensity (independent scattering)."""

    def __init__(self, intensity, window=((0.0, 1.0), (0.0, 1.0)), lambda_max=None):
        super().__init__(window)
        self.intensity_fn = intensity
        self.lambda_max = lambda_max

    def intensity(self, x):
        return np.asarray(self.intensity_fn(_as_pts(x)), dtype=float)

    def _grid(self, W, n_per_axis=40):
        axes = [np.linspace(lo, hi, n_per_axis) for lo, hi in W]
        mesh = np.meshgrid(*axes, indexing="ij")
        return axes, mesh, np.column_stack([m.ravel() for m in mesh])

    def _lambda_max(self, W):
        if self.lambda_max is not None:
            return float(self.lambda_max)
        _, _, pts = self._grid(W, 30)
        return float(self.intensity(pts).max()) * 1.1

    def sample(self, random_state=None):
        rng = _rng(random_state)
        W, vol = self._window_arr()
        lam_max = self._lambda_max(W)
        n = rng.poisson(lam_max * vol)
        cand = rng.uniform(W[:, 0], W[:, 1], size=(n, W.shape[0]))
        keep = rng.uniform(size=n) < self.intensity(cand) / lam_max
        return cand[keep]

    def expected_count(self):
        W, vol = self._window_arr()
        axes, mesh, pts = self._grid(W, 80)
        vals = self.intensity(pts).reshape(mesh[0].shape)
        result = vals
        for axis in reversed(range(len(axes))):
            result = integrate.simpson(result, axes[axis], axis=axis)
        return float(result)

    def K(self, r):
        raise NotImplementedError("K has no closed form for a general intensity function")

    def pcf(self, r):
        return np.ones_like(np.asarray(r, dtype=float))

    def _design(self, X, degree):
        cols = [np.ones(len(X))]
        for k in range(1, degree + 1):
            cols += [X[:, j] ** k for j in range(X.shape[1])]
        return np.column_stack(cols)

    def fit(self, points, window=None, degree=1):
        if window is not None:
            self.window = window
        self.points_ = _as_pts(points)
        W, vol = self._window_arr()
        self.n_ = len(self.points_)
        self.degree = int(degree)
        F = self._design(self.points_, self.degree)

        def negloglik(beta):
            axes, mesh, pts = self._grid(W, 40)
            lam = np.exp(self._design(pts, self.degree) @ beta).reshape(mesh[0].shape)
            integral = lam
            for axis in reversed(range(len(axes))):
                integral = integrate.simpson(integral, axes[axis], axis=axis)
            return -((F @ beta).sum() - integral)

        beta0 = np.zeros(F.shape[1])
        beta0[0] = np.log(max(self.n_ / vol, 1e-6))
        res = optimize.minimize(negloglik, beta0, method="L-BFGS-B")
        self.beta_ = res.x
        self.intensity_fn = lambda X: np.exp(self._design(_as_pts(X), self.degree) @ self.beta_)
        return self


class ThomasProcess(SpatialPointProcess):
    """Thomas cluster process: Poisson(kappa) parents, Poisson(mu) Gaussian-offset children."""

    def __init__(self, kappa=1.0, mu=5.0, sigma=0.1, window=((0.0, 1.0), (0.0, 1.0))):
        super().__init__(window)
        self.kappa, self.mu, self.sigma = float(kappa), float(mu), float(sigma)

    def intensity(self, x):
        return np.full(len(_as_pts(x)), self.kappa * self.mu)

    def expected_count(self):
        return self.kappa * self.mu * self._window_arr()[1]

    def K(self, r):
        r = np.asarray(r, dtype=float)
        return np.pi * r ** 2 + (1.0 - np.exp(-r ** 2 / (4.0 * self.sigma ** 2))) / self.kappa

    def pcf(self, r):
        r = np.asarray(r, dtype=float)
        return 1.0 + np.exp(-r ** 2 / (4.0 * self.sigma ** 2)) / (4.0 * np.pi * self.kappa * self.sigma ** 2)

    def sample(self, random_state=None):
        rng = _rng(random_state)
        W, vol = self._window_arr()
        d = W.shape[0]
        pad = 4.0 * self.sigma
        Wp = W.copy()
        Wp[:, 0] -= pad
        Wp[:, 1] += pad
        vol_p = float(np.prod(Wp[:, 1] - Wp[:, 0]))
        parents = rng.uniform(Wp[:, 0], Wp[:, 1], size=(rng.poisson(self.kappa * vol_p), d))
        chunks = []
        for p in parents:
            nc = rng.poisson(self.mu)
            if nc:
                chunks.append(p + rng.normal(scale=self.sigma, size=(nc, d)))
        pts = np.vstack(chunks) if chunks else np.empty((0, d))
        return pts[_in_window(pts, W)]

    def _fit(self, points, W, vol):
        kappa, sigma = self._minimum_contrast(
            points, W, [self.kappa, self.sigma],
            lambda p: ThomasProcess(kappa=p[0], mu=1.0, sigma=p[1], window=self.window))
        kappa, sigma = float(kappa), float(sigma)
        self.kappa, self.sigma = kappa, sigma
        self.mu = (len(points) / vol) / kappa if kappa > 0 else self.mu


class MaternCluster(SpatialPointProcess):
    """Matern cluster process: Poisson(kappa) parents, Poisson(mu) children uniform in a disc."""

    def __init__(self, kappa=1.0, mu=5.0, radius=0.1, window=((0.0, 1.0), (0.0, 1.0))):
        super().__init__(window)
        self.kappa, self.mu, self.radius = float(kappa), float(mu), float(radius)

    def intensity(self, x):
        return np.full(len(_as_pts(x)), self.kappa * self.mu)

    def expected_count(self):
        return self.kappa * self.mu * self._window_arr()[1]

    def pcf(self, r):
        r = np.asarray(r, dtype=float)
        R = self.radius
        out = np.ones_like(r)
        mask = r < 2.0 * R
        rr = r[mask] / R
        h = (2.0 / np.pi) * (np.arccos(rr / 2.0) - (rr / 2.0) * np.sqrt(np.clip(1.0 - (rr / 2.0) ** 2, 0.0, None)))
        out[mask] = 1.0 + h / (np.pi * R ** 2 * self.kappa)
        return out

    def K(self, r):
        r = np.atleast_1d(np.asarray(r, dtype=float))
        out = np.empty_like(r)
        for i, ri in enumerate(r):
            if ri <= 0:
                out[i] = 0.0
                continue
            rr = np.linspace(1e-9, ri, 60)
            out[i] = 2.0 * np.pi * integrate.simpson(rr * self.pcf(rr), rr)
        return out if out.size > 1 else float(out[0])

    def sample(self, random_state=None):
        rng = _rng(random_state)
        W, vol = self._window_arr()
        if W.shape[0] != 2:
            raise ValueError("MaternCluster.sample supports 2-D windows only")
        pad = self.radius
        Wp = W.copy()
        Wp[:, 0] -= pad
        Wp[:, 1] += pad
        vol_p = float(np.prod(Wp[:, 1] - Wp[:, 0]))
        parents = rng.uniform(Wp[:, 0], Wp[:, 1], size=(rng.poisson(self.kappa * vol_p), 2))
        chunks = []
        for p in parents:
            nc = rng.poisson(self.mu)
            if nc:
                ang = rng.uniform(0.0, 2.0 * np.pi, size=nc)
                rad = self.radius * np.sqrt(rng.uniform(0.0, 1.0, size=nc))
                chunks.append(p + np.column_stack([rad * np.cos(ang), rad * np.sin(ang)]))
        pts = np.vstack(chunks) if chunks else np.empty((0, 2))
        return pts[_in_window(pts, W)]

    def _fit(self, points, W, vol):
        kappa, radius = self._minimum_contrast(
            points, W, [self.kappa, self.radius],
            lambda p: MaternCluster(kappa=p[0], mu=1.0, radius=p[1], window=self.window))
        kappa, radius = float(kappa), float(radius)
        self.kappa, self.radius = kappa, radius
        self.mu = (len(points) / vol) / kappa if kappa > 0 else self.mu


class LogGaussianCox(SpatialPointProcess):
    """Log-Gaussian Cox process: intensity = exp(GRF), sampled on a grid."""

    def __init__(self, mean=0.0, covariance=None, window=((0.0, 1.0), (0.0, 1.0)),
                 grid_shape=(64, 64)):
        super().__init__(window)
        self.mean = float(mean)
        self.covariance = covariance
        self.grid_shape = tuple(int(s) for s in grid_shape)

    def _sill(self):
        cov = self.covariance
        if isinstance(cov, SpatialCovariance):
            return cov.variogram.sill
        if isinstance(cov, Variogram):
            return cov.sill
        return float(np.atleast_1d(cov(np.array([0.0])))[0])

    def _cov_call(self, h):
        cov = self.covariance
        if isinstance(cov, Variogram):
            cov = SpatialCovariance(cov)
        return cov(h)

    def intensity(self, x):
        return np.full(len(_as_pts(x)), np.exp(self.mean + 0.5 * self._sill()))

    def expected_count(self):
        return float(self.intensity(np.zeros((1, self._window_arr()[0].shape[0])))[0]) * self._window_arr()[1]

    def pcf(self, r):
        return np.exp(self._cov_call(np.asarray(r, dtype=float)))

    def K(self, r):
        r = np.atleast_1d(np.asarray(r, dtype=float))
        out = np.empty_like(r)
        for i, ri in enumerate(r):
            if ri <= 0:
                out[i] = 0.0
                continue
            rr = np.linspace(1e-9, ri, 60)
            out[i] = 2.0 * np.pi * integrate.simpson(rr * self.pcf(rr), rr)
        return out if out.size > 1 else float(out[0])

    def sample(self, random_state=None):
        from stochpylib.spatial_statistics.random_fields import GaussianRandomField

        rng = _rng(random_state)
        W, vol = self._window_arr()
        d = W.shape[0]
        spacing = (W[:, 1] - W[:, 0]) / (np.array(self.grid_shape) - 1)
        grf = GaussianRandomField(covariance=self.covariance, mean=self.mean)
        field = grf.sample_grid(self.grid_shape, spacing=spacing, random_state=rng)
        lam = np.exp(field)
        cell_vol = float(np.prod(spacing))
        counts = rng.poisson(lam * cell_vol)
        chunks = []
        for idx in np.argwhere(counts > 0):
            c = int(counts[tuple(idx)])
            lo = W[:, 0] + idx * spacing
            chunks.append(rng.uniform(lo, lo + spacing, size=(c, d)))
        return np.vstack(chunks) if chunks else np.empty((0, d))

    def _fit(self, points, W, vol):
        from stochpylib.spatial_statistics.variogram import Semivariogram

        n = len(points)
        rmax = 0.25 * float(np.min(W[:, 1] - W[:, 0]))
        r = np.linspace(rmax / 20.0, rmax, 12)
        emp = RipleyK(points, self.window, r=r, correction="translation")

        def obj(theta):
            variance, length_scale = np.exp(theta)
            cov = SpatialCovariance(Semivariogram(model="exponential", nugget=0.0,
                                                    sill=variance, range=3.0 * length_scale))
            model = LogGaussianCox(mean=0.0, covariance=cov, window=self.window)
            return float(np.sum((np.asarray(model.K(r)) ** 0.5 - emp.estimate ** 0.5) ** 2))

        res = optimize.minimize(obj, np.log([1.0, max(rmax / 3.0, 1e-3)]), method="Nelder-Mead")
        variance, length_scale = np.exp(res.x)
        self.covariance = SpatialCovariance(Semivariogram(model="exponential", nugget=0.0,
                                                            sill=variance, range=3.0 * length_scale))
        self.mean = float(np.log(max(n / vol, 1e-9)) - variance / 2.0)
