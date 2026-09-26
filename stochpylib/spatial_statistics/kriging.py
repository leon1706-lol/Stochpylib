"""Kriging predictors: simple, ordinary, universal, co-, indicator and disjunctive kriging.

``Kriging``/``OrdinaryKriging``/``UniversalKriging`` share one gamma-Lagrange linear system
(``_drift_matrix`` is the only thing subclasses change), so the unbounded power/linear
variogram models work. ``SimpleKriging`` uses the covariance form directly (known mean, no
unbiasedness constraint). Every predictor's ``predict(X, return_std=True)`` mirrors
``gaussian_processes.ExactInference.predict``; ``predict_result`` wraps the same output in
``timeseries.ForecastResult`` so no new result type is needed.
"""

import copy

import numpy as np
from scipy.interpolate import PchipInterpolator

from stochpylib.experimental_design._common import _pearson, _ranks
from stochpylib.experimental_design.response_surface import _hermite
from stochpylib.spatial_statistics._common import _as_coords, _as_values, _pairwise
from stochpylib.spatial_statistics.variogram import ExperimentalVariogram, VariogramFitting

__all__ = [
    "Kriging", "SimpleKriging", "OrdinaryKriging", "UniversalKriging", "CoKriging",
    "IndicatorKriging", "DisjunctiveKriging",
]


def _auto_variogram(X, z):
    ev = ExperimentalVariogram().fit(X, z)
    return VariogramFitting(model=["spherical", "exponential", "gaussian"]).fit(ev).variogram_


def _loo_cv(fresh, X, z):
    """Leave-one-out cross-validation diagnostics for a fitted kriging model."""
    n = len(X)
    errors = np.empty(n)
    std_errors = np.empty(n)
    for i in range(n):
        mask = np.ones(n, dtype=bool)
        mask[i] = False
        m = fresh()
        m.fit(X[mask], z[mask])
        mu, s = m.predict(X[i:i + 1], return_std=True)
        errors[i] = z[i] - mu[0]
        std_errors[i] = s[0]
    standardized = errors / np.where(std_errors > 0, std_errors, np.nan)
    return {
        "errors": errors,
        "standardized_errors": standardized,
        "rmse": float(np.sqrt(np.mean(errors ** 2))),
        "mean_error": float(np.mean(errors)),
        "msdr": float(np.nanmean(standardized ** 2)),
    }


class Kriging:
    """Ordinary kriging by default (``_drift_matrix`` is a constant-only column).

    ``variogram=None`` auto-fits one (``ExperimentalVariogram`` -> best of
    spherical/exponential/gaussian by AIC). ``n_neighbors`` restricts each prediction to a
    local system over its k nearest data points; otherwise one global system is factored once
    in ``fit``.
    """

    def __init__(self, variogram=None, n_neighbors=None, max_dist=None):
        self.variogram = variogram
        self.n_neighbors = n_neighbors
        self.max_dist = max_dist

    def _drift_matrix(self, X):
        return np.ones((len(X), 1))

    def fit(self, coords, values):
        X = _as_coords(coords)
        z = _as_values(values, len(X))
        self.X_, self.z_ = X, z
        self.variogram_ = self.variogram if self.variogram is not None else _auto_variogram(X, z)
        self._p = self._drift_matrix(X[:1]).shape[1]
        if self.n_neighbors is None:
            self._prepare_global()
        return self

    def _prepare_global(self):
        X = self.X_
        n = len(X)
        Gamma = self.variogram_(_pairwise(X))
        F = self._drift_matrix(X)
        p = F.shape[1]
        A = np.zeros((n + p, n + p))
        A[:n, :n] = Gamma
        A[:n, n:] = F
        A[n:, :n] = F.T
        self._A_inv = np.linalg.inv(A)
        self._n = n

    def _solve_local(self, X0):
        X = self.X_
        k = min(int(self.n_neighbors), len(X))
        D = _pairwise(X0, X)
        idxs = np.argsort(D, axis=1)[:, :k]
        m = len(X0)
        W = np.zeros((len(X), m))
        Lam = np.zeros((self._p, m))
        for i in range(m):
            sub = idxs[i]
            Xi = X[sub]
            Gamma = self.variogram_(_pairwise(Xi))
            F = self._drift_matrix(Xi)
            p = F.shape[1]
            A = np.zeros((k + p, k + p))
            A[:k, :k] = Gamma
            A[:k, k:] = F
            A[k:, :k] = F.T
            g0 = self.variogram_(_pairwise(Xi, X0[i:i + 1])).ravel()
            f0 = self._drift_matrix(X0[i:i + 1]).ravel()
            sol = np.linalg.solve(A, np.concatenate([g0, f0]))
            W[sub, i] = sol[:k]
            Lam[:, i] = sol[k:]
        return W, Lam

    def _solve(self, X0):
        if self.n_neighbors is not None:
            return self._solve_local(X0)
        Gamma0 = self.variogram_(_pairwise(self.X_, X0))
        F0 = self._drift_matrix(X0).T
        RHS = np.vstack([Gamma0, F0])
        sol = self._A_inv @ RHS
        return sol[:self._n, :], sol[self._n:, :]

    def weights(self, x0):
        """The kriging weights (and Lagrange multipliers) at a single point ``x0``."""
        X0 = _as_coords(np.atleast_2d(x0))
        w, lam = self._solve(X0)
        return w[:, 0], lam[:, 0]

    def predict(self, X, return_std=False):
        X0 = _as_coords(X)
        w, lam = self._solve(X0)
        mean = (w * self.z_[:, None]).sum(axis=0)
        if not return_std:
            return mean
        Gamma0 = self.variogram_(_pairwise(self.X_, X0))
        F0 = self._drift_matrix(X0).T
        var = np.sum(w * Gamma0, axis=0) + np.sum(lam * F0, axis=0)
        return mean, np.sqrt(np.maximum(var, 0.0))

    def predict_result(self, X):
        from stochpylib.timeseries import ForecastResult

        mean, std = self.predict(X, return_std=True)
        return ForecastResult(mean, std)

    def cross_validate(self):
        def fresh():
            m = copy.copy(self)
            m.variogram = self.variogram_
            return m

        return _loo_cv(fresh, self.X_, self.z_)

    def __repr__(self):
        n = len(self.X_) if hasattr(self, "X_") else 0
        return f"{type(self).__name__}(n={n})"


class OrdinaryKriging(Kriging):
    """Kriging with the weights constrained to sum to 1 (``Kriging``'s default drift)."""


class UniversalKriging(Kriging):
    """Kriging with a polynomial (or external) drift, fitted by GLS.

    ``drift``: ``"linear"``/``"quadratic"`` polynomial in the coordinates, a callable
    ``f(X) -> (n,)``, or a list of such callables (each becomes one extra drift column).
    ``external_drift``, if given, is a callable ``f(X) -> (n,)`` evaluated at both the data
    and the prediction points (so cross-validation and out-of-sample prediction need no
    external array bookkeeping). With a drift, the variogram (when not given) is auto-fit on
    the OLS-detrended residuals.
    """

    def __init__(self, drift="linear", external_drift=None, variogram=None,
                 n_neighbors=None, max_dist=None):
        super().__init__(variogram=variogram, n_neighbors=n_neighbors, max_dist=max_dist)
        if not (callable(drift) or isinstance(drift, (list, tuple))
                or drift in ("linear", "quadratic")):
            raise ValueError("drift must be 'linear', 'quadratic', a callable, or a list")
        self.drift = drift
        self.external_drift = external_drift

    def _drift_matrix(self, X):
        cols = [np.ones(len(X))]
        d = X.shape[1]
        if self.drift == "linear":
            cols += [X[:, j] for j in range(d)]
        elif self.drift == "quadratic":
            cols += [X[:, j] for j in range(d)]
            cols += [X[:, j] ** 2 for j in range(d)]
            cols += [X[:, j] * X[:, k] for j in range(d) for k in range(j + 1, d)]
        elif callable(self.drift):
            cols.append(np.asarray(self.drift(X), dtype=float).ravel())
        else:
            cols += [np.asarray(f(X), dtype=float).ravel() for f in self.drift]
        if self.external_drift is not None:
            cols.append(np.asarray(self.external_drift(X), dtype=float).ravel())
        return np.column_stack(cols)

    def fit(self, coords, values):
        X = _as_coords(coords)
        z = _as_values(values, len(X))
        self.X_, self.z_ = X, z
        if self.variogram is not None:
            self.variogram_ = self.variogram
        else:
            F = self._drift_matrix(X)
            beta, *_ = np.linalg.lstsq(F, z, rcond=None)
            self.variogram_ = _auto_variogram(X, z - F @ beta)
        self._p = self._drift_matrix(X[:1]).shape[1]
        if self.n_neighbors is None:
            self._prepare_global()
        return self


class SimpleKriging:
    """Kriging with a known (or sample-estimated) mean, using the covariance form directly."""

    def __init__(self, variogram=None, mean=None, n_neighbors=None):
        self.variogram = variogram
        self.mean = mean
        self.n_neighbors = n_neighbors

    def fit(self, coords, values):
        X = _as_coords(coords)
        z = _as_values(values, len(X))
        self.X_, self.z_ = X, z
        self.variogram_ = self.variogram if self.variogram is not None else _auto_variogram(X, z)
        self.mean_ = float(np.mean(z)) if self.mean is None else float(self.mean)
        self._r_ = z - self.mean_
        if self.n_neighbors is None:
            C = self.variogram_.covariance(_pairwise(X))
            self._C_inv = np.linalg.inv(C)
        return self

    def _solve(self, X0):
        if self.n_neighbors is None:
            C0 = self.variogram_.covariance(_pairwise(self.X_, X0))
            return self._C_inv @ C0
        X, r = self.X_, self._r_
        k = min(int(self.n_neighbors), len(X))
        D = _pairwise(X0, X)
        idxs = np.argsort(D, axis=1)[:, :k]
        m = len(X0)
        W = np.zeros((len(X), m))
        for i in range(m):
            sub = idxs[i]
            Xi = X[sub]
            C = self.variogram_.covariance(_pairwise(Xi))
            c0 = self.variogram_.covariance(_pairwise(Xi, X0[i:i + 1])).ravel()
            W[sub, i] = np.linalg.solve(C, c0)
        return W

    def weights(self, x0):
        X0 = _as_coords(np.atleast_2d(x0))
        return self._solve(X0)[:, 0]

    def predict(self, X, return_std=False):
        X0 = _as_coords(X)
        W = self._solve(X0)
        mean = self.mean_ + (W * self._r_[:, None]).sum(axis=0)
        if not return_std:
            return mean
        C0 = self.variogram_.covariance(_pairwise(self.X_, X0))
        var = self.variogram_.sill - np.sum(W * C0, axis=0)
        return mean, np.sqrt(np.maximum(var, 0.0))

    def predict_result(self, X):
        from stochpylib.timeseries import ForecastResult

        mean, std = self.predict(X, return_std=True)
        return ForecastResult(mean, std)

    def cross_validate(self):
        def fresh():
            m = copy.copy(self)
            m.variogram = self.variogram_
            return m

        return _loo_cv(fresh, self.X_, self.z_)

    def __repr__(self):
        n = len(self.X_) if hasattr(self, "X_") else 0
        return f"SimpleKriging(n={n})"


class CoKriging:
    """Ordinary cokriging of a primary variable using one secondary variable.

    Uses the Markov Model 1 (intrinsic collocated cokriging) simplification: the
    cross-covariance is ``rho * sqrt(sill1 * sill2) * corr1(h)``, i.e. proportional to the
    primary's own correlation function, and the secondary shares the primary's spatial
    structure scaled to its own sill. This guarantees a valid (positive semi-definite) model
    with no separate cross-variogram fit, at the cost of assuming the two variables share a
    spatial range (a standard, well-documented geostatistical simplification -- Journel 1999).
    ``rho`` is estimated from the primary and its nearest secondary neighbour at each site.
    """

    def __init__(self, variogram=None, cross_variogram=None):
        self.variogram = variogram
        self.cross_variogram = cross_variogram  # accepted for API symmetry; unused (see above)

    def _corr1(self, h):
        return self.variogram_.covariance(h) / self.variogram_.sill

    def fit(self, coords1, z1, coords2, z2):
        X1 = _as_coords(coords1)
        y1 = _as_values(z1, len(X1))
        X2 = _as_coords(coords2)
        y2 = _as_values(z2, len(X2))
        self.X1_, self.z1_, self.X2_, self.z2_ = X1, y1, X2, y2
        self.variogram_ = self.variogram if self.variogram is not None else _auto_variogram(X1, y1)
        self.sill2_ = float(np.var(y2, ddof=1)) if len(y2) > 1 else float(y2[0] ** 2)
        nearest = np.argmin(_pairwise(X1, X2), axis=1)
        self.rho_ = _pearson(y1, y2[nearest])

        n1, n2 = len(X1), len(X2)
        C11 = self.variogram_.covariance(_pairwise(X1))
        C22 = self.sill2_ * self._corr1(_pairwise(X2))
        cross_scale = self.rho_ * np.sqrt(self.variogram_.sill * self.sill2_)
        C12 = cross_scale * self._corr1(_pairwise(X1, X2))

        m = n1 + n2 + 2
        A = np.zeros((m, m))
        A[:n1, :n1] = C11
        A[:n1, n1:n1 + n2] = C12
        A[n1:n1 + n2, :n1] = C12.T
        A[n1:n1 + n2, n1:n1 + n2] = C22
        A[:n1, n1 + n2] = 1.0
        A[n1 + n2, :n1] = 1.0
        A[n1:n1 + n2, n1 + n2 + 1] = 1.0
        A[n1 + n2 + 1, n1:n1 + n2] = 1.0
        self._A_inv = np.linalg.inv(A)
        self._n1, self._n2 = n1, n2
        self._cross_scale = cross_scale
        return self

    def predict(self, X, return_std=False):
        X0 = _as_coords(X)
        c1_0 = self.variogram_.covariance(_pairwise(self.X1_, X0))
        c2_0 = self._cross_scale * self._corr1(_pairwise(self.X2_, X0))
        m = X0.shape[0]
        RHS = np.vstack([c1_0, c2_0, np.ones((1, m)), np.zeros((1, m))])
        sol = self._A_inv @ RHS
        w1 = sol[:self._n1]
        w2 = sol[self._n1:self._n1 + self._n2]
        mean = (w1 * self.z1_[:, None]).sum(axis=0) + (w2 * self.z2_[:, None]).sum(axis=0)
        if not return_std:
            return mean
        lam1 = sol[self._n1 + self._n2]
        var = self.variogram_.sill - np.sum(w1 * c1_0, axis=0) - np.sum(w2 * c2_0, axis=0) - lam1
        return mean, np.sqrt(np.maximum(var, 0.0))

    def __repr__(self):
        return f"CoKriging(rho_={getattr(self, 'rho_', None)!r})"


class IndicatorKriging:
    """Non-parametric ccdf estimation: one ordinary kriging of ``I(Z <= t)`` per threshold.

    ``predict_cdf`` applies the Deutsch & Journel order-relation correction (average of the
    upward and downward monotone-clipped corrections) so the ccdf is monotone in [0, 1].
    """

    def __init__(self, thresholds, variogram=None, shared_variogram=False):
        self.thresholds = np.asarray(thresholds, dtype=float)
        if self.thresholds.ndim != 1 or len(self.thresholds) < 1:
            raise ValueError("thresholds must be a non-empty 1-D sequence")
        self.variogram = variogram
        self.shared_variogram = bool(shared_variogram)

    def fit(self, coords, values):
        X = _as_coords(coords)
        z = _as_values(values, len(X))
        self.X_, self.z_ = X, z
        v_shared = None
        if self.shared_variogram:
            mid = (z <= self.thresholds[len(self.thresholds) // 2]).astype(float)
            v_shared = self.variogram if self.variogram is not None else _auto_variogram(X, mid)
        self.models_ = []
        for t in self.thresholds:
            ind = (z <= t).astype(float)
            if self.shared_variogram:
                v = v_shared
            else:
                v = self.variogram if self.variogram is not None else _auto_variogram(X, ind)
            self.models_.append(OrdinaryKriging(variogram=v).fit(X, ind))
        return self

    def predict_cdf(self, X):
        X0 = _as_coords(X)
        P = np.column_stack([np.clip(m.predict(X0), 0.0, 1.0) for m in self.models_])
        up = np.maximum.accumulate(P, axis=1)
        down = np.minimum.accumulate(P[:, ::-1], axis=1)[:, ::-1]
        return 0.5 * (up + down)

    def predict_proba(self, X, threshold):
        """``P(Z > threshold | data)`` -- the ccdf at the nearest fitted threshold."""
        cdf = self.predict_cdf(X)
        idx = int(min(np.searchsorted(self.thresholds, threshold), len(self.thresholds) - 1))
        return 1.0 - cdf[:, idx]

    def predict(self, X):
        """E-type estimate: the mean of the ccdf, using class midpoints and data extremes."""
        cdf = self.predict_cdf(X)
        t = self.thresholds
        mids = np.concatenate(([float(self.z_.min())],
                                0.5 * (t[:-1] + t[1:]),
                                [float(self.z_.max())]))
        probs = np.diff(np.column_stack([np.zeros(len(cdf)), cdf, np.ones(len(cdf))]), axis=1)
        return (probs * mids[None, :]).sum(axis=1)

    def __repr__(self):
        return f"IndicatorKriging(n_thresholds={len(self.thresholds)})"


class DisjunctiveKriging:
    """Gaussian anamorphosis disjunctive kriging: ``Z = phi(Y) = sum_k coeffs_k He_k(Y)``.

    ``Y`` is the normal-score transform of the data; ``phi`` (monotone, PCHIP) is fit to the
    empirical (Y, Z) pairs; each Hermite coefficient is estimated by Gauss-Hermite quadrature
    of ``phi``, and each ``He_k(Y)`` (mean 0, unit variance for k >= 1) is simple-kriged with
    covariance ``rho(h)**k``, where ``rho`` is the fitted correlogram of ``Y``. ``He_1`` is the
    identity, so ``predict_proba`` reuses its simple-kriged mean/std of ``Y`` directly:
    ``P(Z > z0) = P(Y > phi^-1(z0))`` under the (Gaussian) simple-kriging distribution of Y.
    """

    def __init__(self, n_hermite=30, variogram=None):
        self.n_hermite = int(n_hermite)
        if self.n_hermite < 2:
            raise ValueError("n_hermite must be >= 2")
        self.variogram = variogram

    def fit(self, coords, values):
        from stochpylib.numerical_methods import GaussHermite
        from stochpylib.statistics._common import _norm_ppf

        X = _as_coords(coords)
        z = _as_values(values, len(X))
        self.X_, self.z_ = X, z
        n = len(z)
        ranks = _ranks(z)
        p = np.clip((ranks - 0.5) / n, 1e-6, 1.0 - 1e-6)
        y = _norm_ppf(p)
        self.y_ = y

        order = np.argsort(y)
        ys, zs = y[order], z[order]
        zs_u, first = np.unique(zs, return_index=True)
        self._phi = PchipInterpolator(ys, zs, extrapolate=True)
        self._phi_inv = PchipInterpolator(zs_u, ys[first], extrapolate=True)

        gh = GaussHermite(self.n_hermite, kind="probabilists")
        He_nodes = _hermite(gh.nodes_, self.n_hermite - 1)          # (n_hermite, n_hermite)
        phi_vals = self._phi(gh.nodes_)
        self.coeffs_ = He_nodes @ (gh.weights_ * phi_vals)          # phi_k = E[Z He_k(Y)]

        self.variogram_ = self.variogram if self.variogram is not None else _auto_variogram(X, y)
        H = _pairwise(X)
        rho_h = self.variogram_.covariance(H) / self.variogram_.sill
        self._He_data = _hermite(y, self.n_hermite - 1)             # (n_hermite, n)
        self._Cinv_k = [np.linalg.inv(rho_h ** k + 1e-10 * np.eye(n))
                         for k in range(1, self.n_hermite)]
        return self

    def _krige_y(self, X0):
        H0 = _pairwise(self.X_, X0)
        rho0 = self.variogram_.covariance(H0) / self.variogram_.sill
        w = self._Cinv_k[0] @ rho0
        muY = (w * self.y_[:, None]).sum(axis=0)
        varY = 1.0 - np.sum(w * rho0, axis=0)
        return muY, np.sqrt(np.maximum(varY, 0.0))

    def predict(self, X, return_std=False):
        X0 = _as_coords(X)
        H0 = _pairwise(self.X_, X0)
        rho0 = self.variogram_.covariance(H0) / self.variogram_.sill
        mean = np.full(len(X0), self.coeffs_[0])
        var = np.zeros(len(X0))
        for i, k in enumerate(range(1, self.n_hermite)):
            c0k = rho0 ** k
            w = self._Cinv_k[i] @ c0k
            hk = (w * self._He_data[k][:, None]).sum(axis=0)
            mean += self.coeffs_[k] * hk
            if return_std:
                var += self.coeffs_[k] ** 2 * np.maximum(1.0 - np.sum(w * c0k, axis=0), 0.0)
        if not return_std:
            return mean
        return mean, np.sqrt(var)

    def predict_proba(self, X, threshold):
        from stochpylib.statistics._common import _norm_cdf

        X0 = _as_coords(X)
        muY, sigY = self._krige_y(X0)
        y0 = float(self._phi_inv(threshold))
        sigY = np.where(sigY > 0, sigY, 1e-12)
        return 1.0 - _norm_cdf((y0 - muY) / sigY)

    def __repr__(self):
        return f"DisjunctiveKriging(n_hermite={self.n_hermite})"
