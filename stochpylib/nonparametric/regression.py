"""Nonparametric smoothers and regressors: local polynomial (Nadaraya-Watson/
local-linear), isotonic (PAVA), spline (P-spline / smoothing-spline), a GP
facade, and local quantile regression.
"""

import math

import numpy as np
from scipy import optimize

from stochpylib.nonparametric._base import NonparametricRegressor
from stochpylib.nonparametric._common import (
    _bspline_design, _bspline_knots, _kernel_fn, _pava_weighted, _select_bandwidth,
)

__all__ = ["LocalPolynomialReg", "IsotonicRegression", "SplineRegression",
           "GPR_Nonparametric", "QuantileRegression"]


class LocalPolynomialReg(NonparametricRegressor):
    """Local polynomial regression: ``degree=0`` is Nadaraya-Watson,
    ``degree=1`` (default) is local-linear. Matches
    ``statsmodels.nonparametric.kernel_regression.KernelReg(reg_type="lc"/"ll",
    bw=[h])`` exactly at a fixed bandwidth. ``bandwidth="gcv"`` minimizes the
    leave-one-out weighted generalized cross-validation score.
    """

    def __init__(self, degree=1, bandwidth="gcv", kernel="gaussian"):
        self.degree = degree
        self.bandwidth = bandwidth
        self.kernel = kernel

    def _weights_row(self, x0, X, h):
        K = _kernel_fn(self.kernel)
        u = (X[:, 0] - x0) / h
        return K(u)

    def _fit_at(self, x0, X, y, h):
        w = self._weights_row(x0, X, h)
        if self.degree == 0:
            sw = np.sum(w)
            return float(np.sum(w * y) / sw) if sw > 0 else float(np.mean(y)), w / sw if sw > 0 else w
        dx = X[:, 0] - x0
        Z = np.column_stack([np.ones_like(dx), dx])
        WZ = Z * w[:, None]
        try:
            beta = np.linalg.solve(WZ.T @ Z + 1e-12 * np.eye(2), WZ.T @ y)
        except np.linalg.LinAlgError:
            beta = np.linalg.lstsq(WZ.T @ Z, WZ.T @ y, rcond=None)[0]
        # equivalent kernel weights (row of the smoother matrix at x0)
        try:
            M = np.linalg.solve(WZ.T @ Z + 1e-12 * np.eye(2), WZ.T)
        except np.linalg.LinAlgError:
            M = np.linalg.pinv(WZ.T @ Z) @ WZ.T
        s_row = M[0, :]
        return float(beta[0]), s_row

    def _fit(self, X, y):
        x = X[:, 0]
        if self.bandwidth == "gcv":
            h = self._select_gcv(x, y)
        else:
            h = _select_bandwidth(x, self.bandwidth, self.kernel)
        self.bandwidth_ = h
        S = np.array([self._fit_at(xi, X, y, h)[1] for xi in x])
        self.smoother_matrix_ = S
        self.df_ = float(np.trace(S))
        self._se = self._pointwise_se(S, y)
        self.fitted_ = S @ y

    def _select_gcv(self, x, y, h_grid=None):
        from stochpylib.nonparametric._common import _bw_silverman
        h0 = _bw_silverman(x)
        if h_grid is None:
            h_grid = np.geomspace(0.2 * h0, 4.0 * h0, 20)
        n = len(x)
        best_h, best_score = float(h_grid[0]), np.inf
        X2 = x[:, None]
        for h in h_grid:
            S = np.array([self._fit_at(xi, X2, y, h)[1] for xi in x])
            fitted = S @ y
            trace_s = np.trace(S)
            resid = y - fitted
            denom = 1.0 - trace_s / n
            score = np.mean(resid ** 2) / denom ** 2 if denom > 1e-6 else np.inf
            if score < best_score:
                best_score, best_h = score, float(h)
        return best_h

    @staticmethod
    def _pointwise_se(S, y):
        n = len(y)
        fitted = S @ y
        resid = y - fitted
        df_res = max(n - 2 * np.trace(S) + np.trace(S @ S.T), 1.0)
        sigma2 = float(np.sum(resid ** 2) / df_res)
        var_row = sigma2 * np.sum(S ** 2, axis=1)
        return np.sqrt(np.clip(var_row, 0, None))

    def predict(self, X, return_std=False):
        X = np.asarray(X, dtype=float)
        if X.ndim == 1:
            X = X[:, None]
        h = self.bandwidth_
        preds, ses = [], []
        for xi in X[:, 0]:
            val, s_row = self._fit_at(xi, self.X_, self.y_, h)
            preds.append(val)
            if return_std:
                n = len(self.y_)
                resid = self.y_ - self.fitted_
                df_res = max(n - 2 * self.df_ + np.trace(self.smoother_matrix_ @
                              self.smoother_matrix_.T), 1.0)
                sigma2 = float(np.sum(resid ** 2) / df_res)
                ses.append(math.sqrt(max(sigma2 * float(np.sum(s_row ** 2)), 0.0)))
        preds = np.array(preds)
        if return_std:
            return preds, np.array(ses)
        return preds


class IsotonicRegression(NonparametricRegressor):
    """Weighted pool-adjacent-violators isotonic regression. Matches
    ``scipy.optimize.isotonic_regression`` exactly (sorted by ``X``).
    """

    def __init__(self, increasing=True, out_of_bounds="clip"):
        self.increasing = increasing
        self.out_of_bounds = out_of_bounds

    def _fit(self, X, y, sample_weight=None):
        x = X[:, 0]
        order = np.argsort(x, kind="mergesort")
        self._x_sorted = x[order]
        w = None if sample_weight is None else np.asarray(sample_weight, dtype=float)[order]
        fitted_sorted = _pava_weighted(y[order], w, increasing=self.increasing)
        self._y_sorted = fitted_sorted
        fitted = np.empty_like(y)
        fitted[order] = fitted_sorted
        self.fitted_ = fitted

    def fit(self, X, y, sample_weight=None):
        X = np.asarray(X, dtype=float)
        if X.ndim == 1:
            X = X[:, None]
        y = np.asarray(y, dtype=float).ravel()
        self.X_, self.y_ = X, y
        self._fit(X, y, sample_weight)
        self.resid_ = y - self.fitted_
        return self

    def predict(self, X, return_std=False):
        X = np.asarray(X, dtype=float)
        x = X[:, 0] if X.ndim == 2 else X
        lo, hi = self._x_sorted[0], self._x_sorted[-1]
        if self.out_of_bounds == "clip":
            xc = np.clip(x, lo, hi)
        else:
            xc = x
        out = np.interp(xc, self._x_sorted, self._y_sorted)
        return out


class SplineRegression(NonparametricRegressor):
    """``method="pspline"``: B-spline basis + 2nd-difference roughness penalty
    (P-spline, Eilers-Marx), GCV-selected ``lam`` by default; at ``lam=0`` it
    is unpenalized B-spline OLS. ``method="smoothing"``: a natural cubic
    smoothing spline via ``scipy.interpolate.make_smoothing_spline`` (raw
    numerical machinery, not a statistical oracle; matches it exactly at a
    given ``lam``).
    """

    def __init__(self, method="pspline", n_knots=20, degree=3, lam="gcv"):
        self.method = method
        self.n_knots = n_knots
        self.degree = degree
        self.lam = lam

    def _fit(self, X, y):
        x = X[:, 0]
        if self.method == "smoothing":
            from scipy.interpolate import make_smoothing_spline
            order = np.argsort(x)
            lam = None if self.lam == "gcv" else self.lam
            spl = make_smoothing_spline(x[order], y[order], lam=lam)
            self._spline = spl
            self.fitted_ = spl(x)
            # scipy does not expose its internally GCV-selected lambda
            self.lam_ = lam
            self.df_ = float("nan")
            return
        if self.method != "pspline":
            raise ValueError("method must be 'pspline' or 'smoothing'")
        knots = _bspline_knots(x, self.n_knots, self.degree)
        self._knots = knots
        B = _bspline_design(x, knots, self.degree)
        p = B.shape[1]
        D = np.diff(np.eye(p), n=2, axis=0)
        P = D.T @ D
        if self.lam == "gcv":
            lam = self._select_gcv(B, P, y)
        else:
            lam = float(self.lam)
        self.lam_ = lam
        A = B.T @ B + lam * P
        beta = np.linalg.solve(A, B.T @ y)
        self._beta = beta
        S = B @ np.linalg.solve(A, B.T)
        self.df_ = float(np.trace(S))
        self.fitted_ = B @ beta

    def _select_gcv(self, B, P, y, lam_grid=None):
        n = len(y)
        if lam_grid is None:
            lam_grid = np.geomspace(1e-4, 1e4, 25)
        best_lam, best_score = 1.0, np.inf
        for lam in lam_grid:
            A = B.T @ B + lam * P
            try:
                S = B @ np.linalg.solve(A, B.T)
            except np.linalg.LinAlgError:
                continue
            fitted = S @ y
            trace_s = np.trace(S)
            denom = 1.0 - trace_s / n
            if denom <= 1e-6:
                continue
            score = np.mean((y - fitted) ** 2) / denom ** 2
            if score < best_score:
                best_score, best_lam = score, float(lam)
        return best_lam

    def predict(self, X, return_std=False):
        X = np.asarray(X, dtype=float)
        x = X[:, 0] if X.ndim == 2 else X
        if self.method == "smoothing":
            return self._spline(x)
        B = _bspline_design(x, self._knots, self.degree)
        return B @ self._beta


class GPR_Nonparametric(NonparametricRegressor):
    """Facade over :class:`stochpylib.gaussian_processes.GPRegression`
    (default RBF + white-noise kernel): same nonparametric-smoother surface
    (``fit``/``predict(return_std=)``), reusing the library's exact GP
    inference rather than reimplementing it.
    """

    def __init__(self, kernel=None, noise=0.1, optimize=True):
        self.kernel = kernel
        self.noise = noise
        self.optimize = optimize

    def _fit(self, X, y):
        from stochpylib.gaussian_processes import GPRegression
        from stochpylib.gaussian_processes.kernels import RBFKernel, WhiteNoiseKernel
        kernel = self.kernel
        if kernel is None:
            scale = float(np.std(X[:, 0])) or 1.0
            kernel = RBFKernel(length_scale=scale, variance=float(np.var(y)) or 1.0) + \
                WhiteNoiseKernel(self.noise)
        self._gp = GPRegression(kernel=kernel, noise=self.noise)
        self._gp.fit(X, y)
        if self.optimize:
            from stochpylib.gaussian_processes.hyperparams import optimize_hyperparams
            optimize_hyperparams(self._gp)
        self.fitted_ = np.asarray(self._gp.predict(X), dtype=float)

    def predict(self, X, return_std=False):
        X = np.asarray(X, dtype=float)
        if X.ndim == 1:
            X = X[:, None]
        return self._gp.predict(X, return_std=return_std)


class QuantileRegression(NonparametricRegressor):
    """Local (kernel-weighted check-loss) or global linear quantile
    regression. ``method="local"`` (default): kernel-weighted linear program
    at each query point. ``method="linear"`` delegates to
    :func:`stochpylib.statistics.quantile_regression`. A very large
    ``bandwidth`` makes ``"local"`` converge to the global linear fit.
    """

    def __init__(self, q=0.5, bandwidth="rule", kernel="gaussian", method="local"):
        self.q = q
        self.bandwidth = bandwidth
        self.kernel = kernel
        self.method = method

    def _fit(self, X, y):
        x = X[:, 0]
        if self.method == "linear":
            from stochpylib.statistics import quantile_regression
            self._lin = quantile_regression(x, y, q=self.q)
            self.fitted_ = self._lin.fitted_
            return
        if self.bandwidth == "rule":
            n = len(x)
            std = float(np.std(x, ddof=1))
            h = 0.9 * std * n ** (-1.0 / 5.0)
        else:
            h = float(self.bandwidth)
        self.bandwidth_ = h
        self.fitted_ = np.array([self._local_fit(xi, x, y, h) for xi in x])

    def _local_fit(self, x0, x, y, h):
        K = _kernel_fn(self.kernel)
        w_all = K((x - x0) / h)
        # drop points whose weight is relatively negligible -- bounds the LP
        # size per query without biasing the fit (exact for a huge bandwidth,
        # where every weight stays comparably large and nothing is dropped).
        max_w = np.max(w_all)
        keep = w_all >= 1e-6 * max_w if max_w > 0 else np.ones_like(w_all, dtype=bool)
        if np.sum(keep) < 5:
            keep = np.argsort(-w_all)[:min(len(x), 20)]
        w = np.clip(w_all[keep], 1e-12, None)
        dx = x[keep] - x0
        yk = y[keep]
        n = len(dx)
        c = np.concatenate([np.zeros(2), self.q * w, (1 - self.q) * w])
        A_eq = np.column_stack([np.ones(n), dx])
        A_eq = np.hstack([A_eq, np.eye(n), -np.eye(n)])
        bounds = [(None, None)] * 2 + [(0, None)] * (2 * n)
        res = optimize.linprog(c, A_eq=A_eq, b_eq=yk, bounds=bounds, method="highs")
        return float(res.x[0])

    def predict(self, X, return_std=False):
        X = np.asarray(X, dtype=float)
        x = X[:, 0] if X.ndim == 2 else X
        if self.method == "linear":
            return self._lin.extras["_predict"](x)
        h = self.bandwidth_
        return np.array([self._local_fit(xi, self.X_[:, 0], self.y_, h) for xi in x])
