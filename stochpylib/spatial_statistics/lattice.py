"""SAR and CAR lattice models (documented extras -- not spec names, closing the vault's
"no CAR/SAR" gap noted in ``ARCHITECTURE.md`` / ``Ratings.md``).

Both maximize a concentrated Gaussian log-likelihood over the spatial autoregressive
parameter ``rho`` by 1-D bounded search, with the mean/scale parameters solved in closed
form at each ``rho`` -- SAR by an OLS regression of the lag-transformed response, CAR by a
GLS regression weighted by the CAR precision structure. Standard errors come from the
numerical Hessian of the full (non-concentrated) log-likelihood, matching the library's
existing ``_numeric_hessian`` convention (``statistics._common``).
"""

import numpy as np
from scipy.optimize import minimize_scalar

from stochpylib.statistics._common import _numeric_hessian

__all__ = ["SARModel", "CARModel"]


def _W_matrix(weights):
    return weights.W if hasattr(weights, "W") else np.asarray(weights, dtype=float)


def _design_matrix(X, n):
    if X is None:
        return np.ones((n, 1))
    X = np.asarray(X, dtype=float)
    if X.ndim == 1:
        X = X[:, None]
    if len(X) != n:
        raise ValueError("X must have one row per observation")
    return np.column_stack([np.ones(n), X])


def _rho_bounds(eig):
    lam_min, lam_max = float(eig.min()), float(eig.max())
    rho_lo = 1.0 / lam_min + 1e-6 if lam_min < 0 else -0.999
    rho_hi = 1.0 / lam_max - 1e-6 if lam_max > 0 else 0.999
    return rho_lo, rho_hi


class SARModel:
    """Simultaneous autoregressive (spatial lag) model: ``y = rho*W*y + X*beta + eps``."""

    def __init__(self, weights):
        self.weights = weights

    def fit(self, y, X=None):
        y = np.asarray(y, dtype=float).ravel()
        n = len(y)
        W = _W_matrix(self.weights)
        Xd = _design_matrix(X, n)
        eig = np.linalg.eigvals(W).real  # real part: exact for symmetric W, a practical
        rho_lo, rho_hi = _rho_bounds(eig)  # approximation for a general asymmetric one

        def logdet(rho):
            return float(np.sum(np.log(np.abs(1.0 - rho * eig))))

        def fit_at(rho):
            Ay = y - rho * (W @ y)
            beta, *_ = np.linalg.lstsq(Xd, Ay, rcond=None)
            resid = Ay - Xd @ beta
            return beta, float(resid @ resid) / n

        def neg_conc_loglik(rho):
            _, sigma2 = fit_at(rho)
            if sigma2 <= 0:
                return np.inf
            ll = -0.5 * n * (np.log(2.0 * np.pi) + 1.0) - 0.5 * n * np.log(sigma2) + logdet(rho)
            return -ll

        res = minimize_scalar(neg_conc_loglik, bounds=(rho_lo, rho_hi), method="bounded")
        rho = float(res.x)
        beta, sigma2 = fit_at(rho)
        self.rho_, self.beta_, self.sigma2_ = rho, beta, sigma2
        self.loglik_ = -neg_conc_loglik(rho)
        self.aic_ = -2.0 * self.loglik_ + 2.0 * (len(beta) + 2)
        self.X_, self._W = Xd, W

        def negloglik_full(theta):
            rho_, beta_, log_s2 = theta[0], theta[1:-1], theta[-1]
            s2 = np.exp(log_s2)
            resid_ = (y - rho_ * (W @ y)) - Xd @ beta_
            return -(-0.5 * n * np.log(2.0 * np.pi * s2) + logdet(rho_)
                      - 0.5 * float(resid_ @ resid_) / s2)

        theta0 = np.concatenate([[rho], beta, [np.log(sigma2)]])
        try:
            H = _numeric_hessian(negloglik_full, theta0)
            se = np.sqrt(np.maximum(np.diag(np.linalg.inv(H)), 0.0))
        except np.linalg.LinAlgError:
            se = np.full(len(theta0), np.nan)
        self.std_errors_ = se[:-1]
        return self

    def predict(self):
        n = len(self.X_)
        A = np.eye(n) - self.rho_ * self._W
        return np.linalg.solve(A, self.X_ @ self.beta_)

    def __repr__(self):
        return f"SARModel(rho_={getattr(self, 'rho_', None)!r})"


class CARModel:
    """Gaussian conditional autoregressive model: precision ``(I - rho*W) / sigma2``.

    ``weights`` must be symmetric (the usual CAR requirement, e.g. a binary rook/queen
    lattice weights matrix).
    """

    def __init__(self, weights):
        self.weights = weights

    def fit(self, y, X=None):
        y = np.asarray(y, dtype=float).ravel()
        n = len(y)
        W = _W_matrix(self.weights)
        if not np.allclose(W, W.T):
            raise ValueError("CARModel requires a symmetric weights matrix")
        Xd = _design_matrix(X, n)
        eig = np.linalg.eigvalsh(W)
        rho_lo, rho_hi = _rho_bounds(eig)

        def logdet(rho):
            return float(np.sum(np.log(1.0 - rho * eig)))

        def fit_at(rho):
            Q = np.eye(n) - rho * W
            XtQ = Xd.T @ Q
            beta = np.linalg.solve(XtQ @ Xd, XtQ @ y)
            resid = y - Xd @ beta
            return beta, float(resid @ (Q @ resid)) / n

        def neg_conc_loglik(rho):
            _, sigma2 = fit_at(rho)
            if sigma2 <= 0:
                return np.inf
            ll = -0.5 * n * np.log(2.0 * np.pi) - 0.5 * n * np.log(sigma2) + 0.5 * logdet(rho) - 0.5 * n
            return -ll

        res = minimize_scalar(neg_conc_loglik, bounds=(rho_lo, rho_hi), method="bounded")
        rho = float(res.x)
        beta, sigma2 = fit_at(rho)
        self.rho_, self.beta_, self.sigma2_ = rho, beta, sigma2
        self.loglik_ = -neg_conc_loglik(rho)
        self.aic_ = -2.0 * self.loglik_ + 2.0 * (len(beta) + 2)
        self.X_, self._W, self._eig = Xd, W, eig

        def negloglik_full(theta):
            rho_, beta_, log_s2 = theta[0], theta[1:-1], theta[-1]
            s2 = np.exp(log_s2)
            Q = np.eye(n) - rho_ * W
            resid_ = y - Xd @ beta_
            return -(-0.5 * n * np.log(2.0 * np.pi * s2) + 0.5 * logdet(rho_)
                      - 0.5 * float(resid_ @ (Q @ resid_)) / s2)

        theta0 = np.concatenate([[rho], beta, [np.log(sigma2)]])
        try:
            H = _numeric_hessian(negloglik_full, theta0)
            se = np.sqrt(np.maximum(np.diag(np.linalg.inv(H)), 0.0))
        except np.linalg.LinAlgError:
            se = np.full(len(theta0), np.nan)
        self.std_errors_ = se[:-1]
        return self

    def sample(self, random_state=None):
        from stochpylib.gaussian_processes._utils import cholesky_with_jitter

        n = len(self.X_)
        Q = (np.eye(n) - self.rho_ * self._W) / self.sigma2_
        L, _ = cholesky_with_jitter(np.linalg.inv(Q))
        rng = np.random.default_rng(random_state)
        return self.X_ @ self.beta_ + L @ rng.standard_normal(n)

    def predict(self):
        return self.X_ @ self.beta_

    def __repr__(self):
        return f"CARModel(rho_={getattr(self, 'rho_', None)!r})"
