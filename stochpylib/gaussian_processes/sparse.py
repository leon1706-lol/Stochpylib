"""Sparse GP regression engines: FITC and VFE/SGPR (Titsias 2009)."""

import numpy as np

from stochpylib.gaussian_processes._utils import _as_2d, cholesky_with_jitter

__all__ = ["FITC", "VFE", "SparseVFE"]


class _SparseRegressionBase:
    objective = "vfe"

    def __init__(self, kernel, inducing_points, noise=0.1):
        self.kernel = kernel
        self.noise = float(noise)
        self.Z = _as_2d(inducing_points)

    def fit(self, X, y):
        X = _as_2d(X)
        y = np.asarray(y, dtype=float).ravel()
        if len(y) != len(X):
            raise ValueError("X and y must have equal length")
        sigma2 = max(self.noise, 1e-10)

        Z = self.Z
        M = len(Z)
        Kuu = self.kernel(Z)
        Luu, _ = cholesky_with_jitter(Kuu)
        Kuf = self.kernel(Z, X)                                     # (M, T)

        # Per-point effective noise
        if self.objective == "fitc":
            V = np.linalg.solve(Luu, Kuf)
            Qff_diag = np.sum(V ** 2, axis=0)
            lam_diag = sigma2 + np.maximum(self.kernel.diag(X) - Qff_diag, 1e-10)
        else:
            lam_diag = np.full(len(X), sigma2)

        # --- Posterior over inducing variables (Bayes rule) ---
        # Prior:  p(u) = N(0, Kuu)
        # Likelihood: p(y|u) = N(y; Kfu Kuu^-1 u, Lambda)
        # Posterior precision: Kuu^-1 + Kfu Lambda^-1 Kuf
        # Posterior covariance: inv(Kuu^-1 + Kfu Lam^-1 Kuf)
        # Posterior mean: Sigma_u @ Kfu Lam^-1 y
        Kuu_inv = np.linalg.inv(Kuu)
        self._Kuu_inv = Kuu_inv
        A = Kuu_inv + Kuf @ np.diag(1.0 / lam_diag) @ Kuf.T         # M x M
        self.Sigma_u_ = np.linalg.inv(A)
        self.mu_u_ = self.Sigma_u_ @ (Kuf @ (y / lam_diag))

        # Predictive operator: pred_var_op = Kuu^-1 - Kuu^-1 Sig_u Kuu^-1
        self._pred_var_op = Kuu_inv - Kuu_inv @ self.Sigma_u_ @ Kuu_inv

        self.X_train = X
        self.y_train = y

        # LML bound
        ll = float(-0.5 * float(np.log(np.linalg.det(
            2 * np.pi * (np.diag(lam_diag) + Kuf.T @ Kuu_inv @ Kuf))).sum()))
        if self.objective == "vfe":
            ll += 0.5 * float(np.sum(np.log(np.clip(np.diag(Kuu), 1e-12, None)))) \
                  - 0.5 * float(len(y) * np.log(sigma2))
        self.log_marginal_likelihood_ = ll
        return self

    def predict(self, X_test, return_std=True, full_cov=False):
        X_test = _as_2d(X_test)
        k_zt = self.kernel(self.Z, X_test)                          # (M, t)
        mean = k_zt.T @ self._Kuu_inv @ self.mu_u_
        var = self.kernel.diag(X_test) \
            - np.sum(k_zt.T * (k_zt.T @ self._pred_var_op), axis=1)
        var = np.clip(var, 0.0, None)
        if full_cov:
            second = np.diag(var)
        else:
            second = np.sqrt(var)
        if not (return_std or full_cov):
            return mean
        return mean, second


class FITC(_SparseRegressionBase):
    objective = "fitc"


class VFE(_SparseRegressionBase):
    objective = "vfe"


class SparseVFE(VFE):
    pass
