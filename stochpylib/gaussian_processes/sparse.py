"""Sparse GP regression engines: FITC and VFE/SGPR (Titsias 2009).

Both engines share one fit/predict surface and are implemented in the **whitened
parameterization**: with ``u ~ N(0, Kuu)`` the inducing values are transformed to
``u~ = Luu^-1 u`` so every subsequent object is well-conditioned by construction
(``A = I + V Lam^-1 V^T`` has eigenvalues >= 1). All linear algebra runs through
jittered Cholesky solves — no raw inverses of near-singular kernel matrices.
"""

import numpy as np

from stochpylib.gaussian_processes._utils import _as_2d, cholesky_with_jitter

__all__ = ["FITC", "VFE", "SparseVFE", "InducingPointGP", "SparseGaussianProcess"]

_LOG_2PI = np.log(2.0 * np.pi)


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
        Luu, jitter_u = cholesky_with_jitter(self.kernel(Z))
        Kuf = self.kernel(Z, X)                                     # (M, T)

        # Whitened cross-covariance V = Luu^-1 Kuf; then diag(Kff) = colsums(V^2)
        # and Qff = V^T V exactly.
        V = np.linalg.solve(Luu, Kuf)
        Qff_diag = np.sum(V**2, axis=0)

        # Per-point effective noise (FITC augments with the low-rank residual).
        if self.objective == "fitc":
            lam_diag = sigma2 + np.maximum(self.kernel.diag(X) - Qff_diag, 1e-10)
        else:
            lam_diag = np.full(len(X), sigma2)

        # --- Posterior over the whitened inducing values u~ = Luu^-1 u ---
        #   q(u~) = N(m~, S~),  S~ = (I + V Lam^-1 V^T)^-1,  m~ = S~ V Lam^-1 y
        A = np.eye(M) + (V / lam_diag[None, :]) @ V.T               # eig >= 1
        LA, _ = cholesky_with_jitter(A)
        b = V @ (y / lam_diag)
        self.m_tilde_ = np.linalg.solve(LA.T, np.linalg.solve(LA, b))

        self.Luu_, self._LA_, self.jitter_used_ = Luu, LA, jitter_u
        self.X_train = X
        self.y_train = y

        # Log marginal: for VFE this is exactly Titsias' SGPR bound (the log evidence
        # of q(y) = N(0, Qff + Lam)); for FITC the analogous pseudo-log-evidence under
        # its own effective-noise Lambda. Both reduce to the same whitened formula:
        #   L = -0.5 [ n log 2pi + sum(log lam) + y' Lam^-1 y - b' A^-1 b ]
        quad = float((y / lam_diag) @ y) - float(b @ self.m_tilde_)
        self.log_marginal_likelihood_ = -0.5 * (
            len(y) * _LOG_2PI + float(np.sum(np.log(lam_diag))) + quad
        )
        return self

    def predict(self, X_test, return_std=True, full_cov=False):
        X_test = _as_2d(X_test)
        k_zt = self.kernel(self.Z, X_test)                          # (M, t)
        Vz = np.linalg.solve(self.Luu_, k_zt)                       # (M, t)
        mean = Vz.T @ self.m_tilde_
        W = np.linalg.solve(self._LA_, Vz)
        var = self.kernel.diag(X_test) \
            - np.sum(Vz**2, axis=0) + np.sum(W**2, axis=0)
        var = np.clip(var, 0.0, None)
        if full_cov:
            second = np.diag(var)
        else:
            second = np.sqrt(var)
        if not (return_std or full_cov):
            return mean
        return mean, second

    def log_marginal_likelihood(self):
        if self.log_marginal_likelihood_ is None:
            raise RuntimeError("fit() must be called first")
        return self.log_marginal_likelihood_


class FITC(_SparseRegressionBase):
    """Fully independent training conditional (Snelson & Ghahramani, 2006)."""

    objective = "fitc"


class VFE(_SparseRegressionBase):
    """Variational free-energy bound (Titsias, 2009)."""

    objective = "vfe"


class SparseVFE(VFE):
    """Spec-facing alias of the VFE engine."""


class InducingPointGP(FITC):
    """Spec name for the inducing-point (pseudo-input) regression approximation.

    Thin alias of :class:`FITC` — the Snelson & Ghahramani pseudo-point method that
    augments the effective noise with the low-rank residual ``diag(K - Qff)``.
    """


class SparseGaussianProcess(VFE):
    """Spec name for sparse GP regression via variational induction.

    Thin alias of :class:`VFE` — Titsias' SGPR bound, the standard "sparse GP".
    """
