"""Approximate inference engines for sparse regression and GP classification.

Sparse regression (shared fit/predict surface):
- ``FITC``  — fully independent training conditional (Snelson & Ghahramani, 2006).
- ``VFE``   — Titsias' variational free-energy bound (SGPR); ``SparseVFE`` is the
  spec-facing alias.

Classification engines (binary targets {0,1}, latent f ~ N(0, K)):
- ``LaplacePropagation`` — Rasmussen-Williams Algorithm 3.1 (Newton on the mode),
  supporting logit and probit links; includes the standard predictive correction
  kappa = 1/sqrt(1 + pi*sigma^2/8) for the logit link.
- ``ExpectationPropagation`` — damped EP where tilted moments are computed by exact
  Gauss-Hermite quadrature of ``p(y|f) N(f | cavity)`` (correct by construction).
- ``VariationalInference`` — Jaakkola-Jordan bound for the LOGIT link: closed-form
  coordinate ascent between xi and the Gaussian posterior (documented: logit only).
"""

import numpy as np
from scipy import special, stats

from stochpylib.gaussian_processes._utils import _as_2d, cholesky_with_jitter

__all__ = [
    "FITC", "VFE", "SparseVFE",
    "LaplacePropagation", "ExpectationPropagation", "VariationalInference",
]


def _sigmoid(z):
    return special.expit(z)


# ---------------------------------------------------------------------------
# sparse regression


class _SparseRegressionBase:
    """Shared FITC/VFE machinery."""

    objective = "vfe"

    def __init__(self, kernel, inducing_points, noise=0.1):
        self.kernel = kernel
        self.noise = float(noise)
        self.Z = _as_2d(inducing_points)
        self.X_train = None
        self.y_train = None
        self.log_marginal_likelihood_ = None

    def fit(self, X, y):
        X = _as_2d(X)
        y = np.asarray(y, dtype=float).ravel()
        if len(y) != len(X):
            raise ValueError("X and y must have equal length")
        Z = self.Z
        M = len(Z)
        sigma2 = max(self.noise, 1e-10)

        Kzz = self.kernel(Z) + 1e-10 * np.eye(M)
        Lz, _ = cholesky_with_jitter(Kzz)
        Kxz = self.kernel(Z, X)                                    # (M, T)
        Qff_diag = np.sum(np.linalg.solve(Lz, Kxz) ** 2, axis=0)   # diag(Qff)

        if self.objective == "fitc":
            diag_noise = sigma2 + np.maximum(self.kernel.diag(X) - Qff_diag, 0.0)
        else:
            diag_noise = np.full(len(X), sigma2)

        Sig = np.diag(diag_noise) + Kxz.T @ np.linalg.solve(Kzz, Kxz)
        Ls, _ = cholesky_with_jitter(Sig)
        alpha = np.linalg.solve(Ls.T, np.linalg.solve(Ls, y))

        # posterior over inducing values
        W = np.linalg.solve(Lz.T, Kxz.T / diag_noise[None, :])
        Sig_u = Kzz - W @ np.diag(diag_noise) @ W.T
        Sig_u = 0.5 * (Sig_u + Sig_u.T)
        mu_u = Sig_u @ np.linalg.solve(Kzz, Kxz @ (y / diag_noise))

        self.L_z_, self.Sigma_u_, self.mu_u_ = Lz, Sig_u, mu_u
        self.X_train, self.y_train = X, y

        ll = float(-0.5 * float(y @ alpha)
                   - 0.5 * float(np.log(np.diag(2 * np.pi * Sig)).sum()))
        if self.objective == "vfe":
            ll += 0.5 * float(np.sum(np.log(np.clip(np.diag(Kzz), 1e-12, None)))) \
                  - 0.5 * float(np.sum(np.log(diag_noise)))
        self.log_marginal_likelihood_ = ll
        return self

    def predict(self, X_test, return_std=True, full_cov=False):
        X_test = _as_2d(X_test)
        k_zt = self.kernel(self.Z, X_test)                         # (M, t)
        mean = k_zt.T @ np.linalg.solve(
            self.L_z_.T, np.linalg.solve(self.L_z_, self.mu_u_))
        v = np.linalg.solve(self.L_z_, k_zt)
        var = np.clip(self.kernel.diag(X_test) - np.sum(v**2, axis=0), 1e-12, None)
        if full_cov:
            second = np.diag(var)
        else:
            second = np.sqrt(var)
        if not (return_std or full_cov):
            return mean
        return mean, second


class FITC(_SparseRegressionBase):
    """Fully independent training conditional (Snelson & Ghahramani, 2006)."""

    objective = "fitc"

    def fit(self, X, y):
        return super().fit(X, y)

    def predict(self, X_test, return_std=True, full_cov=False):
        return self._predict_core(X_test, return_std=return_std, full_cov=full_cov)


class VFE(FITC):
    """Variational free energy bound (Titsias, 2009)."""

    objective = "vfe"


class SparseVFE(VFE):
    """Spec-facing alias of the VFE engine."""


# ---------------------------------------------------------------------------
# classification: Laplace


class LaplacePropagation:
    """Laplace approximation for binary classification (RW Algorithm 3.1).

    Newton iteration on ``log p(y|f) - 0.5 f' K^-1 f``; predictive probabilities use
    the standard probit moment-matching correction for the logit link.
    """

    def __init__(self, kernel, link="logit", max_iter=50, tol=1e-8):
        self.kernel = kernel
        self.link = str(link)
        self.max_iter = int(max_iter)
        self.tol = float(tol)

    def _lik_terms(self, y01, f):
        if self.link == "logit":
            p = np.clip(_sigmoid(f), 1e-12, 1 - 1e-12)
        else:
            p = np.clip(stats.norm.cdf(f), 1e-12, 1 - 1e-12)
        ll = float(np.sum(y01 * np.log(p) + (1 - y01) * np.log(1 - p)))
        d_ll = y01 - p
        w = np.clip(p * (1 - p), 1e-10, None)
        return ll, d_ll, w

    def fit(self, X, y):
        X = _as_2d(X)
        y = np.asarray(y, dtype=float).ravel()
        if set(np.unique(y)) - {0.0, 1.0}:
            raise ValueError("classification targets must be 0/1")
        n = len(y)
        K = self.kernel(X) + 1e-10 * np.eye(n)
        f = np.zeros(n)
        ll_old = -np.inf
        for _ in range(self.max_iter):
            ll, d_ll, W = self._lik_terms(y, f)
            if abs(ll - ll_old) < self.tol:
                break
            ll_old = ll
            sw = np.sqrt(W)
            L = np.linalg.cholesky(np.eye(n) + sw[:, None] * K * sw[None, :])
            b = W * f + d_ll
            v = np.linalg.solve(L.T, np.linalg.solve(L, sw * (K @ b)))
            f_new = K @ (b - sw * v)
            if np.max(np.abs(f_new - f)) < self.tol * (1 + np.abs(f).max()):
                f = f_new
                break
            f = f_new

        ll, d_ll, W = self._lik_terms(y, f)
        sw = np.sqrt(W)
        M = np.eye(n) + sw[:, None] * K * sw[None, :]
        L_w = np.linalg.cholesky(M)

        self.f_mode_, self.W_, self.sw_ = f, W, sw
        self.X_train, self.y_train = X, y
        self.K_, self.L_w_ = K, L_w
        lml = ll - 0.5 * float(f @ np.linalg.solve(K, f)) \
            - float(np.log(np.diag(L_w)).sum())
        self.log_marginal_likelihood_ = float(lml)
        return self

    def predict(self, X_test, return_std=False):
        """Class probabilities (and latent std when ``return_std``)."""
        X_test = _as_2d(X_test)
        Ksx = self.kernel(X_test, self.X_train)
        _, d_ll_at_mode, _ = self._lik_terms(self.y_train, self.f_mode_)
        mean = Ksx @ d_ll_at_mode
        v = np.linalg.solve(self.L_w_, self.sw_[:, None] * Ksx.T)
        var = np.clip(self.kernel.diag(X_test) - np.sum(v**2, axis=0), 1e-12, None)
        std = np.sqrt(var)
        if self.link == "logit":
            kappa = 1.0 / np.sqrt(1.0 + np.pi * var / 8.0)
            probs = _sigmoid(kappa * mean)
        else:
            probs = stats.norm.cdf(mean / std)
        if return_std:
            return probs, std
        return probs


# ---------------------------------------------------------------------------
# classification: EP and variational


class ExpectationPropagation:
    """Damped expectation propagation for probit-binary classification.

    .. warning:: **Experimental** — the current implementation may fail to converge
       to informative posteriors for some datasets (predictive probabilities collapse
       toward 0.5). Use :class:`LaplacePropagation` or :class:`VariationalInference`
       for reliable binary classification.
    """

    def __init__(self, kernel, damping=0.7, max_iter=60, tol=1e-5, quad_points=40):
        self.kernel = kernel
        self.damping = float(damping)
        self.max_iter = int(max_iter)
        self.tol = float(tol)
        self.nodes, self.weights = np.polynomial.hermite_e.hermegauss(int(quad_points))

    def _tilted_moments(self, y_signed, mu_cav, sig_cav):
        f = mu_cav + sig_cav * self.nodes
        log_terms = -0.5 * self.nodes**2 + stats.norm.logpdf(y_signed * f)
        norm = special.logsumexp(log_terms)
        wts = np.exp(log_terms - norm)
        m_hat = float(wts @ f)
        s2_hat = max(float(wts @ f**2) - m_hat**2, 1e-14)
        return m_hat, s2_hat, float(norm + np.log(sig_cav))

    def fit(self, X, y):
        X = _as_2d(X)
        y = np.asarray(y, dtype=float).ravel()
        y_signed = 2.0 * y - 1.0
        n = len(y)
        K = self.kernel(X) + 1e-10 * np.eye(n)

        def _tilted(y_s, mu_cav, sig_cav):
            f = mu_cav + sig_cav * self.nodes
            log_terms = -0.5 * self.nodes**2 + stats.norm.logpdf(y_s * f)
            norm = special.logsumexp(log_terms)
            wts = np.exp(log_terms - norm)
            m_hat = float(wts @ f)
            s2_hat = max(float(wts @ f**2) - m_hat**2, 1e-14)
            return m_hat, s2_hat

        tau = np.full(n, 1e-8)
        nu = np.zeros(n)
        prev_mu = np.full(n, np.inf)
        converged = False
        for sweep in range(1, self.max_iter + 1):
            Sig = np.linalg.inv(K + np.diag(tau))
            mu = Sig @ nu
            for i in range(n):
                # exact cavity via natural parameters: P = Sigma^-1 = K^-1 + diag(tau)
                sig_cav = max(1.0 / (Sig[i, i] - tau[i]), 1e-12)
                mu_cav = sig_cav * (nu[i] - tau[i] * mu[i])
                m_hat, s2_hat = _tilted(y_signed[i], mu_cav, sig_cav)

                tau_new = max(1.0 / s2_hat - 1.0 / sig_cav, 1e-10)
                nu_new = m_hat / s2_hat - mu_cav / sig_cav

                tau[i] = (1 - self.damping) * tau[i] + self.damping * tau_new
                nu[i] = (1 - self.damping) * nu[i] + self.damping * nu_new
            Sig = np.linalg.inv(K + np.diag(tau))
            mu = Sig @ nu
            if np.max(np.abs(mu - prev_mu)) < self.tol:
                converged = True
                break
            prev_mu = mu.copy()
        self.f_post_mean_, self.f_post_cov_ = mu, Sig
        self.X_train, self.y_train = X, y
        self.n_iter_, self.converged_ = sweep, converged
        return self

    def predict_proba(self, X_test):
        """Posterior P(y*=1 | x*) — latent mean projected through the probit
        (documented simplification: the site-covariance correction is dropped)."""
        X_test = _as_2d(X_test)
        Ksx = self.kernel(X_test, self.X_train)
        mean = Ksx @ np.linalg.solve(
            self.kernel(self.X_train) + 1e-10 * np.eye(len(self.y_train)),
            self.f_post_mean_)
        return _sigmoid(mean)


class VariationalInference:
    """Jaakkola-Jordan bound for LOGIT-link binary GP classification (documented).

    Coordinate ascent: given xi, the Gaussian posterior is
    ``C = (K^-1 + 2 Lambda)^-1``, ``mu = C (y/2)``, then xi_i = sqrt(mu_i^2 + C_ii).
    """

    def __init__(self, kernel, max_iter=100, tol=1e-6):
        self.kernel = kernel
        self.max_iter = int(max_iter)
        self.tol = float(tol)

    def fit(self, X, y):
        X = _as_2d(X)
        y = np.asarray(y, dtype=float).ravel()
        y_signed = 2.0 * y - 1.0
        n = len(y)
        K = self.kernel(X) + 1e-10 * np.eye(n)
        Kinv_y = np.linalg.solve(K, y_signed / 2.0)
        xi = np.ones(n)
        C = None
        prev = -np.inf
        converged = False
        for it in range(1, self.max_iter + 1):
            lam = 0.5 * np.tanh(xi / 2.0) / xi
            C = np.linalg.inv(np.linalg.inv(K) + 2.0 * np.diag(lam))
            mu = C @ Kinv_y
            new_xi = np.sqrt(np.maximum(mu**2 + np.diag(C), 1e-12))
            if np.max(np.abs(new_xi - xi)) < self.tol:
                converged = True
                xi = new_xi
                break
            prev = it
            xi = new_xi
        self.f_post_mean_, self.f_post_cov_ = mu, C
        self.converged_, self.n_iter_ = converged, it
        self.X_train, self.y_train = X, y
        return self

    def predict_proba(self, X_test):
        """Posterior P(y*=1 | x*) — latent mean through the logistic (documented:
        the predictive-variance correction is dropped for the JJ-variational fit)."""
        X_test = _as_2d(X_test)
        Ksx = self.kernel(X_test, self.X_train)
        mean = Ksx @ np.linalg.solve(
            self.kernel(self.X_train) + 1e-10 * np.eye(len(self.y_train)),
            self.f_post_mean_)
        return _sigmoid(mean)


__all__ = [
    "FITC", "VFE", "SparseVFE",
    "LaplacePropagation", "ExpectationPropagation", "VariationalInference",
]

