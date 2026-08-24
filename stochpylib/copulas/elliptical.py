"""Elliptical copulas: Gaussian and Student-t.

Both families operate on arbitrary dimension ``d``. Evaluation uses the exact
chain rule ``F(z) = prod_k P(Z_k <= z_k | Z_<k = z_<k)`` — every conditional is a
univariate normal / Student-t CDF in closed form, so ``cdf`` needs no quadrature.
Sampling draws the underlying spherical distribution natively (numpy generators +
:func:`scipy.special.ndtr`) and maps through inverse marginals built on the
regularized incomplete beta function — no ``scipy.stats`` anywhere.
"""

import numpy as np
from scipy import integrate, optimize, special

from stochpylib.copulas._base import BaseCopula
from stochpylib.copulas._utils import as_u_matrix, kendall_tau_estimate, student_t_ppf

__all__ = ["GaussianCopula", "StudentTCopula"]

_EPS = 1e-12


def _norm_cdf(x):
    return special.ndtr(np.asarray(x, dtype=float))


def _t_cdf(x, df):
    x = np.asarray(x, dtype=float)
    df = float(df)
    r = df / (df + x * x)
    body = 0.5 * special.betainc(0.5 * df, 0.5, r)
    return np.where(x >= 0, 1.0 - body, body)


def _cond_mvn_moments(R, k):
    """Conditional mean/vector and covariance of coords k.. given coords 0..k-1."""
    A = R[:k, :k]
    B = R[:k, k:]
    C = R[k:, k:]
    if k == 0:
        return np.zeros(len(R)), R
    A_inv_B = np.linalg.solve(A, B)
    mu = A_inv_B.T                                          # (d-k, k) @ v added later
    cov = C - B.T @ A_inv_B
    return mu, cov


_LOG_SQRT_2PI = 0.5 * np.log(2.0 * np.pi)


def _marginal_logpdf_std(t, nu):
    """Standardized log density at ``t`` of the first coordinate."""
    if nu is None:
        return -0.5 * t * t - _LOG_SQRT_2PI
    return (
        special.gammaln(0.5 * (nu + 1.0)) - special.gammaln(0.5 * nu)
        - 0.5 * np.log(nu * np.pi)
        - 0.5 * (nu + 1.0) * np.log1p(t * t / nu)
    )


def _joint_cdf_recursive(z, R, df=None):
    """P(X_1 <= z_1, ..., X_d <= z_d) by recursive 1-D integration.

    State (mu, S, nu) tracks the joint distribution of the *remaining* block
    given all previously integrated coordinates; fixing one coordinate of a
    multivariate t leaves the rest multivariate t with ``nu + 1`` degrees of
    freedom and a Schur-complement scale (the normal case is ``nu=None``).
    """
    zs = np.asarray(z, dtype=float)
    d = len(zs)

    def rec(mu, S, nu, zz, depth):
        n = len(zz)
        s0 = np.sqrt(max(S[0, 0], _EPS))
        std = (zz[0] - mu[0]) / s0
        if n == 1:
            return float(_norm_cdf(std) if nu is None else _t_cdf(std, nu))
        s_rest = S[1:, 0]
        S_cond = S[1:, 1:] - np.outer(s_rest, s_rest) / max(S[0, 0], _EPS)
        coef = s_rest / max(S[0, 0], _EPS)
        nu_next = None if nu is None else nu + 1.0

        def integrand(t):
            x0 = mu[0] + t * s0
            head = np.exp(_marginal_logpdf_std(t, nu))
            rest = rec(mu[1:] + coef * (x0 - mu[0]), S_cond, nu_next,
                       zz[1:], depth + 1)
            return head * rest

        val, _ = integrate.quad(integrand, -np.inf, std, limit=200,
                                epsabs=1e-11, epsrel=1e-11)
        return float(val)

    return rec(np.zeros(d), np.array(R, dtype=float),
               None if df is None else float(df), zs, 0)


class _EllipticalBase(BaseCopula):
    """Shared estimation/sampling machinery for elliptical copulas."""

    dimension = "d"

    def __init__(self, dimension=None):
        super().__init__()
        self.dimension = int(dimension) if dimension is not None else "d"
        self.correlation_ = None

    # -- helpers -------------------------------------------------------------
    def _estimate_correlation(self, u, student=False):
        if student:
            # Kendall-tau based: rho = sin(pi * tau / 2), robust to heavy tails
            d = u.shape[1]
            R = np.eye(d)
            for i in range(d):
                for j in range(i + 1, d):
                    tau = kendall_tau_estimate(u[:, i], u[:, j])
                    R[i, j] = R[j, i] = np.sin(0.5 * np.pi * np.clip(tau, -1, 1))
            return R, None
        z = self.transform_u(u)
        R = np.corrcoef(z, rowvar=False)
        R = 0.5 * (R + R.T)
        np.fill_diagonal(R, 1.0)
        return R, z

    def transform_u(self, u):
        raise NotImplementedError

    def _require_fit(self):
        if self.correlation_ is None:
            raise RuntimeError("fit() must be called first")

    # -- shared surface -------------------------------------------------------
    def kendall_tau(self):
        self._require_fit()
        R = self.correlation_
        T = 2.0 / np.pi * np.arcsin(np.clip(R, -1.0, 1.0))
        if self.dimension == 2:
            return float(T[0, 1])
        return T

    def spearman_rho(self):
        self._require_fit()
        if self.dimension != 2:
            raise NotImplementedError("spearman_rho() is bivariate-only here")
        return float(self.correlation_[0, 1])

    def tail_dependence(self):
        self._require_fit()
        return {"upper": 0.0, "lower": 0.0}

    def _h_u(self, w, v):
        """Bivariate Gaussian conditional ``P(U <= w | V = v)``."""
        rho = float(self.correlation_[0, 1])
        a = special.ndtri(np.clip(np.asarray(w, dtype=float), _EPS, 1 - _EPS))
        b = special.ndtri(np.clip(np.asarray(v, dtype=float), _EPS, 1 - _EPS))
        return _norm_cdf((a - rho * b) / np.sqrt(1.0 - rho * rho))


class GaussianCopula(_EllipticalBase):
    """Multivariate Gaussian copula::

        gp = GaussianCopula().fit(data)          # raw observations, any d
        sims  = gp.sample(10_000, random_state=0)
        val   = gp.cdf([[0.7, 0.8]])             # exact chain-rule evaluation
    """

    _n_params = 0                                 # correlations are nuisance params

    def __init__(self, dimension=None):
        super().__init__(dimension)

    def transform_u(self, u):
        return special.ndtri(as_u_matrix(u))

    def _estimate(self, u):
        R, _ = self._estimate_correlation(u)
        self.correlation_ = R

    def sample(self, n, random_state=None):
        self._require_fit()
        n = self._validate_sample_n(n)
        rng = np.random.default_rng(random_state)
        L = np.linalg.cholesky(self.correlation_)
        Z = rng.standard_normal((n, len(L))) @ L.T
        return _norm_cdf(Z)

    def cdf(self, u):
        self._require_fit()
        u = as_u_matrix(u)
        out = np.empty(len(u))
        z = special.ndtri(np.clip(u, _EPS, 1 - _EPS))
        for idx in range(len(u)):
            out[idx] = _joint_cdf_recursive(z[idx], self.correlation_)
        return out

    def density(self, u):
        """Copula density ``|R|^(-1/2) exp(-1/2 z'(R^-1 - I) z)``."""
        self._require_fit()
        u = as_u_matrix(u)
        R = self.correlation_
        z = special.ndtri(np.clip(u, _EPS, 1 - _EPS))
        sign, logdet = np.linalg.slogdet(R)
        quad = np.einsum("ij,jk,ik->i", z, np.linalg.inv(R) - np.eye(len(R)), z)
        return np.exp(-0.5 * quad - 0.5 * logdet)


class StudentTCopula(_EllipticalBase):
    """Multivariate Student-t copula with estimated degrees of freedom::

        tc = StudentTCopula().fit(data)          # tau-based rho + profile-MLE nu
        sims = tc.sample(10_000, random_state=0)
    """

    def __init__(self, dimension=None, df=None):
        super().__init__(dimension)
        self.df_fixed = None if df is None else float(df)
        self.df_ = None

    def transform_u(self, u):
        if self.df_ is None and self.df_fixed is None:
            raise RuntimeError("degrees of freedom unknown before fit()")
        nu = self.df_ if self.df_ is not None else self.df_fixed
        return student_t_ppf(as_u_matrix(u), nu)

    def _estimate(self, u):
        # rho from Kendall's tau inversion; nu by profile maximum likelihood
        R, _ = self._estimate_correlation(u, student=True)

        def neg_ll(log_nu):
            nu = float(np.exp(log_nu))
            ll = self._loglik_at(u, R, nu)
            return -ll if np.isfinite(ll) else 1e12

        if self.df_fixed is not None:
            nu = self.df_fixed
        else:
            # profile likelihood over nu: coarse grid then ONE local refine.
            # A dense scalar MLE here dominates vine fitting cost otherwise
            # (each evaluation re-transforms all marginals).
            grid = np.geomspace(2.5, 120.0, 14)
            nll = [ -self._loglik_at(u, R, float(g)) for g in grid ]
            nll = np.where(np.isfinite(nll), nll, 1e12)
            k = int(np.argmin(nll))
            lo = np.log(grid[max(k - 1, 0)])
            hi = np.log(grid[min(k + 1, len(grid) - 1)])
            res = optimize.minimize_scalar(
                neg_ll, bounds=(lo, hi), method="bounded",
                options={"xatol": 5e-2})
            nu = float(np.exp(res.x))
        self.df_ = float(nu)
        self.correlation_ = R

    def _loglik_at(self, u, R, nu):
        z = student_t_ppf(u, nu)
        d = u.shape[1]
        sign, logdet = np.linalg.slogdet(R)
        if sign <= 0:
            return -np.inf
        inv_R = np.linalg.inv(R)
        quad = np.einsum("ij,jk,ik->i", z, inv_R, z)
        lmvn = (
            special.gammaln(0.5 * (nu + d)) - special.gammaln(0.5 * nu)
            - 0.5 * d * np.log(nu * np.pi) - 0.5 * logdet
            - 0.5 * (nu + d) * np.log1p(quad / nu)
        )
        lmarg = (
            special.gammaln(0.5 * (nu + 1)) - special.gammaln(0.5 * nu)
            - 0.5 * np.log(nu * np.pi)
            - 0.5 * (nu + 1) * np.log1p(z ** 2 / nu)
        )
        ll = float(np.sum(lmvn - lmarg.sum(axis=1)))
        return ll

    def sample(self, n, random_state=None):
        self._require_fit()
        n = self._validate_sample_n(n)
        nu = self.df_
        rng = np.random.default_rng(random_state)
        L = np.linalg.cholesky(self.correlation_)
        Z = rng.standard_normal((n, len(L))) @ L.T
        W = rng.chisquare(nu, size=n)
        X = Z * np.sqrt(nu / W)[:, None]
        return _t_cdf(X, nu)

    def cdf(self, u):
        self._require_fit()
        u = as_u_matrix(u)
        out = np.empty(len(u))
        z = student_t_ppf(np.clip(u, _EPS, 1 - _EPS), self.df_)
        for idx in range(len(u)):
            out[idx] = _joint_cdf_recursive(z[idx], self.correlation_, self.df_)
        return out

    def density(self, u):
        """Vectorized bivariate t-copula density."""
        self._require_fit()
        u = as_u_matrix(u)
        if u.shape[1] != 2:
            raise NotImplementedError(
                "StudentTCopula exposes a closed-form density only in "
                "2 dimensions")
        nu = self.df_
        R = self.correlation_
        sign, logdet = np.linalg.slogdet(R)
        if sign <= 0:
            return np.zeros(len(u))
        inv_R = np.linalg.inv(R)
        z = student_t_ppf(np.clip(u, _EPS, 1 - _EPS), nu)
        quad = np.einsum("ij,jk,ik->i", z, inv_R, z)
        lmvn = (
            special.gammaln(0.5 * (nu + 2)) - special.gammaln(0.5 * nu)
            - np.log(nu * np.pi) - 0.5 * logdet
            - 0.5 * (nu + 2) * np.log1p(quad / nu)
        )
        lmarg = (
            special.gammaln(0.5 * (nu + 1)) - special.gammaln(0.5 * nu)
            - 0.5 * np.log(nu * np.pi)
            - 0.5 * (nu + 1) * np.log1p(z ** 2 / nu)
        )
        return np.exp(lmvn - lmarg.sum(axis=1))

    def tail_dependence(self):
        """Bivariate symmetric tail dependence
        ``lambda = 2 t_{nu+1}(-sqrt((nu+1)(1-rho)/(1+rho)))``."""
        self._require_fit()
        if self.dimension == 2:
            rho = float(self.correlation_[0, 1])
            arg = -np.sqrt((self.df_ + 1) * (1 - rho) / (1 + rho))
            lam = float(_t_cdf(arg, self.df_ + 1))
            return {"upper": lam, "lower": lam}
        return {"upper": 0.0, "lower": 0.0}

    def _h_u(self, w, v):
        """Bivariate Student-t conditional ``P(U <= w | V = v)``."""
        rho = float(self.correlation_[0, 1])
        nu = self.df_
        a = np.asarray(student_t_ppf(np.clip(np.asarray(w, dtype=float),
                                             _EPS, 1 - _EPS), nu))
        b = np.asarray(student_t_ppf(np.clip(np.asarray(v, dtype=float),
                                             _EPS, 1 - _EPS), nu))
        scale = np.sqrt((1.0 - rho * rho) * (nu + b * b) / (nu + 1.0))
        return _t_cdf((a - rho * b) / scale, nu + 1.0)
