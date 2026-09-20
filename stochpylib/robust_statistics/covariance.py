"""Robust covariance / scatter estimation.

``MCD``/``MVE`` are the classic high-breakdown (up to 50%) resampling
estimators (Rousseeuw 1985); ``OGK`` is a fast pairwise-orthogonalization
alternative (Maronna & Zamar 2002); ``RobustCovariance``/``RobustCorrelation``
dispatch over all of them; ``CovShrinkage`` covers Ledoit-Wolf/OAS/
constant-correlation shrinkage toward a structured target.
"""

import itertools
import math

import numpy as np
from scipy import special

from stochpylib.robust_statistics._base import RobustCovarianceEstimator
from stochpylib.robust_statistics._common import (
    _as_2d, _chi2_consistency, _chi2_ppf, _cholesky_psd, _mahalanobis, _mad,
    _rng, _tau_scale,
)

__all__ = ["RobustCovariance", "MCD", "MVE", "OGK", "RobustCorrelation", "CovShrinkage"]


# --------------------------------------------------------------------- MCD

class MCD(RobustCovarianceEstimator):
    """Minimum Covariance Determinant (Rousseeuw & Van Driessen 1999,
    "FAST-MCD"): the h-subset (``h ~ n/2`` by default) with minimal
    covariance determinant, consistency-corrected and (by default)
    reweighted.
    """

    def __init__(self, support_fraction=None, n_trials=500, n_keep=10, n_csteps=2,
                reweight=True, random_state=None):
        self.support_fraction = support_fraction
        self.n_trials = int(n_trials)
        self.n_keep = int(n_keep)
        self.n_csteps = int(n_csteps)
        self.reweight = bool(reweight)
        self.random_state = random_state

    def fit(self, X):
        X = _as_2d(X)
        n, p = X.shape
        self._X = X
        h = ((n + p + 1) // 2 if self.support_fraction is None
            else int(np.floor(self.support_fraction * n)))
        h = max(h, p + 1)
        if h > n:
            raise ValueError("support_fraction too large for this sample size")
        rng = _rng(self.random_state)

        total = special.comb(n, h, exact=True) if n >= h else 0
        candidates = []
        if 0 < total <= 5000:
            for rows in itertools.combinations(range(n), h):
                rows = np.array(rows)
                loc0, cov0 = _mean_cov(X[rows])
                loc, cov, subset, det = _mcd_c_steps(X, h, loc0, cov0, max_iter=100)
                candidates.append((det, loc, cov, subset))
            self.exhaustive_ = True
        else:
            starts = []
            for _ in range(self.n_trials):
                rows = rng.choice(n, size=p + 1, replace=False)
                loc0, cov0 = _mean_cov(X[rows])
                if not np.all(np.isfinite(cov0)) or np.linalg.matrix_rank(cov0) < p:
                    continue
                loc, cov, subset, det = _mcd_c_steps(X, h, loc0, cov0, max_iter=self.n_csteps)
                starts.append((det, loc, cov, subset))
            starts.sort(key=lambda t: t[0])
            for det, loc, cov, subset in starts[:self.n_keep]:
                loc, cov, subset, det = _mcd_c_steps(X, h, loc, cov, max_iter=100)
                candidates.append((det, loc, cov, subset))
            self.exhaustive_ = False

        if not candidates:
            raise ValueError("MCD found no valid h-subset (data may be degenerate)")
        candidates.sort(key=lambda t: t[0])
        det, raw_loc, raw_cov, subset = candidates[0]

        c = _chi2_consistency(h / n, p)
        self.raw_location_ = raw_loc
        self.raw_covariance_ = c * raw_cov
        self.raw_support_mask_ = np.zeros(n, dtype=bool)
        self.raw_support_mask_[subset] = True
        self.dist_ = _mahalanobis(X, self.raw_location_, self.raw_covariance_)
        self.h_ = h

        if self.reweight:
            cutoff = _chi2_ppf(0.975, p)
            w = self.dist_ <= cutoff
            if int(np.sum(w)) < p + 1:
                w = self.raw_support_mask_
            loc_r, cov_r = _mean_cov(X[w])
            c_r = _chi2_consistency(0.975, p)
            self.location_ = loc_r
            self.covariance_ = c_r * cov_r
            self.support_mask_ = w
        else:
            self.location_ = self.raw_location_
            self.covariance_ = self.raw_covariance_
            self.support_mask_ = self.raw_support_mask_
        return self


def _mean_cov(rows):
    loc = np.mean(rows, axis=0)
    diff = rows - loc
    cov = (diff.T @ diff) / rows.shape[0]
    return loc, cov


def _mcd_c_steps(X, h, loc, cov, max_iter=50):
    n = X.shape[0]
    det_prev = np.inf
    subset = None
    for _ in range(max_iter):
        try:
            L = _cholesky_psd(cov)
        except np.linalg.LinAlgError:
            break
        d2 = _mahalanobis(X, loc, cov)
        subset = np.argsort(d2)[:h]
        loc, cov = _mean_cov(X[subset])
        sign, logdet = np.linalg.slogdet(cov)
        det = logdet if sign > 0 else np.inf
        if det >= det_prev - 1e-12:
            break
        det_prev = det
    if subset is None:
        d2 = _mahalanobis(X, loc, cov)
        subset = np.argsort(d2)[:h]
    sign, logdet = np.linalg.slogdet(cov)
    det = float(logdet) if sign > 0 else np.inf
    return loc, cov, subset, det


# --------------------------------------------------------------------- MVE

class MVE(RobustCovarianceEstimator):
    """Minimum Volume Ellipsoid (Rousseeuw & Leroy 1987): the (p+1)-point
    subset minimizing the volume of the ellipsoid covering its own h-th
    smallest Mahalanobis distance."""

    def __init__(self, support_fraction=None, n_trials=2000, reweight=True, random_state=None):
        self.support_fraction = support_fraction
        self.n_trials = int(n_trials)
        self.reweight = bool(reweight)
        self.random_state = random_state

    def fit(self, X):
        X = _as_2d(X)
        n, p = X.shape
        self._X = X
        h = ((n + p + 1) // 2 if self.support_fraction is None
            else int(np.floor(self.support_fraction * n)))
        h = max(h, p + 1)
        rng = _rng(self.random_state)

        total = special.comb(n, p + 1, exact=True) if n >= p + 1 else 0
        if 0 < total <= 5000:
            subsets = np.array(list(itertools.combinations(range(n), p + 1)))
            self.exhaustive_ = True
        else:
            subsets = np.array([np.sort(rng.choice(n, size=p + 1, replace=False))
                                for _ in range(self.n_trials)])
            self.exhaustive_ = False

        best_vol = np.inf
        best = None
        for rows in subsets:
            m_J, C_J = _mean_cov(X[rows])
            try:
                sign, logdet_C = np.linalg.slogdet(C_J)
            except np.linalg.LinAlgError:
                continue
            if sign <= 0:
                continue
            d2 = _mahalanobis(X, m_J, C_J)
            m2 = float(np.sort(d2)[h - 1])
            if m2 <= 0:
                continue
            log_vol = 0.5 * logdet_C + p * math.log(m2)
            if log_vol < best_vol:
                best_vol = log_vol
                best = (m_J, C_J, m2)

        if best is None:
            raise ValueError("MVE found no valid (p+1)-point subset")
        m_J, C_J, m2 = best
        q = h / n
        chi2_q = _chi2_ppf(q, p)
        self.raw_location_ = m_J
        self.raw_covariance_ = (m2 / chi2_q) * C_J if chi2_q > 0 else C_J
        self.dist_ = _mahalanobis(X, self.raw_location_, self.raw_covariance_)
        cutoff_raw = float(np.sort(self.dist_)[h - 1])
        self.raw_support_mask_ = self.dist_ <= cutoff_raw

        if self.reweight:
            cutoff = _chi2_ppf(0.975, p)
            w = self.dist_ <= cutoff
            if int(np.sum(w)) < p + 1:
                w = self.raw_support_mask_
            loc_r, cov_r = _mean_cov(X[w])
            c_r = _chi2_consistency(0.975, p)
            self.location_ = loc_r
            self.covariance_ = c_r * cov_r
            self.support_mask_ = w
        else:
            self.location_ = self.raw_location_
            self.covariance_ = self.raw_covariance_
            self.support_mask_ = self.raw_support_mask_
        return self


# --------------------------------------------------------------------- OGK

class OGK(RobustCovarianceEstimator):
    """Orthogonalized Gnanadesikan-Kettenring covariance (Maronna & Zamar
    2002): pairwise robust covariances via a univariate robust scale of
    sums/differences, orthogonalized and iterated."""

    def __init__(self, scale="tau", n_iter=2, reweight=True, beta=0.9):
        if scale not in ("tau", "qn", "mad"):
            raise ValueError("scale must be 'tau', 'qn', or 'mad'")
        self.scale = scale
        self.n_iter = int(n_iter)
        self.reweight = bool(reweight)
        self.beta = float(beta)

    def _loc_scale(self, v):
        if self.scale == "tau":
            return _tau_scale(v)
        if self.scale == "mad":
            return float(np.median(v)), _mad(v, normal=True)
        from stochpylib.robust_statistics.scale import Qn_Estimator
        loc = float(np.median(v))
        sc = Qn_Estimator(n_boot=0).fit(v).estimate_
        return loc, sc

    def _one_pass(self, Y):
        """One Gnanadesikan-Kettenring orthogonalization pass: robust
        per-column loc/scale, pairwise GK covariance via sums/differences,
        eigendecomposition. Returns ``(loc, cov, M, Zc)`` in ``Y``'s own
        coordinates, where ``M`` maps the centered rotated residuals ``Zc``
        back to ``Y - loc`` (i.e. ``Y - loc ~= Zc @ M.T``), so a further pass
        on ``Zc`` composes as ``loc += M @ loc2``, ``cov = M @ cov2 @ M.T``.
        """
        n, p = Y.shape
        m = np.empty(p)
        s = np.empty(p)
        for j in range(p):
            m[j], s[j] = self._loc_scale(Y[:, j])
        s = np.where(s > 0, s, 1.0)
        Ystd = (Y - m) / s
        U = np.eye(p)
        for j in range(p):
            for k in range(j + 1, p):
                _, s_plus = self._loc_scale(Ystd[:, j] + Ystd[:, k])
                _, s_minus = self._loc_scale(Ystd[:, j] - Ystd[:, k])
                U[j, k] = U[k, j] = 0.25 * (s_plus ** 2 - s_minus ** 2)
        _, E = np.linalg.eigh(U)
        Z = Ystd @ E
        m_z = np.empty(p)
        s_z = np.empty(p)
        for k in range(p):
            m_z[k], s_z[k] = self._loc_scale(Z[:, k])
        M = s[:, None] * E
        loc = m + M @ m_z
        cov = M @ np.diag(s_z ** 2) @ M.T
        return loc, cov, M, Z - m_z

    def fit(self, X):
        X = _as_2d(X)
        n, p = X.shape
        self._X = X
        loc, cov, M, Zc = self._one_pass(X)
        for _ in range(max(self.n_iter - 1, 0)):
            loc2, cov2, M2, Zc2 = self._one_pass(Zc)
            loc = loc + M @ loc2
            cov = M @ cov2 @ M.T
            M = M @ M2
            Zc = Zc2
        self.raw_location_ = loc
        self.raw_covariance_ = cov
        self.dist_ = _mahalanobis(X, self.raw_location_, self.raw_covariance_)

        if self.reweight:
            med_d2 = float(np.median(self.dist_))
            cutoff = _chi2_ppf(self.beta, p) * med_d2 / _chi2_ppf(0.5, p) if med_d2 > 0 else np.inf
            w = self.dist_ <= cutoff
            if int(np.sum(w)) < p + 1:
                w = np.ones(n, dtype=bool)
            loc_r, cov_r = _mean_cov(X[w])
            # truncating to the central `beta` fraction of a chi2_p-distributed
            # squared distance shrinks the sample covariance; the same
            # consistency correction MCD/MVE use undoes it.
            c_r = _chi2_consistency(self.beta, p)
            self.location_ = loc_r
            self.covariance_ = c_r * cov_r
            self.weights_ = w
        else:
            self.location_ = self.raw_location_
            self.covariance_ = self.raw_covariance_
            self.weights_ = None
        return self


# ------------------------------------------------------------ dispatchers

class RobustCovariance(RobustCovarianceEstimator):
    """Dispatcher: ``method`` is ``"mcd"``, ``"mve"``, ``"ogk"``, ``"huber"``
    (Maronna-type M-estimator of scatter), or ``"sample"`` (a plain
    baseline)."""

    def __init__(self, method="mcd", **kwargs):
        if method not in ("mcd", "mve", "ogk", "huber", "sample"):
            raise ValueError("method must be 'mcd', 'mve', 'ogk', 'huber', or 'sample'")
        self.method = method
        self.kwargs = kwargs

    def fit(self, X):
        X = _as_2d(X)
        self._X = X
        if self.method == "sample":
            self.location_ = np.mean(X, axis=0)
            self.covariance_ = np.cov(X, rowvar=False, ddof=1)
            self.estimator_ = None
        elif self.method == "huber":
            self.location_, self.covariance_ = _huber_scatter(X, **self.kwargs)
            self.estimator_ = None
        else:
            cls = {"mcd": MCD, "mve": MVE, "ogk": OGK}[self.method]
            est = cls(**self.kwargs).fit(X)
            self.estimator_ = est
            self.location_ = est.location_
            self.covariance_ = est.covariance_
        return self


def _huber_scatter(X, c=None, tol=1e-8, max_iter=100):
    """Maronna-type M-estimator of multivariate location/scatter with a
    Huber-type weight, scaled to be Fisher-consistent at the Gaussian via a
    1-D quadrature of the weight-times-distance-squared function."""
    n, p = X.shape
    c = float(np.sqrt(_chi2_ppf(0.95, p))) if c is None else float(c)
    loc = np.median(X, axis=0)
    cov = np.cov(X, rowvar=False, ddof=1)
    for _ in range(max_iter):
        try:
            d2 = _mahalanobis(X, loc, cov)
        except np.linalg.LinAlgError:
            break
        d = np.sqrt(np.clip(d2, 1e-300, None))
        w = np.minimum(1.0, c / d)
        sw = np.sum(w)
        loc_new = np.sum(w[:, None] * X, axis=0) / sw
        diff = X - loc_new
        cov_new = (diff * w[:, None]).T @ diff / n
        if np.max(np.abs(loc_new - loc)) < tol * max(1.0, np.max(np.abs(loc))):
            loc, cov = loc_new, cov_new
            break
        loc, cov = loc_new, cov_new
    # Fisher consistency at the Gaussian: divide by E[w(||Z||) ||Z||^2 / p]
    from scipy import integrate

    def chi_pdf(r):
        return (2.0 ** (1 - p / 2.0) / special.gamma(p / 2.0)) * r ** (p - 1) * np.exp(-0.5 * r * r)

    def integrand(r):
        w_r = min(1.0, c / r) if r > 0 else 1.0
        return w_r * r * r * chi_pdf(r)

    ez, _ = integrate.quad(integrand, 0, 50)
    factor = ez / p if ez > 0 else 1.0
    return loc, cov / factor


class RobustCorrelation:
    """Robust correlation matrix (or a single pairwise correlation via
    ``pairwise(x, y)``). ``method``: ``"mcd"``/``"ogk"``/``"huber"``
    (covariance-implied), ``"spearman"``, ``"kendall"``, ``"gaussian_rank"``,
    ``"quadrant"``.
    """

    _COV_METHODS = ("mcd", "ogk", "huber")
    _RANK_METHODS = ("spearman", "kendall", "gaussian_rank", "quadrant")

    def __init__(self, method="mcd", **kwargs):
        if method not in self._COV_METHODS + self._RANK_METHODS:
            raise ValueError(f"method must be one of {self._COV_METHODS + self._RANK_METHODS}")
        self.method = method
        self.kwargs = kwargs

    def fit(self, X):
        X = _as_2d(X)
        n, p = X.shape
        if self.method in self._COV_METHODS:
            cov = RobustCovariance(method=self.method, **self.kwargs).fit(X)
            d = np.sqrt(np.diag(cov.covariance_))
            outer_d = np.outer(d, d)
            self.correlation_ = np.where(outer_d > 0, cov.covariance_ / np.where(outer_d == 0, 1, outer_d), 0.0)
        else:
            R = np.eye(p)
            for i in range(p):
                for j in range(i + 1, p):
                    R[i, j] = R[j, i] = self.pairwise(X[:, i], X[:, j])
            self.correlation_ = R
        return self

    def pairwise(self, x, y):
        x = np.asarray(x, dtype=float).ravel()
        y = np.asarray(y, dtype=float).ravel()
        if self.method == "spearman":
            from stochpylib.copulas._utils import spearman_rho_estimate
            return float(spearman_rho_estimate(x, y))
        if self.method == "kendall":
            from stochpylib.copulas._utils import kendall_tau_estimate
            return float(kendall_tau_estimate(x, y))
        if self.method == "gaussian_rank":
            from stochpylib.copulas._utils import _rank_ties
            from stochpylib.robust_statistics._common import _norm_ppf
            n = len(x)
            zx = _norm_ppf(_rank_ties(x) / (n + 1.0))
            zy = _norm_ppf(_rank_ties(y) / (n + 1.0))
            return float(np.corrcoef(zx, zy)[0, 1])
        if self.method == "quadrant":
            mx, my = np.median(x), np.median(y)
            return float(np.sin(0.5 * np.pi * np.mean(np.sign(x - mx) * np.sign(y - my))))
        raise ValueError(f"pairwise is not defined for method {self.method!r}")


# --------------------------------------------------------------------- shrinkage

class CovShrinkage:
    """Shrinkage covariance estimator: ``covariance_ = (1-delta)*S + delta*F``.

    ``method``: ``"ledoit_wolf"`` (analytic shrinkage intensity toward
    ``target``), ``"oas"`` (Oracle Approximating Shrinkage, identity target
    only), ``"fixed"`` (use ``shrinkage``). ``target``: ``"identity"``
    (``mu_bar * I``), ``"diagonal"`` (``diag(S)``, Schafer-Strimmer),
    ``"constant_correlation"`` (Ledoit-Wolf 2003).
    """

    def __init__(self, method="ledoit_wolf", target="identity", shrinkage=None, ddof=0):
        if method not in ("ledoit_wolf", "oas", "fixed"):
            raise ValueError("method must be 'ledoit_wolf', 'oas', or 'fixed'")
        if target not in ("identity", "diagonal", "constant_correlation"):
            raise ValueError("target must be 'identity', 'diagonal', or 'constant_correlation'")
        self.method = method
        self.target = target
        self.shrinkage = shrinkage
        self.ddof = int(ddof)

    def fit(self, X):
        X = _as_2d(X)
        n, p = X.shape
        Xc = X - X.mean(axis=0)
        S = (Xc.T @ Xc) / (n - self.ddof)
        self.sample_covariance_ = S

        if self.method == "fixed":
            delta = float(self.shrinkage)
            F = self._target_matrix(S, Xc, n)
        elif self.method == "oas":
            F = (np.trace(S) / p) * np.eye(p)
            trS2 = float(np.trace(S @ S))
            trS_sq = float(np.trace(S) ** 2)
            num = (1.0 - 2.0 / p) * trS2 + trS_sq
            den = (n + 1.0 - 2.0 / p) * (trS2 - trS_sq / p)
            delta = num / den if den > 0 else 1.0
            delta = float(np.clip(delta, 0.0, 1.0))
        else:  # ledoit_wolf
            F = self._target_matrix(S, Xc, n)
            delta = self._lw_delta(S, F, Xc, n, p)

        self.target_ = F
        self.shrinkage_ = delta
        self.covariance_ = (1.0 - delta) * S + delta * F
        return self

    def _target_matrix(self, S, Xc, n):
        p = S.shape[0]
        if self.target == "identity":
            mu_bar = np.trace(S) / p
            return mu_bar * np.eye(p)
        if self.target == "diagonal":
            return np.diag(np.diag(S))
        # constant_correlation (Ledoit-Wolf 2003)
        d = np.sqrt(np.clip(np.diag(S), 1e-300, None))
        R = S / np.outer(d, d)
        r_bar = (np.sum(R) - p) / (p * (p - 1)) if p > 1 else 0.0
        F = r_bar * np.outer(d, d)
        np.fill_diagonal(F, np.diag(S))
        return F

    def _lw_delta(self, S, F, Xc, n, p):
        if self.target == "identity":
            mu_bar = np.trace(S) / p
            delta2 = float(np.sum((S - F) ** 2))
            pi_hat = 0.0
            for t in range(n):
                outer_t = np.outer(Xc[t], Xc[t])
                pi_hat += np.sum((outer_t - S) ** 2)
            pi_hat /= n * n
            beta2 = min(pi_hat, delta2)
            return float(np.clip(beta2 / delta2 if delta2 > 0 else 0.0, 0.0, 1.0))
        if self.target == "diagonal":
            var_hat = 0.0
            for t in range(n):
                outer_t = np.outer(Xc[t], Xc[t])
                var_hat += np.sum((outer_t - S) ** 2)
            # Schafer-Strimmer: only off-diagonal variance contributes
            off = ~np.eye(p, dtype=bool)
            num = 0.0
            for t in range(n):
                w_t = np.outer(Xc[t], Xc[t])
                num += np.sum(((w_t - S) ** 2)[off])
            num *= n / (n - 1.0) ** 3
            den = np.sum(((S - F) ** 2)[off])
            return float(np.clip(num / den if den > 0 else 0.0, 0.0, 1.0))
        # constant_correlation (Ledoit & Wolf 2003, "Honey I shrunk...")
        d = np.sqrt(np.clip(np.diag(S), 1e-300, None))
        R = S / np.outer(d, d)
        r_bar = (np.sum(R) - p) / (p * (p - 1)) if p > 1 else 0.0
        pi_mat = np.zeros((p, p))
        for t in range(n):
            w_t = np.outer(Xc[t], Xc[t])
            pi_mat += (w_t - S) ** 2
        pi_mat /= n
        pi_hat = float(np.sum(pi_mat))
        rho_hat = 0.0
        for i in range(p):
            for j in range(p):
                if i == j:
                    continue
                theta_ii = np.mean([(Xc[t, i] ** 2 - S[i, i]) * (Xc[t, i] * Xc[t, j] - S[i, j]) for t in range(n)])
                theta_jj = np.mean([(Xc[t, j] ** 2 - S[j, j]) * (Xc[t, i] * Xc[t, j] - S[i, j]) for t in range(n)])
                rho_hat += (d[j] / d[i]) * theta_ii + (d[i] / d[j]) * theta_jj
        rho_hat *= r_bar / 2.0
        gamma_hat = float(np.sum((F - S) ** 2))
        kappa_hat = (pi_hat - rho_hat) / gamma_hat if gamma_hat > 0 else 0.0
        return float(np.clip(kappa_hat / n, 0.0, 1.0))
