"""Empirical distribution/CDF/characteristic-function estimators, the
Glivenko-Cantelli uniform-convergence bound, and Owen's empirical likelihood.
"""

import numpy as np
from scipy import optimize

from stochpylib.nonparametric._base import NonparametricDensity
from stochpylib.nonparametric._common import _dkw_band, _el_logstar, _el_logstar1, _el_logstar2
from stochpylib.statistics._result import TestResult

__all__ = ["EmpiricalDistribution", "EmpiricalCDF", "EmpiricalCharFn",
           "GlivenkoCantelli", "EmpiricalLikelihood"]


class EmpiricalDistribution(NonparametricDensity):
    """The empirical distribution: a discrete atom of mass ``w_i`` at each
    data point (equal weights by default). Full distribution contract:
    ``pmf`` = per-atom weight, step ``cdf``, order-statistic ``ppf``,
    weighted-resample ``rvs``, closed-form ``mean``/``var``/``mgf``/``cf``,
    atom ``entropy``.
    """

    is_discrete = True

    def __init__(self, weights=None):
        self.weights = weights

    def _fit(self, x):
        order = np.argsort(x)
        self._x_sorted = x[order]
        n = len(x)
        w = np.full(n, 1.0 / n) if self.weights is None else \
            np.asarray(self.weights, dtype=float)[order]
        if self.weights is not None:
            w = w / w.sum()
        self._w_sorted = w
        self._cumw = np.cumsum(w)
        return {}

    def support(self):
        return (float(self._x_sorted[0]), float(self._x_sorted[-1]))

    def pmf(self, x):
        scalar = np.asarray(x).ndim == 0
        x = np.atleast_1d(np.asarray(x, dtype=float))
        out = np.zeros(len(x))
        for i, xi in enumerate(x):
            match = self._x_sorted == xi
            if np.any(match):
                out[i] = self._w_sorted[match].sum()
        return float(out[0]) if scalar else out

    pdf = pmf

    def cdf(self, x):
        scalar = np.asarray(x).ndim == 0
        x = np.atleast_1d(np.asarray(x, dtype=float))
        idx = np.searchsorted(self._x_sorted, x, side="right") - 1
        out = np.where(idx >= 0, self._cumw[np.clip(idx, 0, None)], 0.0)
        return float(out[0]) if scalar else out

    def ppf(self, q):
        scalar = np.asarray(q).ndim == 0
        q = np.atleast_1d(np.asarray(q, dtype=float))
        idx = np.searchsorted(self._cumw, q, side="left")
        idx = np.clip(idx, 0, len(self._x_sorted) - 1)
        out = self._x_sorted[idx]
        return float(out[0]) if scalar else out

    def rvs(self, size=1, random_state=None):
        rng = np.random.default_rng(random_state)
        return rng.choice(self._x_sorted, size=size, replace=True, p=self._w_sorted)

    def mean(self):
        return float(np.sum(self._x_sorted * self._w_sorted))

    def var(self):
        mu = self.mean()
        return float(np.sum(self._w_sorted * (self._x_sorted - mu) ** 2))

    def entropy(self):
        p = self._w_sorted[self._w_sorted > 0]
        return float(-np.sum(p * np.log(p)))

    def mgf(self, t):
        return float(np.sum(self._w_sorted * np.exp(t * self._x_sorted)))

    def cf(self, t):
        return complex(np.sum(self._w_sorted * np.exp(1j * t * self._x_sorted)))

    def confidence_band(self, level=0.95):
        """DKW simultaneous confidence band half-width for the CDF."""
        return _dkw_band(len(self._x_sorted), level)

    def ks_test(self, data):
        data = np.sort(np.asarray(data, dtype=float))
        n = len(data)
        cdf_vals = self.cdf(data)
        i = np.arange(1, n + 1)
        d_plus = np.max(i / n - cdf_vals)
        d_minus = np.max(cdf_vals - (i - 1) / n)
        d_stat = max(d_plus, d_minus)
        lam = max((np.sqrt(n) + 0.12 + 0.11 / np.sqrt(n)) * d_stat, 0.0)
        p = 2.0 * sum((-1) ** (k - 1) * np.exp(-2.0 * k ** 2 * lam ** 2) for k in range(1, 101))
        return float(d_stat), float(np.clip(p, 0.0, 1.0))


class EmpiricalCDF:
    """A fluent empirical CDF: ``fit(x)`` then call, or use
    ``evaluate``/``quantile``/``std_error``/``confidence_band``.
    """

    def fit(self, x):
        x = np.asarray(x, dtype=float).ravel()
        self.sample_points_ = np.sort(x)
        self.n_ = len(x)
        return self

    def evaluate(self, t):
        scalar = np.asarray(t).ndim == 0
        t = np.atleast_1d(np.asarray(t, dtype=float))
        out = np.searchsorted(self.sample_points_, t, side="right") / self.n_
        return float(out[0]) if scalar else out

    __call__ = evaluate

    def quantile(self, q):
        return np.quantile(self.sample_points_, q, method="inverted_cdf")

    def std_error(self, t):
        f = np.asarray(self.evaluate(t), dtype=float)
        return np.sqrt(f * (1.0 - f) / self.n_)

    def confidence_band(self, level=0.95):
        """Dvoretzky-Kiefer-Wolfowitz simultaneous half-width: with probability
        >= ``level``, ``|F_n(t) - F(t)| <= eps`` for every ``t`` at once."""
        return _dkw_band(self.n_, level)

    def __repr__(self):
        return f"EmpiricalCDF(n={getattr(self, 'n_', 0)})"


class EmpiricalCharFn:
    """Empirical characteristic function ``phi_n(t) = mean(exp(i t x))``."""

    def fit(self, x):
        self.data_ = np.asarray(x, dtype=float).ravel()
        self.n_ = len(self.data_)
        return self

    def evaluate(self, t):
        scalar = np.asarray(t).ndim == 0
        t = np.atleast_1d(np.asarray(t, dtype=float))
        out = np.array([np.mean(np.exp(1j * ti * self.data_)) for ti in t])
        return out[0] if scalar else out

    __call__ = evaluate

    def modulus(self, t):
        return np.abs(self.evaluate(t))

    def std_error(self, t):
        scalar = np.asarray(t).ndim == 0
        phi = np.abs(np.atleast_1d(self.evaluate(t)))
        se = np.sqrt(np.clip(1.0 - phi ** 2, 0.0, None) / self.n_)
        return float(se[0]) if scalar else se

    def distance(self, dist_or_cf, t_grid):
        """Integrated squared modulus deviation from a reference cf over ``t_grid``
        (a ``Distribution`` instance with ``.cf()``, or a plain callable)."""
        cf_fn = dist_or_cf.cf if hasattr(dist_or_cf, "cf") else dist_or_cf
        phi_n = np.array([self.evaluate(t) for t in t_grid])
        phi_ref = np.array([cf_fn(t) for t in t_grid])
        diff2 = np.abs(phi_n - phi_ref) ** 2
        return float(np.trapezoid(diff2, t_grid))

    def __repr__(self):
        return f"EmpiricalCharFn(n={getattr(self, 'n_', 0)})"


class GlivenkoCantelli:
    """The Glivenko-Cantelli sup-norm distance between the empirical CDF and a
    reference CDF, plus the DKW simultaneous confidence bound.
    """

    def fit(self, x, cdf):
        x = np.sort(np.asarray(x, dtype=float).ravel())
        self.n_ = len(x)
        cdf_fn = cdf.cdf if hasattr(cdf, "cdf") else cdf
        F = np.asarray([cdf_fn(xi) for xi in x], dtype=float)
        i = np.arange(1, self.n_ + 1)
        d_plus = np.max(i / self.n_ - F)
        d_minus = np.max(F - (i - 1) / self.n_)
        self.distance_ = float(max(d_plus, d_minus))
        return self

    def dkw_bound(self, level=0.95):
        return _dkw_band(self.n_, level)

    def n_required(self, eps, level=0.95):
        """Smallest n such that the DKW bound is <= eps at the given level."""
        alpha = 1.0 - level
        return int(np.ceil(np.log(2.0 / alpha) / (2.0 * eps ** 2)))

    def convergence(self, sizes, dist, random_state=None):
        """Sup-distance of the empirical CDF from ``dist`` across sample sizes
        ``sizes`` (fresh draws from ``dist`` at each size)."""
        rng = np.random.default_rng(random_state)
        out = []
        for n in sizes:
            x = np.asarray(dist.rvs(size=int(n), random_state=rng), dtype=float)
            out.append(GlivenkoCantelli().fit(x, dist).distance_)
        return np.array(out)

    def __repr__(self):
        if not hasattr(self, "distance_"):
            return "GlivenkoCantelli(unfitted)"
        return f"GlivenkoCantelli(distance={self.distance_:.4g}, n={self.n_})"


class EmpiricalLikelihood:
    """Owen's empirical likelihood for a (possibly vector) population mean.

    ``test_mean(mu0)`` maximizes ``sum log(n w_i)`` subject to
    ``sum w_i (x_i - mu0) = 0``, ``sum w_i = 1``, ``w_i >= 0`` via the dual
    (Lagrange multiplier ``lambda``, Owen 2001 Ch. 3), returning a
    ``TestResult`` with ``statistic = -2 log R ~ chi2(d)``.
    """

    def fit(self, x):
        x = np.asarray(x, dtype=float)
        if x.ndim == 1:
            x = x[:, None]
        self.data_ = x
        self.n_, self.d_ = x.shape
        return self

    def _lambda(self, mu0):
        Z = self.data_ - np.asarray(mu0, dtype=float)
        n, d = Z.shape
        eps = 1.0 / n
        lam = np.zeros(d)
        for _ in range(200):
            denom = 1.0 + Z @ lam
            g1 = _el_logstar1(denom, eps)
            g2 = _el_logstar2(denom, eps)
            grad = -Z.T @ g1
            H = -(Z * g2[:, None]).T @ Z
            try:
                step = np.linalg.solve(H - 1e-10 * np.eye(d), grad)
            except np.linalg.LinAlgError:
                step = grad
            # backtracking to keep 1+Z.lam bounded away from collapse
            alpha = 1.0
            for _ in range(30):
                lam_new = lam - alpha * step
                if np.all(1.0 + Z @ lam_new > eps / 2):
                    break
                alpha *= 0.5
            if np.max(np.abs(lam_new - lam)) < 1e-10:
                lam = lam_new
                break
            lam = lam_new
        return lam, Z

    def test_mean(self, mu0):
        mu0 = np.atleast_1d(np.asarray(mu0, dtype=float))
        lam, Z = self._lambda(mu0)
        n = self.n_
        denom = 1.0 + Z @ lam
        llr = 2.0 * np.sum(np.log(np.clip(denom, 1e-300, None)))
        from stochpylib.nonparametric._common import _chi2_sf
        pvalue = float(_chi2_sf(llr, self.d_))
        return TestResult(statistic=float(llr), pvalue=pvalue, df=self.d_,
                           null=f"mean = {mu0.tolist()}", method="empirical likelihood",
                           extras={"lambda": lam})

    def weights(self, mu0):
        lam, Z = self._lambda(mu0)
        n = self.n_
        w = 1.0 / (n * (1.0 + Z @ lam))
        return w

    def confidence_interval(self, level=0.95):
        """1-D only: bracket the profile likelihood-ratio at the chi2(1) critical value."""
        if self.d_ != 1:
            raise NotImplementedError("confidence_interval is 1-D only; use test_mean")
        from stochpylib.nonparametric._common import _chi2_ppf
        crit = float(_chi2_ppf(level, 1))
        xbar = float(np.mean(self.data_))
        s = float(np.std(self.data_, ddof=1))

        def f(mu):
            return self.test_mean(np.array([mu])).statistic - crit

        lo_bracket = xbar - 6 * s / np.sqrt(self.n_) - 1e-6
        hi_bracket = xbar + 6 * s / np.sqrt(self.n_) + 1e-6
        lo = optimize.brentq(f, lo_bracket, xbar - 1e-9)
        hi = optimize.brentq(f, xbar + 1e-9, hi_bracket)
        return (lo, hi)

    def __repr__(self):
        return f"EmpiricalLikelihood(n={getattr(self, 'n_', 0)})"
