"""Result objects returned by :mod:`stochpylib.bayesian`.

``Posterior`` (``core.posterior``), ``PosteriorApproximation`` (``computation.py``) and
``ICResult`` (``selection.py``) follow the library-wide convention of a shared result
carrying the point answer plus enough to quantify it further (interval, samples,
diagnostics) -- the ``bayesian`` sibling of ``statistics.EstimateResult``/``TestResult``.
"""

import numpy as np
from scipy import special

__all__ = ["Posterior", "PosteriorApproximation", "ICResult", "EmpiricalPredictive"]


def _weighted_quantile(samples, weights, q):
    order = np.argsort(samples)
    s, w = samples[order], weights[order]
    cw = np.cumsum(w)
    cw = cw / cw[-1]
    return float(np.interp(q, cw, s))


class Posterior:
    """A posterior distribution over one or more parameters.

    Exactly one of ``dist`` (an exact/approximate library ``Distribution``),
    ``samples_``/``weights_`` (MCMC or weighted-particle draws) or ``grid_``/``density_``
    (a numerical grid) is populated, depending on ``method``.
    """

    def __init__(self, method, dim, dist=None, samples_=None, weights_=None, grid_=None,
                 density_=None, mean_=None, cov_=None, mode_=None, log_evidence_=None,
                 extras=None):
        self.method = method
        self.dim = int(dim)
        self.dist = dist
        self.samples_ = samples_
        self.weights_ = weights_
        self.grid_ = grid_
        self.density_ = density_
        self.mean_ = None if mean_ is None else np.atleast_1d(np.asarray(mean_, dtype=float))
        self.cov_ = None if cov_ is None else np.atleast_2d(np.asarray(cov_, dtype=float))
        self.mode_ = None if mode_ is None else np.atleast_1d(np.asarray(mode_, dtype=float))
        self.log_evidence_ = log_evidence_
        self.extras = extras or {}

    def mean(self):
        return self.mean_

    def cov(self):
        return self.cov_

    def var(self):
        return np.diag(self.cov_) if self.cov_ is not None else None

    def std(self):
        v = self.var()
        return None if v is None else np.sqrt(v)

    def mode(self):
        return self.mode_

    def sample(self, n=1, random_state=None):
        rng = np.random.default_rng(random_state)
        if self.dist is not None:
            out = np.atleast_2d(np.asarray(self.dist.rvs(n, random_state=rng)))
            if self.dim == 1 and out.ndim == 2 and out.shape[1] != 1:
                out = out.reshape(n, 1)
            return out
        if self.samples_ is not None:
            w = self.weights_
            if w is None:
                idx = rng.integers(0, len(self.samples_), size=n)
            else:
                idx = rng.choice(len(self.samples_), size=n, p=w / np.sum(w))
            return self.samples_[idx]
        if self.grid_ is not None:
            return self._sample_grid(n, rng)
        raise RuntimeError("posterior has no dist/samples/grid to sample from")

    def _sample_grid(self, n, rng):
        if self.dim == 1:
            grid, dens = self.grid_, self.density_
            widths = np.gradient(grid)
            cell_p = dens * widths
            cell_p = cell_p / cell_p.sum()
            cells = rng.choice(len(grid), size=n, p=cell_p)
            step = np.median(widths)
            return (grid[cells] + rng.uniform(-step / 2, step / 2, size=n))[:, None]
        gx, gy = self.grid_
        dens = self.density_
        flat_p = dens.ravel()
        flat_p = flat_p / flat_p.sum()
        idx = rng.choice(flat_p.size, size=n, p=flat_p)
        iy, ix = np.unravel_index(idx, dens.shape)
        dx = gx[0, 1] - gx[0, 0] if gx.shape[1] > 1 else 1e-6
        dy = gy[1, 0] - gy[0, 0] if gy.shape[0] > 1 else 1e-6
        xs = gx[0, ix] + rng.uniform(-dx / 2, dx / 2, size=n)
        ys = gy[iy, 0] + rng.uniform(-dy / 2, dy / 2, size=n)
        return np.column_stack([xs, ys])

    def logpdf(self, theta):
        if self.dist is not None:
            p = np.clip(np.atleast_1d(np.asarray(self.dist.pdf(theta), dtype=float)), 1e-300, None)
            return float(np.sum(np.log(p))) if self.dim == 1 else float(np.log(p[0]))
        if self.grid_ is not None and self.dim == 1:
            return float(np.log(max(np.interp(theta, self.grid_, self.density_), 1e-300)))
        raise NotImplementedError("logpdf is only available for dist-backed or 1-D grid posteriors")

    def pdf(self, theta):
        return float(np.exp(self.logpdf(theta)))

    def credible_interval(self, level=0.95, kind="equal-tailed"):
        alpha2 = (1.0 - level) / 2.0
        if self.dist is not None and hasattr(self.dist, "ppf") and self.dim == 1:
            if kind == "equal-tailed":
                return (float(self.dist.ppf(alpha2)), float(self.dist.ppf(1 - alpha2)))
            return self._hpd_from_samples(self.sample(20000, random_state=0)[:, 0], level)
        if self.samples_ is not None:
            out = []
            w = self.weights_
            for j in range(self.dim):
                col = self.samples_[:, j]
                if kind == "equal-tailed":
                    if w is None:
                        out.append((float(np.quantile(col, alpha2)), float(np.quantile(col, 1 - alpha2))))
                    else:
                        out.append((_weighted_quantile(col, w, alpha2), _weighted_quantile(col, w, 1 - alpha2)))
                else:
                    out.append(self._hpd_from_samples(col, level))
            return out[0] if self.dim == 1 else out
        if self.grid_ is not None and self.dim == 1:
            grid, dens = self.grid_, self.density_
            widths = np.gradient(grid)
            cdf = np.cumsum(dens * widths)
            cdf = cdf / cdf[-1]
            if kind == "equal-tailed":
                return (float(np.interp(alpha2, cdf, grid)), float(np.interp(1 - alpha2, cdf, grid)))
            return self._hpd_from_grid(grid, dens, widths, level)
        raise NotImplementedError("credible_interval not available for this posterior representation")

    @staticmethod
    def _hpd_from_samples(x, level):
        s = np.sort(x)
        n = len(s)
        n_in = max(int(np.ceil(level * n)), 1)
        widths = s[n_in - 1:] - s[:n - n_in + 1] if n - n_in + 1 > 0 else s[-1:] - s[:1]
        i = int(np.argmin(widths))
        return (float(s[i]), float(s[i + n_in - 1]))

    @staticmethod
    def _hpd_from_grid(grid, dens, widths, level):
        order = np.argsort(dens)[::-1]
        mass = np.cumsum((dens * widths)[order])
        mass = mass / (dens * widths).sum()
        k = np.searchsorted(mass, level) + 1
        idx = np.sort(order[:k])
        return (float(grid[idx[0]]), float(grid[idx[-1]]))

    def summary(self):
        m, s = self.mean(), self.std()
        out = {}
        for j in range(self.dim):
            ci = self.credible_interval(0.95)
            ci_j = ci if self.dim == 1 else ci[j]
            out[f"theta_{j}"] = {
                "mean": float(m[j]) if m is not None else None,
                "std": float(s[j]) if s is not None else None,
                "ci_95": ci_j,
            }
        return out

    def __repr__(self):
        return f"Posterior(method={self.method!r}, dim={self.dim})"


class PosteriorApproximation:
    """A Gaussian (or importance-weighted) approximation to a posterior, returned by
    every function in :mod:`stochpylib.bayesian.computation`."""

    def __init__(self, method, mean_, cov_, log_evidence_=None, samples_=None,
                 weights_=None, n_iter_=None, converged_=None, extras=None):
        self.method = method
        self.mean_ = np.atleast_1d(np.asarray(mean_, dtype=float))
        self.cov_ = np.atleast_2d(np.asarray(cov_, dtype=float))
        self.dim = len(self.mean_)
        self.log_evidence_ = log_evidence_
        self.samples_ = samples_
        self.weights_ = weights_
        self.n_iter_ = n_iter_
        self.converged_ = converged_
        self.extras = extras or {}

    @property
    def std_(self):
        return np.sqrt(np.diag(self.cov_))

    def sample(self, n=1, random_state=None):
        rng = np.random.default_rng(random_state)
        if self.samples_ is not None:
            w = self.weights_
            if w is None:
                idx = rng.integers(0, len(self.samples_), size=n)
            else:
                idx = rng.choice(len(self.samples_), size=n, p=w / np.sum(w))
            return self.samples_[idx]
        from stochpylib.bayesian._common import _cholesky_psd
        L = _cholesky_psd(self.cov_)
        return self.mean_ + rng.standard_normal((n, self.dim)) @ L.T

    def logpdf(self, x):
        from stochpylib.bayesian._common import _mvn_logpdf
        return _mvn_logpdf(np.asarray(x, dtype=float), self.mean_, self.cov_)

    def credible_interval(self, level=0.95):
        alpha2 = (1.0 - level) / 2.0
        if self.samples_ is not None:
            w = self.weights_
            out = []
            for j in range(self.dim):
                col = self.samples_[:, j]
                if w is None:
                    out.append((float(np.quantile(col, alpha2)), float(np.quantile(col, 1 - alpha2))))
                else:
                    out.append((_weighted_quantile(col, w, alpha2), _weighted_quantile(col, w, 1 - alpha2)))
            return out[0] if self.dim == 1 else out
        z = float(special.ndtri(1 - alpha2))
        lo = self.mean_ - z * self.std_
        hi = self.mean_ + z * self.std_
        out = list(zip(lo.tolist(), hi.tolist()))
        return out[0] if self.dim == 1 else out

    def to_posterior(self):
        from stochpylib.distributions import MultivariateNormal, Normal
        if self.dim == 1:
            dist = Normal(float(self.mean_[0]), float(self.std_[0]))
        else:
            dist = MultivariateNormal(self.mean_, self.cov_)
        return Posterior(f"{self.method}->posterior", self.dim, dist=dist,
                          samples_=self.samples_, weights_=self.weights_,
                          mean_=self.mean_, cov_=self.cov_,
                          log_evidence_=self.log_evidence_, extras=dict(self.extras))

    def __repr__(self):
        ev = "" if self.log_evidence_ is None else f", log_evidence={self.log_evidence_:.4g}"
        return f"PosteriorApproximation(method={self.method!r}, dim={self.dim}{ev})"


class ICResult:
    """An information-criterion / Bayes-factor result: ``float(result)`` gives the
    criterion value; ``se``/``p_eff``/``pointwise`` are populated where defined."""

    def __init__(self, name, value, se=None, p_eff=None, pointwise=None, extras=None):
        self.name = name
        self.value = float(value)
        self.se = None if se is None else float(se)
        self.p_eff = None if p_eff is None else float(p_eff)
        self.pointwise = pointwise
        self.extras = extras or {}

    def __float__(self):
        return self.value

    def __repr__(self):
        se = "" if self.se is None else f" (se={self.se:.4g})"
        return f"{self.name}={self.value:.4g}{se}"


class EmpiricalPredictive:
    """A posterior-predictive distribution represented by draws.

    Satisfies the common distribution contract (pdf/cdf/ppf/rvs/mean/var/skewness/
    kurtosis/entropy/mgf/cf/fit/ks_test) via a Gaussian KDE (continuous draws) or
    relative frequency (integer-valued draws, ``is_discrete=True``).
    """

    def __init__(self, draws):
        self.draws = np.asarray(draws, dtype=float).ravel()
        if self.draws.size == 0:
            raise ValueError("draws must be non-empty")
        self.is_discrete = bool(np.all(self.draws == np.round(self.draws)))
        n = len(self.draws)
        std = np.std(self.draws, ddof=1) if n > 1 else 1.0
        self._h = 1.06 * max(std, 1e-12) * n ** (-1 / 5) if not self.is_discrete else None

    def support(self):
        return (float(self.draws.min()), float(self.draws.max()))

    def pmf(self, k):
        if not self.is_discrete:
            raise NotImplementedError("pmf is only defined for integer-valued draws")
        k = np.atleast_1d(np.asarray(k, dtype=float))
        vals, counts = np.unique(self.draws, return_counts=True)
        freq = counts / counts.sum()
        out = np.array([freq[vals == ki][0] if ki in vals else 0.0 for ki in k])
        return out if out.size > 1 else float(out[0])

    def pdf(self, x):
        if self.is_discrete:
            return self.pmf(x)
        x = np.atleast_1d(np.asarray(x, dtype=float))
        z = (x[:, None] - self.draws[None, :]) / self._h
        out = np.mean(np.exp(-0.5 * z ** 2) / np.sqrt(2 * np.pi), axis=1) / self._h
        return out if out.size > 1 else float(out[0])

    def cdf(self, x):
        x = np.atleast_1d(np.asarray(x, dtype=float))
        out = np.mean(self.draws[None, :] <= x[:, None], axis=1)
        return out if out.size > 1 else float(out[0])

    def ppf(self, q):
        return np.quantile(self.draws, q)

    def rvs(self, size=1, random_state=None):
        rng = np.random.default_rng(random_state)
        return rng.choice(self.draws, size=size, replace=True)

    def mean(self):
        return float(np.mean(self.draws))

    def var(self):
        return float(np.var(self.draws, ddof=1))

    def std(self):
        return float(np.std(self.draws, ddof=1))

    def skewness(self):
        m = self.mean()
        s = np.std(self.draws)
        return float(np.mean(((self.draws - m) / s) ** 3)) if s > 0 else 0.0

    def kurtosis(self):
        m = self.mean()
        s = np.std(self.draws)
        return float(np.mean(((self.draws - m) / s) ** 4) - 3.0) if s > 0 else 0.0

    def entropy(self):
        n_bins = max(int(np.sqrt(len(self.draws))), 5)
        hist, edges = np.histogram(self.draws, bins=n_bins, density=True)
        widths = np.diff(edges)
        p = hist * widths
        p = p[p > 1e-300]
        return float(-np.sum(p * np.log(p / widths[: len(p)])))

    def mgf(self, t):
        return float(np.mean(np.exp(t * self.draws)))

    def cf(self, t):
        return complex(np.mean(np.exp(1j * t * self.draws)))

    @classmethod
    def fit(cls, data):
        return cls(data)

    def ks_test(self, data):
        data = np.sort(np.asarray(data, dtype=float))
        n = len(data)
        cdf_vals = self.cdf(data)
        i = np.arange(1, n + 1)
        d_plus = np.max(i / n - cdf_vals)
        d_minus = np.max(cdf_vals - (i - 1) / n)
        d_stat = max(d_plus, d_minus)
        lam = max((np.sqrt(n) + 0.12 + 0.11 / np.sqrt(n)) * d_stat, 0.0)
        p_value = 2.0 * sum((-1) ** (k - 1) * np.exp(-2.0 * k ** 2 * lam ** 2) for k in range(1, 101))
        return float(d_stat), float(np.clip(p_value, 0.0, 1.0))

    def __repr__(self):
        return f"EmpiricalPredictive(n_draws={len(self.draws)}, is_discrete={self.is_discrete})"
