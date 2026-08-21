"""Shared interface for every distribution in :mod:`stochpylib.distributions`.

Every concrete distribution is written from scratch (no ``scipy.stats.<dist>`` wrapping) — but
``scipy.special``/``scipy.optimize``/``scipy.integrate`` are used as numerical building blocks the
same way NumPy itself is, for the parts that have no convenient closed form (root-finding for
``ppf``, quadrature for ``entropy``/``mgf``/``cf``, MLE for ``fit``).

A concrete subclass only has to provide what's actually distribution-specific — its own
``pdf``/``pmf``, ``cdf``, support bounds, and closed-form ``mean``/``var`` — and can rely on the
generic numerical fallbacks below for everything else, overriding them only when a convenient
closed form exists (most well-known distributions have one for at least ``rvs``).
"""

import numpy as np
from scipy import integrate, optimize, special


class Distribution:
    """Base class for univariate distributions (discrete or continuous)."""

    is_discrete = False

    def support(self):
        """Return ``(low, high)`` bounds of the distribution's support."""
        return (-np.inf, np.inf)

    # --- distribution-specific (subclasses must override) ---

    def pdf(self, x):
        raise NotImplementedError

    pmf = pdf

    def cdf(self, x):
        raise NotImplementedError

    def mean(self):
        raise NotImplementedError

    def var(self):
        raise NotImplementedError

    def std(self):
        return np.sqrt(self.var())

    # --- generic numerical fallbacks ---

    def ppf(self, q):
        q = np.asarray(q, dtype=float)
        scalar_input = q.ndim == 0
        q = np.atleast_1d(q)
        if self.is_discrete:
            out = np.array([self._ppf_discrete(qi) for qi in q], dtype=float)
        else:
            out = np.array([self._ppf_continuous(qi) for qi in q], dtype=float)
        return out.item() if scalar_input else out

    def _ppf_discrete(self, q):
        low, high = self.support()
        low = int(low) if np.isfinite(low) else 0
        finite_high = np.isfinite(high)
        high = int(high) if finite_high else None
        # exponential search for an upper bracket k with cdf(k) >= q,
        # clamping to the support's upper bound instead of skipping past it
        k = low
        step = 1
        prev = None  # last k known to satisfy cdf(prev) < q
        while True:
            if finite_high and k > high:
                k = high
            if self.cdf(k) >= q:
                hi = k
                break
            prev = k
            nxt = k + step
            step *= 2
            k = nxt
        lo = max(low, prev) if prev is not None else low
        # binary search for the smallest x in [lo, hi] with cdf(x) >= q
        while lo < hi:
            mid = (lo + hi) // 2
            if self.cdf(mid) < q:
                lo = mid + 1
            else:
                hi = mid
        return float(lo)

    def _ppf_continuous(self, q):
        if q <= 0:
            return self.support()[0]
        if q >= 1:
            return self.support()[1]
        low, high = self.support()
        a = low if np.isfinite(low) else -1.0
        b = high if np.isfinite(high) else 1.0
        # expand bracket until cdf(a) <= q <= cdf(b)
        tries = 0
        while self.cdf(a) > q and tries < 200:
            a = a * 2 if a < 0 else -1.0
            tries += 1
        while self.cdf(b) < q and tries < 200:
            b = b * 2 if b > 0 else 1.0
            tries += 1
        return optimize.brentq(lambda x: self.cdf(x) - q, a, b, xtol=1e-10)

    def rvs(self, size=1, random_state=None):
        rng = np.random.default_rng(random_state)
        u = rng.uniform(0.0, 1.0, size=size)
        return self.ppf(u)

    def _support_grid(self, n=20000):
        low, high = self.support()
        if self.is_discrete:
            lo = int(low) if np.isfinite(low) else int(self.ppf(1e-9))
            hi = int(high) if np.isfinite(high) else int(self.ppf(1 - 1e-9))
            return np.arange(lo, hi + 1)
        lo = low if np.isfinite(low) else self.ppf(1e-9)
        hi = high if np.isfinite(high) else self.ppf(1 - 1e-9)
        return np.linspace(lo, hi, n)

    def _moment_about_mean(self, k):
        mu = self.mean()
        if self.is_discrete:
            grid = self._support_grid()
            return np.sum(((grid - mu) ** k) * self.pmf(grid))
        low, high = self.support()
        val, _ = integrate.quad(lambda x: ((x - mu) ** k) * self.pdf(x), low, high, limit=200)
        return val

    def skewness(self):
        return self._moment_about_mean(3) / self.std() ** 3

    def kurtosis(self):
        return self._moment_about_mean(4) / self.var() ** 2 - 3.0

    def entropy(self):
        if self.is_discrete:
            grid = self._support_grid()
            p = self.pmf(grid)
            p = p[p > 1e-300]
            return float(-np.sum(p * np.log(p)))
        low, high = self.support()
        def integrand(x):
            p = self.pdf(x)
            return -p * np.log(p) if p > 1e-300 else 0.0
        val, _ = integrate.quad(integrand, low, high, limit=200)
        return float(val)

    def mgf(self, t):
        if self.is_discrete:
            grid = self._support_grid()
            return float(np.sum(np.exp(t * grid) * self.pmf(grid)))
        low, high = self.support()
        val, _ = integrate.quad(lambda x: np.exp(t * x) * self.pdf(x), low, high, limit=200)
        return float(val)

    def cf(self, t):
        if self.is_discrete:
            grid = self._support_grid()
            return complex(np.sum(np.exp(1j * t * grid) * self.pmf(grid)))
        low, high = self.support()
        real, _ = integrate.quad(lambda x: np.cos(t * x) * self.pdf(x), low, high, limit=200)
        imag, _ = integrate.quad(lambda x: np.sin(t * x) * self.pdf(x), low, high, limit=200)
        return complex(real, imag)

    @classmethod
    def _generic_fit(cls, data, x0, bounds=None):
        data = np.asarray(data, dtype=float)

        def nll(params):
            try:
                dist = cls(*params)
            except Exception:
                return np.inf
            vals = np.atleast_1d(dist.pdf(data))
            if np.any(~np.isfinite(vals)) or np.any(vals <= 0):
                vals = np.clip(vals, 1e-300, None)
            return float(-np.sum(np.log(vals)))

        method = "L-BFGS-B" if bounds is not None else "Nelder-Mead"
        res = optimize.minimize(nll, x0=np.asarray(x0, dtype=float), bounds=bounds, method=method)
        return cls(*res.x)

    @classmethod
    def fit(cls, data):
        data = np.asarray(data, dtype=float)
        x0 = cls._initial_guess(data)
        bounds = cls._param_bounds()
        return cls._generic_fit(data, x0, bounds)

    @classmethod
    def _initial_guess(cls, data):
        raise NotImplementedError(f"{cls.__name__} must implement _initial_guess or override fit")

    @classmethod
    def _param_bounds(cls):
        return None

    def ks_test(self, data):
        data = np.sort(np.asarray(data, dtype=float))
        n = len(data)
        cdf_vals = np.array([self.cdf(x) for x in data])
        i = np.arange(1, n + 1)
        d_plus = np.max(i / n - cdf_vals)
        d_minus = np.max(cdf_vals - (i - 1) / n)
        d_stat = max(d_plus, d_minus)
        lam = max((np.sqrt(n) + 0.12 + 0.11 / np.sqrt(n)) * d_stat, 0.0)
        p_value = 2.0 * sum(
            (-1) ** (k - 1) * np.exp(-2.0 * k**2 * lam**2) for k in range(1, 101)
        )
        p_value = float(np.clip(p_value, 0.0, 1.0))
        return float(d_stat), p_value


class MultivariateDistribution:
    """Base class for multivariate distributions.

    ``.ppf()``, ``.skewness()``, ``.kurtosis()``, and ``.ks_test()`` have no single standard
    generalization to the multivariate case and raise ``NotImplementedError`` rather than fake a
    scalar/approximate answer. ``.cdf()`` is computed via Monte Carlo integration (approximate by
    construction — exact multivariate CDF integration from scratch is impractical).
    """

    def pdf(self, x):
        raise NotImplementedError

    def mean(self):
        raise NotImplementedError

    def var(self):
        """Covariance matrix (the multivariate analog of variance)."""
        raise NotImplementedError

    def rvs(self, size=1, random_state=None):
        raise NotImplementedError

    def entropy(self):
        raise NotImplementedError

    @classmethod
    def fit(cls, data):
        raise NotImplementedError

    def cdf(self, x, n_samples=20000, random_state=None):
        rng = np.random.default_rng(random_state)
        samples = self.rvs(size=n_samples, random_state=rng)
        x = np.asarray(x, dtype=float)
        return float(np.mean(np.all(samples <= x, axis=-1)))

    def ppf(self, q):
        raise NotImplementedError("not defined for multivariate distributions")

    def skewness(self):
        raise NotImplementedError("not defined for multivariate distributions")

    def kurtosis(self):
        raise NotImplementedError("not defined for multivariate distributions")

    def ks_test(self, data):
        raise NotImplementedError("not defined for multivariate distributions")


def regularized_gamma_lower(a, x):
    return special.gammainc(a, x)


def regularized_gamma_upper(a, x):
    return special.gammaincc(a, x)
