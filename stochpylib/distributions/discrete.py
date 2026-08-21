"""Discrete distributions, implemented from scratch on top of :mod:`stochpylib.distributions._base`.

``Multinomial`` is placed here (rather than in ``multivariate.py``) because that's where the
design spec lists it, even though it is inherently multivariate (vector-valued outcomes) — it
subclasses :class:`MultivariateDistribution` and follows the same ``NotImplementedError``
convention for ``ppf``/``skewness``/``kurtosis``/``ks_test``.
"""

import numpy as np
from scipy import special

from stochpylib.distributions._base import Distribution, MultivariateDistribution


class Bernoulli(Distribution):
    """Bernoulli(p): a single trial, 1 with probability p, 0 otherwise."""

    is_discrete = True

    def __init__(self, p):
        if not (0 <= p <= 1):
            raise ValueError("p must be in [0, 1]")
        self.p = float(p)

    def support(self):
        return (0, 1)

    def pmf(self, k):
        k = np.asarray(k)
        return np.where(k == 1, self.p, np.where(k == 0, 1 - self.p, 0.0))

    pdf = pmf

    def cdf(self, x):
        x = np.asarray(x, dtype=float)
        return np.where(x < 0, 0.0, np.where(x < 1, 1 - self.p, 1.0))

    def mean(self):
        return self.p

    def var(self):
        return self.p * (1 - self.p)

    def rvs(self, size=1, random_state=None):
        rng = np.random.default_rng(random_state)
        return rng.binomial(1, self.p, size=size)

    @classmethod
    def fit(cls, data):
        return cls(float(np.mean(data)))


class Binomial(Distribution):
    """Binomial(n, p): number of successes in n independent Bernoulli(p) trials."""

    is_discrete = True

    def __init__(self, n, p):
        if not (0 <= p <= 1):
            raise ValueError("p must be in [0, 1]")
        self.n = int(n)
        self.p = float(p)

    def support(self):
        return (0, self.n)

    def pmf(self, k):
        k = np.asarray(k)
        valid = (k >= 0) & (k <= self.n) & (k == np.round(k))
        out = np.where(
            valid,
            special.comb(self.n, np.clip(k, 0, self.n)) * self.p**k * (1 - self.p) ** (self.n - k),
            0.0,
        )
        return out

    pdf = pmf

    def cdf(self, x):
        x = np.asarray(x, dtype=float)
        k = np.floor(x)
        return np.where(
            k < 0, 0.0, np.where(k >= self.n, 1.0, special.betainc(self.n - k, k + 1, 1 - self.p))
        )

    def mean(self):
        return self.n * self.p

    def var(self):
        return self.n * self.p * (1 - self.p)

    def rvs(self, size=1, random_state=None):
        rng = np.random.default_rng(random_state)
        return rng.binomial(self.n, self.p, size=size)

    @classmethod
    def fit(cls, data, n=None):
        if n is None:
            raise ValueError("Binomial.fit requires the number of trials n")
        return cls(n, float(np.mean(data)) / n)


class Poisson(Distribution):
    """Poisson(lambda): count of events in a fixed interval at constant rate lambda."""

    is_discrete = True

    def __init__(self, lam):
        if lam <= 0:
            raise ValueError("lam must be > 0")
        self.lam = float(lam)

    def support(self):
        return (0, np.inf)

    def pmf(self, k):
        k = np.asarray(k)
        valid = (k >= 0) & (k == np.round(k))
        kk = np.clip(k, 0, None)
        out = np.where(valid, np.exp(-self.lam) * self.lam**kk / special.gamma(kk + 1), 0.0)
        return out

    pdf = pmf

    def cdf(self, x):
        x = np.asarray(x, dtype=float)
        k = np.floor(x)
        return np.where(k < 0, 0.0, special.gammaincc(k + 1, self.lam))

    def mean(self):
        return self.lam

    def var(self):
        return self.lam

    def rvs(self, size=1, random_state=None):
        rng = np.random.default_rng(random_state)
        return rng.poisson(self.lam, size=size)

    @classmethod
    def fit(cls, data):
        return cls(float(np.mean(data)))


class Geometric(Distribution):
    """Geometric(p): number of trials (>=1) until the first success."""

    is_discrete = True

    def __init__(self, p):
        if not (0 < p <= 1):
            raise ValueError("p must be in (0, 1]")
        self.p = float(p)

    def support(self):
        return (1, np.inf)

    def pmf(self, k):
        k = np.asarray(k)
        valid = (k >= 1) & (k == np.round(k))
        return np.where(valid, (1 - self.p) ** (k - 1) * self.p, 0.0)

    pdf = pmf

    def cdf(self, x):
        x = np.asarray(x, dtype=float)
        k = np.floor(x)
        return np.where(k < 1, 0.0, 1 - (1 - self.p) ** k)

    def mean(self):
        return 1.0 / self.p

    def var(self):
        return (1 - self.p) / self.p**2

    def rvs(self, size=1, random_state=None):
        rng = np.random.default_rng(random_state)
        return rng.geometric(self.p, size=size)

    @classmethod
    def fit(cls, data):
        return cls(1.0 / float(np.mean(data)))


class NegBinomial(Distribution):
    """NegBinomial(r, p): number of failures before the r-th success."""

    is_discrete = True

    def __init__(self, r, p):
        if not (0 < p <= 1):
            raise ValueError("p must be in (0, 1]")
        self.r = float(r)
        self.p = float(p)

    def support(self):
        return (0, np.inf)

    def pmf(self, k):
        k = np.asarray(k)
        valid = (k >= 0) & (k == np.round(k))
        kk = np.clip(k, 0, None)
        coef = special.gamma(kk + self.r) / (special.gamma(self.r) * special.gamma(kk + 1))
        return np.where(valid, coef * self.p**self.r * (1 - self.p) ** kk, 0.0)

    pdf = pmf

    def cdf(self, x):
        x = np.asarray(x, dtype=float)
        k = np.floor(x)
        return np.where(k < 0, 0.0, special.betainc(self.r, k + 1, self.p))

    def mean(self):
        return self.r * (1 - self.p) / self.p

    def var(self):
        return self.r * (1 - self.p) / self.p**2

    def rvs(self, size=1, random_state=None):
        rng = np.random.default_rng(random_state)
        return rng.negative_binomial(self.r, self.p, size=size)

    @classmethod
    def fit(cls, data, r=None):
        if r is None:
            raise ValueError("NegBinomial.fit requires the number of successes r")
        m = float(np.mean(data))
        return cls(r, r / (r + m))


class Hypergeometric(Distribution):
    """Hypergeometric(N, K, n): successes in n draws without replacement from N items, K of which succeed."""

    is_discrete = True

    def __init__(self, N, K, n):
        self.N, self.K, self.n = int(N), int(K), int(n)

    def support(self):
        return (max(0, self.n - (self.N - self.K)), min(self.n, self.K))

    def pmf(self, k):
        k = np.asarray(k)
        lo, hi = self.support()
        valid = (k >= lo) & (k <= hi) & (k == np.round(k))
        kk = np.clip(k, lo, hi)
        out = np.where(
            valid,
            special.comb(self.K, kk) * special.comb(self.N - self.K, self.n - kk) / special.comb(self.N, self.n),
            0.0,
        )
        return out

    pdf = pmf

    def cdf(self, x):
        x = np.asarray(x, dtype=float)
        scalar = x.ndim == 0
        x = np.atleast_1d(x)
        lo, hi = self.support()
        out = np.array([np.sum(self.pmf(np.arange(lo, int(np.floor(xi)) + 1))) if xi >= lo else 0.0 for xi in x])
        out = np.where(x > hi, 1.0, out)
        return out.item() if scalar else out

    def mean(self):
        return self.n * self.K / self.N

    def var(self):
        N, K, n = self.N, self.K, self.n
        p = K / N
        return n * p * (1 - p) * (N - n) / (N - 1)

    def rvs(self, size=1, random_state=None):
        rng = np.random.default_rng(random_state)
        return rng.hypergeometric(self.K, self.N - self.K, self.n, size=size)

    @classmethod
    def fit(cls, data, N=None, n=None):
        if N is None or n is None:
            raise ValueError("Hypergeometric.fit requires population size N and sample size n")
        K = round(float(np.mean(data)) * N / n)
        return cls(N, K, n)


class DiscreteUniform(Distribution):
    """DiscreteUniform(low, high): equally likely integers in [low, high]."""

    is_discrete = True

    def __init__(self, low, high):
        self.low, self.high = int(low), int(high)

    def support(self):
        return (self.low, self.high)

    def pmf(self, k):
        k = np.asarray(k)
        n = self.high - self.low + 1
        valid = (k >= self.low) & (k <= self.high) & (k == np.round(k))
        return np.where(valid, 1.0 / n, 0.0)

    pdf = pmf

    def cdf(self, x):
        x = np.asarray(x, dtype=float)
        n = self.high - self.low + 1
        k = np.floor(x)
        return np.clip((k - self.low + 1) / n, 0.0, 1.0)

    def mean(self):
        return (self.low + self.high) / 2.0

    def var(self):
        n = self.high - self.low + 1
        return (n**2 - 1) / 12.0

    def rvs(self, size=1, random_state=None):
        rng = np.random.default_rng(random_state)
        return rng.integers(self.low, self.high + 1, size=size)

    @classmethod
    def fit(cls, data):
        return cls(int(np.min(data)), int(np.max(data)))


class Multinomial(MultivariateDistribution):
    """Multinomial(n, p): counts of each of len(p) categories over n independent trials.

    Listed under ``distributions.discrete`` per the spec even though it's vector-valued; follows
    the multivariate convention for ``ppf``/``skewness``/``kurtosis``/``ks_test``.
    """

    def __init__(self, n, p):
        p = np.asarray(p, dtype=float)
        if not np.isclose(p.sum(), 1.0):
            raise ValueError("p must sum to 1")
        self.n = int(n)
        self.p = p

    def pmf(self, x):
        x = np.asarray(x, dtype=float)
        coef = special.gamma(self.n + 1) / np.prod(special.gamma(x + 1))
        return float(coef * np.prod(self.p**x))

    pdf = pmf

    def mean(self):
        return self.n * self.p

    def var(self):
        return self.n * (np.diag(self.p) - np.outer(self.p, self.p))

    def rvs(self, size=1, random_state=None):
        rng = np.random.default_rng(random_state)
        return rng.multinomial(self.n, self.p, size=size)

    def entropy(self):
        samples = self.rvs(size=20000)
        _, counts = np.unique(samples, axis=0, return_counts=True)
        probs = counts / counts.sum()
        return float(-np.sum(probs * np.log(probs)))

    @classmethod
    def fit(cls, data):
        data = np.asarray(data, dtype=float)
        n = int(data.sum(axis=-1).mean())
        p = data.mean(axis=0) / n
        return cls(n, p)


class ZipfDistribution(Distribution):
    """Zipf(a): pmf(k) proportional to k^-a for k = 1, 2, 3, ..."""

    is_discrete = True

    def __init__(self, a):
        if a <= 1:
            raise ValueError("a must be > 1")
        self.a = float(a)

    def support(self):
        return (1, np.inf)

    def _zeta(self, s):
        return special.zeta(s, 1)

    def pmf(self, k):
        k = np.asarray(k)
        valid = (k >= 1) & (k == np.round(k))
        kk = np.clip(k, 1, None)
        return np.where(valid, kk ** (-self.a) / self._zeta(self.a), 0.0)

    pdf = pmf

    def cdf(self, x):
        x = np.asarray(x, dtype=float)
        scalar = x.ndim == 0
        x = np.atleast_1d(x)
        out = np.array([np.sum(self.pmf(np.arange(1, int(np.floor(xi)) + 1))) if xi >= 1 else 0.0 for xi in x])
        return out.item() if scalar else out

    def mean(self):
        return self._zeta(self.a - 1) / self._zeta(self.a) if self.a > 2 else np.inf

    def var(self):
        if self.a <= 3:
            return np.inf
        return self._zeta(self.a - 2) / self._zeta(self.a) - self.mean() ** 2

    def rvs(self, size=1, random_state=None):
        rng = np.random.default_rng(random_state)
        return rng.zipf(self.a, size=size)

    @classmethod
    def _initial_guess(cls, data):
        return [2.0]

    @classmethod
    def _param_bounds(cls):
        return [(1.0001, None)]


class BetaBinomial(Distribution):
    """BetaBinomial(n, a, b): Binomial(n, p) with p ~ Beta(a, b) integrated out."""

    is_discrete = True

    def __init__(self, n, a, b):
        self.n = int(n)
        self.a = float(a)
        self.b = float(b)

    def support(self):
        return (0, self.n)

    def pmf(self, k):
        k = np.asarray(k)
        valid = (k >= 0) & (k <= self.n) & (k == np.round(k))
        kk = np.clip(k, 0, self.n)
        out = special.comb(self.n, kk) * special.beta(kk + self.a, self.n - kk + self.b) / special.beta(self.a, self.b)
        return np.where(valid, out, 0.0)

    pdf = pmf

    def cdf(self, x):
        x = np.asarray(x, dtype=float)
        scalar = x.ndim == 0
        x = np.atleast_1d(x)
        out = np.array([np.sum(self.pmf(np.arange(0, int(np.floor(xi)) + 1))) if xi >= 0 else 0.0 for xi in x])
        out = np.where(x > self.n, 1.0, out)
        return out.item() if scalar else out

    def mean(self):
        return self.n * self.a / (self.a + self.b)

    def var(self):
        a, b, n = self.a, self.b, self.n
        return n * a * b * (a + b + n) / ((a + b) ** 2 * (a + b + 1))

    def rvs(self, size=1, random_state=None):
        rng = np.random.default_rng(random_state)
        p = rng.beta(self.a, self.b, size=size)
        return rng.binomial(self.n, p)

    @classmethod
    def _initial_guess(cls, data):
        m = np.mean(data)
        return [m / 2 + 0.5, m / 2 + 0.5]

    @classmethod
    def _param_bounds(cls):
        return [(1e-3, None), (1e-3, None)]

    @classmethod
    def fit(cls, data, n=None):
        if n is None:
            raise ValueError("BetaBinomial.fit requires the number of trials n")
        data = np.asarray(data, dtype=float)

        def nll(params):
            a, b = params
            if a <= 0 or b <= 0:
                return np.inf
            dist = cls(n, a, b)
            vals = np.clip(dist.pmf(data), 1e-300, None)
            return float(-np.sum(np.log(vals)))

        from scipy.optimize import minimize

        m = np.mean(data)
        x0 = [max(m / 2, 0.5), max(n - m, 1) / 2 + 0.5]
        res = minimize(nll, x0=x0, bounds=[(1e-3, None), (1e-3, None)], method="L-BFGS-B")
        return cls(n, *res.x)


class ConwayMaxwellPoisson(Distribution):
    """COM-Poisson(lambda, nu): Poisson-like with an extra dispersion parameter nu.

    No closed-form normalizing constant exists; Z(lambda, nu) = sum_j lambda^j / (j!)^nu is
    truncated once terms become negligible.
    """

    is_discrete = True

    def __init__(self, lam, nu, max_terms=2000):
        if lam <= 0 or nu < 0:
            raise ValueError("lam must be > 0 and nu must be >= 0")
        self.lam = float(lam)
        self.nu = float(nu)
        self.max_terms = max_terms
        self._Z, self._k_max = self._compute_Z()

    def _compute_Z(self):
        total = 0.0
        term = 1.0
        j = 0
        while j < self.max_terms:
            total += term
            j += 1
            term *= self.lam / j**self.nu
            if term < 1e-16 * total and j > 10:
                break
        return total, j

    def support(self):
        return (0, np.inf)

    def pmf(self, k):
        k = np.asarray(k)
        valid = (k >= 0) & (k == np.round(k))
        kk = np.clip(k, 0, None)
        out = np.where(valid, self.lam**kk / (special.gamma(kk + 1) ** self.nu) / self._Z, 0.0)
        return out

    pdf = pmf

    def cdf(self, x):
        x = np.asarray(x, dtype=float)
        scalar = x.ndim == 0
        x = np.atleast_1d(x)
        out = np.array([np.sum(self.pmf(np.arange(0, int(np.floor(xi)) + 1))) if xi >= 0 else 0.0 for xi in x])
        return out.item() if scalar else out

    def mean(self):
        grid = np.arange(0, self._k_max + 1)
        return float(np.sum(grid * self.pmf(grid)))

    def var(self):
        grid = np.arange(0, self._k_max + 1)
        m = self.mean()
        return float(np.sum((grid - m) ** 2 * self.pmf(grid)))

    @classmethod
    def _initial_guess(cls, data):
        return [max(np.mean(data), 0.5), 1.0]

    @classmethod
    def _param_bounds(cls):
        return [(1e-6, None), (0.0, None)]
