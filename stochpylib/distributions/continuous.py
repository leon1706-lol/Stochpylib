"""Continuous distributions, implemented from scratch on top of
:mod:`stochpylib.distributions._base`. ``scipy.special`` is used for standard mathematical
building blocks (gamma/beta/Bessel/erf functions, regularized incomplete gamma/beta and their
inverses) the same way NumPy is — none of these are packaged distribution objects.
"""

import numpy as np
from scipy import integrate, optimize, special

from stochpylib.distributions._base import Distribution

EULER_GAMMA = 0.5772156649015329


class Normal(Distribution):
    def __init__(self, mu=0.0, sigma=1.0):
        if sigma <= 0:
            raise ValueError("sigma must be > 0")
        self.mu, self.sigma = float(mu), float(sigma)

    def pdf(self, x):
        x = np.asarray(x, dtype=float)
        return np.exp(-((x - self.mu) ** 2) / (2 * self.sigma**2)) / (self.sigma * np.sqrt(2 * np.pi))

    def cdf(self, x):
        x = np.asarray(x, dtype=float)
        return special.ndtr((x - self.mu) / self.sigma)

    def ppf(self, q):
        q = np.asarray(q, dtype=float)
        return self.mu + self.sigma * special.ndtri(q)

    def mean(self):
        return self.mu

    def var(self):
        return self.sigma**2

    def rvs(self, size=1, random_state=None):
        rng = np.random.default_rng(random_state)
        return rng.normal(self.mu, self.sigma, size=size)

    def entropy(self):
        return 0.5 * np.log(2 * np.pi * np.e * self.sigma**2)

    def skewness(self):
        return 0.0

    def kurtosis(self):
        return 0.0

    def mgf(self, t):
        return float(np.exp(self.mu * t + 0.5 * self.sigma**2 * t**2))

    def cf(self, t):
        return complex(np.exp(1j * self.mu * t - 0.5 * self.sigma**2 * t**2))

    @classmethod
    def fit(cls, data):
        data = np.asarray(data, dtype=float)
        return cls(float(np.mean(data)), float(np.std(data)))


class Exponential(Distribution):
    def __init__(self, rate=1.0):
        if rate <= 0:
            raise ValueError("rate must be > 0")
        self.rate = float(rate)

    def support(self):
        return (0, np.inf)

    def pdf(self, x):
        x = np.asarray(x, dtype=float)
        return np.where(x >= 0, self.rate * np.exp(-self.rate * x), 0.0)

    def cdf(self, x):
        x = np.asarray(x, dtype=float)
        return np.where(x >= 0, 1 - np.exp(-self.rate * x), 0.0)

    def ppf(self, q):
        q = np.asarray(q, dtype=float)
        return -np.log(1 - q) / self.rate

    def mean(self):
        return 1.0 / self.rate

    def var(self):
        return 1.0 / self.rate**2

    def rvs(self, size=1, random_state=None):
        rng = np.random.default_rng(random_state)
        return rng.exponential(1.0 / self.rate, size=size)

    def entropy(self):
        return 1 - np.log(self.rate)

    def skewness(self):
        return 2.0

    def kurtosis(self):
        return 6.0

    def mgf(self, t):
        if t >= self.rate:
            return np.inf
        return self.rate / (self.rate - t)

    def cf(self, t):
        return complex(self.rate / (self.rate - 1j * t))

    @classmethod
    def fit(cls, data):
        return cls(1.0 / float(np.mean(data)))


class Uniform(Distribution):
    def __init__(self, a=0.0, b=1.0):
        if b <= a:
            raise ValueError("b must be > a")
        self.a, self.b = float(a), float(b)

    def support(self):
        return (self.a, self.b)

    def pdf(self, x):
        x = np.asarray(x, dtype=float)
        return np.where((x >= self.a) & (x <= self.b), 1.0 / (self.b - self.a), 0.0)

    def cdf(self, x):
        x = np.asarray(x, dtype=float)
        return np.clip((x - self.a) / (self.b - self.a), 0.0, 1.0)

    def ppf(self, q):
        q = np.asarray(q, dtype=float)
        return self.a + q * (self.b - self.a)

    def mean(self):
        return (self.a + self.b) / 2.0

    def var(self):
        return (self.b - self.a) ** 2 / 12.0

    def rvs(self, size=1, random_state=None):
        rng = np.random.default_rng(random_state)
        return rng.uniform(self.a, self.b, size=size)

    def entropy(self):
        return np.log(self.b - self.a)

    def skewness(self):
        return 0.0

    def kurtosis(self):
        return -6.0 / 5.0

    @classmethod
    def fit(cls, data):
        return cls(float(np.min(data)), float(np.max(data)))


class Beta(Distribution):
    def __init__(self, a=1.0, b=1.0):
        self.a, self.b = float(a), float(b)

    def support(self):
        return (0, 1)

    def pdf(self, x):
        x = np.asarray(x, dtype=float)
        out = x ** (self.a - 1) * (1 - x) ** (self.b - 1) / special.beta(self.a, self.b)
        return np.where((x >= 0) & (x <= 1), out, 0.0)

    def cdf(self, x):
        x = np.asarray(x, dtype=float)
        return special.betainc(self.a, self.b, np.clip(x, 0, 1))

    def ppf(self, q):
        q = np.asarray(q, dtype=float)
        return special.betaincinv(self.a, self.b, q)

    def mean(self):
        return self.a / (self.a + self.b)

    def var(self):
        a, b = self.a, self.b
        return a * b / ((a + b) ** 2 * (a + b + 1))

    def rvs(self, size=1, random_state=None):
        rng = np.random.default_rng(random_state)
        return rng.beta(self.a, self.b, size=size)

    @classmethod
    def _initial_guess(cls, data):
        m, v = np.mean(data), np.var(data)
        common = m * (1 - m) / v - 1
        return [max(m * common, 0.5), max((1 - m) * common, 0.5)]

    @classmethod
    def _param_bounds(cls):
        return [(1e-3, None), (1e-3, None)]


class Gamma(Distribution):
    def __init__(self, shape=1.0, scale=1.0):
        if shape <= 0 or scale <= 0:
            raise ValueError("shape and scale must be > 0")
        self.shape, self.scale = float(shape), float(scale)

    def support(self):
        return (0, np.inf)

    def pdf(self, x):
        x = np.asarray(x, dtype=float)
        k, theta = self.shape, self.scale
        out = x ** (k - 1) * np.exp(-x / theta) / (special.gamma(k) * theta**k)
        return np.where(x >= 0, out, 0.0)

    def cdf(self, x):
        x = np.asarray(x, dtype=float)
        return np.where(x >= 0, special.gammainc(self.shape, x / self.scale), 0.0)

    def ppf(self, q):
        q = np.asarray(q, dtype=float)
        return self.scale * special.gammaincinv(self.shape, q)

    def mean(self):
        return self.shape * self.scale

    def var(self):
        return self.shape * self.scale**2

    def rvs(self, size=1, random_state=None):
        rng = np.random.default_rng(random_state)
        return rng.gamma(self.shape, self.scale, size=size)

    @classmethod
    def _initial_guess(cls, data):
        m, v = np.mean(data), np.var(data)
        return [m**2 / v, v / m]

    @classmethod
    def _param_bounds(cls):
        return [(1e-3, None), (1e-3, None)]


class Chi2(Distribution):
    def __init__(self, df=1.0):
        if df <= 0:
            raise ValueError("df must be > 0")
        self.df = float(df)

    def support(self):
        return (0, np.inf)

    def pdf(self, x):
        x = np.asarray(x, dtype=float)
        k = self.df
        out = x ** (k / 2 - 1) * np.exp(-x / 2) / (2 ** (k / 2) * special.gamma(k / 2))
        return np.where(x >= 0, out, 0.0)

    def cdf(self, x):
        x = np.asarray(x, dtype=float)
        return np.where(x >= 0, special.gammainc(self.df / 2, x / 2), 0.0)

    def ppf(self, q):
        q = np.asarray(q, dtype=float)
        return 2 * special.gammaincinv(self.df / 2, q)

    def mean(self):
        return self.df

    def var(self):
        return 2 * self.df

    def rvs(self, size=1, random_state=None):
        rng = np.random.default_rng(random_state)
        return rng.chisquare(self.df, size=size)

    @classmethod
    def fit(cls, data):
        return cls(float(np.mean(data)))


class Student_t(Distribution):
    def __init__(self, df=1.0):
        if df <= 0:
            raise ValueError("df must be > 0")
        self.df = float(df)

    def pdf(self, x):
        x = np.asarray(x, dtype=float)
        v = self.df
        coef = special.gamma((v + 1) / 2) / (np.sqrt(v * np.pi) * special.gamma(v / 2))
        return coef * (1 + x**2 / v) ** (-(v + 1) / 2)

    def cdf(self, x):
        return special.stdtr(self.df, np.asarray(x, dtype=float))

    def ppf(self, q):
        return special.stdtrit(self.df, np.asarray(q, dtype=float))

    def mean(self):
        return 0.0 if self.df > 1 else np.nan

    def var(self):
        if self.df > 2:
            return self.df / (self.df - 2)
        return np.inf

    def rvs(self, size=1, random_state=None):
        rng = np.random.default_rng(random_state)
        return rng.standard_t(self.df, size=size)

    def skewness(self):
        return 0.0 if self.df > 3 else np.nan

    @classmethod
    def _initial_guess(cls, data):
        v = np.var(data)
        df0 = 2 * v / (v - 1) if v > 1 else 5.0
        return [max(df0, 2.5)]

    @classmethod
    def _param_bounds(cls):
        return [(1e-3, None)]


class F(Distribution):
    def __init__(self, d1=1.0, d2=1.0):
        if d1 <= 0 or d2 <= 0:
            raise ValueError("d1 and d2 must be > 0")
        self.d1, self.d2 = float(d1), float(d2)

    def support(self):
        return (0, np.inf)

    def pdf(self, x):
        x = np.asarray(x, dtype=float)
        d1, d2 = self.d1, self.d2
        num = (d1 / d2) ** (d1 / 2) * x ** (d1 / 2 - 1)
        den = special.beta(d1 / 2, d2 / 2) * (1 + d1 * x / d2) ** ((d1 + d2) / 2)
        return np.where(x >= 0, num / den, 0.0)

    def cdf(self, x):
        x = np.asarray(x, dtype=float)
        return np.where(x >= 0, special.fdtr(self.d1, self.d2, x), 0.0)

    def ppf(self, q):
        return special.fdtri(self.d1, self.d2, np.asarray(q, dtype=float))

    def mean(self):
        return self.d2 / (self.d2 - 2) if self.d2 > 2 else np.inf

    def var(self):
        d1, d2 = self.d1, self.d2
        if d2 > 4:
            return 2 * d2**2 * (d1 + d2 - 2) / (d1 * (d2 - 2) ** 2 * (d2 - 4))
        return np.inf

    def rvs(self, size=1, random_state=None):
        rng = np.random.default_rng(random_state)
        return rng.f(self.d1, self.d2, size=size)

    @classmethod
    def _initial_guess(cls, data):
        return [5.0, 5.0]

    @classmethod
    def _param_bounds(cls):
        return [(1e-3, None), (1e-3, None)]


class Cauchy(Distribution):
    def __init__(self, loc=0.0, scale=1.0):
        if scale <= 0:
            raise ValueError("scale must be > 0")
        self.loc, self.scale = float(loc), float(scale)

    def pdf(self, x):
        x = np.asarray(x, dtype=float)
        return 1.0 / (np.pi * self.scale * (1 + ((x - self.loc) / self.scale) ** 2))

    def cdf(self, x):
        x = np.asarray(x, dtype=float)
        return 0.5 + np.arctan((x - self.loc) / self.scale) / np.pi

    def ppf(self, q):
        q = np.asarray(q, dtype=float)
        return self.loc + self.scale * np.tan(np.pi * (q - 0.5))

    def mean(self):
        return np.nan

    def var(self):
        return np.inf

    def skewness(self):
        return np.nan

    def kurtosis(self):
        return np.nan

    def rvs(self, size=1, random_state=None):
        rng = np.random.default_rng(random_state)
        return self.loc + self.scale * rng.standard_cauchy(size=size)

    def entropy(self):
        return np.log(4 * np.pi * self.scale)

    def cf(self, t):
        return complex(np.exp(1j * self.loc * t - self.scale * abs(t)))

    @classmethod
    def _initial_guess(cls, data):
        return [float(np.median(data)), float(np.subtract(*np.percentile(data, [75, 25])) / 2) or 1.0]

    @classmethod
    def _param_bounds(cls):
        return [(None, None), (1e-3, None)]


class Laplace(Distribution):
    def __init__(self, loc=0.0, scale=1.0):
        if scale <= 0:
            raise ValueError("scale must be > 0")
        self.loc, self.scale = float(loc), float(scale)

    def pdf(self, x):
        x = np.asarray(x, dtype=float)
        return np.exp(-np.abs(x - self.loc) / self.scale) / (2 * self.scale)

    def cdf(self, x):
        x = np.asarray(x, dtype=float)
        z = (x - self.loc) / self.scale
        return np.where(z >= 0, 1 - 0.5 * np.exp(-z), 0.5 * np.exp(z))

    def ppf(self, q):
        q = np.asarray(q, dtype=float)
        return self.loc - self.scale * np.sign(q - 0.5) * np.log(1 - 2 * np.abs(q - 0.5))

    def mean(self):
        return self.loc

    def var(self):
        return 2 * self.scale**2

    def rvs(self, size=1, random_state=None):
        rng = np.random.default_rng(random_state)
        return rng.laplace(self.loc, self.scale, size=size)

    def entropy(self):
        return np.log(2 * self.scale * np.e)

    def skewness(self):
        return 0.0

    def kurtosis(self):
        return 3.0

    def mgf(self, t):
        if abs(t) >= 1 / self.scale:
            return np.inf
        return float(np.exp(self.loc * t) / (1 - self.scale**2 * t**2))

    def cf(self, t):
        return complex(np.exp(1j * self.loc * t) / (1 + self.scale**2 * t**2))

    @classmethod
    def fit(cls, data):
        loc = float(np.median(data))
        scale = float(np.mean(np.abs(np.asarray(data) - loc)))
        return cls(loc, scale)


class Weibull(Distribution):
    def __init__(self, shape=1.0, scale=1.0):
        if shape <= 0 or scale <= 0:
            raise ValueError("shape and scale must be > 0")
        self.shape, self.scale = float(shape), float(scale)

    def support(self):
        return (0, np.inf)

    def pdf(self, x):
        x = np.asarray(x, dtype=float)
        k, lam = self.shape, self.scale
        out = (k / lam) * (x / lam) ** (k - 1) * np.exp(-((x / lam) ** k))
        return np.where(x >= 0, out, 0.0)

    def cdf(self, x):
        x = np.asarray(x, dtype=float)
        return np.where(x >= 0, 1 - np.exp(-((x / self.scale) ** self.shape)), 0.0)

    def ppf(self, q):
        q = np.asarray(q, dtype=float)
        return self.scale * (-np.log(1 - q)) ** (1 / self.shape)

    def mean(self):
        return self.scale * special.gamma(1 + 1 / self.shape)

    def var(self):
        k, lam = self.shape, self.scale
        return lam**2 * (special.gamma(1 + 2 / k) - special.gamma(1 + 1 / k) ** 2)

    def rvs(self, size=1, random_state=None):
        rng = np.random.default_rng(random_state)
        return self.scale * rng.weibull(self.shape, size=size)

    @classmethod
    def _initial_guess(cls, data):
        return [1.5, float(np.mean(data))]

    @classmethod
    def _param_bounds(cls):
        return [(1e-3, None), (1e-3, None)]


class Pareto(Distribution):
    def __init__(self, alpha=1.0, xm=1.0):
        if alpha <= 0 or xm <= 0:
            raise ValueError("alpha and xm must be > 0")
        self.alpha, self.xm = float(alpha), float(xm)

    def support(self):
        return (self.xm, np.inf)

    def pdf(self, x):
        x = np.asarray(x, dtype=float)
        out = self.alpha * self.xm**self.alpha / x ** (self.alpha + 1)
        return np.where(x >= self.xm, out, 0.0)

    def cdf(self, x):
        x = np.asarray(x, dtype=float)
        return np.where(x >= self.xm, 1 - (self.xm / x) ** self.alpha, 0.0)

    def ppf(self, q):
        q = np.asarray(q, dtype=float)
        return self.xm / (1 - q) ** (1 / self.alpha)

    def mean(self):
        return self.alpha * self.xm / (self.alpha - 1) if self.alpha > 1 else np.inf

    def var(self):
        a, xm = self.alpha, self.xm
        if a > 2:
            return xm**2 * a / ((a - 1) ** 2 * (a - 2))
        return np.inf

    def rvs(self, size=1, random_state=None):
        rng = np.random.default_rng(random_state)
        return self.xm * (1 + rng.pareto(self.alpha, size=size))

    def entropy(self):
        return np.log(self.xm / self.alpha) + 1 + 1 / self.alpha

    @classmethod
    def fit(cls, data):
        data = np.asarray(data, dtype=float)
        xm = float(np.min(data))
        alpha = len(data) / np.sum(np.log(data / xm))
        return cls(alpha, xm)


class LogNormal(Distribution):
    def __init__(self, mu=0.0, sigma=1.0):
        if sigma <= 0:
            raise ValueError("sigma must be > 0")
        self.mu, self.sigma = float(mu), float(sigma)

    def support(self):
        return (0, np.inf)

    def pdf(self, x):
        x = np.asarray(x, dtype=float)
        with np.errstate(divide="ignore", invalid="ignore"):
            out = np.exp(-((np.log(x) - self.mu) ** 2) / (2 * self.sigma**2)) / (
                x * self.sigma * np.sqrt(2 * np.pi)
            )
        return np.where(x > 0, out, 0.0)

    def cdf(self, x):
        x = np.asarray(x, dtype=float)
        with np.errstate(divide="ignore", invalid="ignore"):
            out = special.ndtr((np.log(x) - self.mu) / self.sigma)
        return np.where(x > 0, out, 0.0)

    def ppf(self, q):
        q = np.asarray(q, dtype=float)
        return np.exp(self.mu + self.sigma * special.ndtri(q))

    def mean(self):
        return float(np.exp(self.mu + self.sigma**2 / 2))

    def var(self):
        return float((np.exp(self.sigma**2) - 1) * np.exp(2 * self.mu + self.sigma**2))

    def rvs(self, size=1, random_state=None):
        rng = np.random.default_rng(random_state)
        return rng.lognormal(self.mu, self.sigma, size=size)

    @classmethod
    def fit(cls, data):
        log_data = np.log(np.asarray(data, dtype=float))
        return cls(float(np.mean(log_data)), float(np.std(log_data)))


class Gumbel(Distribution):
    def __init__(self, loc=0.0, scale=1.0):
        if scale <= 0:
            raise ValueError("scale must be > 0")
        self.loc, self.scale = float(loc), float(scale)

    def pdf(self, x):
        x = np.asarray(x, dtype=float)
        z = (x - self.loc) / self.scale
        return np.exp(-(z + np.exp(-z))) / self.scale

    def cdf(self, x):
        x = np.asarray(x, dtype=float)
        z = (x - self.loc) / self.scale
        return np.exp(-np.exp(-z))

    def ppf(self, q):
        q = np.asarray(q, dtype=float)
        return self.loc - self.scale * np.log(-np.log(q))

    def mean(self):
        return self.loc + self.scale * EULER_GAMMA

    def var(self):
        return (np.pi**2 / 6) * self.scale**2

    def rvs(self, size=1, random_state=None):
        rng = np.random.default_rng(random_state)
        return rng.gumbel(self.loc, self.scale, size=size)

    def entropy(self):
        return np.log(self.scale) + EULER_GAMMA + 1

    @classmethod
    def fit(cls, data):
        data = np.asarray(data, dtype=float)
        scale = float(np.std(data) * np.sqrt(6) / np.pi)
        loc = float(np.mean(data) - scale * EULER_GAMMA)
        return cls(loc, scale)


class Frechet(Distribution):
    """Inverse-Weibull parametrization."""

    def __init__(self, shape=1.0, scale=1.0):
        if shape <= 0 or scale <= 0:
            raise ValueError("shape and scale must be > 0")
        self.shape, self.scale = float(shape), float(scale)

    def support(self):
        return (0, np.inf)

    def pdf(self, x):
        x = np.asarray(x, dtype=float)
        a, s = self.shape, self.scale
        with np.errstate(divide="ignore", invalid="ignore"):
            out = (a / s) * (x / s) ** (-a - 1) * np.exp(-((x / s) ** -a))
        return np.where(x > 0, out, 0.0)

    def cdf(self, x):
        x = np.asarray(x, dtype=float)
        with np.errstate(divide="ignore", invalid="ignore"):
            out = np.exp(-((x / self.scale) ** -self.shape))
        return np.where(x > 0, out, 0.0)

    def ppf(self, q):
        q = np.asarray(q, dtype=float)
        return self.scale * (-np.log(q)) ** (-1 / self.shape)

    def mean(self):
        return self.scale * special.gamma(1 - 1 / self.shape) if self.shape > 1 else np.inf

    def var(self):
        a, s = self.shape, self.scale
        if a > 2:
            return s**2 * (special.gamma(1 - 2 / a) - special.gamma(1 - 1 / a) ** 2)
        return np.inf

    def rvs(self, size=1, random_state=None):
        rng = np.random.default_rng(random_state)
        return self.ppf(rng.uniform(size=size))

    @classmethod
    def _initial_guess(cls, data):
        return [2.0, float(np.mean(data))]

    @classmethod
    def _param_bounds(cls):
        return [(1e-3, None), (1e-3, None)]


class GEV(Distribution):
    """Generalized extreme value distribution."""

    def __init__(self, loc=0.0, scale=1.0, shape=0.0):
        if scale <= 0:
            raise ValueError("scale must be > 0")
        self.loc, self.scale, self.shape = float(loc), float(scale), float(shape)

    def _z(self, x):
        x = np.asarray(x, dtype=float)
        if self.shape == 0:
            return np.exp(-(x - self.loc) / self.scale)
        return np.maximum(1 + self.shape * (x - self.loc) / self.scale, 1e-300) ** (-1 / self.shape)

    def pdf(self, x):
        x = np.asarray(x, dtype=float)
        z = self._z(x)
        if self.shape == 0:
            return np.exp(-(x - self.loc) / self.scale) * np.exp(-z) / self.scale
        t = 1 + self.shape * (x - self.loc) / self.scale
        valid = t > 0
        out = np.where(valid, (z ** (1 + self.shape)) * np.exp(-z) / self.scale, 0.0)
        return out

    def cdf(self, x):
        return np.exp(-self._z(x))

    def ppf(self, q):
        q = np.asarray(q, dtype=float)
        if self.shape == 0:
            return self.loc - self.scale * np.log(-np.log(q))
        return self.loc + self.scale * ((-np.log(q)) ** (-self.shape) - 1) / self.shape

    def mean(self):
        if self.shape == 0:
            return self.loc + self.scale * EULER_GAMMA
        if self.shape < 1:
            return self.loc + self.scale * (special.gamma(1 - self.shape) - 1) / self.shape
        return np.inf

    def var(self):
        if self.shape == 0:
            return (np.pi**2 / 6) * self.scale**2
        if self.shape < 0.5:
            g1, g2 = special.gamma(1 - self.shape), special.gamma(1 - 2 * self.shape)
            return self.scale**2 * (g2 - g1**2) / self.shape**2
        return np.inf

    def rvs(self, size=1, random_state=None):
        rng = np.random.default_rng(random_state)
        return self.ppf(rng.uniform(size=size))

    @classmethod
    def _initial_guess(cls, data):
        return [float(np.mean(data)), float(np.std(data)), 0.1]

    @classmethod
    def _param_bounds(cls):
        return [(None, None), (1e-3, None), (-1.0, 1.0)]


class GPareto(Distribution):
    """Generalized Pareto distribution."""

    def __init__(self, loc=0.0, scale=1.0, shape=0.0):
        if scale <= 0:
            raise ValueError("scale must be > 0")
        self.loc, self.scale, self.shape = float(loc), float(scale), float(shape)

    def support(self):
        if self.shape >= 0:
            return (self.loc, np.inf)
        return (self.loc, self.loc - self.scale / self.shape)

    def pdf(self, x):
        x = np.asarray(x, dtype=float)
        z = (x - self.loc) / self.scale
        if self.shape == 0:
            return np.where(z >= 0, np.exp(-z) / self.scale, 0.0)
        t = 1 + self.shape * z
        return np.where((z >= 0) & (t > 0), t ** (-1 / self.shape - 1) / self.scale, 0.0)

    def cdf(self, x):
        x = np.asarray(x, dtype=float)
        z = (x - self.loc) / self.scale
        if self.shape == 0:
            return np.where(z >= 0, 1 - np.exp(-z), 0.0)
        t = np.maximum(1 + self.shape * z, 0)
        return np.where(z >= 0, 1 - t ** (-1 / self.shape), 0.0)

    def ppf(self, q):
        q = np.asarray(q, dtype=float)
        if self.shape == 0:
            return self.loc - self.scale * np.log(1 - q)
        return self.loc + self.scale * ((1 - q) ** (-self.shape) - 1) / self.shape

    def mean(self):
        return self.loc + self.scale / (1 - self.shape) if self.shape < 1 else np.inf

    def var(self):
        xi = self.shape
        if xi < 0.5:
            return self.scale**2 / ((1 - xi) ** 2 * (1 - 2 * xi))
        return np.inf

    def rvs(self, size=1, random_state=None):
        rng = np.random.default_rng(random_state)
        return self.ppf(rng.uniform(size=size))

    @classmethod
    def _initial_guess(cls, data):
        return [float(np.min(data)), float(np.std(data)), 0.1]

    @classmethod
    def _param_bounds(cls):
        return [(None, None), (1e-3, None), (-1.0 + 1e-6, 5.0)]


class InvGamma(Distribution):
    def __init__(self, shape=1.0, scale=1.0):
        if shape <= 0 or scale <= 0:
            raise ValueError("shape and scale must be > 0")
        self.shape, self.scale = float(shape), float(scale)

    def support(self):
        return (0, np.inf)

    def pdf(self, x):
        x = np.asarray(x, dtype=float)
        a, b = self.shape, self.scale
        with np.errstate(divide="ignore", invalid="ignore"):
            out = b**a / special.gamma(a) * x ** (-a - 1) * np.exp(-b / x)
        return np.where(x > 0, out, 0.0)

    def cdf(self, x):
        x = np.asarray(x, dtype=float)
        with np.errstate(divide="ignore", invalid="ignore"):
            out = special.gammaincc(self.shape, self.scale / x)
        return np.where(x > 0, out, 0.0)

    def ppf(self, q):
        q = np.asarray(q, dtype=float)
        return self.scale / special.gammainccinv(self.shape, q)

    def mean(self):
        return self.scale / (self.shape - 1) if self.shape > 1 else np.inf

    def var(self):
        a, b = self.shape, self.scale
        return b**2 / ((a - 1) ** 2 * (a - 2)) if a > 2 else np.inf

    def rvs(self, size=1, random_state=None):
        rng = np.random.default_rng(random_state)
        return 1.0 / rng.gamma(self.shape, 1.0 / self.scale, size=size)

    @classmethod
    def _initial_guess(cls, data):
        m, v = np.mean(data), np.var(data)
        a = m**2 / v + 2
        b = m * (a - 1)
        return [a, b]

    @classmethod
    def _param_bounds(cls):
        return [(1e-3, None), (1e-3, None)]


class InvGaussian(Distribution):
    """Wald / inverse-Gaussian distribution."""

    def __init__(self, mu=1.0, lam=1.0):
        if mu <= 0 or lam <= 0:
            raise ValueError("mu and lam must be > 0")
        self.mu, self.lam = float(mu), float(lam)

    def support(self):
        return (0, np.inf)

    def pdf(self, x):
        x = np.asarray(x, dtype=float)
        mu, lam = self.mu, self.lam
        with np.errstate(divide="ignore", invalid="ignore"):
            out = np.sqrt(lam / (2 * np.pi * x**3)) * np.exp(-lam * (x - mu) ** 2 / (2 * mu**2 * x))
        return np.where(x > 0, out, 0.0)

    def cdf(self, x):
        x = np.asarray(x, dtype=float)
        mu, lam = self.mu, self.lam
        with np.errstate(divide="ignore", invalid="ignore"):
            a = special.ndtr(np.sqrt(lam / x) * (x / mu - 1))
            b = np.exp(2 * lam / mu) * special.ndtr(-np.sqrt(lam / x) * (x / mu + 1))
        return np.where(x > 0, a + b, 0.0)

    def mean(self):
        return self.mu

    def var(self):
        return self.mu**3 / self.lam

    def rvs(self, size=1, random_state=None):
        rng = np.random.default_rng(random_state)
        return rng.wald(self.mu, self.lam, size=size)

    @classmethod
    def fit(cls, data):
        data = np.asarray(data, dtype=float)
        mu = float(np.mean(data))
        lam = 1.0 / (float(np.mean(1.0 / data)) - 1.0 / mu)
        return cls(mu, lam)


class Rayleigh(Distribution):
    def __init__(self, scale=1.0):
        if scale <= 0:
            raise ValueError("scale must be > 0")
        self.scale = float(scale)

    def support(self):
        return (0, np.inf)

    def pdf(self, x):
        x = np.asarray(x, dtype=float)
        s = self.scale
        return np.where(x >= 0, (x / s**2) * np.exp(-(x**2) / (2 * s**2)), 0.0)

    def cdf(self, x):
        x = np.asarray(x, dtype=float)
        return np.where(x >= 0, 1 - np.exp(-(x**2) / (2 * self.scale**2)), 0.0)

    def ppf(self, q):
        q = np.asarray(q, dtype=float)
        return self.scale * np.sqrt(-2 * np.log(1 - q))

    def mean(self):
        return self.scale * np.sqrt(np.pi / 2)

    def var(self):
        return (4 - np.pi) / 2 * self.scale**2

    def rvs(self, size=1, random_state=None):
        rng = np.random.default_rng(random_state)
        return rng.rayleigh(self.scale, size=size)

    def entropy(self):
        return 1 + np.log(self.scale / np.sqrt(2)) + EULER_GAMMA / 2

    @classmethod
    def fit(cls, data):
        data = np.asarray(data, dtype=float)
        return cls(float(np.sqrt(np.mean(data**2) / 2)))


class Maxwell(Distribution):
    """Maxwell-Boltzmann speed distribution."""

    def __init__(self, scale=1.0):
        if scale <= 0:
            raise ValueError("scale must be > 0")
        self.scale = float(scale)

    def support(self):
        return (0, np.inf)

    def pdf(self, x):
        x = np.asarray(x, dtype=float)
        a = self.scale
        out = np.sqrt(2 / np.pi) * x**2 * np.exp(-(x**2) / (2 * a**2)) / a**3
        return np.where(x >= 0, out, 0.0)

    def cdf(self, x):
        x = np.asarray(x, dtype=float)
        return np.where(x >= 0, special.gammainc(1.5, x**2 / (2 * self.scale**2)), 0.0)

    def ppf(self, q):
        q = np.asarray(q, dtype=float)
        return self.scale * np.sqrt(2 * special.gammaincinv(1.5, q))

    def mean(self):
        return 2 * self.scale * np.sqrt(2 / np.pi)

    def var(self):
        return self.scale**2 * (3 * np.pi - 8) / np.pi

    def rvs(self, size=1, random_state=None):
        rng = np.random.default_rng(random_state)
        comps = rng.normal(0, self.scale, size=(np.prod(size, dtype=int) if hasattr(size, "__iter__") else size, 3))
        out = np.sqrt(np.sum(comps**2, axis=-1))
        return out.reshape(size) if hasattr(size, "__iter__") else out

    @classmethod
    def fit(cls, data):
        data = np.asarray(data, dtype=float)
        return cls(float(np.sqrt(np.mean(data**2) / 3)))


class Nakagami(Distribution):
    def __init__(self, shape=1.0, scale=1.0):
        if shape <= 0 or scale <= 0:
            raise ValueError("shape (m) and scale (omega) must be > 0")
        self.shape, self.scale = float(shape), float(scale)

    def support(self):
        return (0, np.inf)

    def pdf(self, x):
        x = np.asarray(x, dtype=float)
        m, w = self.shape, self.scale
        out = (2 * m**m / (special.gamma(m) * w**m)) * x ** (2 * m - 1) * np.exp(-m * x**2 / w)
        return np.where(x >= 0, out, 0.0)

    def cdf(self, x):
        x = np.asarray(x, dtype=float)
        m, w = self.shape, self.scale
        return np.where(x >= 0, special.gammainc(m, m * x**2 / w), 0.0)

    def ppf(self, q):
        q = np.asarray(q, dtype=float)
        m, w = self.shape, self.scale
        return np.sqrt(w / m * special.gammaincinv(m, q))

    def mean(self):
        m, w = self.shape, self.scale
        return (special.gamma(m + 0.5) / special.gamma(m)) * np.sqrt(w / m)

    def var(self):
        m, w = self.shape, self.scale
        return w * (1 - (1 / m) * (special.gamma(m + 0.5) / special.gamma(m)) ** 2)

    def rvs(self, size=1, random_state=None):
        rng = np.random.default_rng(random_state)
        m, w = self.shape, self.scale
        return np.sqrt(rng.gamma(m, w / m, size=size))

    @classmethod
    def _initial_guess(cls, data):
        return [1.0, float(np.mean(np.asarray(data) ** 2))]

    @classmethod
    def _param_bounds(cls):
        return [(1e-3, None), (1e-3, None)]


class Rice(Distribution):
    def __init__(self, nu=0.0, sigma=1.0):
        if nu < 0 or sigma <= 0:
            raise ValueError("nu must be >= 0 and sigma must be > 0")
        self.nu, self.sigma = float(nu), float(sigma)

    def support(self):
        return (0, np.inf)

    def pdf(self, x):
        x = np.asarray(x, dtype=float)
        nu, s = self.nu, self.sigma
        with np.errstate(divide="ignore", invalid="ignore", over="ignore"):
            # evaluated in log space: log I0(y) = y + log(i0e(y)) avoids the overflow to NaN
            # of the naive form (I0 explodes like e^y while the Gaussian factor decays).
            y = x * nu / s**2
            logpdf = (
                np.log(x / s**2)
                - (x**2 + nu**2) / (2 * s**2)
                + y
                + np.log(special.i0e(y))
            )
            out = np.exp(logpdf)
        return np.where(x >= 0, out, 0.0)

    def cdf(self, x):
        x = np.asarray(x, dtype=float)
        nc = (self.nu / self.sigma) ** 2
        return np.where(x >= 0, special.chndtr((x / self.sigma) ** 2, 2, nc), 0.0)

    def mean(self):
        nu, s = self.nu, self.sigma
        x = -(nu**2) / (2 * s**2)
        L = special.eval_genlaguerre(0.5, 0, x)
        return s * np.sqrt(np.pi / 2) * L

    def var(self):
        return 2 * self.sigma**2 + self.nu**2 - (np.pi * self.sigma**2 / 2) * (self.mean() / (self.sigma * np.sqrt(np.pi / 2))) ** 2 if self.sigma > 0 else np.nan

    def rvs(self, size=1, random_state=None):
        rng = np.random.default_rng(random_state)
        z1 = rng.normal(self.nu, self.sigma, size=size)
        z2 = rng.normal(0, self.sigma, size=size)
        return np.sqrt(z1**2 + z2**2)

    @classmethod
    def _initial_guess(cls, data):
        return [float(np.mean(data)), float(np.std(data)) or 1.0]

    @classmethod
    def _param_bounds(cls):
        return [(0.0, None), (1e-3, None)]


class VonMises(Distribution):
    """Circular distribution on (-pi, pi]."""

    def __init__(self, mu=0.0, kappa=1.0):
        if kappa < 0:
            raise ValueError("kappa must be >= 0")
        self.mu, self.kappa = float(mu), float(kappa)

    def support(self):
        return (-np.pi, np.pi)

    def pdf(self, x):
        x = np.asarray(x, dtype=float)
        return np.exp(self.kappa * np.cos(x - self.mu)) / (2 * np.pi * special.i0(self.kappa))

    def cdf(self, x):
        x = np.asarray(x, dtype=float)
        scalar = x.ndim == 0
        x = np.atleast_1d(x)
        out = np.array(
            [integrate.quad(self.pdf, -np.pi, min(max(xi, -np.pi), np.pi), limit=200)[0] for xi in x]
        )
        out = np.where(x >= np.pi, 1.0, out)
        return out.item() if scalar else out

    def mean(self):
        return self.mu

    def var(self):
        """Circular variance ``1 - I1(kappa)/I0(kappa)`` (the standard dispersion measure for
        circular data) — deliberately *not* the ordinary linear variance over [-pi, pi]."""
        return 1 - special.i1(self.kappa) / special.i0(self.kappa)

    def entropy(self):
        return np.log(2 * np.pi * special.i0(self.kappa)) - self.kappa * special.i1(self.kappa) / special.i0(self.kappa)

    def rvs(self, size=1, random_state=None):
        rng = np.random.default_rng(random_state)
        return rng.vonmises(self.mu, self.kappa, size=size)

    @classmethod
    def fit(cls, data):
        data = np.asarray(data, dtype=float)
        c, s = np.mean(np.cos(data)), np.mean(np.sin(data))
        mu = np.arctan2(s, c)
        r = np.sqrt(c**2 + s**2)
        kappa = optimize.brentq(lambda k: special.i1(k) / special.i0(k) - r, 1e-8, 1e4)
        return cls(float(mu), float(kappa))


class Kumaraswamy(Distribution):
    def __init__(self, a=1.0, b=1.0):
        if a <= 0 or b <= 0:
            raise ValueError("a and b must be > 0")
        self.a, self.b = float(a), float(b)

    def support(self):
        return (0, 1)

    def pdf(self, x):
        x = np.asarray(x, dtype=float)
        a, b = self.a, self.b
        out = a * b * x ** (a - 1) * (1 - x**a) ** (b - 1)
        return np.where((x >= 0) & (x <= 1), out, 0.0)

    def cdf(self, x):
        x = np.asarray(x, dtype=float)
        return np.clip(1 - (1 - np.clip(x, 0, 1) ** self.a) ** self.b, 0.0, 1.0)

    def ppf(self, q):
        q = np.asarray(q, dtype=float)
        return (1 - (1 - q) ** (1 / self.b)) ** (1 / self.a)

    def mean(self):
        return self.b * special.beta(1 + 1 / self.a, self.b)

    def var(self):
        m2 = self.b * special.beta(1 + 2 / self.a, self.b)
        return m2 - self.mean() ** 2

    def rvs(self, size=1, random_state=None):
        rng = np.random.default_rng(random_state)
        return self.ppf(rng.uniform(size=size))

    @classmethod
    def _initial_guess(cls, data):
        return [1.5, 1.5]

    @classmethod
    def _param_bounds(cls):
        return [(1e-3, None), (1e-3, None)]
