"""Multivariate distributions, implemented from scratch on top of
:mod:`stochpylib.distributions._base`. See ``MultivariateDistribution`` for the shared
``NotImplementedError`` convention on ``ppf``/``skewness``/``kurtosis``/``ks_test``.
"""

import numpy as np
from scipy import special

from stochpylib.distributions._base import MultivariateDistribution


class MultivariateNormal(MultivariateDistribution):
    def __init__(self, mean, cov):
        self.mean_vec = np.asarray(mean, dtype=float)
        self.cov_matrix = np.asarray(cov, dtype=float)
        self.k = len(self.mean_vec)
        self._L = np.linalg.cholesky(self.cov_matrix)

    def pdf(self, x):
        x = np.asarray(x, dtype=float)
        diff = x - self.mean_vec
        inv = np.linalg.inv(self.cov_matrix)
        det = np.linalg.det(self.cov_matrix)
        exponent = -0.5 * diff @ inv @ diff
        return float(np.exp(exponent) / np.sqrt((2 * np.pi) ** self.k * det))

    def mean(self):
        return self.mean_vec

    def var(self):
        return self.cov_matrix

    def rvs(self, size=1, random_state=None):
        rng = np.random.default_rng(random_state)
        n = size if isinstance(size, int) else int(np.prod(size))
        z = rng.normal(size=(n, self.k))
        out = self.mean_vec + z @ self._L.T
        return out[0] if size == 1 else out

    def entropy(self):
        det = np.linalg.det(self.cov_matrix)
        return 0.5 * np.log((2 * np.pi * np.e) ** self.k * det)

    @classmethod
    def fit(cls, data):
        data = np.asarray(data, dtype=float)
        return cls(np.mean(data, axis=0), np.cov(data, rowvar=False))


class Dirichlet(MultivariateDistribution):
    def __init__(self, alpha):
        self.alpha = np.asarray(alpha, dtype=float)
        if np.any(self.alpha <= 0):
            raise ValueError("all alpha must be > 0")

    def pdf(self, x):
        x = np.asarray(x, dtype=float)
        a = self.alpha
        B = np.prod(special.gamma(a)) / special.gamma(np.sum(a))
        return float(np.prod(x ** (a - 1)) / B)

    def mean(self):
        return self.alpha / np.sum(self.alpha)

    def var(self):
        a0 = np.sum(self.alpha)
        a = self.alpha
        cov = -np.outer(a, a) / (a0**2 * (a0 + 1))
        np.fill_diagonal(cov, a * (a0 - a) / (a0**2 * (a0 + 1)))
        return cov

    def rvs(self, size=1, random_state=None):
        rng = np.random.default_rng(random_state)
        return rng.dirichlet(self.alpha, size=size)

    def entropy(self):
        a = self.alpha
        a0 = np.sum(a)
        log_B = np.sum(special.gammaln(a)) - special.gammaln(a0)
        return float(
            log_B
            + (a0 - len(a)) * special.digamma(a0)
            - np.sum((a - 1) * special.digamma(a))
        )

    @classmethod
    def fit(cls, data):
        data = np.asarray(data, dtype=float)
        m = np.mean(data, axis=0)
        v = np.mean(np.var(data, axis=0))
        a0 = np.mean(m * (1 - m)) / v - 1 if v > 0 else 1.0
        a0 = max(a0, 1e-3)
        return cls(m * a0)


class Wishart(MultivariateDistribution):
    """Wishart(df, scale). ``var()`` returns the per-entry variance matrix
    ``Var(W_ij) = df * (scale_ij^2 + scale_ii * scale_jj)`` — a documented simplification, not
    the full 4-index covariance tensor (which is not represented as a 2D array)."""

    def __init__(self, df, scale):
        self.df = float(df)
        self.scale = np.asarray(scale, dtype=float)
        self.p = self.scale.shape[0]

    def pdf(self, x):
        x = np.asarray(x, dtype=float)
        n, V, p = self.df, self.scale, self.p
        Vinv = np.linalg.inv(V)
        detV, detX = np.linalg.det(V), np.linalg.det(x)
        log_pdf = (
            ((n - p - 1) / 2) * np.log(detX)
            - 0.5 * np.trace(Vinv @ x)
            - (n * p / 2) * np.log(2)
            - (n / 2) * np.log(detV)
            - special.multigammaln(n / 2, p)
        )
        return float(np.exp(log_pdf))

    def mean(self):
        return self.df * self.scale

    def var(self):
        V, n = self.scale, self.df
        diag = np.diag(V)
        return n * (V**2 + np.outer(diag, diag))

    def rvs(self, size=1, random_state=None):
        rng = np.random.default_rng(random_state)
        n = size if isinstance(size, int) else int(np.prod(size))
        out = np.empty((n, self.p, self.p))
        for i in range(n):
            x = rng.multivariate_normal(np.zeros(self.p), self.scale, size=int(self.df))
            out[i] = x.T @ x
        return out[0] if size == 1 else out

    @classmethod
    def fit(cls, data, df=None):
        data = np.asarray(data, dtype=float)
        if df is None:
            raise ValueError("Wishart.fit requires the degrees of freedom df")
        mean_matrix = np.mean(data, axis=0)
        return cls(df, mean_matrix / df)


class InverseWishart(MultivariateDistribution):
    def __init__(self, df, scale):
        self.df = float(df)
        self.scale = np.asarray(scale, dtype=float)
        self.p = self.scale.shape[0]

    def pdf(self, x):
        x = np.asarray(x, dtype=float)
        n, Psi, p = self.df, self.scale, self.p
        detPsi, detX = np.linalg.det(Psi), np.linalg.det(x)
        log_pdf = (
            (n / 2) * np.log(detPsi)
            - ((n + p + 1) / 2) * np.log(detX)
            - 0.5 * np.trace(Psi @ np.linalg.inv(x))
            - (n * p / 2) * np.log(2)
            - special.multigammaln(n / 2, p)
        )
        return float(np.exp(log_pdf))

    def mean(self):
        p = self.p
        return self.scale / (self.df - p - 1) if self.df > p + 1 else np.full_like(self.scale, np.inf)

    def var(self):
        return self.scale**2 / max(self.df - self.p - 1, 1e-6) ** 2

    def rvs(self, size=1, random_state=None):
        rng = np.random.default_rng(random_state)
        n = size if isinstance(size, int) else int(np.prod(size))
        out = np.empty((n, self.p, self.p))
        scale_inv = np.linalg.inv(self.scale)
        for i in range(n):
            x = rng.multivariate_normal(np.zeros(self.p), scale_inv, size=int(self.df))
            w = x.T @ x
            out[i] = np.linalg.inv(w)
        return out[0] if size == 1 else out

    @classmethod
    def fit(cls, data, df=None):
        data = np.asarray(data, dtype=float)
        if df is None:
            raise ValueError("InverseWishart.fit requires the degrees of freedom df")
        p = data.shape[-1]
        mean_matrix = np.mean(data, axis=0)
        return cls(df, mean_matrix * (df - p - 1))


class MultivariateT(MultivariateDistribution):
    def __init__(self, df, loc, shape):
        self.df = float(df)
        self.loc = np.asarray(loc, dtype=float)
        self.shape_matrix = np.asarray(shape, dtype=float)
        self.k = len(self.loc)

    def pdf(self, x):
        x = np.asarray(x, dtype=float)
        v, mu, Sigma, p = self.df, self.loc, self.shape_matrix, self.k
        diff = x - mu
        inv = np.linalg.inv(Sigma)
        det = np.linalg.det(Sigma)
        quad = diff @ inv @ diff
        num = special.gamma((v + p) / 2)
        den = special.gamma(v / 2) * (v * np.pi) ** (p / 2) * np.sqrt(det) * (1 + quad / v) ** ((v + p) / 2)
        return float(num / den)

    def mean(self):
        return self.loc if self.df > 1 else np.full(self.k, np.nan)

    def var(self):
        return self.shape_matrix * self.df / (self.df - 2) if self.df > 2 else np.full_like(self.shape_matrix, np.inf)

    def rvs(self, size=1, random_state=None):
        rng = np.random.default_rng(random_state)
        n = size if isinstance(size, int) else int(np.prod(size))
        L = np.linalg.cholesky(self.shape_matrix)
        z = rng.normal(size=(n, self.k)) @ L.T
        u = rng.chisquare(self.df, size=n) / self.df
        out = self.loc + z / np.sqrt(u)[:, None]
        return out[0] if size == 1 else out

    @classmethod
    def fit(cls, data, df=None):
        data = np.asarray(data, dtype=float)
        if df is None:
            df = 5.0
        loc = np.mean(data, axis=0)
        shape = np.cov(data, rowvar=False) * (df - 2) / df if df > 2 else np.cov(data, rowvar=False)
        return cls(df, loc, shape)


class MultivariatePareto(MultivariateDistribution):
    """Mardia Type II multivariate Pareto: joint survival
    ``S(x) = (1 + sum((x_i - loc_i) / scale_i))^-alpha`` for ``x_i >= loc_i``.

    This is a specific, named, standard formulation chosen for "MultivariatePareto" — it isn't a
    single universal definition in the literature.
    """

    def __init__(self, alpha, loc, scale):
        self.alpha = float(alpha)
        self.loc = np.asarray(loc, dtype=float)
        self.scale = np.asarray(scale, dtype=float)
        self.d = len(self.loc)

    def pdf(self, x):
        x = np.asarray(x, dtype=float)
        a, mu, theta, d = self.alpha, self.loc, self.scale, self.d
        if np.any(x < mu):
            return 0.0
        z = 1 + np.sum((x - mu) / theta)
        coef = np.prod([a + i for i in range(d)]) / np.prod(theta)
        return float(coef * z ** (-(a + d)))

    def mean(self):
        return self.loc + self.scale / (self.alpha - 1) if self.alpha > 1 else np.full(self.d, np.inf)

    def var(self):
        a = self.alpha
        if a <= 2:
            return np.full((self.d, self.d), np.inf)
        theta = self.scale
        cov = np.outer(theta, theta) / ((a - 1) ** 2 * (a - 2))
        np.fill_diagonal(cov, theta**2 * a / ((a - 1) ** 2 * (a - 2)))
        return cov

    def rvs(self, size=1, random_state=None):
        rng = np.random.default_rng(random_state)
        n = size if isinstance(size, int) else int(np.prod(size))
        g = rng.gamma(self.alpha, 1.0, size=n)
        e = rng.exponential(1.0, size=(n, self.d))
        out = self.loc + self.scale * (e / g[:, None])
        return out[0] if size == 1 else out

    @classmethod
    def fit(cls, data):
        data = np.asarray(data, dtype=float)
        loc = np.min(data, axis=0)
        m = np.mean(data - loc, axis=0)
        v = np.var(data - loc, axis=0)
        alpha = np.mean(m**2 / v) + 2
        scale = m * (alpha - 1)
        return cls(float(alpha), loc, scale)
