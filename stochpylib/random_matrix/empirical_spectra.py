"""Limiting spectral laws and tridiagonal beta-ensembles.

``WignerSemicircle``, ``MarchenkoPastur`` and ``TracyWidomDistribution`` subclass
:class:`stochpylib.distributions.Distribution`, so they carry the library-wide 13-method
contract (pdf/cdf/ppf/rvs/mean/var/skewness/kurtosis/entropy/mgf/cf/fit/ks_test) and can
be handed to anything that consumes a distribution object. Each additionally offers
``compare(eigenvalues)`` (a KS test returning ``stochpylib.statistics.TestResult``) and
``histogram_vs_density(eigenvalues)`` -- the plotting-free replacement for the design
spec's ``plot_vs_empirical()``.
"""

import numpy as np
from scipy import integrate, special

from stochpylib.distributions import Beta
from stochpylib.distributions._base import Distribution
from stochpylib.random_matrix._common import (
    _GridInterp,
    _MEAN_RATIO_REFERENCE,
    _as_1d,
    _grid_law,
    _hermitian_eigvals,
    _rng,
)

__all__ = [
    "BetaEnsemble",
    "JacobiEnsemble",
    "MarchenkoPastur",
    "TracyWidomDistribution",
    "WignerSemicircle",
]


class _SpectralLaw(Distribution):
    """Common ``compare`` / ``histogram_vs_density`` surface for the limit laws."""

    def compare(self, eigenvalues):
        """Kolmogorov-Smirnov comparison of an empirical spectrum against this law.

        Returns a :class:`stochpylib.statistics.TestResult` (asymptotic p-value).
        """
        from stochpylib.statistics import TestResult

        eig = np.sort(_as_1d(eigenvalues).real.astype(float))
        stat, p = self.ks_test(eig)
        return TestResult(statistic=float(stat), pvalue=float(p),
                          null=f"spectrum follows {type(self).__name__}",
                          method="Kolmogorov-Smirnov", extras={"n": int(eig.size)})

    def histogram_vs_density(self, eigenvalues, bins=50):
        """``(centers, empirical_density, theoretical_density)`` on a common grid."""
        eig = _as_1d(eigenvalues).real.astype(float)
        counts, edges = np.histogram(eig, bins=bins, density=True)
        centers = 0.5 * (edges[1:] + edges[:-1])
        return centers, counts, np.asarray(self.pdf(centers), dtype=float)


class WignerSemicircle(_SpectralLaw):
    """Wigner's semicircle law on ``[-radius, radius]``.

    The bulk limit of ``eigs / sqrt(n)`` for Wigner matrices whose off-diagonal entries
    have unit variance is the semicircle with ``radius=2``. Even moments are Catalan
    numbers times ``(radius/2)**(2k)``.
    """

    def __init__(self, radius=2.0):
        if radius <= 0:
            raise ValueError("radius must be positive")
        self.radius = float(radius)

    def support(self):
        return (-self.radius, self.radius)

    def pdf(self, x):
        x = np.asarray(x, dtype=float)
        r = self.radius
        inside = np.abs(x) <= r
        out = np.zeros_like(x, dtype=float)
        out[inside] = 2.0 / (np.pi * r * r) * np.sqrt(r * r - x[inside] ** 2)
        return out if out.ndim else float(out)

    def cdf(self, x):
        x = np.asarray(x, dtype=float)
        t = np.clip(x / self.radius, -1.0, 1.0)
        out = 0.5 + (t * np.sqrt(1.0 - t * t) + np.arcsin(t)) / np.pi
        return out if out.ndim else float(out)

    def rvs(self, size=1, random_state=None):
        # semicircle(R) = R * (2 * Beta(3/2, 3/2) - 1)
        b = np.asarray(Beta(1.5, 1.5).rvs(size, random_state=random_state), dtype=float)
        return self.radius * (2.0 * b - 1.0)

    def mean(self):
        return 0.0

    def var(self):
        return self.radius ** 2 / 4.0

    def skewness(self):
        return 0.0

    def kurtosis(self):
        return -1.0  # excess kurtosis of the semicircle

    def moment(self, k):
        """Raw moment E[X^k]: zero for odd k, Catalan(k/2) * (R/2)^k for even k."""
        if k % 2:
            return 0.0
        m = k // 2
        catalan = special.comb(2 * m, m, exact=True) // (m + 1)
        return float(catalan * (self.radius / 2.0) ** k)

    def entropy(self):
        return float(np.log(np.pi * self.radius) - 0.5)

    def mgf(self, t):
        t = float(t)
        if t == 0.0:
            return 1.0
        z = self.radius * t
        return float(2.0 * special.iv(1, z) / z)

    def cf(self, t):
        t = float(t)
        if t == 0.0:
            return 1.0 + 0.0j
        z = self.radius * t
        return complex(2.0 * special.jv(1, z) / z)

    @classmethod
    def fit(cls, data):
        data = np.asarray(data, dtype=float)
        return cls(radius=float(np.sqrt(4.0 * np.mean(data ** 2))))


class MarchenkoPastur(_SpectralLaw):
    """Marchenko-Pastur law with aspect ratio ``gamma = p/n`` and variance ``sigma**2``.

    The bulk limit of ``eigs / n`` for ``W = X.T @ X`` with ``X`` an ``n x p`` matrix of
    i.i.d. ``(0, sigma**2)`` entries. Support ``[sigma^2 (1-sqrt(gamma))^2,
    sigma^2 (1+sqrt(gamma))^2]``; for ``gamma > 1`` an atom of mass ``1 - 1/gamma`` sits
    at zero (``W`` is rank-deficient).
    """

    def __init__(self, gamma, sigma=1.0):
        if gamma <= 0 or sigma <= 0:
            raise ValueError("gamma and sigma must be positive")
        self.gamma = float(gamma)
        self.sigma = float(sigma)
        s2 = self.sigma ** 2
        self.lam_minus = s2 * (1.0 - np.sqrt(self.gamma)) ** 2
        self.lam_plus = s2 * (1.0 + np.sqrt(self.gamma)) ** 2
        self.atom = max(0.0, 1.0 - 1.0 / self.gamma)
        self._table = None

    def support(self):
        return (0.0 if self.atom > 0 else self.lam_minus, self.lam_plus)

    def _density(self, x):
        x = np.asarray(x, dtype=float)
        inside = (x > self.lam_minus) & (x < self.lam_plus)
        out = np.zeros_like(x, dtype=float)
        xi = x[inside]
        out[inside] = np.sqrt((self.lam_plus - xi) * (xi - self.lam_minus)) / (
            2.0 * np.pi * self.sigma ** 2 * self.gamma * xi)
        return out

    def pdf(self, x):
        """Continuous part of the density (the atom at zero, if any, is not a density)."""
        out = self._density(x)
        return out if out.ndim else float(out)

    def _grid(self):
        if self._table is None:
            x, F = _grid_law(self._density, self.lam_minus, self.lam_plus, 6001)
            # analytic total mass of the continuous part: min(1, 1/gamma)
            F = F * (min(1.0, 1.0 / self.gamma) / F[-1])
            self._table = _GridInterp(x, F)
        return self._table

    def cdf(self, x):
        x = np.asarray(x, dtype=float)
        out = self.atom * (x >= 0.0) + self._grid().cdf(x)
        return out if out.ndim else float(out)

    def ppf(self, q):
        q = np.asarray(q, dtype=float)
        cont = self._grid().ppf(np.clip(q - self.atom, 0.0, None))
        out = np.where(q <= self.atom, 0.0, cont)
        return out if out.ndim else float(out)

    def rvs(self, size=1, random_state=None):
        u = _rng(random_state).uniform(0.0, 1.0, size=size)
        return self.ppf(u)

    def mean(self):
        return self.sigma ** 2

    def var(self):
        return self.gamma * self.sigma ** 4

    def moment(self, k):
        """Raw moment: sigma^(2k) * sum_r Narayana(k, r+1) gamma^r (r = 0..k-1)."""
        k = int(k)
        if k == 0:
            return 1.0
        total = 0.0
        for r in range(k):
            narayana = special.comb(k, r, exact=True) * special.comb(k, r + 1, exact=True) // k
            total += narayana * self.gamma ** r
        return float(self.sigma ** (2 * k) * total)

    @classmethod
    def fit(cls, data):
        data = np.asarray(data, dtype=float)
        m, v = float(np.mean(data)), float(np.var(data))
        return cls(gamma=v / (m * m), sigma=float(np.sqrt(m)))

    def ks_test(self, data):
        """KS statistic honouring the atom at zero (ties are compared to F(0-) and F(0))."""
        data = np.sort(np.asarray(data, dtype=float))
        n = data.size
        vals, first = np.unique(data, return_index=True)
        counts = np.diff(np.append(first, n))
        f_right = np.cumsum(counts) / n
        f_left = f_right - counts / n
        F = np.asarray(self.cdf(vals), dtype=float)
        F_left = F - self.atom * (vals == 0.0)
        d_stat = float(max(np.max(np.abs(f_right - F)), np.max(np.abs(f_left - F_left))))
        lam = max((np.sqrt(n) + 0.12 + 0.11 / np.sqrt(n)) * d_stat, 0.0)
        p_value = 2.0 * sum((-1) ** (k - 1) * np.exp(-2.0 * k ** 2 * lam ** 2)
                            for k in range(1, 101))
        return d_stat, float(np.clip(p_value, 0.0, 1.0))


class TracyWidomDistribution(_SpectralLaw):
    """Tracy-Widom law of the scaled largest eigenvalue, ``beta`` in {1, 2, 4}.

    Built from the Hastings-McLeod solution of Painleve II ``q'' = s q + 2 q^3``
    (``q ~ Ai(s)`` as ``s -> +inf``): ``F2(s) = exp(-int_s^inf (x-s) q^2)``,
    ``F1 = sqrt(F2) exp(-int_s^inf q / 2)``, ``F4(s) = sqrt(F2(t)) cosh(int_t^inf q / 2)``
    with ``t = s * sqrt(2)`` -- the convention with mean -2.3069 / variance 0.5177 for
    ``beta=4`` (mean -1.2065 / var 1.6078 for ``beta=1``, -1.7711 / 0.8132 for ``beta=2``).
    Tables are solved once per ``beta`` on ``[-10, 8]`` and cached on the class.
    """

    _S_LO, _S_HI = -10.0, 8.0
    _cache = {}

    def __init__(self, beta=2):
        if beta not in (1, 2, 4):
            raise ValueError("beta must be 1, 2 or 4")
        self.beta = int(beta)

    @classmethod
    def _painleve_tables(cls):
        if "raw" in cls._cache:
            return cls._cache["raw"]
        s0, s1 = cls._S_HI, cls._S_LO
        ai, aip, _, _ = special.airy(s0)

        def rhs(s, y):
            q, qp, iq2, iq, ixq2 = y
            return [qp, s * q + 2.0 * q ** 3, -q * q, -q, -s * q * q]

        sol = integrate.solve_ivp(rhs, (s0, s1), [ai, aip, 0.0, 0.0, 0.0], method="DOP853",
                                  rtol=1e-11, atol=1e-13, dense_output=True)
        s = np.linspace(s1, s0, 3601)
        q, qp, iq2, iq, ixq2 = sol.sol(s)
        cls._cache["raw"] = (s, q, iq2, iq, ixq2)
        return cls._cache["raw"]

    def _table(self):
        key = self.beta
        if key not in self._cache:
            s, q, iq2, iq, ixq2 = self._painleve_tables()
            F2 = np.exp(-(ixq2 - s * iq2))
            if self.beta == 2:
                x, F = s, F2
            elif self.beta == 1:
                x, F = s, np.sqrt(F2) * np.exp(-0.5 * iq)
            else:
                x, F = s / np.sqrt(2.0), np.sqrt(F2) * np.cosh(0.5 * iq)
            F = np.maximum.accumulate(np.clip(F, 0.0, 1.0))
            self._cache[key] = _GridInterp(x, F, total=1.0)
        return self._cache[key]

    def support(self):
        return (-np.inf, np.inf)

    def pdf(self, x):
        out = self._table().pdf(x)
        return out if out.ndim else float(out)

    def cdf(self, x):
        out = self._table().cdf(x)
        return out if out.ndim else float(out)

    def ppf(self, q):
        out = self._table().ppf(q)
        return out if out.ndim else float(out)

    def rvs(self, size=1, random_state=None):
        u = _rng(random_state).uniform(0.0, 1.0, size=size)
        return self.ppf(u)

    def _moment(self, k, about=0.0):
        t = self._table()
        return float(np.trapezoid((t.x - about) ** k * t.pdf(t.x), t.x))

    def mean(self):
        return self._moment(1)

    def var(self):
        return self._moment(2, self.mean())

    def _moment_about_mean(self, k):
        return self._moment(k, self.mean())

    def entropy(self):
        t = self._table()
        f = np.clip(t.pdf(t.x), 1e-300, None)
        return float(-np.trapezoid(f * np.log(f), t.x))

    def mgf(self, t):
        tb = self._table()
        return float(np.trapezoid(np.exp(float(t) * tb.x) * tb.pdf(tb.x), tb.x))

    def cf(self, t):
        tb = self._table()
        return complex(np.trapezoid(np.exp(1j * float(t) * tb.x) * tb.pdf(tb.x), tb.x))

    @classmethod
    def fit(cls, data, beta=2):
        """No free parameters: returns the law for ``beta`` (data is validated only)."""
        np.asarray(data, dtype=float)
        return cls(beta=beta)


class BetaEnsemble:
    """Dumitriu-Edelman tridiagonal beta-Hermite / beta-Laguerre ensembles, any ``beta > 0``.

    ``kind="hermite"``: symmetric tridiagonal with diagonal ``N(0, 1)`` and off-diagonal
    ``chi_{beta (n-k)} / sqrt(2)``; ``normalized_eigenvalues()`` divides by
    ``sqrt(beta n / 2)`` and converges to ``WignerSemicircle(2)``. ``beta`` = 1, 2, 4
    reproduce the GOE/GUE/GSE eigenvalue laws exactly. ``kind="laguerre"``: ``B B^T``
    with bidiagonal chi entries (``a`` = number of columns, default ``n``);
    ``normalized_eigenvalues()`` divides by ``beta a`` and converges to
    ``MarchenkoPastur(n / a)``.
    """

    def __init__(self, n, beta=2.0, kind="hermite", a=None):
        if n < 2 or beta <= 0:
            raise ValueError("n >= 2 and beta > 0 required")
        if kind not in ("hermite", "laguerre"):
            raise ValueError("kind must be 'hermite' or 'laguerre'")
        self.n = int(n)
        self.beta = float(beta)
        self.kind = kind
        self.a = int(a) if a is not None else self.n
        if kind == "laguerre" and self.a < self.n:
            raise ValueError("laguerre: a (columns) must be >= n")

    def sample(self, random_state=None):
        """One tridiagonal matrix draw (dense ``n x n`` array)."""
        rng = _rng(random_state)
        n, b = self.n, self.beta
        if self.kind == "hermite":
            diag = rng.normal(0.0, np.sqrt(2.0), n)
            off = np.sqrt(rng.chisquare(b * np.arange(n - 1, 0, -1)))
            H = np.diag(diag) + np.diag(off, 1) + np.diag(off, -1)
            return H / np.sqrt(2.0)
        d = np.sqrt(rng.chisquare(b * (self.a - np.arange(n))))
        sub = np.sqrt(rng.chisquare(b * np.arange(n - 1, 0, -1)))
        B = np.diag(d) + np.diag(sub, -1)
        return B @ B.T

    def eigenvalues(self, random_state=None):
        return _hermitian_eigvals(self.sample(random_state))

    def _scale(self):
        if self.kind == "hermite":
            return float(np.sqrt(self.beta * self.n / 2.0))
        return float(self.beta * self.a)

    def normalize(self, eigenvalues):
        """Apply the bulk scaling of ``normalized_eigenvalues`` to given eigenvalues."""
        return np.asarray(eigenvalues) / self._scale()

    def normalized_eigenvalues(self, random_state=None):
        return self.normalize(self.eigenvalues(random_state))

    def limit_law(self):
        if self.kind == "hermite":
            return WignerSemicircle(2.0)
        return MarchenkoPastur(self.n / self.a)

    def mean_ratio_reference(self):
        """Large-N ``<r>`` reference for beta in {1, 2, 4} (None otherwise)."""
        return _MEAN_RATIO_REFERENCE.get(int(self.beta) if self.beta in (1, 2, 4) else None)


class JacobiEnsemble:
    """Jacobi (MANOVA) ensemble: eigenvalues of ``(W1 + W2)^{-1/2} W1 (W1 + W2)^{-1/2}``.

    ``W1 ~ Wishart(m1, I_n)``, ``W2 ~ Wishart(m2, I_n)`` (real for ``beta=1``, complex
    for ``beta=2``); eigenvalues lie in ``[0, 1]`` with mean ``m1 / (m1 + m2)`` and, as
    ``n, m1, m2 -> inf`` with fixed ratios, follow the Wachter law (``limit_density``).
    """

    def __init__(self, n, m1, m2, beta=1):
        if beta not in (1, 2):
            raise ValueError("beta must be 1 or 2")
        if m1 < n or m2 < n:
            raise ValueError("m1 and m2 must be >= n for full-rank Wisharts")
        self.n, self.m1, self.m2, self.beta = int(n), int(m1), int(m2), int(beta)

    def _gaussian(self, rng, rows):
        if self.beta == 1:
            return rng.standard_normal((rows, self.n))
        z = rng.standard_normal((rows, self.n)) + 1j * rng.standard_normal((rows, self.n))
        return z / np.sqrt(2.0)

    def sample(self, random_state=None):
        rng = _rng(random_state)
        X1, X2 = self._gaussian(rng, self.m1), self._gaussian(rng, self.m2)
        W1 = X1.conj().T @ X1
        W2 = X2.conj().T @ X2
        w, V = np.linalg.eigh(W1 + W2)
        inv_sqrt = (V * (1.0 / np.sqrt(w))) @ V.conj().T
        J = inv_sqrt @ W1 @ inv_sqrt
        return 0.5 * (J + J.conj().T)

    def eigenvalues(self, random_state=None):
        return np.clip(_hermitian_eigvals(self.sample(random_state)), 0.0, 1.0)

    def _f_density(self, f):
        """Limiting density of the F-matrix eigenvalues (Wachter 1980 / Bai-Silverstein)."""
        y1, y2 = self.n / self.m1, self.n / self.m2
        h = np.sqrt(1.0 - (1.0 - y1) * (1.0 - y2))
        a = ((1.0 - h) / (1.0 - y2)) ** 2
        b = ((1.0 + h) / (1.0 - y2)) ** 2
        f = np.asarray(f, dtype=float)
        out = np.zeros_like(f, dtype=float)
        inside = (f > a) & (f < b)
        fi = f[inside]
        out[inside] = (1.0 - y2) * np.sqrt((b - fi) * (fi - a)) / (2.0 * np.pi * fi * (y1 + y2 * fi))
        return out

    def limit_density(self, x):
        """Wachter law: bulk density of the eigenvalues on (0, 1) as ``n, m1, m2 -> inf``.

        Obtained from the multivariate-F limit through ``f = (m2/m1) x / (1 - x)``.
        """
        x = np.asarray(x, dtype=float)
        out = np.zeros_like(x, dtype=float)
        inside = (x > 0.0) & (x < 1.0)
        xi = x[inside]
        k = self.m2 / self.m1
        f = k * xi / (1.0 - xi)
        out[inside] = self._f_density(f) * k / (1.0 - xi) ** 2
        return out if out.ndim else float(out)
