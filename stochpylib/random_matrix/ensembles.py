"""Classical random-matrix ensembles.

Normalization convention (shared by the whole module): the Hermitian Gaussian ensembles
have off-diagonal entries with ``E|h_ij|^2 = 1``, so ``eigenvalues / sqrt(n)`` converge
to ``WignerSemicircle(radius=2)``; Wishart ``eigenvalues / n`` converge to
``MarchenkoPastur(p / n)``; Ginibre-type ``eigenvalues / sqrt(n)`` fill the unit disk.
``normalized_eigenvalues()`` applies exactly that scaling and ``limit_law()`` returns the
matching object from :mod:`stochpylib.random_matrix.empirical_spectra`.
"""

import numpy as np

from stochpylib.distributions import InverseWishart as _InverseWishartDist
from stochpylib.distributions import Wishart as _WishartDist
from stochpylib.distributions._base import Distribution
from stochpylib.random_matrix._common import _hermitian_eigvals, _rng
from stochpylib.random_matrix.empirical_spectra import MarchenkoPastur, WignerSemicircle
from stochpylib.random_matrix.random_rotations import HaarMeasure

__all__ = [
    "CUE",
    "CircularLaw",
    "GOE",
    "GSE",
    "GUE",
    "InverseWishart",
    "MatrixEnsemble",
    "MuresanMatrix",
    "WignerMatrix",
    "WishartMatrix",
]


class MatrixEnsemble:
    """Base class: ``sample`` one matrix, ``eigenvalues`` of one draw, ``normalized_eigenvalues``
    scaled to the bulk limit ``limit_law()``. Subclasses set ``beta`` (Dyson index) and
    ``_scale`` (the divisor applied by ``normalized_eigenvalues``)."""

    beta = None
    hermitian = True

    def __init__(self, n):
        if n < 1:
            raise ValueError("n must be >= 1")
        self.n = int(n)

    def sample(self, random_state=None):
        raise NotImplementedError

    def samples(self, n_samples, random_state=None):
        rng = _rng(random_state)
        return np.stack([self.sample(rng) for _ in range(int(n_samples))])

    def eigenvalues(self, random_state=None):
        M = self.sample(random_state)
        if self.hermitian:
            return _hermitian_eigvals(M)
        return np.linalg.eigvals(M)

    def _scale(self):
        return np.sqrt(self.n)

    def normalize(self, eigenvalues):
        """Apply the bulk scaling of ``normalized_eigenvalues`` to given eigenvalues."""
        return np.asarray(eigenvalues) / self._scale()

    def normalized_eigenvalues(self, random_state=None):
        return self.normalize(self.eigenvalues(random_state))

    def limit_law(self):
        return WignerSemicircle(2.0)

    def __repr__(self):
        return f"{type(self).__name__}(n={self.n})"


class GOE(MatrixEnsemble):
    """Gaussian Orthogonal Ensemble: ``H = (A + A.T) / sqrt(2)``, ``A`` i.i.d. ``N(0, 1)``.

    Real symmetric, off-diagonal variance 1, diagonal variance 2; ``beta = 1``.
    """

    beta = 1

    def sample(self, random_state=None):
        A = _rng(random_state).standard_normal((self.n, self.n))
        return (A + A.T) / np.sqrt(2.0)


class GUE(MatrixEnsemble):
    """Gaussian Unitary Ensemble: ``H = (A + A^H) / 2`` with complex Gaussian ``A``
    (``E|a_ij|^2 = 2``), so off-diagonal ``E|h_ij|^2 = 1``; ``beta = 2``."""

    beta = 2

    def sample(self, random_state=None):
        rng = _rng(random_state)
        A = rng.standard_normal((self.n, self.n)) + 1j * rng.standard_normal((self.n, self.n))
        return (A + A.conj().T) / 2.0


class GSE(MatrixEnsemble):
    """Gaussian Symplectic Ensemble: quaternion self-dual ``2n x 2n`` Hermitian matrix.

    Built as ``[[A, B], [-conj(B), conj(A)]]`` with ``A`` Hermitian, ``B`` antisymmetric,
    scaled so off-diagonal quaternion entries have ``E|h|^2 = 1``. Every eigenvalue is
    Kramers-doubled; ``eigenvalues()`` returns the ``n`` distinct values (``beta = 4``).
    """

    beta = 4

    def sample(self, random_state=None):
        rng = _rng(random_state)
        n = self.n
        X = rng.standard_normal((n, n)) + 1j * rng.standard_normal((n, n))
        Y = rng.standard_normal((n, n)) + 1j * rng.standard_normal((n, n))
        A = (X + X.conj().T) / 2.0        # Hermitian
        B = (Y - Y.T) / 2.0               # antisymmetric (complex)
        H = np.block([[A, B], [-B.conj(), A.conj()]])
        return H / np.sqrt(2.0)

    def eigenvalues(self, random_state=None):
        return _hermitian_eigvals(self.sample(random_state))[::2]


class WignerMatrix(MatrixEnsemble):
    """Real symmetric Wigner matrix with i.i.d. off-diagonal entries of any distribution.

    ``entries`` is a library ``Distribution`` (its ``rvs`` is used and the draws are
    standardized to mean 0 / variance 1), a callable ``f(size, rng) -> array`` (also
    standardized), or ``None`` for Gaussian entries. Semicircle universality:
    ``eigenvalues / sqrt(n)`` -> ``WignerSemicircle(2)`` regardless of ``entries``.
    """

    beta = 1

    def __init__(self, n, entries=None):
        super().__init__(n)
        self.entries = entries

    def _draw(self, size, rng):
        if self.entries is None:
            return rng.standard_normal(size)
        if isinstance(self.entries, Distribution):
            x = np.asarray(self.entries.rvs(size, random_state=rng), dtype=float)
            mu, sd = self.entries.mean(), self.entries.std()
        else:
            x = np.asarray(self.entries(size, rng), dtype=float)
            mu, sd = float(np.mean(x)), float(np.std(x))
        if not np.isfinite(sd) or sd <= 0:
            raise ValueError("entry distribution must have finite positive variance")
        return (x - mu) / sd

    def sample(self, random_state=None):
        rng = _rng(random_state)
        n = self.n
        iu = np.triu_indices(n, 1)
        H = np.zeros((n, n))
        H[iu] = self._draw(iu[0].size, rng)
        H = H + H.T
        H[np.diag_indices(n)] = self._draw(n, rng) * np.sqrt(2.0)
        return H


class WishartMatrix(MatrixEnsemble):
    """Real Wishart matrix ``W = X.T @ X`` with ``X`` an ``n x p`` matrix of ``N(0, Sigma)`` rows.

    Sampled through :class:`stochpylib.distributions.Wishart` (``df = n``, ``scale = Sigma``).
    ``normalized_eigenvalues()`` divides by ``n`` and, for ``Sigma = I``, converges to
    ``MarchenkoPastur(gamma = p / n)``.
    """

    beta = 1

    def __init__(self, p, n, sigma=None):
        if p < 1 or n < 1:
            raise ValueError("p and n must be >= 1")
        self.p, self.n = int(p), int(n)
        self.sigma = np.eye(self.p) if sigma is None else np.asarray(sigma, dtype=float)
        if self.sigma.shape != (self.p, self.p):
            raise ValueError("sigma must be p x p")
        self._dist = _WishartDist(df=self.n, scale=self.sigma)

    @property
    def gamma(self):
        return self.p / self.n

    def sample(self, random_state=None):
        return np.asarray(self._dist.rvs(1, random_state=random_state), dtype=float).reshape(
            self.p, self.p)

    def _scale(self):
        return float(self.n)

    def limit_law(self):
        return MarchenkoPastur(self.gamma, sigma=1.0)

    def __repr__(self):
        return f"WishartMatrix(p={self.p}, n={self.n})"


class InverseWishart(WishartMatrix):
    """Inverse-Wishart matrix ``W^{-1}`` for ``W ~ Wishart(n, Sigma)``, ``p x p``.

    Sampled through :class:`stochpylib.distributions.InverseWishart`; eigenvalues are the
    reciprocals of the Wishart eigenvalues, so ``normalized_eigenvalues()`` multiplies by
    ``n`` and ``1 / normalized_eigenvalues()`` follows ``MarchenkoPastur(p / n)`` for
    ``Sigma = I``. Note ``stochpylib.distributions.InverseWishart`` shares the name:
    ``spl show InverseWishart`` resolves to the distribution (first in module order).
    """

    def __init__(self, p, n, sigma=None):
        super().__init__(p, n, sigma)
        if self.n <= self.p + 1:
            raise ValueError("InverseWishart needs n > p + 1 for a finite mean")
        self._dist = _InverseWishartDist(df=self.n, scale=self.sigma)

    def _scale(self):
        return 1.0 / float(self.n)

    def __repr__(self):
        return f"InverseWishart(p={self.p}, n={self.n})"


class CUE(MatrixEnsemble):
    """Circular Unitary Ensemble: Haar-random ``n x n`` unitary matrices (``beta = 2``).

    ``eigenvalues()`` are complex with unit modulus; ``eigenangles()`` their arguments in
    ``(-pi, pi]``, whose marginal is exactly uniform.
    """

    beta = 2
    hermitian = False

    def __init__(self, n):
        super().__init__(n)
        self._haar = HaarMeasure("U", self.n)

    def sample(self, random_state=None):
        return self._haar.sample(random_state)

    def eigenangles(self, random_state=None):
        return np.sort(np.angle(self.eigenvalues(random_state)))

    def normalized_eigenvalues(self, random_state=None):
        """Eigenangles scaled to unit mean spacing (``angle * n / 2pi``)."""
        return self.eigenangles(random_state) * self.n / (2.0 * np.pi)

    def limit_law(self):
        return None


class CircularLaw:
    """Uniform law on the unit disk (limit of Ginibre-type ``eigenvalues / sqrt(n)``).

    ``pdf(z) = 1/pi`` inside the disk; ``radial_cdf(r) = r^2``; ``compare(eigenvalues)``
    KS-tests ``|z|^2`` against ``Uniform(0, 1)``.
    """

    def pdf(self, z):
        z = np.asarray(z)
        out = np.where(np.abs(z) <= 1.0, 1.0 / np.pi, 0.0)
        return out if out.ndim else float(out)

    def radial_cdf(self, r):
        r = np.asarray(r, dtype=float)
        out = np.clip(r, 0.0, 1.0) ** 2
        return out if out.ndim else float(out)

    def rvs(self, size=1, random_state=None):
        rng = _rng(random_state)
        r = np.sqrt(rng.uniform(0.0, 1.0, size))
        th = rng.uniform(-np.pi, np.pi, size)
        return r * np.exp(1j * th)

    def compare(self, eigenvalues):
        from stochpylib.distributions import Uniform
        from stochpylib.statistics import TestResult

        r2 = np.abs(np.asarray(eigenvalues).ravel()) ** 2
        stat, p = Uniform(0.0, 1.0).ks_test(r2)
        return TestResult(statistic=float(stat), pvalue=float(p),
                          null="spectrum follows the circular law",
                          method="Kolmogorov-Smirnov on |z|^2", extras={"n": int(r2.size)})


class MuresanMatrix(MatrixEnsemble):
    """Ginibre-type non-Hermitian ``n x n`` matrix with i.i.d. entries.

    The design spec names this ensemble without defining it; it is implemented here as the
    general i.i.d.-entry non-Hermitian matrix (real Gaussian = real Ginibre, ``"complex"`` =
    complex Ginibre, or any library ``Distribution`` / callable ``f(size, rng)``, standardized
    to mean 0 / variance 1). ``eigenvalues()`` are complex; ``eigenvalues / sqrt(n)`` obey the
    circular law (``limit_law()`` -> :class:`CircularLaw`).
    """

    hermitian = False
    beta = None

    def __init__(self, n, entries="gaussian"):
        super().__init__(n)
        self.entries = entries

    def _draw(self, size, rng):
        e = self.entries
        if e in (None, "gaussian", "real"):
            return rng.standard_normal(size)
        if e == "complex":
            return (rng.standard_normal(size) + 1j * rng.standard_normal(size)) / np.sqrt(2.0)
        if isinstance(e, Distribution):
            x = np.asarray(e.rvs(size, random_state=rng), dtype=float)
            return (x - e.mean()) / e.std()
        x = np.asarray(e(size, rng))
        return (x - np.mean(x)) / np.std(x)

    def sample(self, random_state=None):
        return self._draw((self.n, self.n), _rng(random_state))

    def limit_law(self):
        return CircularLaw()
