"""Empirical copulas: order-statistics, checkerboard binning and Bernstein
(beta-kernel) smoothing.

All three are built from rank pseudo-observations of the training data.

- ``EmpiricalCopula``     : multivariate empirical distribution function of the
  pseudo-observations (Glivenko-Cantelli consistent); sampling is a row bootstrap.
- ``CheckerboardCopula``  : empirical masses on an ``m^d`` grid; the CDF is the
  multilinear interpolation of the cumulative cell grid; sampling picks a cell
  by mass and fills it uniformly.
- ``BetaCopula``          : Bernstein-polynomial smoothing of the checkerboard
  grid — ``C_M(u) = sum_k prod_j b_{k_j,M}(u_j) * F_hat(k/m)`` over the node
  multi-index ``k`` with the Bernstein basis ``b_{k,M}(x) = C(M,k)x^k(1-x)^{M-k}``;
  converges to the checkerboard/empirical copula as ``m -> inf``. Sampling draws
  a cell by mass and then i.i.d. ``Beta(k_j+1, M+1-k_j)`` coordinates (documented
  approximation of the full conditional).
"""

import numpy as np
from scipy import interpolate, special

from stochpylib.copulas._base import BaseCopula
from stochpylib.copulas._utils import as_u_matrix

__all__ = ["EmpiricalCopula", "CheckerboardCopula", "BetaCopula"]


class EmpiricalCopula(BaseCopula):
    """Multivariate empirical copula::

        ec = EmpiricalCopula().fit(data)
        val = ec.cdf([[0.7, 0.8]])
        sims = ec.sample(1000, random_state=0)
    """

    _n_params = 0
    dimension = "d"

    def __init__(self):
        super().__init__()
        self.u_obs_ = None

    def _require_fit(self):
        if self.u_obs_ is None:
            raise RuntimeError("fit() must be called first")

    def _estimate(self, u):
        self.u_obs_ = u

    def sample(self, n, random_state=None):
        """Row bootstrap of the pseudo-observations."""
        self._require_fit()
        n = self._validate_sample_n(n)
        rng = np.random.default_rng(random_state)
        idx = rng.integers(0, len(self.u_obs_), size=n)
        return self.u_obs_[idx]

    def cdf(self, u):
        self._require_fit()
        u = as_u_matrix(u)
        obs = self.u_obs_
        n = len(obs)
        out = np.empty(len(u))
        chunk = max(1, 2_000_000 // max(n, 1))
        for start in range(0, len(u), chunk):
            stop = min(start + chunk, len(u))
            # count observations dominated by the query point
            le = np.all(obs[None, :, :] <= u[start:stop, None, :], axis=2)
            out[start:stop] = le.sum(axis=1) / n
        return out


class _GridCopulaBase(BaseCopula):
    """Shared binning machinery for the smoothed empirical copulas."""

    _n_params = 0
    dimension = "d"

    def __init__(self, n_bins=10):
        super().__init__()
        self.n_bins = int(n_bins)
        if self.n_bins < 1:
            raise ValueError("n_bins must be >= 1")
        self.cell_mass_ = None            # (m, ..., m)
        self.node_cdf_ = None             # (m+1, ..., m+1)

    def _require_fit(self):
        if self.cell_mass_ is None:
            raise RuntimeError("fit() must be called first")

    def _bin(self, u):
        m = self.n_bins
        idx = np.minimum((u * m).astype(int), m - 1)
        flat = np.ravel_multi_index(idx.T, (m,) * u.shape[1])
        counts = np.bincount(flat, minlength=m ** u.shape[1])
        self.cell_mass_ = counts.reshape((m,) * u.shape[1]) / float(len(u))

    def _build_node_cdf(self):
        """F_hat on the node grid k/m, k = 0..m per dimension."""
        m = self.n_bins
        padded = np.pad(self.cell_mass_,
                        [(1, 0)] * self.cell_mass_.ndim,
                        mode="constant", constant_values=0.0)
        cum = padded.copy()
        for axis in range(cum.ndim):
            cum = np.cumsum(cum, axis=axis)
        return cum


class CheckerboardCopula(_GridCopulaBase):
    """Checkerboard (empirical grid) copula with multilinear-interpolated CDF."""

    def _estimate(self, u):
        self._bin(u)
        self.node_cdf_ = self._build_node_cdf()
        return u

    def cdf(self, u):
        self._require_fit()
        u = as_u_matrix(u)
        m = self.n_bins
        interp = interpolate.RegularGridInterpolator(
            [np.linspace(0.0, 1.0, m + 1)] * self.cell_mass_.ndim,
            self.node_cdf_, method="linear", bounds_error=True)
        return interp(np.clip(u, 0.0, 1.0))

    def density(self, u):
        """Piecewise-constant cell density (cell mass / cell volume)."""
        self._require_fit()
        u = as_u_matrix(u)
        m = self.n_bins
        idx = np.minimum((np.clip(u, 0.0, 1.0 - 1e-12) * m).astype(int),
                         m - 1)
        vol = (1.0 / m) ** u.shape[1]
        return self.cell_mass_[tuple(idx.T)] / vol

    def sample(self, n, random_state=None):
        """Draw a cell by empirical mass, then fill it uniformly."""
        self._require_fit()
        n = self._validate_sample_n(n)
        rng = np.random.default_rng(random_state)
        m = self.n_bins
        d = self.cell_mass_.ndim
        flat_p = self.cell_mass_.ravel()
        cell = rng.choice(flat_p.size, size=n, p=flat_p)
        idx = np.unravel_index(cell, (m,) * d)
        return np.column_stack([
            (idx[j] + rng.random(n)) / m for j in range(d)
        ])


class BetaCopula(_GridCopulaBase):
    """Bernstein (beta-kernel) smoothed checkerboard copula."""

    def __init__(self, n_bins=20):
        super().__init__(n_bins)

    def _estimate(self, u):
        self._bin(u)
        self.node_cdf_ = self._build_node_cdf()
        return u

    def _bernstein_factors(self, x):
        """(len(x), m+1) matrix of b_{k,M}(x)."""
        m = self.n_bins
        ks = np.arange(m + 1)
        log_binom = special.gammaln(m + 1) - special.gammaln(ks + 1) \
            - special.gammaln(m - ks + 1)
        # keep strictly inside the open interval so logs stay finite even at
        # the cube boundary (weights there are limiting values anyway)
        xc = np.clip(x, 1e-12, 1.0 - 1e-12)
        logw = log_binom[None, :] \
            + ks[None, :] * np.log(xc[:, None]) \
            + (m - ks)[None, :] * np.log1p(-xc[:, None])
        return np.exp(logw)

    def cdf(self, u):
        self._require_fit()
        u = as_u_matrix(u)
        m = self.n_bins
        d = self.cell_mass_.ndim
        if d > 8 or (m + 1) ** d > 4_000_000:
            raise ValueError("BetaCopula evaluation supports small "
                             "dimension/bin combinations only")
        letters = "abcdefgh"[:d]
        factors = [self._bernstein_factors(np.clip(u[:, j], 0.0, 1.0))
                   for j in range(d)]
        expr = ",".join(f"q{ch}" for ch in letters) + "," + letters + "->q"
        return np.einsum(expr, *(factors + [self.node_cdf_]))

    def sample(self, n, random_state=None):
        """Cell by mass, then i.i.d. Beta(k_j+1, M+1-k_j) coordinates
        (documented approximation of the Bernstein conditional)."""
        self._require_fit()
        n = self._validate_sample_n(n)
        rng = np.random.default_rng(random_state)
        m = self.n_bins
        d = self.cell_mass_.ndim
        flat_p = self.cell_mass_.ravel()
        cell = rng.choice(flat_p.size, size=n, p=flat_p)
        idx = np.unravel_index(cell, (m,) * d)
        return np.column_stack([
            rng.beta(idx[j] + 1.0, m - idx[j] + 1.0) for j in range(d)
        ])
