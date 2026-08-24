"""Base class for all copulas.

Conventions (mirroring the ARCHITECTURE.md cross-cutting rules):

- ``.fit(data)`` is fluent: it accepts *raw* observations, converts them to rank-based
  pseudo-observations internally and returns ``self``; fitted parameters live on the
  instance as attributes ending in ``_``.
- ``.sample(n, random_state=None)`` takes anything :func:`numpy.random.default_rng`
  accepts — never a bare global seed.
- ``.cdf(u)`` evaluates on points of the unit cube (rows = d-dimensional coordinates).
- Every family exposes analytic ``kendall_tau()`` / ``tail_dependence()`` where they
  exist (bivariate), plus ``loglik``/``aic`` for model selection.
"""

import numpy as np

from stochpylib.copulas._utils import as_u_matrix, pseudo_obs

__all__ = ["BaseCopula"]


class BaseCopula:
    """Abstract copula with the shared fit/sample/cdf surface."""

    #: number of dimensions; subclasses fix it or set it in fit() for d-dim families
    dimension = None

    #: parameter name(s) estimated from data; used by AIC bookkeeping
    _n_params = 1

    def __init__(self):
        self.n_obs_ = None

    # ------------------------------------------------------------------ fitting
    def _check_dimension(self, data):
        data = np.asarray(data, dtype=float)
        if self.dimension is None:
            raise NotImplementedError(
                f"{type(self).__name__} does not define its dimension")
        if self.dimension != "d" and data.shape[1] != self.dimension:
            raise ValueError(
                f"{type(self).__name__} is {self.dimension}-dimensional, "
                f"got data with {data.shape[1]} columns")
        return data

    def fit(self, data):
        """Fit to raw observations via rank pseudo-observations (fluent)."""
        data = self._check_dimension(np.asarray(data, dtype=float))
        if data.ndim != 2 or data.shape[0] < 2:
            raise ValueError("data must be a (n, d) array with n >= 2")
        if self.dimension == "d":
            self.dimension = int(data.shape[1])
        u = pseudo_obs(data)
        self.n_obs_ = len(u)
        self._estimate(u)
        return self

    def _estimate(self, u):
        raise NotImplementedError

    # ------------------------------------------------------------------ sampling
    def sample(self, n, random_state=None):
        raise NotImplementedError

    # ------------------------------------------------------------------ evaluation
    def cdf(self, u):
        raise NotImplementedError

    def density(self, u):
        raise NotImplementedError(
            f"{type(self).__name__} does not expose a closed-form copula density")

    # ------------------------------------------------------------------ dependence
    def kendall_tau(self):
        """Analytic bivariate Kendall's tau, or None when no closed form exists."""
        return None

    def tail_dependence(self):
        """Dict with ``upper`` / ``lower`` tail-dependence coefficients."""
        return {"upper": 0.0, "lower": 0.0}

    # ------------------------------------------------------------------ selection
    def loglik(self, u):
        """Log-likelihood at pseudo-observations ``u`` (rows inside (0,1)^d)."""
        u = as_u_matrix(u)
        dens = np.asarray(self.density(u), dtype=float)
        if np.any(dens <= 0) or not np.all(np.isfinite(dens)):
            return -np.inf
        return float(np.sum(np.log(dens)))

    def aic(self, u):
        ll = self.loglik(u)
        return np.inf if not np.isfinite(ll) \
            else -2.0 * ll + 2.0 * self._n_params

    # ------------------------------------------------------------------ helpers
    def _validate_sample_n(self, n):
        n = int(n)
        if n < 1:
            raise ValueError("n must be >= 1")
        return n
