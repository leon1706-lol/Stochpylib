"""Private helpers shared across the random_matrix submodules.

Not part of the public API (no ``__all__``). Everything here is numpy/scipy.special/
scipy.integrate only -- library code never wraps ``scipy.stats`` (test oracle only).
"""

import numpy as np
from scipy import interpolate


def _rng(random_state):
    return np.random.default_rng(random_state)


def _as_1d(x, name="eigenvalues"):
    arr = np.asarray(x).ravel()
    if arr.size == 0:
        raise ValueError(f"{name} must be non-empty")
    return arr


def _hermitian_eigvals(H):
    """Sorted real eigenvalues of a (numerically) Hermitian matrix."""
    return np.sort(np.linalg.eigvalsh(H))


def _grid_law(pdf, lo, hi, n=4001):
    """Tabulate a continuous density on [lo, hi] and return (x, cdf) with cdf(lo)=0.

    Cumulative trapezoid on a fine grid; callers wrap the result in ``_GridInterp``.
    """
    x = np.linspace(lo, hi, n)
    f = np.asarray(pdf(x), dtype=float)
    F = np.concatenate([[0.0], np.cumsum(0.5 * (f[1:] + f[:-1]) * np.diff(x))])
    return x, F


class _GridInterp:
    """Monotone (PCHIP) cdf interpolant with a derived pdf and inverse."""

    def __init__(self, x, F, total=None):
        self.x = np.asarray(x, dtype=float)
        F = np.asarray(F, dtype=float)
        self.total = float(F[-1]) if total is None else float(total)
        self._cdf = interpolate.PchipInterpolator(self.x, F, extrapolate=False)
        self._pdf = self._cdf.derivative()
        # strictly increasing subset for the inverse
        keep = np.concatenate([[True], np.diff(F) > 1e-15])
        self._ppf = interpolate.PchipInterpolator(F[keep], self.x[keep], extrapolate=False)

    def cdf(self, s):
        s = np.asarray(s, dtype=float)
        out = self._cdf(np.clip(s, self.x[0], self.x[-1]))
        out = np.where(s < self.x[0], 0.0, out)
        out = np.where(s > self.x[-1], self.total, out)
        return out

    def pdf(self, s):
        s = np.asarray(s, dtype=float)
        out = self._pdf(np.clip(s, self.x[0], self.x[-1]))
        out = np.where((s < self.x[0]) | (s > self.x[-1]), 0.0, out)
        return np.clip(out, 0.0, None)

    def ppf(self, q):
        q = np.asarray(q, dtype=float)
        qq = np.clip(q, 0.0, self.total)
        out = self._ppf(qq)
        out = np.where(q <= 0.0, self.x[0], out)
        out = np.where(q >= self.total, self.x[-1], out)
        return out


# ------------------------------------------------------------- spacing references

def _wigner_surmise(s, beta):
    """Wigner surmise density of the unit-mean nearest-neighbour spacing.

    ``beta`` in {1, 2, 4} (Dyson index) or 0 / "poisson" for uncorrelated levels.
    """
    s = np.asarray(s, dtype=float)
    if beta in (0, "poisson"):
        return np.exp(-s)
    if beta == 1:
        return (np.pi / 2.0) * s * np.exp(-np.pi * s * s / 4.0)
    if beta == 2:
        return (32.0 / np.pi ** 2) * s ** 2 * np.exp(-4.0 * s * s / np.pi)
    if beta == 4:
        return (2 ** 18 / (3 ** 6 * np.pi ** 3)) * s ** 4 * np.exp(-64.0 * s * s / (9.0 * np.pi))
    raise ValueError("beta must be 1, 2, 4 or 0/'poisson'")


def _wigner_surmise_cdf(s, beta):
    s = np.asarray(s, dtype=float)
    if beta in (0, "poisson"):
        return 1.0 - np.exp(-s)
    if beta == 1:
        return 1.0 - np.exp(-np.pi * s * s / 4.0)
    x, F = _grid_law(lambda t: _wigner_surmise(t, beta), 0.0, 6.0, 3001)
    return np.interp(s, x, F / F[-1])


# Mean adjacent-gap ratio <r> = E[min(s_i, s_{i+1}) / max(s_i, s_{i+1})]: large-N values from
# Atas et al. (2013), Poisson exactly 2 ln 2 - 1. Needs no unfolding, so it is the primary
# spacing oracle throughout the module.
_MEAN_RATIO_REFERENCE = {"poisson": 2.0 * np.log(2.0) - 1.0, 1: 0.5307, 2: 0.5996, 4: 0.6744}
