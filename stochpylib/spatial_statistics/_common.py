"""Shared numerics for :mod:`stochpylib.spatial_statistics`.

Pairwise/nearest-neighbour distances reuse ``experimental_design._common`` rather than
duplicating them. Everything else here (box windows, ball volumes, lag binning) is new to
this module. Native numpy/scipy.special/optimize/integrate only -- scipy.stats is the test
suite's oracle only.
"""

import numpy as np
from scipy import special

from stochpylib.experimental_design._common import _min_distance, _nearest_distance, _pairwise

__all__ = []  # private module


def _rng(random_state):
    return np.random.default_rng(random_state)


def _as_coords(X, name="coords"):
    arr = np.asarray(X, dtype=float)
    if arr.ndim == 1:
        arr = arr[:, None]
    if arr.ndim != 2 or arr.shape[0] == 0:
        raise ValueError(f"{name} must be a non-empty (n, d) array")
    if not np.all(np.isfinite(arr)):
        raise ValueError(f"{name} contains non-finite values")
    return arr


def _as_values(y, n, name="values"):
    arr = np.asarray(y, dtype=float).ravel()
    if arr.size != n:
        raise ValueError(f"{name} must have {n} entries, one per coordinate")
    if not np.all(np.isfinite(arr)):
        raise ValueError(f"{name} contains non-finite values")
    return arr


def _box_window(window):
    """Normalize a window to a ``(d, 2)`` array of (lo, hi) bounds, plus its volume."""
    W = np.atleast_2d(np.asarray(window, dtype=float))
    if W.shape[1] != 2:
        raise ValueError("window must be a sequence of (lo, hi) pairs, one per dimension")
    if np.any(W[:, 1] <= W[:, 0]):
        raise ValueError("every window bound needs hi > lo")
    volume = float(np.prod(W[:, 1] - W[:, 0]))
    return W, volume


def _ball_volume(d, r):
    """Volume of a d-dimensional ball of radius r: omega_d * r^d."""
    r = np.asarray(r, dtype=float)
    omega_d = np.pi ** (d / 2.0) / special.gamma(d / 2.0 + 1.0)
    return omega_d * r ** d


def _in_window(points, W):
    inside = np.ones(len(points), dtype=bool)
    for j in range(W.shape[0]):
        inside &= (points[:, j] >= W[j, 0]) & (points[:, j] <= W[j, 1])
    return inside


def _lag_bins(h, bins, max_dist=None):
    """Bin edges for a set of pairwise lag distances h."""
    h = np.asarray(h, dtype=float)
    if max_dist is None:
        max_dist = 0.5 * float(h.max()) if h.size else 1.0
    return np.linspace(0.0, max_dist, int(bins) + 1)


# ``_pairwise``/``_min_distance``/``_nearest_distance`` are imported above from
# experimental_design._common and re-exported here so every file in this package imports
# distance helpers from one place (`from stochpylib.spatial_statistics._common import ...`).
