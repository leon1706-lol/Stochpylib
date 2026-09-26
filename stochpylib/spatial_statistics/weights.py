"""Spatial weights matrices for the ``spatial_statistics.tests`` autocorrelation family."""

import numpy as np

from stochpylib.spatial_statistics._common import _as_coords, _pairwise

__all__ = ["SpatialWeights"]


class SpatialWeights:
    """A spatial weights matrix ``W`` (n, n), zero diagonal enforced.

    ``S0 = sum(W)``, ``S1 = 0.5 * sum((W + W.T)**2)``, ``S2 = sum((row_sum + col_sum)**2)``
    are the Cliff & Ord moment sums used throughout ``spatial_statistics.tests``.
    """

    def __init__(self, W):
        W = np.asarray(W, dtype=float)
        if W.ndim != 2 or W.shape[0] != W.shape[1]:
            raise ValueError("W must be a square (n, n) matrix")
        np.fill_diagonal(W, 0.0)
        self.W = W

    @property
    def n(self):
        return self.W.shape[0]

    @property
    def S0(self):
        return float(self.W.sum())

    @property
    def S1(self):
        return float(0.5 * np.sum((self.W + self.W.T) ** 2))

    @property
    def S2(self):
        row = self.W.sum(axis=1)
        col = self.W.sum(axis=0)
        return float(np.sum((row + col) ** 2))

    def row_standardize(self):
        row = self.W.sum(axis=1, keepdims=True)
        row = np.where(row > 0, row, 1.0)
        return SpatialWeights(self.W / row)

    def symmetrize(self):
        return SpatialWeights(0.5 * (self.W + self.W.T))

    def __array__(self, dtype=None):
        return self.W.astype(dtype) if dtype is not None else self.W

    def __repr__(self):
        return f"SpatialWeights(n={self.n}, S0={self.S0:.4g})"

    # ------------------------------------------------------------------- constructors
    @classmethod
    def knn(cls, coords, k):
        X = _as_coords(coords)
        n = len(X)
        k = int(k)
        if not (0 < k < n):
            raise ValueError("k must satisfy 0 < k < n")
        D = _pairwise(X)
        np.fill_diagonal(D, np.inf)
        W = np.zeros((n, n))
        idx = np.argsort(D, axis=1)[:, :k]
        rows = np.repeat(np.arange(n), k)
        W[rows, idx.ravel()] = 1.0
        return cls(W)

    @classmethod
    def distance_band(cls, coords, threshold, binary=True):
        X = _as_coords(coords)
        D = _pairwise(X)
        within = (D > 0) & (D <= float(threshold))
        if binary:
            W = within.astype(float)
        else:
            W = np.where(within, D, 0.0)
        return cls(W)

    @classmethod
    def inverse_distance(cls, coords, power=1.0, threshold=None):
        X = _as_coords(coords)
        D = _pairwise(X)
        n = len(X)
        W = np.zeros((n, n))
        mask = D > 0
        if threshold is not None:
            mask &= D <= float(threshold)
        W[mask] = 1.0 / D[mask] ** float(power)
        return cls(W)

    @classmethod
    def kernel(cls, coords, bandwidth, kind="gaussian"):
        X = _as_coords(coords)
        D = _pairwise(X)
        b = float(bandwidth)
        u = D / b
        if kind == "gaussian":
            W = np.exp(-0.5 * u ** 2)
        elif kind == "epanechnikov":
            W = np.where(u < 1, 0.75 * (1 - u ** 2), 0.0)
        elif kind == "triangular":
            W = np.where(u < 1, 1 - u, 0.0)
        else:
            raise ValueError("kind must be 'gaussian', 'epanechnikov' or 'triangular'")
        np.fill_diagonal(W, 0.0)
        return cls(W)

    @classmethod
    def lattice(cls, shape, rule="rook"):
        rows, cols = int(shape[0]), int(shape[1])
        n = rows * cols

        def idx(i, j):
            return i * cols + j

        W = np.zeros((n, n))
        if rule not in ("rook", "queen"):
            raise ValueError("rule must be 'rook' or 'queen'")
        offsets = [(-1, 0), (1, 0), (0, -1), (0, 1)]
        if rule == "queen":
            offsets += [(-1, -1), (-1, 1), (1, -1), (1, 1)]
        for i in range(rows):
            for j in range(cols):
                for di, dj in offsets:
                    ii, jj = i + di, j + dj
                    if 0 <= ii < rows and 0 <= jj < cols:
                        W[idx(i, j), idx(ii, jj)] = 1.0
        return cls(W)
