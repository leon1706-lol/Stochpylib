"""Mutual information and related measures."""

import numpy as np

from stochpylib.information_theory._base import (
    _validate_probs, _normalise, _joint_table, _safe_log2,
)

__all__ = [
    "MutualInformation", "NormalizedMutualInformation",
    "VariationOfInformation", "ConditionalMutualInfo",
    "InteractionInformation", "MultiInformation",
]


def _entropy_bits(p):
    p = np.asarray(p, float)
    p = p[p > 0]
    return float(-np.sum(p * _safe_log2(p)))


class MutualInformation:
    """I(X;Y) = H(X) + H(Y) - H(X,Y) from paired observations.

    For discrete data uses contingency table; set ``bins`` > 0 for
    continuous data (equal-width discretisation).
    """

    def __init__(self, bins=0):
        self.bins = int(bins)

    def fit(self, x, y):
        joint_flat, table = _joint_table(x, y, bins=self.bins)
        pxy = table / max(table.sum(), 1e-300)
        px = pxy.sum(axis=1)
        py = pxy.sum(axis=0)
        hx = _entropy_bits(px)
        hy = _entropy_bits(py)
        hxy = _entropy_bits(pxy.ravel())
        self.result_ = max(hx + hy - hxy, 0.0)
        self.h_x_, self.h_y_, self.h_xy_ = hx, hy, hxy
        self.joint_table_ = pxy
        return self

    @classmethod
    def compute(cls, x, y, **kw):
        return cls(**kw).fit(x, y).result_


class NormalizedMutualInformation:
    """NMI = I(X;Y) / sqrt(H(X)*H(Y)), in [0,1]."""

    def __init__(self, bins=0):
        self.bins = int(bins)

    def fit(self, x, y):
        mi = MutualInformation(bins=self.bins).fit(x, y)
        denom = np.sqrt(mi.h_x_ * mi.h_y_)
        self.result_ = mi.result_ / max(denom, 1e-12)
        self.mi_ = mi.result_
        return self

    @classmethod
    def compute(cls, x, y, **kw):
        return cls(**kw).fit(x, y).result_


class VariationOfInformation:
    """VI(X;Y) = H(X|Y) + H(Y|X) = H(X,Y) - I(X;Y); in [0, log2(n)]."""

    def __init__(self, bins=0):
        self.bins = int(bins)

    def fit(self, x, y):
        mi = MutualInformation(bins=self.bins).fit(x, y)
        joint_flat, table = _joint_table(x, y, bins=self.bins)
        pxy = table / max(table.sum(), 1e-300)
        hxy = _entropy_bits(_normalise(pxy.ravel()))
        self.result_ = max(hxy - mi.result_, 0.0)
        self.mi_ = mi.result_
        return self

    @classmethod
    def compute(cls, x, y, **kw):
        return cls(**kw).fit(x, y).result_


class ConditionalMutualInfo:
    """I(X;Y|Z) = sum_z p(z) I(X;Y | Z=z).

    Accepts three paired observation arrays (discrete/categorical).
    """

    def __init__(self):
        pass

    def fit(self, x, y, z):
        x = np.asarray(x).ravel()
        y = np.asarray(y).ravel()
        z = np.asarray(z).ravel()
        if not (len(x) == len(y) == len(z)):
            raise ValueError("x/y/z must have equal length")
        uz = np.unique(z)
        total_weight = len(z)
        cmi = 0.0
        for zv in uz:
            mask = z == zv
            frac = mask.sum() / total_weight
            if frac < 1e-12:
                continue
            # I(X;Y | Z=zv): MI on the subset
            sub_x = x[mask]
            sub_y = y[mask]
            ux = np.unique(sub_x)
            uy = np.unique(sub_y)
            xi = np.searchsorted(ux, sub_x)
            yi = np.searchsorted(uy, sub_y)
            table = np.zeros((len(ux), len(uy)))
            np.add.at(table, (xi, yi), 1.0)
            pxy = table / table.sum()
            px = pxy.sum(axis=1)
            py = pxy.sum(axis=0)
            hx = _entropy_bits(px)
            hy = _entropy_bits(py)
            hxy = _entropy_bits(pxy.ravel())
            cmi += frac * max(hx + hy - hxy, 0.0)
        self.result_ = cmi
        return self

    @classmethod
    def compute(cls, x, y, z):
        return cls().fit(x, y, z).result_


class InteractionInformation:
    """Interaction information I(X;Y;Z) = I(X;Y) - I(X;Y|Z).

    Can be negative (synergy) or positive (redundancy).
    """

    def __init__(self):
        pass

    def fit(self, x, y, z):
        mi_xy = MutualInformation().fit(x, y).result_
        cmi = ConditionalMutualInfo.compute(x, y, z)
        self.result_ = mi_xy - cmi
        return self

    @classmethod
    def compute(cls, x, y, z):
        return cls().fit(x, y, z).result_


class MultiInformation:
    """Multi-information (total correlation): sum(H(X_i)) - H(X_1,...,X_n).

    Generalisation of mutual information to more than two variables.
    """

    def __init__(self):
        pass

    def fit(self, *columns):
        cols = [np.asarray(c).ravel() for c in columns]
        n_vars = len(cols)
        if n_vars < 2:
            raise ValueError("need at least two variables")
        individual_h = sum(EntropyProxy(c).h for c in cols)
        # joint entropy via discretised multi-way contingency table
        encoded = np.zeros(len(cols[0]))
        for j, col in enumerate(cols):
            _, codes = np.unique(col, return_inverse=True)
            if j == 0:
                encoded = codes.astype(float)
            else:
                prev_cardinality = int(encoded.max()) + 1
                encoded = encoded * prev_cardinality + codes
        h_joint = _entropy_bits(_normalise(
            np.bincount(encoded.astype(int))))
        self.result_ = individual_h - h_joint
        self.individual_entropies_ = [EntropyProxy(c).h for c in cols]
        self.joint_entropy_ = h_joint
        return self

    @classmethod
    def compute(cls, *columns):
        return cls().fit(*columns).result_


def EntropyProxy(arr):
    class _P:
        def __init__(self, a):
            _, counts = np.unique(a, return_counts=True)
            self.h = _entropy_bits(_normalise(counts))
    return _P(np.asarray(arr).ravel())