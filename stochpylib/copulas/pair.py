"""Pair-copula construction: rotations, h-functions and AIC-based selection."""

import numpy as np

from stochpylib.copulas._utils import kendall_tau_estimate

__all__ = ["PairCopulaConstruction", "PAIR_FAMILIES", "attach_h_inverse"]

_EPS = 1e-12

PAIR_FAMILIES = ("gaussian", "t", "clayton", "gumbel", "frank", "joe")


def _make_pair(family):
    """Fresh unfitted bivariate copula for a family name."""
    from stochpylib.copulas.archimedean import (
        ClaytonCopula, FrankCopula, GumbelCopula, JoeCopula,
    )
    from stochpylib.copulas.elliptical import GaussianCopula, StudentTCopula

    makers = {
        "gaussian": lambda: GaussianCopula(dimension=2),
        "t": lambda: StudentTCopula(dimension=2),
        "clayton": lambda: ClaytonCopula(),
        "gumbel": lambda: GumbelCopula(),
        "frank": lambda: FrankCopula(),
        "joe": lambda: JoeCopula(),
    }
    if family not in makers:
        raise ValueError(f"unknown pair family {family!r}")
    return makers[family]()


def attach_h_inverse(copula_cls):
    """Class decorator adding a generic vectorized ``_h_u_inverse`` bisection."""

    def _h_u_inverse(self, p, v):
        p = np.clip(np.asarray(p, dtype=float), _EPS, 1 - _EPS)
        v = np.broadcast_to(np.asarray(v, dtype=float), p.shape)
        lo = np.full(p.shape, _EPS)
        hi = np.full(p.shape, 1.0 - _EPS)
        for _ in range(60):
            mid = 0.5 * (lo + hi)
            vals = self._h_u(mid, v)
            too_low = vals < p
            lo = np.where(too_low, mid, lo)
            hi = np.where(too_low, hi, mid)
        return 0.5 * (lo + hi)

    copula_cls._h_u_inverse = _h_u_inverse
    return copula_cls


def _pair_loglik(copula, x, y):
    pts = np.column_stack([np.clip(x, _EPS, 1 - _EPS),
                           np.clip(y, _EPS, 1 - _EPS)])
    dens = np.asarray(copula.density(pts), dtype=float)
    good = np.isfinite(dens) & (dens > 0)
    if not np.any(good):
        return -np.inf
    return float(np.sum(np.log(dens[good])) + int(np.sum(~good)) * (-700.0))


class PairCopulaConstruction:
    """One vine edge: a fitted bivariate copula plus a rotation (0/90/180/270).

    Rotation conventions (Brechmann & Schepsmeier):

        rot 0   : C(u,v)       = Q(u, v)
        rot 90  : C(u,v)       = v - Q(1-u, v)
        rot 180 : C(u,v)       = u + v - 1 + Q(1-u, 1-v)
        rot 270 : C(u,v)       = u - Q(u, 1-v)

    ``fit`` evaluates every family x rotation on the appropriately transformed
    columns and keeps the best AIC — this is what lets e.g. a rotated Clayton
    capture negative lower-tail dependence.
    """

    def __init__(self, copula, rotation=0, family=None):
        self.copula = copula
        self.rotation = int(rotation) % 360
        self.family = family

    @staticmethod
    def _rotate_xy(x, y, rotation):
        """Columns handed to the BASE copula so that ``c_base(rx, ry)``
        equals the rotated density at ``(x, y)``:

        rot 0   : c_Q(x, y)          -> (x, y)
        rot 90  : c_Q(1-x, y)        -> (1-x, y)
        rot 180 : c_Q(1-x, 1-y)      -> (1-x, 1-y)
        rot 270 : c_Q(x, 1-y)        -> (x, 1-y)
        """
        if rotation == 0:
            return x, y
        if rotation == 90:
            return 1.0 - x, y
        if rotation == 180:
            return 1.0 - x, 1.0 - y
        if rotation == 270:
            return x, 1.0 - y
        raise ValueError("rotation must be one of 0, 90, 180, 270")

    @classmethod
    def fit(cls, x, y, families=PAIR_FAMILIES, allow_rotations=True):
        x = np.clip(np.asarray(x, dtype=float), _EPS, 1 - _EPS)
        y = np.clip(np.asarray(y, dtype=float), _EPS, 1 - _EPS)
        best = None
        rotations = (0, 90, 180, 270) if allow_rotations else (0,)
        for family in families:
            for rotation in rotations:
                rx, ry = cls._rotate_xy(x, y, rotation)
                cand = _make_pair(family)
                try:
                    cand.fit(np.column_stack([rx, ry]))
                except (ValueError, RuntimeError):
                    continue
                ll = _pair_loglik(cand, rx, ry)
                score = -2.0 * ll + 2.0 * getattr(cand, "_n_params", 1)
                if np.isfinite(score) and (best is None or score < best[0]):
                    best = (score, family, rotation, cand)
        if best is None:
            raise RuntimeError("no pair-copula family could be fitted")
        _, family, rotation, copula = best
        return cls(copula, rotation, family=family)

    # -- rotated-scale conditionals ---------------------------------------------
    def h(self, u, given_v):
        """``P(U <= u | V = v)`` on the rotated scale.

        Each is ``d C_rot(u,v)/dv`` with the conventions above:
        rot0   : Q2(u, v)
        rot180 : 1 - Q2(1-u, 1-v)
        rot90  : 1 - Q2(1-u, v)
        rot270 : Q2(u, 1-v)
        where ``Q2(w, v) = d Q(w,v)/dv`` is the base conditional.
        """
        u = np.asarray(u, dtype=float)
        v = np.asarray(given_v, dtype=float)
        q = self.copula
        r = self.rotation
        if r == 0:
            return q._h_u(u, v)
        if r == 180:
            return 1.0 - q._h_u(1.0 - u, 1.0 - v)
        if r == 90:
            return 1.0 - q._h_u(1.0 - u, v)
        # r == 270
        return q._h_u(u, 1.0 - v)

    def h_inv(self, p, given_v):
        """Vectorized inverse of :meth:`h` via monotone bisection."""
        p = np.clip(np.asarray(p, dtype=float), _EPS, 1 - _EPS)
        v = np.broadcast_to(np.asarray(given_v, dtype=float), p.shape)
        lo = np.full(p.shape, _EPS)
        hi = np.full(p.shape, 1.0 - _EPS)
        for _ in range(60):
            mid = 0.5 * (lo + hi)
            too_low = self.h(mid, v) < p
            lo = np.where(too_low, mid, lo)
            hi = np.where(too_low, hi, mid)
        return 0.5 * (lo + hi)

    def loglik(self, x, y):
        rx, ry = self._rotate_xy(x, y, self.rotation)
        return _pair_loglik(self.copula, rx, ry)

    def describe(self):
        name = type(self.copula).__name__.replace("Copula", "")
        rot = {0: "", 90: " (90)", 180: " (180)", 270: " (270)"}[self.rotation]
        return f"{name}{rot}"
