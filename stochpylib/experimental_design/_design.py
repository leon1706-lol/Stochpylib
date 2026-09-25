"""The shared result object of :mod:`stochpylib.experimental_design`.

Every generator returns a :class:`Design`: the run matrix plus the metadata needed to use
it (factor names, which space the points live in, natural-unit bounds, and the design's
own properties -- resolution, alpha, efficiencies, ...). A design matrix is not an
estimate, so it gets its own type rather than bending ``MCResult``/``TestResult`` into one.
"""

from dataclasses import dataclass, field, replace

import numpy as np

from stochpylib.experimental_design._common import (
    _as_2d,
    _discrepancy,
    _factor_names,
    _logdet,
    _max_abs_correlation,
    _min_distance,
    _model_matrix,
    _nearest_distance,
    _normalize_bounds,
    _rng,
    _sobol_reference,
)

__all__ = ["Design"]


@dataclass
class Design:
    """A design matrix with its metadata.

    ``points`` is ``(n_runs, n_factors)``. ``space`` is ``"coded"`` (factor levels in
    ``[-1, 1]``: classical and optimal designs) or ``"unit"`` (``[0, 1]``: space-filling
    designs) -- or ``"natural"`` for designs built directly on user bounds. ``bounds`` maps
    the points to natural units via :meth:`to_natural`. ``np.asarray(design)`` yields the
    points, so a Design can be passed anywhere an array is expected.
    """

    points: np.ndarray
    factor_names: list = None
    space: str = "coded"
    kind: str = ""
    bounds: np.ndarray = None
    properties: dict = field(default_factory=dict)
    run_order: np.ndarray = None

    def __post_init__(self):
        self.points = np.asarray(self.points, dtype=float)
        if self.points.ndim == 1:
            self.points = self.points[:, None]
        if self.factor_names is None:
            self.factor_names = _factor_names(self.points.shape[1])
        if len(self.factor_names) != self.points.shape[1]:
            raise ValueError("factor_names must have one entry per column")
        if self.bounds is not None:
            self.bounds = _normalize_bounds(self.bounds, self.points.shape[1])

    # ------------------------------------------------------------------ array protocol
    def __array__(self, dtype=None, copy=None):
        out = self.points if dtype is None else self.points.astype(dtype)
        return out.copy() if copy else out

    def __len__(self):
        return self.points.shape[0]

    @property
    def n_runs(self):
        return self.points.shape[0]

    @property
    def n_factors(self):
        return self.points.shape[1]

    @property
    def shape(self):
        return self.points.shape

    # ---------------------------------------------------------------- unit conversion
    def _bounds_or(self, bounds):
        b = self.bounds if bounds is None else _normalize_bounds(bounds, self.n_factors)
        if b is None:
            raise ValueError("no bounds given and none stored on the design")
        return b

    def unit_points(self):
        """The points rescaled into the unit cube (used by the space-filling metrics)."""
        if self.space == "unit":
            return self.points.copy()
        if self.space == "coded":
            return (np.clip(self.points, -1.0, 1.0) + 1.0) / 2.0
        b = self._bounds_or(None)
        return (self.points - b[:, 0]) / (b[:, 1] - b[:, 0])

    def to_natural(self, bounds=None):
        """The points in natural units, ``lo + fraction * (hi - lo)`` per factor."""
        if self.space == "natural":
            return self.points.copy()
        b = self._bounds_or(bounds)
        frac = self.points if self.space == "unit" else (self.points + 1.0) / 2.0
        return b[:, 0] + frac * (b[:, 1] - b[:, 0])

    def to_coded(self, X_natural, bounds=None):
        """Inverse of :meth:`to_natural` for arbitrary natural-unit points."""
        b = self._bounds_or(bounds)
        frac = (_as_2d(X_natural) - b[:, 0]) / (b[:, 1] - b[:, 0])
        return frac if self.space == "unit" else 2.0 * frac - 1.0

    # ------------------------------------------------------------------ manipulation
    def randomized(self, random_state=None):
        """A copy with the runs in random order; ``run_order`` holds the permutation."""
        perm = _rng(random_state).permutation(self.n_runs)
        return replace(self, points=self.points[perm].copy(), run_order=perm,
                       properties=dict(self.properties))

    def with_center_points(self, n):
        """A copy with ``n`` centre runs appended (coded designs only)."""
        if self.space != "coded":
            raise ValueError("centre points are defined for coded designs only")
        pts = np.vstack([self.points, np.zeros((int(n), self.n_factors))])
        props = dict(self.properties)
        props["n_center"] = props.get("n_center", 0) + int(n)
        return replace(self, points=pts, properties=props, run_order=None)

    # ------------------------------------------------------------------------ metrics
    def model_matrix(self, model="linear"):
        """``(F, term_names)`` for a polynomial model over this design."""
        return _model_matrix(self.points, model, self.factor_names)

    def d_efficiency(self, model="linear"):
        """``det(F'F)^(1/p) / n``: 1 for an orthogonal two-level design, 0 if singular."""
        F, _ = self.model_matrix(model)
        ld = _logdet(F.T @ F)
        return 0.0 if not np.isfinite(ld) else float(np.exp(ld / F.shape[1]) / len(F))

    def min_distance(self):
        """Smallest pairwise Euclidean distance between runs (in the unit cube)."""
        return _min_distance(self.unit_points())

    def fill_distance(self, n_ref=4096, random_state=0):
        """Largest distance from the unit cube to the nearest run.

        Estimated on ``n_ref`` Sobol reference points, so it is a (tight) lower bound on
        the true fill distance.
        """
        U = self.unit_points()
        ref = _sobol_reference(U.shape[1], n_ref, random_state)
        return float(_nearest_distance(ref, U).max())

    def discrepancy(self, method="CD"):
        """L2-type discrepancy of the runs rescaled to the unit cube."""
        return _discrepancy(self.unit_points(), method)

    def max_abs_correlation(self):
        """Largest absolute pairwise correlation between factor columns."""
        return _max_abs_correlation(self.points)

    def __repr__(self):
        return (f"Design(kind={self.kind!r}, n_runs={self.n_runs}, "
                f"n_factors={self.n_factors}, space={self.space!r})")
