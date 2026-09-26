"""Result object for point-process summary functions."""

from dataclasses import dataclass, field

import numpy as np

__all__ = ["SpatialFunction"]


@dataclass
class SpatialFunction:
    """A summary function (K, pair-correlation, G, ...) evaluated over a grid of radii.

    ``lower``/``upper`` are a Monte Carlo simulation envelope (``None`` when
    ``n_simulations=0``); ``pvalue`` is the global MAD envelope test's p-value in that case.
    """

    r: np.ndarray
    estimate: np.ndarray
    theoretical: np.ndarray
    name: str = ""
    correction: str = ""
    lower: np.ndarray | None = None
    upper: np.ndarray | None = None
    pvalue: float | None = None
    extras: dict = field(default_factory=dict)

    def __post_init__(self):
        self.r = np.asarray(self.r, dtype=float)
        self.estimate = np.asarray(self.estimate, dtype=float)
        self.theoretical = np.asarray(self.theoretical, dtype=float)

    def L(self):
        """Besag's L transform: ``(K / omega_d) ** (1/d)`` (variance-stabilized K)."""
        from scipy import special

        d = int(self.extras.get("dim", 2))
        omega_d = np.pi ** (d / 2.0) / special.gamma(d / 2.0 + 1.0)
        with np.errstate(invalid="ignore", divide="ignore"):
            est = (self.estimate / omega_d) ** (1.0 / d)
            theo = (self.theoretical / omega_d) ** (1.0 / d)
        return SpatialFunction(self.r, est, theo, name=f"L ({self.name})",
                                correction=self.correction, extras=self.extras)

    def __repr__(self):
        return (f"SpatialFunction(name={self.name!r}, correction={self.correction!r}, "
                f"n_r={len(self.r)}, pvalue={self.pvalue!r})")
