"""Shared result object for stochpylib.optimization.

Mirrors the library-wide convention (montecarlo.MCResult, numerical_methods.RootResult):
a solution is never shipped without its objective value, effort counters and convergence
status. Estimators whose answer is itself a Monte Carlo quantity (``SAA``'s optimality
gap, ``CEM``'s objective estimate) put an :class:`stochpylib.montecarlo.MCResult` in
``extras`` rather than defining a second result type.
"""

from dataclasses import dataclass, field

import numpy as np

__all__ = ["OptimizeResult"]


@dataclass
class OptimizeResult:
    """Outcome of a minimization: the point, its value, and how it got there.

    ``converged`` reflects the optimizer's own stopping test; a run that exhausted
    ``max_iter`` returns ``converged=False`` rather than raising, so a caller can always
    inspect the best point found. ``history`` is the per-iteration objective value and is
    always populated; ``trajectory`` holds the per-iteration iterate and is only filled in
    when the optimizer was constructed with ``track_trajectory=True`` (it is O(nit * dim)).
    """

    x: np.ndarray
    fun: float
    nit: int = 0
    nfev: int = 0
    njev: int = 0
    converged: bool = False
    method: str = ""
    message: str = ""
    jac: np.ndarray = None
    hess_inv: np.ndarray = None
    history: list = None
    trajectory: list = None
    extras: dict = field(default_factory=dict)

    def __post_init__(self):
        self.x = np.atleast_1d(np.asarray(self.x, dtype=float))
        self.fun = float(self.fun)
        if self.history is None:
            self.history = []
        if self.trajectory is None:
            self.trajectory = []

    @property
    def x_scalar(self):
        """The solution as a plain float; raises when the problem is not one-dimensional."""
        if self.x.size != 1:
            raise ValueError(f"solution has dimension {self.x.size}; index .x directly")
        return float(self.x.reshape(()))

    def __float__(self):
        return float(self.fun)

    def __repr__(self):
        return (f"OptimizeResult(fun={self.fun:.6g}, x={np.round(self.x, 6)!r}, "
                f"nit={self.nit}, nfev={self.nfev}, converged={self.converged}, "
                f"method={self.method!r})")
