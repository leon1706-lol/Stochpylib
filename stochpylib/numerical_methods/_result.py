"""Shared result objects for stochpylib.numerical_methods.

Mirrors the library-wide convention (montecarlo.MCResult, timeseries.ForecastResult):
a point value/estimate is never shipped without its uncertainty or convergence status.
"""

from dataclasses import dataclass, field

import numpy as np

__all__ = ["QuadratureResult", "RootResult", "ODESolution", "SDESolution"]


@dataclass
class QuadratureResult:
    """Outcome of a numerical integration: value + error estimate + diagnostics."""

    value: float
    error: float = float("nan")
    n_evals: int = 0
    method: str = ""
    converged: bool = True
    extras: dict = field(default_factory=dict)

    def __float__(self):
        return float(self.value)

    def __repr__(self):
        return (f"QuadratureResult({self.value:.6g} +/- {self.error:.3g}, "
                f"n_evals={self.n_evals}, method={self.method!r}, converged={self.converged})")


@dataclass
class RootResult:
    """Outcome of a root-finding routine."""

    root: object
    f_root: object
    iterations: int
    converged: bool
    method: str
    bracket: object = None
    history: list = None

    def __post_init__(self):
        if self.history is None:
            self.history = []

    def __float__(self):
        r = np.asarray(self.root)
        if r.ndim != 0 and r.size != 1:
            raise TypeError("root is not a scalar; cannot coerce to float")
        return float(np.asarray(self.root).reshape(()))

    def __repr__(self):
        return (f"RootResult(root={self.root!r}, iterations={self.iterations}, "
                f"converged={self.converged}, method={self.method!r})")


@dataclass
class ODESolution:
    """Grid solution of an ODE IVP: times, states, and dense (interpolated) output.

    ``y`` is always ``(n, d)`` even for a scalar state (``d == 1``, use ``.y1d``).
    ``dy`` (optional) stores derivatives at the grid points for cubic-Hermite dense output;
    without it, ``__call__`` falls back to linear interpolation.
    """

    t: np.ndarray
    y: np.ndarray
    method: str = ""
    n_evals: int = 0
    n_steps: int = 0
    success: bool = True
    message: str = ""
    dy: np.ndarray = None

    def __post_init__(self):
        self.t = np.asarray(self.t, dtype=float)
        y = np.asarray(self.y, dtype=float)
        if y.ndim == 1:
            y = y[:, None]
        self.y = y
        if self.dy is not None:
            dy = np.asarray(self.dy, dtype=float)
            if dy.ndim == 1:
                dy = dy[:, None]
            self.dy = dy

    @property
    def y1d(self):
        if self.y.shape[1] != 1:
            raise ValueError("state is not scalar (d != 1); index .y directly")
        return self.y[:, 0]

    @property
    def y_final(self):
        return self.y[-1]

    def __len__(self):
        return len(self.t)

    def __call__(self, t_query):
        """Dense output: linear interpolation, or cubic Hermite if ``dy`` is stored."""
        tq = np.atleast_1d(np.asarray(t_query, dtype=float))
        scalar_in = np.ndim(t_query) == 0
        d = self.y.shape[1]
        out = np.empty((len(tq), d))
        idx = np.clip(np.searchsorted(self.t, tq) - 1, 0, len(self.t) - 2)
        for j in range(d):
            if self.dy is not None:
                out[:, j] = _hermite_eval(self.t, self.y[:, j], self.dy[:, j], tq, idx)
            else:
                t0, t1 = self.t[idx], self.t[idx + 1]
                y0, y1 = self.y[idx, j], self.y[idx + 1, j]
                frac = np.where(t1 > t0, (tq - t0) / np.where(t1 > t0, t1 - t0, 1.0), 0.0)
                out[:, j] = y0 + frac * (y1 - y0)
        if scalar_in:
            return out[0, 0] if d == 1 else out[0]
        return out[:, 0] if d == 1 else out

    def __repr__(self):
        return (f"ODESolution(n={len(self.t)}, d={self.y.shape[1]}, method={self.method!r}, "
                f"success={self.success})")


def _hermite_eval(t, y, dy, tq, idx):
    t0, t1 = t[idx], t[idx + 1]
    y0, y1 = y[idx], y[idx + 1]
    m0, m1 = dy[idx], dy[idx + 1]
    h = np.where(t1 > t0, t1 - t0, 1.0)
    s = (tq - t0) / h
    h00 = 2 * s ** 3 - 3 * s ** 2 + 1
    h10 = s ** 3 - 2 * s ** 2 + s
    h01 = -2 * s ** 3 + 3 * s ** 2
    h11 = s ** 3 - s ** 2
    return h00 * y0 + h10 * h * m0 + h01 * y1 + h11 * h * m1


@dataclass
class SDESolution:
    """Grid solution of an SDE: shared time grid, per-path states, driving Brownian paths."""

    t: np.ndarray
    paths: np.ndarray
    method: str = ""
    brownian: np.ndarray = None

    def __post_init__(self):
        self.t = np.asarray(self.t, dtype=float)
        p = np.asarray(self.paths, dtype=float)
        if p.ndim == 2:
            p = p[:, :, None]
        self.paths = p

    @property
    def terminal(self):
        return self.paths[:, -1, :]

    def mean_path(self):
        return self.paths.mean(axis=0)

    def var_path(self):
        return self.paths.var(axis=0, ddof=1) if self.paths.shape[0] > 1 else np.zeros(self.paths.shape[1:])

    def __len__(self):
        return self.paths.shape[0]

    def __repr__(self):
        return (f"SDESolution(n_paths={self.paths.shape[0]}, n_steps={self.paths.shape[1] - 1}, "
                f"d={self.paths.shape[2]}, method={self.method!r})")
