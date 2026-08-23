"""Forecast result object for the timeseries module."""

from dataclasses import dataclass

import numpy as np

_Z95 = 1.959963984540054


@dataclass
class ForecastResult:
    """Multi-step forecast: point path plus per-step standard deviation.

    Returned by every ``forecast(horizon=...)`` in :mod:`stochpylib.timeseries`.
    ``confidence_interval(level)`` gives normal-approximation ``(low, high)`` paths.
    """

    mean: np.ndarray
    std: np.ndarray

    def __post_init__(self):
        self.mean = np.asarray(self.mean, dtype=float)
        self.std = np.asarray(self.std, dtype=float)
        if self.mean.shape != self.std.shape:
            raise ValueError("mean and std must share one shape")

    def confidence_interval(self, level: float = 0.95):
        if level == 0.95:
            z = _Z95
        else:
            from scipy import optimize, special

            alpha2 = (1.0 - level) / 2.0
            z = float(optimize.brentq(lambda q: 2.0 * special.ndtr(q) - 1.0 - 2.0 * alpha2, 0.0, 50.0))
        return self.mean - z * self.std, self.mean + z * self.std

    def __len__(self):
        return len(self.mean)

    def __repr__(self):
        head = ", ".join(f"{v:.4g}" for v in self.mean[:5])
        more = "..." if len(self.mean) > 5 else ""
        return f"ForecastResult(mean=[{head}{more}], n={len(self.mean)})"


@dataclass
class TestResult:
    """Outcome of a classical time-series hypothesis test.

    ``statistic`` is the test statistic, ``pvalue`` its (approximate) p-value,
    ``critical_values`` maps significance levels to critical values where published
    tables exist, and ``null`` states the null hypothesis in words.
    """

    statistic: float
    pvalue: float | None
    null: str
    critical_values: dict = None

    def __post_init__(self):
        self.critical_values = self.critical_values or {}

    def __repr__(self):
        pv = f"{self.pvalue:.4f}" if self.pvalue is not None else "n/a"
        return f"TestResult(stat={self.statistic:.4f}, p={pv}, {self.null})"
