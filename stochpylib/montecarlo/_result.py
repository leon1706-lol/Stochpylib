"""Shared result object for Monte Carlo estimators.

Every estimator in :mod:`stochpylib.montecarlo` returns an :class:`MCResult` so callers
always get the point estimate together with its standard error and a confidence interval
(ARCHITECTURE.md convention: ``random_state=`` for seeds, ``MCResult`` for outputs).
"""

from dataclasses import dataclass, field

import numpy as np

_Z_95 = 1.959963984540054  # two-sided normal quantile for the default level


@dataclass
class MCResult:
    """Point estimate + uncertainty from a Monte Carlo experiment.

    Supports ``float(result)`` (yields :attr:`estimate`) so results can be used directly
    where a plain number is expected.
    """

    estimate: float
    std_error: float = float("nan")
    n_samples: int = 0
    method: str = "monte carlo"
    extras: dict = field(default_factory=dict)

    def confidence_interval(self, level: float = 0.95):
        """Normal-approximation two-sided confidence interval ``(low, high)``."""
        if not np.isfinite(self.std_error):
            return (float("nan"), float("nan"))
        # invert two-sided normal quantile without hardcoding tables
        from scipy import optimize, special

        if level == 0.95:
            z = _Z_95
        else:
            alpha2 = (1.0 - level) / 2.0
            z = float(optimize.brentq(lambda q: 2.0 * special.ndtr(q) - 1.0 - 2.0 * alpha2, 0.0, 50.0))
        return (self.estimate - z * self.std_error, self.estimate + z * self.std_error)

    def __float__(self):
        return float(self.estimate)

    def __repr__(self):
        lo, hi = self.confidence_interval()
        return (
            f"MCResult({self.estimate:.6g} ± {self.std_error:.3g} "
            f"[{lo:.4g}, {hi:.4g}] n={self.n_samples}, method={self.method!r})"
        )
