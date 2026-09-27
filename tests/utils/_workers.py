"""Module-level (picklable) helpers for the ``backend="process"`` test cases.

A lambda or a closure can't be pickled, so anything exercised under a real
``ProcessPoolExecutor`` needs a plain module-level function -- this file holds them,
kept separate from ``tests.py``/``e2e.py`` so it is never collected as a test module
itself (its name doesn't match ``python_files``).
"""

import numpy as np

from stochpylib.montecarlo import MCResult


def crude_square_mc(n, rng):
    """One picklable ``ParallelSimulation.estimate`` chunk: mean of ``x**2`` for
    ``x ~ U(0,1)``."""
    x = rng.random(n)
    v = x ** 2
    se = float(v.std(ddof=1) / np.sqrt(n)) if n > 1 else float("nan")
    return MCResult(float(v.mean()), se, n, "crude_square_mc")


def logp_std_normal(theta):
    """Standard-normal log-density, used as a picklable MCMC target for the
    ``backend="process"`` sampler test."""
    return float(-0.5 * np.sum(theta ** 2))
