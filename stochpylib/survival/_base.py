"""Base class and shared conventions for survival analysis."""

import numpy as np

__all__ = ["SurvivalFitter"]


def _check_durations_events(durations, events):
    """Validate (durations, events); returns float/int arrays."""
    t = np.asarray(durations, dtype=float).ravel()
    e = np.asarray(events, dtype=float).ravel() if events is not None \
        else np.ones(len(t))
    if len(t) != len(e) or len(t) == 0:
        raise ValueError("durations/events must be equal-length, non-empty")
    if not np.all(np.isfinite(t)) or np.any(t < 0):
        raise ValueError("durations must be finite and non-negative")
    if not np.all(np.isin(e, (0.0, 1.0))):
        raise ValueError("events must be 0/1 (1 = event observed, 0 = censored)")
    return t, e.astype(int)


class SurvivalFitter:
    """Shared conventions for all survival fitters.

    - ``fit(durations, events=None)`` is fluent and returns ``self``; fitted
      quantities live on the instance as attributes ending in ``_``.
    - ``survival_function_`` / ``cumulative_hazard_`` are structured arrays
      with fields ``("time", "value")`` sorted by time.
    - ``predict(times)`` evaluates the fitted quantity at arbitrary times
      (right-continuous step functions where nonparametric).
    """

    def fit(self, durations, events=None):
        raise NotImplementedError

    def predict(self, times):
        raise NotImplementedError

    @staticmethod
    def _step_array(times, values):
        arr = np.empty(len(times), dtype=[("time", float), ("value", float)])
        arr["time"] = np.asarray(times, dtype=float)
        arr["value"] = np.asarray(values, dtype=float)
        order = np.argsort(arr["time"], kind="mergesort")
        return arr[order]

    @staticmethod
    def _step_evaluate(step_arr, times):
        """Right-continuous evaluation of a ('time','value') step array."""
        times = np.atleast_1d(np.asarray(times, dtype=float))
        idx = np.searchsorted(step_arr["time"], times, side="right") - 1
        out = np.empty(len(idx))
        ok = idx >= 0
        out[~ok] = 1.0            # before first event: S=1, H=0 convention
        if np.any(ok):
            out[ok] = step_arr["value"][idx[ok]]
        return out
