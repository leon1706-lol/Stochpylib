"""Survival-function wrappers bridging raw data fits and distribution objects.

Every wrapper accepts either
- a *fitted nonparametric* object exposing ``survival_function_`` /
  ``cumulative_hazard_`` (KaplanMeier / NelsonAalen), or
- a *library distribution* instance exposing ``cdf`` / ``ppf``
  (stochpylib.distributions), or
- a *fitted parametric* survival model exposing ``survival_(t)`` /
  ``hazard_(t)``.

and provides a uniform ``predict(times)`` surface.
"""

import numpy as np

from stochpylib.survival.nonparametric import KaplanMeier, NelsonAalen
from stochpylib.survival._base import (
    SurvivalFitter,
    _check_durations_events,
)

__all__ = [
    "SurvivalFunction", "HazardFunction", "CumulativeHazard",
    "ResidualLifetime", "MeanResidualLife",
]


def _as_sf_callable(source):
    """Return sf(times) callable for any accepted source."""
    if hasattr(source, "survival_function_"):          # KaplanMeier
        return lambda t: np.asarray(
            SurvivalFitter._step_evaluate(source.survival_function_, t))
    if hasattr(source, "survival_") and callable(source.survival_):
        return lambda t: np.asarray(source.survival_(t), dtype=float)
    if hasattr(source, "cdf") and callable(source.cdf):
        return lambda t: 1.0 - np.asarray(source.cdf(np.asarray(t)), float)
    raise TypeError("source must be a KaplanMeier fit, a parametric "
                    "survival model with survival_(t), or a distribution "
                    "object with cdf(t)")


def _as_hazard_callable(source):
    """Return hazard(times) callable for any accepted source: NelsonAalen
    fits (discrete step rates), parametric survival models (hazard_) or
    library distributions (hazard(t))."""
    if hasattr(source, "cumulative_hazard_") and hasattr(source, "predict"):
        arr = source.cumulative_hazard_
        times = np.asarray(arr["time"])
        vals = np.asarray(arr["value"])
        inc = np.diff(np.concatenate([[0.0], vals]))
        dur = np.diff(np.concatenate([[times[0] - 1e-9], times]))
        rates = inc / np.maximum(dur, 1e-12)

        def haz(t):
            idx = np.searchsorted(times, np.atleast_1d(np.asarray(
                t, dtype=float)), side="right")
            out = np.zeros(len(idx))
            okk = idx > 0
            out[okk] = rates[idx[okk] - 1]
            return out
        return haz
    if hasattr(source, "hazard_") and callable(source.hazard_):
        return lambda t: np.asarray(source.hazard_(t), dtype=float)
    if hasattr(source, "hazard") and callable(source.hazard):
        return lambda t: np.asarray(
            source.hazard(np.atleast_1d(np.asarray(t, dtype=float))),
            dtype=float)
    raise TypeError("source must be a NelsonAalen fit, a parametric survival "
                    "model with hazard_(t), or a distribution with hazard(t)")


class _FromDataOrSourceMixin:
    @classmethod
    def _unwrap_source(cls, source=None, durations=None, events=None):
        if source is not None:
            return source, None
        if durations is not None:
            return None, (np.asarray(durations, float),
                          None if events is None else
                          np.asarray(events, float))
        raise ValueError("provide either `source` or `durations`")


class SurvivalFunction(_FromDataOrSourceMixin):
    """S(t) from data (Kaplan-Meier) or directly from a distribution/model::

        SurvivalFunction(durations=t, events=e).predict([1, 2])
        SurvivalFunction(source=Weibull(1.5, 10)).predict([1, 2])
    """

    def __init__(self, source=None, durations=None, events=None):
        src, data = self._unwrap_source(source, durations, events)
        if src is None:
            src = KaplanMeier().fit(*data)
        self._sf = _as_sf_callable(src)
        self.source_ = src

    def predict(self, times):
        return np.asarray(self._sf(np.asarray(times, dtype=float)),
                          dtype=float)


class CumulativeHazard(_FromDataOrSourceMixin):
    """H(t) from data (Nelson-Aalen), from a survival source (-log S), or a
    hazard-bearing model integrated numerically."""

    def __init__(self, source=None, durations=None, events=None):
        src, data = self._unwrap_source(source, durations, events)
        if src is None:
            src = NelsonAalen().fit(*data)
        if hasattr(src, "cumulative_hazard_"):
            self._haz = _as_hazard_callable(src)
            self._mode = "na_step"
        elif hasattr(src, "hazard_") and callable(src.hazard_) \
                or (hasattr(src, "hazard") and callable(src.hazard)):
            self._haz = _as_hazard_callable_dist(src)
            self._mode = "integrate"
        else:
            self._sf = _as_sf_callable(src)
            self._mode = "from_survival"
        self.source_ = src

    def predict(self, times):
        times = np.atleast_1d(np.asarray(times, dtype=float))
        if self._mode == "na_step":
            # evaluate the cumulative-hazard step function directly
            from stochpylib.survival._base import SurvivalFitter
            return np.asarray(
                SurvivalFitter._step_evaluate(
                    self.source_.cumulative_hazard_, times), dtype=float)
        if self._mode == "integrate":
            lo = max(float(times.min()) * .5, 1e-6)
            hi = float(times.max()) * 1.02 + 1e-6
            grid = np.linspace(lo, hi, 20001)
            h = np.asarray(self._haz(grid), dtype=float)
            H = np.concatenate([[0.0], np.cumsum(h * (grid[1] - grid[0]))])
            idx = np.clip(np.searchsorted(grid, times), 0, len(H) - 1)
            return H[idx]
        s = np.clip(self._sf(times), 1e-300, 1.0)
        return -np.log(s)


def _as_hazard_callable_dist(source):
    if hasattr(source, "hazard_") and callable(source.hazard_):
        return lambda t: np.asarray(source.hazard_(t), dtype=float)
    if hasattr(source, "hazard") and callable(source.hazard):
        return lambda t: np.asarray(
            source.hazard(np.asarray(t, dtype=float)), dtype=float)
    raise TypeError("no hazard callable")


class HazardFunction(_FromDataOrSourceMixin):
    """Discrete hazard rates from data (Nelson-Aalen increments) or the
    parametric/distribution hazard."""

    def __init__(self, source=None, durations=None, events=None):
        src, data = self._unwrap_source(source, durations, events)
        if src is None:
            src = NelsonAalen().fit(*data)
        self._haz = _as_hazard_callable(src)
        self.source_ = src

    def predict(self, times):
        return np.asarray(self._haz(np.asarray(times, dtype=float)),
                          dtype=float)


def _mrl_from_survival(sf, t0, upper=1e6, n_points=6000):
    """Mean residual life mrl(t0)= int_{t0}^inf S(u) du / S(t0).

    Integrated on a log-time grid so heavy-decay survival curves are resolved
    even when the effective upper bound spans orders of magnitude."""
    t0 = float(t0)
    hi = float(min(max(upper, t0 * 2.0), 1e8))
    lo_u = np.log1p(t0)
    hi_u = np.log1p(hi)
    ug = np.linspace(lo_u, hi_u, n_points)
    tg = np.expm1(ug)
    su = np.asarray(sf(tg), dtype=float).ravel()
    area = float(np.trapezoid(su * np.exp(ug), ug))
    s0 = float(np.asarray(sf(np.array([t0]))).ravel()[0])
    return area / max(s0, 1e-300)


class ResidualLifetime(_FromDataOrSourceMixin):
    """Expected remaining lifetime given survival to ``t``::

        rl = ResidualLifetime(durations=t_obs, events=e)
        rl.value(2.0)

    Also accepts any supported source (distribution objects included).
    """

    def __init__(self, source=None, durations=None, events=None, upper=1e6):
        src, data = self._unwrap_source(source, durations, events)
        if src is None:
            src = KaplanMeier().fit(*data)
        self._sf = _as_sf_callable(src)
        self.source_ = src
        self.upper = float(upper)
        # nonparametric survival functions are only defined up to the last
        # observed time — never integrate their plateau beyond it
        if hasattr(src, "survival_function_"):
            arr = src.survival_function_
            self.upper = min(self.upper, float(np.asarray(arr["time"])[-1]))

    def value(self, t):
        return float(_mrl_from_survival(self._sf, t, upper=self.upper))

    def predict(self, times):
        return np.array([self.value(float(t)) for t in
                         np.atleast_1d(np.asarray(times, dtype=float))])


class MeanResidualLife(ResidualLifetime):
    """Curve view of the mean residual life over a time grid::

        mrl = MeanResidualLife(source=fitted_km)
        mrl.curve([0, 1, 2])   -> array of mrl values
    """

    def curve(self, times):
        times = np.atleast_1d(np.asarray(times, dtype=float))
        return np.array([self.value(float(t)) for t in times])

    def predict(self, times):
        return self.curve(times)
