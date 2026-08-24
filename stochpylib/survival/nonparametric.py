"""Nonparametric survival estimation: Kaplan-Meier, Nelson-Aalen, life tables.

All estimators follow the shared conventions in
:mod:`stochpylib.survival._base` — fluent ``fit(durations, events)``,
structured step arrays, right-continuous evaluation.
"""

import numpy as np

from stochpylib.survival._base import SurvivalFitter, _check_durations_events

__all__ = [
    "KaplanMeier", "NelsonAalen", "LifeTable", "EmpiricalSurvival",
    "BreslowEstimator",
]

_EPS = 1e-12


def _event_table(times, events):
    """Event-time table: unique times WITH >=1 event, (n_at_risk, n_events).

    Censoring-only times carry no KM/NA jump and are excluded from the grid
    (they still reduce subsequent risk-set sizes through ``at_risk``)."""
    order = np.argsort(times, kind="mergesort")
    ts = times[order]
    es = events[order]
    uniq = np.unique(ts)
    idx_of = np.searchsorted(uniq, ts)          # each subject -> its block
    deaths_all = np.bincount(idx_of, weights=es.astype(float),
                             minlength=len(uniq))
    keep = deaths_all > 0
    ev_uniq = uniq[keep]
    deaths = deaths_all[keep]
    at_risk = len(ts) - np.searchsorted(ts, ev_uniq, side="left")
    return ev_uniq, at_risk.astype(float), deaths.astype(float)


class KaplanMeier(SurvivalFitter):
    """Kaplan-Meier product-limit estimator of the survival function::

        km = KaplanMeier().fit(durations, events)
        km.median_survival_time()
        km.predict([1.0, 2.0])            # S(t) at arbitrary times

    Confidence intervals use Greenwood's variance on the complementary
    log-minus-log scale by default (``ci_method="loglog"`` or ``"linear"``).
    """

    def __init__(self, ci_method="loglog", alpha=0.05):
        if ci_method not in ("loglog", "linear"):
            raise ValueError("ci_method must be 'loglog' or 'linear'")
        self.ci_method = ci_method
        self.alpha = float(alpha)
        self.survival_function_ = None
        self.confidence_interval_ = None
        self.median_survival_time_ = None

    def fit(self, durations, events=None):
        t, e = _check_durations_events(durations, events)
        uniq, at_risk, deaths = _event_table(t, e)
        ratios = 1.0 - deaths / np.maximum(at_risk, _EPS)
        s = np.cumprod(ratios)
        self.survival_function_ = self._step_array(uniq, s)

        # Greenwood variance of log S
        greenwood = np.cumsum(deaths / np.maximum(
            at_risk * (at_risk - deaths), _EPS))
        var_log_s = s ** 2 * greenwood          # var(S) = S^2 * sum(...)
        z = stats_z(self.alpha)
        if self.ci_method == "loglog":
            with np.errstate(divide="ignore", invalid="ignore", over="ignore"):
                c = np.log(-np.log(np.clip(s, _EPS, 1 - _EPS)))
                se_cll = np.sqrt(greenwood) / np.abs(np.log(
                    np.clip(s, _EPS, 1 - _EPS)))
                arg_lo = np.clip(c + z * se_cll, -700.0, 700.0)
                arg_hi = np.clip(c - z * se_cll, -700.0, 700.0)
            lo = np.exp(-np.exp(arg_lo))
            hi = np.exp(-np.exp(arg_hi))
        else:
            lo = s - z * np.sqrt(var_log_s)
            hi = s + z * np.sqrt(var_log_s)
        lo = np.clip(lo, 0.0, 1.0)
        hi = np.clip(hi, 0.0, 1.0)
        ci = np.empty(len(s), dtype=[("time", float),
                                     ("lower", float), ("upper", float)])
        ci["time"], ci["lower"], ci["upper"] = uniq, lo, hi
        self.confidence_interval_ = ci[np.argsort(ci["time"])]

        # median: first time S drops to <= 0.5 (inf if never)
        below = np.flatnonzero(s <= 0.5)
        self.median_survival_time_ = float(uniq[below[0]]) if len(below) \
            else float("inf")
        return self

    def predict(self, times):
        self._require_fit()
        return self._step_evaluate(self.survival_function_, times)

    def _require_fit(self):
        if self.survival_function_ is None:
            raise RuntimeError("fit() must be called first")


def stats_z(alpha):
    """Two-sided normal quantile for confidence level 1-alpha."""
    from scipy import special
    return float(special.ndtri(1.0 - alpha / 2.0))


class NelsonAalen(SurvivalFitter):
    """Nelson-Aalen estimator of the cumulative hazard function::

        na = NelsonAalen().fit(durations, events)
        na.cumulative_hazard_      # ('time','value') steps
        na.predict([1.0, 2.0])     # H(t)
    """

    def __init__(self):
        self.cumulative_hazard_ = None

    def fit(self, durations, events=None):
        t, e = _check_durations_events(durations, events)
        uniq, at_risk, deaths = _event_table(t, e)
        h = np.cumsum(deaths / np.maximum(at_risk, _EPS))
        self.cumulative_hazard_ = self._step_array(uniq, h)
        return self

    def predict(self, times):
        self._require_fit()
        return self._step_evaluate(self.cumulative_hazard_, times)

    def _require_fit(self):
        if self.cumulative_hazard_ is None:
            raise RuntimeError("fit() must be called first")


class LifeTable(SurvivalFitter):
    """Actuarial life table over fixed intervals.

    ``fit(durations, events, bins=None, width=1.0)`` — ``bins`` is an ordered
    sequence of interval edges (default: integer widths up to max time).
    Exposes arrays: n_entering_, n_censored_, n_deaths_, n_risk_,
    conditional_death_prob_, survival_, hazard_rate_, density_.
    """

    def __init__(self):
        self.interval_edges_ = None

    def fit(self, durations, events=None, bins=None, width=1.0):
        t, e = _check_durations_events(durations, events)
        tmax = float(t.max())
        if bins is None:
            top = int(np.ceil(max(tmax, 1e-9))) + 1
            edges = width * np.arange(top + 1, dtype=float)
        else:
            edges = np.asarray(bins, dtype=float)
            if edges.ndim != 1 or len(edges) < 2 or np.any(np.diff(edges) <= 0):
                raise ValueError("bins must be an increasing edge sequence")
        idx = np.clip(np.searchsorted(edges, t, side="right") - 1,
                      0, len(edges) - 2)
        k = len(edges) - 1
        deaths = np.bincount(idx, weights=e.astype(float), minlength=k)
        censured = np.bincount(idx, weights=(1 - e).astype(float),
                               minlength=k)
        # entering numbers via reverse sweep
        enter = np.zeros(k + 1)
        enter[0] = len(t)
        for j in range(k):
            enter[j + 1] = enter[j] - deaths[j] - censured[j]
        eff = enter[:k] - censured / 2.0                      # effective risk
        q = np.where(eff > 0, deaths / np.maximum(eff, _EPS), 0.0)
        p = 1.0 - q
        surv = np.cumprod(p)
        h = edges[:-1]
        hw = np.diff(edges)
        dens = q * np.concatenate([[1.0], surv[:-1]]) / hw
        mrate = np.where(p < 1, 2.0 * q / (hw * (1.0 + p)), 0.0)
        for attr, arr in (("n_entering_", enter), ("n_censored_", censured),
                          ("n_deaths_", deaths), ("n_risk_", eff),
                          ("conditional_death_prob_", q),
                          ("survival_", surv), ("hazard_rate_", mrate),
                          ("density_", dens)):
            setattr(self, attr, arr)
        self.interval_edges_ = edges
        self.n_obs_ = len(t)
        return self


class EmpiricalSurvival(SurvivalFitter):
    """Uncensored empirical survival function S(t) = 1 - F_n(t)."""

    def __init__(self):
        self.sorted_times_ = None

    def fit(self, durations, events=None):
        t, _ = _check_durations_events(durations, None)
        self.sorted_times_ = np.sort(t)
        self.n_obs_ = len(t)
        return self

    def predict(self, times):
        if self.sorted_times_ is None:
            raise RuntimeError("fit() must be called first")
        times = np.asarray(times, dtype=float)
        return 1.0 - np.searchsorted(self.sorted_times_, times,
                                     side="right") / self.n_obs_


class BreslowEstimator:
    """Breslow baseline cumulative-hazard estimator given linear predictors::

        be = BreslowEstimator().fit(durations, events, partial_hazard)
        be.cumulative_hazard_     # H0(t) steps
        be.baseline_survival_(t)  # exp(-H0(t))

    Used as the baseline layer of the Cox models in
    :mod:`stochpylib.survival.regression`.
    """

    def fit(self, durations, events, partial_hazard):
        t, e = _check_durations_events(durations, events)
        w = np.asarray(partial_hazard, dtype=float).ravel()
        if len(w) != len(t):
            raise ValueError("partial_hazard must match durations")
        order = np.argsort(t, kind="mergesort")
        ts, ws = t[order], w[order]
        uniq, at_risk, deaths = _event_table(t, e)
        # risk-sum of exp(x beta) among subjects still at risk (t >= u)
        first_at_risk = np.searchsorted(ts, uniq, side="left")
        prefix = np.concatenate([[0.0], np.cumsum(ws)])
        risk_sum = prefix[-1] - prefix[first_at_risk]
        h0 = np.cumsum(deaths / np.maximum(risk_sum, _EPS))
        self.cumulative_hazard_ = SurvivalFitter._step_array(uniq, h0)
        self.times_ = uniq
        self.values_ = h0
        return self

    def baseline_survival_(self, times=None):
        times = self.times_ if times is None else np.asarray(times, float)
        idx = np.searchsorted(self.times_, times, side="right") - 1
        out = np.ones(len(np.atleast_1d(times)))
        ok = idx >= 0
        out[ok] = np.exp(-self.values_[idx[ok]])
        return out

    def predict(self, times):
        from stochpylib.survival._base import SurvivalFitter
        return SurvivalFitter._step_evaluate(self.cumulative_hazard_, times)
