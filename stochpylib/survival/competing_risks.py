"""Competing-risks analysis: cause-specific hazards, Aalen-Johansen
cumulative incidence functions and a facade model."""

import numpy as np

from stochpylib.survival._base import _check_durations_events
from stochpylib.survival.nonparametric import KaplanMeier, NelsonAalen

__all__ = [
    "CauseSpecificHazard", "CumulativeIncidenceFunction",
    "CompetingRisksModel",
]


class CauseSpecificHazard:
    """Nelson-Aalen estimator of the cause-specific cumulative hazard for one
    cause code::

        csh = CauseSpecificHazard().fit(durations, cause, cause_of_interest=1)
        csh.predict(times)      # cumulative cause-specific hazard
    """

    def __init__(self):
        self.cumulative_hazard_ = None

    def fit(self, durations, cause, cause_of_interest=1):
        t = np.asarray(durations, dtype=float).ravel()
        c = np.asarray(cause).ravel().astype(int)
        if len(t) != len(c) or len(t) == 0:
            raise ValueError("durations/cause length mismatch")
        ev = (c == int(cause_of_interest)).astype(float)
        na = NelsonAalen().fit(t, ev)
        self.cumulative_hazard_ = na.cumulative_hazard_
        self.cause_ = int(cause_of_interest)
        self.n_obs_ = len(t)
        return self

    def predict(self, times):
        if self.cumulative_hazard_ is None:
            raise RuntimeError("fit() must be called first")
        from stochpylib.survival._base import SurvivalFitter
        return SurvivalFitter._step_evaluate(self.cumulative_hazard_, times,
                                             default=0.0)


class CumulativeIncidenceFunction:
    """Aalen-Johansen cumulative incidence for one cause::

        cif = CumulativeIncidenceFunction().fit(durations, cause,
                                                cause_of_interest=1)
        cif.predict([1.0, 2.0])     # P(T <= t, cause = k)
    """

    def __init__(self):
        self.cif_ = None
        self.overall_survival_ = None

    def fit(self, durations, cause, cause_of_interest=1):
        t = np.asarray(durations, dtype=float).ravel()
        c = np.asarray(cause).ravel().astype(int)
        if len(t) != len(c) or len(t) == 0:
            raise ValueError("durations/cause length mismatch")
        order = np.argsort(t, kind="mergesort")
        ts, cs = t[order], c[order]
        uniq = np.unique(ts)
        n_total = len(ts)

        k = int(cause_of_interest)
        s_prev = 1.0
        f_k = 0.0
        times_out, cif_out, surv_out = [], [], []
        ptr = 0
        for u in uniq:
            end = int(np.searchsorted(ts, u, side="right"))
            n_at_risk = float(n_total - ptr)
            blk = slice(ptr, end)
            blk_cause = cs[blk]
            d_any = float(np.sum(blk_cause > 0))          # all-cause events
            d_k = float(np.sum((blk_cause == k) &
                               np.isclose(ts[blk], u)))
            if n_at_risk > 0 and d_any > 0:
                f_k += s_prev * (d_k / n_at_risk)
                s_prev *= 1.0 - d_any / n_at_risk
            times_out.append(u)
            cif_out.append(f_k)
            surv_out.append(s_prev)
            ptr = end
        arr = np.empty(len(times_out),
                       dtype=[("time", float), ("value", float)])
        arr["time"] = times_out
        arr["value"] = cif_out
        self.cif_ = arr
        surv_arr = np.empty(len(times_out),
                            dtype=[("time", float), ("value", float)])
        surv_arr["time"], surv_arr["value"] = times_out, surv_out
        self.overall_survival_ = surv_arr
        self.cause_ = k
        return self

    def predict(self, times):
        if self.cif_ is None:
            raise RuntimeError("fit() must be called first")
        from stochpylib.survival._base import SurvivalFitter
        return SurvivalFitter._step_evaluate(self.cif_, times, default=0.0)


class CompetingRisksModel:
    """Facade fitting every cause's Nelson-Aalen hazard and Aalen-Johansen
    cumulative incidence at once::

        crm = CompetingRisksModel().fit(durations, cause)
        sorted(crm.causes_)              # [1, 2]
        crm.cif_[1].predict([1.0])       # Aalen-Johansen CIF for cause 1
        crm.overall_survival_.predict([1.0])
        crm.check_identity()             # sum of CIFs + KM == 1 (max abs dev)
    """

    def __init__(self):
        self.causes_ = None

    def fit(self, durations, cause):
        t = np.asarray(durations, dtype=float).ravel()
        c = np.asarray(cause).ravel().astype(int)
        if len(t) != len(c) or len(t) == 0:
            raise ValueError("durations/cause length mismatch")
        self.causes_ = sorted(set(c[c > 0].tolist()))
        self.n_obs_ = len(t)
        self.csh_ = {}
        self.cif_ = {}
        for k in self.causes_:
            self.csh_[k] = CauseSpecificHazard().fit(
                t, c, cause_of_interest=k)
            self.cif_[k] = CumulativeIncidenceFunction().fit(
                t, c, cause_of_interest=k)
        self.overall_survival_ = KaplanMeier().fit(
            t, (c > 0).astype(float))
        return self

    def check_identity(self, grid=None):
        """max |sum_k CIF_k + KM - 1| on the common event-time grid."""
        if not self.causes_:
            raise RuntimeError("fit() must be called first")
        times = np.asarray(self.overall_survival_.survival_function_["time"])
        total = np.zeros(len(times))
        for k in self.causes_:
            total += np.asarray(self.cif_[k].predict(times), dtype=float)
        kmv = np.asarray(self.overall_survival_.predict(times), dtype=float)
        return float(np.max(np.abs(total + kmv - 1.0)))
