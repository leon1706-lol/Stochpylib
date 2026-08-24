"""Weighted log-rank family: hypothesis tests comparing survival curves.

All five tests share one statistic: at each pooled event time the O-E and its
hypergeometric variance are accumulated per group under a weight, giving a
chi-square test with (groups - 1) degrees of freedom.

- ``LogRankTest``        weight 1
- ``WilcoxonSurvival``   weight = number at risk (Gehan-Breslow)
- ``TaroneWareTest``     weight = sqrt(number at risk)
- ``PetoTest``           weight = Peto-Peto S̃(t)
- ``FlemingHarrington``  weight = S̃(t-)^rho (1-S̃(t-))^gamma
"""

import numpy as np
from scipy import stats as sps

from stochpylib.survival._base import _check_durations_events

__all__ = [
    "LogRankTest", "WilcoxonSurvival", "TaroneWareTest", "PetoTest",
    "FlemingHarrington",
]


def _weighted_logrank(durations, events, groups, weight_fn):
    durations = np.asarray(durations, dtype=float).ravel()
    events = np.asarray(events, dtype=float).ravel()
    groups = np.asarray(groups).ravel()
    if not (len(durations) == len(events) == len(groups)):
        raise ValueError("durations/events/groups must have equal length")
    g_labels = sorted(set(groups.tolist()), key=str)
    if len(g_labels) < 2:
        raise ValueError("need at least two groups")
    g_idx = {g: j for j, g in enumerate(g_labels)}
    gnum = np.array([g_idx[g] for g in groups])
    G = len(g_labels)

    order = np.argsort(durations, kind="mergesort")
    ts, es, gs = durations[order], events[order], gnum[order]
    n_total = len(ts)
    uniq = np.unique(ts[es == 1])
    starts_all = np.searchsorted(ts, uniq, side="left")
    ends_all = np.searchsorted(ts, uniq, side="right")

    O_minus_E = np.zeros(G)
    obs_vec = np.zeros(G)
    exp_vec = np.zeros(G)
    V_full = np.zeros((G, G))

    s_prev = 1.0                 # pooled KM just before current event time
    ptr = 0

    for _k, u in enumerate(uniq):
        # advance through everything strictly before u; censoring does not
        # change the KM, deaths do
        while ptr < n_total and ts[ptr] < u - 1e-12:
            if es[ptr] > 0:
                at_risk_here = float(n_total - ptr)
                s_prev *= 1.0 - 1.0 / max(at_risk_here, 1e-12)
            ptr += 1
        blk = slice(int(starts_all[_k]), int(ends_all[_k]))
        risk = slice(int(starts_all[_k]), n_total)
        nrisk_g = np.bincount(gs[risk], minlength=G).astype(float)
        n_risk = float(nrisk_g.sum())
        ev_in_block = gs[blk][es[blk] > 0].astype(int)
        drisk_g = np.bincount(ev_in_block, minlength=G).astype(float)
        d_total = float(drisk_g.sum())
        if d_total == 0 or n_risk <= 0:
            continue
        w = float(max(weight_fn(u, n_risk, n_total, s_prev), 0.0))
        pg = nrisk_g / max(n_risk, 1e-12)
        exp_g = d_total * pg
        O_minus_E += w * (drisk_g - exp_g)
        obs_vec += w * drisk_g
        exp_vec += w * exp_g
        corr = (n_risk - d_total) / max(n_risk - 1.0, 1.0)
        for i in range(G):
            vii = d_total * pg[i] * (1.0 - pg[i]) * corr
            V_full[i, i] += w ** 2 * vii
            for j in range(i + 1, G):
                vij = -d_total * pg[i] * pg[j] * corr
                V_full[i, j] += w ** 2 * vij
                V_full[j, i] += w ** 2 * vij
        # update pooled KM through the deaths AT u
        s_prev *= 1.0 - d_total / max(n_risk, 1e-12)

    stat = float(O_minus_E @ np.linalg.pinv(V_full) @ O_minus_E)
    dfree = G - 1
    return {
        "test_statistic": stat,
        "degrees_of_freedom": dfree,
        "p_value": float(sps.chi2.sf(stat, dfree)),
        "observed": obs_vec,
        "expected": exp_vec,
        "groups": g_labels,
    }


class _WeightedLogRankBase:
    """Shared fluent surface for the weighted log-rank family."""

    weight_desc = "1"

    def __init__(self):
        self.result_ = None

    def _weight(self, u, n_at_risk, n_total, s_pooled):
        raise NotImplementedError

    def fit(self, durations, events, groups):
        t, e = _check_durations_events(durations, events)
        self.result_ = _weighted_logrank(
            t, e, groups,
            lambda u, n, N, s: self._weight(u, n, N, s))
        self.test_statistic_ = self.result_["test_statistic"]
        self.p_value_ = self.result_["p_value"]
        self.degrees_of_freedom_ = self.result_["degrees_of_freedom"]
        self.groups_ = self.result_["groups"]
        return self


class LogRankTest(_WeightedLogRankBase):
    """Standard Mantel-Haenzel log-rank test (weight 1)::

        res = LogRankTest().fit(durations, events, groups)
        res.p_value
    """

    weight_desc = "1"

    def _weight(self, u, n_at_risk, n_total, s_pooled):
        return 1.0


class WilcoxonSurvival(_WeightedLogRankBase):
    """Gehan-Breslow weighted log-rank (weight = number at risk)."""

    weight_desc = "number at risk"

    def _weight(self, u, n_at_risk, n_total, s_pooled):
        return float(n_at_risk)


class TaroneWareTest(_WeightedLogRankBase):
    """Tarone-Ware weighted log-rank (weight = sqrt(number at risk))."""

    weight_desc = "sqrt(number at risk)"

    def _weight(self, u, n_at_risk, n_total, s_pooled):
        return float(np.sqrt(max(n_at_risk, 0.0)))


class PetoTest(_WeightedLogRankBase):
    """Peto-Peto weighted log-rank (weight = pooled S̃(t))."""

    weight_desc = "pooled S(t)"

    def _weight(self, u, n_at_risk, n_total, s_pooled):
        return float(s_pooled)


class FlemingHarrington(_WeightedLogRankBase):
    """Fleming-Harrington weighted log-rank:
    weight = S̃(t-)^rho (1-S̃(t-))^gamma."""

    def __init__(self, rho=1.0, gamma=0.0):
        super().__init__()
        self.rho = float(rho)
        self.gamma = float(gamma)

    @property
    def weight_desc(self):
        return f"S^({self.rho})*(1-S)^({self.gamma})"

    def _weight(self, u, n_at_risk, n_total, s_pooled):
        sv = float(np.clip(s_pooled, 1e-12, 1.0))
        return sv ** self.rho * (1.0 - sv) ** self.gamma
