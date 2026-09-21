"""Rank-based and distance-based bivariate dependence measures."""

import math

import numpy as np

from stochpylib.copulas._utils import kendall_tau_estimate, spearman_rho_estimate
from stochpylib.nonparametric._base import DependenceMeasure
from stochpylib.nonparametric._common import (
    _chi2_sf, _double_center, _norm_cdf, _pairwise_dist, _rank_ties, _u_center,
)

__all__ = ["SpearmanCorrelation", "KendallTau", "RankCorrelation",
           "DistanceCorrelation", "BrownianCorrelation", "HoeffdingD"]


class SpearmanCorrelation(DependenceMeasure):
    """Spearman's rank correlation: rho via
    :func:`stochpylib.copulas._utils.spearman_rho_estimate`, a t-approximation
    p-value matching ``scipy.stats.spearmanr`` exactly, Fisher-z confidence
    interval.
    """

    def __init__(self, alternative="two-sided"):
        self.alternative = alternative

    def _fit(self, x, y):
        n = len(x)
        rho = spearman_rho_estimate(x, y)
        if n > 2:
            t = rho * math.sqrt((n - 2) / max(1e-300, 1 - rho ** 2))
            from stochpylib.nonparametric._common import _t_sf
            if self.alternative == "two-sided":
                p = 2.0 * float(_t_sf(abs(t), n - 2))
            elif self.alternative == "greater":
                p = float(_t_sf(t, n - 2))
            else:
                p = 1.0 - float(_t_sf(t, n - 2))
            p = float(np.clip(p, 0.0, 1.0))
        else:
            p = float("nan")
        se = 1.0 / math.sqrt(n - 3) if n > 3 else float("nan")
        return rho, p, {"std_error": se, "fisher_z": True}

    def confidence_interval(self, level=0.95):
        se = self.extras_.get("std_error")
        if se is None or not np.isfinite(se):
            raise NotImplementedError("need n > 3 for a Fisher-z confidence interval")
        from stochpylib.nonparametric._common import _norm_ppf
        z = math.atanh(self.estimate_)
        crit = float(_norm_ppf(1 - (1 - level) / 2))
        lo, hi = math.tanh(z - crit * se), math.tanh(z + crit * se)
        return (lo, hi)


class KendallTau(DependenceMeasure):
    """Kendall's tau: tau-b via
    :func:`stochpylib.copulas._utils.kendall_tau_estimate` (default), or
    tau-a/tau-c. ``method="exact"`` uses the exact null permutation
    distribution for small untied samples (matches
    ``scipy.stats.kendalltau(method="exact")``); ``"asymptotic"`` (default
    "auto") uses the tie-corrected normal approximation, matching
    ``scipy.stats.kendalltau(method="asymptotic")`` exactly.
    """

    def __init__(self, variant="b", method="auto"):
        self.variant = variant
        self.method = method

    def _fit(self, x, y):
        n = len(x)
        tau_b = kendall_tau_estimate(x, y)

        # nc - nd (excluding ties) recovered from tau-b's own denominator --
        # avoids re-deriving the O(n log n) inversion count: tau_b = (nc-nd)/
        # sqrt((N-tx)(N-ty)), so (nc-nd) = tau_b * sqrt((N-tx)(N-ty)).
        con_minus_dis = self._con_minus_dis(x, y, tau_b)

        if self.variant == "a":
            N = n * (n - 1) / 2.0
            tau = con_minus_dis / N
        elif self.variant == "c":
            m = min(len(np.unique(x)), len(np.unique(y)))
            tau = 2.0 * con_minus_dis / (n * n * (m - 1) / m) if m > 1 else float("nan")
        else:
            tau = tau_b

        n0 = n * (n - 1) / 2.0
        no_ties = len(np.unique(x)) == n and len(np.unique(y)) == n
        method = self.method
        if method == "auto":
            method = "exact" if (no_ties and n <= 33) else "asymptotic"

        if method == "exact" and no_ties:
            pvalue = self._exact_pvalue(n, con_minus_dis)
        else:
            vx = self._tie_groups(x)
            vy = self._tie_groups(y)
            v0 = n * (n - 1) * (2 * n + 5)
            vt = np.sum(vx * (vx - 1) * (2 * vx + 5))
            vu = np.sum(vy * (vy - 1) * (2 * vy + 5))
            v1 = np.sum(vx * (vx - 1)) * np.sum(vy * (vy - 1)) / (2.0 * n * (n - 1)) \
                if n > 1 else 0.0
            v2 = np.sum(vx * (vx - 1) * (vx - 2)) * np.sum(vy * (vy - 1) * (vy - 2)) / \
                (9.0 * n * (n - 1) * (n - 2)) if n > 2 else 0.0
            var_s = (v0 - vt - vu) / 18.0 + v1 + v2
            z = con_minus_dis / math.sqrt(var_s) if var_s > 0 else 0.0
            pvalue = float(2.0 * _norm_cdf(-abs(z)))
        pvalue = float(np.clip(pvalue, 0.0, 1.0))
        se = math.sqrt((2.0 * (2 * n + 5)) / (9.0 * n * (n - 1))) if n > 1 else float("nan")
        return tau, pvalue, {"std_error": se}

    @staticmethod
    def _tie_groups(a):
        _, counts = np.unique(a, return_counts=True)
        return counts.astype(float)

    @classmethod
    def _con_minus_dis(cls, x, y, tau_b=None):
        """``nc - nd`` (ties excluded from both), recovered from the already
        tie-corrected ``tau_b`` -- avoids re-deriving the O(n log n) inversion
        count that :func:`kendall_tau_estimate` already computes internally."""
        n = len(x)
        if tau_b is None:
            tau_b = kendall_tau_estimate(x, y)
        N = n * (n - 1) / 2.0
        tx = np.sum(cls._tie_groups(x) * (cls._tie_groups(x) - 1) / 2.0)
        ty = np.sum(cls._tie_groups(y) * (cls._tie_groups(y) - 1) / 2.0)
        denom = math.sqrt(max(N - tx, 0.0) * max(N - ty, 0.0))
        return tau_b * denom

    @staticmethod
    def _exact_pvalue(n, con_minus_dis):
        """Exact two-sided p-value from the null distribution of the number
        of discordant pairs among ``n!`` permutations (the classical Kendall
        S-statistic recursion: the generating function ``prod_{m=2}^n (1 + q
        + ... + q^(m-1))`` gives ``counts[k]`` = #permutations with exactly
        ``k`` discordant pairs)."""
        counts = np.array([1.0])
        for m in range(2, n + 1):
            new_counts = np.zeros(len(counts) + m - 1)
            for j in range(m):
                new_counts[j:j + len(counts)] += counts
            counts = new_counts
        total = counts.sum()
        s_obs = abs(con_minus_dis)
        k_grid = np.arange(len(counts))
        s_grid = (n * (n - 1) // 2) - 2 * k_grid          # S = nc - nd for k discordant pairs
        p = float(np.sum(counts[np.abs(s_grid) >= s_obs - 1e-9]) / total)
        return p


class RankCorrelation(DependenceMeasure):
    """Dispatcher over ``method="spearman"|"kendall"|"gamma"|"somers_d"``.
    ``"gamma"`` is Goodman-Kruskal's gamma; ``"somers_d"`` is Somers' D
    (asymmetric, D(y|x)), matching ``scipy.stats.somersd`` exactly.
    """

    def __init__(self, method="spearman"):
        self.method = method

    def _fit(self, x, y):
        if self.method == "spearman":
            m = SpearmanCorrelation().fit(x, y)
            return m.estimate_, m.pvalue_, m.extras_
        if self.method == "kendall":
            m = KendallTau().fit(x, y)
            return m.estimate_, m.pvalue_, m.extras_
        con_minus_dis = KendallTau._con_minus_dis(x, y)
        if self.method == "gamma":
            n = len(x)
            # pairs tied on x or y contribute to neither nc nor nd
            all_pairs = n * (n - 1) / 2.0
            tx = np.sum(KendallTau._tie_groups(x) * (KendallTau._tie_groups(x) - 1) / 2.0)
            ty = np.sum(KendallTau._tie_groups(y) * (KendallTau._tie_groups(y) - 1) / 2.0)
            nc_plus_nd = all_pairs - tx - ty
            nc = (con_minus_dis + nc_plus_nd) / 2.0
            nd = nc_plus_nd - nc
            gamma = (nc - nd) / (nc + nd) if (nc + nd) > 0 else float("nan")
            return float(gamma), float("nan"), {}
        if self.method == "somers_d":
            # D(Y|X) = tau_a(X,Y) / tau_a(X,X): the denominator excludes ties
            # in X, the independent/row variable (matches scipy.stats.somersd(x, y)).
            n = len(x)
            tx = np.sum(KendallTau._tie_groups(x) * (KendallTau._tie_groups(x) - 1) / 2.0)
            all_pairs = n * (n - 1) / 2.0
            d = con_minus_dis / (all_pairs - tx) if (all_pairs - tx) > 0 else float("nan")
            return float(d), float("nan"), {}
        raise ValueError("method must be 'spearman', 'kendall', 'gamma', or 'somers_d'")


class DistanceCorrelation(DependenceMeasure):
    """Szekely-Rizzo distance correlation/covariance/variance: dCor in [0, 1],
    zero iff independent (in the population). ``unbiased=True`` uses the
    U-centered bias-corrected estimator; the p-value is from a permutation
    test with ``n_resamples`` shuffles of ``y``.
    """

    def __init__(self, unbiased=False, n_resamples=999, random_state=None):
        self.unbiased = unbiased
        self.n_resamples = n_resamples
        self.random_state = random_state

    def _dcov2(self, x, y):
        A = _u_center(_pairwise_dist(x)) if self.unbiased else _double_center(_pairwise_dist(x))
        B = _u_center(_pairwise_dist(y)) if self.unbiased else _double_center(_pairwise_dist(y))
        n = len(x)
        if self.unbiased:
            return float(np.sum(A * B) / (n * (n - 3)))
        return float(np.mean(A * B))

    def _fit(self, x, y):
        n = len(x)
        dcov2 = self._dcov2(x, y)
        dvarx2 = self._dcov2(x, x)
        dvary2 = self._dcov2(y, y)
        denom = math.sqrt(max(dvarx2, 0.0) * max(dvary2, 0.0))
        dcor = math.sqrt(max(dcov2, 0.0) / denom) if denom > 0 else 0.0
        rng = np.random.default_rng(self.random_state)
        obs = dcov2
        null = np.empty(self.n_resamples)
        for b in range(self.n_resamples):
            perm = rng.permutation(n)
            null[b] = self._dcov2(x, y[perm])
        pvalue = float((np.sum(null >= obs) + 1) / (self.n_resamples + 1))
        return float(dcor), pvalue, {"dcov2": dcov2, "dvarx2": dvarx2, "dvary2": dvary2}


class BrownianCorrelation(DependenceMeasure):
    """Szekely-Rizzo Brownian distance correlation generalized to fractional
    Brownian motion of Hurst exponent ``hurst``: an alpha-distance covariance
    with ``alpha = 2 * hurst`` (``hurst=0.5`` reduces exactly to
    :class:`DistanceCorrelation`).
    """

    def __init__(self, hurst=0.5, n_resamples=999, random_state=None):
        self.hurst = hurst
        self.n_resamples = n_resamples
        self.random_state = random_state

    def _dist(self, x):
        alpha = 2.0 * self.hurst
        d = _pairwise_dist(x)
        return d ** alpha

    def _dcov2(self, x, y):
        A = _double_center(self._dist(x))
        B = _double_center(self._dist(y))
        return float(np.mean(A * B))

    def _fit(self, x, y):
        n = len(x)
        dcov2 = self._dcov2(x, y)
        dvarx2 = self._dcov2(x, x)
        dvary2 = self._dcov2(y, y)
        denom = math.sqrt(max(dvarx2, 0.0) * max(dvary2, 0.0))
        dcor = math.sqrt(max(dcov2, 0.0) / denom) if denom > 0 else 0.0
        rng = np.random.default_rng(self.random_state)
        null = np.empty(self.n_resamples)
        for b in range(self.n_resamples):
            perm = rng.permutation(n)
            null[b] = self._dcov2(x, y[perm])
        pvalue = float((np.sum(null >= dcov2) + 1) / (self.n_resamples + 1))
        return float(dcor), pvalue, {"dcov2": dcov2}


class HoeffdingD(DependenceMeasure):
    """Hoeffding's D dependence measure (Blum-Kiefer-Rosenblatt 1961 formula,
    R ``Hmisc::hoeffd`` scaling: D in roughly [-0.5, 1], 0 under independence,
    1 for a perfectly monotone relation). P-value from a permutation test.
    """

    def __init__(self, n_resamples=999, random_state=None):
        self.n_resamples = n_resamples
        self.random_state = random_state

    @staticmethod
    def _stat(x, y):
        """Hoeffding's D (Hollander & Wolfe eq. 5.34-5.37, matching R's
        Hmisc::hoeffd): bivariate rank ``Q_i`` with tie corrections, then the
        D1/D2/D3 combination."""
        n = len(x)
        if n < 5:
            return float("nan")
        R = _rank_ties(x)
        S = _rank_ties(y)
        Q = np.empty(n)
        for i in range(n):
            less_x = x < x[i]
            less_y = y < y[i]
            eq_x = x == x[i]
            eq_y = y == y[i]
            both_less = np.sum(less_x & less_y)
            eq_both_others = np.sum(eq_x & eq_y) - 1.0
            x_eq_y_less = np.sum(eq_x & less_y)
            x_less_y_eq = np.sum(less_x & eq_y)
            Q[i] = 1 + both_less + 0.25 * eq_both_others + 0.5 * x_eq_y_less + 0.5 * x_less_y_eq
        D1 = np.sum((Q - 1) * (Q - 2))
        D2 = np.sum((R - 1) * (R - 2) * (S - 1) * (S - 2))
        D3 = np.sum((R - 2) * (S - 2) * (Q - 1))
        D = 30 * ((n - 2) * (n - 3) * D1 + D2 - 2 * (n - 2) * D3) / \
            (n * (n - 1) * (n - 2) * (n - 3) * (n - 4))
        return float(D)

    def _fit(self, x, y):
        n = len(x)
        obs = self._stat(x, y)
        rng = np.random.default_rng(self.random_state)
        null = np.empty(self.n_resamples)
        for b in range(self.n_resamples):
            perm = rng.permutation(n)
            null[b] = self._stat(x, y[perm])
        pvalue = float((np.sum(null >= obs) + 1) / (self.n_resamples + 1))
        return obs, pvalue, {}
