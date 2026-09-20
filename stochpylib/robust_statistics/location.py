"""Robust location (central-tendency) estimators.

``TrimmedMean``/``WinsorizedMean`` down-weight tail observations by a fixed
proportion; ``Median``/``HodgesLehmann`` are exact order-statistic/rank
estimators with distribution-free confidence intervals; ``L_Estimator``
generalizes to arbitrary order-statistic weights; ``M_Estimator``/
``R_Estimator`` are the M- and R- analogues of the classic robust-regression
psi-functions, specialized to the one-sample (and, for R, two-sample) case.
"""

import math

import numpy as np
from scipy import optimize, special

from stochpylib.robust_statistics._base import RobustEstimator
from stochpylib.robust_statistics._common import (
    _as_1d, _kth_cross_diff, _kth_pairwise, _mad, _norm_ppf, _PSI, _psi_prime_of,
    _psi_value, _psi_weight, _rng,
)

__all__ = [
    "TrimmedMean", "WinsorizedMean", "Median", "HodgesLehmann",
    "L_Estimator", "M_Estimator", "R_Estimator",
]


def _trim_winsorize(x, proportion):
    x = np.sort(x)
    n = len(x)
    g = int(math.floor(proportion * n))
    return x, g


class TrimmedMean(RobustEstimator):
    """Mean after removing ``proportion`` of the sample from each tail.

    Matches ``scipy.stats.trim_mean`` exactly. Standard error via the
    Tukey-McLaughlin formula; confidence interval is a Student-t interval
    with ``df = n - 2g - 1``.
    """

    method = "trimmed mean"

    def __init__(self, proportion=0.1):
        if not (0.0 <= proportion < 0.5):
            raise ValueError("proportion must be in [0, 0.5)")
        self.proportion = float(proportion)

    def _fit(self, x):
        xs, g = _trim_winsorize(x, self.proportion)
        n = len(xs)
        core = xs[g:n - g] if g > 0 else xs
        estimate = float(np.mean(core))
        if g > 0:
            win = np.concatenate([np.full(g, xs[g]), core, np.full(g, xs[n - g - 1])])
        else:
            win = core
        s_w = float(np.std(win, ddof=1)) if n > 1 else 0.0
        df = n - 2 * g - 1
        se = s_w / ((1.0 - 2.0 * self.proportion) * math.sqrt(n)) if n > 0 else float("nan")
        return estimate, se, {"df": df, "n_trimmed": g}


class WinsorizedMean(RobustEstimator):
    """Mean of the sample after winsorizing ``proportion`` from each tail
    (tail values replaced by the nearest retained order statistic)."""

    method = "winsorized mean"

    def __init__(self, proportion=0.1):
        if not (0.0 <= proportion < 0.5):
            raise ValueError("proportion must be in [0, 0.5)")
        self.proportion = float(proportion)

    def _fit(self, x):
        xs, g = _trim_winsorize(x, self.proportion)
        n = len(xs)
        if g > 0:
            win = np.concatenate([np.full(g, xs[g]), xs[g:n - g], np.full(g, xs[n - g - 1])])
        else:
            win = xs
        estimate = float(np.mean(win))
        s_w = float(np.std(win, ddof=1)) if n > 1 else 0.0
        df = n - 2 * g - 1
        se = s_w / ((1.0 - 2.0 * self.proportion) * math.sqrt(n)) if n > 0 else float("nan")
        return estimate, se, {"df": df, "n_winsorized": g}


class Median(RobustEstimator):
    """Sample median, with a Maritz-Jarrett or bootstrap standard error and a
    distribution-free order-statistic confidence interval (falling back to a
    normal approximation for ``n < 6``)."""

    method = "median"

    def __init__(self, se_method="maritz-jarrett", n_boot=1000, random_state=None):
        if se_method not in ("maritz-jarrett", "bootstrap"):
            raise ValueError("se_method must be 'maritz-jarrett' or 'bootstrap'")
        self.se_method = se_method
        self.n_boot = int(n_boot)
        self.random_state = random_state

    def _fit(self, x):
        n = len(x)
        estimate = float(np.median(x))
        if n == 1:
            return estimate, float("nan"), {}
        xs = np.sort(x)
        if self.se_method == "maritz-jarrett":
            se = self._mj_se(xs)
        else:
            se = self._boot_se(x)
        ci = self._order_stat_ci(xs, level=0.95) if n >= 6 else None
        extras = {"method": "order-statistic" if ci is not None else "normal-approx"}
        if ci is not None:
            extras["confidence_interval"] = ci
        return estimate, se, extras

    @staticmethod
    def _mj_se(xs):
        n = len(xs)
        m = (n + 1) // 2
        i = np.arange(1, n + 1)
        W = special.betainc(m, n - m + 1, i / n) - special.betainc(m, n - m + 1, (i - 1) / n)
        c1 = float(np.sum(W * xs))
        c2 = float(np.sum(W * xs ** 2))
        return math.sqrt(max(c2 - c1 * c1, 0.0))

    def _boot_se(self, x):
        rng = _rng(self.random_state)
        n = len(x)
        idx = rng.integers(0, n, size=(self.n_boot, n))
        boot = np.median(x[idx], axis=1)
        return float(np.std(boot, ddof=1))

    def confidence_interval(self, level=0.95):
        if not hasattr(self, "estimate_"):
            raise RuntimeError("Median is not fitted")
        if level == 0.95 and "confidence_interval" in self.extras_:
            return self.extras_["confidence_interval"]
        if self.n_ >= 6:
            xs = np.sort(np.asarray(self._x, dtype=float))
            return self._order_stat_ci(xs, level)
        return super().confidence_interval(level)

    def fit(self, x):
        self._x = np.asarray(x, dtype=float).ravel()
        return super().fit(x)

    @staticmethod
    def _order_stat_ci(xs, level):
        n = len(xs)
        alpha2 = (1.0 - level) / 2.0
        k = 1
        for cand in range(1, n // 2 + 1):
            if float(special.bdtr(cand - 1, n, 0.5)) <= alpha2:
                k = cand
            else:
                break
        lo = float(xs[k - 1])
        hi = float(xs[n - k])
        return (lo, hi)


class HodgesLehmann(RobustEstimator):
    """One- or two-sample Hodges-Lehmann location estimator: the median of
    Walsh averages (one-sample) or of all pairwise differences (two-sample),
    with the Hollander-Wolfe signed-rank / two-sample confidence interval."""

    method = "Hodges-Lehmann"

    def __init__(self):
        pass

    def fit(self, x, y=None):
        x = _as_1d(x)
        if y is None:
            self.n_ = len(x)
            self.estimate_, self.std_error_, self.extras_ = self._fit_one(x)
        else:
            y = _as_1d(y, "y")
            self.n_ = len(x) + len(y)
            self.estimate_, self.std_error_, self.extras_ = self._fit_two(x, y)
        return self

    def _fit(self, x):
        return self._fit_one(x)

    @staticmethod
    def _fit_one(x, level=0.95):
        n = len(x)
        M = n * (n + 1) // 2
        if M % 2 == 1:
            estimate = _kth_pairwise(x, (M + 1) // 2, kind="walsh")
        else:
            lo_med = _kth_pairwise(x, M // 2, kind="walsh")
            hi_med = _kth_pairwise(x, M // 2 + 1, kind="walsh")
            estimate = 0.5 * (lo_med + hi_med)
        z = float(-_norm_ppf((1.0 - level) / 2.0))
        C = int(math.floor(n * (n + 1) / 4.0 - z * math.sqrt(n * (n + 1) * (2 * n + 1) / 24.0))) + 1
        C = int(np.clip(C, 1, max(M // 2, 1)))
        lo = _kth_pairwise(x, C, kind="walsh")
        hi = _kth_pairwise(x, M + 1 - C, kind="walsh")
        se = (hi - lo) / (2.0 * z) if z > 0 else float("nan")
        return float(estimate), float(se), {"confidence_interval": (float(lo), float(hi)), "n_walsh": M}

    @staticmethod
    def _fit_two(x, y, level=0.95):
        m, n = len(x), len(y)
        mn = m * n
        if mn % 2 == 1:
            estimate = _kth_cross_diff(x, y, (mn + 1) // 2)
        else:
            lo_med = _kth_cross_diff(x, y, mn // 2)
            hi_med = _kth_cross_diff(x, y, mn // 2 + 1)
            estimate = 0.5 * (lo_med + hi_med)
        z = float(-_norm_ppf((1.0 - level) / 2.0))
        C = int(math.floor(mn / 2.0 - z * math.sqrt(mn * (m + n + 1) / 12.0))) + 1
        C = int(np.clip(C, 1, max(mn // 2, 1)))
        lo = _kth_cross_diff(x, y, C)
        hi = _kth_cross_diff(x, y, mn + 1 - C)
        se = (hi - lo) / (2.0 * z) if z > 0 else float("nan")
        return float(estimate), float(se), {"confidence_interval": (float(lo), float(hi)), "n_pairs": mn}


_L_QUANTILE_WEIGHTS = {
    "trimean": [(0.25, 0.25), (0.5, 0.5), (0.25, 0.75)],
    "midhinge": [(0.5, 0.25), (0.5, 0.75)],
    "gastwirth": [(0.3, 1.0 / 3.0), (0.4, 0.5), (0.3, 2.0 / 3.0)],
}


def _quantile_at(xs, p):
    """Linear-interpolation quantile of a sorted array (1-based positions)."""
    n = len(xs)
    h = (n - 1) * p
    lo = int(math.floor(h))
    hi = min(lo + 1, n - 1)
    frac = h - lo
    return xs[lo] * (1 - frac) + xs[hi] * frac


class L_Estimator(RobustEstimator):
    """Linear combination of order statistics.

    ``weights``: ``"trimean"`` (0.25 Q1 + 0.5 median + 0.25 Q3, the default),
    ``"gastwirth"``, ``"midhinge"``, ``"trimmed"``/``"winsorized"`` (with
    ``proportion=``), a callable ``J(u)`` on (0, 1) sampled at
    ``(i - 0.5)/n`` and normalized to sum to 1, or an explicit length-``n``
    array of weights on the sorted sample.
    """

    method = "L-estimator"

    def __init__(self, weights="trimean", proportion=0.1, n_boot=500, random_state=None):
        self.weights = weights
        self.proportion = float(proportion)
        self.n_boot = int(n_boot)
        self.random_state = random_state

    def _statistic(self, x):
        xs = np.sort(np.asarray(x, dtype=float))
        n = len(xs)
        w = self.weights
        if isinstance(w, str) and w in ("trimmed", "winsorized"):
            g = int(math.floor(self.proportion * n))
            if w == "trimmed":
                core = xs[g:n - g] if g > 0 else xs
                return float(np.mean(core))
            win = np.concatenate([np.full(g, xs[g]), xs[g:n - g], np.full(g, xs[n - g - 1])]) if g > 0 else xs
            return float(np.mean(win))
        if isinstance(w, str) and w in _L_QUANTILE_WEIGHTS:
            terms = _L_QUANTILE_WEIGHTS[w]
            return float(sum(coef * _quantile_at(xs, p) for coef, p in terms))
        if callable(w):
            i = np.arange(1, n + 1)
            weights = w((i - 0.5) / n)
            weights = np.asarray(weights, dtype=float)
            weights = weights / weights.sum()
            return float(np.sum(weights * xs))
        arr = np.asarray(w, dtype=float)
        if arr.shape[0] != n:
            raise ValueError("explicit weights array must have length n")
        arr = arr / arr.sum()
        return float(np.sum(arr * xs))

    def _fit(self, x):
        estimate = self._statistic(x)
        rng = _rng(self.random_state)
        n = len(x)
        idx = rng.integers(0, n, size=(self.n_boot, n))
        boot = np.array([self._statistic(x[row]) for row in idx])
        se = float(np.std(boot, ddof=1))
        return estimate, se, {"bootstrap_distribution": boot}


_M_LOCATION_SCALES = {"mad", "huber"}


class M_Estimator(RobustEstimator):
    """Huber-type M-estimator of location via IRLS.

    ``scale``: ``"mad"`` (fixed at ``mad(x - median(x), center=0)``, the
    statsmodels RLM convention), ``"huber"`` (Huber's proposal 2, jointly
    estimated), or a fixed float. Standard error uses the H1 sandwich formula
    (matches ``statsmodels.RLM`` with ``cov="H1"`` for an intercept-only
    design).
    """

    method = "M-estimator"

    def __init__(self, psi="huber", c=None, scale="mad", tol=1e-8, max_iter=100):
        if psi not in _PSI:
            raise ValueError(f"psi must be one of {sorted(_PSI)}")
        self.psi = psi
        self.scale = scale
        if c is None:
            c = 1.5 if (psi == "huber" and scale == "huber") else _PSI[psi]["c"]
        self.c = c
        self.tol = float(tol)
        self.max_iter = int(max_iter)

    def _fit(self, x):
        n = len(x)
        from stochpylib.robust_statistics._common import _huber_proposal2

        update_scale = self.scale == "mad"
        if self.scale == "huber":
            _, s = _huber_proposal2(x)
            theta = float(np.median(x))
        elif self.scale == "mad":
            theta = float(np.mean(x))          # matches statsmodels RLM's OLS start
            s = _mad(x - theta, center=0.0)
        else:
            s = float(self.scale)
            theta = float(np.median(x))
        s = s if s > 0 else 1.0

        n_iter = 0
        for _ in range(self.max_iter):
            n_iter += 1
            u = (x - theta) / s
            w = _psi_weight(self.psi, u, self.c)
            sw = np.sum(w)
            theta_new = float(np.sum(w * x) / sw) if sw > 0 else theta
            if update_scale:
                s_new = _mad(x - theta_new, center=0.0)
                s = s_new if s_new > 0 else s
            if abs(theta_new - theta) < self.tol * max(1.0, abs(theta)):
                theta = theta_new
                break
            theta = theta_new

        u = (x - theta) / s
        psi_vals = _psi_value(self.psi, u, self.c)
        psi_prime_vals = _psi_prime_of(self.psi, u, self.c)
        m = float(np.mean(psi_prime_vals))
        var_psi_prime = float(np.var(psi_prime_vals, ddof=0))
        k = 1.0 + (1.0 / n) * var_psi_prime / m ** 2 if m != 0 else float("nan")
        ss_psi = float(np.sum(psi_vals ** 2))
        se = (k * s * math.sqrt(ss_psi / (n - 1)) / (m * math.sqrt(n))
              if (m != 0 and n > 1) else float("nan"))
        weights_final = _psi_weight(self.psi, u, self.c)
        return float(theta), float(se), {"scale_": float(s), "weights_": weights_final, "n_iter_": n_iter}


class R_Estimator(RobustEstimator):
    """Rank-based (R-) location estimator: root of the signed-rank (or
    two-sample rank) score equation.

    ``score``: ``"wilcoxon"`` (linear score -- equals ``HodgesLehmann``
    exactly), ``"sign"`` (equals the median), ``"normal"`` (van der Waerden
    normal scores).
    """

    method = "R-estimator"

    def __init__(self, score="wilcoxon"):
        if score not in ("wilcoxon", "sign", "normal"):
            raise ValueError("score must be 'wilcoxon', 'sign', or 'normal'")
        self.score = score

    def fit(self, x, y=None):
        x = _as_1d(x)
        if y is None:
            self.n_ = len(x)
            self.estimate_, self.std_error_, self.extras_ = self._fit_one(x)
        else:
            y = _as_1d(y, "y")
            self.n_ = len(x) + len(y)
            self.estimate_, self.std_error_, self.extras_ = self._fit_two(x, y)
        return self

    def _fit(self, x):
        return self._fit_one(x)

    def _point_one(self, x):
        """The point estimate only (used directly and inside the bootstrap)."""
        if self.score == "wilcoxon":
            return HodgesLehmann._fit_one(x)[0]
        if self.score == "sign":
            return float(np.median(x))
        n = len(x)
        a_all = np.array([_norm_ppf(0.5 + 0.5 * k / (n + 1.0)) for k in range(1, n + 1)])

        def S(theta):
            d = x - theta
            r = np.argsort(np.argsort(np.abs(d)))
            return float(np.sum(np.sign(d) * a_all[r]))

        lo, hi = float(np.min(x)), float(np.max(x))
        s_lo, s_hi = S(lo + 1e-9 * max(1.0, abs(lo))), S(hi - 1e-9 * max(1.0, abs(hi)))
        if not (s_lo >= 0.0 >= s_hi):
            return float(np.median(x))
        return float(optimize.brentq(S, lo, hi, xtol=1e-10, maxiter=200))

    def _fit_one(self, x, n_boot=500, random_state=0):
        if self.score == "wilcoxon":
            return HodgesLehmann._fit_one(x)
        est = self._point_one(x)
        rng = _rng(random_state)
        n = len(x)
        idx = rng.integers(0, n, size=(n_boot, n))
        boot = np.array([self._point_one(x[row]) for row in idx])
        se = float(np.std(boot, ddof=1))
        return est, se, {}

    def _point_two(self, x, y):
        if self.score == "wilcoxon":
            return HodgesLehmann._fit_two(x, y)[0]
        if self.score == "sign":
            return float(np.median(x) - np.median(y))
        m, n = len(x), len(y)
        N = m + n
        a_all = np.array([_norm_ppf(0.5 + 0.5 * k / (N + 1.0)) for k in range(1, N + 1)])
        target = m * float(np.mean(a_all))

        def T(theta):
            pooled = np.concatenate([x, y + theta])
            ranks = np.argsort(np.argsort(pooled))
            rx = ranks[:m]
            return float(np.sum(a_all[rx])) - target

        lo, hi = float(np.min(x) - np.max(y)), float(np.max(x) - np.min(y))
        t_lo, t_hi = T(lo + 1e-9), T(hi - 1e-9)
        if t_lo * t_hi > 0:
            return float(np.median(x) - np.median(y))
        return float(optimize.brentq(T, lo, hi, xtol=1e-10, maxiter=200))

    def _fit_two(self, x, y, n_boot=300, random_state=0):
        if self.score == "wilcoxon":
            return HodgesLehmann._fit_two(x, y)
        est = self._point_two(x, y)
        rng = _rng(random_state)
        idxx = rng.integers(0, len(x), size=(n_boot, len(x)))
        idxy = rng.integers(0, len(y), size=(n_boot, len(y)))
        boot = np.array([self._point_two(x[idxx[i]], y[idxy[i]]) for i in range(n_boot)])
        se = float(np.std(boot, ddof=1))
        return float(est), se, {}
