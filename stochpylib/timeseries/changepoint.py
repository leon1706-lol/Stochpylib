"""Change-point detection for piecewise-constant mean signals.

Documented scope: all detectors model segments by their mean (Gaussian squared-error
cost). The default penalty/threshold is a BIC-flavored heuristic —
``2 * var(y) * log(T)`` — and can be overridden explicitly. ``BayesianChangePoint``
implements Adams & MacKay (2007) online Bayesian changepoint detection with a
Normal-Inverse-Gamma conjugate model (Student-t predictive), with the run-length
support truncated at ``max_run`` for memory.
"""

from dataclasses import dataclass, field

import numpy as np
from scipy import stats
from scipy.special import logsumexp

from stochpylib.timeseries._utils import as_1d

__all__ = [
    "ChangePointResult",
    "ChangePointDetection",
    "BinarySegmentation",
    "BottomUp",
    "PELT",
    "BayesianChangePoint",
]


@dataclass
class ChangePointResult:
    """Detected change points: 0-based indices where a **new segment starts**."""

    points: list
    method: str
    extra: dict = field(default_factory=dict)

    def __repr__(self):
        return f"ChangePointResult(method={self.method!r}, points={self.points})"


# ---------------------------------------------------------------------------
# shared cost machinery


def _prefix_sums(y):
    S = np.concatenate([[0.0], np.cumsum(y)])
    SS = np.concatenate([[0.0], np.cumsum(y**2)])
    return S, SS


def _sse(S, SS, a, b):
    """Sum of squared deviations of y[a:b] from its own mean."""
    n = b - a
    if n <= 0:
        return 0.0
    s = S[b] - S[a]
    ss = SS[b] - SS[a]
    return max(ss - s * s / n, 0.0)


def _default_penalty(y):
    return float(2.0 * np.var(y) * np.log(max(len(y), 3)))


# ---------------------------------------------------------------------------
# classical detectors


def BinarySegmentation(y, threshold=None, min_segment=5):
    """Recursive binary segmentation: split where the SSE reduction is maximal."""
    y = as_1d(y)
    T = len(y)
    threshold = _default_penalty(y) if threshold is None else float(threshold)
    S, SS = _prefix_sums(y)

    points = []

    def recurse(a, b):
        if b - a < 2 * min_segment:
            return
        full = _sse(S, SS, a, b)
        best_gain, best_m = 0.0, None
        for m in range(a + min_segment, b - min_segment + 1):
            gain = full - _sse(S, SS, a, m) - _sse(S, SS, m, b)
            if gain > best_gain:
                best_gain, best_m = gain, m
        if best_m is not None and best_gain > threshold:
            points.append(best_m)
            recurse(a, best_m)
            recurse(best_m, b)

    recurse(0, T)
    points.sort()
    return ChangePointResult(points=points, method="binary_segmentation")


def BottomUp(y, threshold=None, initial_segment=10):
    """Bottom-up segmentation: start fine, greedily merge the cheapest adjacent pair."""
    y = as_1d(y)
    T = len(y)
    threshold = _default_penalty(y) if threshold is None else float(threshold)
    initial_segment = max(int(initial_segment), 2)
    S, SS = _prefix_sums(y)

    # initial boundaries every `initial_segment` points (excluding 0 and T)
    bounds = list(range(initial_segment, T - initial_segment // 2, initial_segment))
    seg_starts = [0] + bounds + [T]

    def total_cost():
        return sum(
            _sse(S, SS, seg_starts[i], seg_starts[i + 1]) for i in range(len(seg_starts) - 1)
        )

    while len(seg_starts) > 2:
        best_cost, best_i = None, None
        for i in range(len(seg_starts) - 2):
            merged = (
                _sse(S, SS, seg_starts[i], seg_starts[i + 2])
                - _sse(S, SS, seg_starts[i], seg_starts[i + 1])
                - _sse(S, SS, seg_starts[i + 1], seg_starts[i + 2])
            )
            if best_cost is None or merged < best_cost:
                best_cost, best_i = merged, i
        if best_cost is None or best_cost > threshold:
            break
        del seg_starts[best_i + 1]

    return ChangePointResult(points=sorted(b for b in seg_starts if 0 < b < T),
                             method="bottom_up")


def PELT(y, penalty=None):
    """Pruned Exact Linear Time changepoint search (Killick et al., 2012).

    Exact under a linear penalty; candidates are pruned with K = 0, which is valid for
    the superadditive Gaussian mean cost used here.
    """
    y = as_1d(y)
    T = len(y)
    beta = _default_penalty(y) if penalty is None else float(penalty)
    S, SS = _prefix_sums(y)

    F = np.full(T + 1, np.inf)
    F[0] = -beta
    backptr = np.zeros(T + 1, dtype=int)
    R = [0]
    for t in range(1, T + 1):
        costs = [F[s] + _sse(S, SS, s, t) + beta for s in R]
        best_i = int(np.argmin(costs))
        F[t] = costs[best_i]
        backptr[t] = R[best_i]
        R = [s for s in R if F[s] < F[t]]
        R.append(t)

    points = []
    t = T
    while t > 0:
        prev = int(backptr[t])
        if prev > 0:
            points.append(prev)
        t = prev
    points.sort()
    return ChangePointResult(points=points, method="pelt", extra={"penalty": beta})


# ---------------------------------------------------------------------------
# Bayesian (Adams & MacKay 2007)


@dataclass
class BOCPDResult:
    probability_of_change: np.ndarray   # (T,) posterior prob of a change at each t
    points: list                        # indices where that posterior exceeds `threshold`
    method: str = "bocpd"
    extra: dict = field(default_factory=dict)


class BayesianChangePoint:
    """Online Bayesian changepoint detection for piecewise-constant Gaussian data.

    Normal-Inverse-Gamma conjugate updates per run length give a Student-t predictive;
    the run-length support is truncated at ``max_run`` (documented approximation).
    """

    def __init__(self, hazard_rate=None, mu0=None, kappa0=1.0, alpha0=2.0,
                 beta0=None, threshold=0.4, max_run=400):
        self.hazard_rate = hazard_rate
        self.mu0 = mu0
        self.kappa0 = float(kappa0)
        self.alpha0 = float(alpha0)
        self.beta0 = None if beta0 is None else float(beta0)
        self.threshold = float(threshold)
        self.max_run = int(max_run)

    @staticmethod
    def _student_logpdf(x, df, loc, scale):
        scale = max(scale, 1e-300)
        return stats.t.logpdf(x, df, loc=loc, scale=scale)

    def fit(self, y):
        y = as_1d(y)
        T = len(y)
        mu0 = float(np.mean(y)) if self.mu0 is None else self.mu0
        kappa0, alpha0 = self.kappa0, self.alpha0
        beta0 = float(np.var(y)) / 2.0 if self.beta0 is None else self.beta0
        hazard = self.hazard_rate if self.hazard_rate is not None else min(1.0 / 100.0, 5.0 / T)
        max_run = min(self.max_run, T)

        # Normal-Inverse-Gamma posterior sufficient stats per run length:
        # (n, sample_mean, centered sum of squares) — Welford-style updates
        stats_by_run = {0: (0, mu0, 0.0)}

        def student_pred(r, x):
            n, xbar, M2 = stats_by_run[r]
            kappa_n = kappa0 + n
            alpha_n = alpha0 + n / 2.0
            mu_post = (kappa0 * mu0 + n * xbar) / kappa_n
            beta_n = beta0 + 0.5 * M2 + kappa0 * n * (xbar - mu0) ** 2 / (2.0 * kappa_n)
            scale2 = beta_n * (kappa_n + 1.0) / (alpha_n * kappa_n)
            return stats.t.logpdf(x, df=2 * alpha_n, loc=mu_post, scale=np.sqrt(scale2))

        def add_observation(r, x):
            n, xbar, M2 = stats_by_run[r]
            n1 = n + 1
            delta = x - xbar
            return (n1, xbar + delta / n1, M2 + delta * (x - xbar))

        def student_prior_pred(x):
            """Predictive of the RESET hypothesis: NIG prior, empty run."""
            scale2 = beta0 * (kappa0 + 1.0) / (alpha0 * kappa0)
            return stats.t.logpdf(x, df=2 * alpha0, loc=mu0, scale=np.sqrt(scale2))

        log_R = np.full(max_run + 1, -np.inf)
        log_R[0] = 0.0
        prob_change = np.empty(T)

        for t in range(T):
            x = y[t]
            active = [r for r in range(max_run + 1) if log_R[r] > -np.inf]

            new_log_R = np.full(max_run + 1, -np.inf)
            # changepoint hypothesis: reset to the prior, then observe x
            growth = np.log(hazard) + student_prior_pred(x) + logsumexp(log_R)
            for r in active:
                lp = student_pred(r, x)
                if r + 1 <= max_run:
                    stay = np.log(1.0 - hazard) + lp + log_R[r]
                    new_log_R[r + 1] = np.logaddexp(new_log_R[r + 1], stay)
            new_log_R[0] = growth

            norm = logsumexp(new_log_R)
            log_R = new_log_R - norm
            prob_change[t] = float(np.exp(log_R[0]))

            # advance sufficient statistics: surviving runs grow by one observation;
            # the changepoint hypothesis resets to an empty run
            new_stats = {0: (0, mu0, 0.0)}
            for r in active:
                if r + 1 <= max_run and log_R[r + 1] > -np.inf:
                    new_stats[r + 1] = add_observation(r, x)
            stats_by_run = new_stats

        prob_change = np.clip(prob_change, 0.0, 1.0)
        points = [
            i for i in range(1, T)
            if prob_change[i] > self.threshold and prob_change[i] >= prob_change[i - 1]
        ]
        self.probability_of_change_ = prob_change
        self.runlength_matrix_note_ = "run-length support truncated at max_run"
        result = BOCPDResult(probability_of_change=prob_change, points=points,
                             extra={"hazard": hazard, "max_run": max_run})
        self.result_ = result
        return self


# --------------------------------------------------------------------------- facade


def ChangePointDetection(y, method="pelt", penalty=None, threshold=None):
    """Facade dispatching by name.

    - ``"pelt"`` → :func:`PELT` (uses ``penalty``; BIC-like default when None)
    - ``"binary_segmentation"`` / ``"binseg"`` → :func:`BinarySegmentation`
    - ``"bottom_up"`` → :func:`BottomUp`
    - ``"bocpd"`` → :class:`BayesianChangePoint` (uses ``threshold``)

    All methods model segments by their mean (Gaussian cost); see module docstring.
    """
    method = str(method).lower()
    if method == "pelt":
        return PELT(y, penalty=penalty)
    if method in ("binary_segmentation", "binseg"):
        thr = threshold if threshold is not None else penalty
        return BinarySegmentation(y, threshold=thr)
    if method == "bottom_up":
        thr = threshold if threshold is not None else penalty
        return BottomUp(y, threshold=thr)
    if method in ("bocpd", "bayesian"):
        return BayesianChangePoint(threshold=threshold or 0.4).fit(y).result_
    raise ValueError(
        f"unknown method {method!r}; choose 'pelt', 'binary_segmentation', "
        "'bottom_up' or 'bocpd'"
    )
