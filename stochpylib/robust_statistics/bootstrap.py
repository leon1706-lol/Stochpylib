"""Resampling-based standard errors and confidence intervals.

``RobustBootstrap`` is the general-purpose i.i.d. bootstrap (any statistic,
including this module's own estimators); ``WildBootstrap`` resamples
regression residuals with a mean-zero, unit-variance multiplier, preserving
heteroskedasticity structure; ``BlockBootstrap``/``StationaryBootstrap``
resample contiguous blocks for dependent (time-series) data.
"""

import math

import numpy as np

from stochpylib.robust_statistics._base import Resampler, RobustEstimator
from stochpylib.robust_statistics._common import _as_1d, _design, _mad, _norm_ppf, _rng

__all__ = ["RobustBootstrap", "WildBootstrap", "BlockBootstrap", "StationaryBootstrap"]


_STRING_STATISTICS = {
    "median": lambda x: float(np.median(x)),
    "mad": lambda x: _mad(x, normal=True),
}


def _named_factory(name):
    from stochpylib.robust_statistics import location, scale
    factories = {
        "trimmed_mean": lambda: location.TrimmedMean(),
        "huber": lambda: location.M_Estimator("huber"),
        "hodges_lehmann": lambda: location.HodgesLehmann(),
        "qn": lambda: scale.Qn_Estimator(n_boot=0),
    }
    if name not in factories:
        raise ValueError(f"unknown statistic name {name!r}")
    return factories[name]


def _resolve_statistic(statistic):
    if isinstance(statistic, str):
        if statistic in _STRING_STATISTICS:
            return _STRING_STATISTICS[statistic]
        factory = _named_factory(statistic)
        return lambda x: float(factory().fit(x).estimate_)
    if isinstance(statistic, RobustEstimator):
        cls = type(statistic)
        kwargs = {k: v for k, v in vars(statistic).items()
                 if not k.startswith("_") and k not in ("estimate_", "std_error_", "n_", "extras_")}

        def stat_fn(x):
            return float(cls(**kwargs).fit(x).estimate_)
        return stat_fn
    if callable(statistic):
        return statistic
    raise TypeError("statistic must be a callable, a RobustEstimator instance, or a known name")


def _ci_from_boot(theta_hat, boot, level, method, se, jackknife_fn=None):
    alpha2 = (1.0 - level) / 2.0
    if method == "percentile":
        lo, hi = np.quantile(boot, [alpha2, 1.0 - alpha2])
        return float(lo), float(hi)
    if method == "basic":
        q_lo, q_hi = np.quantile(boot, [alpha2, 1.0 - alpha2])
        return float(2 * theta_hat - q_hi), float(2 * theta_hat - q_lo)
    if method == "normal":
        z = float(_norm_ppf(1.0 - alpha2))
        return float(theta_hat - z * se), float(theta_hat + z * se)
    if method == "bca":
        if jackknife_fn is None:
            raise ValueError("bca requires a cheap jackknife (n <= 2000)")
        jack_stats = jackknife_fn()
        jack_mean = np.mean(jack_stats)
        d = jack_mean - jack_stats
        denom = 6.0 * np.sum(d ** 2) ** 1.5
        a_hat = float(np.sum(d ** 3) / denom) if denom != 0 else 0.0
        prop_less = np.mean(boot < theta_hat) + 0.5 * np.mean(boot == theta_hat)
        z0 = float(_norm_ppf(np.clip(prop_less, 1e-10, 1 - 1e-10)))
        z_lo, z_hi = float(_norm_ppf(alpha2)), float(_norm_ppf(1.0 - alpha2))
        from stochpylib.robust_statistics._common import _norm_cdf
        p_lo = float(_norm_cdf(z0 + (z0 + z_lo) / (1.0 - a_hat * (z0 + z_lo))))
        p_hi = float(_norm_cdf(z0 + (z0 + z_hi) / (1.0 - a_hat * (z0 + z_hi))))
        lo, hi = np.quantile(boot, [p_lo, p_hi])
        return float(lo), float(hi)
    raise ValueError("method must be 'percentile', 'basic', 'normal', or 'bca'")


class RobustBootstrap(Resampler):
    """General-purpose i.i.d. bootstrap for an arbitrary statistic.

    ``statistic``: a callable on a 1-D array, a fitted-or-unfitted
    :class:`~stochpylib.robust_statistics.RobustEstimator` instance (its
    constructor arguments are reused, never its stale fit), or a name --
    ``"median"``, ``"mad"``, ``"trimmed_mean"``, ``"huber"``,
    ``"hodges_lehmann"``, ``"qn"``.
    """

    def __init__(self, statistic="median", n_boot=1000, level=0.95, method="percentile",
                random_state=None):
        super().__init__(n_boot, level, random_state)
        self.statistic = statistic
        self.method = method
        self._stat_fn = _resolve_statistic(statistic)

    def fit(self, x):
        x = _as_1d(x)
        n = len(x)
        self.n_ = n
        self.estimate_ = self._stat_fn(x)
        rng = _rng(self.random_state)
        idx = rng.integers(0, n, size=(self.n_boot, n))
        self.distribution_ = np.array([self._stat_fn(x[row]) for row in idx])
        self.std_error_ = float(np.std(self.distribution_, ddof=1))
        self.bias_ = float(np.mean(self.distribution_) - self.estimate_)
        self._x = x
        return self

    def confidence_interval(self, level=None, method=None):
        level = self.level if level is None else level
        method = self.method if method is None else method
        jackknife_fn = None
        if method == "bca":
            n = self.n_
            if n > 2000:
                raise ValueError("bca requires a cheap jackknife (n <= 2000)")
            jackknife_fn = lambda: np.array(
                [self._stat_fn(np.delete(self._x, i)) for i in range(n)])
        return _ci_from_boot(self.estimate_, self.distribution_, level, method,
                             self.std_error_, jackknife_fn)


_WILD_WEIGHTS = {}


def _rademacher(rng, n):
    return rng.choice([-1.0, 1.0], size=n)


def _mammen(rng, n):
    """Mammen (1993) two-point distribution: mean 0, variance 1, E[v^3]=1."""
    a = -(math.sqrt(5.0) - 1.0) / 2.0
    b = (math.sqrt(5.0) + 1.0) / 2.0
    p_a = (math.sqrt(5.0) + 1.0) / (2.0 * math.sqrt(5.0))
    return np.where(rng.uniform(size=n) < p_a, a, b)


def _normal_w(rng, n):
    return rng.standard_normal(n)


def _webb(rng, n):
    vals = np.array([-math.sqrt(1.5), -1.0, -math.sqrt(0.5), math.sqrt(0.5), 1.0, math.sqrt(1.5)])
    idx = rng.integers(0, 6, size=n)
    return vals[idx]


_WILD_WEIGHTS = {"rademacher": _rademacher, "mammen": _mammen, "normal": _normal_w, "webb": _webb}


class WildBootstrap(Resampler):
    """Wild bootstrap for regression: resamples residuals with a mean-zero,
    unit-variance multiplier (preserving each observation's own residual
    magnitude, so heteroskedasticity structure survives), refitting ``model``
    (default OLS, or any ``RobustRegressor`` instance) on each replicate.
    """

    def __init__(self, model=None, n_boot=1000, weights="rademacher", level=0.95,
                fit_intercept=True, random_state=None):
        super().__init__(n_boot, level, random_state)
        if weights not in _WILD_WEIGHTS:
            raise ValueError(f"weights must be one of {sorted(_WILD_WEIGHTS)}")
        self.model = model
        self.weights = weights
        self.fit_intercept = fit_intercept

    def indices(self, rng, n):
        raise NotImplementedError("WildBootstrap resamples residual multipliers, not row indices")

    def _fit_once(self, Xd, y):
        if self.model is None:
            beta, *_ = np.linalg.lstsq(Xd, y, rcond=None)
            return beta
        from copy import deepcopy
        m = deepcopy(self.model)
        m.fit_intercept = False  # Xd already carries the intercept column
        m.fit(Xd, y)
        return m.coef_

    def fit(self, X, y):
        Xd = _design(X, self.fit_intercept)
        y = _as_1d(y, "y")
        n, p = Xd.shape
        base = self._fit_once(Xd, y)
        fitted = Xd @ base
        resid = y - fitted

        rng = _rng(self.random_state)
        w_fn = _WILD_WEIGHTS[self.weights]
        self.distribution_ = np.empty((self.n_boot, p))
        for b in range(self.n_boot):
            v = w_fn(rng, n)
            y_star = fitted + resid * v
            self.distribution_[b] = self._fit_once(Xd, y_star)

        self.coef_ = base
        self.std_errors_ = np.std(self.distribution_, axis=0, ddof=1)
        self.n_ = n
        return self

    def conf_int(self, level=None, method="percentile"):
        level = self.level if level is None else level
        alpha2 = (1.0 - level) / 2.0
        p = len(self.coef_)
        out = []
        for j in range(p):
            col = self.distribution_[:, j]
            if method == "percentile":
                lo, hi = np.quantile(col, [alpha2, 1.0 - alpha2])
            elif method == "basic":
                q_lo, q_hi = np.quantile(col, [alpha2, 1.0 - alpha2])
                lo, hi = 2 * self.coef_[j] - q_hi, 2 * self.coef_[j] - q_lo
            elif method == "normal":
                z = float(_norm_ppf(1.0 - alpha2))
                lo, hi = self.coef_[j] - z * self.std_errors_[j], self.coef_[j] + z * self.std_errors_[j]
            else:
                raise ValueError("method must be 'percentile', 'basic', or 'normal'")
            out.append((float(lo), float(hi)))
        return out

    def to_result(self):
        from stochpylib.statistics._result import RegressionResult
        with np.errstate(invalid="ignore", divide="ignore"):
            zvals = self.coef_ / self.std_errors_
            from scipy import special
            pvals = 2.0 * special.ndtr(-np.abs(zvals))
        return RegressionResult(coef_=self.coef_, std_errors_=self.std_errors_, pvalues_=pvals,
                                method="WildBootstrap", fit_intercept=self.fit_intercept,
                                zvalues_=zvals, extras={"distribution": self.distribution_})


def _moving_block_indices(rng, n, L):
    n_blocks = math.ceil(n / L)
    starts = rng.integers(0, max(n - L + 1, 1), size=n_blocks)
    idx = np.concatenate([np.arange(s, s + L) for s in starts])[:n]
    return idx


def _circular_block_indices(rng, n, L):
    n_blocks = math.ceil(n / L)
    starts = rng.integers(0, n, size=n_blocks)
    idx = np.concatenate([(np.arange(s, s + L) % n) for s in starts])[:n]
    return idx


def _nonoverlapping_block_indices(rng, n, L):
    n_blocks_total = n // L
    if n_blocks_total == 0:
        return rng.integers(0, n, size=n)
    block_starts = np.arange(n_blocks_total) * L
    chosen = rng.integers(0, n_blocks_total, size=math.ceil(n / L))
    idx = np.concatenate([np.arange(block_starts[c], block_starts[c] + L) for c in chosen])[:n]
    return idx


_BLOCK_KINDS = {
    "moving": _moving_block_indices,
    "circular": _circular_block_indices,
    "nonoverlapping": _nonoverlapping_block_indices,
}


class BlockBootstrap(Resampler):
    """Block bootstrap for dependent (e.g. time-series) data.

    ``kind``: ``"moving"`` (Kunsch 1989, overlapping blocks of fixed length
    ``block_length``), ``"circular"`` (Politis & Romano 1992, wraps modulo
    ``n``), ``"nonoverlapping"`` (Carlstein 1986, tiles ``0..n-1`` and
    resamples whole tiles).
    """

    def __init__(self, statistic=np.mean, block_length=None, kind="moving", n_boot=1000,
                level=0.95, random_state=None):
        super().__init__(n_boot, level, random_state)
        if kind not in _BLOCK_KINDS:
            raise ValueError(f"kind must be one of {sorted(_BLOCK_KINDS)}")
        self.statistic = statistic
        self.block_length = block_length
        self.kind = kind

    def _stat(self, x):
        return float(self.statistic(x))

    def fit(self, x):
        x = np.asarray(x, dtype=float)
        n = len(x)
        L = max(int(round(n ** (1.0 / 3.0))), 2) if self.block_length is None else int(self.block_length)
        self.n_ = n
        self.estimate_ = self._stat(x)
        rng = _rng(self.random_state)
        gen = _BLOCK_KINDS[self.kind]
        boot = np.empty(self.n_boot)
        for b in range(self.n_boot):
            idx = gen(rng, n, L)
            boot[b] = self._stat(x[idx])
        self.distribution_ = boot
        self.std_error_ = float(np.std(boot, ddof=1))
        self.bias_ = float(np.mean(boot) - self.estimate_)
        self.block_length_ = L
        return self

    def indices(self, rng, n=None):
        n = self.n_ if n is None else n
        L = self.block_length_ if hasattr(self, "block_length_") else (
            max(int(round(n ** (1.0 / 3.0))), 2) if self.block_length is None else int(self.block_length))
        return _BLOCK_KINDS[self.kind](rng, n, L)

    def confidence_interval(self, level=None, method="percentile"):
        level = self.level if level is None else level
        return _ci_from_boot(self.estimate_, self.distribution_, level, method, self.std_error_)


class StationaryBootstrap(Resampler):
    """Politis & Romano (1994) stationary bootstrap: block lengths are
    geometric with mean ``mean_block_length`` rather than fixed, which (unlike
    the fixed-length block bootstrap) yields a stationary resampled series.
    """

    def __init__(self, statistic=np.mean, mean_block_length=None, n_boot=1000, level=0.95,
                random_state=None):
        super().__init__(n_boot, level, random_state)
        self.statistic = statistic
        self.mean_block_length = mean_block_length

    def _stat(self, x):
        return float(self.statistic(x))

    def _p(self, n):
        L = (max(round(n ** (1.0 / 3.0)), 2) if self.mean_block_length is None
            else float(self.mean_block_length))
        return 1.0 / L

    def indices(self, rng, n=None):
        n = self.n_ if n is None else n
        p = self._p(n)
        idx = np.empty(n, dtype=int)
        idx[0] = rng.integers(0, n)
        cont = rng.uniform(size=n) >= p
        for i in range(1, n):
            idx[i] = (idx[i - 1] + 1) % n if cont[i] else rng.integers(0, n)
        return idx

    def fit(self, x):
        x = np.asarray(x, dtype=float)
        n = len(x)
        self.n_ = n
        self.estimate_ = self._stat(x)
        rng = _rng(self.random_state)
        boot = np.empty(self.n_boot)
        for b in range(self.n_boot):
            idx = self.indices(rng, n)
            boot[b] = self._stat(x[idx])
        self.distribution_ = boot
        self.std_error_ = float(np.std(boot, ddof=1))
        self.bias_ = float(np.mean(boot) - self.estimate_)
        return self

    def confidence_interval(self, level=None, method="percentile"):
        level = self.level if level is None else level
        return _ci_from_boot(self.estimate_, self.distribution_, level, method, self.std_error_)
