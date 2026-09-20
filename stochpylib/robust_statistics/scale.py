"""Robust scale estimators.

``MedianAbsoluteDeviation``/``IQR_Scale`` are the classic simple robust
scales; ``Qn_Estimator``/``Sn_Estimator`` (Rousseeuw & Croux 1993) trade the
25% breakdown point of the MAD for higher Gaussian efficiency without
requiring a location estimate; ``RobustStd`` dispatches to all of them plus
Huber/biweight/tau scale M-estimators.
"""

import numpy as np

from stochpylib.robust_statistics._base import RobustEstimator
from stochpylib.robust_statistics._common import (
    _biweight_midvariance, _kth_pairwise, _lomed, _mad, _QN_CONST, _rng,
    _SN_CONST, _tau_scale,
)

__all__ = ["MedianAbsoluteDeviation", "Qn_Estimator", "Sn_Estimator", "RobustStd", "IQR_Scale"]


def _bootstrap_se(x, statistic, n_boot, random_state):
    if n_boot is None or n_boot <= 0:
        return float("nan"), None
    rng = _rng(random_state)
    n = len(x)
    idx = rng.integers(0, n, size=(n_boot, n))
    boot = np.array([statistic(x[row]) for row in idx])
    return float(np.std(boot, ddof=1)), boot


class MedianAbsoluteDeviation(RobustEstimator):
    """Median absolute deviation from the median, ``normal=True`` scaling it
    to be consistent for sigma at the Gaussian (matches
    ``scipy.stats.median_abs_deviation(x, scale="normal")``)."""

    method = "MAD"

    def __init__(self, center=None, normal=True, n_boot=200, random_state=None):
        self.center = center
        self.normal = bool(normal)
        self.n_boot = int(n_boot)
        self.random_state = random_state

    def _fit(self, x):
        est = _mad(x, center=self.center, normal=self.normal)
        se, boot = _bootstrap_se(
            x, lambda xx: _mad(xx, center=self.center, normal=self.normal),
            self.n_boot, self.random_state)
        return est, se, {"bootstrap_distribution": boot}


class IQR_Scale(RobustEstimator):
    """Interquartile-range scale, ``normal=True`` scaling by
    ``1 / (2*Phi^-1(0.75))`` (matches ``scipy.stats.iqr(x, scale="normal")``)."""

    method = "IQR scale"

    def __init__(self, normal=True, q=(0.25, 0.75), interpolation="linear", n_boot=200,
                random_state=None):
        self.normal = bool(normal)
        self.q = q
        self.interpolation = interpolation
        self.n_boot = int(n_boot)
        self.random_state = random_state

    def _statistic(self, x):
        lo, hi = np.quantile(x, self.q, method=self.interpolation)
        iqr = float(hi - lo)
        from stochpylib.robust_statistics._common import _IQR_CONST
        return iqr / _IQR_CONST if self.normal else iqr

    def _fit(self, x):
        est = self._statistic(x)
        se, boot = _bootstrap_se(x, self._statistic, self.n_boot, self.random_state)
        return est, se, {"bootstrap_distribution": boot}


class Qn_Estimator(RobustEstimator):
    """Rousseeuw & Croux (1993) Qn scale: ``2.2219 * {|x_i - x_j|}_(k)`` for
    ``k = C(h, 2)``, ``h = floor(n/2) + 1`` -- 50% breakdown, ~82% Gaussian
    efficiency, no location estimate needed. Matches
    ``statsmodels.robust.scale.qn_scale`` exactly (no finite-sample
    correction beyond the asymptotic constant)."""

    method = "Qn"

    def __init__(self, normal=True, n_boot=200, random_state=None):
        self.normal = bool(normal)
        self.n_boot = int(n_boot)
        self.random_state = random_state

    def _statistic(self, x):
        n = len(x)
        if n < 2:
            return 0.0
        h = n // 2 + 1
        k = h * (h - 1) // 2
        d = _kth_pairwise(x, k, kind="diff")
        return _QN_CONST * d if self.normal else d

    def _fit(self, x):
        est = self._statistic(x)
        se, boot = _bootstrap_se(x, self._statistic, self.n_boot, self.random_state)
        return est, se, {"bootstrap_distribution": boot}


class Sn_Estimator(RobustEstimator):
    """Rousseeuw & Croux (1993) Sn scale: ``1.1926 * lomed_i himed_j |x_i -
    x_j|`` -- 50% breakdown, ~58% Gaussian efficiency, O(n log n) in
    principle (implemented here as O(n^2), row-chunked)."""

    method = "Sn"

    def __init__(self, normal=True, n_boot=200, random_state=None, chunk=256):
        self.normal = bool(normal)
        self.n_boot = int(n_boot)
        self.random_state = random_state
        self.chunk = int(chunk)

    def _statistic(self, x):
        n = len(x)
        if n < 2:
            return 0.0
        himeds = np.empty(n)
        for start in range(0, n, self.chunk):
            block = x[start:start + self.chunk]
            diffs = np.abs(block[:, None] - x[None, :])
            himeds[start:start + len(block)] = np.sort(diffs, axis=1)[:, n // 2]
        s = _lomed(himeds)
        return _SN_CONST * s if self.normal else s

    def _fit(self, x):
        est = self._statistic(x)
        se, boot = _bootstrap_se(x, self._statistic, self.n_boot, self.random_state)
        return est, se, {"bootstrap_distribution": boot}


_ROBUST_STD_METHODS = ("mad", "qn", "sn", "iqr", "huber", "biweight", "tau")


class RobustStd(RobustEstimator):
    """Dispatcher over every scale-M-estimator in this submodule: ``"mad"``,
    ``"qn"``, ``"sn"``, ``"iqr"`` (delegate to the classes above),
    ``"huber"`` (Huber's proposal 2), ``"biweight"`` (biweight midvariance,
    c=9), ``"tau"`` (Yohai-Zamar tau-scale)."""

    method = "robust std"

    def __init__(self, method="mad", n_boot=200, random_state=None, **kwargs):
        if method not in _ROBUST_STD_METHODS:
            raise ValueError(f"method must be one of {_ROBUST_STD_METHODS}")
        self.method_ = method
        self.kwargs = kwargs
        self.n_boot = int(n_boot)
        self.random_state = random_state

    def _statistic(self, x):
        m = self.method_
        if m == "mad":
            return _mad(x, normal=True, **self.kwargs)
        if m == "qn":
            return Qn_Estimator(**self.kwargs)._statistic(x)
        if m == "sn":
            return Sn_Estimator(**self.kwargs)._statistic(x)
        if m == "iqr":
            return IQR_Scale(**self.kwargs)._statistic(x)
        if m == "huber":
            from stochpylib.robust_statistics._common import _huber_proposal2
            _, s = _huber_proposal2(x, **self.kwargs)
            return s
        if m == "biweight":
            return _biweight_midvariance(x, **self.kwargs)
        if m == "tau":
            _, s = _tau_scale(x, **self.kwargs)
            return s
        raise ValueError(f"unknown method {m!r}")

    def _fit(self, x):
        est = self._statistic(x)
        location = None
        if self.method_ == "huber":
            from stochpylib.robust_statistics._common import _huber_proposal2
            location, est = _huber_proposal2(x, **self.kwargs)
        elif self.method_ == "tau":
            location, est = _tau_scale(x, **self.kwargs)
        se, boot = _bootstrap_se(x, self._statistic, self.n_boot, self.random_state)
        extras = {"bootstrap_distribution": boot}
        if location is not None:
            extras["location_"] = location
        return est, se, extras
