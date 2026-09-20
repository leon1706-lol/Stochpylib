"""Shared base classes for stochpylib.robust_statistics.

Mirrors the library-wide convention (AGENTS.md Sec 3): every estimator's
``.fit()`` returns ``self``, every stochastic method takes ``random_state=``,
and every estimator exposes a shared result object. Location/scale estimators
and regressors reuse ``stochpylib.statistics``'s ``EstimateResult``/
``RegressionResult`` rather than defining new ones -- this module's siblings.
"""

import numpy as np

from stochpylib.robust_statistics._common import _as_1d, _design, _norm_ppf, _t_ppf
from stochpylib.statistics._result import EstimateResult, RegressionResult

__all__ = ["RobustEstimator", "RobustRegressor", "RobustCovarianceEstimator", "Resampler"]


class RobustEstimator:
    """Base for location/scale estimators.

    Subclasses implement ``_fit(x) -> (estimate, std_error, extras)``.
    ``fit(x)`` sets ``estimate_``/``std_error_``/``n_``/``extras_`` and returns
    ``self``. ``extras_['df']`` (if present) switches ``confidence_interval``
    to a Student-t interval instead of the default normal approximation.
    """

    method = ""

    def fit(self, x):
        x = _as_1d(x)
        self.n_ = len(x)
        self.estimate_, self.std_error_, self.extras_ = self._fit(x)
        return self

    def _fit(self, x):
        raise NotImplementedError

    def confidence_interval(self, level=0.95):
        if not hasattr(self, "estimate_"):
            raise RuntimeError(f"{type(self).__name__} is not fitted")
        # the cached CI (if any) was computed at fit time for level=0.95 only;
        # any other level falls through to the normal/t approximation below.
        if level == 0.95 and "confidence_interval" in self.extras_:
            return self.extras_["confidence_interval"]
        alpha2 = (1.0 - level) / 2.0
        df = self.extras_.get("df")
        z = float(_t_ppf(1.0 - alpha2, df)) if df else float(_norm_ppf(1.0 - alpha2))
        lo = self.estimate_ - z * self.std_error_
        hi = self.estimate_ + z * self.std_error_
        return (float(lo), float(hi))

    def to_result(self):
        return EstimateResult(self.estimate_, self.std_error_, self.n_,
                              self.method, dict(self.extras_))

    def __float__(self):
        return float(self.estimate_)

    def __repr__(self):
        if not hasattr(self, "estimate_"):
            return f"{type(self).__name__}(unfitted)"
        return f"{type(self).__name__}(estimate={self.estimate_!r}, std_error={self.std_error_!r})"


class RobustRegressor:
    """Base for robust regressors.

    Subclasses implement ``_fit(Xd, y)`` and set at least ``coef_`` (intercept
    first when ``fit_intercept``). ``fitted_``/``resid_`` are computed
    generically if the subclass does not set them itself.
    """

    def __init__(self, fit_intercept=True):
        self.fit_intercept = fit_intercept

    def fit(self, X, y):
        Xd = _design(X, self.fit_intercept)
        y = _as_1d(y, "y")
        self._fit(Xd, y)
        if not hasattr(self, "fitted_") or self.fitted_ is None:
            self.fitted_ = Xd @ self.coef_
        if not hasattr(self, "resid_") or self.resid_ is None:
            self.resid_ = y - self.fitted_
        return self

    def _fit(self, Xd, y):
        raise NotImplementedError

    @property
    def intercept_(self):
        return float(self.coef_[0]) if self.fit_intercept else 0.0

    @property
    def slope_(self):
        return self.coef_[1:] if self.fit_intercept else self.coef_

    def predict(self, X):
        Xd = _design(X, self.fit_intercept)
        return Xd @ self.coef_

    def score(self, X, y):
        """Robust R^2-like measure: 1 - sum(resid^2) / sum((y - median(y))^2)."""
        y = _as_1d(y, "y")
        pred = self.predict(X)
        ss_res = float(np.sum((y - pred) ** 2))
        ss_tot = float(np.sum((y - np.median(y)) ** 2))
        return 1.0 - ss_res / ss_tot if ss_tot > 0 else 0.0

    def conf_int(self, level=0.95):
        alpha2 = (1.0 - level) / 2.0
        se = getattr(self, "std_errors_", None)
        if se is None:
            raise NotImplementedError("this estimator does not report standard errors")
        df = getattr(self, "df_resid_", None)
        z = float(_t_ppf(1.0 - alpha2, df)) if df else float(_norm_ppf(1.0 - alpha2))
        lo = self.coef_ - z * se
        hi = self.coef_ + z * se
        return lo, hi

    def summary(self):
        lines = [f"{type(self).__name__} (n={len(self.resid_)})"]
        se = getattr(self, "std_errors_", np.full(len(self.coef_), np.nan))
        for i, (c, s) in enumerate(zip(self.coef_, se)):
            name = "intercept" if (i == 0 and self.fit_intercept) else f"x{i - int(self.fit_intercept) + 1}"
            lines.append(f"  {name:>10s}: coef={c:.6g}  se={s:.6g}")
        return "\n".join(lines)

    def to_result(self):
        se = getattr(self, "std_errors_", np.full(len(self.coef_), np.nan))
        with np.errstate(invalid="ignore", divide="ignore"):
            zvals = self.coef_ / se
            from scipy import special
            pvals = 2.0 * special.ndtr(-np.abs(zvals))
        return RegressionResult(
            coef_=self.coef_, std_errors_=se, pvalues_=pvals,
            method=type(self).__name__, fit_intercept=self.fit_intercept,
            zvalues_=zvals, fitted_=self.fitted_, resid_=self.resid_,
            df_resid_=getattr(self, "df_resid_", None),
            extras={"_predict": self.predict, **getattr(self, "extras_", {})},
        )


class RobustCovarianceEstimator:
    """Base for robust scatter (location + covariance) estimators."""

    def fit(self, X):
        raise NotImplementedError

    @property
    def precision_(self):
        return np.linalg.pinv(self.covariance_)

    @property
    def correlation_(self):
        d = np.sqrt(np.diag(self.covariance_))
        outer_d = np.outer(d, d)
        return np.where(outer_d > 0, self.covariance_ / np.where(outer_d == 0, 1, outer_d), 0.0)

    def mahalanobis(self, X=None):
        from stochpylib.robust_statistics._common import _mahalanobis
        X = self._X if X is None else np.asarray(X, dtype=float)
        return _mahalanobis(X, self.location_, self.covariance_)

    def outliers(self, X=None, level=0.975):
        from stochpylib.robust_statistics._common import _chi2_ppf
        X_arr = self._X if X is None else np.asarray(X, dtype=float)
        p = X_arr.shape[1]
        d2 = self.mahalanobis(X_arr)
        cutoff = _chi2_ppf(level, p)
        return d2 > cutoff


class Resampler:
    """Base for the resampling estimators (RobustBootstrap/WildBootstrap/
    BlockBootstrap/StationaryBootstrap)."""

    def __init__(self, n_boot=1000, level=0.95, random_state=None):
        self.n_boot = int(n_boot)
        self.level = float(level)
        self.random_state = random_state

    def indices(self, rng, n):
        raise NotImplementedError

    def to_result(self):
        return EstimateResult(self.estimate_, self.std_error_, self.n_, type(self).__name__,
                              {"bias": self.bias_, "distribution": self.distribution_})
