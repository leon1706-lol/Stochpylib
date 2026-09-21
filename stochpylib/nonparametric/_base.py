"""Shared base classes for stochpylib.nonparametric.

Mirrors the library-wide convention (AGENTS.md Sec 3): every estimator's
``.fit()`` returns ``self``, every stochastic method takes ``random_state=``.
Density estimators satisfy the full 13-method distribution contract by
subclassing :class:`stochpylib.distributions._base.Distribution` (one
sanctioned deviation: ``fit`` here is a fluent instance method, since kernel
density estimates carry their bandwidth/kernel configuration as constructor
state rather than being re-derived purely from data). Hypothesis tests and
dependence measures reuse :class:`stochpylib.statistics.TestResult`/
:class:`~stochpylib.statistics.EstimateResult` rather than defining new
result types -- this module's siblings of ``robust_statistics``'s bases.
"""

import numpy as np

from stochpylib.distributions._base import Distribution
from stochpylib.nonparametric._common import _norm_ppf, _t_ppf
from stochpylib.statistics._result import EstimateResult, TestResult

__all__ = ["NonparametricDensity", "NonparametricTest", "DependenceMeasure",
           "NonparametricRegressor"]


class NonparametricDensity(Distribution):
    """Base for data-driven density estimators.

    Subclasses implement ``_fit(x) -> dict`` (stored on ``self.params_``) and
    override ``pdf``/``support`` (and ``cdf``/``rvs`` where a better-than-generic
    form exists). ``fit(x)`` is a fluent *instance* method (unlike
    ``Distribution.fit``'s classmethod contract) because these estimators keep
    their own bandwidth/kernel configuration; ``mean``/``var`` fall back to the
    sample moments unless overridden.
    """

    def fit(self, x):
        x = np.asarray(x, dtype=float).ravel()
        if x.size == 0 or not np.all(np.isfinite(x)):
            raise ValueError("x must be a non-empty array of finite values")
        self.data_ = x
        self.n_ = len(x)
        self.params_ = self._fit(x)
        return self

    def _fit(self, x):
        raise NotImplementedError

    def logpdf(self, x):
        p = np.asarray(self.pdf(x), dtype=float)
        with np.errstate(divide="ignore"):
            return np.log(np.clip(p, 1e-300, None))

    score_samples = logpdf

    def mean(self):
        return float(np.mean(self.data_))

    def var(self):
        return float(np.var(self.data_, ddof=1))

    def support(self):
        pad = 3.0 * float(np.std(self.data_, ddof=1) + 1e-12)
        return (float(np.min(self.data_) - pad), float(np.max(self.data_) + pad))

    def __repr__(self):
        return f"{type(self).__name__}(n={getattr(self, 'n_', 0)})"


class NonparametricTest:
    """Base for hypothesis tests that don't fit a single closed statistic/pvalue
    pair generically -- subclasses implement ``_fit(*samples) -> TestResult``.
    ``fit(*samples)`` sets ``statistic_``/``pvalue_``/``result_`` and returns self.
    """

    def fit(self, *samples):
        self.result_ = self._fit(*samples)
        self.statistic_ = self.result_.statistic
        self.pvalue_ = self.result_.pvalue
        return self

    def _fit(self, *samples):
        raise NotImplementedError

    def to_result(self):
        return self.result_

    def reject(self, alpha=0.05):
        return self.pvalue_ is not None and self.pvalue_ < alpha

    def __repr__(self):
        if not hasattr(self, "statistic_"):
            return f"{type(self).__name__}(unfitted)"
        pv = "n/a" if self.pvalue_ is None else f"{self.pvalue_:.4g}"
        return f"{type(self).__name__}(statistic={self.statistic_:.4g}, pvalue={pv})"


class DependenceMeasure:
    """Base for bivariate dependence/association measures.

    Subclasses implement ``_fit(x, y) -> (estimate, pvalue, extras)``.
    ``fit(x, y)`` sets ``estimate_``/``pvalue_``/``n_``/``extras_`` and returns
    ``self``.
    """

    def fit(self, x, y):
        x = np.asarray(x, dtype=float).ravel()
        y = np.asarray(y, dtype=float).ravel()
        if len(x) != len(y) or len(x) < 2:
            raise ValueError("x and y must be equal-length arrays with n >= 2")
        self.n_ = len(x)
        self.estimate_, self.pvalue_, self.extras_ = self._fit(x, y)
        return self

    def _fit(self, x, y):
        raise NotImplementedError

    def confidence_interval(self, level=0.95):
        se = self.extras_.get("std_error")
        if se is None:
            raise NotImplementedError("this measure does not report a standard error")
        alpha2 = (1.0 - level) / 2.0
        z = float(_norm_ppf(1.0 - alpha2))
        lo = self.estimate_ - z * se
        hi = self.estimate_ + z * se
        return (float(lo), float(hi))

    def to_result(self):
        return EstimateResult(self.estimate_, self.extras_.get("std_error", float("nan")),
                               self.n_, type(self).__name__,
                               {**self.extras_, "pvalue": self.pvalue_})

    def __float__(self):
        return float(self.estimate_)

    def __repr__(self):
        if not hasattr(self, "estimate_"):
            return f"{type(self).__name__}(unfitted)"
        return f"{type(self).__name__}(estimate={self.estimate_!r})"


class NonparametricRegressor:
    """Base for smoothers/regressors: ``fit(X, y) -> self``,
    ``predict(X, return_std=False)``, ``fitted_``/``resid_`` set generically
    from ``predict`` on the training data unless the subclass sets them itself.
    """

    def fit(self, X, y):
        X = np.asarray(X, dtype=float)
        if X.ndim == 1:
            X = X[:, None]
        y = np.asarray(y, dtype=float).ravel()
        if X.shape[0] != len(y):
            raise ValueError("X and y must have matching length")
        self.X_ = X
        self.y_ = y
        self._fit(X, y)
        if not hasattr(self, "fitted_") or self.fitted_ is None:
            self.fitted_ = np.asarray(self.predict(X), dtype=float)
        if not hasattr(self, "resid_") or self.resid_ is None:
            self.resid_ = y - self.fitted_
        return self

    def _fit(self, X, y):
        raise NotImplementedError

    def predict(self, X, return_std=False):
        raise NotImplementedError

    def score(self, X, y):
        y = np.asarray(y, dtype=float).ravel()
        pred = np.asarray(self.predict(X), dtype=float)
        ss_res = float(np.sum((y - pred) ** 2))
        ss_tot = float(np.sum((y - np.mean(y)) ** 2))
        return 1.0 - ss_res / ss_tot if ss_tot > 0 else 0.0

    @property
    def residual_scale_(self):
        return float(np.std(self.resid_, ddof=1)) if len(self.resid_) > 1 else float("nan")

    def __repr__(self):
        n = len(self.y_) if hasattr(self, "y_") else 0
        return f"{type(self).__name__}(n={n})"
