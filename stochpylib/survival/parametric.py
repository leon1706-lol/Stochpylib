"""Parametric survival regression-free fits with right censoring.

Every fitter maximises the censored likelihood
``L = prod f(t_i)^d_i * S(t_i)^(1-d_i)`` over its parameters (positive
parameters optimised in log space), exposes ``params_`` (dict), ``aic``,
and closed-form ``survival_`` / ``hazard_`` evaluators. Exponential uses its
closed-form MLE; the rest run bounded Newton-style optimisation.
"""

import numpy as np
from scipy import optimize

from stochpylib.survival._base import SurvivalFitter, _check_durations_events

__all__ = [
    "WeibullSurvival", "ExponentialSurvival", "LogNormalSurvival",
    "LogLogisticSurvival", "GompertzSurvival",
]

_EPS = 1e-12


def _neg_ll(f, sf, t, e, theta):
    dens = np.clip(np.asarray(f(t, theta), dtype=float), _EPS, None)
    surv = np.clip(np.asarray(sf(t, theta), dtype=float), _EPS, 1.0)
    ll = float(np.sum(e * np.log(dens) + (1 - e) * np.log(surv)))
    return -ll if np.isfinite(ll) else 1e12


class _ParametricBase(SurvivalFitter):
    """Subclasses define: _n_params names, _pack/_unpack (log-space mapping),
    survival_(t,*theta), hazard_(t,*theta), density_(t,*theta),
    _initial(t,e), _bounds()."""

    def __init__(self):
        self.params_ = None
        self.loglik_ = None
        self.n_obs_ = 0

    # -- to override ---------------------------------------------------------
    def _survival(self, t, theta):
        raise NotImplementedError

    def _hazard(self, t, theta):
        raise NotImplementedError

    def _density(self, t, theta):
        raise NotImplementedError

    def _initial(self, t, e):
        raise NotImplementedError

    def _bounds(self):
        return None

    # -- shared fit ------------------------------------------------------------
    def fit(self, durations, events=None):
        t, e = _check_durations_events(durations, events)
        self.n_obs_ = len(t)

        def neg_ll(log_theta):
            theta = np.exp(np.asarray(log_theta, dtype=float))
            return _neg_ll(self._density, self._survival, t, e,
                           np.atleast_1d(theta))

        x0 = np.log(np.maximum(np.asarray(self._initial(t, e), float), 1e-8))
        res = optimize.minimize(neg_ll, x0, method="Nelder-Mead",
                                options={"xatol": 1e-6, "fatol": 1e-8,
                                         "maxiter": 4000})
        theta = np.exp(res.x)
        self.params_ = dict(zip(self._names, [float(x) for x in theta]))
        self.loglik_ = -float(neg_ll(res.x))
        self.aic_ = -2.0 * self.loglik_ + 2.0 * len(theta)
        return self

    def _require_fit(self):
        if self.params_ is None:
            raise RuntimeError("fit() must be called first")

    def _theta(self):
        return np.array([self.params_[k] for k in self._names], dtype=float)

    def survival_(self, times):
        self._require_fit()
        return self._survival(np.asarray(times, dtype=float), self._theta())

    def hazard_(self, times):
        self._require_fit()
        return self._hazard(np.asarray(times, dtype=float), self._theta())

    def aic(self):
        self._require_fit()
        return self.aic_

    def loglik(self):
        self._require_fit()
        return self.loglik_


class WeibullSurvival(_ParametricBase):
    """Weibull survival: S(t) = exp(-(t/scale)^shape)."""

    _names = ("shape", "scale")

    def _initial(self, t, e):
        m = max(float(np.mean(t[e == 1])) if e.any() else float(np.mean(t)),
                1e-3)
        return np.array([1.0, m])

    def _survival(self, t, th):
        k, lam = th
        return np.exp(-np.maximum(t / lam, 0.0) ** k)

    def _hazard(self, t, th):
        k, lam = th
        tt = np.maximum(t, 0.0)
        return (k / lam) * np.maximum(tt / lam, 0.0) ** (k - 1.0)

    def _density(self, t, th):
        return self._hazard(t, th) * self._survival(t, th)


class ExponentialSurvival(_ParametricBase):
    """Exponential survival with closed-form censored MLE for the rate."""

    _names = ("rate",)

    def fit(self, durations, events=None):
        t, e = _check_durations_events(durations, events)
        self.n_obs_ = len(t)
        total_time = float(np.sum(t))
        rate = float(np.sum(e)) / max(total_time, _EPS)
        self.params_ = {"rate": rate}
        ll = float(np.sum(e * np.log(rate) - rate * t))
        self.loglik_ = ll
        self.aic_ = -2.0 * ll + 2.0
        return self

    def _survival(self, t, th):
        (rate,) = th
        return np.exp(-rate * np.maximum(t, 0.0))

    def _hazard(self, t, th):
        (rate,) = th
        return np.full_like(np.asarray(t, dtype=float), rate)

    def _density(self, t, th):
        (rate,) = th
        return rate * self._survival(t, th)


class LogNormalSurvival(_ParametricBase):
    """Log-normal survival: log T ~ N(mu, sigma^2)."""

    _names = ("mu", "sigma")

    def _initial(self, t, e):
        lt = np.log(np.maximum(t[e == 1] if e.any() else t, 1e-6))
        return np.array([float(np.mean(lt)), max(float(np.std(lt)), 1e-2)])

    def _survival(self, t, th):
        mu, sig = th
        from scipy import special
        z = (np.log(np.maximum(t, 1e-300)) - mu) / sig
        return 0.5 * special.erfc(z / np.sqrt(2.0))

    def _hazard(self, t, th):
        mu, sig = th
        from scipy import special
        tt = np.maximum(t, 1e-300)
        z = (np.log(tt) - mu) / sig
        pdf = np.exp(-0.5 * z ** 2) / (tt * sig * np.sqrt(2 * np.pi))
        s = 0.5 * special.erfc(z / np.sqrt(2.0))
        return pdf / np.maximum(s, _EPS)

    def _density(self, t, th):
        mu, sig = th
        tt = np.maximum(t, 1e-300)
        z = (np.log(tt) - mu) / sig
        return np.exp(-0.5 * z ** 2) / (tt * sig * np.sqrt(2 * np.pi))


class LogLogisticSurvival(_ParametricBase):
    """Log-logistic survival: S(t) = 1/(1+(t/alpha)^beta)."""

    _names = ("alpha", "beta")

    def _initial(self, t, e):
        m = max(float(np.median(t)), 1e-3)
        return np.array([m, 1.5])

    def _survival(self, t, th):
        a, b = th
        r = np.maximum(t / a, 0.0) ** b
        return 1.0 / (1.0 + r)

    def _hazard(self, t, th):
        a, b = th
        tt = np.maximum(t, 1e-300)
        r = (tt / a) ** b
        return (b / tt) * r / (1.0 + r)

    def _density(self, t, th):
        a, b = th
        tt = np.maximum(t, 1e-300)
        r = (tt / a) ** b
        return (b / tt) * r / (1.0 + r) ** 2


class GompertzSurvival(_ParametricBase):
    """Gompertz survival: h(t)=a*exp(b*t), S(t)=exp(a/b*(1-exp(b t)))."""

    _names = ("a", "b")

    def _initial(self, t, e):
        m = max(float(np.mean(t)), 1e-3)
        return np.array([max(1.0 / m, 1e-3), 0.05])

    def _survival(self, t, th):
        a, b = th
        tt = np.maximum(t, 0.0)
        if abs(b) < 1e-10:
            return np.exp(-a * tt)
        return np.exp(a / b * (1.0 - np.exp(b * tt)))

    def _hazard(self, t, th):
        a, b = th
        return a * np.exp(b * np.maximum(t, 0.0))

    def _density(self, t, th):
        return self._hazard(t, th) * self._survival(t, th)
