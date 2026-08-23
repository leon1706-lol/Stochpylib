"""Univariate and multivariate linear time-series models (native implementations).

Conventions (ARCHITECTURE.md):

- fluent fit: ``ARIMA(p, d, q).fit(y)`` returns *self*; estimated parameters live on the
  instance as public attributes ending in ``_``;
- ``forecast(horizon=...)`` returns a :class:`~stochpylib.timeseries._result.ForecastResult`
  with per-step standard deviation;
- estimation is by least squares / conditional sum of squares (CSS) via
  ``scipy.optimize.minimize`` — initialized by Hannan–Rissanen where helpful;
- ``random_state=`` on every ``simulate``.

Documented simplifications: ARIMA ``d`` is restricted to {0, 1, 2}; SARIMA seasonal orders to
D in {0, 1}; VARMA uses a diagonal innovation covariance; ARFIMA uses the fixed-width truncated
fractional filter with convolution-based forecast variance. All are flagged in docstrings.
"""

import warnings

import numpy as np
from scipy import optimize

from stochpylib.timeseries._result import ForecastResult
from stochpylib.timeseries._utils import (
    aic_bic,
    as_1d,
    difference,
    frac_diff_weights,
    integrate_levels,
    lag_matrix,
    psi_weights_ar,
    seasonal_difference,
)


# ---------------------------------------------------------------------------
# shared machinery


def _innovations(y, c, ar_map, ma_map):
    """Conditional-sum-of-squares innovations with zero pre-sample residuals."""
    n = len(y)
    e = np.zeros(n)
    ar_items = list(ar_map.items())
    ma_items = list(ma_map.items())
    for t in range(n):
        s = c
        for lag, coef in ar_items:
            if t - lag >= 0:
                s += coef * y[t - lag]
        for lag, coef in ma_items:
            if t - lag >= 0:
                s += coef * e[t - lag]
        e[t] = y[t] - s
    return e


def _psi_from_maps(ar_map, ma_map, h):
    """Impulse response psi_0..psi_{h-1} of an AR/MA lag-map pair."""
    psi = np.zeros(h)
    psi[0] = 1.0
    for i in range(1, h):
        val = 0.0
        for lag, coef in ar_map.items():
            if i - lag >= 0:
                val += coef * psi[i - lag]
        if i in ma_map:
            val += ma_map[i]
        psi[i] = val
    return psi


def _lag_maps(ar_coefs, ma_coefs, seasonal=None):
    ar_map = {i + 1: float(c) for i, c in enumerate(np.atleast_1d(ar_coefs))}
    ma_map = {i + 1: float(c) for i, c in enumerate(np.atleast_1d(ma_coefs))}
    if seasonal:
        s_ar, s_ma, s = seasonal
        for i, c in enumerate(np.atleast_1d(s_ar)):
            ar_map[s * (i + 1)] = float(c)
        for i, c in enumerate(np.atleast_1d(s_ma)):
            ma_map[s * (i + 1)] = float(c)
    return ar_map, ma_map


def _check_invertible(ma_coefs, name="MA"):
    poly = np.concatenate([[1.0], np.atleast_1d(ma_coefs)])
    roots = np.roots(poly[::-1]) if len(poly) > 1 else np.array([])
    if len(roots) and np.min(np.abs(roots)) <= 1 + 1e-9:
        warnings.warn(f"{name} polynomial has a root inside/on the unit circle; "
                      "the fitted model may not be invertible", stacklevel=3)


class _UnivariateBase:
    """Shared fit/forecast plumbing for stationary univariate CSS models."""

    is_integrated = False

    def __init__(self):
        self.intercept_ = None
        self.ar_coefs_ = None
        self.ma_coefs_ = None
        self.sigma2_ = None
        self.aic_ = None
        self.bic_ = None
        self._resid = None
        self._y = None
        self._fit_series = None  # the transformed series the CSS machinery actually fit

    # subclasses override these -------------------------------------------------
    def _maps(self):
        return _lag_maps(self.ar_coefs_, self.ma_coefs_, getattr(self, "_seasonal", None))

    def _n_params(self):
        return int(self.intercept_ is not None) + len(self.ar_coefs_) + len(self.ma_coefs_)

    def _pre_transform(self, y):
        return y, []

    def _integrate_forecast(self, diff_path):
        return diff_path

    def _integrated_psi(self, h):
        ar_map, ma_map = self._maps()
        return _psi_from_maps(ar_map, ma_map, h)

    # API -----------------------------------------------------------------------
    def residuals(self):
        """In-sample CSS innovations of the fitted model."""
        if self._resid is None:
            raise RuntimeError("fit() must be called first")
        return self._resid

    def _integrate_forecast(self, path):
        return path

    def forecast(self, horizon=1):
        horizon = int(horizon)
        if horizon < 1:
            raise ValueError("horizon must be >= 1")
        ar_map, ma_map = self._maps()
        # innovations aligned to the FIT series (zero pre-sample values) so the
        # MA recursion indexes consistently with the history array below
        eh_full = _innovations(self._fit_series, self.intercept_, ar_map, ma_map)
        hist = np.concatenate([self._fit_series, np.full(horizon, np.nan)])
        eh = np.concatenate([eh_full, np.zeros(horizon)])
        T = len(self._fit_series)
        means = []
        for hstep in range(horizon):
            t = T + hstep
            s = self.intercept_
            for lag, coef in ar_map.items():
                s += coef * hist[t - lag]
            for lag, coef in ma_map.items():
                s += coef * eh[t - lag]
            hist[t] = s
            means.append(s)
        means = np.array(means)

        psi = self._integrated_psi(horizon)
        std = np.sqrt(self.sigma2_ * np.cumsum(psi**2))
        out_mean = self._integrate_forecast(means)
        n_out = len(np.atleast_1d(out_mean))
        return ForecastResult(out_mean, std[:n_out])

    def simulate(self, n=500, burnin=200, random_state=None):
        if self.sigma2_ is None:
            raise RuntimeError("fit() must be called before simulate()")
        rng = np.random.default_rng(random_state)
        total = int(n) + int(burnin)
        ar_map, ma_map = self._maps()
        max_lag = max(max(ar_map, default=0), max(ma_map, default=0), 1)
        e = rng.normal(scale=float(np.sqrt(self.sigma2_)), size=total)
        y = np.zeros(total)
        for t in range(total):
            s = self.intercept_
            for lag, coef in ar_map.items():
                if t - lag >= 0:
                    s += coef * y[t - lag]
            for lag, coef in ma_map.items():
                if t - lag >= 0:
                    s += coef * e[t - lag]
            y[t] = s + e[t]
        out = y[burnin:]
        return out[0] if n == 1 else out


# --------------------------------------------------------------------------- AR


class AR(_UnivariateBase):
    """Autoregressive model AR(p): y_t = c + sum_i phi_i y_{t-i} + e_t.

    Estimated exactly by ordinary least squares.
    """

    def __init__(self, p):
        super().__init__()
        self.p = int(p)
        if self.p < 1:
            raise ValueError("p must be >= 1")

    def fit(self, y):
        y = as_1d(y)
        X, target = lag_matrix(y, self.p)
        beta, *_ = np.linalg.lstsq(X, target, rcond=None)
        resid = target - X @ beta
        self.intercept_, self.ar_coefs_, self.ma_coefs_ = float(beta[0]), beta[1:], np.array([])
        self._resid = resid
        self.sigma2_ = float(resid @ resid) / len(resid)
        self.aic_, self.bic_ = aic_bic(self.sigma2_, len(resid), self.p + 1)
        self._y = y
        self._fit_series = y
        return self


# --------------------------------------------------------------------------- MA


class MA(_UnivariateBase):
    """Moving-average model MA(q), estimated by conditional sum of squares."""

    def __init__(self, q):
        super().__init__()
        self.q = int(q)
        if self.q < 1:
            raise ValueError("q must be >= 1")

    def fit(self, y):
        y = as_1d(y)
        self._y = y

        def css(params):
            e = _innovations(y, params[0], {}, dict(enumerate(params[1:], start=1)))
            return float(e @ e)

        x0 = np.concatenate([[float(np.mean(y))], np.zeros(self.q)])
        res = optimize.minimize(css, x0, method="Nelder-Mead",
                                options={"maxiter": 4000, "xatol": 1e-10, "fatol": 1e-12})
        params = res.x
        self.intercept_, self.ma_coefs_ = float(params[0]), params[1:]
        self.ar_coefs_ = np.array([])
        _check_invertible(self.ma_coefs_)
        e = _innovations(y, self.intercept_, {}, dict(enumerate(self.ma_coefs_, start=1)))
        self._resid = e
        self.sigma2_ = float(e @ e) / len(e)
        self.aic_, self.bic_ = aic_bic(self.sigma2_, len(e), self.q + 1)
        self._y = y
        self._fit_series = y
        return self


# --------------------------------------------------------------------------- ARMA


class ARMA(_UnivariateBase):
    """ARMA(p, q): Hannan-Rissanen initialization followed by CSS refinement."""

    def __init__(self, p, q):
        super().__init__()
        self.p, self.q = int(p), int(q)
        if self.p < 1 or self.q < 1:
            raise ValueError("p and q must both be >= 1")

    def fit(self, y):
        y = as_1d(y)
        self._y = y

        # step 1: long autoregression to approximate innovations
        k = min(max(10, round(len(y) ** (1 / 3)) + self.p + self.q), len(y) // 4)
        Xk, tk = lag_matrix(y, k)
        bk, *_ = np.linalg.lstsq(Xk, tk, rcond=None)
        ehat_full = np.concatenate([np.zeros(k), tk - Xk @ bk])

        # step 2: regress y on its lags and the innovation proxies
        start = k
        rows = []
        targets = []
        for t in range(start, len(y)):
            row = [1.0]
            row += [y[t - j] for j in range(1, self.p + 1)]
            row += [ehat_full[t - j] for j in range(1, self.q + 1)]
            rows.append(row)
            targets.append(y[t])
        H = np.asarray(rows)
        b_init, *_ = np.linalg.lstsq(H, np.asarray(targets), rcond=None)

        # step 3: CSS polish
        ar_map0 = {j: b_init[j] for j in range(1, self.p + 1)}
        ma_map0 = {j: b_init[self.p + j] for j in range(1, self.q + 1)}

        def css(params):
            c = params[0]
            amap = {j: params[j] for j in range(1, self.p + 1)}
            mmap = {j: params[self.p + j] for j in range(1, self.q + 1)}
            e = _innovations(y, c, amap, mmap)
            return float(e @ e)

        x0 = np.concatenate([[b_init[0]], b_init[1:]])
        res = optimize.minimize(css, x0, method="Nelder-Mead",
                                options={"maxiter": 8000, "maxfev": 8000,
                                         "xatol": 1e-10, "fatol": 1e-12})
        params = res.x if css(res.x) <= css(x0) else x0
        self.intercept_ = float(params[0])
        self.ar_coefs_ = params[1 : self.p + 1]
        self.ma_coefs_ = params[self.p + 1 :]
        _check_invertible(self.ma_coefs_)
        e = _innovations(y, self.intercept_,
                         dict(enumerate(self.ar_coefs_, start=1)),
                         dict(enumerate(self.ma_coefs_, start=1)))
        self._resid = e
        self.sigma2_ = float(e @ e) / len(e)
        self.aic_, self.bic_ = aic_bic(self.sigma2_, len(e), self.p + self.q + 1)
        self._y = y
        self._fit_series = y
        return self


# --------------------------------------------------------------------------- ARIMA


class ARIMA(_UnivariateBase):
    """ARIMA(p, d, q) with d restricted to {0, 1, 2} (documented simplification).

    Fitting happens on the differenced series; forecasting integrates the differenced
    path back to levels and propagates variance through the integrated impulse response
    (exact for integer d).
    """

    is_integrated = True

    def __init__(self, p, d, q):
        super().__init__()
        self.p, self.d, self.q = int(p), int(d), int(q)
        if self.d not in (0, 1, 2):
            raise ValueError("d must be in {0, 1, 2}")
        if self.p < 0 or self.q < 0 or (self.p == 0 and self.q == 0):
            raise ValueError("need p > 0 or q > 0")
        if self.q == 0:
            self._inner = AR(self.p)
        elif self.p == 0:
            self._inner = MA(self.q)
        else:
            self._inner = ARMA(self.p, self.q)

    def fit(self, y):
        y = as_1d(y)
        self._levels_tail = list(np.asarray(y, dtype=float)[-self.d :]) if self.d else []
        w = difference(y, self.d) if self.d else y
        self._inner.fit(w)
        self.intercept_ = self._inner.intercept_
        self.ar_coefs_, self.ma_coefs_ = self._inner.ar_coefs_, self._inner.ma_coefs_
        self.sigma2_ = self._inner.sigma2_
        self.aic_, self.bic_ = self._inner.aic_, self._inner.bic_
        self._resid = self._inner.residuals()
        self._y = y
        self._fit_series = w
        return self

    def residuals(self):
        """Innovations of the *differenced* series (documented convention)."""
        return self._inner.residuals()

    def _maps(self):
        return self._inner._maps()

    def _integrated_psi(self, h):
        psi = _psi_from_maps(*self._maps(), h)
        for _ in range(self.d):  # each integration convolves with the all-ones kernel
            psi = np.cumsum(psi)
        return psi

    def _integrate_forecast(self, diff_path):
        if self.d == 0:
            return diff_path
        return integrate_levels(diff_path, self._levels_tail)


# --------------------------------------------------------------------------- SARIMA
class SARIMA(_UnivariateBase):
    """SARIMA(p, d, q)(P, D, Q)s with D restricted to {0, 1} (documented).

    Seasonal terms enter as extra lag offsets: AR/P at lags {1..p} U {s, 2s, ...} and
    likewise for MA/Q. Estimation is CSS over the full lag structure after applying
    regular then seasonal differencing (in that order).
    """

    is_integrated = True

    def __init__(self, p, d, q, P, D, Q, s):
        super().__init__()
        self.p, self.d, self.q = int(p), int(d), int(q)
        self.P, self.D, self.Q, self.s = int(P), int(D), int(Q), int(s)
        if not (0 <= self.d <= 1 and 0 <= self.D <= 1):
            raise ValueError("d and D must each be 0 or 1")
        if self.s < 1:
            raise ValueError("seasonal period s must be >= 1")
        self.ar_coefs_seasonal_ = None
        self.ma_coefs_seasonal_ = None
        self._seasonal = None

    def fit(self, y):
        y = as_1d(y)
        z = difference(y, self.d) if self.d else y
        z = seasonal_difference(z, self.s, self.D) if self.D else z
        # tails needed to undo differencing when forecasting
        self._tail_regular = list(np.asarray(y, dtype=float)[-self.d :]) if self.d else []
        pre = difference(y, self.d) if self.d else y
        self._tail_seasonal = list(pre[-self.s * self.D :]) if self.D else []

        ar_map0 = {}
        for i in range(self.p):
            ar_map0[i + 1] = 0.0
        for i in range(self.P):
            ar_map0[self.s * (i + 1)] = 0.0
        ma_map0 = {}
        for i in range(self.q):
            ma_map0[i + 1] = 0.0
        for i in range(self.Q):
            ma_map0[self.s * (i + 1)] = 0.0

        def unpack(params):
            c = params[0]
            k = 1
            ar = {}
            for i in range(self.p):
                ar[i + 1] = params[k]; k += 1
            for i in range(self.P):
                ar[self.s * (i + 1)] = params[k]; k += 1
            ma = {}
            for i in range(self.q):
                ma[i + 1] = params[k]; k += 1
            for i in range(self.Q):
                ma[self.s * (i + 1)] = params[k]; k += 1
            return c, ar, ma

        def css(params):
            c, amap, mmap = unpack(params)
            e = _innovations(z, c, amap, mmap)
            return float(e @ e)

        x0 = np.concatenate([[float(np.mean(z))], np.zeros(len(ar_map0) + len(ma_map0))])
        res = optimize.minimize(css, x0, method="Nelder-Mead",
                                options={"maxiter": 12000, "maxfev": 12000,
                                         "xatol": 1e-9, "fatol": 1e-11})
        c, amap, mmap = unpack(res.x if css(res.x) <= css(x0) else x0)
        self.intercept_ = c
        self.ar_coefs_ = np.array([amap[i + 1] for i in range(self.p)]) if self.p else np.array([])
        self.ar_coefs_seasonal_ = (
            np.array([amap[self.s * (i + 1)] for i in range(self.P)]) if self.P else np.array([])
        )
        self.ma_coefs_ = np.array([mmap[i + 1] for i in range(self.q)]) if self.q else np.array([])
        self.ma_coefs_seasonal_ = (
            np.array([mmap[self.s * (i + 1)] for i in range(self.Q)]) if self.Q else np.array([])
        )
        self._seasonal = (self.ar_coefs_seasonal_, self.ma_coefs_seasonal_, self.s)

        e = _innovations(z, c, amap, mmap)
        self._resid = e
        self.sigma2_ = float(e @ e) / len(e)
        n_par = len(ar_map0) + len(ma_map0) + 1
        from stochpylib.timeseries._utils import aic_bic as _ab

        self.aic_, self.bic_ = _ab(self.sigma2_, len(e), n_par)
        self._ar_map, self._ma_map = amap, mmap
        self._y = y
        self._fit_series = z
        return self

    def _maps(self):
        return self._ar_map, self._ma_map

    def _integrated_psi(self, h):
        psi = _psi_from_maps(self._ar_map, self._ma_map, h)
        if self.d:
            psi = np.cumsum(psi)
        if self.D:
            # (1 - B^s)^{-1} truncated: unit coefficients at multiples of s
            kernel = np.zeros(h + 2 * self.s)
            kernel[:: self.s] = 1.0
            psi = np.convolve(psi, kernel)[:h]
        return psi[:h]

    def _integrate_forecast(self, diff_path):
        out = np.asarray(diff_path, dtype=float)
        if self.D:  # inverse runs in reverse order of the forward transforms
            from collections import deque

            window = deque(self._tail_seasonal[-self.s :], maxlen=self.s)
            rec = []
            for step in out:
                nxt = step + window[0]
                window.append(nxt)
                rec.append(nxt)
            out = np.array(rec)
        if self.d:
            out = integrate_levels(out, self._tail_regular)
        return out


# --------------------------------------------------------------------------- ARFIMA


class ARFIMA(_UnivariateBase):
    """ARFIMA(p, d, q) with fixed fractional differencing parameter d (documented).

    The fractional filter is the fixed-width truncated binomial window; forecasting
    inverts it by convolving the ARMA forecast path with the inverse-filter weights, and
    forecast variance composes the two impulse responses (documented approximation).
    """

    def __init__(self, p, d, q):
        super().__init__()
        self.p, self.d, self.q = int(p), float(d), int(q)
        if abs(self.d) >= 1:
            raise ValueError("|d| must be < 1")

    def fit(self, y):
        y = as_1d(y)
        # differencing filter (1-B)^{+d}; the inverse (integration) kernel used in
        # forecasting is built separately below
        self._w = frac_diff_weights(self.d)
        L = len(self._w)
        filtered = np.convolve(y, self._w, mode="valid")
        self._filtered_history = filtered
        if self.p >= 1 and self.q >= 1:
            inner = ARMA(self.p, self.q)
        elif self.p >= 1:
            inner = AR(self.p)
        elif self.q >= 1:
            inner = MA(self.q)
        else:
            inner = None  # pure fractional noise ARFIMA(0, d, 0): constant only

        if inner is not None:
            inner.fit(filtered)
            self.intercept_ = inner.intercept_
            self.ar_coefs_, self.ma_coefs_ = inner.ar_coefs_, inner.ma_coefs_
            self.sigma2_ = inner.sigma2_
            self.aic_, self.bic_ = inner.aic_, inner.bic_
            self._resid = inner.residuals()
        else:
            mu = float(np.mean(filtered))
            resid = filtered - mu
            self.intercept_ = mu
            self.ar_coefs_, self.ma_coefs_ = np.array([]), np.array([])
            self.sigma2_ = float(resid @ resid) / len(resid)
            from stochpylib.timeseries._utils import aic_bic as _ab

            self.aic_, self.bic_ = _ab(self.sigma2_, len(resid), 1)
            self._resid = resid
        self._inner = inner
        self._y = y
        return self

    def residuals(self):
        return self._inner.residuals()

    def _maps(self):
        return self._inner._maps()

    def forecast(self, horizon=1):
        horizon = int(horizon)
        g = frac_diff_weights(-self.d)[: horizon + 1]  # inverse-filter kernel
        if self._inner is not None:
            inner_fc = self._inner.forecast(horizon)
            inner_mean = inner_fc.mean
            psi_inner = _psi_from_maps(*self._inner._maps(), horizon)
            sigma2_inner = self._inner.sigma2_
        else:
            inner_mean = np.full(horizon, self.intercept_)
            psi_inner = np.zeros(horizon)
            sigma2_inner = self.sigma2_
        ext = np.concatenate([self._filtered_history, inner_mean])
        level = np.convolve(ext, g)[len(ext) - len(g) + 1 :][:horizon]
        if self._inner is not None:
            total_psi = np.convolve(psi_inner, g)[:horizon]
        else:  # pure fractional noise: impulse response IS the inverse kernel
            total_psi = g[:horizon]
        var = sigma2_inner * np.cumsum(total_psi**2)
        return ForecastResult(level, np.sqrt(var))


# --------------------------------------------------------------------------- VAR


class VAR:
    """Vector autoregression VAR(p), estimated equation-by-equation by OLS."""

    def __init__(self, p):
        self.p = int(p)
        if self.p < 1:
            raise ValueError("p must be >= 1")

    def fit(self, Y):
        Y = np.asarray(Y, dtype=float)
        if Y.ndim != 2:
            raise ValueError("VAR requires a (T, k) matrix")
        self.k = Y.shape[1]
        T = Y.shape[0]
        Z = np.ones((T - self.p, self.k * self.p + 1))
        for j in range(1, self.p + 1):
            Z[:, 1 + (j - 1) * self.k : 1 + j * self.k] = Y[self.p - j : T - j]
        targets = Y[self.p :]
        B, *_ = np.linalg.lstsq(Z, targets, rcond=None)
        resid = targets - Z @ B
        self.coef_matrices_ = [
            B[1 + (j - 1) * self.k : 1 + j * self.k].T for j in range(1, self.p + 1)
        ]
        self.intercept_ = B[0]
        self.sigma_cov_ = resid.T @ resid / (len(targets) - self.k * self.p - 1)
        self._resid = resid
        self._Y = Y
        self.sigma2_ = float(np.trace(self.sigma_cov_) / self.k)  # aggregate scalar for AIC
        self.aic_, self.bic_ = aic_bic(
            self.sigma2_, len(targets), self.k * (self.k * self.p + 1)
        )
        return self

    def residuals(self):
        return self._resid

    def forecast(self, horizon=1):
        horizon = int(horizon)
        hist = [row.copy() for row in self._Y[-self.p :]][::-1]  # hist[0]=y_T
        means = np.empty((horizon, self.k))
        psis = [np.eye(self.k)]
        for i in range(1, horizon):
            acc = np.zeros((self.k, self.k))
            for j, A in enumerate(self.coef_matrices_, start=1):
                if i - j >= 0:
                    acc = acc + A @ psis[i - j]
            psis.append(acc)
        for h in range(horizon):
            nxt = self.intercept_.copy()
            for j, A in enumerate(self.coef_matrices_, start=1):
                if h - j >= 0:
                    nxt = nxt + A @ hist[h - j]
            hist.append(nxt)
            means[h] = nxt
        std = np.empty((horizon, self.k))
        for h in range(horizon):
            cov = np.zeros((self.k, self.k))
            for i in range(h + 1):
                P = psis[h - i]
                cov = cov + P @ self.sigma_cov_ @ P.T
            std[h] = np.sqrt(np.clip(np.diag(cov), 0, None))
        return ForecastResult(means, std)

    def simulate(self, n=500, burnin=200, random_state=None):
        rng = np.random.default_rng(random_state)
        total = int(n) + int(burnin)
        y = np.zeros((total, self.k))
        innov = rng.multivariate_normal(np.zeros(self.k), self.sigma_cov_, size=total)
        for t in range(total):
            s = self.intercept_.copy()
            for j, A in enumerate(self.coef_matrices_, start=1):
                if t - j >= 0:
                    s = s + A @ y[t - j]
            y[t] = s + innov[t]
        return y[burnin:]


# --------------------------------------------------------------------------- VARMA


class VARMA(VAR):
    """VARMA(p, q) via multivariate CSS with a diagonal innovation covariance
    (documented simplification — full VARMA MLE is intentionally out of scope)."""

    def __init__(self, p, q):
        super().__init__(p)
        self.q = int(q)
        if self.q < 0:
            raise ValueError("q must be >= 0")

    def fit(self, Y):
        Y = np.asarray(Y, dtype=float)
        if Y.ndim != 2:
            raise ValueError("VARMA requires a (T, k) matrix")
        self.k = Y.shape[1]
        T = Y.shape[0]

        def unpack(theta):
            c = theta[: self.k]
            pos = self.k
            mats = []
            for _ in range(self.p):
                mats.append(theta[pos : pos + self.k**2].reshape(self.k, self.k))
                pos += self.k**2
            thmats = []
            for _ in range(self.q):
                thmats.append(theta[pos : pos + self.k**2].reshape(self.k, self.k))
                pos += self.k**2
            return c, mats, thmats

        def css(theta):
            cc, Amats, Thmats = unpack(theta)
            e = np.zeros_like(Y)
            for t in range(T):
                s = cc.copy()
                for j, A in enumerate(Amats, start=1):
                    if t - j >= 0:
                        s = s + A @ Y[t - j]
                for j, TH in enumerate(Thmats, start=1):
                    if t - j >= 0:
                        s = s + TH @ e[t - j]
                e[t] = Y[t] - s
            return float(np.sum(e**2))

        # initialize from VAR fit
        var_fit = VAR(self.p).fit(Y)
        init = [var_fit.intercept_]
        for A in var_fit.coef_matrices_:
            init.append(A.reshape(-1))
        init.extend([np.zeros(self.k**2)] * self.q)
        x0 = np.concatenate(init)
        res = optimize.minimize(css, x0, method="Nelder-Mead",
                                options={"maxiter": 20000, "maxfev": 20000})
        theta = res.x if css(res.x) <= css(x0) else x0
        self.intercept_, self.coef_matrices_, self.ma_matrices_ = unpack(theta)

        e = np.zeros_like(Y)
        for t in range(T):
            s = self.intercept_.copy()
            for j, A in enumerate(self.coef_matrices_, start=1):
                if t - j >= 0:
                    s = s + A @ Y[t - j]
            for j, TH in enumerate(self.ma_matrices_, start=1):
                if t - j >= 0:
                    s = s + TH @ e[t - j]
            e[t] = Y[t] - s
        self._resid = e
        self.sigma_cov_ = np.diag(np.var(e, axis=0, ddof=self.k * (self.p + self.q) + 1))
        self.sigma2_ = float(np.mean(np.diag(self.sigma_cov_)))
        self._Y = Y
        self.aic_, self.bic_ = aic_bic(
            self.sigma2_, len(Y), self.k + self.k**2 * (self.p + self.q)
        )
        return self

    def forecast(self, horizon=1):
        horizon = int(horizon)
        hist = [r.copy() for r in self._Y[-max(self.p, self.q) :]][::-1]
        ehist = [r.copy() for r in self._resid[-max(self.q, 1) :]][::-1]
        means = np.empty((horizon, self.k))
        psis = [np.eye(self.k)]
        for i in range(1, horizon):
            acc = np.zeros((self.k, self.k))
            for j, A in enumerate(self.coef_matrices_, start=1):
                if i - j >= 0:
                    acc += A @ psis[i - j]
            for j, TH in enumerate(self.ma_matrices_, start=1):
                if i == j:
                    acc += TH
            psis.append(acc)
        for h in range(horizon):
            nxt = self.intercept_.copy()
            for j, A in enumerate(self.coef_matrices_, start=1):
                if h - j >= 0:
                    nxt += A @ hist[h - j]
            for j, TH in enumerate(self.ma_matrices_, start=1):
                if h - j < 0:  # only in-sample innovations enter the recursion
                    idx = len(ehist) + (h - j)
                    if 0 <= idx < len(ehist):
                        nxt += TH @ ehist[idx]
            hist.append(nxt)
            means[h] = nxt
        std = np.empty((horizon, self.k))
        for h in range(horizon):
            cov = np.zeros((self.k, self.k))
            for i in range(h + 1):
                P = psis[h - i]
                cov += P @ self.sigma_cov_ @ P.T
            std[h] = np.sqrt(np.clip(np.diag(cov), 0, None))
        return ForecastResult(means, std)


# --------------------------------------------------------------------------- VECM


class VECM:
    """Vector error-correction model via reduced-rank regression (Johansen-style).

    Fits Delta_y_t = alpha beta' y_{t-1} + sum_i Gamma_i Delta y_{t-i} + c for a given
    cointegration rank. Critical values for rank testing live in
    `timeseries.tests.johansen_test`.
    """

    def __init__(self, rank, p=1):
        self.rank = int(rank)
        self.p = max(int(p), 1)
        if self.rank < 1:
            raise ValueError("rank must be >= 1")

    def fit(self, Y):
        from stochpylib.timeseries._utils import as_2d

        Y = as_2d(Y, "VECM input")
        k = Y.shape[1]
        T = Y.shape[0]
        m = self.p  # number of lagged-difference terms
        if T - self.p - m < k + 2:
            raise ValueError("sample too short for VECM fit")
        dY = Y[1:] - Y[:-1]
        endog = dY[m:]
        levels = Y[m:-1]
        lags = np.hstack([dY[m - j : T - 1 - j] for j in range(1, m + 1)]) if m else None
        Dmat = np.hstack([np.ones((len(endog), 1))] + ([lags] if m else []))

        def resid_against(M):
            beta_ls, *_ = np.linalg.lstsq(Dmat, M, rcond=None)
            return M - Dmat @ beta_ls

        R0 = resid_against(endog)
        R1 = resid_against(levels)
        S00 = R0.T @ R0 / len(R0)
        S01 = R0.T @ R1 / len(R0)
        S10 = S01.T
        S11 = R1.T @ R1 / len(R1)
        S11_inv = np.linalg.inv(S11)
        Mmat = S11_inv @ S10 @ np.linalg.inv(S00) @ S01
        eigvals, eigvecs = np.linalg.eig(Mmat)
        order = np.argsort(eigvals.real)[::-1]
        eigvals, eigvecs = eigvals.real[order], eigvecs.real[:, order]
        beta_raw = eigvecs[:, : self.rank]
        # normalize: beta' S11 beta = I
        norm = np.linalg.inv(np.sqrt(beta_raw.T @ S11 @ beta_raw))
        self.beta_ = beta_raw @ norm
        self.alpha_ = S01 @ self.beta_
        self.eigenvalues_ = np.clip(eigvals, 0.0, 1.0)

        # unrestricted short-run dynamics: levels + lagged differences
        full_design = np.hstack([levels, lags]) if m else levels
        Cols, *_ = np.linalg.lstsq(full_design, endog, rcond=None)
        self.Pi_mat_ = Cols[:k].T
        self.gamma_matrices_ = [Cols[k + j * k : k + (j + 1) * k].T for j in range(m)]
        resid = endog - full_design @ Cols
        self._resid = resid
        self.sigma_cov_ = resid.T @ resid / (len(resid) - k * (m + 1))
        self.sigma2_ = float(np.mean(np.diag(self.sigma_cov_)))
        self._Y = Y
        self.aic_, self.bic_ = aic_bic(self.sigma2_, len(resid), k * k * (m + 1))
        return self

    def residuals(self):
        return self._resid

    def forecast(self, horizon=1):
        horizon = int(horizon)
        L = np.concatenate([self._Y, np.full((horizon, self.k), np.nan)], axis=0)
        T = len(self._Y)
        means = np.empty((horizon, self.k))
        for h in range(horizon):
            t = T + h
            step = self.Pi_mat_ @ L[t - 1]
            for j, G in enumerate(self.gamma_matrices_, start=1):
                step += G @ (L[t - j] - L[t - j - 1])
            L[t] = L[t - 1] + step
            means[h] = L[t]
        std = np.tile(np.sqrt(np.diag(self.sigma_cov_)), (horizon, 1))
        return ForecastResult(means, std)
