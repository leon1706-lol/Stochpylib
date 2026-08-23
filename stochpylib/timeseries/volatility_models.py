"""Conditional-heteroskedasticity models (GARCH family), native Gaussian-QMLE fits.

All models share the same shape::

    garch = GARCH(p=1, q=1).fit(returns)      # returns are mean-adjusted already
    vol = garch.conditional_volatility()       # sqrt of the conditional variance path
    fc = garch.forecast(horizon=10)             # predicted conditional VARIANCE path
                                                # (.mean holds sigma^2; documented)

Documented simplifications: TGARCH and GJRGARCH are aliases of one leverage core (they are
the same model in the literature); FIGARCH uses truncated fractional weights with ``d``
fixed at construction; MGARCH is constant-correlation; DCC_GARCH is the scalar two-step
DCC. Innovation distribution is Gaussian throughout (QMLE).
"""

import warnings

import numpy as np
from scipy import optimize

from stochpylib.timeseries._result import ForecastResult
from stochpylib.timeseries._utils import as_1d, frac_diff_weights


# ---------------------------------------------------------------------------
# recursion cores


def _garch_recursion(e, omega, alpha, beta):
    """sigma2_t = omega + sum a_i e^2_{t-i} + sum b_j sigma2_{t-j}, backcast = early var."""
    T = len(e)
    m = max(len(alpha), len(beta))
    backcast = float(np.mean(e[: min(T, 75)] ** 2)) if T else 0.0
    # plain scalar loops beat tiny-array dot products here (numpy call overhead)
    e2_list = (e**2).tolist()
    s2 = [backcast] * T
    alpha_l, beta_l = list(np.atleast_1d(alpha)), list(np.atleast_1d(beta))
    for t in range(m, T):
        s = omega
        for i, a in enumerate(alpha_l, start=1):
            s += a * e2_list[t - i]
        for j, b in enumerate(beta_l, start=1):
            s += b * s2[t - j]
        s2[t] = s
    return np.array(s2)


_LEVERAGE_MAXFEV_MAIN = 3500
_LEVERAGE_MAXFEV_ALT = 1800


def _egarch_recursion(e, omega, alpha, gamma, beta):
    """log sigma2_t = w + a(|z|-E|z|) + g z + b log sigma2_{t-1}; z = e/sigma."""
    T = len(e)
    Ez = np.sqrt(2 / np.pi)
    lns2 = np.full(T, np.log(np.var(e[: min(T, 75)]) if T else 1.0))
    z = np.zeros(T)
    for t in range(1, T):
        sig2_prev = np.exp(lns2[t - 1])
        z[t - 1] = e[t - 1] / np.sqrt(sig2_prev)
        lns2[t] = (
            omega
            + alpha * (abs(z[t - 1]) - Ez)
            + gamma * z[t - 1]
            + beta * lns2[t - 1]
        )
    return np.exp(lns2)


def _aparch_recursion(e, omega, alpha, gamma, beta, delta):
    """sigma^delta_t = w + sum a(|e_{t-1}| - g e_{t-1})^delta + sum b sigma^delta."""
    T = len(e)
    m = max(len(alpha), len(beta))
    backcast = float(np.mean(e[: min(T, 75)] ** 2))
    spd = np.full(T, backcast ** (delta / 2.0))
    for t in range(m, T):
        s = 0.0
        for i, a in enumerate(alpha, start=1):
            ae = abs(e[t - i])
            s += a * max(abs(ae - gamma * e[t - i]), 0.0) ** delta
        for j, b in enumerate(beta, start=1):
            s += b * spd[t - j]
        spd[t] = omega + s
    return spd ** (2.0 / delta)


def _figarch_weights(d, truncation=100):
    """Fractional weights of (1-B)^d, normalized to sum to one, truncated."""
    w = frac_diff_weights(d, tol=1e-8, max_terms=truncation)
    return w / w.sum()


class _VolatilityBase:
    """Shared fit/query plumbing: demeaning, QMLE objective, forecast/simulate."""

    def __init__(self, mean="zero"):
        if mean not in ("zero", "const"):
            raise ValueError("mean must be 'zero' or 'const'")
        self.mean_spec = mean
        self.mu_ = None
        self.params_ = {}
        self.sigma2_ = None
        self._e = None
        self.aic_ = None
        self.bic_ = None

    def _unpack(self, theta):  # subclass hook -> dict(omega, alpha[], beta[], extra)
        raise NotImplementedError

    def _variance(self, e, unpacked):  # subclass hook
        raise NotImplementedError

    def fit(self, y):
        y = as_1d(y)
        self.mu_ = float(np.mean(y)) if self.mean_spec == "const" else 0.0
        e = y - self.mu_
        x0 = self._initial_guess(e)

        def nll(theta):
            try:
                un = self._unpack(theta)
            except ValueError:
                return 1e12
            s2 = self._variance(e, un)
            if np.any(s2 <= 1e-14) or not np.all(np.isfinite(s2)):
                return 1e12
            nll_val = float(np.sum(np.log(s2) + e**2 / s2))
            pen = getattr(self, "_penalty", None)
            if pen is not None:
                nll_val += pen(un)
            return nll_val

        # multi-start: subclasses may offer alternative initializations so that
        # Nelder-Mead does not settle on a constraint-wall local optimum
        main_budget, alt_budget = getattr(self, "_fit_budget", (3500, 1800))
        starts = [x0] + list(getattr(self, "_alternative_starts", lambda e, x0: [])(e, x0))
        best_theta, best_val = x0, nll(x0)
        for si, start in enumerate(starts):
            budget = main_budget if si == 0 else min(alt_budget, main_budget)
            res = optimize.minimize(nll, start, method="Nelder-Mead",
                                    options={"maxiter": budget, "maxfev": budget,
                                             "xatol": 1e-8, "fatol": 1e-10})
            val = nll(res.x)
            if val < best_val:
                best_theta, best_val = res.x, val
        theta = best_theta
        self.theta_ = theta
        un = self._unpack(theta)
        self.params_ = un
        self.sigma2_ = self._variance(e, un)
        k = len(theta)
        T = len(e)
        ll = -0.5 * float(np.sum(np.log(self.sigma2_) + e**2 / self.sigma2_)) \
             - 0.5 * T * np.log(2 * np.pi)
        self.aic_, self.bic_ = float(-2 * ll + 2 * k), float(-2 * ll + np.log(T) * k)
        self._e = e
        return self

    def residuals(self):
        """Mean-adjusted series used in the fit."""
        if self._e is None:
            raise RuntimeError("fit() must be called first")
        return self._e

    def standardized_residuals(self):
        return self._e / np.sqrt(self.sigma2_)

    def conditional_volatility(self):
        """sqrt of the fitted conditional-variance path."""
        if self.sigma2_ is None:
            raise RuntimeError("fit() must be called first")
        return np.sqrt(self.sigma2_)

    def _forecast_variance_path(self, horizon):
        raise NotImplementedError

    def forecast(self, horizon=10):
        """Predicted conditional **variance** path (``.mean`` holds sigma^2)."""
        horizon = int(horizon)
        var = self._forecast_variance_path(horizon)
        return ForecastResult(np.asarray(var, dtype=float), np.zeros(horizon))

    def simulate(self, n=500, burnin=200, random_state=None):
        if self.sigma2_ is None:
            raise RuntimeError("fit() must be called before simulate()")
        rng = np.random.default_rng(random_state)
        total = int(n) + int(burnin)
        e = rng.standard_normal(total)
        y = np.zeros(total)
        s2 = float(np.atleast_1d(self.sigma2_)[-1])  # scalar seed from the fitted path
        for t in range(total):
            y[t] = self.mu_ + np.sqrt(max(s2, 1e-14)) * e[t]
            s2 = float(self._one_step_ahead_variance(e[t], s2, y[t]))
        return y[burnin:]

    def _one_step_ahead_variance(self, last_e, last_s2, _last_y=None):
        un = self.params_
        return un["omega"] + sum(a * last_e**2 for a in un["alpha"]) + sum(
            b * last_s2 for b in un["beta"]
        )


# --------------------------------------------------------------------------- GARCH family


class GARCH(_VolatilityBase):
    """GARCH(p, q) via Gaussian QMLE.

    sigma2_t = omega + sum_i alpha_i e_{t-i}^2 + sum_j b_j sigma2_{t-j}.
    Stationarity sum(alpha)+sum(beta) < 1 is enforced by penalty.
    """

    def __init__(self, p, q, mean="zero"):
        super().__init__(mean)
        self.p, self.q = int(p), int(q)

    def _initial_guess(self, e):
        v = float(np.var(e))
        return [v * 0.05] + [0.08] * self.q + ([0.85 / self.p] * self.p if self.p else [])

    def _unpack(self, theta):
        omega = float(theta[0])
        alpha = np.asarray(theta[1 : 1 + self.q], dtype=float)
        beta = np.asarray(theta[1 + self.q : 1 + self.q + self.p], dtype=float)
        if omega <= 1e-14 or np.any(alpha < 0) or np.any(beta < 0):
            raise ValueError("constraint violation")
        return {"omega": omega, "alpha": alpha, "beta": beta}

    def _penalty(self, un):
        pers = float(np.sum(un["alpha"]) + np.sum(un["beta"]))
        return 1e6 * max(0.0, pers - 0.9995) ** 2

    def _variance(self, e, un):
        return _garch_recursion(e, un["omega"], un["alpha"], un["beta"])

    @property
    def persistence_(self):
        return float(np.sum(self.params_["alpha"]) + np.sum(self.params_["beta"]))

    def _forecast_variance_path(self, horizon):
        un = self.params_
        e = self._e
        s2_last = self.sigma2_[-max(self.p, 1) :]
        e2_last = e[-self.q :] ** 2 if self.q else np.array([])
        path = []
        s_hist = list(s2_last)[::-1]  # newest first
        e_hist = list(e2_last)[::-1]
        pers = self.persistence_
        long_run = un["omega"] / max(1e-12, 1.0 - min(pers, 0.999999))
        # near-integrated models: future squared shocks grow with sigma^2 itself
        shock_proxy = long_run if pers < 0.999 else float(self.sigma2_[-1])
        for h in range(horizon):
            s = un["omega"]
            for i, a in enumerate(un["alpha"], start=1):
                s += a * (e_hist[i - 1] if i <= len(e_hist) else max(shock_proxy, long_run))
            for j, b in enumerate(un["beta"], start=1):
                s += b * (s_hist[j - 1] if j <= len(s_hist) else long_run)
            s_hist.insert(0, s)
            e_hist.insert(0, max(shock_proxy, long_run))
            path.append(s)
        return path


class ARCH(GARCH):
    """ARCH(q): GARCH with no autoregressive variance terms."""

    def __init__(self, q, mean="zero"):
        super().__init__(p=0, q=q, mean=mean)


class IGARCH(GARCH):
    """Integrated GARCH: unit persistence, beta_last = 1 - sum(alpha) (documented)."""

    def __init__(self, p, q, mean="zero"):
        super().__init__(p=p, q=max(q, 1), mean=mean)

    def fit(self, y):
        base = super().fit(y)
        alphas_sum = float(np.sum(base.params_["alpha"]))
        betas = base.params_["beta"].copy()
        betas[-1] = 1.0 - alphas_sum
        base.params_["beta"] = betas
        base.sigma2_ = _garch_recursion(base._e, base.params_["omega"],
                                        base.params_["alpha"], base.params_["beta"])
        return base


class _LeverageGARCH(GARCH):
    """GJR/TGARCH core: adds gamma_i e_{t-i}^2 * I(e<0) terms."""

    def __init__(self, p, q, mean="zero"):
        super().__init__(p=p, q=q, mean=mean)

    def _initial_guess(self, e):
        x0 = super()._initial_guess(e)
        # [omega, (alpha_q, gamma_q)..., beta_p]
        return [x0[0]] + [v for pair in zip(x0[1 : 1 + self.q], [0.03] * self.q)
                          for v in pair] + x0[1 + self.q :]

    def _alternative_starts(self, e, x0):
        """Second start with stronger leverage/asymmetry values to escape gamma=0 walls."""
        alt = list(x0)
        for i in range(1, 1 + 2 * self.q):
            if i % 2 == 0:  # gamma slots
                alt[i] = max(alt[i], 0.15)
        return [alt]

    def _unpack(self, theta):
        omega = float(theta[0])
        alpha, gamma, beta = [], [], []
        pos = 1
        for _ in range(self.q):
            alpha.append(float(theta[pos])); gamma.append(float(theta[pos + 1])); pos += 2
        beta = [float(v) for v in theta[pos : pos + self.p]]
        arrs = list(map(np.asarray, (alpha, gamma, beta)))
        if omega <= 1e-14 or any(np.any(a < 0) for a in arrs[:2]) or np.any(arrs[2] < 0):
            raise ValueError("constraint violation")
        return {"omega": omega, "alpha": arrs[0], "gamma": arrs[1], "beta": arrs[2]}

    def _variance(self, e, un):
        T = len(e)
        m = max(self.q, self.p)
        backcast = float(np.mean(e[: min(T, 75)] ** 2))
        e2_list = (e**2).tolist()
        neg2_list = (np.where(e < 0, e**2, 0.0)).tolist()
        s2 = [backcast] * T
        alpha_l, gamma_l = list(un["alpha"]), list(un["gamma"])
        beta_l = list(un["beta"])
        for t in range(m, T):
            s = un["omega"]
            for i in range(1, self.q + 1):
                s += alpha_l[i - 1] * e2_list[t - i] + gamma_l[i - 1] * neg2_list[t - i]
            for j in range(1, self.p + 1):
                s += beta_l[j - 1] * s2[t - j]
            s2[t] = s
        return np.array(s2)

    def _penalty(self, un):
        pers = float(np.sum(un["alpha"]) + 0.5 * np.sum(un["gamma"]) + np.sum(un["beta"]))
        return 1e6 * max(0.0, pers - 1.0) ** 2

    def _one_step_ahead_variance(self, last_e, last_s2=None, _last_y=None):
        un = self.params_
        s = un["omega"]
        ind = 1.0 if last_e < 0 else 0.0
        for a in un["alpha"]:
            s += a * last_e**2
        for g in un["gamma"]:
            s += g * last_e**2 * ind
        for b in un["beta"]:
            s += b * (last_s2 or 1.0)
        return max(s, 1e-14)

    def _forecast_variance_path(self, horizon):
        un = self.params_
        e = self._e
        s_hist = list(self.sigma2_[-self.p :])[::-1]
        e_hist = list(e[-self.q :])[::-1]
        long_run = un["omega"] / max(1e-12, 1.0 - self._penalty_free_persistence())
        path = []
        for h in range(horizon):
            s = un["omega"]
            for i, a in enumerate(un["alpha"], start=1):
                s += a * long_run
            for i, g in enumerate(un["gamma"], start=1):
                s += g * 0.5 * long_run
            for j, b in enumerate(un["beta"], start=1):
                s += b * (s_hist[j - 1] if j <= len(s_hist) else long_run)
            s_hist.insert(0, s)
            e_hist.insert(0, long_run)
            path.append(s)
        return path

    def _penalty_free_persistence(self):
        un = self.params_
        return float(np.sum(un["alpha"]) + 0.5 * np.sum(un["gamma"]) + np.sum(un["beta"]))


class TGARCH(_LeverageGARCH):
    """Threshold ARCH — identical recursion to GJR-GARCH (documented alias)."""


class GJRGARCH(_LeverageGARCH):
    """Glosten–Jagannathan–Runkle GARCH (leverage effects)."""


class EGARCH(_VolatilityBase):
    """Exponential GARCH(1, 1) in log-variance form (Nelson 1991)."""

    _fit_budget = (1400, 700)

    def __init__(self, mean="zero"):
        super().__init__(mean)

    def _initial_guess(self, e):
        return [float(np.log(np.var(e))) * 0.0, 0.15, -0.05, 0.95]

    def _unpack(self, theta):
        return {"omega": float(theta[0]), "alpha": np.array([theta[1]]),
                "gamma": np.array([theta[2]]), "beta": np.array([theta[3]])}

    def _variance(self, e, un):
        return _egarch_recursion(e, un["omega"], un["alpha"][0], un["gamma"][0], un["beta"][0])

    def _penalty(self, un):
        return 1e6 * max(0.0, abs(float(un["beta"][0])) - 0.9999) ** 2

    def _one_step_ahead_variance(self, last_e, last_s2=None, _last_y=None):
        un = self.params_
        Ez = np.sqrt(2 / np.pi)
        lns2 = np.log(max(last_s2 or self.sigma2_[-1], 1e-14))
        z = last_e / np.sqrt(max(last_s2 or self.sigma2_[-1], 1e-14))
        ln_next = (
            un["omega"]
            + un["alpha"][0] * (abs(z) - Ez)
            + un["gamma"][0] * z
            + un["beta"][0] * lns2
        )
        return float(np.exp(ln_next))

    def _forecast_variance_path(self, horizon):
        un = self.params_
        Ez = np.sqrt(2 / np.pi)
        lns2 = float(np.log(self.sigma2_[-1]))
        z_last = self._e[-1] / self.sigma2_[-1] ** 0.5
        path = []
        for h in range(horizon):
            if h == 0:
                # first step conditions on the realized innovation
                ln_next = (
                    un["omega"]
                    + un["alpha"][0] * (abs(z_last) - Ez)
                    + un["gamma"][0] * z_last
                    + un["beta"][0] * lns2
                )
            else:
                # beyond step one: E|z| = sqrt(2/pi), E[z] = 0 under normal QMLE
                ln_next = (
                    un["omega"]
                    + un["alpha"][0] * 0.0
                    + 0.0
                    + un["beta"][0] * np.log(path[-1])
                )
            path.append(float(np.exp(ln_next)))
        return path


class APARCH(_LeverageGARCH):
    """Power ARCH with asymmetry: sigma^delta_t = w + a(|e|-g e)^delta + b sigma^delta."""

    _fit_budget = (800, 450)  # the delta-power recursion is expensive per evaluation

    def __init__(self, p, q, delta_bounds=(0.3, 4.0), mean="zero"):
        super().__init__(p=p, q=q, mean=mean)
        self.delta_bounds = delta_bounds

    def _initial_guess(self, e):
        x0 = super()._initial_guess(e)
        return x0 + [2.0]

    def _unpack(self, theta):
        base = super()._unpack(theta[:-1])
        delta = float(theta[-1])
        if not (self.delta_bounds[0] <= delta <= self.delta_bounds[1]):
            raise ValueError("delta out of bounds")
        base["delta"] = delta
        return base

    def _variance(self, e, un):
        return _aparch_recursion(e, un["omega"], un["alpha"], un["gamma"][0],
                                 un["beta"], un["delta"])

    def _one_step_ahead_variance(self, last_e, last_s2=None, _last_y=None):
        un = self.params_
        delta = un["delta"]
        spd_last = max(last_s2 or 1.0, 1e-14) ** (delta / 2.0)
        s = un["omega"]
        gamma = un["gamma"][0] if len(un["gamma"]) else 0.0
        s += un["alpha"][0] * max(abs(last_e) - gamma * last_e, 0.0) ** delta
        s += un["beta"][0] * spd_last if len(un["beta"]) else 0.0
        return max(s, 1e-14) ** (2.0 / delta)

    def _penalty(self, un):
        return super()._penalty(un)


class FIGARCH(_VolatilityBase):
    """Truncated fractional-integration GARCH approximation (documented).

    sigma2_t = omega + lambda(B) e^2_t with lambda built from the fractional weights of
    (1-B)^d (d fixed at construction, truncated at `truncation` lags), scaled so that
    sum(lambda) < 1 keeps the long-run variance finite.
    """

    def __init__(self, d=0.45, truncation=100, mean="zero"):
        super().__init__(mean)
        if not (0.0 < d < 1.0):
            raise ValueError("d must be in (0, 1)")
        self.d = float(d)
        self.truncation = int(truncation)

    def _initial_guess(self, e):
        return [float(np.var(e)) * (1 - 0.3), 0.25]

    def _weights(self):
        w = frac_diff_weights(self.d, tol=1e-7, max_terms=self.truncation)
        return w / (1.0 - w.sum()) * 0.98  # keep sum(lambda) = 0.98 < 1

    def _unpack(self, theta):
        lam_total = float(theta[1])
        if not (1e-8 < lam_total < 0.999):
            raise ValueError("lambda total out of range")
        return {"longrun": float(theta[0]), "lam_total": lam_total}

    def _variance(self, e, un):
        w = self._weights()
        lam = un["lam_total"] * w
        L = len(lam)
        T = len(e)
        backcast = float(np.mean(e[: min(T, 75)] ** 2))
        # fractional weights alternate in sign; the conditional variance is floored
        # at a small positive constant (standard practice for FIGARCH variants)
        s2 = np.empty(T)
        e2_ext = np.concatenate([np.full(L, backcast), e**2])
        for t in range(T):
            raw = un["longrun"] * (1.0 - un["lam_total"]) + float(lam[::-1] @ e2_ext[t : t + L])
            s2[t] = max(raw, 1e-14)
        return s2

    def _one_step_ahead_variance(self, last_e, last_s2=None, _last_y=None):
        """FIGARCH one-step: full truncated weight window over recent squared shocks."""
        un = self.params_
        lam = self._weights() * un["lam_total"]
        e = self._e
        total = un["longrun"] * (1.0 - un["lam_total"])
        used = min(len(lam), len(e))
        for j in range(used):
            total += lam[j] * e[len(e) - 1 - j] ** 2
        return max(total, 1e-14)

    def _forecast_variance_path(self, horizon):
        un = self.params_
        lam = self._weights() * un["lam_total"]
        e2_hist = list((self._e[-len(lam) :] ** 2))[::-1]  # newest first
        long_run = un["longrun"] * (1.0 - un["lam_total"])
        path = []
        for h in range(horizon):
            s = long_run
            for i, lk in enumerate(lam, start=1):
                past_e2 = e2_hist[i - 1] if i <= len(e2_hist) else long_run
                s += lk * past_e2
            e2_hist.insert(0, long_run)
            path.append(s)
        return path


# --------------------------------------------------------------------------- multivariate


class MGARCH:
    """Constant-correlation multivariate GARCH(1, 1) (diagonal vech, documented).

    Each series gets its own scalar GARCH(1, 1); correlations are constant, estimated
    from the standardized residuals.
    """

    def __init__(self, mean="zero"):
        self.mean_spec = mean

    def fit(self, Y):
        Y = np.asarray(Y, dtype=float)
        if Y.ndim != 2:
            raise ValueError("MGARCH requires a (T, k) matrix")
        self.k = Y.shape[1]
        self.models_ = [GARCH(1, 1).fit(Y[:, j]) for j in range(self.k)]
        Z = np.column_stack([m.standardized_residuals() for m in self.models_])
        self.corr_ = np.corrcoef(Z, rowvar=False)
        self.sigma_cov_ = None
        self._Y = Y
        return self

    def conditional_covariance(self):
        """(T, k, k) conditional covariance path."""
        vols = np.column_stack([m.conditional_volatility() for m in self.models_])
        T = len(vols)
        out = np.empty((T, self.k, self.k))
        for t in range(T):
            D = np.diag(vols[t])
            out[t] = D @ self.corr_ @ D
        return out

    def forecast(self, horizon=10):
        """Covariance forecasts as (horizon, k, k) wrapped in a ForecastResult-like dict."""
        vols = np.column_stack([
            m.forecast(horizon).mean ** 0.5 for m in self.models_
        ])
        covs = np.empty((horizon, self.k, self.k))
        for h in range(horizon):
            D = np.diag(vols[h])
            covs[h] = D @ self.corr_ @ D
        return {"covariance_forecast": covs}


class DCC_GARCH(MGARCH):
    """Scalar two-step Dynamic Conditional Correlation GARCH (Engle 2002, simplified).

    Step 1: per-series GARCH(1,1). Step 2: scalar (a, b) correlation recursion on the
    standardized residuals.
    """

    def fit(self, Y):
        base = super().fit(Y)
        Z = np.column_stack([m.standardized_residuals() for m in self.models_])
        self.Qbar_ = np.cov(Z, rowvar=False)

        def dcc_nll(params):
            a, b = params
            if a < 0 or b < 0 or a + b >= 0.999:
                return 1e12
            Tn = Z.shape[0]
            Q = self.Qbar_.copy()
            nll = 0.0
            for t in range(1, Tn):
                Q = (1 - a - b) * self.Qbar_ + a * np.outer(Z[t - 1], Z[t - 1]) + b * Q
                d_inv = 1.0 / np.sqrt(np.diag(Q))
                R = d_inv[:, None] * Q * d_inv[None, :]
                sign, logdet = np.linalg.slogdet(R)
                if sign <= 0:
                    return 1e12
                quad = Z[t] @ np.linalg.solve(R, Z[t])
                nll += logdet + quad
            return 0.5 * nll

        res = optimize.minimize(dcc_nll, [0.05, 0.90], method="Nelder-Mead",
                                options={"maxiter": 250, "xatol": 1e-5})
        self.dcc_params_ = {"a": float(res.x[0]), "b": float(res.x[1])}
        # rebuild full paths with fitted params
        T = Z.shape[0]
        self.dynamic_correlations_ = np.empty((T, self.k, self.k))
        Q = self.Qbar_.copy()
        for t in range(T):
            if t > 0:
                Q = (1 - self.dcc_params_["a"] - self.dcc_params_["b"]) * self.Qbar_ \
                    + self.dcc_params_["a"] * np.outer(Z[t - 1], Z[t - 1]) \
                    + self.dcc_params_["b"] * Q
            d_inv = 1.0 / np.sqrt(np.diag(Q))
            self.dynamic_correlations_[t] = d_inv[:, None] * Q * d_inv[None, :]
        return base


__all__ = [
    "GARCH", "ARCH", "IGARCH", "TGARCH", "GJRGARCH", "EGARCH", "APARCH", "FIGARCH",
    "MGARCH", "DCC_GARCH",
]
