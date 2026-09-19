"""Short-rate and forward-rate models: Vasicek, CIR, Hull-White, Ho-Lee,
G2++, Black-Karasinski (one-factor short-rate models), LIBOR Market Model,
and Heath-Jarrow-Morton.
"""

import math

import numpy as np
from scipy import optimize

from stochpylib.financial_stochastics._common import _bs_price, _mc_result, _norm_cdf, _rng

__all__ = [
    "HullWhiteModel", "CIRProcess", "VasicekModel", "HoLeeModel",
    "LMM", "HJM", "G2ppModel", "BlackKarasinski",
]


def _jamshidian_bond_option(zcb_price_fn, sigma_p_fn, T, S, K, kind="call"):
    """Jamshidian-style bond-option price for a Gaussian one-factor model:
    ``P(T,S) N(+-h) - K P(T,T)? `` — the standard result for Vasicek/HW/Ho-Lee/G2++
    is ``h = ln(P(0,S)/(K P(0,T))) / sigma_p + sigma_p/2``."""
    P_T = zcb_price_fn(T)
    P_S = zcb_price_fn(S)
    sigma_p = sigma_p_fn(T, S)
    if sigma_p <= 0 or T <= 0:
        intrinsic = max(P_S - K * P_T, 0.0) if kind == "call" else max(K * P_T - P_S, 0.0)
        return intrinsic
    h = math.log(P_S / (K * P_T)) / sigma_p + sigma_p / 2.0
    if kind == "call":
        return P_S * _norm_cdf(h) - K * P_T * _norm_cdf(h - sigma_p)
    return K * P_T * _norm_cdf(sigma_p - h) - P_S * _norm_cdf(-h)


# --------------------------------------------------------------------------- Vasicek

class VasicekModel:
    """``dr = kappa (theta - r) dt + sigma dW``."""

    def __init__(self, r0, kappa, theta, sigma):
        if kappa <= 0 or sigma <= 0:
            raise ValueError("kappa and sigma must be positive")
        self.r0, self.kappa, self.theta, self.sigma = float(r0), float(kappa), float(theta), float(sigma)

    def mean(self, t):
        t = np.asarray(t, dtype=float)
        return self.theta + (self.r0 - self.theta) * np.exp(-self.kappa * t)

    def variance(self, t):
        t = np.asarray(t, dtype=float)
        return (self.sigma**2 / (2.0 * self.kappa)) * (1.0 - np.exp(-2.0 * self.kappa * t))

    def zcb_price(self, T, t=0.0, r=None):
        tau = T - t
        r = self.r0 if r is None else r
        kappa, theta, sigma = self.kappa, self.theta, self.sigma
        B = (1.0 - math.exp(-kappa * tau)) / kappa
        A = math.exp((theta - sigma**2 / (2.0 * kappa**2)) * (B - tau) - sigma**2 * B**2 / (4.0 * kappa))
        return A * math.exp(-B * r)

    def zero_rate(self, T):
        return -math.log(self.zcb_price(T)) / T if T > 0 else self.r0

    def yield_curve(self, maturities):
        return np.array([self.zero_rate(T) for T in maturities])

    def simulate(self, T, N=252, n_paths=1, random_state=None):
        rng = _rng(random_state)
        dt = T / N
        kappa, theta, sigma = self.kappa, self.theta, self.sigma
        ekt = math.exp(-kappa * dt)
        std = sigma * math.sqrt((1.0 - ekt**2) / (2.0 * kappa)) if kappa > 1e-10 else sigma * math.sqrt(dt)
        r = np.empty((n_paths, N + 1))
        r[:, 0] = self.r0
        for k in range(N):
            Z = rng.standard_normal(n_paths)
            r[:, k + 1] = theta + (r[:, k] - theta) * ekt + std * Z
        return r

    def fit(self, rates, dt):
        rates = np.asarray(rates, dtype=float)
        x, y = rates[:-1], rates[1:]
        A = np.column_stack([np.ones_like(x), x])
        coef, *_ = np.linalg.lstsq(A, y, rcond=None)
        c, phi = coef
        phi = np.clip(phi, 1e-8, 1 - 1e-12)
        kappa = -math.log(phi) / dt
        theta = c / (1.0 - phi)
        resid = y - (c + phi * x)
        sigma = float(np.std(resid, ddof=2)) * math.sqrt(2.0 * kappa / (1.0 - phi**2))
        self.kappa, self.theta, self.sigma = float(kappa), float(theta), float(sigma)
        self.r0 = float(rates[-1])
        return self

    def _sigma_p(self, T, S):
        return self.sigma * math.sqrt((1.0 - math.exp(-2.0 * self.kappa * T)) / (2.0 * self.kappa)) \
            * (1.0 - math.exp(-self.kappa * (S - T))) / self.kappa

    def bond_option_price(self, T, S, K, kind="call"):
        return _jamshidian_bond_option(self.zcb_price, self._sigma_p, T, S, K, kind)


# --------------------------------------------------------------------------- CIR

class CIRProcess:
    """``dr = kappa (theta - r) dt + sigma sqrt(r) dW``."""

    def __init__(self, r0, kappa, theta, sigma):
        if r0 < 0 or kappa <= 0 or theta <= 0 or sigma <= 0:
            raise ValueError("r0 >= 0 and kappa, theta, sigma > 0 required")
        self.r0, self.kappa, self.theta, self.sigma = float(r0), float(kappa), float(theta), float(sigma)

    def feller_condition(self):
        return 2.0 * self.kappa * self.theta >= self.sigma**2

    def mean(self, t):
        t = np.asarray(t, dtype=float)
        return self.theta + (self.r0 - self.theta) * np.exp(-self.kappa * t)

    def variance(self, t):
        t = np.asarray(t, dtype=float)
        kappa, theta, sigma, r0 = self.kappa, self.theta, self.sigma, self.r0
        ekt = np.exp(-self.kappa * t)
        return (r0 * sigma**2 / kappa) * (ekt - np.exp(-2 * kappa * t)) \
            + (theta * sigma**2 / (2 * kappa)) * (1 - ekt) ** 2

    def zcb_price(self, T, t=0.0, r=None):
        tau = T - t
        r = self.r0 if r is None else r
        kappa, theta, sigma = self.kappa, self.theta, self.sigma
        gamma = math.sqrt(kappa**2 + 2.0 * sigma**2)
        denom = (gamma + kappa) * (math.exp(gamma * tau) - 1.0) + 2.0 * gamma
        B = 2.0 * (math.exp(gamma * tau) - 1.0) / denom
        A = (2.0 * gamma * math.exp((gamma + kappa) * tau / 2.0) / denom) ** (2.0 * kappa * theta / sigma**2)
        return A * math.exp(-B * r)

    def zero_rate(self, T):
        return -math.log(self.zcb_price(T)) / T if T > 0 else self.r0

    def yield_curve(self, maturities):
        return np.array([self.zero_rate(T) for T in maturities])

    def simulate(self, T, N=252, n_paths=1, random_state=None, scheme="exact"):
        rng = _rng(random_state)
        dt = T / N
        kappa, theta, sigma = self.kappa, self.theta, self.sigma
        r = np.empty((n_paths, N + 1))
        r[:, 0] = self.r0
        if scheme == "exact":
            ekt = math.exp(-kappa * dt)
            d = 4.0 * kappa * theta / sigma**2
            for k in range(N):
                r_pos = np.maximum(r[:, k], 0.0)
                nc = 4.0 * kappa * ekt * r_pos / (sigma**2 * (1.0 - ekt))
                scale = sigma**2 * (1.0 - ekt) / (4.0 * kappa)
                r[:, k + 1] = scale * rng.noncentral_chisquare(d, nc)
        elif scheme == "full_truncation":
            sqrt_dt = math.sqrt(dt)
            for k in range(N):
                r_pos = np.maximum(r[:, k], 0.0)
                Z = rng.standard_normal(n_paths)
                r[:, k + 1] = r[:, k] + kappa * (theta - r_pos) * dt + sigma * np.sqrt(r_pos) * sqrt_dt * Z
        else:
            raise ValueError("scheme must be 'exact' or 'full_truncation'")
        return r

    def fit(self, rates, dt):
        rates = np.asarray(rates, dtype=float)
        r = rates[:-1]
        dr = np.diff(rates)
        r_safe = np.maximum(r, 1e-8)
        y = dr / np.sqrt(r_safe)
        X = np.column_stack([dt / np.sqrt(r_safe), np.sqrt(r_safe) * dt])
        coef, *_ = np.linalg.lstsq(X, y, rcond=None)
        a, b = coef
        kappa = -b
        theta = a / kappa if kappa > 1e-10 else r.mean()
        resid = y - X @ coef
        sigma = float(np.std(resid, ddof=2) / math.sqrt(dt))
        self.kappa, self.theta, self.sigma = float(abs(kappa)), float(abs(theta)), float(abs(sigma))
        self.r0 = float(rates[-1])
        return self


# --------------------------------------------------------------------------- Hull-White

class HullWhiteModel:
    """``dr = (theta(t) - kappa r) dt + sigma dW`` (extended Vasicek).

    Two modes: constant ``theta`` (delegates its closed forms to
    :class:`VasicekModel`), or curve-fitted ``theta(t)`` calibrated to match
    an input discount curve exactly (``fit``/``discount_curve``).
    """

    def __init__(self, kappa, sigma, r0=None, theta=None, discount_curve=None):
        if kappa <= 0 or sigma <= 0:
            raise ValueError("kappa and sigma must be positive")
        self.kappa, self.sigma = float(kappa), float(sigma)
        self.discount_curve_ = discount_curve
        if discount_curve is not None:
            self.r0 = float(r0) if r0 is not None else -self._log_curve_deriv(1e-6)
            self._const_theta = None
        elif theta is None or r0 is None:
            # unfitted: fit(maturities, discount_factors) must supply the curve before use
            self.r0 = None if r0 is None else float(r0)
            self._const_theta = None
        else:
            self.r0 = float(r0)
            self._const_theta = float(theta)

    def _require_curve(self):
        if self.discount_curve_ is None and self._const_theta is None:
            raise RuntimeError("no term structure yet: construct with r0 and theta, or with a "
                               "discount_curve, or call fit(maturities, discount_factors)")

    def _log_curve_deriv(self, t, h=1e-4):
        P = self.discount_curve_
        t = max(t, h)
        return (math.log(P(t + h)) - math.log(P(max(t - h, 1e-8)))) / (h + (t - max(t - h, 1e-8)))

    def forward_rate(self, t):
        if self.discount_curve_ is None:
            raise RuntimeError("forward_rate requires a discount_curve")
        return -self._log_curve_deriv(t)

    def theta(self, t):
        self._require_curve()
        if self.discount_curve_ is None:
            return self._const_theta
        h = 1e-4
        f_p, f_m = self.forward_rate(t + h), self.forward_rate(max(t - h, 1e-8))
        df_dt = (f_p - f_m) / (h + (t - max(t - h, 1e-8)))
        f_t = self.forward_rate(t)
        return df_dt + self.kappa * f_t + self.sigma**2 / (2.0 * self.kappa) * (1.0 - math.exp(-2.0 * self.kappa * t))

    def fit(self, maturities, discount_factors):
        maturities = np.asarray(maturities, dtype=float)
        discount_factors = np.asarray(discount_factors, dtype=float)
        log_df = np.log(discount_factors)

        def curve(t):
            if t <= maturities[0]:
                return math.exp(log_df[0] * t / maturities[0])
            if t >= maturities[-1]:
                slope = (log_df[-1] - log_df[-2]) / (maturities[-1] - maturities[-2])
                return math.exp(log_df[-1] + slope * (t - maturities[-1]))
            return math.exp(float(np.interp(t, maturities, log_df)))

        self.discount_curve_ = curve
        self.r0 = -self._log_curve_deriv(1e-4)
        self._const_theta = None
        return self

    def zcb_price(self, T, t=0.0, r=None):
        self._require_curve()
        kappa, sigma = self.kappa, self.sigma
        r = self.r0 if r is None else r
        B = (1.0 - math.exp(-kappa * (T - t))) / kappa
        if self.discount_curve_ is not None:
            P0T = self.discount_curve_(T)
            P0t = self.discount_curve_(t) if t > 0 else 1.0
            f0t = self.forward_rate(t) if t > 0 else self.forward_rate(1e-6)
            A_term = math.log(P0T / P0t) + B * f0t \
                - sigma**2 / (4.0 * kappa) * (1.0 - math.exp(-2.0 * kappa * t)) * B**2
            return math.exp(A_term - B * r)
        # HW's constant-theta convention is dr=(theta-kappa r)dt+sigma dW, so
        # Vasicek's mean-reversion LEVEL is theta/kappa, not theta itself.
        v = VasicekModel(self.r0, kappa, self._const_theta / kappa, sigma)
        return v.zcb_price(T, t, r)

    def zero_rate(self, T):
        return -math.log(self.zcb_price(T)) / T if T > 0 else self.r0

    def yield_curve(self, maturities):
        return np.array([self.zero_rate(T) for T in maturities])

    def simulate(self, T, N=252, n_paths=1, random_state=None):
        rng = _rng(random_state)
        dt = T / N
        kappa, sigma = self.kappa, self.sigma
        ekt = math.exp(-kappa * dt)
        std = sigma * math.sqrt((1.0 - ekt**2) / (2.0 * kappa))
        r = np.empty((n_paths, N + 1))
        r[:, 0] = self.r0
        t = 0.0
        for k in range(N):
            th_mid = self.theta(t + dt / 2.0)
            Z = rng.standard_normal(n_paths)
            r[:, k + 1] = r[:, k] * ekt + (th_mid / kappa) * (1.0 - ekt) + std * Z
            t += dt
        return r

    def bond_option_price(self, T, S, K, kind="call"):
        def sigma_p(t, s):
            return self.sigma * math.sqrt((1.0 - math.exp(-2.0 * self.kappa * t)) / (2.0 * self.kappa)) \
                * (1.0 - math.exp(-self.kappa * (s - t))) / self.kappa
        return _jamshidian_bond_option(self.zcb_price, sigma_p, T, S, K, kind)


# --------------------------------------------------------------------------- Ho-Lee

class HoLeeModel:
    """``dr = theta(t) dt + sigma dW``."""

    def __init__(self, sigma, r0=None, theta=None, discount_curve=None):
        if sigma <= 0:
            raise ValueError("sigma must be positive")
        self.sigma = float(sigma)
        self.discount_curve_ = discount_curve
        if discount_curve is not None:
            self.r0 = float(r0) if r0 is not None else self._forward(1e-6)
            self._const_theta = None
        elif theta is None or r0 is None:
            # unfitted: fit(maturities, discount_factors) must supply the curve before use
            self.r0 = None if r0 is None else float(r0)
            self._const_theta = None
        else:
            self.r0 = float(r0)
            self._const_theta = float(theta)

    def _require_curve(self):
        if self.discount_curve_ is None and self._const_theta is None:
            raise RuntimeError("no term structure yet: construct with r0 and theta, or with a "
                               "discount_curve, or call fit(maturities, discount_factors)")

    def _forward(self, t, h=1e-4):
        P = self.discount_curve_
        t = max(t, h)
        return -(math.log(P(t + h)) - math.log(P(max(t - h, 1e-8)))) / (h + (t - max(t - h, 1e-8)))

    def theta(self, t):
        self._require_curve()
        if self.discount_curve_ is None:
            return self._const_theta
        h = 1e-4
        f_p, f_m = self._forward(t + h), self._forward(max(t - h, 1e-8))
        return (f_p - f_m) / (h + (t - max(t - h, 1e-8))) + self.sigma**2 * t

    def fit(self, maturities, discount_factors):
        maturities = np.asarray(maturities, dtype=float)
        discount_factors = np.asarray(discount_factors, dtype=float)
        log_df = np.log(discount_factors)

        def curve(t):
            if t <= maturities[0]:
                return math.exp(log_df[0] * t / maturities[0])
            if t >= maturities[-1]:
                slope = (log_df[-1] - log_df[-2]) / (maturities[-1] - maturities[-2])
                return math.exp(log_df[-1] + slope * (t - maturities[-1]))
            return math.exp(float(np.interp(t, maturities, log_df)))

        self.discount_curve_ = curve
        self.r0 = self._forward(1e-4)
        self._const_theta = None
        return self

    def zcb_price(self, T, t=0.0, r=None):
        self._require_curve()
        r = self.r0 if r is None else r
        tau = T - t
        sigma = self.sigma
        if self.discount_curve_ is not None:
            P0T = self.discount_curve_(T)
            P0t = self.discount_curve_(t) if t > 0 else 1.0
            f0t = self._forward(t) if t > 0 else self._forward(1e-6)
            ln_A = math.log(P0T / P0t) + tau * f0t - 0.5 * sigma**2 * t * tau**2
            return math.exp(ln_A - tau * r)
        ln_P = -r * tau - self._const_theta * tau**2 / 2.0 + sigma**2 * tau**3 / 6.0
        return math.exp(ln_P)

    def zero_rate(self, T):
        return -math.log(self.zcb_price(T)) / T if T > 0 else self.r0

    def yield_curve(self, maturities):
        return np.array([self.zero_rate(T) for T in maturities])

    def simulate(self, T, N=252, n_paths=1, random_state=None):
        rng = _rng(random_state)
        dt = T / N
        sqrt_dt = math.sqrt(dt)
        r = np.empty((n_paths, N + 1))
        r[:, 0] = self.r0
        t = 0.0
        for k in range(N):
            Z = rng.standard_normal(n_paths)
            r[:, k + 1] = r[:, k] + self.theta(t + dt / 2.0) * dt + self.sigma * sqrt_dt * Z
            t += dt
        return r

    def bond_option_price(self, T, S, K, kind="call"):
        def sigma_p(t, s):
            return self.sigma * (s - t) * math.sqrt(t)
        return _jamshidian_bond_option(self.zcb_price, sigma_p, T, S, K, kind)


# --------------------------------------------------------------------------- G2++

class G2ppModel:
    """Brigo-Mercurio two-factor Gaussian model: ``r_t = x_t + y_t + phi(t)``,
    ``dx = -a x dt + sigma dW1``, ``dy = -b y dt + eta dW2``, ``corr = rho``."""

    def __init__(self, a, b, sigma, eta, rho, r0=0.03, discount_curve=None):
        if a <= 0 or b <= 0 or sigma <= 0 or eta <= 0:
            raise ValueError("a, b, sigma, eta must be positive")
        if not -1.0 <= rho <= 1.0:
            raise ValueError("rho must lie in [-1, 1]")
        self.a, self.b, self.sigma, self.eta, self.rho = float(a), float(b), float(sigma), float(eta), float(rho)
        self.r0 = float(r0)
        self.discount_curve_ = discount_curve or (lambda t: math.exp(-r0 * t))

    def _V(self, t, T):
        a, b, sigma, eta, rho = self.a, self.b, self.sigma, self.eta, self.rho
        tau = T - t
        term1 = sigma**2 / a**2 * (tau + 2.0 / a * math.exp(-a * tau) - 1.0 / (2.0 * a) * math.exp(-2.0 * a * tau) - 3.0 / (2.0 * a))
        term2 = eta**2 / b**2 * (tau + 2.0 / b * math.exp(-b * tau) - 1.0 / (2.0 * b) * math.exp(-2.0 * b * tau) - 3.0 / (2.0 * b))
        term3 = 2.0 * rho * sigma * eta / (a * b) * (
            tau + (math.exp(-a * tau) - 1.0) / a + (math.exp(-b * tau) - 1.0) / b
            - (math.exp(-(a + b) * tau) - 1.0) / (a + b))
        return term1 + term2 + term3

    def zcb_price(self, t, T=None, x=0.0, y=0.0):
        if T is None:
            T, t = t, 0.0
        P0T, P0t = self.discount_curve_(T), (self.discount_curve_(t) if t > 0 else 1.0)
        B_a = (1.0 - math.exp(-self.a * (T - t))) / self.a
        B_b = (1.0 - math.exp(-self.b * (T - t))) / self.b
        expo = 0.5 * (self._V(t, T) - self._V(0.0, T) + self._V(0.0, t)) - B_a * x - B_b * y
        return (P0T / P0t) * math.exp(expo)

    def zero_rate(self, T):
        return -math.log(self.zcb_price(0.0, T)) / T if T > 0 else self.r0

    def yield_curve(self, maturities):
        return np.array([self.zero_rate(T) for T in maturities])

    def simulate(self, T, N=252, n_paths=1, random_state=None):
        rng = _rng(random_state)
        dt = T / N
        a, b, sigma, eta, rho = self.a, self.b, self.sigma, self.eta, self.rho
        eat, ebt = math.exp(-a * dt), math.exp(-b * dt)
        var_x = sigma**2 * (1.0 - eat**2) / (2.0 * a)
        var_y = eta**2 * (1.0 - ebt**2) / (2.0 * b)
        cov_xy = rho * sigma * eta * (1.0 - math.exp(-(a + b) * dt)) / (a + b)
        cov = np.array([[var_x, cov_xy], [cov_xy, var_y]])
        L = np.linalg.cholesky(cov + 1e-16 * np.eye(2))

        x = np.zeros(n_paths)
        y = np.zeros(n_paths)
        x_hist = np.empty((n_paths, N + 1))
        y_hist = np.empty((n_paths, N + 1))
        r_hist = np.empty((n_paths, N + 1))
        x_hist[:, 0] = x
        y_hist[:, 0] = y
        r_hist[:, 0] = self.r0
        for k in range(N):
            Z = rng.standard_normal((n_paths, 2)) @ L.T
            x = x * eat + Z[:, 0]
            y = y * ebt + Z[:, 1]
            x_hist[:, k + 1] = x
            y_hist[:, k + 1] = y
            t = (k + 1) * dt
            f0t = -(math.log(self.discount_curve_(t + 1e-4)) - math.log(self.discount_curve_(max(t - 1e-4, 1e-8)))) / 2e-4
            r_hist[:, k + 1] = x + y + f0t
        return r_hist, x_hist, y_hist

    def zcb_option_price(self, T, S, K, kind="call"):
        a, b, sigma, eta, rho = self.a, self.b, self.sigma, self.eta, self.rho
        Sigma2 = (
            sigma**2 / (2.0 * a**3) * (1.0 - math.exp(-a * (S - T))) ** 2 * (1.0 - math.exp(-2.0 * a * T))
            + eta**2 / (2.0 * b**3) * (1.0 - math.exp(-b * (S - T))) ** 2 * (1.0 - math.exp(-2.0 * b * T))
            + 2.0 * rho * sigma * eta / (a * b * (a + b)) * (1.0 - math.exp(-a * (S - T)))
            * (1.0 - math.exp(-b * (S - T))) * (1.0 - math.exp(-(a + b) * T))
        )
        Sigma = math.sqrt(max(Sigma2, 0.0))
        P_T, P_S = self.zcb_price(0.0, T), self.zcb_price(0.0, S)
        if Sigma <= 0 or T <= 0:
            return max(P_S - K * P_T, 0.0) if kind == "call" else max(K * P_T - P_S, 0.0)
        h = (1.0 / Sigma) * math.log(P_S / (K * P_T)) + Sigma / 2.0
        if kind == "call":
            return P_S * _norm_cdf(h) - K * P_T * _norm_cdf(h - Sigma)
        return K * P_T * _norm_cdf(Sigma - h) - P_S * _norm_cdf(-h)


# --------------------------------------------------------------------------- Black-Karasinski

class BlackKarasinski:
    """``d ln r = kappa (ln theta - ln r) dt + sigma dW`` — no closed-form ZCB;
    priced by Monte Carlo (log-space exact OU simulation)."""

    def __init__(self, r0, kappa, theta, sigma):
        if r0 <= 0 or kappa <= 0 or theta <= 0 or sigma <= 0:
            raise ValueError("r0, kappa, theta, sigma must be positive")
        self.r0, self.kappa, self.theta, self.sigma = float(r0), float(kappa), float(theta), float(sigma)

    def simulate(self, T, N=252, n_paths=1, random_state=None):
        rng = _rng(random_state)
        dt = T / N
        kappa, sigma = self.kappa, self.sigma
        ln_theta = math.log(self.theta)
        ekt = math.exp(-kappa * dt)
        std = sigma * math.sqrt((1.0 - ekt**2) / (2.0 * kappa)) if kappa > 1e-10 else sigma * math.sqrt(dt)
        x = np.empty((n_paths, N + 1))
        x[:, 0] = math.log(self.r0)
        for k in range(N):
            Z = rng.standard_normal(n_paths)
            x[:, k + 1] = ln_theta + (x[:, k] - ln_theta) * ekt + std * Z
        return np.exp(x)

    def mean(self, t):
        t = float(t)
        kappa, sigma = self.kappa, self.sigma
        ln_theta = math.log(self.theta)
        ekt = math.exp(-kappa * t)
        E_x = ln_theta + (math.log(self.r0) - ln_theta) * ekt
        Var_x = sigma**2 * (1.0 - ekt**2) / (2.0 * kappa) if kappa > 1e-10 else sigma**2 * t
        return math.exp(E_x + 0.5 * Var_x)

    def zcb_price(self, T, N=100, n_paths=100_000, random_state=None):
        r = self.simulate(T, N=N, n_paths=n_paths, random_state=random_state)
        dt = T / N
        integral = np.trapezoid(r, dx=dt, axis=1) if hasattr(np, "trapezoid") \
            else np.trapz(r, dx=dt, axis=1)
        discount = np.exp(-integral)
        return _mc_result(discount, "bk-mc-discount")


# --------------------------------------------------------------------------- LMM

class LMM:
    """LIBOR Market Model: log-Euler simulation of forward rates under the
    terminal/spot measure with a correlation matrix."""

    def __init__(self, forwards, tau, vols, corr=None, beta=None):
        self.forwards = np.asarray(forwards, dtype=float)
        self.tau = np.asarray(tau, dtype=float)
        self.vols = np.asarray(vols, dtype=float)
        n = len(self.forwards)
        if corr is not None:
            self.corr = np.asarray(corr, dtype=float)
        elif beta is not None:
            idx = np.arange(n)
            self.corr = np.exp(-beta * np.abs(idx[:, None] - idx[None, :]))
        else:
            self.corr = np.eye(n)
        self.n = n
        self._maturities = np.concatenate([[0.0], np.cumsum(self.tau)])

    def discount_factors(self):
        return np.cumprod(1.0 / (1.0 + self.tau * self.forwards))

    def simulate(self, n_paths, substeps=4, random_state=None):
        """Spot-measure log-Euler simulation over the tenor structure.

        Forward ``F_j`` is stochastic until its own reset date ``T_j`` and
        frozen (constant) thereafter — the output at step ``j*substeps``
        (time ``T_j``) is therefore its realized fixing. Drift during period
        ``p = [T_p, T_{p+1})`` runs over the still-live forwards
        ``j >= p+1``, summing correlated forwards ``k`` from ``p+1`` to ``j``
        (the standard spot-LIBOR-measure LMM drift).
        """
        rng = _rng(random_state)
        n = self.n
        L = np.linalg.cholesky(self.corr + 1e-12 * np.eye(n))
        F = np.tile(self.forwards, (n_paths, 1)).astype(float)
        out = np.empty((n_paths, n * substeps + 1, n))
        out[:, 0, :] = F
        step_idx = 1
        for p in range(n):
            dt = self.tau[p] / substeps
            sqrt_dt = math.sqrt(dt)
            live = list(range(p + 1, n))
            for _ in range(substeps):
                Z = rng.standard_normal((n_paths, n)) @ L.T
                for j in live:
                    acc = np.zeros(n_paths)
                    for k in range(p + 1, j + 1):
                        acc += (self.corr[j, k] * self.tau[k] * self.vols[k] * F[:, k]
                               / (1.0 + self.tau[k] * F[:, k]))
                    drift = self.vols[j] * acc
                    F[:, j] = F[:, j] * np.exp((drift - 0.5 * self.vols[j] ** 2) * dt
                                               + self.vols[j] * sqrt_dt * Z[:, j])
                out[:, step_idx, :] = F
                step_idx += 1
        return out

    def caplet_price_black(self, i, K):
        F_i = self.forwards[i]
        T_i = self._maturities[i]
        vol = self.vols[i]
        disc = 1.0
        for k in range(i + 1):
            disc *= 1.0 / (1.0 + self.tau[k] * self.forwards[k])
        return self.tau[i] * disc * _bs_price(F_i, K, T_i, 0.0, vol, 0.0, "call")

    def caplet_price_mc(self, i, K, n_paths=50_000, substeps=4, random_state=None):
        paths = self.simulate(n_paths, substeps, random_state)
        F_iT = paths[:, i * substeps, i]
        payoff = self.tau[i] * np.maximum(F_iT - K, 0.0)
        numeraire = np.ones(n_paths)
        for k in range(i + 1):
            F_kT = paths[:, k * substeps, k]
            numeraire *= (1.0 + self.tau[k] * F_kT)
        disc_payoff = payoff / numeraire
        return _mc_result(disc_payoff, "lmm-caplet-mc")


# --------------------------------------------------------------------------- HJM

class HJM:
    """Gaussian one-factor Heath-Jarrow-Morton in the Musiela parametrization
    ``df(t,x) = [sigma(x) int_0^x sigma(y) dy] dt + sigma(x) dW``, ``x = T - t``."""

    def __init__(self, forward_curve, vol_fn, x_max, N):
        self.forward_curve = forward_curve
        self.vol_fn = vol_fn
        self.x_max = float(x_max)
        self.N = int(N)
        self.dx = self.x_max / self.N

    def _initial_curve(self):
        x_grid = np.linspace(0.0, self.x_max, self.N + 1)
        return x_grid, np.array([self.forward_curve(x) for x in x_grid])

    def simulate(self, T, n_paths=1, random_state=None):
        rng = _rng(random_state)
        dx = self.dx
        dt = dx
        n_steps = int(round(T / dt))
        x_grid, f0 = self._initial_curve()
        sigma = np.array([self.vol_fn(x) for x in x_grid])
        cum_sigma_int = np.concatenate([[0.0], np.cumsum(0.5 * (sigma[:-1] + sigma[1:]) * dx)])
        drift = sigma * cum_sigma_int

        f = np.tile(f0, (n_paths, 1))
        out = np.empty((n_paths, n_steps + 1, self.N + 1))
        out[:, 0, :] = f
        sqrt_dt = math.sqrt(dt)
        for k in range(n_steps):
            Z = rng.standard_normal(n_paths)
            f_new = np.empty_like(f)
            f_new[:, :-1] = f[:, 1:] + drift[1:] * dt + sigma[1:] * sqrt_dt * Z[:, None]
            f_new[:, -1] = f_new[:, -2] + (f_new[:, -2] - f_new[:, -3]) if self.N >= 2 else f_new[:, -2]
            f = f_new
            out[:, k + 1, :] = f
        return out

    def short_rate(self, paths):
        return paths[:, :, 0]

    def zcb_from_curve(self, f_row, tau):
        x_grid = np.linspace(0.0, self.x_max, self.N + 1)
        idx = int(round(tau / self.dx))
        idx = min(max(idx, 0), self.N)
        integral = np.trapezoid(f_row[: idx + 1], dx=self.dx) if hasattr(np, "trapezoid") \
            else np.trapz(f_row[: idx + 1], dx=self.dx)
        return math.exp(-integral)

    def zcb_price(self, T):
        _, f0 = self._initial_curve()
        return self.zcb_from_curve(f0, T)

    def zcb_price_mc(self, T, n_paths=10_000, random_state=None):
        dt = self.dx
        n_steps = int(round(T / dt))
        paths = self.simulate(T, n_paths=n_paths, random_state=random_state)
        r = self.short_rate(paths)[:, : n_steps + 1]
        integral = np.trapezoid(r, dx=dt, axis=1) if hasattr(np, "trapezoid") \
            else np.trapz(r, dx=dt, axis=1)
        discount = np.exp(-integral)
        return _mc_result(discount, "hjm-mc-discount")
