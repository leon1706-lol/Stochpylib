"""European/American option pricing: closed form, lattice, Monte Carlo, and
Fourier-transform methods, all sharing the Black-76 kernel in ``_common``.
"""

import math

import numpy as np
from scipy import optimize

from stochpylib.financial_stochastics import greeks as _greeks
from stochpylib.financial_stochastics._common import (
    _bs_d1d2,
    _bs_price,
    _check_kind,
    _implied_vol,
    _mc_result,
    _norm_cdf,
    _rng,
)
from stochpylib.levy_processes.jump_diffusion import carr_madan_call
from stochpylib.montecarlo.quasi_random import SobolSequence

__all__ = [
    "BlackScholes", "BlackScholes_American", "BinomialTree", "TrinomialTree",
    "MonteCarloOptionPricing", "LongstaffSchwartz", "FourierOptionPricing",
]


class BlackScholes:
    """Closed-form European option price and Greeks.

    ``bs = BlackScholes(S=100, K=105, T=1, r=0.05, sigma=0.2)``;
    ``bs.call_price()``; ``bs.Delta``, ``bs.Gamma``, ... (call-side Greeks
    exposed as properties, matching the vault quickstart example).
    """

    def __init__(self, S, K, T, r, sigma, q=0.0):
        self.S, self.K, self.T = float(S), float(K), float(T)
        self.r, self.sigma, self.q = float(r), float(sigma), float(q)

    def price(self, kind="call"):
        return float(_bs_price(self.S, self.K, self.T, self.r, self.sigma, self.q, kind))

    def call_price(self):
        return self.price("call")

    def put_price(self):
        return self.price("put")

    @property
    def d1(self):
        d1, _ = _bs_d1d2(self.S, self.K, self.T, self.r, self.sigma, self.q)
        return float(d1)

    @property
    def d2(self):
        _, d2 = _bs_d1d2(self.S, self.K, self.T, self.r, self.sigma, self.q)
        return float(d2)

    def delta(self, kind="call"):
        return float(_greeks.Delta(self.S, self.K, self.T, self.r, self.sigma, self.q, kind))

    def theta(self, kind="call"):
        return float(_greeks.Theta(self.S, self.K, self.T, self.r, self.sigma, self.q, kind))

    def rho(self, kind="call"):
        return float(_greeks.Rho(self.S, self.K, self.T, self.r, self.sigma, self.q, kind))

    def greeks(self, kind="call"):
        args = (self.S, self.K, self.T, self.r, self.sigma, self.q, kind)
        return {
            "delta": float(_greeks.Delta(*args)), "gamma": float(_greeks.Gamma(*args)),
            "vega": float(_greeks.Vega(*args)), "theta": float(_greeks.Theta(*args)),
            "rho": float(_greeks.Rho(*args)), "vanna": float(_greeks.Vanna(*args)),
            "volga": float(_greeks.Volga(*args)),
        }

    def implied_vol(self, price, kind="call"):
        return _implied_vol(price, self.S, self.K, self.T, self.r, self.q, kind)

    @property
    def Delta(self):
        return self.delta("call")

    @property
    def Gamma(self):
        return float(_greeks.Gamma(self.S, self.K, self.T, self.r, self.sigma, self.q, "call"))

    @property
    def Vega(self):
        return float(_greeks.Vega(self.S, self.K, self.T, self.r, self.sigma, self.q, "call"))

    @property
    def Theta(self):
        return self.theta("call")

    @property
    def Rho(self):
        return self.rho("call")

    @property
    def Vanna(self):
        return float(_greeks.Vanna(self.S, self.K, self.T, self.r, self.sigma, self.q, "call"))

    @property
    def Volga(self):
        return float(_greeks.Volga(self.S, self.K, self.T, self.r, self.sigma, self.q, "call"))


class BlackScholes_American:
    """American option price via Barone-Adesi-Whaley (1987) quadratic approximation."""

    def __init__(self, S, K, T, r, sigma, q=0.0):
        self.S, self.K, self.T = float(S), float(K), float(T)
        self.r, self.sigma, self.q = float(r), float(sigma), float(q)

    def _european(self, kind):
        return _bs_price(self.S, self.K, self.T, self.r, self.sigma, self.q, kind)

    def _baw_put(self):
        S, K, T, r, sigma, q = self.S, self.K, self.T, self.r, self.sigma, self.q
        if T <= 0:
            return max(K - S, 0.0), K
        M = 2.0 * r / sigma**2
        N = 2.0 * (r - q) / sigma**2
        K_T = 1.0 - math.exp(-r * T)
        q1 = (-(N - 1.0) - math.sqrt((N - 1.0) ** 2 + 4.0 * M / K_T)) / 2.0

        def resid(Sx):
            euro = _bs_price(Sx, K, T, r, sigma, q, "put")
            d1, _ = _bs_d1d2(Sx, K, T, r, sigma, q)
            return K - Sx - euro + (1.0 - math.exp(-q * T) * _norm_cdf(-d1)) * Sx / q1

        lo, hi = 1e-8 * K, K
        if resid(lo) * resid(hi) > 0:
            S_star = lo
        else:
            S_star = optimize.brentq(resid, lo, hi, xtol=1e-10)
        return S_star, q1

    def put_price(self):
        S, K, T, r, sigma, q = self.S, self.K, self.T, self.r, self.sigma, self.q
        if T <= 0:
            return max(K - S, 0.0)
        S_star, q1 = self._baw_put()
        if S <= S_star:
            return K - S
        euro_star = _bs_price(S_star, K, T, r, sigma, q, "put")
        d1_star, _ = _bs_d1d2(S_star, K, T, r, sigma, q)
        A1 = -(S_star / q1) * (1.0 - math.exp(-q * T) * _norm_cdf(-d1_star))
        return float(self._european("put") + A1 * (S / S_star) ** q1)

    def _baw_call(self):
        S, K, T, r, sigma, q = self.S, self.K, self.T, self.r, self.sigma, self.q
        M = 2.0 * r / sigma**2
        N = 2.0 * (r - q) / sigma**2
        K_T = 1.0 - math.exp(-r * T)
        q2 = (-(N - 1.0) + math.sqrt((N - 1.0) ** 2 + 4.0 * M / K_T)) / 2.0

        def resid(Sx):
            euro = _bs_price(Sx, K, T, r, sigma, q, "call")
            d1, _ = _bs_d1d2(Sx, K, T, r, sigma, q)
            return Sx - K - euro - (1.0 - math.exp(-q * T) * _norm_cdf(d1)) * Sx / q2

        lo, hi = K, K * 50.0
        while resid(hi) < 0 and hi < K * 1e6:
            hi *= 2.0
        S_star = optimize.brentq(resid, lo, hi, xtol=1e-10)
        return S_star, q2

    def call_price(self):
        S, K, T, r, sigma, q = self.S, self.K, self.T, self.r, self.sigma, self.q
        if T <= 0:
            return max(S - K, 0.0)
        if q <= 0:
            return float(self._european("call"))
        S_star, q2 = self._baw_call()
        if S >= S_star:
            return S - K
        euro_star = _bs_price(S_star, K, T, r, sigma, q, "call")
        d1_star, _ = _bs_d1d2(S_star, K, T, r, sigma, q)
        A2 = (S_star / q2) * (1.0 - math.exp(-q * T) * _norm_cdf(d1_star))
        return float(self._european("call") + A2 * (S / S_star) ** q2)

    def early_exercise_boundary(self, kind="call"):
        _check_kind(kind)
        if kind == "put":
            S_star, _ = self._baw_put()
        else:
            if self.q <= 0:
                return float("inf")
            S_star, _ = self._baw_call()
        return float(S_star)

    def price(self, kind="call", method="baw"):
        _check_kind(kind)
        if method == "tree":
            return BinomialTree(self.S, self.K, self.T, self.r, self.sigma, 1000, self.q
                                ).price(kind, american=True)
        return self.call_price() if kind == "call" else self.put_price()


class BinomialTree:
    """Cox-Ross-Rubinstein lattice, European or American exercise."""

    def __init__(self, S, K, T, r, sigma, n_steps=500, q=0.0):
        self.S, self.K, self.T = float(S), float(K), float(T)
        self.r, self.sigma, self.q = float(r), float(sigma), float(q)
        self.n_steps = int(n_steps)

    def _params(self):
        dt = self.T / self.n_steps
        u = math.exp(self.sigma * math.sqrt(dt))
        d = 1.0 / u
        disc = math.exp(-self.r * dt)
        p = (math.exp((self.r - self.q) * dt) - d) / (u - d)
        if not (0.0 < p < 1.0):
            raise ValueError(f"risk-neutral probability {p} out of (0,1); increase n_steps")
        return dt, u, d, disc, p

    def price(self, kind="call", american=False):
        _check_kind(kind)
        n = self.n_steps
        dt, u, d, disc, p = self._params()
        j = np.arange(n + 1)
        S_T = self.S * u**j * d**(n - j)
        V = np.maximum(S_T - self.K, 0.0) if kind == "call" else np.maximum(self.K - S_T, 0.0)
        for step in range(n - 1, -1, -1):
            V = disc * (p * V[1:] + (1.0 - p) * V[:-1])
            if american:
                jj = np.arange(step + 1)
                S_t = self.S * u**jj * d**(step - jj)
                intrinsic = np.maximum(S_t - self.K, 0.0) if kind == "call" else np.maximum(self.K - S_t, 0.0)
                V = np.maximum(V, intrinsic)
        return float(V[0])

    def call_price(self, american=False):
        return self.price("call", american)

    def put_price(self, american=False):
        return self.price("put", american)

    def delta(self, kind="call", american=False):
        dt, u, d, disc, p = self._params()
        n = self.n_steps
        if n < 1:
            raise ValueError("need at least 1 step for delta")
        S_up = self.S * u
        S_dn = self.S * d
        tree_up = BinomialTree(S_up, self.K, self.T - dt, self.r, self.sigma, n - 1, self.q)
        tree_dn = BinomialTree(S_dn, self.K, self.T - dt, self.r, self.sigma, n - 1, self.q)
        V_up = tree_up.price(kind, american) if n > 1 else self._terminal(S_up, kind)
        V_dn = tree_dn.price(kind, american) if n > 1 else self._terminal(S_dn, kind)
        return (V_up - V_dn) / (S_up - S_dn)

    def gamma(self, kind="call", american=False):
        dt, u, d, disc, p = self._params()
        n = self.n_steps
        if n < 2:
            raise ValueError("need at least 2 steps for gamma")
        S_uu, S_ud, S_dd = self.S * u * u, self.S, self.S * d * d
        t2 = BinomialTree(S_uu, self.K, self.T - 2 * dt, self.r, self.sigma, n - 2, self.q)
        t1 = BinomialTree(S_ud, self.K, self.T - 2 * dt, self.r, self.sigma, n - 2, self.q)
        t0 = BinomialTree(S_dd, self.K, self.T - 2 * dt, self.r, self.sigma, n - 2, self.q)
        f = (lambda tr, S: tr.price(kind, american)) if n > 2 else (lambda tr, S: self._terminal(S, kind))
        V_uu, V_ud, V_dd = f(t2, S_uu), f(t1, S_ud), f(t0, S_dd)
        h = 0.5 * (S_uu - S_dd)
        return ((V_uu - V_ud) / (S_uu - S_ud) - (V_ud - V_dd) / (S_ud - S_dd)) / h

    def _terminal(self, S, kind):
        return max(S - self.K, 0.0) if kind == "call" else max(self.K - S, 0.0)


class TrinomialTree:
    """Kamrad-Ritchken (1991) log-space trinomial lattice."""

    def __init__(self, S, K, T, r, sigma, n_steps=300, q=0.0):
        self.S, self.K, self.T = float(S), float(K), float(T)
        self.r, self.sigma, self.q = float(r), float(sigma), float(q)
        self.n_steps = int(n_steps)

    def price(self, kind="call", american=False):
        _check_kind(kind)
        n = self.n_steps
        dt = self.T / n
        dx = self.sigma * math.sqrt(3.0 * dt)
        nu = self.r - self.q - 0.5 * self.sigma**2
        disc = math.exp(-self.r * dt)
        p_u = 0.5 * ((self.sigma**2 * dt + nu**2 * dt**2) / dx**2 + nu * dt / dx)
        p_d = 0.5 * ((self.sigma**2 * dt + nu**2 * dt**2) / dx**2 - nu * dt / dx)
        p_m = 1.0 - p_u - p_d
        if not (0.0 <= p_u <= 1.0 and 0.0 <= p_d <= 1.0 and 0.0 <= p_m <= 1.0):
            raise ValueError("trinomial probabilities out of [0,1]; increase n_steps")

        j = np.arange(-n, n + 1)
        S_T = self.S * np.exp(j * dx)
        V = np.maximum(S_T - self.K, 0.0) if kind == "call" else np.maximum(self.K - S_T, 0.0)
        for step in range(n - 1, -1, -1):
            V = disc * (p_u * V[2:] + p_m * V[1:-1] + p_d * V[:-2])
            if american:
                jj = np.arange(-step, step + 1)
                S_t = self.S * np.exp(jj * dx)
                intrinsic = np.maximum(S_t - self.K, 0.0) if kind == "call" else np.maximum(self.K - S_t, 0.0)
                V = np.maximum(V, intrinsic)
        return float(V[0])

    def call_price(self, american=False):
        return self.price("call", american)

    def put_price(self, american=False):
        return self.price("put", american)


class MonteCarloOptionPricing:
    """European/path-dependent option pricing via plain, antithetic,
    control-variate, or quasi-Monte-Carlo simulation."""

    def __init__(self, S, K, T, r, sigma, q=0.0):
        self.S, self.K, self.T = float(S), float(K), float(T)
        self.r, self.sigma, self.q = float(r), float(sigma), float(q)

    def simulate_paths(self, n_paths, n_steps, random_state=None):
        rng = _rng(random_state)
        dt = self.T / n_steps
        drift = (self.r - self.q - 0.5 * self.sigma**2) * dt
        vol = self.sigma * math.sqrt(dt)
        Z = rng.standard_normal((n_paths, n_steps))
        log_incr = drift + vol * Z
        log_paths = np.concatenate(
            [np.zeros((n_paths, 1)), np.cumsum(log_incr, axis=1)], axis=1)
        return self.S * np.exp(log_paths)

    def _payoff(self, S_T, kind):
        return np.maximum(S_T - self.K, 0.0) if kind == "call" else np.maximum(self.K - S_T, 0.0)

    def price(self, kind="call", n_paths=100_000, antithetic=True,
              control_variate=False, qmc=False, random_state=None):
        _check_kind(kind)
        disc = math.exp(-self.r * self.T)
        if qmc:
            return self._price_qmc(kind, n_paths, disc, random_state)
        rng = _rng(random_state)
        if antithetic:
            m = n_paths // 2
            Z = rng.standard_normal(m)
            drift = (self.r - self.q - 0.5 * self.sigma**2) * self.T
            vol = self.sigma * math.sqrt(self.T)
            up = self.S * np.exp(drift + vol * Z)
            down = self.S * np.exp(drift - vol * Z)
            payoff = 0.5 * (self._payoff(up, kind) + self._payoff(down, kind))
        else:
            Z = rng.standard_normal(n_paths)
            drift = (self.r - self.q - 0.5 * self.sigma**2) * self.T
            vol = self.sigma * math.sqrt(self.T)
            S_T = self.S * np.exp(drift + vol * Z)
            payoff = self._payoff(S_T, kind)

        if control_variate:
            S_T_full = self.S * np.exp((self.r - self.q - 0.5 * self.sigma**2) * self.T
                                       + self.sigma * math.sqrt(self.T) * Z)
            control_mean = self.S * math.exp((self.r - self.q) * self.T)
            cov = np.cov(payoff, S_T_full, ddof=1)
            beta = cov[0, 1] / cov[1, 1] if cov[1, 1] > 0 else 0.0
            adjusted = payoff - beta * (S_T_full - control_mean)
            res = _mc_result(adjusted, "mc-control-variate", {"beta": float(beta)}, discount=disc)
            res.n_samples = n_paths
            return res

        method = "mc-antithetic" if antithetic else "mc-plain"
        res = _mc_result(payoff, method, discount=disc)
        # antithetic pairing halves the array of *statistical replicates*
        # (each pair contributes one averaged payoff) but the convention set
        # by montecarlo.applications.option_pricing_mc reports n_samples as
        # the number of paths actually simulated, not the replicate count.
        res.n_samples = n_paths
        return res

    def _price_qmc(self, kind, n_paths, disc, random_state, n_scrambles=8):
        from scipy.special import ndtri

        from stochpylib.montecarlo._result import MCResult

        seeds = np.random.SeedSequence(random_state).spawn(n_scrambles)
        per = max(n_paths // n_scrambles, 1)
        estimates = []
        for s in seeds:
            seq = SobolSequence(dim=1, random_state=s)
            u = np.clip(seq.generate(per)[:, 0], 1e-12, 1 - 1e-12)
            Z = ndtri(u)
            drift = (self.r - self.q - 0.5 * self.sigma**2) * self.T
            vol = self.sigma * math.sqrt(self.T)
            S_T = self.S * np.exp(drift + vol * Z)
            payoff = disc * self._payoff(S_T, kind)
            estimates.append(float(payoff.mean()))
        estimates = np.array(estimates)
        mean = float(estimates.mean())
        se = float(estimates.std(ddof=1) / math.sqrt(n_scrambles))
        return MCResult(mean, se, per * n_scrambles, "mc-qmc-sobol", {"n_scrambles": n_scrambles})

    def price_path_dependent(self, payoff, n_paths=50_000, n_steps=252, random_state=None):
        """``payoff(paths)`` maps the ``(n_paths, n_steps+1)`` price grid to payoffs."""
        paths = self.simulate_paths(n_paths, n_steps, random_state)
        disc = math.exp(-self.r * self.T)
        result = np.asarray(payoff(paths), dtype=float)
        return _mc_result(result, "mc-path-dependent", discount=disc)

    def asian_price(self, kind="call", average="arithmetic", n_paths=50_000,
                     n_steps=252, random_state=None):
        _check_kind(kind)
        paths = self.simulate_paths(n_paths, n_steps, random_state)
        avg_prices = paths[:, 1:].mean(axis=1) if average == "arithmetic" \
            else np.exp(np.log(paths[:, 1:]).mean(axis=1))
        payoff = self._payoff(avg_prices, kind)
        disc = math.exp(-self.r * self.T)
        return _mc_result(payoff, f"mc-asian-{average}", discount=disc)


class LongstaffSchwartz:
    """American option pricing via Longstaff-Schwartz (2001) least-squares MC."""

    def __init__(self, S, K, T, r, sigma, q=0.0, n_steps=50, basis="laguerre", degree=3):
        self.S, self.K, self.T = float(S), float(K), float(T)
        self.r, self.sigma, self.q = float(r), float(sigma), float(q)
        self.n_steps = int(n_steps)
        self.basis = basis
        self.degree = int(degree)
        self.exercise_boundary_ = None
        self.coefficients_ = None

    def _basis_matrix(self, x):
        if self.basis == "laguerre":
            cols = [np.exp(-x / 2.0) * np.polynomial.laguerre.lagval(x, [0] * k + [1])
                    for k in range(self.degree + 1)]
        else:
            cols = [x**k for k in range(self.degree + 1)]
        return np.column_stack(cols)

    def _payoff(self, S_t, kind):
        return np.maximum(S_t - self.K, 0.0) if kind == "call" else np.maximum(self.K - S_t, 0.0)

    def price(self, kind="put", n_paths=100_000, random_state=None, paths=None, payoff=None):
        _check_kind(kind)
        n = self.n_steps
        dt = self.T / n
        disc = math.exp(-self.r * dt)
        if paths is None:
            rng = _rng(random_state)
            drift = (self.r - self.q - 0.5 * self.sigma**2) * dt
            vol = self.sigma * math.sqrt(dt)
            Z = rng.standard_normal((n_paths, n))
            log_paths = np.concatenate(
                [np.zeros((n_paths, 1)), np.cumsum(drift + vol * Z, axis=1)], axis=1)
            paths = self.S * np.exp(log_paths)
        else:
            n_paths = paths.shape[0]
            n = paths.shape[1] - 1

        pay_fn = payoff if payoff is not None else (lambda S_t: self._payoff(S_t, kind))
        CF = pay_fn(paths[:, -1])
        exercise_time = np.full(n_paths, n)
        boundary = [None] * (n + 1)

        for t in range(n - 1, 0, -1):
            S_t = paths[:, t]
            intrinsic = pay_fn(S_t)
            itm = intrinsic > 0
            discounted_cf = CF * disc ** (exercise_time - t)
            if itm.sum() >= self.degree + 2:
                X = self._basis_matrix(S_t[itm] / self.K)
                y = discounted_cf[itm]
                coef, *_ = np.linalg.lstsq(X, y, rcond=None)
                continuation = X @ coef
                exercise_now = intrinsic[itm] > continuation
                idx = np.where(itm)[0][exercise_now]
                if idx.size:
                    boundary[t] = float(S_t[idx].min())
            else:
                idx = np.array([], dtype=int)
            if idx.size:
                CF[idx] = intrinsic[idx]
                exercise_time[idx] = t

        self.exercise_boundary_ = boundary
        discounted_cf = CF * disc ** exercise_time
        return _mc_result(discounted_cf, "longstaff-schwartz")


class FourierOptionPricing:
    """European option pricing via characteristic-function inversion
    (Carr-Madan damped Fourier, or the Fang-Oosterlee COS method)."""

    def __init__(self, cf_log_price, S, T, r, q=0.0):
        self.cf_log_price = cf_log_price
        self.S, self.T, self.r, self.q = float(S), float(T), float(r), float(q)

    @classmethod
    def from_black_scholes(cls, S, T, r, sigma, q=0.0):
        def cf(u):
            drift = math.log(S) + (r - q - 0.5 * sigma**2) * T
            return np.exp(1j * u * drift - 0.5 * sigma**2 * u**2 * T)
        return cls(cf, S, T, r, q)

    @classmethod
    def from_heston(cls, model, T):
        return cls(lambda u: model.characteristic_function(u, T), model.S0, T, model.r,
                   getattr(model, "q", 0.0))

    @classmethod
    def from_model(cls, obj, T):
        S0 = getattr(obj, "S0", getattr(obj, "S", None))
        r = getattr(obj, "r", 0.0)
        q = getattr(obj, "q", 0.0)
        return cls(lambda u: obj.characteristic_function(u, T), S0, T, r, q)

    def _cos_price(self, K, kind, N_cos=256, L=10.0):
        """Fang-Oosterlee (2008) COS method. ``cf_log_price`` is the
        characteristic function of the *absolute* ``ln S_T`` (same convention
        as :func:`carr_madan_call`); truncation range from finite-difference
        cumulants (correctly ``Var = E[z^2] - E[z]^2``, not the raw second
        moment) so it stays centered on the true log-price distribution
        regardless of ``S``/``K`` scale.
        """
        K_arr = np.atleast_1d(np.asarray(K, dtype=float))
        cf = self.cf_log_price
        h = 1e-4
        phi0 = complex(cf(np.array([0.0]))[0])
        phi_h = complex(cf(np.array([h]))[0])
        phi_mh = complex(cf(np.array([-h]))[0])
        c1 = complex(-1j * (phi_h - phi_mh) / (2.0 * h)).real
        c2 = complex(-(phi_h - 2.0 * phi0 + phi_mh) / h**2 - complex(c1)**2).real
        c2 = max(c2, 1e-8)
        a = c1 - L * math.sqrt(c2)
        b = c1 + L * math.sqrt(c2)

        k_idx = np.arange(N_cos)
        u = k_idx * math.pi / (b - a)
        phi = cf(u) * np.exp(-1j * u * a)

        out = np.empty(len(K_arr))
        for i, strike in enumerate(K_arr):
            kappa = math.log(strike)
            if kind == "call":
                c, d = max(kappa, a), b
            else:
                c, d = a, min(kappa, b)
            chi, psi = self._chi_psi(k_idx, a, b, c, d)
            if kind == "call":
                Uk = 2.0 / (b - a) * (chi - strike * psi)
            else:
                Uk = 2.0 / (b - a) * (-chi + strike * psi)
            terms = np.real(phi) * Uk
            terms[0] *= 0.5
            out[i] = math.exp(-self.r * self.T) * terms.sum()
        return out if np.ndim(K) else float(out[0])

    @staticmethod
    def _chi_psi(k_idx, a, b, c, d):
        w = k_idx * math.pi / (b - a)
        chi = (1.0 / (1.0 + w**2)) * (
            np.cos(w * (d - a)) * math.exp(d) - np.cos(w * (c - a)) * math.exp(c)
            + w * np.sin(w * (d - a)) * math.exp(d) - w * np.sin(w * (c - a)) * math.exp(c)
        )
        psi = np.where(
            k_idx == 0, d - c,
            (np.sin(w * (d - a)) - np.sin(w * (c - a))) / np.where(w == 0, 1.0, w),
        )
        return chi, psi

    def call_price(self, K, method="carr_madan", alpha=1.5, n_points=20_000,
                   v_max=200.0, N_cos=256, L=10.0):
        if method == "carr_madan":
            K_arr = np.atleast_1d(np.asarray(K, dtype=float))
            out = np.array([carr_madan_call(self.cf_log_price, self.S, k, self.T, self.r,
                                            alpha, n_points, v_max) for k in K_arr])
            return out if np.ndim(K) else float(out[0])
        if method == "cos":
            return self._cos_price(K, "call", N_cos, L)
        raise ValueError("method must be 'carr_madan' or 'cos'")

    def put_price(self, K, method="carr_madan", **kw):
        call = self.call_price(K, method, **kw)
        K_arr = np.asarray(K, dtype=float)
        disc = math.exp(-self.r * self.T)
        fwd_disc = self.S * math.exp(-self.q * self.T)
        put = np.asarray(call) - fwd_disc + K_arr * disc
        return put if np.ndim(K) else float(put)

    def price(self, K, kind="call", method="carr_madan", **kw):
        _check_kind(kind)
        return self.call_price(K, method, **kw) if kind == "call" else self.put_price(K, method, **kw)

    def implied_vol(self, K, kind="call", method="carr_madan", **kw):
        price = self.price(K, kind, method, **kw)
        return _implied_vol(float(price), self.S, float(K), self.T, self.r, self.q, kind)
