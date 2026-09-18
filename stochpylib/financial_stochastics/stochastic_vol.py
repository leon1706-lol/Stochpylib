"""Stochastic and local volatility models: Heston, SABR, rough volatility
(rough Heston, rough Bergomi), local vol (Dupire), local-stochastic vol, and
variance swaps.
"""

import math

import numpy as np
from scipy import optimize, special

from stochpylib.financial_stochastics._common import (
    _bs_price,
    _check_kind,
    _cholesky_psd,
    _implied_vol,
    _mc_result,
    _rng,
    _solve_tridiagonal,
)
from stochpylib.financial_stochastics.option_pricing import BlackScholes
from stochpylib.levy_processes.jump_diffusion import carr_madan_call

__all__ = [
    "HestonModel", "SABRModel", "RoughHeston", "RoughBergomi",
    "LVSV", "LocalVol", "Dupire", "VarianceSwap",
]


# --------------------------------------------------------------------------- Heston

class HestonModel:
    """Heston (1993) stochastic volatility.

    ``dS = (r-q) S dt + sqrt(v) S dW1``, ``dv = kappa(theta-v) dt + xi sqrt(v) dW2``,
    ``corr(dW1,dW2) = rho``.
    """

    def __init__(self, S0, v0, kappa, theta, xi, rho, r, q=0.0):
        if v0 <= 0 or theta <= 0 or kappa <= 0 or xi <= 0:
            raise ValueError("v0, theta, kappa, xi must be positive")
        if not -1.0 <= rho <= 1.0:
            raise ValueError("rho must lie in [-1, 1]")
        self.S0, self.v0 = float(S0), float(v0)
        self.kappa, self.theta, self.xi, self.rho = float(kappa), float(theta), float(xi), float(rho)
        self.r, self.q = float(r), float(q)

    def feller_condition(self):
        return 2.0 * self.kappa * self.theta >= self.xi**2

    def expected_integrated_variance(self, T):
        T = float(T)
        kappa, theta, v0 = self.kappa, self.theta, self.v0
        if kappa * T < 1e-10:
            return v0 * T
        return theta * T + (v0 - theta) * (1.0 - math.exp(-kappa * T)) / kappa

    def characteristic_function(self, u, T):
        """cf of the *absolute* ``ln S_T`` — the 'little trap' formulation
        (Albrecher et al. 2007) to avoid the branch-cut instability of the
        original Heston (1993) form."""
        u = np.asarray(u, dtype=complex)
        T = float(T)
        kappa, theta, xi, rho = self.kappa, self.theta, self.xi, self.rho
        v0, S0, r, q = self.v0, self.S0, self.r, self.q

        if xi < 1e-8:
            V_bar = self.expected_integrated_variance(T)
            drift = math.log(S0) + (r - q) * T - 0.5 * V_bar
            return np.exp(1j * u * drift - 0.5 * u**2 * V_bar)

        beta = kappa - rho * xi * 1j * u
        d = np.sqrt(beta**2 + xi**2 * (1j * u + u**2))
        g = (beta - d) / (beta + d)
        e = np.exp(-d * T)
        C = 1j * u * (math.log(S0) + (r - q) * T) \
            + (kappa * theta / xi**2) * ((beta - d) * T - 2.0 * np.log((1.0 - g * e) / (1.0 - g)))
        D = ((beta - d) / xi**2) * (1.0 - e) / (1.0 - g * e)
        return np.exp(C + D * v0)

    def call_price(self, K, T, alpha=1.5, n_points=20_000, v_max=200.0):
        return carr_madan_call(lambda u: self.characteristic_function(u, T),
                               self.S0, K, T, self.r, alpha, n_points, v_max)

    def put_price(self, K, T, **kw):
        call = self.call_price(K, T, **kw)
        return call - self.S0 * math.exp(-self.q * T) + K * math.exp(-self.r * T)

    def price(self, K, T, kind="call", **kw):
        _check_kind(kind)
        return self.call_price(K, T, **kw) if kind == "call" else self.put_price(K, T, **kw)

    def implied_vol(self, K, T, kind="call", **kw):
        price = self.price(K, T, kind, **kw)
        return _implied_vol(price, self.S0, K, T, self.r, self.q, kind)

    def _qe_step(self, v, dt, rng):
        kappa, theta, xi = self.kappa, self.theta, self.xi
        psi_c = 1.5
        ekt = math.exp(-kappa * dt)
        m = theta + (v - theta) * ekt
        m = np.maximum(m, 1e-12)
        s2 = (v * xi**2 * ekt / kappa) * (1.0 - ekt) + (theta * xi**2 / (2.0 * kappa)) * (1.0 - ekt) ** 2
        psi = s2 / m**2
        v_next = np.empty_like(v)

        small = psi <= psi_c
        if np.any(small):
            psi_s = np.clip(psi[small], 1e-12, None)
            b2 = 2.0 / psi_s - 1.0 + np.sqrt(2.0 / psi_s) * np.sqrt(np.maximum(2.0 / psi_s - 1.0, 0.0))
            a = m[small] / (1.0 + b2)
            Z = rng.standard_normal(np.sum(small))
            v_next[small] = a * (np.sqrt(b2) + Z) ** 2

        big = ~small
        if np.any(big):
            psi_b = psi[big]
            p = (psi_b - 1.0) / (psi_b + 1.0)
            beta = (1.0 - p) / m[big]
            U = rng.random(np.sum(big))
            vb = np.where(U <= p, 0.0, np.log((1.0 - p) / (1.0 - U + 1e-300)) / beta)
            v_next[big] = vb
        return v_next

    def simulate(self, T, N=252, n_paths=1, scheme="qe", random_state=None,
                 return_variance=False):
        rng = _rng(random_state)
        dt = T / N
        S = np.empty((n_paths, N + 1))
        v = np.empty((n_paths, N + 1))
        S[:, 0] = self.S0
        v[:, 0] = self.v0
        kappa, theta, xi, rho, r, q = (self.kappa, self.theta, self.xi, self.rho,
                                        self.r, self.q)

        if scheme == "qe":
            gamma1 = gamma2 = 0.5
            K0 = -rho * kappa * theta * dt / xi
            K1 = gamma1 * dt * (kappa * rho / xi - 0.5) - rho / xi
            K2 = gamma2 * dt * (kappa * rho / xi - 0.5) + rho / xi
            K3 = gamma1 * dt * (1.0 - rho**2)
            K4 = gamma2 * dt * (1.0 - rho**2)
            for t in range(N):
                v_t = v[:, t]
                v_next = self._qe_step(v_t, dt, rng)
                Zs = rng.standard_normal(n_paths)
                log_incr = ((r - q) * dt + K0 + K1 * v_t + K2 * v_next
                            + np.sqrt(np.maximum(K3 * v_t + K4 * v_next, 0.0)) * Zs)
                S[:, t + 1] = S[:, t] * np.exp(log_incr)
                v[:, t + 1] = v_next
        elif scheme == "euler":
            sqrt_dt = math.sqrt(dt)
            for t in range(N):
                v_pos = np.maximum(v[:, t], 0.0)
                Z1 = rng.standard_normal(n_paths)
                Z2 = rng.standard_normal(n_paths)
                W2 = rho * Z1 + math.sqrt(max(1.0 - rho**2, 0.0)) * Z2
                v[:, t + 1] = v[:, t] + kappa * (theta - v_pos) * dt + xi * np.sqrt(v_pos) * sqrt_dt * W2
                S[:, t + 1] = S[:, t] * np.exp((r - q - 0.5 * v_pos) * dt + np.sqrt(v_pos) * sqrt_dt * Z1)
        else:
            raise ValueError("scheme must be 'qe' or 'euler'")

        if return_variance:
            return S, v
        return S

    def call_price_mc(self, K, T, n_paths=100_000, N=100, scheme="qe", random_state=None):
        S = self.simulate(T, N=N, n_paths=n_paths, scheme=scheme, random_state=random_state)
        disc = math.exp(-self.r * T)
        payoff = np.maximum(S[:, -1] - K, 0.0)
        return _mc_result(payoff, f"heston-mc-{scheme}", discount=disc)

    def fit(self, strikes, maturities, prices, kind="call", x0=None, n_points=4_000, v_max=150.0):
        """Calibrate ``(kappa, theta, xi, rho, v0)`` to market prices by
        least squares. Uses a coarser Carr-Madan grid than ``call_price``'s
        pricing-accuracy default (20k points) — each residual evaluation
        prices every quoted strike, and ``least_squares`` needs many such
        evaluations for its finite-difference Jacobian, so the full-accuracy
        grid made a single calibration take minutes; ``n_points``/``v_max``
        are still adjustable for a more precise (slower) fit.
        """
        strikes = np.asarray(strikes, dtype=float)
        maturities = np.asarray(maturities, dtype=float)
        prices = np.asarray(prices, dtype=float)
        x0 = x0 or [self.kappa, self.theta, self.xi, self.rho, self.v0]

        def resid(params):
            kappa, theta, xi, rho, v0 = params
            model = HestonModel(self.S0, max(v0, 1e-6), max(kappa, 1e-6), max(theta, 1e-6),
                                max(xi, 1e-6), np.clip(rho, -0.999, 0.999), self.r, self.q)
            out = np.empty(len(strikes))
            for i in range(len(strikes)):
                out[i] = model.price(strikes[i], maturities[i], kind,
                                     n_points=n_points, v_max=v_max) - prices[i]
            return out

        bounds = ([1e-3, 1e-4, 1e-3, -0.999, 1e-4], [20.0, 2.0, 5.0, 0.999, 2.0])
        res = optimize.least_squares(resid, x0, bounds=bounds)
        self.kappa, self.theta, self.xi, self.rho, self.v0 = res.x
        self.kappa_, self.theta_, self.xi_, self.rho_, self.v0_ = res.x
        self.rmse_ = float(np.sqrt(np.mean(res.fun**2)))
        return self


# --------------------------------------------------------------------------- SABR

class SABRModel:
    """SABR (Hagan et al. 2002) stochastic-alpha-beta-rho volatility model.

    ``dF = alpha F^beta dW1``, ``d alpha = nu alpha dW2``, ``corr = rho``.
    """

    def __init__(self, alpha, beta, rho, nu, shift=0.0):
        if alpha <= 0 or nu < 0:
            raise ValueError("alpha must be > 0, nu must be >= 0")
        if not -1.0 <= rho <= 1.0:
            raise ValueError("rho must lie in [-1, 1]")
        if not 0.0 <= beta <= 1.0:
            raise ValueError("beta must lie in [0, 1]")
        self.alpha, self.beta, self.rho, self.nu = float(alpha), float(beta), float(rho), float(nu)
        self.shift = float(shift)

    def implied_vol(self, F, K, T):
        F = np.asarray(F, dtype=float) + self.shift
        K = np.asarray(K, dtype=float) + self.shift
        T = np.asarray(T, dtype=float)
        alpha, beta, rho, nu = self.alpha, self.beta, self.rho, self.nu

        FK = F * K
        FK_beta = FK ** ((1.0 - beta) / 2.0)
        log_FK = np.log(F / K)

        z = (nu / alpha) * FK_beta * log_FK
        # z_over_x = z / x(z); the small-z expansion of x(z) is
        # z - (rho/2) z^2 + O(z^3), so z/x(z) -> 1 + (rho/2) z (not the
        # reciprocal, and not the sign-flipped 1 - (rho/2) z).
        x_of_z = np.log((np.sqrt(1.0 - 2.0 * rho * z + z**2) + z - rho) / (1.0 - rho))
        z_over_x = np.where(
            np.abs(z) < 1e-7,
            1.0 + 0.5 * rho * z,
            z / np.where(np.abs(x_of_z) < 1e-300, 1.0, x_of_z),
        )

        term1 = alpha / (FK_beta * (1.0 + (1.0 - beta) ** 2 / 24.0 * log_FK**2
                                    + (1.0 - beta) ** 4 / 1920.0 * log_FK**4))
        term2 = ((1.0 - beta) ** 2 / 24.0 * alpha**2 / FK_beta**2
                 + 0.25 * rho * beta * nu * alpha / FK_beta
                 + (2.0 - 3.0 * rho**2) / 24.0 * nu**2)
        atm_vol = alpha / (F ** (1.0 - beta)) * (1.0 + term2 * T)

        near_atm = np.abs(np.log(F / K)) < 1e-10
        vol = term1 * z_over_x * (1.0 + term2 * T)
        vol = np.where(near_atm, atm_vol, vol)
        return vol if vol.ndim else float(vol)

    def price(self, F, K, T, df=1.0, kind="call"):
        from stochpylib.financial_stochastics._common import _black_price
        vol = self.implied_vol(F, K, T)
        return _black_price(F, K, T, vol, df=df, kind=kind)

    def fit(self, F, strikes, T, market_vols, beta=None):
        strikes = np.asarray(strikes, dtype=float)
        market_vols = np.asarray(market_vols, dtype=float)
        fit_beta = beta is None
        x0 = [self.alpha, self.rho, self.nu] + ([self.beta] if fit_beta else [])

        def resid(params):
            alpha, rho, nu = params[0], np.clip(params[1], -0.999, 0.999), max(params[2], 1e-6)
            b = np.clip(params[3], 0.0, 1.0) if fit_beta else self.beta
            model = SABRModel(max(alpha, 1e-6), b, rho, nu, self.shift)
            return model.implied_vol(F, strikes, T) - market_vols

        bounds_lo = [1e-4, -0.999, 1e-6] + ([0.0] if fit_beta else [])
        bounds_hi = [5.0, 0.999, 5.0] + ([1.0] if fit_beta else [])
        res = optimize.least_squares(resid, x0, bounds=(bounds_lo, bounds_hi))
        self.alpha, self.rho, self.nu = res.x[0], res.x[1], res.x[2]
        if fit_beta:
            self.beta = res.x[3]
            self.beta_ = self.beta
        self.alpha_, self.rho_, self.nu_ = self.alpha, self.rho, self.nu
        self.rmse_ = float(np.sqrt(np.mean(res.fun**2)))
        return self

    def simulate(self, F0, T, N=252, n_paths=1, random_state=None):
        rng = _rng(random_state)
        dt = T / N
        sqrt_dt = math.sqrt(dt)
        F = np.empty((n_paths, N + 1))
        A = np.empty((n_paths, N + 1))
        F[:, 0] = F0
        A[:, 0] = self.alpha
        beta, rho, nu = self.beta, self.rho, self.nu
        for t in range(N):
            Z1 = rng.standard_normal(n_paths)
            Z2 = rng.standard_normal(n_paths)
            W2 = rho * Z1 + math.sqrt(max(1.0 - rho**2, 0.0)) * Z2
            A[:, t + 1] = A[:, t] * np.exp(-0.5 * nu**2 * dt + nu * sqrt_dt * W2)
            if beta >= 1.0 - 1e-12:
                F[:, t + 1] = F[:, t] * np.exp(-0.5 * A[:, t]**2 * dt + A[:, t] * sqrt_dt * Z1)
            else:
                F_pos = np.maximum(F[:, t], 0.0)
                incr = A[:, t] * F_pos**beta * sqrt_dt * Z1
                F[:, t + 1] = np.maximum(F[:, t] + incr, 0.0)
        return F, A


# --------------------------------------------------------------------------- Rough Heston

class RoughHeston:
    """El Euch-Rosenbaum (2019) rough Heston: fractional Riccati equation
    solved via a fractional Adams predictor-corrector scheme."""

    def __init__(self, S0, v0, kappa, theta, xi, rho, r, H=0.1, q=0.0):
        if not 0.0 < H < 0.5 and H != 0.5:
            raise ValueError("H must lie in (0, 0.5]")
        if v0 <= 0 or theta <= 0 or kappa <= 0 or xi <= 0:
            raise ValueError("v0, theta, kappa, xi must be positive")
        self.S0, self.v0 = float(S0), float(v0)
        self.kappa, self.theta, self.xi, self.rho = float(kappa), float(theta), float(xi), float(rho)
        self.r, self.q, self.H = float(r), float(q), float(H)

    def _riccati_adams(self, u, T, n_steps):
        """Returns arrays h_j (complex, shape (n_steps+1, len(u))) solving
        ``D^alpha h = F(h)``, ``alpha = H + 1/2``, via fractional Adams."""
        alpha = self.H + 0.5
        kappa, theta, xi, rho = self.kappa, self.theta, self.xi, self.rho
        dt = T / n_steps
        u = np.asarray(u, dtype=complex)

        def F(h):
            return 0.5 * (-u * u - 1j * u) + (1j * u * rho * xi - kappa) * h + 0.5 * xi**2 * h**2

        h = np.zeros((n_steps + 1, len(u)), dtype=complex)
        Fh = np.zeros((n_steps + 1, len(u)), dtype=complex)
        Fh[0] = F(h[0])

        gam_a = special.gamma(alpha)
        gam_a2 = special.gamma(alpha + 2.0)
        j = np.arange(n_steps + 1)

        for k in range(n_steps):
            jj = np.arange(k + 1)
            b_wts = (dt**alpha / alpha) * ((k + 1 - jj) ** alpha - (k - jj) ** alpha)
            h_pred = np.sum(b_wts[:, None] * Fh[: k + 1], axis=0) / gam_a

            if k == 0:
                a0 = 1.0
                a_wts = np.array([a0])
            else:
                kk = float(k)
                jj_f = jj.astype(float)
                a_wts = np.empty(k + 1)
                a_wts[0] = kk ** (alpha + 1.0) - (kk - alpha) * (kk + 1.0) ** alpha
                m = jj_f[1:]
                a_wts[1:] = ((kk - m + 2.0) ** (alpha + 1.0) + (kk - m) ** (alpha + 1.0)
                            - 2.0 * (kk - m + 1.0) ** (alpha + 1.0))
            F_pred = F(h_pred)
            h[k + 1] = (dt**alpha / gam_a2) * (F_pred + np.sum(a_wts[:, None] * Fh[: k + 1], axis=0))
            Fh[k + 1] = F(h[k + 1])
        return h, Fh

    def characteristic_function(self, u, T, n_steps=200):
        """``log phi = iu(...) + kappa*theta * I^1[h](T) + v0 * I^(1-alpha)[h](T)``
        (El Euch-Rosenbaum 2019) — both integral terms are fractional
        integrals of the *solution* ``h`` of the Riccati equation, not of its
        derivative ``F(h)``."""
        u = np.asarray(u, dtype=complex)
        alpha = self.H + 0.5
        h, _ = self._riccati_adams(u, T, n_steps)
        t_grid = np.linspace(0.0, T, n_steps + 1)

        I1 = np.trapezoid(h, t_grid, axis=0) if hasattr(np, "trapezoid") \
            else np.trapz(h, t_grid, axis=0)

        if abs(self.H - 0.5) < 1e-12:
            I_frac = h[-1]
        else:
            weights = (T - t_grid[:-1]) ** (1.0 - alpha) - (T - t_grid[1:]) ** (1.0 - alpha)
            avg = 0.5 * (h[:-1] + h[1:])
            I_frac = np.sum(weights[:, None] * avg, axis=0) / (special.gamma(1.0 - alpha) * (1.0 - alpha))

        log_phi = (1j * u * (math.log(self.S0) + (self.r - self.q) * T)
                  + self.kappa * self.theta * I1 + self.v0 * I_frac)
        return np.exp(log_phi)

    def call_price(self, K, T, n_steps=200, alpha=1.5, n_points=2_000, v_max=100.0):
        return carr_madan_call(lambda u: self.characteristic_function(u, T, n_steps),
                               self.S0, K, T, self.r, alpha, n_points, v_max)

    def put_price(self, K, T, **kw):
        call = self.call_price(K, T, **kw)
        return call - self.S0 * math.exp(-self.q * T) + K * math.exp(-self.r * T)

    def simulate(self, T, N=252, n_paths=1, random_state=None, return_variance=False):
        rng = _rng(random_state)
        dt = T / N
        sqrt_dt = math.sqrt(dt)
        alpha = self.H + 0.5
        kappa, theta, xi, rho, r, q, v0 = (self.kappa, self.theta, self.xi, self.rho,
                                           self.r, self.q, self.v0)
        t_grid = np.arange(N + 1) * dt
        w = np.zeros((N, N))
        for k in range(N):
            for jidx in range(k + 1):
                w[k, jidx] = ((t_grid[k + 1] - t_grid[jidx]) ** alpha
                             - (t_grid[k + 1] - t_grid[jidx + 1]) ** alpha) / special.gamma(alpha + 1.0)

        v = np.full((n_paths, N + 1), v0)
        S = np.full((n_paths, N + 1), self.S0)
        incr = np.zeros((n_paths, N))
        for k in range(N):
            v_pos = np.maximum(v[:, k], 0.0)
            Z1 = rng.standard_normal(n_paths)
            Z2 = rng.standard_normal(n_paths)
            W2 = rho * Z1 + math.sqrt(max(1.0 - rho**2, 0.0)) * Z2
            dW = sqrt_dt * W2
            incr[:, k] = kappa * (theta - v_pos) * dt + xi * np.sqrt(v_pos) * dW
            v[:, k + 1] = v0 + incr[:, : k + 1] @ w[k, : k + 1]
            S[:, k + 1] = S[:, k] * np.exp((r - q - 0.5 * v_pos) * dt + np.sqrt(v_pos) * sqrt_dt * Z1)
        if return_variance:
            return S, v
        return S

    def call_price_mc(self, K, T, n_paths=50_000, N=100, random_state=None):
        S = self.simulate(T, N=N, n_paths=n_paths, random_state=random_state)
        payoff = np.maximum(S[:, -1] - K, 0.0)
        disc = math.exp(-self.r * T)
        return _mc_result(payoff, "rough-heston-mc", discount=disc)


# --------------------------------------------------------------------------- Rough Bergomi

class RoughBergomi:
    """Bayer-Friz-Gatheral (2016) rough Bergomi via exact Cholesky simulation
    of the joint Gaussian ``(Y, W)`` where ``Y`` is Riemann-Liouville fBM."""

    def __init__(self, S0, xi0, eta, rho, H, r=0.0, q=0.0):
        if not 0.0 < H < 0.5:
            raise ValueError("H must lie in (0, 0.5)")
        if not -1.0 <= rho <= 1.0:
            raise ValueError("rho must lie in [-1, 1]")
        self.S0 = float(S0)
        self.xi0 = xi0 if callable(xi0) else (lambda t, v=float(xi0): v)
        self.eta, self.rho, self.H = float(eta), float(rho), float(H)
        self.r, self.q = float(r), float(q)

    def _xi0(self, t):
        return self.xi0(t)

    def covariance_matrix(self, T, N):
        H = self.H
        t = np.linspace(T / N, T, N)
        n = N
        Cyy = np.empty((n, n))
        for i in range(n):
            for j in range(n):
                s, u = min(t[i], t[j]), max(t[i], t[j])
                if s <= 0:
                    Cyy[i, j] = 0.0
                    continue
                ratio = np.clip(s / u, 0.0, 1.0)
                Cyy[i, j] = (2.0 * H * s ** (H + 0.5) * u ** (H - 0.5) / (H + 0.5)
                            * special.hyp2f1(0.5 - H, 1.0, H + 1.5, ratio))
        for i in range(n):
            Cyy[i, i] = t[i] ** (2.0 * H)

        Cyw = np.empty((n, n))
        for i in range(n):
            for j in range(n):
                ti, sj = t[i], t[j]
                m = min(ti, sj)
                Cyw[i, j] = (math.sqrt(2.0 * H) / (H + 0.5)) * (ti ** (H + 0.5) - (ti - m) ** (H + 0.5))

        Cww = np.minimum.outer(t, t)
        top = np.concatenate([Cyy, Cyw], axis=1)
        bot = np.concatenate([Cyw.T, Cww], axis=1)
        return np.concatenate([top, bot], axis=0)

    def simulate(self, T, N=252, n_paths=1, random_state=None, return_variance=False):
        rng = _rng(random_state)
        Sigma = self.covariance_matrix(T, N)
        L = _cholesky_psd(Sigma)
        Z = rng.standard_normal((n_paths, 2 * N))
        draws = Z @ L.T
        Y = draws[:, :N]
        W = draws[:, N:]

        t = np.linspace(T / N, T, N)
        xi0_t = np.array([self._xi0(ti) for ti in t])
        v = xi0_t[None, :] * np.exp(self.eta * Y - 0.5 * self.eta**2 * t[None, :] ** (2 * self.H))
        v_full = np.concatenate([np.full((n_paths, 1), self._xi0(0.0)), v], axis=1)

        dt = T / N
        dW = np.diff(np.concatenate([np.zeros((n_paths, 1)), W], axis=1), axis=1)
        rho = self.rho
        Zperp = rng.standard_normal((n_paths, N))
        dB = rho * dW + math.sqrt(max(1.0 - rho**2, 0.0)) * math.sqrt(dt) * Zperp

        log_S = np.zeros((n_paths, N + 1))
        for k in range(N):
            v_k = np.maximum(v_full[:, k], 0.0)
            log_S[:, k + 1] = log_S[:, k] + (self.r - self.q - 0.5 * v_k) * dt + np.sqrt(v_k) * dB[:, k]
        S = self.S0 * np.exp(log_S)

        if return_variance:
            return S, v_full
        return S

    def call_price_mc(self, K, T, n_paths=20_000, N=100, random_state=None):
        S = self.simulate(T, N=N, n_paths=n_paths, random_state=random_state)
        payoff = np.maximum(S[:, -1] - K, 0.0)
        disc = math.exp(-self.r * T)
        return _mc_result(payoff, "rbergomi-mc", discount=disc)

    def implied_vol_mc(self, K, T, n_paths=20_000, N=100, random_state=None, kind="call"):
        res = self.call_price_mc(K, T, n_paths, N, random_state)
        price = res.estimate if kind == "call" else res.estimate - self.S0 * math.exp(-self.q * T) + K * math.exp(-self.r * T)
        return _implied_vol(price, self.S0, K, T, self.r, self.q, kind)


# --------------------------------------------------------------------------- Local vol / Dupire

class LocalVol:
    """Local volatility model: ``dS = (r-q) S dt + sigma(t,S) S dW``."""

    def __init__(self, sigma_fn, S0, r, q=0.0):
        self.sigma_fn = sigma_fn
        self.S0, self.r, self.q = float(S0), float(r), float(q)

    def simulate(self, T, N=252, n_paths=1, random_state=None):
        rng = _rng(random_state)
        dt = T / N
        sqrt_dt = math.sqrt(dt)
        S = np.empty((n_paths, N + 1))
        S[:, 0] = self.S0
        t = 0.0
        for k in range(N):
            sig = np.asarray(self.sigma_fn(t, S[:, k]), dtype=float)
            Z = rng.standard_normal(n_paths)
            S[:, k + 1] = S[:, k] * np.exp((self.r - self.q - 0.5 * sig**2) * dt + sig * sqrt_dt * Z)
            t += dt
        return S

    def price_mc(self, K, T, kind="call", n_paths=50_000, N=252, random_state=None):
        _check_kind(kind)
        S = self.simulate(T, N=N, n_paths=n_paths, random_state=random_state)
        payoff = np.maximum(S[:, -1] - K, 0.0) if kind == "call" else np.maximum(K - S[:, -1], 0.0)
        disc = math.exp(-self.r * T)
        return _mc_result(payoff, "localvol-mc", discount=disc)

    def price_pde(self, K, T, kind="call", n_space=400, n_time=400, S_max=None):
        _check_kind(kind)
        S0, r, q = self.S0, self.r, self.q
        S_max = S_max or 4.0 * max(S0, K)
        dS = S_max / n_space
        dt = T / n_time
        S_grid = np.linspace(0.0, S_max, n_space + 1)

        if kind == "call":
            V = np.maximum(S_grid - K, 0.0)
        else:
            V = np.maximum(K - S_grid, 0.0)

        for step in range(n_time, 0, -1):
            t = (step - 1) * dt
            sig = np.asarray(self.sigma_fn(t, S_grid[1:-1]), dtype=float)
            j = np.arange(1, n_space)
            a = 0.25 * dt * (sig**2 * j**2 - (r - q) * j)
            b = -0.5 * dt * (sig**2 * j**2 + r)
            c = 0.25 * dt * (sig**2 * j**2 + (r - q) * j)

            rhs = a * V[:-2] + (1.0 + b) * V[1:-1] + c * V[2:]
            if kind == "call":
                bc_lo, bc_hi = 0.0, S_max - K * math.exp(-r * (T - t))
            else:
                bc_lo, bc_hi = K * math.exp(-r * (T - t)), 0.0
            rhs[0] += a[0] * bc_lo
            rhs[-1] += c[-1] * bc_hi

            lower = np.concatenate([[0.0], -a[1:]])
            diag = 1.0 - b
            upper = np.concatenate([-c[:-1], [0.0]])
            V_inner = _solve_tridiagonal(lower, diag, upper, rhs)
            V = np.concatenate([[bc_lo], V_inner, [bc_hi]])

        idx = int(round(S0 / dS))
        idx = min(max(idx, 1), n_space - 1)
        x0, x1 = S_grid[idx], S_grid[idx + 1]
        w = (S0 - x0) / (x1 - x0) if x1 > x0 else 0.0
        return float((1 - w) * V[idx] + w * V[idx + 1])

    def delta_pde(self, K, T, kind="call", n_space=400, n_time=400, S_max=None, bump=1e-3):
        h = bump * self.S0
        up = LocalVol(self.sigma_fn, self.S0 + h, self.r, self.q).price_pde(K, T, kind, n_space, n_time, S_max)
        dn = LocalVol(self.sigma_fn, self.S0 - h, self.r, self.q).price_pde(K, T, kind, n_space, n_time, S_max)
        return (up - dn) / (2.0 * h)


class Dupire:
    """Dupire (1994) local volatility extracted from an implied-vol surface,
    via Gatheral's total-implied-variance form of the local-vol formula."""

    def __init__(self, implied_vol_fn, S0, r, q=0.0, dy=1e-3, dT=1e-4):
        self.implied_vol_fn = implied_vol_fn
        self.S0, self.r, self.q = float(S0), float(r), float(q)
        self.dy, self.dT = float(dy), float(dT)

    def _w(self, y, T):
        T = max(T, self.dT)
        F_T = self.S0 * math.exp((self.r - self.q) * T)
        K = F_T * math.exp(y)
        iv = self.implied_vol_fn(K, T)
        return iv**2 * T

    def local_vol(self, K, T):
        K = np.atleast_1d(np.asarray(K, dtype=float))
        T_arr = np.atleast_1d(np.asarray(T, dtype=float))
        out = np.empty(max(len(K), len(T_arr)))
        Kb = np.broadcast_to(K, out.shape)
        Tb = np.broadcast_to(T_arr, out.shape)
        for i in range(out.shape[0]):
            Ti = max(float(Tb[i]), self.dT)
            F_T = self.S0 * math.exp((self.r - self.q) * Ti)
            y = math.log(float(Kb[i]) / F_T)
            dy = self.dy

            w0 = self._w(y, Ti)
            w_yp = self._w(y + dy, Ti)
            w_ym = self._w(y - dy, Ti)
            dw_dy = (w_yp - w_ym) / (2.0 * dy)
            d2w_dy2 = (w_yp - 2.0 * w0 + w_ym) / dy**2

            w_Tp = self._w(y, Ti + self.dT)
            w_Tm = self._w(y, max(Ti - self.dT, self.dT / 2.0))
            dw_dT = (w_Tp - w_Tm) / (Ti + self.dT - max(Ti - self.dT, self.dT / 2.0))

            denom = 1.0 - (y / max(w0, 1e-12)) * dw_dy \
                + 0.25 * (-0.25 - 1.0 / max(w0, 1e-12) + y**2 / max(w0, 1e-12) ** 2) * dw_dy**2 \
                + 0.5 * d2w_dy2
            denom = max(denom, 1e-8)
            sigma_loc2 = max(dw_dT / denom, 0.0)
            out[i] = math.sqrt(sigma_loc2)
        return out if out.size > 1 else float(out[0])

    def to_local_vol(self):
        def sigma_fn(t, S):
            t_eff = max(t, self.dT)
            return self.local_vol(S, t_eff)
        return LocalVol(sigma_fn, self.S0, self.r, self.q)


# --------------------------------------------------------------------------- LVSV

class LVSV:
    """Local-stochastic volatility: Heston-type variance with a leverage
    function ``L(t,S)`` matching a target local-vol surface (Guyon &
    Henry-Labordere particle method, histogram-binned)."""

    def __init__(self, local_vol_fn, S0, v0, kappa, theta, xi, rho, r, q=0.0,
                 leverage=None, n_bins=30):
        self.local_vol_fn = local_vol_fn
        self.S0, self.v0 = float(S0), float(v0)
        self.kappa, self.theta, self.xi, self.rho = float(kappa), float(theta), float(xi), float(rho)
        self.r, self.q = float(r), float(q)
        self.leverage = leverage
        self.n_bins = int(n_bins)
        self.leverage_grid_ = None

    def simulate(self, T, N=252, n_paths=10_000, random_state=None, return_variance=False):
        rng = _rng(random_state)
        dt = T / N
        sqrt_dt = math.sqrt(dt)
        S = np.full(n_paths, self.S0)
        v = np.full(n_paths, self.v0)
        S_hist = np.empty((n_paths, N + 1))
        v_hist = np.empty((n_paths, N + 1))
        S_hist[:, 0], v_hist[:, 0] = S, v
        leverage_grid = []
        t = 0.0

        for k in range(N):
            if self.leverage is not None:
                L = np.asarray(self.leverage(t, S), dtype=float)
            elif self.xi < 1e-10:
                L = np.ones(n_paths)
            else:
                sig_loc = np.asarray(self.local_vol_fn(t, S), dtype=float)
                bins = np.quantile(S, np.linspace(0, 1, self.n_bins + 1))
                bins[0] -= 1e-8
                bins[-1] += 1e-8
                bin_idx = np.clip(np.digitize(S, bins) - 1, 0, self.n_bins - 1)
                v_pos = np.maximum(v, 1e-10)
                bin_ev = np.array([v_pos[bin_idx == b].mean() if np.any(bin_idx == b) else self.theta
                                   for b in range(self.n_bins)])
                bin_centers = 0.5 * (bins[:-1] + bins[1:])
                Ev_at_S = bin_ev[bin_idx]
                L = sig_loc / np.sqrt(np.maximum(Ev_at_S, 1e-10))
                leverage_grid.append((t, bin_centers.copy(), np.sqrt(np.maximum(bin_ev, 1e-10))))

            Z1 = rng.standard_normal(n_paths)
            Z2 = rng.standard_normal(n_paths)
            W2 = self.rho * Z1 + math.sqrt(max(1.0 - self.rho**2, 0.0)) * Z2
            v_pos = np.maximum(v, 0.0)
            v = v + self.kappa * (self.theta - v_pos) * dt + self.xi * np.sqrt(v_pos) * sqrt_dt * W2
            S = S * np.exp((self.r - self.q - 0.5 * (L**2) * v_pos) * dt + L * np.sqrt(v_pos) * sqrt_dt * Z1)
            S_hist[:, k + 1], v_hist[:, k + 1] = S, v
            t += dt

        self.leverage_grid_ = leverage_grid
        if return_variance:
            return S_hist, v_hist
        return S_hist

    def price_mc(self, K, T, kind="call", n_paths=10_000, N=252, random_state=None):
        _check_kind(kind)
        S = self.simulate(T, N=N, n_paths=n_paths, random_state=random_state)
        payoff = np.maximum(S[:, -1] - K, 0.0) if kind == "call" else np.maximum(K - S[:, -1], 0.0)
        disc = math.exp(-self.r * T)
        return _mc_result(payoff, "lvsv-mc", discount=disc)


# --------------------------------------------------------------------------- Variance swap

class VarianceSwap:
    """Fair variance-swap strike (annualized), by closed form (Heston),
    replication from an implied-vol surface, or realized from simulated
    paths."""

    def __init__(self, T, r=0.0, notional=1.0):
        self.T, self.r, self.notional = float(T), float(r), float(notional)

    def fair_strike_heston(self, model):
        return model.expected_integrated_variance(self.T) / self.T

    def fair_strike_from_surface(self, implied_vol_fn, S0, q=0.0, n=4_000, width=8.0):
        T, r = self.T, self.r
        F = S0 * math.exp((r - q) * T)
        atm_vol = implied_vol_fn(F, T)
        lo = F * math.exp(-width * atm_vol * math.sqrt(T))
        hi = F * math.exp(width * atm_vol * math.sqrt(T))
        log_lo, log_hi = math.log(lo), math.log(hi)
        log_K = np.linspace(log_lo, log_hi, n)
        K = np.exp(log_K)
        iv = np.asarray([implied_vol_fn(k, T) for k in K])

        prices = np.empty(n)
        below = K <= F
        prices[below] = _bs_price(S0, K[below], T, r, iv[below], q, "put")
        prices[~below] = _bs_price(S0, K[~below], T, r, iv[~below], q, "call")

        integrand = prices / K**2
        integral = np.trapezoid(integrand, K) if hasattr(np, "trapezoid") else np.trapz(integrand, K)
        return (2.0 * math.exp(r * T) / T) * integral

    def fair_strike_mc(self, paths):
        paths = np.asarray(paths, dtype=float)
        log_ret = np.diff(np.log(paths), axis=1)
        realized_var = np.sum(log_ret**2, axis=1) / self.T
        return _mc_result(realized_var, "variance-swap-mc")

    def payoff(self, paths, strike):
        res = self.fair_strike_mc(paths)
        realized = res.estimate
        return self.notional * (realized - strike)
