"""Credit risk: hazard-rate/default-intensity models, CDS pricing, structural
(Merton) credit risk, rating migration, and portfolio credit-loss models
(independent-default and copula-dependent).
"""

import math

import numpy as np
from scipy import optimize

from stochpylib.copulas.elliptical import GaussianCopula, StudentTCopula
from stochpylib.financial_stochastics._common import (
    _bs_price,
    _matrix_exp,
    _matrix_log,
    _mc_result,
    _norm_cdf,
    _norm_ppf,
    _rng,
)
from stochpylib.montecarlo._result import MCResult
from stochpylib.montecarlo.applications import RiskResult, risk_analysis

__all__ = [
    "CreditRiskModel", "DefaultIntensity", "CreditMigration", "CDSPricing",
    "MertonCreditModel", "CopulaCreditModel",
]


class DefaultIntensity:
    """Piecewise-constant hazard-rate curve ``lambda(t)``."""

    def __init__(self, times, hazards):
        self.times = np.asarray(times, dtype=float)
        self.hazards = np.asarray(hazards, dtype=float)
        if len(self.times) != len(self.hazards):
            raise ValueError("times and hazards must have the same length")

    @classmethod
    def flat(cls, lam):
        return cls([0.0], [lam])

    @classmethod
    def from_survival(cls, times, probs):
        times = np.asarray(times, dtype=float)
        probs = np.asarray(probs, dtype=float)
        edges = np.concatenate([[0.0], times])
        log_s = np.concatenate([[0.0], np.log(probs)])
        hazards = -(np.diff(log_s)) / np.diff(edges)
        return cls(edges[:-1], hazards)

    def intensity(self, t):
        t = np.atleast_1d(np.asarray(t, dtype=float))
        idx = np.searchsorted(self.times, t, side="right") - 1
        idx = np.clip(idx, 0, len(self.hazards) - 1)
        out = self.hazards[idx]
        return out if out.size > 1 else float(out[0])

    def cumulative_hazard(self, t):
        t = np.atleast_1d(np.asarray(t, dtype=float))
        out = np.empty_like(t)
        for i, ti in enumerate(t):
            if ti <= 0:
                out[i] = 0.0
                continue
            edges = np.sort(np.unique(np.concatenate([self.times[self.times < ti], [ti]])))
            total = 0.0
            for k in range(len(edges) - 1):
                idx = np.searchsorted(self.times, edges[k], side="right") - 1
                idx = max(idx, 0)
                total += self.hazards[idx] * (edges[k + 1] - edges[k])
            out[i] = total
        return out if out.size > 1 else float(out[0])

    def survival(self, t):
        return np.exp(-self.cumulative_hazard(t))

    def default_prob(self, t):
        return 1.0 - self.survival(t)

    def simulate_default_times(self, n, random_state=None):
        rng = _rng(random_state)
        E = rng.exponential(1.0, n)
        out = np.empty(n)
        max_t = self.times[-1] + 1000.0 / max(self.hazards[-1], 1e-6)
        grid = np.linspace(0.0, max_t, 20_000)
        cum = np.array([self.cumulative_hazard(t) for t in grid])
        out = np.interp(E, cum, grid)
        return out

    def fit(self, default_times, observed=None, times=None):
        default_times = np.asarray(default_times, dtype=float)
        if times is None:
            times = np.array([0.0, np.median(default_times), default_times.max()])
        times = np.asarray(times, dtype=float)
        observed = np.ones(len(default_times), dtype=bool) if observed is None else np.asarray(observed, dtype=bool)
        hazards = np.empty(len(times))
        for k in range(len(times)):
            t0 = times[k]
            t1 = times[k + 1] if k + 1 < len(times) else np.inf
            exposure = np.sum(np.clip(np.minimum(default_times, t1) - t0, 0.0, None))
            defaults = np.sum(observed & (default_times >= t0) & (default_times < t1))
            hazards[k] = defaults / exposure if exposure > 0 else 0.0
        self.times, self.hazards = times, hazards
        return self


class CDSPricing:
    """Reduced-form CDS pricing from a hazard curve (or a flat hazard rate)
    with a piecewise-constant discount curve."""

    def __init__(self, hazard, r, recovery=0.4, payment_freq=4, accrual=True):
        self.hazard = hazard if isinstance(hazard, DefaultIntensity) else DefaultIntensity.flat(hazard)
        self.discount = r if callable(r) else (lambda t, r=r: math.exp(-r * t))
        self.recovery = float(recovery)
        self.payment_freq = int(payment_freq)
        self.accrual = bool(accrual)

    def _payment_times(self, T):
        n = int(round(T * self.payment_freq))
        return np.linspace(T / n, T, n)

    def premium_leg_pv01(self, T):
        times = self._payment_times(T)
        dt = times[0]
        prev = 0.0
        pv01 = 0.0
        for t in times:
            Q_t = self.hazard.survival(t)
            Q_prev = self.hazard.survival(prev) if prev > 0 else 1.0
            pv01 += dt * self.discount(t) * Q_t
            if self.accrual:
                pv01 += 0.5 * dt * self.discount(t) * (Q_prev - Q_t)
            prev = t
        return pv01

    def protection_leg(self, T, n_steps=None):
        n_steps = n_steps or max(int(T * 52), 10)
        grid = np.linspace(0.0, T, n_steps + 1)
        Q = np.array([self.hazard.survival(t) if t > 0 else 1.0 for t in grid])
        mid = 0.5 * (grid[:-1] + grid[1:])
        disc_mid = np.array([self.discount(t) for t in mid])
        return float((1.0 - self.recovery) * np.sum(disc_mid * (Q[:-1] - Q[1:])))

    def par_spread(self, T):
        return self.protection_leg(T) / self.premium_leg_pv01(T)

    def pv(self, T, spread, notional=1.0):
        return notional * (self.protection_leg(T) - spread * self.premium_leg_pv01(T))

    def implied_hazard(self, T, spread):
        def resid(lam):
            cds = CDSPricing(max(lam, 1e-8), self.discount, self.recovery, self.payment_freq, self.accrual)
            return cds.pv(T, spread)
        return optimize.brentq(resid, 1e-6, 5.0, xtol=1e-10)

    def bootstrap(self, maturities, spreads):
        maturities = np.asarray(maturities, dtype=float)
        spreads = np.asarray(spreads, dtype=float)
        times, hazards = [], []
        prev_T = 0.0
        for T, spread in zip(maturities, spreads):
            def resid(lam, T=T, spread=spread, prev_hazards=list(hazards), prev_times=list(times), prev_T=prev_T):
                curve = DefaultIntensity(prev_times + [prev_T], prev_hazards + [lam])
                cds = CDSPricing(curve, self.discount, self.recovery, self.payment_freq, self.accrual)
                return cds.pv(T, spread)
            lam = optimize.brentq(resid, 1e-6, 5.0, xtol=1e-10)
            times.append(prev_T)
            hazards.append(lam)
            prev_T = T
        return DefaultIntensity(times, hazards)


class MertonCreditModel:
    """Merton (1974) structural credit-risk model: equity as a call option
    on firm value ``V``, debt as the residual."""

    def __init__(self, V0, D, T, r, sigma_V, mu=None):
        self.V0, self.D, self.T = float(V0), float(D), float(T)
        self.r, self.sigma_V = float(r), float(sigma_V)
        self.mu = float(mu) if mu is not None else None

    def _d1_d2(self, rate):
        d1 = (math.log(self.V0 / self.D) + (rate + 0.5 * self.sigma_V**2) * self.T) / (self.sigma_V * math.sqrt(self.T))
        d2 = d1 - self.sigma_V * math.sqrt(self.T)
        return d1, d2

    def equity_value(self):
        return _bs_price(self.V0, self.D, self.T, self.r, self.sigma_V, 0.0, "call")

    def debt_value(self):
        return self.V0 - self.equity_value()

    def default_probability(self, risk_neutral=True):
        rate = self.r if risk_neutral else (self.mu if self.mu is not None else self.r)
        _, d2 = self._d1_d2(rate)
        return float(_norm_cdf(-d2))

    def distance_to_default(self):
        rate = self.mu if self.mu is not None else self.r
        _, d2 = self._d1_d2(rate)
        return float(d2)

    def credit_spread(self):
        debt = self.debt_value()
        return -math.log(debt / self.D) / self.T - self.r

    def recovery_rate(self):
        # Expected recovery given default: E[V_T | V_T < D] / D under Q.
        d1, d2 = self._d1_d2(self.r)
        exp_VT_given_default = self.V0 * math.exp(self.r * self.T) * _norm_cdf(-d1) / _norm_cdf(-d2) \
            if _norm_cdf(-d2) > 1e-14 else self.D
        return float(min(exp_VT_given_default / self.D, 1.0))

    @classmethod
    def from_equity(cls, E, sigma_E, D, T, r):
        def equations(params):
            V0, sigma_V = params
            if V0 <= 0 or sigma_V <= 0:
                return [1e6, 1e6]
            m = cls(V0, D, T, r, sigma_V)
            d1, _ = m._d1_d2(r)
            eq_price = m.equity_value() - E
            eq_vol = _norm_cdf(d1) * sigma_V * V0 - sigma_E * E
            return [eq_price, eq_vol]

        x0 = [E + D * math.exp(-r * T), sigma_E * E / (E + D * math.exp(-r * T))]
        sol = optimize.fsolve(equations, x0, full_output=False)
        return cls(float(sol[0]), D, T, r, float(sol[1]))


class CreditMigration:
    """Discrete-time rating-transition matrix model (last state absorbing
    default by convention)."""

    def __init__(self, transition_matrix=None, states=None):
        self.transition_matrix_ = None if transition_matrix is None else np.asarray(transition_matrix, dtype=float)
        self.states = states

    def fit(self, sequences):
        sequences = [np.asarray(s) for s in sequences]
        n_states = int(max(s.max() for s in sequences)) + 1
        counts = np.zeros((n_states, n_states))
        for s in sequences:
            for a, b in zip(s[:-1], s[1:]):
                counts[a, b] += 1
        row_sums = counts.sum(axis=1, keepdims=True)
        P = np.where(row_sums > 0, counts / np.where(row_sums == 0, 1, row_sums), 0.0)
        P[-1, :] = 0.0
        P[-1, -1] = 1.0
        self.transition_matrix_ = P
        return self

    def n_step(self, n):
        return np.linalg.matrix_power(self.transition_matrix_, n)

    def cumulative_default_prob(self, state, n):
        Pn = self.n_step(n)
        return float(Pn[state, -1])

    def state_distribution(self, state, n):
        Pn = self.n_step(n)
        return Pn[state, :]

    def generator(self, regularize=True):
        """Matrix logarithm of the transition matrix; with ``regularize``,
        applies Israel-Rosenthal-Wei: any negative off-diagonal entry (an
        artifact of a non-embeddable ``P``) is zeroed and its (negative)
        value moved onto the diagonal, which preserves the zero-row-sum
        generator constraint while leaving already-valid rows untouched."""
        P = self.transition_matrix_
        G = _matrix_log(P)
        if regularize:
            n = G.shape[0]
            for i in range(n):
                for j in range(n):
                    if j != i and G[i, j] < 0:
                        G[i, i] += G[i, j]
                        G[i, j] = 0.0
        return G

    def simulate(self, state, n_steps, n_paths=1, random_state=None):
        rng = _rng(random_state)
        P = self.transition_matrix_
        n_states = P.shape[0]
        out = np.empty((n_paths, n_steps + 1), dtype=int)
        out[:, 0] = state
        cur = np.full(n_paths, state)
        for t in range(n_steps):
            U = rng.random(n_paths)
            cdf = np.cumsum(P[cur, :], axis=1)
            nxt = (U[:, None] < cdf).argmax(axis=1)
            cur = nxt
            out[:, t + 1] = cur
        return out


class CreditRiskModel:
    """Independent-default portfolio credit-loss model (homogeneous or
    heterogeneous exposures)."""

    def __init__(self, exposures, pds, lgds=0.6):
        self.exposures = np.asarray(exposures, dtype=float)
        self.pds = np.asarray(pds, dtype=float)
        n = len(self.exposures)
        self.lgds = np.full(n, lgds, dtype=float) if np.isscalar(lgds) else np.asarray(lgds, dtype=float)

    def expected_loss(self):
        return float(np.sum(self.exposures * self.pds * self.lgds))

    def unexpected_loss(self):
        var_loss = np.sum((self.exposures * self.lgds) ** 2 * self.pds * (1.0 - self.pds))
        return float(math.sqrt(var_loss))

    def simulate_losses(self, n, random_state=None):
        rng = _rng(random_state)
        defaults = rng.random((n, len(self.pds))) < self.pds[None, :]
        return defaults @ (self.exposures * self.lgds)

    def loss_distribution(self, confidence=0.99, n=100_000, random_state=None):
        losses = self.simulate_losses(n, random_state)
        pnl = -losses
        res = risk_analysis(pnl, confidence)
        return res

    def vasicek_quantile(self, alpha, rho):
        pd_bar = float(np.mean(self.pds))
        z_pd, z_a = _norm_ppf(pd_bar), _norm_ppf(alpha)
        p_alpha = _norm_cdf((z_pd + math.sqrt(rho) * z_a) / math.sqrt(1.0 - rho))
        total_exposure = float(np.sum(self.exposures * self.lgds))
        return float(p_alpha) * total_exposure


class CopulaCreditModel(CreditRiskModel):
    """Dependent default times via a Gaussian or Student-t copula on the
    default indicators (single-period asset-correlation model)."""

    def __init__(self, exposures, pds, lgds, correlation, copula="gaussian", df=4):
        super().__init__(exposures, pds, lgds)
        self.correlation = np.asarray(correlation, dtype=float)
        self.copula_kind = copula
        self.df = float(df)

    @classmethod
    def one_factor(cls, exposures, pds, lgds, rho, copula="gaussian", df=4):
        n = len(exposures)
        R = np.full((n, n), rho)
        np.fill_diagonal(R, 1.0)
        return cls(exposures, pds, lgds, R, copula, df)

    def simulate_losses(self, n, random_state=None):
        d = len(self.pds)
        if self.copula_kind == "gaussian":
            cop = GaussianCopula(dimension=d)
        elif self.copula_kind == "t":
            cop = StudentTCopula(dimension=d, df=self.df)
            cop.df_ = self.df
        else:
            raise ValueError("copula must be 'gaussian' or 't'")
        cop.correlation_ = self.correlation
        cop.dimension = d
        U = cop.sample(n, random_state=random_state)
        defaults = U <= self.pds[None, :]
        return defaults @ (self.exposures * self.lgds)
