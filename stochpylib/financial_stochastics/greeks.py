"""Option sensitivities: analytic Black-Scholes Greeks plus finite-difference
and Monte Carlo estimators that work with *any* pricer.

``Delta``/``Gamma``/.../``Volga`` are plain functions (the library's existing
PascalCase-function precedent, e.g. ``Euler_Maruyama`` in levy_processes.sde)
so :class:`~stochpylib.financial_stochastics.option_pricing.BlackScholes`
exposes them as properties while any other code can call them directly.
"""

import numpy as np

from stochpylib.financial_stochastics._common import (
    _bs_d1d2,
    _check_kind,
    _mc_result,
    _norm_cdf,
    _norm_pdf,
    _rng,
)

__all__ = [
    "Delta", "Gamma", "Vega", "Theta", "Rho", "Vanna", "Volga",
    "Greeks_MC", "Greeks_FD",
]


def Delta(S, K, T, r, sigma, q=0.0, kind="call"):
    _check_kind(kind)
    d1, _ = _bs_d1d2(S, K, T, r, sigma, q)
    disc_q = np.exp(-q * np.asarray(T, dtype=float))
    if kind == "call":
        return disc_q * _norm_cdf(d1)
    return disc_q * (_norm_cdf(d1) - 1.0)


def Gamma(S, K, T, r, sigma, q=0.0, kind="call"):
    _check_kind(kind)
    S = np.asarray(S, dtype=float)
    T = np.asarray(T, dtype=float)
    sigma = np.asarray(sigma, dtype=float)
    d1, _ = _bs_d1d2(S, K, T, r, sigma, q)
    disc_q = np.exp(-q * T)
    return disc_q * _norm_pdf(d1) / (S * sigma * np.sqrt(np.where(T <= 0, 1e-12, T)))


def Vega(S, K, T, r, sigma, q=0.0, kind="call"):
    """Sensitivity to a unit change in volatility (i.e. per 100 vol points)."""
    _check_kind(kind)
    S = np.asarray(S, dtype=float)
    T = np.asarray(T, dtype=float)
    d1, _ = _bs_d1d2(S, K, T, r, sigma, q)
    disc_q = np.exp(-q * T)
    return S * disc_q * _norm_pdf(d1) * np.sqrt(np.where(T <= 0, 1e-12, T))


def Theta(S, K, T, r, sigma, q=0.0, kind="call"):
    """Sensitivity to the passage of time, per unit of ``T`` (annualized)."""
    _check_kind(kind)
    S = np.asarray(S, dtype=float)
    K = np.asarray(K, dtype=float)
    T = np.asarray(T, dtype=float)
    r = np.asarray(r, dtype=float)
    sigma = np.asarray(sigma, dtype=float)
    q = np.asarray(q, dtype=float)
    d1, d2 = _bs_d1d2(S, K, T, r, sigma, q)
    T_safe = np.where(T <= 0, 1e-12, T)
    disc_q = np.exp(-q * T)
    disc_r = np.exp(-r * T)
    term1 = -S * disc_q * _norm_pdf(d1) * sigma / (2.0 * np.sqrt(T_safe))
    if kind == "call":
        return term1 - r * K * disc_r * _norm_cdf(d2) + q * S * disc_q * _norm_cdf(d1)
    return term1 + r * K * disc_r * _norm_cdf(-d2) - q * S * disc_q * _norm_cdf(-d1)


def Rho(S, K, T, r, sigma, q=0.0, kind="call"):
    """Sensitivity to a unit change in the risk-free rate."""
    _check_kind(kind)
    K = np.asarray(K, dtype=float)
    T = np.asarray(T, dtype=float)
    r = np.asarray(r, dtype=float)
    _, d2 = _bs_d1d2(S, K, T, r, sigma, q)
    disc_r = np.exp(-r * T)
    if kind == "call":
        return K * T * disc_r * _norm_cdf(d2)
    return -K * T * disc_r * _norm_cdf(-d2)


def Vanna(S, K, T, r, sigma, q=0.0, kind="call"):
    """``d Delta / d sigma`` = ``d Vega / d S``; identical for call and put."""
    T = np.asarray(T, dtype=float)
    sigma = np.asarray(sigma, dtype=float)
    d1, d2 = _bs_d1d2(S, K, T, r, sigma, q)
    disc_q = np.exp(-q * T)
    return -disc_q * _norm_pdf(d1) * d2 / np.where(sigma <= 0, 1e-12, sigma)


def Volga(S, K, T, r, sigma, q=0.0, kind="call"):
    """``d Vega / d sigma`` (vomma); identical for call and put."""
    S = np.asarray(S, dtype=float)
    T = np.asarray(T, dtype=float)
    sigma = np.asarray(sigma, dtype=float)
    d1, d2 = _bs_d1d2(S, K, T, r, sigma, q)
    disc_q = np.exp(-q * T)
    vega = S * disc_q * _norm_pdf(d1) * np.sqrt(np.where(T <= 0, 1e-12, T))
    return vega * d1 * d2 / np.where(sigma <= 0, 1e-12, sigma)


_ALL_GREEKS = {
    "delta": Delta, "gamma": Gamma, "vega": Vega,
    "theta": Theta, "rho": Rho, "vanna": Vanna, "volga": Volga,
}


class Greeks_FD:
    """Central-difference Greeks for *any* pricer ``price_fn(S, K, T, r, sigma)``.

    Works on pricers with no analytic sensitivities (trees, PDE solvers,
    Monte Carlo point estimates), unlike the closed-form functions above.
    """

    def __init__(self, price_fn, h_S=None, h_sigma=1e-4, h_T=1e-5, h_r=1e-5):
        self.price_fn = price_fn
        self.h_S = h_S
        self.h_sigma = float(h_sigma)
        self.h_T = float(h_T)
        self.h_r = float(h_r)

    def compute(self, S, K, T, r, sigma):
        S, K, T, r, sigma = float(S), float(K), float(T), float(r), float(sigma)
        hS = self.h_S if self.h_S is not None else max(1e-4 * S, 1e-6)
        hs = self.h_sigma
        hT = min(self.h_T, T / 2) if T > 0 else self.h_T
        hr = self.h_r
        f = self.price_fn

        p0 = f(S, K, T, r, sigma)
        p_Sp, p_Sm = f(S + hS, K, T, r, sigma), f(S - hS, K, T, r, sigma)
        p_sp, p_sm = f(S, K, T, r, sigma + hs), f(S, K, T, r, sigma - hs)
        p_Tp, p_Tm = f(S, K, T + hT, r, sigma), f(S, K, max(T - hT, 1e-12), r, sigma)
        p_rp, p_rm = f(S, K, T, r + hr, sigma), f(S, K, T, r - hr, sigma)
        p_Sp_sp = f(S + hS, K, T, r, sigma + hs)
        p_Sm_sp = f(S - hS, K, T, r, sigma + hs)
        p_Sp_sm = f(S + hS, K, T, r, sigma - hs)
        p_Sm_sm = f(S - hS, K, T, r, sigma - hs)

        delta = (p_Sp - p_Sm) / (2 * hS)
        gamma = (p_Sp - 2 * p0 + p_Sm) / hS**2
        vega = (p_sp - p_sm) / (2 * hs)
        theta = -(p_Tp - p_Tm) / (2 * hT)
        rho = (p_rp - p_rm) / (2 * hr)
        vanna = ((p_Sp_sp - p_Sm_sp) - (p_Sp_sm - p_Sm_sm)) / (4 * hS * hs)
        volga = (p_sp - 2 * p0 + p_sm) / hs**2

        return {
            "delta": float(delta), "gamma": float(gamma), "vega": float(vega),
            "theta": float(theta), "rho": float(rho), "vanna": float(vanna),
            "volga": float(volga),
        }


class Greeks_MC:
    """Monte Carlo sensitivities for the vanilla Black-Scholes payoff.

    Three estimators: ``"pathwise"`` (delta/vega/rho, biased-free but
    undefined for the discontinuous gamma), ``"likelihood"`` (all five,
    including gamma, via the score-function/likelihood-ratio method), and
    ``"bump"`` (central finite differences driven by common random numbers).
    """

    def __init__(self, S, K, T, r, sigma, q=0.0, kind="call"):
        _check_kind(kind)
        self.S, self.K, self.T = float(S), float(K), float(T)
        self.r, self.sigma, self.q, self.kind = float(r), float(sigma), float(q), kind

    def _terminal(self, Z):
        S, T, r, sigma, q = self.S, self.T, self.r, self.sigma, self.q
        return S * np.exp((r - q - 0.5 * sigma**2) * T + sigma * np.sqrt(T) * Z)

    def _payoff(self, S_T):
        K = self.K
        return np.maximum(S_T - K, 0.0) if self.kind == "call" else np.maximum(K - S_T, 0.0)

    def compute(self, n_paths=200_000, method="pathwise", random_state=None,
                greeks=("delta", "gamma", "vega", "rho")):
        rng = _rng(random_state)
        S, K, T, r, sigma, q = self.S, self.K, self.T, self.r, self.sigma, self.q
        disc = np.exp(-r * T)
        sqrtT = np.sqrt(T)
        sign = 1.0 if self.kind == "call" else -1.0

        if method == "bump":
            return self._bump(n_paths, greeks, rng)

        Z = rng.standard_normal(n_paths)
        S_T = self._terminal(Z)
        payoff = self._payoff(S_T)
        itm = (S_T > K) if self.kind == "call" else (S_T < K)

        out = {}
        if method == "pathwise":
            if "delta" in greeks:
                out["delta"] = _mc_result(disc * itm * S_T / S, "pathwise MC", {"greek": "delta"})
            if "vega" in greeks:
                integrand = itm * S_T * (np.log(S_T / S) - (r - q + 0.5 * sigma**2) * T) / sigma
                out["vega"] = _mc_result(disc * integrand, "pathwise MC", {"greek": "vega"})
            if "rho" in greeks:
                out["rho"] = _mc_result(disc * T * K * itm, "pathwise MC", {"greek": "rho"})
            if "gamma" in greeks:
                out["gamma"] = self._likelihood_greek("gamma", Z, S_T, payoff, disc, sqrtT, rng)
            if "theta" in greeks:
                out["theta"] = self._likelihood_greek("theta", Z, S_T, payoff, disc, sqrtT, rng)
        elif method == "likelihood":
            for g in greeks:
                out[g] = self._likelihood_greek(g, Z, S_T, payoff, disc, sqrtT, rng)
        else:
            raise ValueError("method must be 'pathwise', 'likelihood', or 'bump'")
        return out

    def _likelihood_greek(self, greek, Z, S_T, payoff, disc, sqrtT, rng):
        S, sigma, T, r = self.S, self.sigma, self.T, self.r
        if greek == "delta":
            score = Z / (S * sigma * sqrtT)
        elif greek == "gamma":
            score = (Z**2 - 1.0) / (S**2 * sigma**2 * T) - Z / (S**2 * sigma * sqrtT)
        elif greek == "vega":
            score = (Z**2 - 1.0) / sigma - Z * sqrtT
        elif greek == "rho":
            score = T * np.ones_like(Z)
        elif greek == "theta":
            score = -((Z**2 - 1.0) / (2.0 * T) - Z * sigma / (2.0 * sqrtT)) if T > 0 else np.zeros_like(Z)
        else:
            raise ValueError(f"unknown greek {greek!r}")
        return _mc_result(disc * payoff * score, "likelihood-ratio MC", {"greek": greek})

    def _bump(self, n_paths, greeks, rng):
        S, K, T, r, sigma, q = self.S, self.K, self.T, self.r, self.sigma, self.q
        Z = rng.standard_normal(n_paths)
        disc = np.exp(-r * T)

        def priced(S_, T_, r_, sigma_):
            drift = (r_ - q - 0.5 * sigma_**2) * T_
            S_T = S_ * np.exp(drift + sigma_ * np.sqrt(T_) * Z)
            return np.exp(-r_ * T_) * self._payoff_at(S_T, K)

        out = {}
        if "delta" in greeks:
            hS = 1e-3 * S
            out["delta"] = _mc_result((priced(S + hS, T, r, sigma) - priced(S - hS, T, r, sigma)) / (2 * hS),
                                      "bump MC", {"greek": "delta"})
        if "gamma" in greeks:
            hS = 1e-2 * S
            p0 = priced(S, T, r, sigma)
            out["gamma"] = _mc_result((priced(S + hS, T, r, sigma) - 2 * p0 + priced(S - hS, T, r, sigma)) / hS**2,
                                      "bump MC", {"greek": "gamma"})
        if "vega" in greeks:
            hs = 1e-4
            out["vega"] = _mc_result((priced(S, T, r, sigma + hs) - priced(S, T, r, sigma - hs)) / (2 * hs),
                                     "bump MC", {"greek": "vega"})
        if "rho" in greeks:
            hr = 1e-5
            out["rho"] = _mc_result((priced(S, T, r + hr, sigma) - priced(S, T, r - hr, sigma)) / (2 * hr),
                                    "bump MC", {"greek": "rho"})
        return out

    def _payoff_at(self, S_T, K):
        return np.maximum(S_T - K, 0.0) if self.kind == "call" else np.maximum(K - S_T, 0.0)
