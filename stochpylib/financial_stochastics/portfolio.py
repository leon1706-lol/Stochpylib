"""Portfolio construction: covariance estimation, mean-variance optimization,
Black-Litterman, and risk parity.
"""

import math

import numpy as np
from scipy import optimize

__all__ = [
    "PortfolioOptimization", "MeanVariance", "BlackLitterman",
    "RiskParity", "CovarianceEstimation",
]


class CovarianceEstimation:
    """Sample, EWMA (RiskMetrics), or Ledoit-Wolf (2004) shrinkage covariance."""

    def __init__(self, method="sample", lam=0.94, shrinkage=None):
        self.method, self.lam, self.shrinkage = method, float(lam), shrinkage
        self.covariance_ = None
        self.correlation_ = None
        self.shrinkage_ = None

    def fit(self, returns):
        R = np.asarray(returns, dtype=float)
        n, p = R.shape
        if self.method == "sample":
            self.covariance_ = np.cov(R, rowvar=False, ddof=1)
        elif self.method == "ewma":
            X = R - R.mean(axis=0)
            S = np.outer(X[0], X[0])
            for t in range(1, n):
                S = self.lam * S + (1.0 - self.lam) * np.outer(X[t], X[t])
            self.covariance_ = S
        elif self.method in ("ledoit_wolf", "shrinkage"):
            S = np.cov(R, rowvar=False, ddof=1)
            mu_bar = np.trace(S) / p
            target = mu_bar * np.eye(p)
            if self.method == "shrinkage" and self.shrinkage is not None:
                delta = float(self.shrinkage)
            else:
                # Ledoit-Wolf (2004) sec. 2: b_bar^2 = (1/n^2) sum_t ||x_t x_t' - S||_F^2
                # (an average-of-n-terms estimator of an O(1/n) asymptotic
                # variance, hence the *second* factor of n) — dividing by n
                # only left the shrinkage saturated at 1.0 for any non-tiny n.
                X = R - R.mean(axis=0)
                delta2 = np.sum((S - target) ** 2)
                pi_hat = 0.0
                for t in range(n):
                    outer_t = np.outer(X[t], X[t])
                    pi_hat += np.sum((outer_t - S) ** 2)
                pi_hat /= n * n
                beta2 = min(pi_hat, delta2)
                delta = beta2 / delta2 if delta2 > 0 else 0.0
                delta = float(np.clip(delta, 0.0, 1.0))
            self.covariance_ = (1.0 - delta) * S + delta * target
            self.shrinkage_ = delta
        else:
            raise ValueError("method must be 'sample', 'ewma', 'ledoit_wolf', or 'shrinkage'")

        d = np.sqrt(np.diag(self.covariance_))
        outer_d = np.outer(d, d)
        self.correlation_ = np.where(outer_d > 0, self.covariance_ / np.where(outer_d == 0, 1, outer_d), 0.0)
        return self


class MeanVariance:
    """Markowitz mean-variance portfolio construction."""

    def __init__(self, mu, cov, risk_free=0.0, long_only=True):
        self.mu = np.asarray(mu, dtype=float)
        self.cov = np.asarray(cov, dtype=float)
        self.risk_free = float(risk_free)
        self.long_only = long_only
        self.n = len(self.mu)

    def portfolio_return(self, w):
        return float(np.dot(w, self.mu))

    def portfolio_volatility(self, w):
        return float(math.sqrt(np.dot(w, self.cov @ w)))

    def sharpe(self, w):
        vol = self.portfolio_volatility(w)
        return (self.portfolio_return(w) - self.risk_free) / vol if vol > 0 else 0.0

    def _closed_form_unconstrained(self, target_return=None, max_sharpe=False):
        inv_cov = np.linalg.inv(self.cov)
        ones = np.ones(self.n)
        A = ones @ inv_cov @ ones
        B = ones @ inv_cov @ self.mu
        C = self.mu @ inv_cov @ self.mu
        D = A * C - B**2
        if max_sharpe:
            excess = self.mu - self.risk_free
            w = inv_cov @ excess
            return w / np.sum(w)
        if target_return is None:
            return (inv_cov @ ones) / A
        m = target_return
        w = ((C - B * m) * (inv_cov @ ones) + (A * m - B) * (inv_cov @ self.mu)) / D
        return w

    def _slsqp(self, objective, target_return=None):
        n = self.n
        x0 = np.full(n, 1.0 / n)
        constraints = [{"type": "eq", "fun": lambda w: np.sum(w) - 1.0}]
        if target_return is not None:
            constraints.append({"type": "eq", "fun": lambda w: np.dot(w, self.mu) - target_return})
        bounds = [(0.0, 1.0)] * n if self.long_only else [(None, None)] * n
        res = optimize.minimize(objective, x0, method="SLSQP", bounds=bounds, constraints=constraints,
                                options={"maxiter": 500, "ftol": 1e-12})
        if not res.success:
            raise RuntimeError(f"portfolio optimization failed: {res.message}")
        return res.x

    def min_variance(self):
        if not self.long_only:
            return self._closed_form_unconstrained()
        return self._slsqp(lambda w: np.dot(w, self.cov @ w))

    def max_sharpe(self):
        if not self.long_only:
            return self._closed_form_unconstrained(max_sharpe=True)
        return self._slsqp(lambda w: -self.sharpe(w))

    def efficient(self, target_return):
        if not self.long_only:
            return self._closed_form_unconstrained(target_return=target_return)
        return self._slsqp(lambda w: np.dot(w, self.cov @ w), target_return=target_return)

    def efficient_frontier(self, n=50):
        lo, hi = self.mu.min(), self.mu.max()
        targets = np.linspace(lo, hi, n)
        W = np.empty((n, self.n))
        rets = np.empty(n)
        vols = np.empty(n)
        for i, t in enumerate(targets):
            try:
                w = self.efficient(t)
            except (RuntimeError, np.linalg.LinAlgError):
                w = np.full(self.n, np.nan)
            W[i] = w
            rets[i] = self.portfolio_return(w)
            vols[i] = self.portfolio_volatility(w)
        return rets, vols, W


class PortfolioOptimization:
    """Fluent fit/optimize interface over sample moments."""

    def __init__(self, expected_returns=None, covariance=None, risk_free=0.0,
                 bounds=(0.0, 1.0), budget=True):
        self.expected_returns = None if expected_returns is None else np.asarray(expected_returns, dtype=float)
        self.covariance = None if covariance is None else np.asarray(covariance, dtype=float)
        self.risk_free = float(risk_free)
        self.bounds, self.budget = bounds, budget
        self.mu_ = self.cov_ = None

    def fit(self, returns, annualize=None):
        R = np.asarray(returns, dtype=float)
        cov_est = CovarianceEstimation("sample").fit(R)
        scale = annualize if annualize else 1.0
        self.mu_ = R.mean(axis=0) * scale
        self.cov_ = cov_est.covariance_ * scale
        if self.expected_returns is None:
            self.expected_returns = self.mu_
        if self.covariance is None:
            self.covariance = self.cov_
        return self

    def _mv(self):
        long_only = self.bounds == (0.0, 1.0)
        return MeanVariance(self.expected_returns, self.covariance, self.risk_free, long_only)

    def optimize(self, objective="max_sharpe", target=None, risk_aversion=1.0):
        mv = self._mv()
        if objective == "max_sharpe":
            w = mv.max_sharpe()
        elif objective == "min_variance":
            w = mv.min_variance()
        elif objective == "target_return":
            if target is None:
                raise ValueError("target_return objective requires `target`")
            w = mv.efficient(target)
        elif objective == "max_utility":
            n = len(self.expected_returns)
            if self.bounds == (0.0, 1.0):
                def neg_util(w):
                    return -(np.dot(w, self.expected_returns) - 0.5 * risk_aversion * np.dot(w, self.covariance @ w))
                w = mv._slsqp(neg_util)
            else:
                inv_cov = np.linalg.inv(self.covariance)
                ones = np.ones(n)
                gamma = (ones @ inv_cov @ self.expected_returns - risk_aversion) / (ones @ inv_cov @ ones)
                w = (inv_cov @ (self.expected_returns - gamma * ones)) / risk_aversion
        else:
            raise ValueError("unknown objective")
        return w

    def portfolio_return(self, w):
        return float(np.dot(w, self.expected_returns))

    def portfolio_volatility(self, w):
        return float(math.sqrt(np.dot(w, self.covariance @ w)))

    def sharpe(self, w):
        vol = self.portfolio_volatility(w)
        return (self.portfolio_return(w) - self.risk_free) / vol if vol > 0 else 0.0


class BlackLitterman:
    """Black-Litterman (1992) posterior returns from equilibrium plus views
    (He-Litterman 1999 parametrization)."""

    def __init__(self, cov, market_weights, risk_aversion=2.5, tau=0.05, risk_free=0.0):
        self.cov = np.asarray(cov, dtype=float)
        self.market_weights = np.asarray(market_weights, dtype=float)
        self.risk_aversion, self.tau, self.risk_free = float(risk_aversion), float(tau), float(risk_free)

    def equilibrium_returns(self):
        return self.risk_aversion * self.cov @ self.market_weights

    def posterior(self, P=None, Q=None, omega=None):
        Pi = self.equilibrium_returns()
        if P is None or len(np.atleast_2d(P)) == 0 or np.size(P) == 0:
            cov_bl = (1.0 + self.tau) * self.cov
            return Pi, cov_bl
        P = np.atleast_2d(np.asarray(P, dtype=float))
        Q = np.asarray(Q, dtype=float).reshape(-1)
        tau_cov = self.tau * self.cov
        if omega is None:
            omega = np.diag(np.diag(P @ tau_cov @ P.T))
        else:
            omega = np.asarray(omega, dtype=float)
            if omega.ndim == 0:
                omega = omega * np.eye(len(Q))

        inv_tau_cov = np.linalg.inv(tau_cov)
        inv_omega = np.linalg.inv(omega)
        M = np.linalg.inv(inv_tau_cov + P.T @ inv_omega @ P)
        mu_bl = M @ (inv_tau_cov @ Pi + P.T @ inv_omega @ Q)
        cov_bl = self.cov + M
        return mu_bl, cov_bl

    def optimal_weights(self, mu_bl=None, cov_bl=None):
        if mu_bl is None or cov_bl is None:
            mu_bl, cov_bl = self.posterior()
        return np.linalg.inv(self.risk_aversion * cov_bl) @ mu_bl


class RiskParity:
    """Equal-risk-contribution portfolio (Spinu 2013 convex formulation via
    cyclical coordinate descent)."""

    def __init__(self, cov, budgets=None):
        self.cov = np.asarray(cov, dtype=float)
        n = self.cov.shape[0]
        self.budgets = np.full(n, 1.0 / n) if budgets is None else np.asarray(budgets, dtype=float)
        self.n = n

    def optimize(self, tol=1e-10, max_iter=10_000):
        """Cyclical coordinate descent on Spinu's (2013) convex risk-parity
        objective ``min 1/2 w'Sigma w - sum(b_i ln w_i)`` over the *unnormalized*
        ``w > 0`` (its minimizer is unique up to no scaling ambiguity, since
        the log-barrier fixes the scale) — normalizing to the simplex only
        happens once, after convergence. Renormalizing every sweep (as a
        naive implementation might) changes each coordinate's fixed-point
        equation mid-iteration and prevents convergence to equal risk
        contributions.
        """
        Sigma = self.cov
        n = self.n
        w = np.full(n, 1.0 / n)
        b = self.budgets
        for _ in range(max_iter):
            w_old = w.copy()
            for i in range(n):
                c_i = np.dot(Sigma[i, :], w) - Sigma[i, i] * w[i]
                sigma_ii = Sigma[i, i]
                disc = c_i**2 + 4.0 * sigma_ii * b[i]
                w[i] = max((-c_i + math.sqrt(max(disc, 0.0))) / (2.0 * sigma_ii), 1e-12)
            if np.max(np.abs(w - w_old)) < tol * max(1.0, np.max(np.abs(w_old))):
                break
        w = w / w.sum()
        self.weights_ = w
        return w

    def risk_contributions(self, w):
        w = np.asarray(w, dtype=float)
        port_vol = math.sqrt(np.dot(w, self.cov @ w))
        marginal = self.cov @ w
        return w * marginal / port_vol if port_vol > 0 else np.zeros_like(w)
