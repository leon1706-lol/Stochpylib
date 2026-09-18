"""Value-at-Risk, Expected Shortfall, stress testing, and scenario analysis.

Convention (matching :func:`stochpylib.montecarlo.applications.risk_analysis`):
losses are positive numbers (``loss = -return``); VaR/ES are quantiles/tail
means of the *loss* distribution.
"""

import math

import numpy as np
from scipy import integrate, optimize, special

from stochpylib.distributions.continuous import Student_t
from stochpylib.distributions.multivariate import MultivariateNormal
from stochpylib.financial_stochastics._common import _norm_cdf, _norm_pdf, _norm_ppf, _rng
from stochpylib.montecarlo._result import MCResult
from stochpylib.montecarlo.applications import RiskResult, risk_analysis

__all__ = [
    "ValueAtRisk", "ExpectedShortfall", "ConditionalVaR", "HistoricalVaR",
    "ParametricVaR", "StressTest", "ScenarioAnalysis",
]


def _overlapping_sums(returns, horizon):
    if horizon <= 1:
        return returns
    n = len(returns) - horizon + 1
    if n <= 0:
        raise ValueError("not enough observations for the requested horizon")
    kernel = np.ones(horizon)
    return np.convolve(returns, kernel, mode="valid")[:n]


def _age_weighted_quantile(losses, weights, alpha):
    order = np.argsort(losses)
    sl, sw = losses[order], weights[order]
    csum = np.cumsum(sw) / sw.sum()
    idx = np.searchsorted(csum, alpha)
    idx = min(idx, len(sl) - 1)
    return float(sl[idx])


class HistoricalVaR:
    """Historical (empirical-quantile) VaR/ES, optionally age-weighted
    (BRW exponential weighting) and multi-period (overlapping sums)."""

    def __init__(self, confidence=0.99, horizon=1, weights=None, lam=0.94):
        self.confidence, self.horizon = float(confidence), int(horizon)
        self.weights, self.lam = weights, float(lam)
        self.var_ = None
        self.es_ = None

    def compute(self, returns):
        returns = np.asarray(returns, dtype=float).reshape(-1)
        h_returns = _overlapping_sums(returns, self.horizon)
        losses = -h_returns
        n = len(losses)

        if self.weights == "exponential":
            w = self.lam ** np.arange(n - 1, -1, -1)
            w = w / w.sum()
            q = _age_weighted_quantile(losses, w, self.confidence)
            tail = losses[losses >= q]
            es = float(tail.mean()) if len(tail) else q
            f_hat = self._kde_density(losses, q)
            se = math.sqrt(self.confidence * (1 - self.confidence) / n) / max(f_hat, 1e-8)
            res = RiskResult(q, se, n, "historical-weighted", {"alpha": self.confidence}, es)
        else:
            base = risk_analysis(h_returns, self.confidence)
            f_hat = self._kde_density(losses, base.estimate)
            se = math.sqrt(self.confidence * (1 - self.confidence) / n) / max(f_hat, 1e-8)
            res = RiskResult(base.estimate, se, n, "historical", base.extras, base.expected_shortfall)
        self.var_, self.es_ = res.estimate, res.expected_shortfall
        return res

    @staticmethod
    def _kde_density(x, at):
        x = np.asarray(x, dtype=float)
        n = len(x)
        std = x.std(ddof=1) if n > 1 else 1.0
        h = 1.06 * std * n ** (-0.2) if std > 0 else 1.0
        u = (at - x) / h
        return float(np.mean(_norm_pdf(u)) / h)

    def fit(self, returns):
        self.compute(returns)
        return self


class ParametricVaR:
    """Delta-normal / Student-t / Cornish-Fisher / EWMA / GARCH parametric VaR."""

    def __init__(self, confidence=0.99, horizon=1, dist="normal", df=None,
                 method="analytic", lam=0.94):
        self.confidence, self.horizon = float(confidence), int(horizon)
        self.dist, self.df, self.method, self.lam = dist, df, method, float(lam)
        self.mu_ = self.sigma_ = self.skew_ = self.kurt_ = None

    def fit(self, returns):
        returns = np.asarray(returns, dtype=float).reshape(-1)
        self.mu_ = float(returns.mean())
        self.sigma_ = float(returns.std(ddof=1))
        m = returns - self.mu_
        s = self.sigma_
        self.skew_ = float(np.mean(m**3) / s**3) if s > 0 else 0.0
        self.kurt_ = float(np.mean(m**4) / s**4 - 3.0) if s > 0 else 0.0
        self._returns = returns
        return self

    @classmethod
    def from_moments(cls, mu, sigma, skew=0.0, kurt=0.0, confidence=0.99, horizon=1):
        obj = cls(confidence, horizon)
        obj.mu_, obj.sigma_, obj.skew_, obj.kurt_ = float(mu), float(sigma), float(skew), float(kurt)
        return obj

    def _ewma_sigma(self):
        r = self._returns
        var = np.empty(len(r))
        var[0] = r[0] ** 2
        for i in range(1, len(r)):
            var[i] = self.lam * var[i - 1] + (1.0 - self.lam) * r[i - 1] ** 2
        return math.sqrt(var[-1])

    def _garch_sigma(self):
        from stochpylib.timeseries.volatility_models import GARCH
        model = GARCH(1, 1).fit(self._returns)
        fc = model.forecast(self.horizon)
        return math.sqrt(float(np.sum(fc.mean)))

    def compute(self):
        alpha = self.confidence
        h = self.horizon
        mu, sigma = self.mu_ * h, self.sigma_ * math.sqrt(h)

        if self.method == "ewma":
            sigma = self._ewma_sigma() * math.sqrt(h)
        elif self.method == "garch":
            sigma = self._garch_sigma()

        if self.dist == "normal":
            z = _norm_ppf(alpha)
            var = -(mu - z * sigma)
            es = -mu + sigma * float(_norm_pdf(z)) / (1.0 - alpha)
        elif self.dist == "t":
            df = self.df or 5.0
            t = Student_t(df)
            tq = float(t.ppf(alpha))
            scale = sigma * math.sqrt((df - 2.0) / df) if df > 2 else sigma
            var = -(mu - tq * scale)
            f_t = float(t.pdf(tq))
            es = -mu + scale * (df + tq**2) / (df - 1.0) * f_t / (1.0 - alpha)
        else:
            raise ValueError("dist must be 'normal' or 't'")

        if self.method == "cornish_fisher":
            z = _norm_ppf(alpha)
            skew, kurt = self.skew_ or 0.0, self.kurt_ or 0.0
            z_cf = (z + (z**2 - 1.0) * skew / 6.0 + (z**3 - 3.0 * z) * kurt / 24.0
                   - (2.0 * z**3 - 5.0 * z) * skew**2 / 36.0)
            dz_cf = (1.0 + (2.0 * z) * skew / 6.0 + (3.0 * z**2 - 3.0) * kurt / 24.0
                    - (6.0 * z**2 - 5.0) * skew**2 / 36.0)
            if dz_cf <= 0:
                raise ValueError("Cornish-Fisher expansion non-monotone at this quantile "
                                 "(moments too extreme)")
            var = -(mu - z_cf * sigma)

            def var_at_level(p):
                """Cornish-Fisher VaR at confidence level ``p`` (increasing
                toward the loss tail as ``p -> 1``)."""
                zp = _norm_ppf(p)
                zp_cf = (zp + (zp**2 - 1.0) * skew / 6.0 + (zp**3 - 3.0 * zp) * kurt / 24.0
                        - (2.0 * zp**3 - 5.0 * zp) * skew**2 / 36.0)
                return -(mu - zp_cf * sigma)

            # ES(alpha) = (1/(1-alpha)) * integral_alpha^1 VaR_u du (average
            # VaR over the tail confidence levels above alpha, not below).
            es, _ = integrate.quad(var_at_level, alpha, 1.0 - 1e-8, limit=200)
            es = es / (1.0 - alpha)

        return RiskResult(float(var), float("nan"), 0, f"parametric-{self.dist}-{self.method}",
                          {"alpha": alpha}, float(es))


class ValueAtRisk:
    """Facade over historical / parametric / Monte Carlo VaR, plus backtesting."""

    def __init__(self, confidence=0.99, horizon=1):
        self.confidence, self.horizon = float(confidence), int(horizon)

    def historical(self, returns, **kw):
        return HistoricalVaR(self.confidence, self.horizon, **kw).compute(returns)

    def parametric(self, returns=None, mu=None, sigma=None, dist="normal", df=None,
                   cornish_fisher=False):
        method = "cornish_fisher" if cornish_fisher else "analytic"
        pv = ParametricVaR(self.confidence, self.horizon, dist, df, method)
        if returns is not None:
            pv.fit(returns)
        else:
            pv.mu_, pv.sigma_, pv.skew_, pv.kurt_ = mu, sigma, 0.0, 0.0
        return pv.compute()

    def monte_carlo(self, sampler_or_samples, n=100_000, random_state=None):
        if callable(sampler_or_samples):
            rng = _rng(random_state)
            samples = sampler_or_samples(n, rng)
        else:
            samples = np.asarray(sampler_or_samples, dtype=float)
        res = risk_analysis(samples, self.confidence)
        res.method = "monte-carlo"
        return res

    def backtest(self, returns, var_series):
        returns = np.asarray(returns, dtype=float)
        var_series = np.asarray(var_series, dtype=float)
        n = len(returns)
        exceptions = int(np.sum(-returns > var_series))
        p = 1.0 - self.confidence
        expected = n * p
        if exceptions == 0:
            lr_pof = -2.0 * n * math.log(1.0 - p)
        elif exceptions == n:
            lr_pof = -2.0 * n * math.log(p)
        else:
            p_hat = exceptions / n
            lr_pof = -2.0 * (
                (n - exceptions) * math.log(1.0 - p) + exceptions * math.log(p)
                - (n - exceptions) * math.log(1.0 - p_hat) - exceptions * math.log(p_hat)
            )
        p_value = float(special.chdtrc(1, lr_pof))
        return {"exceptions": exceptions, "expected": expected, "lr_pof": float(lr_pof),
               "p_value": p_value}


class ExpectedShortfall:
    def __init__(self, confidence=0.99):
        self.confidence = float(confidence)

    def historical(self, returns):
        returns = np.asarray(returns, dtype=float).reshape(-1)
        losses = -returns
        q = float(np.quantile(losses, self.confidence))
        tail = losses[losses >= q]
        n_tail = len(tail)
        se = float(tail.std(ddof=1) / math.sqrt(n_tail)) if n_tail > 1 else float("nan")
        return MCResult(float(tail.mean()) if n_tail else q, se, n_tail, "historical-es",
                        {"alpha": self.confidence, "var": q})

    def parametric(self, mu, sigma, dist="normal", df=None):
        z = _norm_ppf(self.confidence)
        if dist == "normal":
            return float(-mu + sigma * float(_norm_pdf(z)) / (1.0 - self.confidence))
        df = df or 5.0
        t = Student_t(df)
        tq = float(t.ppf(self.confidence))
        scale = sigma * math.sqrt((df - 2.0) / df) if df > 2 else sigma
        f_t = float(t.pdf(tq))
        return float(-mu + scale * (df + tq**2) / (df - 1.0) * f_t / (1.0 - self.confidence))

    def compute(self, returns, method="historical"):
        if method == "historical":
            return self.historical(returns)
        raise ValueError("method must be 'historical' (use .parametric() for closed form)")


class ConditionalVaR(ExpectedShortfall):
    """Alias for Expected Shortfall, plus CVaR-portfolio optimization
    (Rockafellar-Uryasev 2000 linear program)."""

    def optimize_portfolio(self, returns_matrix, target_return=None, long_only=True):
        R = np.asarray(returns_matrix, dtype=float)
        n_obs, n_assets = R.shape
        alpha = self.confidence

        n_vars = n_assets + 1 + n_obs  # w, zeta, u
        c = np.zeros(n_vars)
        c[n_assets] = 1.0
        c[n_assets + 1:] = 1.0 / ((1.0 - alpha) * n_obs)

        A_ub = np.zeros((n_obs, n_vars))
        A_ub[:, :n_assets] = -R
        A_ub[:, n_assets] = -1.0
        A_ub[:, n_assets + 1:] = -np.eye(n_obs)
        b_ub = np.zeros(n_obs)

        A_eq = np.zeros((1, n_vars))
        A_eq[0, :n_assets] = 1.0
        b_eq = [1.0]
        if target_return is not None:
            mean_ret = R.mean(axis=0)
            A_eq = np.vstack([A_eq, np.concatenate([mean_ret, np.zeros(1 + n_obs)])])
            b_eq = b_eq + [target_return]

        bounds = [(0.0, 1.0) if long_only else (None, None)] * n_assets \
            + [(None, None)] + [(0.0, None)] * n_obs

        res = optimize.linprog(c, A_ub=A_ub, b_ub=b_ub, A_eq=A_eq, b_eq=b_eq,
                               bounds=bounds, method="highs")
        if not res.success:
            raise RuntimeError(f"CVaR optimization failed: {res.message}")
        w = res.x[:n_assets]
        self.weights_ = w
        self.cvar_ = float(res.fun)
        return w


class StressTest:
    """Apply named factor shocks to a pricer ``pricer(factors: dict) -> float``."""

    def __init__(self, pricer, base_factors, shock_type="absolute"):
        self.pricer = pricer
        self.base_factors = dict(base_factors)
        self.shock_type = shock_type

    def apply(self, shocks):
        base_value = self.pricer(self.base_factors)
        shocked = dict(self.base_factors)
        for name, shock in shocks.items():
            kind, magnitude = shock if isinstance(shock, tuple) else (self.shock_type, shock)
            if kind == "relative":
                shocked[name] = shocked[name] * (1.0 + magnitude)
            else:
                shocked[name] = shocked[name] + magnitude
        stressed_value = self.pricer(shocked)
        return {"base_value": base_value, "stressed_value": stressed_value,
               "pnl": stressed_value - base_value}

    def run(self, scenarios):
        return {name: self.apply(shocks)["pnl"] for name, shocks in scenarios.items()}


class ScenarioAnalysis:
    """Distribution of P&L under a pricer and factor scenarios, from historical
    factor returns, a multivariate-normal Monte Carlo, or a custom sampler."""

    def __init__(self, pricer, confidence=0.99):
        self.pricer = pricer
        self.confidence = float(confidence)
        self.pnl_ = None
        self.scenarios_ = None

    def from_historical(self, factor_returns, base_factors, factors=None):
        """``factors`` names the (relatively-shocked) stochastic keys, in the
        column order of ``factor_returns`` — defaults to every key in
        ``base_factors``, but a pricer commonly takes extra fixed parameters
        (e.g. a strike) that should stay untouched, hence the override."""
        factor_returns = np.asarray(factor_returns, dtype=float)
        names = factors if factors is not None else list(base_factors.keys())
        base_value = self.pricer(base_factors)
        pnl = np.empty(len(factor_returns))
        for i, row in enumerate(factor_returns):
            shocked = dict(base_factors)
            for j, n in enumerate(names):
                shocked[n] = base_factors[n] * (1.0 + row[j])
            pnl[i] = self.pricer(shocked) - base_value
        self.pnl_, self.scenarios_ = pnl, factor_returns
        return self

    def from_mc(self, mean, cov, base_factors, factors=None, n=100_000, random_state=None):
        """``factors`` names the stochastic keys shocked by the ``mean``/``cov``
        multivariate normal, in that order — defaults to every key in
        ``base_factors`` (see :meth:`from_historical`)."""
        mvn = MultivariateNormal(mean, cov)
        draws = mvn.rvs(size=n, random_state=random_state)
        names = factors if factors is not None else list(base_factors.keys())
        base_value = self.pricer(base_factors)
        pnl = np.empty(n)
        for i in range(n):
            shocked = dict(base_factors)
            for j, nm in enumerate(names):
                shocked[nm] = base_factors[nm] * (1.0 + draws[i, j])
            pnl[i] = self.pricer(shocked) - base_value
        self.pnl_, self.scenarios_ = pnl, draws
        return self

    def from_sampler(self, fn, n, random_state=None):
        rng = _rng(random_state)
        pnl = np.asarray([fn(rng) for _ in range(n)], dtype=float)
        self.pnl_ = pnl
        return self

    def run(self):
        return risk_analysis(self.pnl_, self.confidence)
