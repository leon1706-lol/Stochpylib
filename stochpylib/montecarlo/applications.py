"""Application-level Monte Carlo tools built on the simulation/variance-reduction core."""

import numpy as np
from dataclasses import dataclass

from stochpylib.montecarlo._result import MCResult
from stochpylib.montecarlo.simulation import crude_mc
from stochpylib.montecarlo.variance_reduction import AntitheticVariates

__all__ = [
    "MonteCarloIntegration",
    "pi_estimation",
    "option_pricing_mc",
    "risk_analysis",
    "RiskResult",
    "reliability_mc",
    "sensitivity_analysis",
]


class MonteCarloIntegration:
    """Integral over a box via plain MC or a low-discrepancy sequence."""

    def __init__(self, integrand, bounds=None, method="crude", sequence="sobol",
                 random_state=None):
        self.integrand = integrand
        self.bounds = bounds
        self.method = method
        self.sequence = sequence
        self.random_state = random_state

    def estimate(self, n=100_000):
        if self.method == "qmc":
            from stochpylib.montecarlo.simulation import quasi_montecarlo

            dim = 1 if self.bounds is None else np.asarray(self.bounds).shape[0]
            return quasi_montecarlo(self.integrand, dim=dim, n=n, sequence=self.sequence,
                                    bounds=self.bounds, random_state=self.random_state)
        if self.bounds is None:
            dim = 1
        else:
            dim = np.asarray(self.bounds).shape[0]
        return crude_mc(self.integrand, n=n, dim=dim, bounds=self.bounds,
                        random_state=self.random_state)


def pi_estimation(n=1_000_000, random_state=None):
    """Estimate pi by sampling the unit square and counting quarter-circle hits."""
    rng = np.random.default_rng(random_state)
    pts = rng.uniform(size=(n, 2))
    hits = np.sum(pts[:, 0] ** 2 + pts[:, 1] ** 2 <= 1.0)
    p_hat = hits / n
    se = float(np.sqrt(p_hat * (1 - p_hat) / n))
    return MCResult(estimate=4.0 * p_hat, std_error=4.0 * se, n_samples=n,
                    method="pi_estimation")


def option_pricing_mc(S=100.0, K=100.0, T=1.0, r=0.05, sigma=0.2, n=100_000,
                      kind="call", antithetic=True, random_state=None):
    """European option price under GBM via Monte Carlo (antithetic by default)."""
    if kind not in ("call", "put"):
        raise ValueError("kind must be 'call' or 'put'")
    if antithetic:
        av = AntitheticVariates(n_simulations=n, random_state=random_state)
        m = n // 2
        z = av.rng.standard_normal(m)
        up = S * np.exp((r - sigma**2 / 2) * T + sigma * np.sqrt(T) * z)
        down = S * np.exp((r - sigma**2 / 2) * T - sigma * np.sqrt(T) * z)
        if kind == "call":
            payoffs = 0.5 * (np.maximum(up - K, 0) + np.maximum(down - K, 0))
        else:
            payoffs = 0.5 * (np.maximum(K - up, 0) + np.maximum(K - down, 0))
        disc = np.exp(-r * T)
        est = disc * float(payoffs.mean())
        se = disc * float(payoffs.std(ddof=1) / np.sqrt(m))
        return MCResult(estimate=est, std_error=se, n_samples=n,
                        method=f"mc-{kind}-antithetic")
    rng = np.random.default_rng(random_state)
    z = rng.standard_normal(n)
    terminal = S * np.exp((r - sigma**2 / 2) * T + sigma * np.sqrt(T) * z)
    if kind == "call":
        payoffs = np.maximum(terminal - K, 0)
    else:
        payoffs = np.maximum(K - terminal, 0)
    disc = np.exp(-r * T)
    return MCResult(estimate=disc * float(payoffs.mean()),
                    std_error=disc * float(payoffs.std(ddof=1) / np.sqrt(n)),
                    n_samples=n, method=f"mc-{kind}")


@dataclass
class RiskResult(MCResult):
    """Value-at-Risk result carrying the accompanying expected shortfall."""

    expected_shortfall: float = float("nan")


def risk_analysis(samples, alpha=0.95):
    """Historical VaR/ES from a 1-D array of P&L or returns (loss = negative outcome).

    Returns a :class:`RiskResult`: ``estimate`` is the alpha-VaR (the quantile of the
    loss distribution), ``extras['expected_shortfall']`` is the mean loss beyond VaR.
    """
    arr = np.asarray(samples, dtype=float).reshape(-1)
    losses = -arr
    q = float(np.quantile(losses, alpha))
    tail = losses[losses >= q]
    es = float(tail.mean()) if len(tail) else q
    res = RiskResult(
        estimate=q, std_error=float("nan"), n_samples=len(arr), method="historical",
        extras={"alpha": alpha, "expected_shortfall": es},
    )
    res.expected_shortfall = es
    return res


def reliability_mc(performance_fn, input_distributions, threshold=0.0, n=100_000,
                   random_state=None):
    """Failure probability ``P(performance_fn(X) <= threshold)``.

    ``input_distributions`` is a list of stochpylib ``Distribution`` objects (their
    ``.rvs()`` supplies independent inputs).
    """
    rng = np.random.default_rng(random_state)
    draws = [np.atleast_1d(d.rvs(n, random_state=rng)).astype(float) for d in input_distributions]
    X = np.column_stack(draws)
    g = np.asarray(performance_fn(X), dtype=float).reshape(-1)
    failed = int(np.sum(g <= threshold))
    p_hat = failed / n
    if 0 < failed < n:
        se = float(np.sqrt(p_hat * (1 - p_hat) / n))
    else:
        se = float("nan")  # degenerate estimate (zero or all failures)
    return MCResult(estimate=p_hat, std_error=se, n_samples=n, method="reliability_mc",
                    extras={"failures": failed})


def sensitivity_analysis(fn, input_distributions, n=100_000, method="correlation",
                         random_state=None):
    """First-order input sensitivity via output correlation.

    Samples each input independently from its distribution, evaluates ``fn``, and reports
    Pearson and Spearman correlations between every input and the output.
    """
    from scipy import stats as spt

    rng = np.random.default_rng(random_state)
    cols = [np.atleast_1d(d.rvs(n, random_state=rng)).astype(float) for d in input_distributions]
    X = np.column_stack(cols)
    y = np.asarray(fn(X), dtype=float).reshape(-1)
    out = {}
    for j, _ in enumerate(cols):
        pearson = float(spt.pearsonr(X[:, j], y)[0])
        spearman = float(spt.spearmanr(X[:, j], y)[0])
        out[j] = {"pearson": pearson, "spearman": spearman}
    return out
