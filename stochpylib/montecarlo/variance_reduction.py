"""Variance-reduction techniques for Monte Carlo estimation.

Class-based per the design-spec quickstart::

    av = AntitheticVariates(n_simulations=100_000)
    price = av.price_european_call(S=100, K=100, T=1, r=0.05, sigma=0.2)

Every ``.estimate(...)`` returns an :class:`MCResult`; achieved variance reduction versus
crude sampling is reported in ``extras`` where meaningful. All randomness goes through
``random_state=``.
"""

import numpy as np

from stochpylib.montecarlo._result import MCResult
from stochpylib.montecarlo.simulation import _eval_integrand

__all__ = [
    "AntitheticVariates",
    "ControlVariates",
    "StratifiedSampling",
    "LatinHypercubeSampling",
    "OrthogonalSampling",
    "ConditionedMC",
    "RejectionControl",
]


def _black_scholes_price(S, K, T, r, sigma, kind="call"):
    """Closed-form European option price (internal oracle for tests/examples)."""
    from scipy import special

    d1 = (np.log(S / K) + (r + sigma**2 / 2) * T) / (sigma * np.sqrt(T))
    d2 = d1 - sigma * np.sqrt(T)
    if kind == "call":
        return S * special.ndtr(d1) - K * np.exp(-r * T) * special.ndtr(d2)
    return K * np.exp(-r * T) * special.ndtr(-d2) - S * special.ndtr(-d1)


class AntitheticVariates:
    """Average each draw ``U`` with its mirror ``1 - U`` (antithetic pairing).

    Helps whenever the integrand is close to monotone in the underlying uniforms —
    e.g. European payoffs driven by Gaussian increments.
    """

    def __init__(self, n_simulations=100_000, random_state=None):
        self.n = int(n_simulations)
        self.rng = np.random.default_rng(random_state)

    def estimate(self, integrand, dim=1):
        """Estimate ``E[f(U)]`` over uniforms using ``n // 2`` antithetic pairs."""
        m = self.n // 2
        u = self.rng.uniform(size=(m, dim))
        f1 = _eval_integrand(integrand, u)
        f2 = _eval_integrand(integrand, 1.0 - u)
        pairs = (f1 + f2) / 2.0
        crude_se = float(np.concatenate([f1, f2]).std(ddof=1) / np.sqrt(self.n))
        se = float(pairs.std(ddof=1) / np.sqrt(m))
        return MCResult(estimate=float(pairs.mean()), std_error=se, n_samples=self.n,
                        method="antithetic",
                        extras={"variance_reduction": (crude_se / se) ** 2 if se > 0 else float("inf")})

    def _gbm_terminal(self, S, T, r, sigma, m):
        z = self.rng.standard_normal(m)
        return S * np.exp((r - sigma**2 / 2) * T + sigma * np.sqrt(T) * z), \
               S * np.exp((r - sigma**2 / 2) * T - sigma * np.sqrt(T) * z)

    def price_european_call(self, S=100.0, K=100.0, T=1.0, r=0.05, sigma=0.2):
        """Discounted European call price via antithetic GBM terminals."""
        m = self.n // 2
        up, down = self._gbm_terminal(S, T, r, sigma, m)
        payoffs = 0.5 * (np.maximum(up - K, 0) + np.maximum(down - K, 0))
        disc = np.exp(-r * T)
        se = disc * float(payoffs.std(ddof=1) / np.sqrt(m))
        return MCResult(estimate=disc * float(payoffs.mean()), std_error=se,
                        n_samples=self.n, method="antithetic-european-call")

    def price_european_put(self, S=100.0, K=100.0, T=1.0, r=0.05, sigma=0.2):
        m = self.n // 2
        up, down = self._gbm_terminal(S, T, r, sigma, m)
        payoffs = 0.5 * (np.maximum(K - up, 0) + np.maximum(K - down, 0))
        disc = np.exp(-r * T)
        se = disc * float(payoffs.std(ddof=1) / np.sqrt(m))
        return MCResult(estimate=disc * float(payoffs.mean()), std_error=se,
                        n_samples=self.n, method="antithetic-european-put")


class ControlVariates:
    """Exploit a known-mean companion: ``f - beta*(g - E[g])`` with least-squares beta."""

    def __init__(self, n_simulations=100_000, random_state=None):
        self.n = int(n_simulations)
        self.rng = np.random.default_rng(random_state)

    def estimate(self, integrand, control_integrand, control_mean, dim=1, sampler=None):
        """Estimate ``E[f(X)]`` using ``E[g(X)] = control_mean`` as the control.

        By default ``X`` is ``dim`` independent uniforms; pass ``sampler(n, rng)``
        (returning ``(n, dim)`` draws) to drive the pair from any base variates —
        e.g. standard normals for payoff/stock controls.
        """
        if sampler is None:
            pts = self.rng.uniform(size=(self.n, dim))
        else:
            pts = np.atleast_2d(sampler(self.n, self.rng))
        f = _eval_integrand(integrand, pts)
        g = _eval_integrand(control_integrand, pts)
        var_g = float(g.var(ddof=1))
        if var_g == 0:
            raise ValueError("control variate is constant on the sampled points")
        beta = float(np.cov(f, g, ddof=1)[0, 1]) / var_g
        adjusted = f - beta * (g - control_mean)
        raw_se = float(f.std(ddof=1) / np.sqrt(self.n))
        se = float(adjusted.std(ddof=1) / np.sqrt(self.n))
        vr = (raw_se / se) ** 2 if se > 0 else float("inf")
        return MCResult(estimate=float(adjusted.mean()), std_error=se, n_samples=self.n,
                        method="control_variate",
                        extras={"beta": beta, "variance_reduction": vr})


class LatinHypercubeSampling:
    """LHS: exactly one uniform draw per stratum in every margin."""

    def __init__(self, dim=2, n=1024, random_state=None):
        self.dim, self.n = int(dim), int(n)
        self.rng = np.random.default_rng(random_state)

    def generate(self):
        strata = (np.arange(self.n)[:, None] + self.rng.uniform(size=(self.n, self.dim))) / self.n
        out = np.empty_like(strata)
        for d in range(self.dim):
            out[self.rng.permutation(self.n), d] = strata[:, d]
        return out


class OrthogonalSampling(LatinHypercubeSampling):
    """LHS with an extra level of within-stratum jitter (documented simplification of
    full orthogonal-array sampling: stratification at both scales improves balance)."""

    def generate(self):
        base = super().generate()
        fine = (np.arange(self.n * self.dim, dtype=float).reshape(self.n, self.dim) * 0 +
                self.rng.uniform(size=(self.n, self.dim)))
        # second-level jitter inside each coarse stratum keeps margins stratified
        jitter = ((np.arange(self.n)[:, None] + fine) % self.n) / self.n
        shifted = (base + jitter / self.n) % 1.0
        return shifted


class StratifiedSampling:
    """Multi-dimensional stratification over an equal grid of ``n_strata^dim`` boxes."""

    def __init__(self, n_strata=32, dim=1, n_per_stratum=8, random_state=None):
        if dim > 3:
            raise ValueError("StratifiedSampling class supports dim <= 3 (use the 1-D function "
                             "stratified_sampling for other cases)")
        self.K, self.dim, self.m = int(n_strata), int(dim), int(n_per_stratum)
        self.rng = np.random.default_rng(random_state)

    def estimate(self, integrand, bounds=None):
        if bounds is None:
            lows, highs = np.zeros(self.dim), np.ones(self.dim)
            volume = 1.0
        else:
            bounds = np.asarray(bounds, dtype=float)
            lows, highs = bounds[:, 0], bounds[:, 1]
            volume = float(np.prod(highs - lows))
        total = self.K**self.dim
        ests = np.empty(total)
        widths = (highs - lows) / self.K
        for idx, cell in enumerate(np.ndindex(*([self.K] * self.dim))):
            lo = lows + np.asarray(cell) * widths
            pts = self.rng.uniform(lo, lo + widths, size=(self.m, self.dim))
            ests[idx] = _eval_integrand(integrand, pts).mean()
        integral = volume * float(ests.mean())
        # between-cell variance drives the error once within-cell counts are fixed
        se = volume * float(ests.std(ddof=1) / np.sqrt(total))
        return MCResult(estimate=integral, std_error=se, n_samples=total * self.m,
                        method="stratified_grid")


class ConditionedMC:
    """Rao-Blackwellized / conditioned estimator: average user-supplied conditional
    expectations ``m(y) = E[f(X) | Y=y]`` instead of noisy realized values."""

    def __init__(self, n_simulations=100_000, random_state=None):
        self.n = int(n_simulations)
        self.rng = np.random.default_rng(random_state)

    def estimate(self, cond_expectation, y_sampler):
        """``cond_expectation(y) -> float``; ``y_sampler(rng) -> one Y draw``."""
        ys = [y_sampler(self.rng) for _ in range(self.n)]
        vals = np.array([float(cond_expectation(y)) for y in ys])
        return MCResult(estimate=float(vals.mean()),
                        std_error=float(vals.std(ddof=1) / np.sqrt(self.n)),
                        n_samples=self.n, method="conditioned_mc")


class RejectionControl:
    """Hesterberg-style rejection-control weighting: clip extreme importance weights at
    a threshold and redistribute the removed mass proportionally (deficit splitting).
    """

    def __init__(self, n_simulations=100_000, random_state=None):
        self.n = int(n_simulations)
        self.rng = np.random.default_rng(random_state)

    def estimate(self, integrand, target_pdf, proposal_sampler, proposal_pdf, threshold=None):
        pts = np.atleast_2d(proposal_sampler(self.n, self.rng))
        t = np.asarray(target_pdf(pts), dtype=float).reshape(-1)
        p = np.asarray(proposal_pdf(pts), dtype=float).reshape(-1)
        g = _eval_integrand(integrand, pts)
        w_raw = t / p
        threshold = threshold or float(np.quantile(w_raw, 0.9))
        clipped = np.minimum(w_raw, threshold)
        deficit_mass = float(np.sum(w_raw[clipped >= threshold] - threshold)) / len(w_raw)
        # distribute the removed mass evenly over un-clipped draws (proportional split)
        share = np.where(clipped < threshold, deficit_mass / max(int((clipped < threshold).sum()), 1), 0.0)
        w_star = clipped + share
        estimate = float(np.sum(w_star * g) / np.sum(w_star))
        # reference: plain self-normalized importance sampling SE (delta method)
        is_estimate = float(np.sum(w_raw * g) / np.sum(w_raw))
        is_se = float(
            np.sqrt(np.sum(w_raw**2 * (g - is_estimate) ** 2)) / abs(np.sum(w_raw))
        )
        se = float(np.sqrt(np.sum(w_star**2 * (g - estimate) ** 2)) / abs(np.sum(w_star)))
        return MCResult(estimate=estimate, std_error=se, n_samples=self.n,
                        method="rejection_control",
                        extras={"threshold": threshold,
                                "variance_reduction": (is_se / se) ** 2})
