"""Core Monte Carlo estimation routines.

Conventions (ARCHITECTURE.md):

- every estimator takes ``random_state=None`` (anything ``np.random.default_rng`` accepts);
- integration is over the **unit hypercube** by default — pass ``bounds`` for a box domain;
- integrands are callables mapping an ``(m, dim)`` array of points to an ``(m,)`` array of
  values (write scalar functions as ``lambda pts: f(pts[:, 0])`` in 1-D);
- estimators return :class:`~stochpylib.montecarlo._result.MCResult`.
"""

import numpy as np

from stochpylib.montecarlo._result import MCResult

__all__ = [
    "simulate",
    "crude_mc",
    "importance_sampling",
    "rejection_sampling",
    "stratified_sampling",
    "quasi_montecarlo",
]


def _rng(random_state):
    return np.random.default_rng(random_state)


def _eval_integrand(f, pts):
    vals = np.asarray(f(pts), dtype=float)
    if vals.shape != (pts.shape[0],):
        raise ValueError(
            f"integrand must map ({pts.shape[0]}, {pts.shape[1]}) -> ({pts.shape[0]},); "
            f"got shape {vals.shape}"
        )
    return vals


def simulate(statistic, sampler, n_simulations=10_000, random_state=None):
    """Generic driver: repeat ``sampler() -> statistic(sample)`` ``n_simulations`` times.

    ``statistic`` maps one realized sample to a float. Returns the mean over repetitions
    with its standard error.
    """
    rng = _rng(random_state)
    draws = np.empty(n_simulations, dtype=float)
    for i in range(n_simulations):
        value = statistic(sampler(rng))
        draws[i] = float(value)
    return MCResult(
        estimate=float(draws.mean()),
        std_error=float(draws.std(ddof=1) / np.sqrt(n_simulations)) if n_simulations > 1 else float("nan"),
        n_samples=n_simulations,
        method="simulate",
    )


def crude_mc(integrand, n=100_000, dim=1, bounds=None, random_state=None):
    """Crude Monte Carlo integral of ``integrand`` over a box domain.

    Default domain is the unit hypercube ``[0,1)^dim``; ``bounds`` is a ``(dim, 2)``
    array-like of ``(low, high)`` pairs.
    """
    rng = _rng(random_state)
    if bounds is None:
        lows, highs = np.zeros(dim), np.ones(dim)
        volume = 1.0
    else:
        bounds = np.asarray(bounds, dtype=float)
        lows, highs = bounds[:, 0], bounds[:, 1]
        if np.any(highs <= lows):
            raise ValueError("bounds must satisfy low < high in every dimension")
        volume = float(np.prod(highs - lows))
    pts = rng.uniform(lows, highs, size=(n, dim))
    vals = _eval_integrand(integrand, pts)
    return MCResult(
        estimate=volume * float(vals.mean()),
        std_error=volume * float(vals.std(ddof=1) / np.sqrt(n)),
        n_samples=n,
        method="crude_mc",
    )


def importance_sampling(integrand, target_pdf, proposal_sampler, proposal_pdf,
                        n=100_000, random_state=None):
    """Importance sampling: ``E_f[integrand]`` estimated with draws from a proposal.

    ``proposal_sampler(n, rng)`` returns ``(n, dim)`` draws; ``target_pdf`` /
    ``proposal_pdf`` evaluate densities on those draws (any shape; flattened).
    Weights are self-normalized; the effective sample size (ESS) is reported in
    ``extras['ess']``.
    """
    rng = _rng(random_state)
    pts = np.atleast_2d(proposal_sampler(n, rng))
    if pts.shape[0] != n:
        raise ValueError("proposal_sampler must return exactly n draws")
    t = np.asarray(target_pdf(pts), dtype=float).reshape(-1)
    d = np.asarray(proposal_pdf(pts), dtype=float)
    d = d.reshape(-1)
    if np.any(d <= 0):
        raise ValueError("proposal_pdf must be positive at proposed points")
    g = _eval_integrand(integrand, pts)
    weighted = t * g
    estimate = float(weighted.sum() / d.sum())
    # self-normalized delta-method standard error
    w = t / d
    se = float(np.sqrt(np.mean(w * (g - estimate) ** 2) / n))
    w_norm = w / w.sum()
    ess = 1.0 / float(np.sum(w_norm**2))
    return MCResult(estimate=estimate, std_error=se, n_samples=n,
                    method="importance_sampling", extras={"ess": ess})


def rejection_sampling(target_pdf, proposal_sampler, proposal_pdf, n_samples=1000,
                       k_bound=None, max_trials=None, batch_size=None, random_state=None):
    """Acceptance-rejection sampling from ``target_pdf`` using a proposal envelope.

    Requires ``target_pdf(x) <= k_bound * proposal_pdf(x)`` for all ``x``; pass the known
    envelope constant as ``k_bound`` (auto-calibrated on a pilot batch when omitted —
    verify coverage, auto-calibration can under-estimate heavy tails).

    Returns ``(samples, acceptance_rate)`` where ``samples`` has exactly ``n_samples``
    rows (raises ``RuntimeError`` if ``max_trials`` is exhausted first).
    """
    rng = _rng(random_state)
    max_trials = max_trials or max(50 * n_samples, 10_000)
    batch_size = batch_size or max(n_samples, 1000)
    accepted = []
    trials = 0
    if k_bound is None:  # pilot calibration
        pilot_pts = np.atleast_2d(proposal_sampler(min(batch_size, 5000), rng))
        t = np.asarray(target_pdf(pilot_pts), dtype=float).reshape(-1)
        p = np.asarray(proposal_pdf(pilot_pts), dtype=float).reshape(-1)
        ratio = np.where(p > 0, t / p, 0.0)
        k_bound = 1.2 * float(np.nanmax(ratio))
    while sum(len(a) for a in accepted) < n_samples and trials < max_trials:
        pts = np.atleast_2d(proposal_sampler(batch_size, rng))
        t = np.asarray(target_pdf(pts), dtype=float).reshape(-1)
        p = np.asarray(proposal_pdf(pts), dtype=float).reshape(-1)
        with np.errstate(divide="ignore", invalid="ignore"):
            ratio = np.where(p > 0, t / (k_bound * p), np.inf)
        accept = rng.uniform(size=batch_size) < ratio
        kept = pts[accept]
        if len(kept):
            accepted.append(kept)
        trials += batch_size
    total = sum(len(a) for a in accepted)
    if total < n_samples:
        raise RuntimeError(
            f"rejection_sampling: only {total}/{n_samples} accepted after {trials} trials "
            "(check k_bound / proposal overlap)"
        )
    samples = np.vstack(accepted)[:n_samples]
    return samples, total / trials


def stratified_sampling(integrand, bounds=(0.0, 1.0), n_strata=100, n_per_stratum=100,
                        random_state=None):
    """1-D stratified integral over ``[low, high)`` with equal-width strata.

    Variance within strata removes between-strata variance — strictly better than crude
    MC for monotone integrands.
    """
    rng = _rng(random_state)
    low, high = map(float, bounds)
    width = (high - low) / n_strata
    estimates = np.empty(n_strata)
    for s in range(n_strata):
        lo = low + s * width
        pts = rng.uniform(lo, lo + width, size=(n_per_stratum, 1))
        estimates[s] = _eval_integrand(integrand, pts).mean()
    integral = width * float(estimates.sum())
    se = width * float(estimates.std(ddof=1) / np.sqrt(n_strata))
    return MCResult(estimate=integral, std_error=se, n_samples=n_strata * n_per_stratum,
                    method="stratified_sampling")


def quasi_montecarlo(integrand, dim=1, n=16_384, sequence="sobol", bounds=None,
                     random_state=None):
    """Quasi-Monte Carlo integral over a box using a low-discrepancy sequence.

    Deterministic (no standard error); ``random_state`` only seeds the optional digital
    shift scrambling of the sequence.
    """
    from stochpylib.montecarlo.quasi_random import LowDiscrepancy

    seq = LowDiscrepancy(sequence, dim=dim, random_state=random_state)
    pts = seq.generate(n)
    if bounds is not None:
        bounds = np.asarray(bounds, dtype=float)
        lows, highs = bounds[:, 0], bounds[:, 1]
        volume = float(np.prod(highs - lows))
        pts = lows + pts * (highs - lows)
    else:
        volume = 1.0
    vals = _eval_integrand(integrand, pts)
    return MCResult(estimate=volume * float(vals.mean()), std_error=float("nan"),
                    n_samples=n, method=f"qmc({sequence})")
