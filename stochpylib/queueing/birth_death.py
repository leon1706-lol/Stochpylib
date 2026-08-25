"""Birth-death queueing models and classical blocking formulas."""

import numpy as np

__all__ = [
    "BirthDeathQueue", "erlang_b_formula", "erlang_c_formula",
    "engset_formula",
]


class BirthDeathQueue:
    """General birth-death process with state-dependent rates.

    Computes the steady-state distribution via the detailed-balance product
    form, truncated at ``max_population`` states.
    """

    def __init__(self, max_population=10000):
        self.max_population = int(max_population)

    def steady_state(self, lambda_fn, mu_fn):
        """Return normalised pi_0..pi_{max_pop-1}."""
        pi = [1.0]
        for n in range(1, self.max_population):
            ln = float(lambda_fn(n - 1))
            mn = float(mu_fn(n))
            if mn <= 0:
                break
            pi.append(pi[-1] * ln / mn)
        total = sum(pi)
        return np.array(pi[:len(pi)]) / total

    def fit(self, arrival_rate, service_rate, n_servers=1):
        """M/M/c steady-state distribution via birth-death framework."""
        lam = float(arrival_rate)
        mu = float(service_rate)
        c = int(n_servers)
        if lam >= c * mu:
            raise ValueError("system unstable")
        pi = self.steady_state(
            lambda_fn=lambda n: lam,
            mu_fn=lambda n: min(n, c) * mu,
        )
        N = len(pi)
        ns = np.arange(N, dtype=float)
        L = float(np.sum(ns * pi))
        Lq = float(np.sum(np.maximum(ns - c, 0.0) * pi))
        Wq = Lq / lam if lam > 0 else 0.0
        W = Wq + 1.0 / mu
        rho = lam / (c * mu)
        return {
            "steady_state": pi,
            "L": L, "Lq": Lq, "W": W, "Wq": Wq, "rho": rho,
        }


def _comb(n, k):
    if k < 0 or k > n:
        return 0
    r = 1
    for i in range(k):
        r = r * (n - i) // (i + 1)
    return r


def erlang_b_formula(n_servers, offered_load):
    """Erlang-B blocking probability for M/M/c/c loss system.

    Parameters
    ----------
    n_servers : int
        Number of servers / trunks.
    offered_load : float
        Offered traffic in Erlangs.

    Returns P(block), the probability that all servers are busy.
    """
    c = int(n_servers)
    a = float(offered_load)
    if c < 1 or a < 0:
        raise ValueError("need n_servers >= 1 and offered_load >= 0")
    if a == 0:
        return 0.0
    inv_b = 1.0
    for k in range(1, c + 1):
        inv_b = 1.0 + inv_b * k / max(a, 1e-15)
    return float(1.0 / inv_b)


def erlang_c_formula(n_servers, offered_load):
    """Erlang-C waiting probability for M/M/c queue.

    Returns P(wait > 0).
    """
    c = int(n_servers)
    a = float(offered_load)
    if c < 1 or a < 0:
        raise ValueError("need n_servers >= 1 and offered_load >= 0")
    if a >= c:
        return 1.0
    eb = erlang_b_formula(c, a)
    ec = eb / (1.0 - a / c * (1.0 - eb))
    return float(min(max(ec, 0.0), 1.0))


def engset_formula(n_sources, n_servers, per_source_offered_load):
    """Engset loss formula for finite-source systems.

    Parameters
    ----------
    n_sources : int
        Number of traffic sources (m > c required for nonzero congestion).
    n_servers : int
        Number of servers (c).
    per_source_offered_load : float
        Offered load per idle source.

    Returns the blocking probability B(m, c, alpha).
    """
    m = int(n_sources)
    c = int(n_servers)
    alpha = float(per_source_offered_load)
    if m <= c:
        return 0.0
    numerator = _comb(m - 1, c) * alpha ** c
    denominator = sum(_comb(m - 1, j) * alpha ** j
                      for j in range(min(c + 1, m)))
    if denominator == 0:
        return 0.0
    return float(min(1.0, numerator / denominator))