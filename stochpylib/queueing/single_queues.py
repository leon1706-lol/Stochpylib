"""Analytic single-station queueing models: M/M/1 through M/G/1 priority."""

import numpy as np
import math

from stochpylib.queueing._base import QueueResult, BaseQueue, _validate_rates

__all__ = [
    "MM1Queue", "MMCQueue", "MMInfinityQueue", "MD1Queue", "MG1Queue",
    "GI1Queue", "GIGQueue", "MG1PriorityQueue",
]

_EPS = 1e-12


class MM1Queue(BaseQueue):
    """M/M/1 queue (Poisson arrivals, exponential service, 1 server)."""

    def fit(self, arrival_rate, service_rate, **kw):
        _validate_rates(arrival_rate, service_rate)
        lam, mu = float(arrival_rate), float(service_rate)
        rho = lam / mu
        L = lam / (mu - lam)
        Lq = lam * lam / (mu * (mu - lam))
        W = 1.0 / (mu - lam)
        Wq = lam / (mu * (mu - lam))
        return QueueResult(L, Lq, W, Wq, rho)


class MMCQueue(BaseQueue):
    """M/M/c queue (c parallel exponential servers)."""

    def __init__(self, n_servers=2):
        self.n_servers = int(n_servers)
        if self.n_servers < 1:
            raise ValueError("n_servers must be >= 1")

    def fit(self, arrival_rate, service_rate, **kw):
        c = self.n_servers
        _validate_rates(arrival_rate, service_rate, n_servers=c)
        lam, mu = float(arrival_rate), float(service_rate)
        a = lam / mu
        rho = a / c
        # Erlang C via birth-death module (verified implementation)
        from stochpylib.queueing.birth_death import erlang_c_formula as _ec
        erlang_c = _ec(c, a)
        # Mean queue length via Erlang C formula
        Lq = erlang_c * a / max(c - a, 1e-15) if c > a else float('inf')
        L = a + Lq
        W = L / lam
        Wq = Lq / lam
        return QueueResult(L, Lq, W, Wq, rho,
                           erlang_c=erlang_c, n_servers=c)


class MMInfinityQueue(BaseQueue):
    """M/M/∞ queue (infinitely many servers — no waiting)."""

    def fit(self, arrival_rate, service_rate, **kw):
        if arrival_rate <= 0 or service_rate <= 0:
            raise ValueError("rates must be positive")
        lam, mu = float(arrival_rate), float(service_rate)
        a = lam / mu
        return QueueResult(L=a, Lq=0.0, W=1.0 / mu, Wq=0.0, rho=0.0)


class MD1Queue(BaseQueue):
    """M/D/1 queue (deterministic service time = 1/mu)."""

    def fit(self, arrival_rate, service_rate, **kw):
        _validate_rates(arrival_rate, service_rate)
        lam, mu = float(arrival_rate), float(service_rate)
        rho = lam / mu
        es2 = 1.0 / mu ** 2                    # deterministic: E[S^2]=1/mu^2
        Lq = lam ** 2 * es2 / (2.0 * (1.0 - rho))
        L = rho + Lq
        Wq = Lq / lam
        W = Wq + 1.0 / mu
        return QueueResult(L, Lq, W, Wq, rho, deterministic=True)


class MG1Queue(BaseQueue):
    """M/G/1 queue via Pollaczek–Khinchine formula.

    Requires ``second_moment`` = E[S²]; defaults to the M/M/1 value 2/mu²
    if not supplied.
    """

    def fit(self, arrival_rate, service_rate, second_moment=None, **kw):
        _validate_rates(arrival_rate, service_rate)
        lam, mu = float(arrival_rate), float(service_rate)
        rho = lam / mu
        if second_moment is None:
            second_moment = 2.0 / mu ** 2     # exponential default
        es2 = float(second_moment)
        cv_squared = es2 * mu * mu - 1.0       # coefficient of variation²
        Lq = rho ** 2 * (1.0 + cv_squared) / (2.0 * (1.0 - rho))
        L = rho + Lq
        Wq = Lq / lam
        W = Wq + 1.0 / mu
        return QueueResult(L, Lq, W, Wq, rho,
                           second_moment=es2, cv_squared=cv_squared)


class GIGQueue(BaseQueue):
    """GI/G/1 approximate queue (Kingman / Allen-Cunneen).

    Uses the heavy-traffic approximation with separate arrival and service
    coefficients of variation.  Accuracy degrades as rho → 1 from below for
    highly variable inputs; documented approximation.
    """

    def fit(self, arrival_rate, service_rate,
            arrival_cv=1.0, service_cv=1.0, **kw):
        _validate_rates(arrival_rate, service_rate)
        lam, mu = float(arrival_rate), float(service_rate)
        rho = lam / mu
        ca = float(arrival_cv)
        cs = float(service_cv)
        # Kingman's approximation for mean waiting time
        wq_kingman = rho / (1 - rho) * \
            (ca ** 2 + cs ** 2) / 2.0 / mu
        # Allen-Cunneen correction factor
        ac_factor = 0.5 * (ca ** 2 + cs ** 2)
        wq = wq_kingman * ac_factor / max(ac_factor, 1e-12)  # already same
        wq = rho / (1.0 - rho) * (ca ** 2 + cs ** 2) / (2.0 * mu)
        lq = lam * wq
        w = wq + 1.0 / mu
        big_l = rho + lq
        return QueueResult(big_l, lq, w, wq, rho,
                           kingman=True, ca=ca, cs=cs)


class GI1Queue(BaseQueue):
    """GI/G/1 queue — alias of :class:`GIGQueue` (same model, spec naming)."""

    def fit(self, arrival_rate, service_rate,
            arrival_cv=1.0, service_cv=1.0, **kw):
        return GIGQueue().fit(arrival_rate, service_rate,
                              arrival_cv=arrival_cv,
                              service_cv=service_cv, **kw)


class MG1PriorityQueue(BaseQueue):
    """M/G/1 two-class non-preemptive priority queue.

    Class 1 has strict non-preemptive priority over class 2.
    Both classes share the same server.

    Parameters
    ----------
    arrival_rate_1, arrival_rate_2 : float
    service_rate_1, service_rate_2 : float
    second_moment_1, second_moment_2 : float or None
        E[S_i^2]; defaults to exponential 2/mu_i².

    Returns per-class and system-wide QueueResults in ``self.results_`` dict.
    """

    def fit(self, arrival_rate_1, arrival_rate_2,
            service_rate_1, service_rate_2,
            second_moment_1=None, second_moment_2=None, **kw):
        lam1 = float(arrival_rate_1)
        lam2 = float(arrival_rate_2)
        mu1 = float(service_rate_1)
        mu2 = float(service_rate_2)
        if lam1 >= mu1:
            raise ValueError("priority class unstable: lambda_1 >= mu_1")
        total_load = lam1 / mu1 + lam2 / mu2
        if total_load >= 1.0:
            raise ValueError(f"system unstable: total load {total_load:.4f} >= 1")

        es2_1 = float(second_moment_1) if second_moment_1 else \
            2.0 / mu1 ** 2
        es2_2 = float(second_moment_2) if second_moment_2 else \
            2.0 / mu2 ** 2

        R = lam1 * es2_1 / 2.0 + lam2 * es2_2 / 2.0   # residual work
        rho1 = lam1 / mu1
        rho2 = lam2 / mu2

        # class-1 waiting time (non-preemptive priority)
        Wq1 = R / (1.0 - rho1)
        # class-2 waiting time
        Wq2 = R / ((1.0 - rho1) * (1.0 - rho1 - rho2))

        W1 = Wq1 + 1.0 / mu1
        W2 = Wq2 + 1.0 / mu2
        Lq1 = lam1 * Wq1
        Lq2 = lam2 * Wq2
        L1 = rho1 + Lq1
        L2 = rho2 + Lq2

        self.results_ = {
            "class_1": QueueResult(L1, Lq1, W1, Wq1, rho1),
            "class_2": QueueResult(L2, Lq2, W2, Wq2, rho2),
        }
        lam_tot = lam1 + lam2
        W_sys = (L1 + L2) / max(lam_tot, _EPS)
        Wq_sys = (Lq1 + Lq2) / max(lam_tot, _EPS)
        self.results_["system"] = QueueResult(
            L1 + L2, Lq1 + Lq2, W_sys, Wq_sys, rho1 + rho2)
        return self
