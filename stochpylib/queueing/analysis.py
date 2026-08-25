"""Classical queueing analysis: Little's Law and derived metrics.

These are thin, dependency-free wrappers around the fundamental relationships
that hold for any stable queueing system in steady state.
"""

import numpy as np

__all__ = [
    "LittleLaw", "traffic_intensity", "mean_waiting_time",
    "mean_queue_length", "server_utilization", "SojournTime",
    "WaitingTimeDistribution",
]


def LittleLaw(L=None, arrival_rate=None, waiting_time=None,
              through_rate=None):
    """Little's Law ``L = lambda * W`` solved for whichever argument is None.

    Parameters
    ----------
    L : float or None
        Mean number in system.
    arrival_rate : float or None
        Effective arrival rate lambda (also aliased as *through_rate*).
    waiting_time : float or None
        Mean sojourn time W.

    Exactly one of L, arrival_rate/through_rate, waiting_time must be None;
    it is computed from the other two.  Also returns Wq via Little's law on
    the queue if both Lq and Wq are available in the caller's context.
    """
    lam = arrival_rate if arrival_rate is not None else through_rate
    args = [L, lam, waiting_time]
    nones = sum(1 for a in args if a is None)
    if nones != 1:
        raise ValueError(
            "exactly one of L, arrival_rate/through_rate, "
            "waiting_time must be None")
    if L is None:
        return {"L": lam * waiting_time}
    if lam is None:
        return {"arrival_rate": L / waiting_time}
    return {"waiting_time": L / lam}


def traffic_intensity(arrival_rate, service_rate, n_servers=1):
    """rho = lambda / (c * mu)."""
    rho = arrival_rate / (n_servers * service_rate)
    return float(rho)


def mean_waiting_time(Lq, arrival_rate):
    """Wq = Lq / lambda (Little's Law applied to the queue only)."""
    if arrival_rate <= 0:
        raise ValueError("arrival_rate must be positive")
    return float(Lq / arrival_rate)


def mean_queue_length(Wq, arrival_rate):
    """Lq = lambda * Wq."""
    return float(arrival_rate * Wq)


def server_utilization(arrival_rate, service_rate, n_servers=1):
    """rho = lambda / (c * mu) — alias for traffic_intensity."""
    return traffic_intensity(arrival_rate, service_rate, n_servers)


class SojournTime:
    """Container computing sojourn time from waiting + service."""

    def __init__(self, waiting_time, service_time):
        self.waiting = float(waiting_time)
        self.service = float(service_time)
        self.total = self.waiting + self.service

    def __repr__(self):
        return (f"SojournTime(wait={self.waiting:.4g}, "
                f"service={self.service:.4g}, total={self.total:.4g})")


class WaitingTimeDistribution:
    """Exact / approximate waiting-time distribution for M/M/1 and M/M/c.

    For M/M/1 the waiting-time distribution is exponential with rate
    mu - lambda above zero, with an atom of probability 1 - rho at zero.
    For M/M/c the delay probability is Erlang C.
    """

    def __init__(self, model_type="MM1", arrival_rate=1.0, service_rate=2.0,
                 n_servers=1):
        self.model_type = str(model_type).upper()
        self.arrival_rate = float(arrival_rate)
        self.service_rate = float(service_rate)
        self.n_servers = int(n_servers)
        self.rho = self.arrival_rate / (self.n_servers * self.service_rate)

    def cdf(self, t):
        """P(W <= t), the waiting-time CDF."""
        t = np.atleast_1d(np.asarray(t, dtype=float))
        out = np.empty_like(t)
        if self.model_type == "MM1":
            rate = self.service_rate - self.arrival_rate
            atom = 1.0 - self.rho
            cont = self.rho * (1.0 - np.exp(-rate * t))
            out[:] = atom + cont
        elif self.model_type == "MMC":
            from stochpylib.queueing.birth_death import erlang_c_formula
            pc = erlang_c_formula(self.n_servers,
                                  self.arrival_rate / self.service_rate)
            mu_c = self.n_servers * self.service_rate - self.arrival_rate
            atom = 1.0 - pc
            cont = pc * (1.0 - np.exp(-mu_c * t))
            out[:] = atom + cont
        else:
            raise ValueError(f"unsupported model_type {self.model_type!r}")
        return np.clip(out, 0.0, 1.0)

    def sf(self, t):
        """P(W > t)."""
        return 1.0 - self.cdf(t)

    def mean(self):
        """Mean waiting time."""
        if self.model_type == "MM1":
            return self.rho / (self.service_rate - self.arrival_rate)
        if self.model_type == "MMC":
            from stochpylib.queueing.birth_death import erlang_c_formula
            pc = erlang_c_formula(self.n_servers,
                                  self.arrival_rate / self.service_rate)
            mu_c = self.n_servers * self.service_rate - self.arrival_rate
            return pc / mu_c
        raise ValueError(f"unsupported model_type {self.model_type!r}")
