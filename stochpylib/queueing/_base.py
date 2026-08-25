"""Base class and shared result object for queueing models."""

import numpy as np

__all__ = ["QueueResult", "BaseQueue"]


class QueueResult:
    """Immutable container for steady-state queueing metrics.

    Attributes
    ----------
    L : float
        Mean number in system (including service).
    Lq : float
        Mean number waiting in queue.
    W : float
        Mean sojourn time (waiting + service).
    Wq : float
        Mean waiting time in queue.
    rho : float
        Server utilisation (traffic intensity).
    extras : dict
        Model-specific additional metrics.
    """

    __slots__ = ("L", "Lq", "W", "Wq", "rho", "extras")

    def __init__(self, L, Lq, W, Wq, rho, **extras):
        object.__setattr__(self, "L", float(L))
        object.__setattr__(self, "Lq", float(Lq))
        object.__setattr__(self, "W", float(W))
        object.__setattr__(self, "Wq", float(Wq))
        object.__setattr__(self, "rho", float(rho))
        object.__setattr__(self, "extras", dict(extras))

    def __setattr__(self, name, value):
        raise AttributeError("QueueResult is immutable")

    def to_dict(self):
        out = {"L": self.L, "Lq": self.Lq,
               "W": self.W, "Wq": self.Wq, "rho": self.rho}
        out.update(self.extras)
        return out

    def __repr__(self):
        inner = ", ".join(f"{k}={v:.4g}" if isinstance(v, float)
                          else f"{k}={v}"
                          for k, v in self.to_dict().items())
        return f"QueueResult({inner})"

    def __eq__(self, other):
        if not isinstance(other, QueueResult):
            return NotImplemented
        return (abs(self.L - other.L) < 1e-12 and
                abs(self.Lq - other.Lq) < 1e-12 and
                abs(self.W - other.W) < 1e-12 and
                abs(self.Wq - other.Wq) < 1e-12 and
                abs(self.rho - other.rho) < 1e-12)


class BaseQueue:
    """Abstract base for analytic queueing models."""

    def fit(self, arrival_rate: float, service_rate: float,
            **kw) -> QueueResult:
        raise NotImplementedError

    @classmethod
    def compute(cls, arrival_rate: float, service_rate: float,
                **init_kw) -> "BaseQueue":
        """Return a fitted instance (convenience factory)."""
        inst = cls(**init_kw)
        return inst


def _validate_rates(arrival_rate, service_rate, n_servers=1):
    if arrival_rate <= 0:
        raise ValueError(f"arrival_rate must be > 0, got {arrival_rate}")
    if service_rate <= 0:
        raise ValueError(f"service_rate must be > 0, got {service_rate}")
    total_service = n_servers * service_rate
    if arrival_rate >= total_service:
        raise ValueError(
            f"system is unstable: lambda={arrival_rate} >= "
            f"c*mu={total_service}")
