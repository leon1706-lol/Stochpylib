"""Queueing networks: Jackson (open), Gordon-Newell (closed), BCMP."""

import numpy as np

__all__ = [
    "JacksonNetwork", "OpenNetwork", "ClosedNetwork", "GordonNewell",
    "BCMP", "ProductFormNetwork",
]


def _solve_traffic_equations(external_arrivals, routing_matrix):
    """Solve lambda = gamma + lambda @ P for open networks."""
    K = len(external_arrivals)
    P = np.asarray(routing_matrix, dtype=float)
    if P.shape != (K, K):
        raise ValueError(f"routing_matrix must be ({K},{K}), got {P.shape}")
    gamma = np.asarray(external_arrivals, dtype=float)
    A = np.eye(K) - P.T
    det = np.linalg.det(A)
    if abs(det) < 1e-14:
        raise ValueError("network is not open: I-P^T is near-singular")
    # lambda = gamma + lambda @ P  =>  (I - P)^T lambda = gamma
    lam = np.linalg.solve(A, gamma)
    return lam


class JacksonNetwork:
    """Open Jackson network with Poisson external arrivals and probabilistic
    routing.  Each node is an M/M/c queue; the product-form solution applies::

        jn = JacksonNetwork(gamma=[1., .5], mu=[2., 1.5],
                            c=[2, 2], P=[[0,.3],[.2,0]])
        results = jn.fit()
        total_L = jn.total_mean_number_in_system()
    """

    def __init__(self, external_arrivals, service_rates,
                 routing_matrix=None, n_servers=None):
        self.gamma = np.asarray(external_arrivals, dtype=float).ravel()
        self.K = len(self.gamma)
        self.mu = np.asarray(service_rates, dtype=float).ravel()
        if len(self.mu) != self.K:
            raise ValueError("service_rates length must match nodes")
        if routing_matrix is None:
            self.P = np.zeros((self.K, self.K))
        else:
            self.P = np.asarray(routing_matrix, dtype=float)
        self.c = (np.full(self.K, 1, dtype=int) if n_servers is None
                  else np.broadcast_to(np.asarray(n_servers, int),
                                       (self.K,)).copy())
        try:
            self.lam = _solve_traffic_equations(self.gamma, self.P)
        except ValueError:
            lam = self.gamma.copy()
            for _ in range(500):
                lam_new = self.gamma + lam @ self.P
                if np.max(np.abs(lam_new - lam)) < 1e-12:
                    break
                lam = lam_new
            self.lam = lam

    def fit(self, **kw):
        from stochpylib.queueing.single_queues import MMCQueue
        results = {}
        for k in range(self.K):
            c_k = int(self.c[k])
            q = MMCQueue(n_servers=c_k)
            results[k] = q.fit(self.lam[k], self.mu[k])
        self.results_ = results
        return self

    def total_mean_number_in_system(self):
        if not hasattr(self, "results_"):
            self.fit()
        return sum(r.L for r in self.results_.values())

    def total_mean_sojourn_time(self):
        if not hasattr(self, "results_"):
            self.fit()
        total_lambda = float(np.sum(np.abs(self.lam)))
        weighted = sum(self.results_[k].L for k in range(self.K))
        return weighted / max(total_lambda, 1e-12)


class OpenNetwork(JacksonNetwork):
    """Alias for :class:`JacksonNetwork`."""


class ProductFormNetwork:
    """Base class for product-form queueing networks."""

    pass


class ClosedNetwork(ProductFormNetwork):
    """Closed network with fixed population and mean-value analysis::

        cn = ClosedNetwork(population=5, service_demands=[.5, .3])
        mva = cn.mean_value_analysis()
        mva["system_throughput"]
    """

    def __init__(self, population, service_demands, n_servers=None):
        self.N = int(population)
        self.D = np.asarray(service_demands, dtype=float).ravel()
        self.K = len(self.D)
        if n_servers is None:
            self.c = np.ones(self.K, dtype=int)
        else:
            self.c = np.broadcast_to(
                np.asarray(n_servers, int), (self.K,)).copy()

    def mean_value_analysis(self):
        D, K, N = self.D, self.K, self.N
        L = np.zeros((N + 1, K))
        X = np.zeros((N + 1, K))
        R = np.zeros((N + 1, K))
        for nn in range(1, N + 1):
            for k in range(K):
                R[nn, k] = D[k] * (1 + L[nn - 1, k])
            denom = float(np.sum(R[nn]))
            X[nn] = nn / denom if denom > 0 else 0.0
            L[nn] = X[nn] * R[nn]
        self.throughputs_ = X
        self.response_times_ = R
        self.queue_lengths_ = L
        self.system_throughput_ = float(X[N].max()) if K > 0 else 0.0
        self.total_response_time_ = float(np.sum(R[N]))
        return {
            "throughput": X,
            "response_time": R,
            "queue_length": L,
            "system_throughput": self.system_throughput_,
            "total_response_time": self.total_response_time_,
        }


class GordonNewell(ClosedNetwork):
    """Gordon-Newell closed queueing network (product form).

    Alias of :class:`ClosedNetwork` — same MVA computation.
    """


class BCMP(ProductFormNetwork):
    """BCMP theorem supporting type-1 (FCFS) and type-3 (IS) stations::

        bcmp = BCMP(population=4, service_demands=[.5, .2],
                    station_types=[1, 3])
        result = bcmp.mean_value_analysis()
    """

    def __init__(self, population, service_demands,
                 station_types=None, n_servers=None):
        self.N = int(population)
        self.D = np.asarray(service_demands, dtype=float).ravel()
        self.K = len(self.D)
        self.types = list(station_types) if station_types else [1] * self.K
        if any(t not in (1, 3) for t in self.types):
            raise ValueError("only types 1 (FCFS) and 3 (IS) supported")
        self.c = (np.ones(self.K, dtype=int) if n_servers is None
                  else np.broadcast_to(np.asarray(n_servers, int),
                                       (self.K,)).copy())

    def mean_value_analysis(self):
        K, N, D = self.K, self.N, self.D
        L = np.zeros((N + 1, K))
        R = np.zeros((N + 1, K))
        X_sys = np.zeros(N + 1)
        for nn in range(1, N + 1):
            for k in range(K):
                if self.types[k] == 3:
                    R[nn, k] = D[k]
                else:
                    R[nn, k] = D[k] * (1 + L[nn - 1, k])
            denom = float(np.sum(R[nn]))
            X_sys[nn] = nn / denom if denom > 0 else 0.0
            X_row = X_sys[nn]
            for k in range(K):
                if self.types[k] == 3:
                    # IS: X_k(n) = n_k * / D_k; use arrival-instant avg
                    L[nn, k] = X_row * R[nn, k]
                else:
                    L[nn, k] = X_row * R[nn, k]
        self.throughputs_ = np.array([
            [X_sys[nn] * L[nn, k] / max(L[nn].sum(), 1e-12)
             if L[nn].sum() > 0 else 0.0
             for k in range(K)] for nn in range(N + 1)])
        self.response_times_ = R
        self.queue_lengths_ = L
        self.system_throughput_ = float(X_sys[N])
        self.total_response_time_ = float(np.sum(R[N]))
        return {
            "throughput": self.throughputs_,
            "response_time": R,
            "queue_length": L,
            "system_throughput": self.system_throughput_,
            "total_response_time": self.total_response_time_,
        }