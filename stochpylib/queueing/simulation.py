"""Discrete-event queueing simulation with event-calendar engine."""

import heapq

import numpy as np

from stochpylib.queueing._base import QueueResult

__all__ = [
    "DiscreteEventSim", "EventDrivenSim", "SimStats", "QueueSimulation",
]


class SimStats:
    """Collects and reports queueing simulation statistics."""

    def __init__(self, n_servers=1):
        self.n_servers = n_servers
        self.waiting_times = []
        self.sojourn_times = []
        self.service_times = []
        self.n_served = 0
        self.n_arrivals = 0
        self.area_n_system = 0.0
        self.last_time = 0.0
        self._cur_n_sys = 0
        self._t_last = 0.0

    def _tick(self, now):
        dt = now - self._t_last
        if dt > 0:
            self.area_n_system += self._cur_n_sys * dt
        self._t_last = now

    def set_state(self, n_sys, now):
        self._tick(now)
        self._cur_n_sys = n_sys

    def record_service(self, svc):
        self.service_times.append(svc)

    def finalise(self, end_time):
        self._tick(end_time)
        self.last_time = end_time

    @property
    def mean_waiting_time(self):
        return float(np.mean(self.waiting_times)) if self.waiting_times else 0.

    @property
    def mean_sojourn_time(self):
        return float(np.mean(self.sojourn_times)) if self.sojourn_times else 0.

    @property
    def mean_service_time(self):
        return float(np.mean(self.service_times)) if self.service_times else 1.

    @property
    def mean_number_in_system(self):
        return self.area_n_system / max(self.last_time, 1e-12)

    def to_result(self):
        """Compute metrics via Little's Law (exact, avoids area bias)."""
        duration = max(self.last_time - self._warmup_start_time,
                       self.last_time * .01, 1e-12)
        lam = self.n_served / duration
        W = self.mean_sojourn_time
        Wq = self.mean_waiting_time
        L = lam * W
        Lq = lam * Wq
        mean_svc = self.mean_service_time
        c = max(self.n_servers, 1)
        rho = min(max(lam * mean_svc / c, 0.), 1.)
        return QueueResult(L, Lq, W, Wq, rho, n_served=self.n_served)

    @property
    def _warmup_start_time(self):
        return getattr(self, '_warmup', 0.0)


class DiscreteEventSim:
    """Exact discrete-event simulation for a single-station FIFO queue.

    Parameters
    ----------
    arrival_dist : callable(rng) -> interarrival time
    service_dist : callable(rng) -> service duration
    n_servers : int
    simulate_duration : float
    warmup : float — stats collected only after this time
    random_state : int or None
    """

    def __init__(self, arrival_dist=None, service_dist=None, n_servers=1,
                 simulate_duration=50000.0, warmup=2000.0,
                 random_state=None):
        self._arr_fn = arrival_dist or (lambda r: r.exponential(1.25))
        self._svc_fn = service_dist or (lambda r: r.exponential(0.8333))
        self._c = max(1, int(n_servers))
        self._duration = float(simulate_duration)
        self._warmup = min(float(warmup), self._duration * 0.9)
        self._seed = int(random_state) if random_state is not None else 42
        self.stats = SimStats(n_servers=self._c)
        self.stats._warmup = self._warmup

    def run(self):
        rng = np.random.default_rng(self._seed)
        st = self.stats
        cal = []                        # heap of (time, seq, kind)
        seq = [0]

        def push(t, kind):
            seq[0] += 1
            heapq.heappush(cal, (t, seq[0], kind))

        busy = 0
        waiting = []                    # (arrival_time,) FIFO
        end_T = self._duration

        first_arr = self._arr_fn(rng)
        if first_arr <= end_T:
            push(first_arr, "arr")

        while cal:
            ev_t, _, ev_kind = heapq.heappop(cal)
            if ev_t > end_T:
                break
            st.set_state(len(waiting) + busy, ev_t)

            if ev_kind == "arr":
                st.n_arrivals += 1
                nxt = ev_t + self._arr_fn(rng)
                if nxt <= end_T:
                    push(nxt, "arr")
                collect = ev_t >= self._warmup
                if busy < self._c:
                    busy += 1
                    svc = self._svc_fn(rng)
                    push(ev_t + svc, "dep")
                    st.record_service(svc)
                    if collect:
                        st.sojourn_times.append(svc)
                        st.waiting_times.append(0.0)
                        st.n_served += 1
                else:
                    waiting.append(ev_t)
            elif ev_kind == "dep":
                busy -= 1
                if waiting:
                    arr_t = waiting.pop(0)
                    wait = ev_t - arr_t
                    svc = self._svc_fn(rng)
                    push(ev_t + svc, "dep")
                    busy += 1
                    soj = wait + svc
                    st.record_service(svc)
                    if ev_t >= self._warmup:
                        st.waiting_times.append(wait)
                        st.sojourn_times.append(soj)
                        st.n_served += 1

        # drain remaining departures past end_time
        while cal:
            ev_t, _, ev_kind = heapq.heappop(cal)
            if ev_kind != "dep":
                continue
            if waiting:
                arr_t = waiting.pop(0)
                svc = self._svc_fn(rng)
                st.record_service(svc)
                if ev_t >= self._warmup:
                    st.waiting_times.append(ev_t - arr_t)
                    st.sojourn_times.append(ev_t + svc - arr_t)
                    st.n_served += 1
            else:
                break

        st.finalise(end_T)
        return st.to_result()


class EventDrivenSim(DiscreteEventSim):
    pass


class QueueSimulation:
    """Facade running both analytical and simulated results::

        qs = QueueSimulation("MM1", arrival_rate=.8, service_rate=1.2,
                             simulate_duration=50000, random_state=42)
        analytic = qs.analytic()
        simulated = qs.simulate()
    """

    def __init__(self, model="MM1", arrival_rate=.8, service_rate=1.2,
                 n_servers=1, second_moment=None,
                 simulate_duration=50000.0, warmup=2000.0,
                 random_state=None):
        from stochpylib.queueing.single_queues import (
            MM1Queue, MMCQueue, MD1Queue, MG1Queue,
        )
        makers = {"MM1": MM1Queue, "MMC": MMCQueue, "MD1": MD1Queue,
                  "MG1": MG1Queue}
        if model not in makers:
            raise ValueError(f"unknown model {model!r}; "
                             f"expected one of {sorted(makers)}")
        self.model_name = model
        self.arrival_rate = float(arrival_rate)
        self.service_rate = float(service_rate)
        self.n_servers = max(1, int(n_servers))
        self.second_moment = second_moment
        self.simulate_duration = float(simulate_duration)
        self.warmup = float(warmup)
        self.random_state = random_state
        self._maker = makers[model]

    def analytic(self):
        inst = self._maker()
        if hasattr(inst, 'n_servers'):
            inst.n_servers = self.n_servers
        kw = {}
        if self.model_name == "MG1" and self.second_moment is not None:
            kw["second_moment"] = self.second_moment
        return inst.fit(self.arrival_rate, self.service_rate, **kw)

    def simulate(self):
        import numpy as _np
        rng = _np.random.default_rng(
            12345 if self.random_state is None else int(self.random_state))
        mu = self.service_rate
        if self.model_name == "MD1":
            svc_fn = lambda r: 1.0 / mu
        else:
            svc_fn = lambda r: r.exponential(1.0 / mu)
        sim = DiscreteEventSim(
            arrival_dist=lambda r: r.exponential(1.0 / self.arrival_rate),
            service_dist=svc_fn,
            n_servers=self.n_servers,
            simulate_duration=self.simulate_duration,
            warmup=self.warmup,
            random_state=self.random_state,
        )
        return sim.run()