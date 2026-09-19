"""End-to-end API sweep for stochpylib.queueing: one realistic exercise per public
name. ``test_every_public_name_is_exercised`` fails the moment a name ships without one;
``test_exercise[<name>]`` runs each exercise as its own pytest case."""

import numpy as np
import pytest

from stochpylib import queueing as mod

EXERCISES = {}


def exercise(name):
    def deco(fn):
        EXERCISES[name] = fn
        return fn
    return deco


@exercise("BaseQueue")
def _base():
    assert issubclass(mod.MM1Queue, mod.BaseQueue)
    assert isinstance(mod.MM1Queue.compute(0.5, 1.0), mod.BaseQueue)


@exercise("QueueResult")
def _result():
    r = mod.QueueResult(L=1.0, Lq=0.5, W=2.0, Wq=1.0, rho=0.5, n_served=10)
    d = r.to_dict()
    assert d["L"] == 1.0 and r.rho == 0.5


@exercise("LittleLaw")
def _little():
    r = mod.LittleLaw(L=10, arrival_rate=2.0)
    assert r["waiting_time"] == pytest.approx(5.0)


@exercise("traffic_intensity")
def _traffic():
    assert mod.traffic_intensity(0.8, 1.0) == pytest.approx(0.8)
    assert mod.traffic_intensity(1.6, 1.0, n_servers=2) == pytest.approx(0.8)


@exercise("mean_waiting_time")
def _mean_wait():
    assert mod.mean_waiting_time(Lq=4.0, arrival_rate=0.8) == pytest.approx(5.0)


@exercise("mean_queue_length")
def _mean_queue():
    assert mod.mean_queue_length(Wq=5.0, arrival_rate=0.8) == pytest.approx(4.0)


@exercise("server_utilization")
def _utilization():
    assert mod.server_utilization(0.5, 1.0) == pytest.approx(0.5)


@exercise("SojournTime")
def _sojourn():
    s = mod.SojournTime(waiting_time=1.5, service_time=0.5)
    assert s.total == pytest.approx(2.0) and "SojournTime" in repr(s)


@exercise("WaitingTimeDistribution")
def _wtd():
    w = mod.WaitingTimeDistribution("MM1", arrival_rate=0.5, service_rate=1.0)
    assert 0 <= w.cdf(1.0) <= 1 and w.sf(1.0) == pytest.approx(1 - w.cdf(1.0))
    assert w.mean() == pytest.approx(1.0)  # Wq = rho / (mu - lambda) = 0.5 / 0.5


@exercise("MM1Queue")
def _mm1():
    q = mod.MM1Queue().fit(0.8, 1.0)
    assert q.L == pytest.approx(4.0) and q.Wq == pytest.approx(4.0)


@exercise("MMCQueue")
def _mmc():
    q = mod.MMCQueue(n_servers=2).fit(1.2, 1.0)
    assert 0 < q.rho < 1 and q.L > q.Lq > 0


@exercise("MMInfinityQueue")
def _mminf():
    q = mod.MMInfinityQueue().fit(3.0, 1.0)
    assert q.L == pytest.approx(3.0) and q.Wq == pytest.approx(0.0)


@exercise("MD1Queue")
def _md1():
    q = mod.MD1Queue().fit(0.5, 1.0)
    assert q.Lq == pytest.approx(0.25)  # rho^2 / (2 (1 - rho))


@exercise("MG1Queue")
def _mg1():
    q = mod.MG1Queue().fit(0.5, 1.0, second_moment=2.0)   # exponential service -> M/M/1
    assert q.L == pytest.approx(1.0)


@exercise("GI1Queue")
def _gi1():
    q = mod.GI1Queue().fit(0.5, 1.0, arrival_cv=0.5)
    assert 0 < q.Wq < 1.0


@exercise("GIGQueue")
def _gig():
    q = mod.GIGQueue().fit(0.5, 1.0, arrival_cv=1.0, service_cv=1.0)
    assert q.Wq == pytest.approx(1.0, abs=1e-9)  # Kingman reduces to M/M/1


@exercise("MG1PriorityQueue")
def _priority():
    q = mod.MG1PriorityQueue().fit(0.3, 0.3, 1.0, 1.0)
    assert q.results_["class_1"].Wq < q.results_["class_2"].Wq


@exercise("BirthDeathQueue")
def _birth_death():
    bd = mod.BirthDeathQueue(max_population=200)
    res = bd.fit(arrival_rate=0.5, service_rate=1.0, n_servers=1)
    assert abs(res["L"] - 1.0) < 0.05
    pi = bd.steady_state(lambda n: 0.5, lambda n: 1.0)
    assert abs(sum(pi) - 1.0) < 1e-9


@exercise("erlang_b_formula")
def _erlang_b():
    assert 0 < mod.erlang_b_formula(5, 3.0) < 1


@exercise("erlang_c_formula")
def _erlang_c():
    assert mod.erlang_c_formula(5, 3.0) >= mod.erlang_b_formula(5, 3.0)


@exercise("engset_formula")
def _engset():
    assert 0 < mod.engset_formula(20, 3, 2.0) <= 1


@exercise("ErlangBFormula")
def _erlang_b_alias():
    assert mod.ErlangBFormula(5, 3.0) == mod.erlang_b_formula(5, 3.0)


@exercise("ErlangCFormula")
def _erlang_c_alias():
    assert mod.ErlangCFormula(5, 3.0) == mod.erlang_c_formula(5, 3.0)


@exercise("EngsetFormula")
def _engset_alias():
    assert mod.EngsetFormula(20, 3, 2.0) == mod.engset_formula(20, 3, 2.0)


@exercise("DiscreteEventSim")
def _des():
    sim = mod.DiscreteEventSim(lambda r: r.exponential(2.0), lambda r: r.exponential(1.0),
                               simulate_duration=3000, warmup=200, random_state=0)
    res = sim.run()
    assert abs(res.rho - 0.5) < 0.1 and res.L > 0


@exercise("EventDrivenSim")
def _eds():
    res = mod.EventDrivenSim(simulate_duration=2000, warmup=100, random_state=1).run()
    assert 0 < res.rho < 1


@exercise("SimStats")
def _simstats():
    st = mod.SimStats(n_servers=1)
    st.set_state(0, 0.0)
    st.set_state(1, 1.0)
    st.record_service(0.5)
    st.finalise(2.0)
    assert st.n_servers == 1


@exercise("QueueSimulation")
def _qsim():
    qs = mod.QueueSimulation("MM1", arrival_rate=0.5, service_rate=1.0,
                             simulate_duration=5000, warmup=500, random_state=2)
    an, sm = qs.analytic(), qs.simulate()
    assert abs(sm.L - an.L) < 0.4


@exercise("JacksonNetwork")
def _jackson():
    jn = mod.JacksonNetwork([1.0, 0.0], [3.0, 3.0], routing_matrix=[[0.0, 1.0], [0.0, 0.0]])
    jn.fit()
    assert np.allclose(jn.lam, [1.0, 1.0]) and jn.total_mean_number_in_system() > 0
    assert jn.total_mean_sojourn_time() > 0


@exercise("OpenNetwork")
def _open():
    on = mod.OpenNetwork([1.0], [2.0]).fit()
    assert on.total_mean_number_in_system() == pytest.approx(1.0)


@exercise("GordonNewell")
def _gordon():
    mva = mod.GordonNewell(population=3, service_demands=[0.5, 0.3]).mean_value_analysis()
    assert mva["system_throughput"] > 0


@exercise("BCMP")
def _bcmp():
    mva = mod.BCMP(population=4, service_demands=[0.5, 0.2], station_types=[1, 3]).mean_value_analysis()
    assert np.isfinite(mva["system_throughput"])


@exercise("ClosedNetwork")
def _closed():
    mva = mod.ClosedNetwork(population=3, service_demands=[0.5, 0.25]).mean_value_analysis()
    assert mva["system_throughput"] > 0


@exercise("ProductFormNetwork")
def _product_form():
    assert issubclass(mod.ClosedNetwork, mod.ProductFormNetwork)


def test_every_public_name_is_exercised():
    assert set(EXERCISES) == set(mod.__all__), (
        set(mod.__all__) - set(EXERCISES), set(EXERCISES) - set(mod.__all__))


@pytest.mark.parametrize("name", sorted(mod.__all__))
def test_exercise(name):
    EXERCISES[name]()
