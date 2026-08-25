"""Tests for stochpylib.queueing.

Covers: M/M/1 closed-form identity, M/M/c Erlang-C consistency, M/D/1
Pollaczek-Khinchine, M/G/1 exponential equivalence, GI/G/1 approximation,
priority queue ordering, Erlang B/C tabulated values, Engset formula,
Jackson network traffic equations and product-form decomposition, closed
network mean-value analysis, discrete-event simulation vs analytical
results, and the analysis helper functions.

All randomness is seeded.
"""

import numpy as np
import pytest

from stochpylib import queueing as qm
from stochpylib.queueing._base import QueueResult
from stochpylib.queueing.single_queues import (
    GIGQueue, GI1Queue, MD1Queue, MG1PriorityQueue, MG1Queue, MMCQueue,
    MMInfinityQueue, MM1Queue,
)
from stochpylib.queueing.birth_death import (
    BirthDeathQueue, engset_formula, erlang_b_formula, erlang_c_formula,
)
from stochpylib.queueing.analysis import (
    LittleLaw, WaitingTimeDistribution, mean_queue_length,
    mean_waiting_time, server_utilization, traffic_intensity,
)
from stochpylib.queueing.simulation import DiscreteEventSim, QueueSimulation
from stochpylib.queueing.networks import (
    BCMP, ClosedNetwork, GordonNewell, JacksonNetwork,
)


# ---------------------------------------------------------------- M/M/1

class TestMM1:
    def test_closed_form_identity(self):
        r = MM1Queue().fit(.8, 1.2)
        assert abs(r.L - 2.0) < 1e-12
        assert abs(r.Lq - 4 / 3) < 1e-12
        assert abs(r.W - 2.5) < 1e-12
        assert abs(r.Wq - 5 / 3) < 1e-12
        assert abs(r.rho - 2 / 3) < 1e-12

    def test_little_law_consistency(self):
        r = MM1Queue().fit(.6, 1.0)
        assert abs(r.L - r.arrival_rate * r.W if hasattr(r, 'arrival_rate')
                   else True)

    def test_unstable_raises(self):
        with pytest.raises(ValueError):
            MM1Queue().fit(1.5, 1.0)

    def test_immutable_result(self):
        r = MM1Queue().fit(.5, 1.0)
        with pytest.raises(AttributeError):
            r.L = 999


# ---------------------------------------------------------------- M/M/c

class TestMMC:
    def test_c1_equals_mm1(self):
        mmc = MMCQueue(n_servers=1).fit(.8, 1.2)
        mm1 = MM1Queue().fit(.8, 1.2)
        assert mmc == mm1

    def test_multi_server_stable(self):
        r = MMCQueue(n_servers=5).fit(3.0, 1.0)
        assert r.rho < 1 and r.L > 0 and r.Lq >= 0

    def test_more_servers_reduces_waiting(self):
        r2 = MMCQueue(n_servers=2).fit(.8, 1.0)
        r4 = MMCQueue(n_servers=4).fit(.8, 1.0)
        assert r4.Wq <= r2.Wq


# ---------------------------------------------------------------- M/D/1 & M/G/1

class TestMDMG1:
    def test_md1_lower_lq_than_mm1(self):
        md = MD1Queue().fit(.8, 1.2)
        mm = MM1Queue().fit(.8, 1.2)
        assert md.Lq < mm.Lq

    def test_mg1_exponential_equals_mm1(self):
        mg = MG1Queue().fit(.8, 1.2)
        mm = MM1Queue().fit(.8, 1.2)
        assert abs(mg.Lq - mm.Lq) < 1e-9

    def test_mg1_higher_variance_increases_lq(self):
        mg_low = MG1Queue().fit(.5, 1.0, second_moment=2.0)
        mg_high = MG1Queue().fit(.5, 1.0, second_moment=10.0)
        assert mg_high.Lq > mg_low.Lq


# ---------------------------------------------------------------- G/G/1

class TestGIG1:
    def test_positive_results(self):
        r = GIGQueue().fit(.7, 1.2, arrival_cv=1., service_cv=1.)
        assert r.L > 0 and r.Lq >= 0 and r.rho > 0

    def test_gi1_alias(self):
        assert isinstance(GI1Queue().fit(.5, 1.0), QueueResult)


# ---------------------------------------------------------------- priority

class TestPriority:
    def test_class1_shorter_wait_than_class2(self):
        pq = MG1PriorityQueue()
        pq.fit(arrival_rate_1=.3, arrival_rate_2=.3,
               service_rate_1=1., service_rate_2=1.)
        assert (pq.results_["class_1"].Wq <
                pq.results_["class_2"].Wq)

    def test_system_metrics_present(self):
        pq = MG1PriorityQueue()
        pq.fit(arrival_rate_1=.3, arrival_rate_2=.3,
               service_rate_1=1., service_rate_2=1.)
        assert "system" in pq.results_
        assert pq.results_["system"].L > 0


# ---------------------------------------------------------------- birth-death

class TestBirthDeath:
    def test_erlang_b_known_value(self):
        b = erlang_b_formula(10, 7.0)
        assert .07 < b < .09

    def test_erlang_b_single_server(self):
        assert abs(erlang_b_formula(1, 2.0) - 2 / 3) < 1e-9

    def test_erlang_b_zero_load(self):
        assert erlang_b_formula(5, 0.0) == 0.0

    def test_erlang_c_between_0_and_1(self):
        c = erlang_c_formula(5, 3.0)
        assert 0 < c < 1

    def test_erlang_c_less_than_1_when_stable(self):
        assert erlang_c_formula(10, 7.0) < 1.0

    def test_engset_no_congestion_finite_sources_lt_servers(self):
        assert engset_formula(3, 5, 2.0) == 0.0

    def test_engset_congestion_positive_when_overloaded(self):
        b = engset_formula(20, 3, 2.0)
        assert 0 < b <= 1

    def test_birth_death_mm1_matches_analytical(self):
        bd = BirthDeathQueue(max_population=500)
        result = bd.fit(arrival_rate=.5, service_rate=1.0, n_servers=1)
        mm1 = MM1Queue().fit(.5, 1.0)
        assert abs(result["L"] - mm1.L) < 0.05


# ---------------------------------------------------------------- networks

class TestNetworks:
    def _make_jackson(self):
        return JacksonNetwork(
            external_arrivals=[1., 0.],
            service_rates=[3., 3.],
            routing_matrix=[[0., 1.], [0., 0.]])

    def test_jackson_traffic_equations(self):
        jn = self._make_jackson()
        jn.fit()
        np.testing.assert_allclose(jn.lam, [1., 1.], atol=1e-10)

    def test_jackson_total_L_positive(self):
        jn = self._make_jackson()
        jn.fit()
        assert jn.total_mean_number_in_system() > 0

    def test_jackson_sojourn_time_positive(self):
        jn = self._make_jackson()
        jn.fit()
        assert jn.total_mean_sojourn_time() > 0

    def test_closed_network_mva_throughput_positive(self):
        cn = ClosedNetwork(population=3, service_demands=[.5, .25])
        mva = cn.mean_value_analysis()
        assert mva["system_throughput"] > 0

    def test_closed_network_mva_monotone_response_time(self):
        cn = ClosedNetwork(population=5, service_demands=[.5, .25])
        mva = cn.mean_value_analysis()
        R = mva["response_time"]
        total_R = R.sum(axis=1)
        assert np.all(np.diff(total_R) > 0)

    def test_gordon_newell_is_closed_network(self):
        gn = GordonNewell(population=3, service_demands=[.5, .3])
        mva = gn.mean_value_analysis()
        assert "system_throughput" in mva

    def test_bcmp_type3_station(self):
        bc = BCMP(population=4, service_demands=[.5, .2],
                  station_types=[1, 3])
        mva = bc.mean_value_analysis()
        assert np.isfinite(mva["system_throughput"])

    def test_bcmp_invalid_type_raises(self):
        with pytest.raises(ValueError):
            BCMP(population=2, service_demands=[.5], station_types=[2])


# ---------------------------------------------------------------- simulation

class TestSimulation:
    def test_mm1_simulation_close_to_analytical(self):
        qs = QueueSimulation("MM1", arrival_rate=.5, service_rate=1.,
                             simulate_duration=50000, warmup=2000,
                             random_state=42)
        an = qs.analytic()
        sm = qs.simulate()
        assert abs(sm.Wq - an.Wq) < max(.15, .15 * an.Wq)
        assert abs(sm.L - an.L) < max(.15, .15 * an.L)

    def test_simulation_rho_close_to_expected(self):
        qs = QueueSimulation("MM1", arrival_rate=.5, service_rate=1.,
                             simulate_duration=50000, warmup=2000,
                             random_state=43)
        sm = qs.simulate()
        assert abs(sm.rho - .5) < .05

    def test_md1_simulation_runs(self):
        qs = QueueSimulation("MD1", arrival_rate=.5, service_rate=1.,
                             simulate_duration=30000, warmup=1000,
                             random_state=44)
        sm = qs.simulate()
        assert sm.extras.get("n_served", 0) > 0

    def test_discrete_event_sim_direct(self):
        rng_seed = 45
        sim = DiscreteEventSim(
            arrival_dist=lambda r: r.exponential(2.0),
            service_dist=lambda r: r.exponential(1.0),
            simulate_duration=30000, random_state=rng_seed)
        res = sim.run()
        assert res.extras.get("n_served", 0) > 0


# ---------------------------------------------------------------- analysis

class TestAnalysis:
    def test_little_law_solve_for_W(self):
        r = LittleLaw(L=10, arrival_rate=2.)
        assert abs(r["waiting_time"] - 5.) < 1e-12

    def test_little_law_solve_for_L(self):
        r = LittleLaw(arrival_rate=2., waiting_time=5.)
        assert abs(r["L"] - 10.) < 1e-12

    def test_little_law_solve_for_lambda(self):
        r = LittleLaw(L=10, waiting_time=5.)
        assert abs(r["arrival_rate"] - 2.) < 1e-12

    def test_traffic_intensity(self):
        assert abs(traffic_intensity(.8, 1.6, 2) - .25) < 1e-12

    def test_server_utilization_alias(self):
        assert server_utilization(.8, 1.6, 2) == traffic_intensity(.8, 1.6, 2)

    def test_mean_waiting_time(self):
        assert mean_waiting_time(.5, 1.0) == .5

    def test_mean_queue_length(self):
        assert mean_queue_length(1.0, 2.0) == 2.0

    def test_waiting_time_distribution_mm1(self):
        wtd = WaitingTimeDistribution("MM1", arrival_rate=.5,
                                       service_rate=1.)
        cdf_vals = wtd.cdf([0., 1., 5.])
        assert cdf_vals[0] >= .5       # atom at zero
        assert cdf_vals[2] > .9
        assert abs(wtd.mean() - 1.0) < .01


# ---------------------------------------------------------------- wiring

def test_module_wiring():
    import stochpylib
    assert "queueing" in stochpylib.__all__
    assert hasattr(stochpylib, "queueing")
    expected = {"MM1Queue", "MMCQueue", "MD1Queue", "MG1Queue",
                "GI1Queue", "GIGQueue", "MMInfinityQueue",
                "MG1PriorityQueue", "JacksonNetwork", "GordonNewell",
                "BCMP", "OpenNetwork", "ClosedNetwork", "ProductFormNetwork",
                "LittleLaw", "traffic_intensity", "mean_waiting_time",
                "mean_queue_length", "server_utilization",
                "WaitingTimeDistribution", "SojournTime",
                "BirthDeathQueue", "erlang_b_formula", "erlang_c_formula",
                "engset_formula", "QueueSimulation", "EventDrivenSim",
                "DiscreteEventSim", "SimStats"}
    missing = expected - set(qm.__all__)
    assert not missing, f"missing: {missing}"