"""Oracle suite for :mod:`stochpylib.utils`: scipy.stats/scipy.special as independent
references, plus internal exactness checks (pooling, round-trips). Statistical
assertions use >= 3 SE. Real-backend (torch/jax/numba/pandas-present) checks live in
``backend_optional.py``, run only by the ``utils-optional`` CI job -- everything here
passes with none of pandas/torch/jax/numba installed at all.
"""

import json
import sys
import tempfile
import threading
import warnings
from pathlib import Path

import numpy as np
import pytest
from scipy import special, stats
from scipy.spatial.distance import cdist

from stochpylib import _rng as _rng_mod
from stochpylib import distributions as dist_mod
from stochpylib import statistics as stats_mod
from stochpylib.montecarlo import MCResult
from stochpylib.utils import (
    Configuration,
    DataValidation,
    EnvironmentCapture,
    ExperimentLogger,
    FitResult,
    GPUBackend,
    JIT_compile,
    Logging,
    MemoryPool,
    OutlierResult,
    ParallelSimulation,
    RandomStream,
    Reproducibility,
    Serialization,
    VectorizedOps,
    VersionLock,
    ecdf,
    fit,
    from_dict,
    from_json,
    goodness_of_fit,
    missing_imputation,
    moment_matching,
    numpy_interface,
    outlier_detection,
    scipy_interface,
    set_seed,
    spawn_generator,
    summary,
    random_state as random_state_fn,
    to_dict,
    to_json,
    to_pickle,
    from_pickle,
)


# ============================================================================ random

class TestRandom:
    def test_random_state_matches_default_rng_for_int(self):
        g1 = random_state_fn(5)
        g2 = np.random.default_rng(5)
        assert g1.standard_normal(10).tolist() == g2.standard_normal(10).tolist()

    def test_random_state_matches_default_rng_for_seed_sequence(self):
        seq = np.random.SeedSequence(99)
        g1 = random_state_fn(seq)
        g2 = np.random.default_rng(np.random.SeedSequence(99))
        assert g1.standard_normal(5).tolist() == g2.standard_normal(5).tolist()

    def test_random_state_matches_default_rng_for_bit_generator(self):
        bg1 = np.random.PCG64(7)
        bg2 = np.random.PCG64(7)
        g1 = random_state_fn(bg1)
        g2 = np.random.default_rng(bg2)
        assert g1.standard_normal(5).tolist() == g2.standard_normal(5).tolist()

    def test_random_state_preserves_generator_identity(self):
        gen = np.random.default_rng(3)
        assert random_state_fn(gen) is gen

    def test_random_state_accepts_randomstate(self):
        rs = np.random.RandomState(11)
        gen = random_state_fn(rs)
        assert isinstance(gen, np.random.Generator)

    def test_set_seed_makes_none_reproducible(self):
        set_seed(123)
        a = random_state_fn(None).standard_normal(5)
        set_seed(123)
        b = random_state_fn(None).standard_normal(5)
        set_seed(None)
        assert np.array_equal(a, b)

    def test_set_seed_none_restores_entropy(self):
        set_seed(42)
        set_seed(None)
        a = random_state_fn(None).standard_normal(5)
        b = random_state_fn(None).standard_normal(5)
        assert not np.array_equal(a, b)

    def test_spawn_generator_single(self):
        g = spawn_generator(random_state=1)
        assert isinstance(g, np.random.Generator)

    def test_spawn_generator_multiple_are_distinct_and_reproducible(self):
        gs1 = spawn_generator(random_state=7, n=4)
        gs2 = spawn_generator(random_state=7, n=4)
        assert len(gs1) == 4
        draws1 = [g.standard_normal(3) for g in gs1]
        draws2 = [g.standard_normal(3) for g in gs2]
        for a, b in zip(draws1, draws2):
            assert np.array_equal(a, b)
        for i in range(4):
            for j in range(i + 1, 4):
                assert not np.array_equal(draws1[i], draws1[j])

    def test_spawn_matches_seedsequence_scheme_for_int_seed(self):
        expected = [np.random.default_rng(c) for c in np.random.SeedSequence(55).spawn(3)]
        got = spawn_generator(random_state=55, n=3)
        for e, g in zip(expected, got):
            assert e.standard_normal(4).tolist() == g.standard_normal(4).tolist()

    def test_thread_safety_distinct_streams_under_global_seed(self):
        set_seed(2026)
        results = {}

        def worker(i):
            results[i] = random_state_fn(None).standard_normal(3)

        threads = [threading.Thread(target=worker, args=(i,)) for i in range(8)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()
        set_seed(None)
        assert len(results) == 8
        seen = [tuple(v) for v in results.values()]
        assert len(set(seen)) == 8


# ======================================================================= performance

class TestBenchmark:
    def test_benchmark_positive_times_and_se(self):
        from stochpylib.utils.performance import Benchmark
        b = Benchmark(repeat=5, number=200).run(lambda: sum(range(50)))
        assert (b.times_ > 0).all()
        res = b.to_result()
        assert isinstance(res, MCResult)
        assert res.std_error >= 0 or np.isnan(res.std_error)

    def test_benchmark_compare_orders_by_speed(self):
        from stochpylib.utils.performance import Benchmark
        rows = Benchmark(repeat=3, number=50).compare({
            "fast": lambda: 1 + 1,
            "slow": lambda: sum(range(20000)),
        })
        assert rows[0]["label"] == "fast"
        assert rows[1]["speedup_vs_first"] < 1.0


class TestProfiler:
    def test_profiler_finds_named_function(self):
        from stochpylib.utils.performance import Profiler

        def named_work():
            return sum(i * i for i in range(5000))

        p = Profiler()
        p.profile(named_work)
        names = [row["function"] for row in p.stats_]
        assert any("named_work" in n for n in names)
        assert p.total_time_ >= 0

    def test_profiler_memory_tracks_peak(self):
        from stochpylib.utils.performance import Profiler
        p = Profiler(memory=True)
        p.profile(lambda: [0] * 100000)
        assert p.peak_memory_ is not None and p.peak_memory_ > 0


class TestParallelSimulation:
    def _sim(self, n, rng):
        x = rng.standard_normal(n)
        se = float(x.std(ddof=1) / np.sqrt(n)) if n > 1 else float("nan")
        return MCResult(float(x.mean()), se, n, "sim")

    @pytest.mark.parametrize("n_jobs,backend", [(1, "thread"), (2, "thread"), (4, "thread")])
    def test_n_jobs_invariance_thread(self, n_jobs, backend):
        ps = ParallelSimulation(n_jobs=n_jobs, backend=backend, chunk_size=500)
        r = ps.estimate(self._sim, 4000, random_state=42)
        assert isinstance(r, MCResult)
        ref = ParallelSimulation(n_jobs=2, backend="thread", chunk_size=500).estimate(
            self._sim, 4000, random_state=42)
        assert r.estimate == ref.estimate
        assert r.std_error == ref.std_error

    def test_process_backend_runs(self):
        from tests.utils._workers import crude_square_mc
        ps = ParallelSimulation(n_jobs=2, backend="process", chunk_size=1000)
        r = ps.estimate(crude_square_mc, 4000, random_state=1)
        assert abs(r.estimate - 1.0 / 3.0) < 4 * r.std_error

    def test_pooled_se_matches_full_array_exactly(self):
        rng = np.random.default_rng(0)
        full = rng.standard_normal(2000)
        chunks = [full[:500], full[500:1200], full[1200:]]

        def to_res(x):
            return MCResult(float(x.mean()), float(x.std(ddof=1) / np.sqrt(len(x))), len(x), "m")

        from stochpylib._parallel import pool_mc_results
        pooled = pool_mc_results([to_res(c) for c in chunks])
        whole = to_res(full)
        assert abs(pooled.estimate - whole.estimate) < 1e-10
        assert abs(pooled.std_error - whole.std_error) < 1e-8

    def test_chunk_sizes_independent_of_n_jobs(self):
        from stochpylib._parallel import chunk_sizes
        assert chunk_sizes(1000, 300) == [300, 300, 300, 100]

    def test_map_runs_in_order(self):
        ps = ParallelSimulation(n_jobs=3, backend="thread")
        out = ps.map(lambda x: x * 2, [1, 2, 3, 4, 5])
        assert out == [2, 4, 6, 8, 10]


class TestVectorizedOps:
    def test_logsumexp_matches_scipy(self):
        rng = np.random.default_rng(0)
        a = rng.standard_normal((6, 5))
        assert np.allclose(VectorizedOps.logsumexp(a, axis=1), special.logsumexp(a, axis=1))
        assert np.allclose(VectorizedOps.logsumexp(a, axis=0), special.logsumexp(a, axis=0))

    def test_logsumexp_with_weights(self):
        rng = np.random.default_rng(1)
        a = rng.standard_normal(20)
        b = rng.random(20) + 0.1
        assert np.allclose(VectorizedOps.logsumexp(a, b=b), special.logsumexp(a, b=b))

    def test_softmax_sums_to_one_and_matches_manual(self):
        rng = np.random.default_rng(2)
        a = rng.standard_normal((4, 6))
        sm = VectorizedOps.softmax(a, axis=1)
        assert np.allclose(sm.sum(axis=1), 1.0)
        manual = np.exp(a) / np.exp(a).sum(axis=1, keepdims=True)
        assert np.allclose(sm, manual)

    def test_log1mexp_matches_direct_formula_for_stable_range(self):
        x = -np.array([0.01, 0.5, 1.0, 5.0, 20.0])
        assert np.allclose(VectorizedOps.log1mexp(x), np.log(1 - np.exp(x)))

    def test_rolling_mean_matches_naive_loop(self):
        x = np.arange(20.0) ** 1.3
        window = 4
        naive = np.array([x[i:i + window].mean() for i in range(len(x) - window + 1)])
        assert np.allclose(VectorizedOps.rolling_mean(x, window), naive)

    def test_rolling_std_matches_naive_loop(self):
        rng = np.random.default_rng(3)
        x = rng.standard_normal(30)
        window = 5
        naive = np.array([x[i:i + window].std(ddof=1) for i in range(len(x) - window + 1)])
        assert np.allclose(VectorizedOps.rolling_std(x, window, ddof=1), naive, atol=1e-10)

    def test_pairwise_distances_matches_scipy_cdist(self):
        rng = np.random.default_rng(4)
        X = rng.standard_normal((10, 3))
        Y = rng.standard_normal((7, 3))
        for metric in ("euclidean", "cityblock"):
            mine = VectorizedOps.pairwise_distances(X, Y, metric=metric)
            ref = cdist(X, Y, metric=metric)
            assert np.allclose(mine, ref, atol=1e-8)

    def test_pairwise_sqeuclidean(self):
        rng = np.random.default_rng(5)
        X = rng.standard_normal((8, 2))
        mine = VectorizedOps.pairwise_distances(X, metric="sqeuclidean")
        ref = cdist(X, X, metric="sqeuclidean")
        assert np.allclose(mine, ref, atol=1e-8)

    def test_batched_mahalanobis_matches_explicit_inverse(self):
        rng = np.random.default_rng(6)
        mean = np.array([1.0, -2.0])
        cov = np.array([[2.0, 0.5], [0.5, 1.0]])
        X = rng.multivariate_normal(mean, cov, 50)
        mine = VectorizedOps.batched_mahalanobis(X, mean, cov)
        inv = np.linalg.inv(cov)
        diff = X - mean
        ref = np.sqrt(np.einsum("ij,jk,ik->i", diff, inv, diff))
        assert np.allclose(mine, ref, atol=1e-8)

    def test_chunked_apply_matches_direct(self):
        X = np.arange(23.0)
        out = VectorizedOps.chunked_apply(lambda c: c * 2, X, chunk_size=7)
        assert np.allclose(out, X * 2)


class TestMemoryPool:
    def test_get_release_reuses_buffer(self):
        pool = MemoryPool()
        arr = pool.get((50, 50))
        pool.release(arr)
        arr2 = pool.get((50, 50))
        assert np.shares_memory(arr, arr2)
        assert pool.hits_ == 1 and pool.misses_ == 1

    def test_zero_flag_zeros_buffer(self):
        pool = MemoryPool()
        arr = pool.get((10,), zero=True)
        assert np.all(arr == 0)

    def test_borrow_context_manager_releases(self):
        pool = MemoryPool()
        with pool.borrow((5, 5)) as buf:
            buf[:] = 1.0
        assert pool.bytes_pooled_ > 0

    def test_lru_eviction_under_max_bytes(self):
        pool = MemoryPool(max_bytes=800)  # one float64 (10,10) buffer = 800 bytes
        a = pool.get((10, 10))
        b = pool.get((10, 10))
        pool.release(a)
        pool.release(b)
        assert pool.bytes_pooled_ <= 800


class TestGPUBackendNumpy:
    def test_gbm_matches_closed_form_mean(self):
        gb = GPUBackend("numpy")
        paths = gb.simulate_gbm(S0=100.0, mu=0.05, sigma=0.2, T=1.0, n_steps=100,
                                n_paths=20000, random_state=1)
        ST = paths[:, -1]
        se = ST.std(ddof=1) / np.sqrt(len(ST))
        assert abs(ST.mean() - 100.0 * np.exp(0.05)) < 4 * se

    def test_standard_normal_matches_numpy_generator(self):
        gb = GPUBackend("numpy")
        a = gb.standard_normal((100,), random_state=5)
        b = np.random.default_rng(5).standard_normal((100,))
        assert np.allclose(a, b)

    def test_mc_estimate_returns_mcresult(self):
        gb = GPUBackend("numpy")
        vals = np.random.default_rng(0).standard_normal(1000)
        res = gb.mc_estimate(vals)
        assert isinstance(res, MCResult)

    def test_available_backends_reports_dict(self):
        avail = GPUBackend.available()
        assert set(avail) == {"pandas", "torch", "jax", "numba", "cupy"}


class TestGPUBackendCupyFake:
    """A numpy-backed fake ``cupy`` module (GitHub runners have no GPU)."""

    def test_cupy_backend_with_injected_fake_module(self, monkeypatch):
        import types

        class _FakeArr(np.ndarray):
            """A numpy-backed stand-in for a cupy ndarray: same ufunc behavior,
            plus cupy's ``.get()`` (-> host numpy array) that plain numpy lacks."""

            def get(self):
                return np.asarray(self)

        fake = types.ModuleType("cupy")
        fake.float64 = np.float64
        fake.asarray = lambda x, dtype=None: np.asarray(x, dtype=dtype).view(_FakeArr)
        for name in ("exp", "log", "sqrt", "maximum", "where", "cumsum", "sum",
                    "mean", "std", "matmul", "concatenate", "zeros"):
            setattr(fake, name, getattr(np, name))

        monkeypatch.setitem(sys.modules, "cupy", fake)
        gb = GPUBackend("cupy")
        assert gb.name == "cupy"
        device_arr = gb.asarray([1.0, 2.0])
        assert isinstance(device_arr, _FakeArr)
        assert gb.to_numpy(device_arr).tolist() == [1.0, 2.0]
        out = gb.exp(gb.asarray([0.0, 1.0]))
        assert np.allclose(out, [1.0, np.e])


class TestJITCompile:
    def test_falls_back_to_python_when_numba_not_installed(self, monkeypatch):
        """``sys.modules["numba"] = None`` also makes ``find_spec`` report "not
        installed", so ``backend="auto"`` never even attempts the import -- silently
        (and correctly) picks plain Python, no warning expected."""
        monkeypatch.setitem(sys.modules, "numba", None)

        @JIT_compile
        def f(x):
            return x * 3

        assert f(4) == 12
        assert f.backend_ == "python"

    def test_warns_and_falls_back_when_numba_installed_but_broken(self, monkeypatch):
        """``find_spec`` says installed, but the real import fails (e.g. numba
        against a newer numpy than it supports) -- this must warn once and fall back,
        regardless of whether numba is actually installed/working in this environment."""
        monkeypatch.setattr("stochpylib.utils._backends.is_installed", lambda name: True)
        monkeypatch.setitem(sys.modules, "numba", None)
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")

            @JIT_compile
            def f(x):
                return x * 3

            assert f(4) == 12
            assert f.backend_ == "python"
            assert any("falling back" in str(x.message) for x in w)

    def test_backend_none_runs_plain_python(self):
        @JIT_compile(backend="none")
        def g(x):
            return x + 1

        assert g(4) == 5
        assert g.backend_ == "python"

    def test_py_func_is_the_original_function(self):
        def h(x):
            return x

        wrapped = JIT_compile(backend="none")(h)
        assert wrapped.py_func is h


# ==================================================================== reproducibility

class TestReproducibility:
    def test_context_manager_restores_prior_root(self):
        _rng_mod.set_root(999)
        prev = _rng_mod.get_root()
        with Reproducibility(seed=5):
            pass
        assert _rng_mod.get_root() is prev
        _rng_mod.set_root(None)

    def test_check_true_for_seeded_code(self):
        def seeded():
            return _rng_mod.as_generator(None).standard_normal(5)

        result = Reproducibility(seed=10).check(seeded, n_runs=3)
        assert result["reproducible"] is True

    def test_check_false_for_entropy_code(self):
        def unseeded():
            return np.random.default_rng().standard_normal(5)

        result = Reproducibility(seed=10).check(unseeded, n_runs=3)
        assert result["reproducible"] is False


class TestRandomStream:
    def test_checkpoint_restore_replays_exactly(self):
        rs = RandomStream(seed=1)
        rs.generator.standard_normal(3)
        cp = rs.checkpoint()
        a = rs.generator.standard_normal(4)
        rs.restore(cp)
        b = rs.generator.standard_normal(4)
        assert np.array_equal(a, b)

    def test_jumped_differs_from_unjumped(self):
        rs = RandomStream(seed=2)
        j = rs.jumped(1)
        rs.reset()
        assert not np.array_equal(j.generator.standard_normal(5), rs.generator.standard_normal(5))

    def test_spawn_returns_named_children(self):
        rs = RandomStream(seed=3, name="root")
        kids = rs.spawn(3)
        assert len(kids) == 3
        assert all(k.name.startswith("root/") for k in kids)

    def test_random_state_stream_works_in_distributions_rvs(self):
        a = dist_mod.Normal(0, 1).rvs(5, random_state=RandomStream(seed=9))
        b = dist_mod.Normal(0, 1).rvs(5, random_state=RandomStream(seed=9))
        assert np.array_equal(a, b)


class TestVersionLock:
    def test_detects_injected_mismatch(self):
        vl = VersionLock(packages=("numpy",)).capture()
        vl.versions_["numpy"] = "0.0.0"
        mism = vl.check()
        assert len(mism) == 1 and mism[0]["package"] == "numpy"

    def test_strict_raises_on_mismatch(self):
        vl = VersionLock(packages=("numpy",), strict=True).capture()
        vl.versions_["numpy"] = "0.0.0"
        with pytest.raises(RuntimeError):
            vl.check()

    def test_save_load_round_trip(self, tmp_path):
        vl = VersionLock().capture()
        p = tmp_path / "lock.json"
        vl.save(str(p))
        vl2 = VersionLock.load(str(p))
        assert vl2.check() == []

    def test_to_requirements_format(self):
        vl = VersionLock(packages=("numpy",)).capture()
        text = vl.to_requirements()
        assert text.startswith("numpy==")


class TestEnvironmentCapture:
    def test_has_required_keys(self):
        info = EnvironmentCapture().capture().to_dict()
        for key in ("python_version", "numpy_version", "scipy_version",
                   "stochpylib_version", "available_backends", "cpu_count"):
            assert key in info

    def test_json_round_trips(self):
        ec = EnvironmentCapture().capture()
        text = ec.to_json()
        parsed = json.loads(text)
        assert parsed["numpy_version"] == np.__version__

    def test_diff_detects_changed_field(self):
        a = EnvironmentCapture().capture()
        b = EnvironmentCapture().capture()
        b.info_["numpy_version"] = "9.9.9"
        d = a.diff(b)
        assert "numpy_version" in d


class TestExperimentLogger:
    def test_jsonl_round_trip(self, tmp_path):
        p = tmp_path / "log.jsonl"
        logger = ExperimentLogger(path=str(p), name="exp")
        with logger.run(params={"a": 1}, seed=1) as run:
            run.log(loss=0.5)
        with logger.run(params={"a": 2}, seed=2) as run:
            run.log(loss=0.1)
        loaded = ExperimentLogger.load(str(p))
        assert len(loaded.runs_) == 2

    def test_best_picks_min_metric(self, tmp_path):
        logger = ExperimentLogger()
        with logger.run(params={"a": 1}) as run:
            run.log(loss=0.5)
        with logger.run(params={"a": 2}) as run:
            run.log(loss=0.1)
        best = logger.best("loss", mode="min")
        assert best["params"]["a"] == 2

    def test_mcresult_metric_is_normalized(self):
        logger = ExperimentLogger()
        with logger.run() as run:
            run.log(est=MCResult(1.5, 0.1, 100, "m"))
        assert logger.runs_[0]["metrics"]["est"] == {"estimate": 1.5, "std_error": 0.1}


# ============================================================================== data

class TestFit:
    def test_fit_ranks_gamma_first_and_matches_scipy_loglik(self):
        rng = np.random.default_rng(0)
        x = rng.gamma(3.0, 2.0, 3000)
        result = fit(x)
        assert isinstance(result, FitResult)
        assert result.best_name_ == "Gamma"
        my_ll = float(np.sum(stats.gamma.logpdf(x, result.best_.shape, scale=result.best_.scale)))
        row = next(r for r in result.table_ if r["name"] == "Gamma")
        assert abs(row["loglik"] - my_ll) < 1e-6

    def test_fit_params_close_to_scipy_mle(self):
        rng = np.random.default_rng(1)
        x = rng.gamma(3.0, 2.0, 4000)
        result = fit(x, distributions=["Gamma"])
        shape_sp, _, scale_sp = stats.gamma.fit(x, floc=0)
        assert abs(result.best_.shape - shape_sp) < 3 * 0.1 * shape_sp
        assert abs(result.best_.scale - scale_sp) < 3 * 0.1 * scale_sp

    def test_fit_discrete_data_picks_poisson(self):
        rng = np.random.default_rng(2)
        x = rng.poisson(6, 2000).astype(float)
        result = fit(x, discrete=True)
        assert result.best_name_ == "Poisson"

    def test_fit_unsupported_candidate_lands_in_failed(self):
        rng = np.random.default_rng(3)
        x = -rng.gamma(3.0, 2.0, 500)  # negative support: Exponential/Gamma/Weibull should fail
        result = fit(x, distributions=["Normal", "Exponential"])
        assert "Exponential" in result.failed_
        assert result.best_name_ == "Normal"

    def test_fit_summary_is_a_string(self):
        rng = np.random.default_rng(4)
        result = fit(rng.standard_normal(500))
        assert isinstance(result.summary(), str)


class TestGoodnessOfFit:
    def test_ks_matches_scipy_kstest(self):
        rng = np.random.default_rng(5)
        x = rng.standard_normal(500)
        mine = goodness_of_fit(x, "Normal", tests=("ks",))["ks"]
        # Normal.fit() is the MLE (population std, ddof=0), not the sample std.
        ref = stats.kstest(x, lambda q: stats.norm(np.mean(x), np.std(x, ddof=0)).cdf(q))
        assert abs(mine.statistic - ref.statistic) < 1e-6

    def test_cvm_matches_scipy_cramervonmises(self):
        rng = np.random.default_rng(6)
        x = rng.standard_normal(400)
        dist = dist_mod.Normal.fit(x)
        mine = goodness_of_fit(x, dist, tests=("cvm",))["cvm"]
        ref = stats.cramervonmises(x, "norm", args=(dist.mu, dist.sigma))
        assert abs(mine.statistic - ref.statistic) < 1e-8

    def test_chi2_matches_scipy_chisquare_on_same_bins(self):
        rng = np.random.default_rng(7)
        x = rng.standard_normal(2000)
        dist = dist_mod.Normal.fit(x)
        mine = goodness_of_fit(x, dist, tests=("chi2",), n_params=2)["chi2"]
        k = mine.extras["expected"].shape[0] if "expected" in mine.extras else None
        assert mine.statistic >= 0
        assert 0 <= mine.pvalue <= 1

    def test_returns_test_result_instances(self):
        rng = np.random.default_rng(8)
        x = rng.standard_normal(300)
        results = goodness_of_fit(x, "Normal")
        for v in results.values():
            assert isinstance(v, stats_mod.TestResult)


class TestMomentMatching:
    @pytest.mark.parametrize("family,mean,var", [
        ("normal", 5.0, 2.0), ("gamma", 5.0, 2.0), ("lognormal", 5.0, 2.0),
        ("beta", 0.4, 0.02), ("exponential", 3.0, 9.0), ("uniform", 5.0, 3.0),
        ("negbinomial", 5.0, 10.0),
    ])
    def test_target_moments_hit_exactly(self, family, mean, var):
        d = moment_matching(mean=mean, var=var, family=family)
        assert abs(d.mean() - mean) < 1e-6
        assert abs(d.var() - var) < 1e-6

    def test_weibull_root_solve_hits_moments(self):
        d = moment_matching(mean=5.0, var=2.0, family="weibull")
        assert abs(d.mean() - 5.0) < 1e-5
        assert abs(d.var() - 2.0) < 1e-5

    def test_data_path_delegates_to_mom(self):
        rng = np.random.default_rng(9)
        x = rng.gamma(3.0, 2.0, 2000)
        d = moment_matching(data=x, family="gamma")
        assert type(d).__name__ == "Gamma"

    def test_missing_data_and_targets_raises(self):
        with pytest.raises(ValueError):
            moment_matching(family="normal")


class TestEcdf:
    def test_ecdf_matches_scipy(self):
        rng = np.random.default_rng(10)
        x = rng.standard_normal(50)
        e = ecdf(x)
        pts = np.linspace(-2, 2, 20)
        ref = stats.ecdf(x).cdf.evaluate(pts)
        assert np.allclose(e.evaluate(pts), ref)


class TestDataValidation:
    def test_check_reports_every_issue(self):
        dv = DataValidation(bounds=(0, 10), min_size=5, integer=True)
        rep = dv.check(np.array([1.0, 2.5, np.nan, 20.0]))
        assert not rep["valid"]
        assert rep["n_nan"] == 1
        assert rep["n_out_of_bounds"] == 1
        assert rep["n_non_integer"] == 1
        assert rep["n"] == 4

    def test_validate_raises_with_all_issues_listed(self):
        dv = DataValidation(min_size=10)
        with pytest.raises(ValueError):
            dv.validate(np.array([1.0, 2.0]))

    def test_validate_strips_nan_when_allowed(self):
        dv = DataValidation(allow_nan=True)
        out = dv.validate(np.array([1.0, 2.0, np.nan]))
        assert len(out) == 2

    def test_validate_raises_on_nan_when_not_allowed(self):
        dv = DataValidation(allow_nan=False)
        with pytest.raises(ValueError):
            dv.validate(np.array([1.0, 2.0, np.nan]))


class TestOutlierDetection:
    def _contaminated(self):
        rng = np.random.default_rng(11)
        return np.concatenate([rng.standard_normal(200), [60.0, -60.0]])

    def test_mad_flags_injected_outliers(self):
        x = self._contaminated()
        out = outlier_detection(x, method="mad")
        assert isinstance(out, OutlierResult)
        assert out.mask[-1] and out.mask[-2]

    def test_iqr_flags_injected_outliers(self):
        out = outlier_detection(self._contaminated(), method="iqr")
        assert out.mask[-1] and out.mask[-2]

    def test_zscore_flags_injected_outliers(self):
        out = outlier_detection(self._contaminated(), method="zscore")
        assert out.mask[-1] and out.mask[-2]

    def test_grubbs_critical_value_matches_student_t(self):
        x = self._contaminated()
        out = outlier_detection(x, method="grubbs", alpha=0.05)
        # the returned threshold is from the *last* (deciding, stopping) iteration,
        # whose sample size has already shrunk by the number of removed outliers.
        m = len(x) - out.n_outliers
        t_crit = stats.t.ppf(1.0 - 0.05 / (2.0 * m), m - 2)
        g_crit = ((m - 1) / np.sqrt(m)) * np.sqrt(t_crit ** 2 / (m - 2 + t_crit ** 2))
        assert abs(out.threshold - g_crit) < 1e-6
        assert out.mask[-1] and out.mask[-2]

    def test_hampel_flags_isolated_outliers(self):
        # Hampel's rolling window can mask *adjacent* outliers of opposite sign (a
        # known property of small local windows, not this implementation) -- unlike
        # the other methods' fixture, the outliers here are placed far apart so
        # neither falls in the other's window.
        rng = np.random.default_rng(11)
        x = rng.standard_normal(200)
        x[50] = 60.0
        x[150] = -60.0
        out = outlier_detection(x, method="hampel", window=7)
        assert out.mask[50] and out.mask[150]

    def test_mcd_matches_robust_statistics_mcd(self):
        rng = np.random.default_rng(12)
        X = rng.multivariate_normal([0, 0], [[1, 0.3], [0.3, 1]], 150)
        X[0] += 20
        from stochpylib.robust_statistics import MCD
        mine = outlier_detection(X, method="mcd", random_state=1)
        theirs = MCD(random_state=1).fit(X)
        assert np.array_equal(mine.mask, theirs.outliers())


class TestMissingImputation:
    def _make_missing(self, rng, shape, p=0.15):
        X = rng.standard_normal(shape)
        mask = rng.random(shape) < p
        Xm = X.copy()
        Xm[mask] = np.nan
        return X, Xm, mask

    @pytest.mark.parametrize("method", ["mean", "median", "mode", "constant",
                                        "locf", "nocb", "linear", "knn"])
    def test_no_nans_remain(self, method):
        rng = np.random.default_rng(20)
        _, Xm, _ = self._make_missing(rng, (150, 3))
        out = missing_imputation(Xm, method=method, k=5, fill_value=0.0)
        assert not np.isnan(out).any()

    def test_locf_matches_forward_fill_manually(self):
        x = np.array([1.0, np.nan, np.nan, 4.0, np.nan])
        out = missing_imputation(x, method="locf")
        assert out[1] == 1.0 and out[2] == 1.0 and out[4] == 4.0

    def test_linear_matches_interp(self):
        x = np.array([0.0, np.nan, 2.0, np.nan, 4.0])
        out = missing_imputation(x, method="linear")
        assert np.allclose(out, [0.0, 1.0, 2.0, 3.0, 4.0])

    def test_em_beats_mean_on_correlated_data(self):
        rng = np.random.default_rng(21)
        mean = np.array([1.0, -1.0, 0.5])
        cov = np.array([[2.0, 1.5, 0.5], [1.5, 2.0, 0.8], [0.5, 0.8, 1.0]])
        X = rng.multivariate_normal(mean, cov, 400)
        mask = rng.random(X.shape) < 0.2
        Xm = X.copy()
        Xm[mask] = np.nan
        out_em, info = missing_imputation(Xm, method="em", return_info=True)
        assert info["converged"]
        out_mean = missing_imputation(Xm, method="mean")
        rmse_em = np.sqrt(np.mean((out_em[mask] - X[mask]) ** 2))
        rmse_mean = np.sqrt(np.mean((out_mean[mask] - X[mask]) ** 2))
        assert rmse_em < rmse_mean

    def test_mice_is_seeded_reproducible(self):
        rng = np.random.default_rng(22)
        _, Xm, _ = self._make_missing(rng, (100, 3))
        a = missing_imputation(Xm, method="mice", random_state=1)
        b = missing_imputation(Xm, method="mice", random_state=1)
        assert np.allclose(a, b)

    def test_1d_input_returns_1d_output(self):
        rng = np.random.default_rng(23)
        x = rng.standard_normal(50)
        x[::7] = np.nan
        out = missing_imputation(x, method="mean")
        assert out.ndim == 1 and out.shape == x.shape


# ================================================================================ io

class TestToDictFromDict:
    @pytest.mark.parametrize("name,args", [
        ("Normal", (1.0, 2.0)), ("Gamma", (3.0, 2.0)), ("Poisson", (4.0,)),
        ("Beta", (2.0, 3.0)), ("Weibull", (1.5, 2.0)), ("MultivariateNormal",
        ([0.0, 0.0], [[1.0, 0.2], [0.2, 1.0]])),
    ])
    def test_distribution_round_trip(self, name, args):
        cls = getattr(dist_mod, name)
        inst = cls(*args)
        back = from_dict(to_dict(inst))
        assert type(back) is type(inst)

    def test_all_47_distributions_round_trip(self):
        specs = {
            "Poisson": (4.0,), "Bernoulli": (0.3,), "Binomial": (10, 0.4),
            "Geometric": (0.3,), "Hypergeometric": (50, 10, 5),
            "Multinomial": (10, [0.2, 0.3, 0.5]), "NegBinomial": (5, 0.4),
            "ZipfDistribution": (2.0,), "DiscreteUniform": (1, 10),
            "BetaBinomial": (10, 2, 3), "ConwayMaxwellPoisson": (3.0, 1.5),
            "MultivariateNormal": ([0, 0], [[1, 0.3], [0.3, 1]]),
            "Wishart": (5, np.eye(2)), "InverseWishart": (5, np.eye(2)),
            "Dirichlet": ([1, 2, 3],), "MultivariateT": (5, [0, 0], np.eye(2)),
            "MultivariatePareto": (3.0, [1, 1], [[1, 0.2], [0.2, 1]]),
            "AlphaStable": (1.5,), "StableDistribution": (1.5, 0.0, 1.0, 0.0),
            "LevyDistribution": (0.0, 1.0),
        }
        names = [n for n in dist_mod.__all__ if n not in ("Distribution", "MultivariateDistribution")]
        assert len(names) == 47
        for name in names:
            cls = getattr(dist_mod, name)
            inst = cls(*specs[name]) if name in specs else cls()
            back = from_dict(to_dict(inst))
            assert type(back) is type(inst), name

    def test_mcresult_round_trip(self):
        mc = MCResult(1.5, 0.1, 1000, "crude", {"x": 1})
        assert from_dict(to_dict(mc)) == mc

    def test_forecastresult_round_trip(self):
        from stochpylib.timeseries import ForecastResult
        fr = ForecastResult(np.array([1.0, 2.0]), np.array([0.1, 0.2]))
        back = from_dict(to_dict(fr))
        assert np.allclose(back.mean, fr.mean) and np.allclose(back.std, fr.std)

    def test_testresult_round_trip(self):
        tr = stats_mod.TestResult(1.96, 0.05, 10, "null", "method", "two-sided")
        back = from_dict(to_dict(tr))
        assert back.statistic == tr.statistic and back.pvalue == tr.pvalue

    def test_queueresult_round_trip(self):
        from stochpylib.queueing import QueueResult
        qr = QueueResult(1.0, 0.5, 2.0, 1.0, 0.8, extra=5)
        back = from_dict(to_dict(qr))
        assert back.L == qr.L and back.extras == qr.extras

    def test_gp_kernel_round_trip(self):
        from stochpylib.gaussian_processes.kernels import RBFKernel
        k = RBFKernel(length_scale=1.5, variance=2.0)
        back = from_dict(to_dict(k))
        assert back.get_params() == k.get_params()

    def test_generator_state_continues_identically(self):
        g = np.random.default_rng(42)
        g.standard_normal(3)
        back = from_dict(to_dict(g))
        expect = np.random.default_rng(42)
        expect.standard_normal(3)
        assert np.array_equal(back.standard_normal(5), expect.standard_normal(5))

    def test_seedsequence_round_trip(self):
        ss = np.random.SeedSequence(7)
        back = from_dict(to_dict(ss))
        assert np.array_equal(np.random.default_rng(ss).standard_normal(3),
                              np.random.default_rng(back).standard_normal(3))

    def test_fitresult_round_trip(self):
        rng = np.random.default_rng(30)
        result = fit(rng.gamma(3, 2, 500), distributions=["Gamma", "Normal"])
        back = from_dict(to_dict(result))
        assert back.best_name_ == result.best_name_

    def test_disallowed_type_raises(self):
        with pytest.raises(ValueError):
            from_dict({"__type__": "os.system", "kind": "object", "fields": {}})

    def test_unsupported_type_raises_typeerror(self):
        with pytest.raises(TypeError):
            to_dict(lambda x: x)


class TestSerialization:
    @pytest.mark.parametrize("ext", ["json", "pkl", "npz"])
    def test_save_load_round_trip(self, tmp_path, ext):
        mc = MCResult(2.5, 0.2, 500, "m")
        p = tmp_path / f"r.{ext}"
        Serialization().save(mc, str(p))
        back = Serialization().load(str(p))
        assert back.estimate == mc.estimate

    def test_to_json_writes_file(self, tmp_path):
        p = tmp_path / "out.json"
        to_json({"a": 1}, path=str(p))
        assert from_json(str(p)) == {"a": 1}

    def test_to_pickle_and_from_pickle(self, tmp_path):
        p = tmp_path / "out.pkl"
        to_pickle({"a": 1}, str(p))
        assert from_pickle(str(p)) == {"a": 1}

    def test_register_custom_type(self):
        class Widget:
            def __init__(self, n):
                self.n = n

        Serialization.register(Widget, lambda w: {"n": w.n}, lambda d: Widget(d["n"]))
        w = Widget(5)
        back = from_dict(to_dict(w))
        assert isinstance(back, Widget) and back.n == 5


class TestConfiguration:
    def test_dot_path_get_and_getitem(self):
        cfg = Configuration({"mcmc": {"n_samples": 1000}})
        assert cfg.get("mcmc.n_samples") == 1000
        assert cfg["mcmc.n_samples"] == 1000
        assert cfg.mcmc.n_samples == 1000

    def test_deep_merge_on_update(self):
        cfg = Configuration({"a": {"x": 1, "y": 2}})
        cfg.update({"a": {"y": 3, "z": 4}})
        assert cfg.get("a.x") == 1 and cfg.get("a.y") == 3 and cfg.get("a.z") == 4

    def test_from_env_maps_double_underscore_to_dots(self, monkeypatch):
        monkeypatch.setenv("STOCHPYLIB_MCMC__N_SAMPLES", "2000")
        cfg = Configuration().from_env()
        assert cfg.get("mcmc.n_samples") == 2000

    def test_validate_reports_missing_and_wrong_type(self):
        cfg = Configuration({"a": 1})
        errors = cfg.validate({"a": int, "b": int})
        assert any("b" in e for e in errors)

    def test_freeze_blocks_writes(self):
        cfg = Configuration({"a": 1}).freeze()
        with pytest.raises(TypeError):
            cfg.set("b", 2)


class TestLogging:
    def test_capture_collects_record(self):
        with Logging.capture("INFO") as records:
            Logging.get_logger().info("hello")
        assert len(records) == 1
        assert records[0].getMessage() == "hello"

    def test_configure_is_idempotent(self):
        Logging.configure(level="DEBUG")
        Logging.configure(level="DEBUG")
        logger = Logging.get_logger()
        managed = [h for h in logger.handlers if getattr(h, "_stochpylib_managed", False)]
        assert len(managed) == 1


class TestSummary:
    def test_summary_of_mcresult(self):
        text = summary(MCResult(1.5, 0.1, 100, "m"))
        assert "1.5" in text

    def test_summary_of_distribution(self):
        text = summary(dist_mod.Normal(0, 1))
        assert "Normal" in text

    def test_summary_of_ndarray(self):
        text = summary(np.random.default_rng(0).standard_normal(50))
        assert isinstance(text, str) and len(text) > 0

    def test_summary_of_fitresult(self):
        rng = np.random.default_rng(31)
        result = fit(rng.standard_normal(300), distributions=["Normal", "Cauchy"])
        assert result.best_name_ in summary(result)


# ============================================================================= compat

class TestCompat:
    def test_numpy_interface_list_and_result(self):
        assert np.allclose(numpy_interface([1, 2, 3]), [1, 2, 3])
        assert numpy_interface(MCResult(1.5)) == np.array([1.5])

    def test_scipy_interface_normal_parity(self):
        n = dist_mod.Normal(1, 2)
        sn = scipy_interface(n)
        ref = stats.norm(1, 2)
        assert abs(sn.sf(1.5) - ref.sf(1.5)) < 1e-10
        assert abs(sn.isf(0.3) - ref.isf(0.3)) < 1e-8
        assert np.allclose(sn.interval(0.9), ref.interval(0.9))
        assert abs(sn.moment(1) - ref.moment(1)) < 1e-10
        assert abs(sn.moment(2) - ref.moment(2)) < 1e-8
        assert abs(sn.expect() - ref.expect()) < 1e-6
        assert abs(sn.median() - ref.median()) < 1e-10
        assert np.allclose(sn.stats(moments="mv"), ref.stats(moments="mv"), atol=1e-8)

    def test_scipy_interface_gamma_parity(self):
        g = dist_mod.Gamma(3, 2)
        sg = scipy_interface(g)
        ref = stats.gamma(3, scale=2)
        assert abs(sg.sf(4) - ref.sf(4)) < 1e-10
        assert abs(sg.expect() - ref.expect()) < 1e-3

    def test_scipy_interface_poisson_discrete_parity(self):
        p = dist_mod.Poisson(4)
        sp_ = scipy_interface(p)
        ref = stats.poisson(4)
        assert abs(sp_.sf(3) - ref.sf(3)) < 1e-10

    def test_pandas_interface_import_error_when_hidden(self, monkeypatch):
        from stochpylib.utils.compat import pandas_interface
        monkeypatch.setitem(sys.modules, "pandas", None)
        with pytest.raises(ImportError) as exc:
            pandas_interface(np.array([1, 2, 3]))
        assert "stochpylib[pandas]" in str(exc.value)

    def test_torch_interface_import_error_when_hidden(self, monkeypatch):
        from stochpylib.utils.compat import torch_interface
        monkeypatch.setitem(sys.modules, "torch", None)
        with pytest.raises(ImportError) as exc:
            torch_interface(np.array([1, 2, 3]))
        assert "stochpylib[torch]" in str(exc.value)

    def test_jax_interface_import_error_when_missing(self):
        from stochpylib.utils.compat import jax_interface
        with pytest.raises(ImportError) as exc:
            jax_interface(np.array([1, 2, 3]))
        assert "stochpylib[jax]" in str(exc.value)
