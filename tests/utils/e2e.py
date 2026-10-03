"""End-to-end exercise sweep: one realistic use per public name in
``stochpylib.utils.__all__``. Kept fast (small n, cheap constructors) -- exhaustive
oracle checks live in ``tests.py``. Every exercise builds a real object from another
stochpylib module where relevant, never mock data.
"""

import json
import tempfile
from pathlib import Path

import numpy as np
import pytest

from stochpylib import utils as mod

EXERCISES = {}


def exercise(name):
    def deco(fn):
        EXERCISES[name] = fn
        return fn
    return deco


_RNG = np.random.default_rng(0)


@exercise("set_seed")
def _set_seed():
    mod.set_seed(7)
    a = mod.random_state(None).standard_normal(3)
    mod.set_seed(7)
    b = mod.random_state(None).standard_normal(3)
    mod.set_seed(None)
    assert np.array_equal(a, b)


@exercise("random_state")
def _random_state():
    g = mod.random_state(5)
    assert isinstance(g, np.random.Generator)


@exercise("Generator")
def _generator():
    assert mod.Generator is np.random.Generator


@exercise("SeedSequence")
def _seed_sequence():
    assert mod.SeedSequence is np.random.SeedSequence


@exercise("spawn_generator")
def _spawn_generator():
    kids = mod.spawn_generator(random_state=1, n=3)
    assert len(kids) == 3


@exercise("Benchmark")
def _benchmark():
    b = mod.Benchmark(repeat=3, number=50).run(lambda: sum(range(100)))
    assert b.mean_ >= 0


@exercise("Profiler")
def _profiler():
    p = mod.Profiler()
    p.profile(lambda: sum(range(2000)))
    assert p.total_time_ >= 0


@exercise("ParallelSimulation")
def _parallel_simulation():
    def sim(n, rng):
        from stochpylib.montecarlo import MCResult
        x = rng.standard_normal(n)
        return MCResult(float(x.mean()), float(x.std(ddof=1) / np.sqrt(n)), n, "e2e")

    res = mod.ParallelSimulation(n_jobs=2, chunk_size=500).estimate(sim, 2000, random_state=1)
    assert isinstance(res, object) and hasattr(res, "estimate")


@exercise("GPUBackend")
def _gpu_backend():
    gb = mod.GPUBackend("numpy")
    paths = gb.simulate_gbm(100.0, 0.05, 0.2, 1.0, 20, 200, random_state=1)
    assert paths.shape == (200, 21)


@exercise("JIT_compile")
def _jit_compile():
    @mod.JIT_compile(backend="none")
    def f(x):
        return x * 2

    assert f(3) == 6


@exercise("VectorizedOps")
def _vectorized_ops():
    a = _RNG.standard_normal(10)
    assert np.isfinite(mod.VectorizedOps.logsumexp(a))


@exercise("MemoryPool")
def _memory_pool():
    pool = mod.MemoryPool()
    arr = pool.get((10, 10))
    pool.release(arr)
    assert pool.hits_ + pool.misses_ >= 1


@exercise("Reproducibility")
def _reproducibility():
    with mod.Reproducibility(seed=3):
        pass


@exercise("RandomStream")
def _random_stream():
    rs = mod.RandomStream(seed=2)
    assert rs.generator.standard_normal(1).shape == (1,)


@exercise("VersionLock")
def _version_lock():
    vl = mod.VersionLock().capture()
    assert vl.check() == []


@exercise("EnvironmentCapture")
def _environment_capture():
    ec = mod.EnvironmentCapture().capture()
    assert "numpy_version" in ec.to_dict()


@exercise("ExperimentLogger")
def _experiment_logger():
    logger = mod.ExperimentLogger()
    with logger.run(params={"a": 1}) as run:
        run.log(loss=0.1)
    assert len(logger.runs_) == 1


@exercise("fit")
def _fit():
    from stochpylib.distributions import Gamma
    data = Gamma(3.0, 2.0).rvs(1000, random_state=0)
    result = mod.fit(data)
    assert result.best_name_ == "Gamma"


@exercise("goodness_of_fit")
def _goodness_of_fit():
    data = _RNG.standard_normal(300)
    res = mod.goodness_of_fit(data, "Normal")
    assert "ks" in res


@exercise("moment_matching")
def _moment_matching():
    d = mod.moment_matching(mean=5.0, var=2.0, family="gamma")
    assert abs(d.mean() - 5.0) < 1e-6


@exercise("ecdf")
def _ecdf():
    e = mod.ecdf(_RNG.standard_normal(100))
    assert 0.0 <= e.evaluate(0.0) <= 1.0


@exercise("DataValidation")
def _data_validation():
    dv = mod.DataValidation(min_size=1)
    rep = dv.check(np.array([1.0, 2.0]))
    assert rep["valid"]


@exercise("outlier_detection")
def _outlier_detection():
    data = np.concatenate([_RNG.standard_normal(100), [50.0]])
    out = mod.outlier_detection(data, method="mad")
    assert out.mask[-1]


@exercise("missing_imputation")
def _missing_imputation():
    x = _RNG.standard_normal(50)
    x[::5] = np.nan
    out = mod.missing_imputation(x, method="mean")
    assert not np.isnan(out).any()


@exercise("to_dict")
def _to_dict():
    from stochpylib.distributions import Normal
    d = mod.to_dict(Normal(0, 1))
    assert d["kind"] == "distribution"


@exercise("from_dict")
def _from_dict():
    from stochpylib.distributions import Normal
    back = mod.from_dict(mod.to_dict(Normal(0, 1)))
    assert type(back).__name__ == "Normal"


@exercise("to_json")
def _to_json():
    text = mod.to_json({"a": 1})
    assert json.loads(text) == {"a": 1}


@exercise("from_json")
def _from_json():
    assert mod.from_json('{"a": 1}') == {"a": 1}


@exercise("to_pickle")
def _to_pickle():
    with tempfile.TemporaryDirectory() as td:
        p = Path(td) / "x.pkl"
        mod.to_pickle({"a": 1}, str(p))
        assert p.exists()


@exercise("from_pickle")
def _from_pickle():
    with tempfile.TemporaryDirectory() as td:
        p = Path(td) / "x.pkl"
        mod.to_pickle({"a": 1}, str(p))
        assert mod.from_pickle(str(p)) == {"a": 1}


@exercise("Serialization")
def _serialization():
    with tempfile.TemporaryDirectory() as td:
        p = Path(td) / "x.json"
        mod.Serialization().save({"a": 1}, str(p))
        assert mod.Serialization().load(str(p)) == {"a": 1}


@exercise("Configuration")
def _configuration():
    cfg = mod.Configuration({"a": {"b": 1}})
    assert cfg.get("a.b") == 1


@exercise("Logging")
def _logging():
    with mod.Logging.capture("INFO") as records:
        mod.Logging.get_logger().info("hi")
    assert len(records) == 1


@exercise("summary")
def _summary():
    from stochpylib.montecarlo import MCResult
    text = mod.summary(MCResult(1.0, 0.1, 10, "m"))
    assert isinstance(text, str)


@exercise("FitResult")
def _fit_result():
    data = _RNG.standard_normal(300)
    result = mod.fit(data, distributions=["Normal", "Cauchy"])
    assert isinstance(result, mod.FitResult)


@exercise("OutlierResult")
def _outlier_result():
    data = np.concatenate([_RNG.standard_normal(100), [80.0]])
    out = mod.outlier_detection(data, method="iqr")
    assert isinstance(out, mod.OutlierResult)


@exercise("numpy_interface")
def _numpy_interface():
    assert np.allclose(mod.numpy_interface([1, 2, 3]), [1, 2, 3])


@exercise("scipy_interface")
def _scipy_interface():
    from stochpylib.distributions import Normal
    view = mod.scipy_interface(Normal(0, 1))
    assert abs(view.mean() - 0.0) < 1e-10


@exercise("pandas_interface")
def _pandas_interface():
    from stochpylib.utils._backends import is_installed
    if is_installed("pandas"):
        df = mod.pandas_interface(_RNG.standard_normal((5, 3)))
        assert df.shape == (5, 3)
    else:
        with pytest.raises(ImportError):
            mod.pandas_interface(_RNG.standard_normal((5, 3)))


@exercise("torch_interface")
def _torch_interface():
    from stochpylib.distributions import Normal
    from stochpylib.utils._backends import is_installed
    if is_installed("torch"):
        t = mod.torch_interface(Normal(0, 1))
        assert hasattr(t, "log_prob")
    else:
        with pytest.raises(ImportError):
            mod.torch_interface(Normal(0, 1))


@exercise("jax_interface")
def _jax_interface():
    from stochpylib.utils._backends import is_installed
    if is_installed("jax"):
        arr = mod.jax_interface(np.array([1.0, 2.0, 3.0]))
        assert np.allclose(np.asarray(arr), [1.0, 2.0, 3.0])
    else:
        with pytest.raises(ImportError):
            mod.jax_interface(np.array([1.0, 2.0, 3.0]))


def test_every_public_name_is_exercised():
    assert set(EXERCISES) == set(mod.__all__), (
        set(mod.__all__) - set(EXERCISES), set(EXERCISES) - set(mod.__all__))


@pytest.mark.parametrize("name", sorted(mod.__all__))
def test_exercise(name):
    EXERCISES[name]()
