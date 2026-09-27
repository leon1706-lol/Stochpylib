"""Real-backend tests for torch/jax/numba/pandas -- NOT auto-collected by the main
suite (``pyproject.toml``'s ``python_files = ["tests.py", "e2e.py"]`` never matches
this filename, the same mechanism that keeps ``tests/viz/backend_mpl.py`` out). Run
explicitly by the ``utils-optional`` CI job, after installing the real backends.
Everything in ``tests.py``/``e2e.py`` already passes with none of them installed --
this file only adds coverage for the real libraries themselves.

Each backend uses its own ``pytest.importorskip`` so this file can also be run
locally with whatever subset happens to be installed. The
``STOCHPYLIB_REQUIRE_BACKENDS=1`` env var (set by the CI job) makes a missing
backend fail loudly instead of silently skipping, so CI can never quietly lose
coverage.
"""

import os

import numpy as np
import pytest

from stochpylib.distributions import Gamma, Normal, Poisson
from stochpylib.utils import GPUBackend, JIT_compile, pandas_interface, torch_interface
from stochpylib.utils.compat import jax_interface
from stochpylib.utils._backends import available_backends

_REQUIRE = os.environ.get("STOCHPYLIB_REQUIRE_BACKENDS") == "1"


def _require_or_skip(name):
    """Unlike ``is_installed`` (a cheap ``find_spec`` check used by the library's own
    lazy-import guard), this actually imports the package -- a package can have an
    import spec but still be unusable (e.g. numba against a newer numpy than it
    supports), and this file exists specifically to test the real, working backend."""
    if available_backends().get(name):
        return
    if _REQUIRE:
        pytest.fail(f"{name} is required (STOCHPYLIB_REQUIRE_BACKENDS=1) but not usable")
    pytest.skip(f"{name} not installed/usable")


class TestTorchBackend:
    def setup_method(self):
        _require_or_skip("torch")

    def test_distribution_log_prob_matches_library_logpdf(self):
        import torch
        n = Normal(1.0, 2.0)
        td = torch_interface(n)
        x = torch.tensor(1.7, dtype=torch.float64)
        lp = float(td.log_prob(x))
        ref = float(np.log(n.pdf(1.7)))
        assert abs(lp - ref) < 1e-8

    def test_autodiff_grad_matches_analytic_gradient(self):
        import torch

        def logp(theta):
            return -0.5 * torch.sum(theta ** 2)

        ad = torch_interface(logp)
        x = np.array([1.0, -2.0, 0.5])
        assert np.allclose(ad.grad(x), -x, atol=1e-8)
        assert abs(ad(x) - (-0.5 * np.sum(x ** 2))) < 1e-8

    def test_hmc_driven_by_torch_autodiff_gradient(self):
        import torch
        from stochpylib.advanced_mcmc import ESS, HamiltonianMonteCarlo

        def logp_np(theta):
            return -0.5 * np.sum(theta ** 2)

        def logp_torch(theta):
            return -0.5 * torch.sum(theta ** 2)

        grad = torch_interface(logp_torch).grad
        sampler = HamiltonianMonteCarlo(logp_np, grad_log_prob=grad, n_samples=300,
                                        n_warmup=150, step_size=0.5, n_leapfrog=10)
        sampler.sample(np.zeros(2), random_state=1)
        samples = sampler.get_samples()
        # MCSE-based bound (this module's own ESS), not a fixed magic number -- the
        # leapfrog trajectory is chaotic, so a hardcoded threshold flakes across
        # platforms even at a fixed random_state (torch's float rounding differs).
        for j in range(samples.shape[1]):
            col = samples[:, j]
            ess = ESS(col[None, :, None])
            se = col.std(ddof=1) / np.sqrt(max(ess, 1.0))
            assert abs(col.mean()) < 5 * se

    def test_gpu_backend_torch_gbm_matches_numpy(self):
        gb_np = GPUBackend("numpy")
        gb_torch = GPUBackend("torch")
        a = gb_np.simulate_gbm(100.0, 0.05, 0.2, 1.0, 50, 500, random_state=3)
        b = gb_torch.simulate_gbm(100.0, 0.05, 0.2, 1.0, 50, 500, random_state=3)
        assert np.allclose(a, b, atol=1e-6)


class TestJaxBackend:
    def setup_method(self):
        _require_or_skip("jax")

    def test_distribution_log_prob_matches_library_logpdf(self):
        n = Normal(0.0, 1.0)
        shim = jax_interface(n)
        lp = float(shim.log_prob(0.5))
        ref = float(np.log(n.pdf(0.5)))
        assert abs(lp - ref) < 1e-6

    def test_autodiff_grad_matches_analytic_gradient(self):
        import jax.numpy as jnp

        def logp(theta):
            return -0.5 * jnp.sum(theta ** 2)

        from stochpylib.utils._backends import JaxAutodiffLogDensity
        ad = JaxAutodiffLogDensity(logp)
        x = np.array([1.0, -2.0])
        assert np.allclose(ad.grad(x), -x, atol=1e-6)

    def test_jit_compile_backend_jax(self):
        import jax.numpy as jnp

        @JIT_compile(backend="jax")
        def f(x):
            return x * 2.0

        out = f(jnp.array(3.0))
        assert float(out) == 6.0
        assert f.backend_ == "jax"

    def test_gpu_backend_jax_gbm_matches_numpy(self):
        gb_np = GPUBackend("numpy")
        gb_jax = GPUBackend("jax")
        a = gb_np.simulate_gbm(100.0, 0.05, 0.2, 1.0, 50, 500, random_state=4)
        b = gb_jax.simulate_gbm(100.0, 0.05, 0.2, 1.0, 50, 500, random_state=4)
        assert np.allclose(a, b, atol=1e-5)


class TestNumbaBackend:
    def setup_method(self):
        _require_or_skip("numba")

    def test_jit_compile_uses_numba_and_matches_python(self):
        def f(x):
            total = 0.0
            for i in range(x.shape[0]):
                total += x[i] * x[i]
            return total

        compiled = JIT_compile(f)
        x = np.arange(10.0)
        result = compiled(x)
        assert compiled.backend_ == "numba"
        assert abs(result - f(x)) < 1e-10


class TestPandasBackend:
    def setup_method(self):
        _require_or_skip("pandas")

    def test_pandas_interface_round_trips_mcresult(self):
        from stochpylib.montecarlo import MCResult
        s = pandas_interface(MCResult(1.5, 0.1, 100, "m"))
        assert s["estimate"] == 1.5


class TestEnvironmentCaptureListsInstalledBackends:
    def test_all_four_available(self):
        for name in ("pandas", "torch", "jax", "numba"):
            _require_or_skip(name)
        from stochpylib.utils import EnvironmentCapture
        avail = EnvironmentCapture().capture().to_dict()["available_backends"]
        for name in ("pandas", "torch", "jax", "numba"):
            assert avail[name] is True
