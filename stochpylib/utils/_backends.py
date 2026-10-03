"""The ONLY file in stochpylib allowed to import pandas / torch / jax / jaxlib /
numba / cupy -- and only lazily, inside function bodies, never at module import
time. ``tests/library/tests.py`` enforces both rules with an AST walk.

``utils.performance``/``utils.compat`` call into this module for anything that
actually needs one of these packages; ``stochpylib.utils`` itself, and every
other module in the library, stays importable with none of them installed.
"""

import functools
import importlib
import importlib.util

__all__ = []

_EXTRAS = {"pandas": "pandas", "torch": "torch", "jax": "jax", "numba": "numba",
           "cupy": "gpu"}


def is_installed(name):
    """Whether ``name`` has an importable spec, without importing it."""
    return importlib.util.find_spec(name) is not None


def import_backend(name):
    """Import ``name`` (e.g. ``"torch"``), or raise a clear ``ImportError`` naming
    the extra to install. Catches *any* import-time failure, including a package
    that is installed but broken for the running environment (e.g. numba raising
    ``Numba needs NumPy 2.3 or less`` against a newer numpy)."""
    extra = _EXTRAS.get(name, name)
    try:
        return importlib.import_module(name)
    except ImportError as exc:
        raise ImportError(
            f"stochpylib.utils: {name} is optional and not usable ({exc}); "
            f"pip install 'stochpylib[{extra}]'") from exc


@functools.lru_cache(maxsize=1)
def available_backends():
    """``{name: bool}`` from a real import attempt (cached per process)."""
    out = {}
    for name in _EXTRAS:
        try:
            importlib.import_module(name)
            out[name] = True
        except ImportError:
            out[name] = False
    return out


# name -> (torch class name, {torch kwarg: fn(dist) -> python float/int})
_TORCH_DIST_MAP = {
    "Normal": ("Normal", {"loc": lambda d: d.mu, "scale": lambda d: d.sigma}),
    "Exponential": ("Exponential", {"rate": lambda d: d.rate}),
    "Gamma": ("Gamma", {"concentration": lambda d: d.shape, "rate": lambda d: 1.0 / d.scale}),
    "Beta": ("Beta", {"concentration1": lambda d: d.a, "concentration0": lambda d: d.b}),
    "Uniform": ("Uniform", {"low": lambda d: d.a, "high": lambda d: d.b}),
    "LogNormal": ("LogNormal", {"loc": lambda d: d.mu, "scale": lambda d: d.sigma}),
    "Student_t": ("StudentT", {"df": lambda d: d.df}),
    "Laplace": ("Laplace", {"loc": lambda d: d.loc, "scale": lambda d: d.scale}),
    "Cauchy": ("Cauchy", {"loc": lambda d: d.loc, "scale": lambda d: d.scale}),
    "Poisson": ("Poisson", {"rate": lambda d: d.lam}),
    "Bernoulli": ("Bernoulli", {"probs": lambda d: d.p}),
    "Binomial": ("Binomial", {"total_count": lambda d: d.n, "probs": lambda d: d.p}),
}


def torch_distribution(dist, device, dtype):
    """A native ``torch.distributions`` object for a stochpylib ``Distribution``."""
    torch = import_backend("torch")
    name = type(dist).__name__
    if name not in _TORCH_DIST_MAP:
        raise TypeError(
            f"torch_interface: no torch.distributions mapping for {name!r}; "
            f"supported: {sorted(_TORCH_DIST_MAP)}")
    torch_name, kwarg_fns = _TORCH_DIST_MAP[name]
    cls = getattr(torch.distributions, torch_name)
    kwargs = {k: torch.tensor(float(fn(dist)), device=device, dtype=dtype)
              for k, fn in kwarg_fns.items()}
    return cls(**kwargs)


class TorchAutodiffLogDensity:
    """Wraps a torch-written log-density ``fn(tensor) -> scalar tensor`` as a plain
    numpy callable with an exact ``.grad(x)`` via autograd -- usable directly as
    ``advanced_mcmc``'s ``grad_log_prob=``."""

    def __init__(self, fn, device="cpu", dtype="float64"):
        torch = import_backend("torch")
        self._torch = torch
        self._fn = fn
        self._device = device
        self._dtype = getattr(torch, dtype) if isinstance(dtype, str) else dtype

    def __call__(self, x):
        torch = self._torch
        t = torch.as_tensor(x, device=self._device, dtype=self._dtype)
        with torch.no_grad():
            return float(self._fn(t))

    def grad(self, x):
        torch = self._torch
        t = torch.as_tensor(x, device=self._device, dtype=self._dtype).clone()
        t.requires_grad_(True)
        val = self._fn(t)
        val.backward()
        return t.grad.detach().cpu().numpy().astype(float)


def jax_enable_x64():
    """Enable jax float64 globally (jax defaults to float32). Documented global
    side effect -- called once from every ``jax_interface``/``JIT_compile`` entry
    point that touches jax."""
    jax = import_backend("jax")
    jax.config.update("jax_enable_x64", True)
    return jax


def jax_numpy():
    """``jax.numpy``, imported here so no other file needs a raw ``jax`` import."""
    jax_enable_x64()
    import jax.numpy as jnp
    return jnp


def jax_scipy_stats():
    """``jax.scipy.stats`` (jax's own reimplementation, not ``scipy.stats``)."""
    jax_enable_x64()
    import jax.scipy.stats as jsp
    return jsp


def jax_key(random_state):
    """A ``jax.random`` PRNGKey derived from a stochpylib ``random_state=``."""
    jax = jax_enable_x64()
    from stochpylib import _rng

    seed = int(_rng.as_generator(random_state).integers(0, 2 ** 31 - 1))
    return jax.random.PRNGKey(seed)


class JaxAutodiffLogDensity:
    """Same contract as :class:`TorchAutodiffLogDensity`, backed by ``jax.grad``."""

    def __init__(self, fn):
        jax = jax_enable_x64()
        import jax.numpy as jnp

        self._jax = jax
        self._jnp = jnp
        self._fn = fn
        self._grad_fn = jax.grad(fn)

    def __call__(self, x):
        return float(self._fn(self._jnp.asarray(x)))

    def grad(self, x):
        import numpy as np

        return np.asarray(self._grad_fn(self._jnp.asarray(x)), dtype=float)


def cupy_has_device():
    """Whether cupy is installed *and* sees a CUDA device."""
    if not is_installed("cupy"):
        return False
    try:
        cupy = importlib.import_module("cupy")
        return cupy.cuda.runtime.getDeviceCount() > 0
    except Exception:
        return False
