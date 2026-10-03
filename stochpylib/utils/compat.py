"""Interop with numpy / a scipy.stats-shaped frozen-distribution view / pandas /
torch / jax. Only ``numpy_interface`` needs no import at all (pure duck-typing);
everything else routes optional imports through ``utils._backends``.

``scipy_interface`` is a one-directional *view* -- library code never wraps
``scipy.stats`` (AGENTS.md), so this only ever reads a stochpylib ``Distribution``
through scipy's method names, never the reverse.
"""

import numpy as np

from stochpylib.utils import _backends

__all__ = ["jax_interface", "numpy_interface", "pandas_interface", "scipy_interface",
           "torch_interface"]


def numpy_interface(obj):
    """Best-effort conversion of ``obj`` (torch/jax/cupy array, pandas object,
    an MCResult/ForecastResult, or any array-like) to a numpy array. Duck-types
    on the object's originating package, so it needs no import at all."""
    top = type(obj).__module__.split(".")[0]
    if top == "torch":
        return obj.detach().cpu().numpy()
    if top in ("jax", "jaxlib"):
        return np.asarray(obj)
    if top == "cupy":
        return obj.get()
    if top == "pandas":
        return obj.to_numpy()

    from stochpylib.montecarlo import MCResult
    if isinstance(obj, MCResult):
        return np.array([obj.estimate])
    from stochpylib.timeseries import ForecastResult
    if isinstance(obj, ForecastResult):
        return np.asarray(obj.mean)

    return np.asarray(obj)


class _ScipyFrozenAdapter:
    """Exposes a stochpylib ``Distribution`` through the ``scipy.stats`` frozen-
    distribution method names, built entirely from the distribution's own
    ``pdf/pmf/cdf/ppf/rvs/mean/var/skewness/kurtosis/entropy/support`` -- no
    import of ``scipy.stats`` anywhere in this class."""

    def __init__(self, dist):
        self._d = dist
        self.is_discrete = getattr(dist, "is_discrete", False)

    def pdf(self, x):
        return self._d.pmf(x) if self.is_discrete else self._d.pdf(x)

    pmf = pdf

    def logpdf(self, x):
        return np.log(np.asarray(self.pdf(x), dtype=float))

    logpmf = logpdf

    def cdf(self, x):
        return self._d.cdf(x)

    def sf(self, x):
        return 1.0 - np.asarray(self._d.cdf(x), dtype=float)

    def ppf(self, q):
        return self._d.ppf(q)

    def isf(self, q):
        return self._d.ppf(1.0 - np.asarray(q, dtype=float))

    def rvs(self, size=1, random_state=None):
        return self._d.rvs(size, random_state=random_state)

    def mean(self):
        return self._d.mean()

    def var(self):
        return self._d.var()

    def std(self):
        return self._d.std() if hasattr(self._d, "std") else float(np.sqrt(self._d.var()))

    def median(self):
        return self._d.ppf(0.5)

    def interval(self, confidence):
        alpha = (1.0 - confidence) / 2.0
        return self._d.ppf(alpha), self._d.ppf(1.0 - alpha)

    def stats(self, moments="mv"):
        table = {"m": self.mean, "v": self.var, "s": self._d.skewness, "k": self._d.kurtosis}
        return tuple(table[m]() for m in moments)

    def entropy(self):
        return self._d.entropy()

    def support(self):
        return self._d.support()

    def moment(self, n):
        if n == 1:
            return self.mean()
        if n == 2:
            return self.var() + self.mean() ** 2
        raise NotImplementedError("moment() beyond n=2 is not implemented generically")

    def expect(self, func=None):
        from scipy import integrate
        f = (lambda x: x) if func is None else func
        if self.is_discrete:
            lo, hi = self.support()
            lo = max(lo, -1000)
            hi = min(hi, 1000)
            xs = np.arange(int(lo), int(hi) + 1)
            return float(np.sum([f(x) * self._d.pmf(x) for x in xs]))
        lo, hi = self.support()
        val, _ = integrate.quad(lambda x: f(x) * self._d.pdf(x), lo, hi, limit=200)
        return float(val)

    def __repr__(self):
        return f"_ScipyFrozenAdapter({self._d!r})"


def scipy_interface(dist):
    """A ``scipy.stats``-frozen-distribution-shaped read-only view over a
    stochpylib ``Distribution`` (never imports ``scipy.stats`` itself)."""
    return _ScipyFrozenAdapter(dist)


def pandas_interface(obj, index=None, name=None):
    """Convert ``obj`` to the natural pandas shape for its type."""
    pd = _backends.import_backend("pandas")

    if isinstance(obj, np.ndarray):
        if obj.ndim == 1:
            return pd.Series(obj, index=index, name=name)
        if obj.ndim == 2:
            return pd.DataFrame(obj, index=index)
        if obj.ndim == 3:
            c, s, dim = obj.shape
            chain_idx, draw_idx = np.meshgrid(np.arange(c), np.arange(s), indexing="ij")
            data = {"chain": chain_idx.ravel(), "draw": draw_idx.ravel()}
            for d in range(dim):
                data[f"theta{d}"] = obj[:, :, d].ravel()
            return pd.DataFrame(data)
        raise ValueError(f"pandas_interface: unsupported ndarray ndim={obj.ndim}")

    from stochpylib.timeseries import ForecastResult
    if isinstance(obj, ForecastResult):
        lo, hi = obj.confidence_interval()
        return pd.DataFrame({"mean": obj.mean, "std": obj.std, "lower": lo, "upper": hi}, index=index)

    from stochpylib.montecarlo import MCResult
    if isinstance(obj, MCResult):
        lo, hi = obj.confidence_interval()
        return pd.Series({"estimate": obj.estimate, "std_error": obj.std_error,
                          "n_samples": obj.n_samples, "ci_low": lo, "ci_high": hi}, name=name)

    from stochpylib.statistics import TestResult
    if isinstance(obj, TestResult):
        return pd.Series({"statistic": obj.statistic, "pvalue": obj.pvalue, "method": obj.method},
                         name=name)

    from stochpylib.utils._results import FitResult
    if isinstance(obj, FitResult):
        return pd.DataFrame(obj.table_)

    if isinstance(obj, dict) and obj and all(hasattr(v, "statistic") for v in obj.values()):
        from stochpylib.statistics import TestResult as _TR
        rows = [{"name": k, "statistic": v.statistic, "pvalue": v.pvalue} for k, v in obj.items()]
        return pd.DataFrame(rows)

    return pd.DataFrame(obj) if np.ndim(obj) >= 2 else pd.Series(obj, index=index, name=name)


_TORCH_DIST_NAMES = sorted(_backends._TORCH_DIST_MAP)


def torch_interface(obj, device="cpu", dtype="float64", requires_grad=False):
    """Convert an array/result to a torch tensor; a ``Distribution`` to a native
    ``torch.distributions`` object; a torch-written callable log-density to an
    ``AutodiffLogDensity``-style wrapper (numpy ``__call__``, exact ``.grad``)."""
    torch = _backends.import_backend("torch")
    torch_dtype = getattr(torch, dtype) if isinstance(dtype, str) else dtype

    from stochpylib.distributions._base import Distribution
    if isinstance(obj, Distribution):
        return _backends.torch_distribution(obj, device, torch_dtype)

    if callable(obj) and not isinstance(obj, (np.ndarray,)):
        return _backends.TorchAutodiffLogDensity(obj, device=device, dtype=torch_dtype)

    arr = numpy_interface(obj) if not isinstance(obj, np.ndarray) else obj
    return torch.tensor(np.asarray(arr, dtype=float), device=device, dtype=torch_dtype,
                        requires_grad=requires_grad)


class _JaxDistributionShim:
    """Small distribution-like shim over ``jax.scipy.stats`` (jax's own
    reimplementation, not ``scipy.stats``) for the families it covers."""

    _LOGPDF_MAP = {
        "Normal": lambda jsp, d, x: jsp.norm.logpdf(x, d.mu, d.sigma),
        "Exponential": lambda jsp, d, x: jsp.expon.logpdf(x, scale=1.0 / d.rate),
        "Gamma": lambda jsp, d, x: jsp.gamma.logpdf(x, d.shape, scale=d.scale),
        "Beta": lambda jsp, d, x: jsp.beta.logpdf(x, d.a, d.b),
        "Uniform": lambda jsp, d, x: jsp.uniform.logpdf(x, d.a, d.b - d.a),
        "Poisson": lambda jsp, d, x: jsp.poisson.logpmf(x, d.lam),
        "Bernoulli": lambda jsp, d, x: jsp.bernoulli.logpmf(x, d.p),
    }

    def __init__(self, dist):
        self._dist = dist
        name = type(dist).__name__
        if name not in self._LOGPDF_MAP:
            raise TypeError(f"jax_interface: no jax.scipy.stats mapping for {name!r}; "
                            f"supported: {sorted(self._LOGPDF_MAP)}")
        self._name = name

    def log_prob(self, x):
        jsp = _backends.jax_scipy_stats()
        return self._LOGPDF_MAP[self._name](jsp, self._dist, x)

    def sample(self, shape, random_state=None):
        jnp = _backends.jax_numpy()
        rvs = np.asarray(self._dist.rvs(int(np.prod(shape)), random_state=random_state),
                         dtype=float).reshape(shape)
        return jnp.asarray(rvs)


def jax_interface(obj, dtype="float64"):
    """Convert an array/result to a ``jax.numpy`` array; a ``Distribution`` to a
    small ``log_prob``/``sample`` shim; a jax-written callable to an autodiff
    wrapper via ``jax.grad`` (numpy ``__call__``/``.grad``)."""
    jnp = _backends.jax_numpy()

    from stochpylib.distributions._base import Distribution
    if isinstance(obj, Distribution):
        return _JaxDistributionShim(obj)

    if callable(obj) and not isinstance(obj, np.ndarray):
        return _backends.JaxAutodiffLogDensity(obj)

    arr = numpy_interface(obj) if not isinstance(obj, np.ndarray) else obj
    return jnp.asarray(np.asarray(arr, dtype=float))
