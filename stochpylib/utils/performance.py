"""Timing, profiling, parallel/GPU execution, JIT compilation, vectorized numeric
helpers, and buffer reuse.

``ParallelSimulation``/``GPUBackend``/``JIT_compile`` are the shared backend that
simulation-heavy modules (``montecarlo``, ``financial_stochastics``, ``advanced_mcmc``)
opt into via ``n_jobs=``/``backend=`` rather than reimplementing (ARCHITECTURE.md).
Heavy optional dependencies (numba/torch/jax/cupy) are never imported here directly --
everything routes through ``utils._backends``, the one file allowed to import them.
"""

import cProfile
import functools
import io as _io
import pstats
import time
import tracemalloc
import warnings

import numpy as np

from stochpylib import _parallel
from stochpylib.utils import _backends

__all__ = ["Benchmark", "GPUBackend", "JIT_compile", "MemoryPool", "ParallelSimulation",
           "Profiler", "VectorizedOps"]


class Benchmark:
    """Repeated wall-clock timing of a callable.

    ``repeat`` independent measurements of ``number`` calls each (``number=None``
    auto-ranges, like ``timeit``, until one repeat takes >= 0.05s).
    """

    def __init__(self, repeat=5, number=None, warmup=1):
        self.repeat = int(repeat)
        self.number = number
        self.warmup = int(warmup)

    def _auto_number(self, fn, args, kwargs):
        number = 1
        while True:
            t0 = time.perf_counter()
            for _ in range(number):
                fn(*args, **kwargs)
            elapsed = time.perf_counter() - t0
            if elapsed >= 0.05 or number >= 10_000_000:
                return number
            number *= 10

    def run(self, fn, *args, **kwargs):
        for _ in range(self.warmup):
            self.result_ = fn(*args, **kwargs)
        number = self.number if self.number is not None else self._auto_number(fn, args, kwargs)
        self.number_ = number
        times = np.empty(self.repeat)
        for i in range(self.repeat):
            t0 = time.perf_counter()
            for _ in range(number):
                self.result_ = fn(*args, **kwargs)
            times[i] = (time.perf_counter() - t0) / number
        self.times_ = times
        self.mean_ = float(times.mean())
        self.std_ = float(times.std(ddof=1)) if self.repeat > 1 else 0.0
        self.min_ = float(times.min())
        self.median_ = float(np.median(times))
        return self

    def to_result(self):
        from stochpylib.montecarlo import MCResult

        se = self.std_ / np.sqrt(self.repeat) if self.repeat > 1 else float("nan")
        return MCResult(self.mean_, se, self.repeat, "benchmark",
                        {"number": self.number_, "unit": "seconds/call"})

    def compare(self, fns, *args, **kwargs):
        """``fns``: ``{label: callable}``. Returns a list of dicts sorted fastest
        first, each with ``speedup_vs_first`` relative to the first-listed label."""
        rows = []
        first_mean = None
        for label, fn in fns.items():
            b = Benchmark(self.repeat, self.number, self.warmup).run(fn, *args, **kwargs)
            if first_mean is None:
                first_mean = b.mean_
            rows.append({"label": label, "mean": b.mean_, "std": b.std_,
                        "speedup_vs_first": first_mean / b.mean_ if b.mean_ > 0 else float("inf")})
        return rows

    def __repr__(self):
        if not hasattr(self, "mean_"):
            return "Benchmark(unrun)"
        return f"Benchmark(mean={self.mean_:.4g}s, std={self.std_:.2g}s, repeat={self.repeat})"


class Profiler:
    """Wraps stdlib ``cProfile``/``pstats`` (plus ``tracemalloc`` when
    ``memory=True``). Usable as a context manager or via ``.profile(fn, ...)``."""

    def __init__(self, sort="cumulative", memory=False):
        self.sort = sort
        self.memory = memory
        self.peak_memory_ = None

    def __enter__(self):
        self._pr = cProfile.Profile()
        if self.memory:
            tracemalloc.start()
        self._pr.enable()
        return self

    def __exit__(self, *exc):
        self._pr.disable()
        if self.memory:
            _, peak = tracemalloc.get_traced_memory()
            self.peak_memory_ = peak
            tracemalloc.stop()
        self._finalize()
        return False

    def profile(self, fn, *args, **kwargs):
        with self:
            result = fn(*args, **kwargs)
        return result

    def _finalize(self):
        stats = pstats.Stats(self._pr)
        rows = []
        for func, (cc, nc, tt, ct, _callers) in stats.stats.items():
            file_, line, name = func
            rows.append({"function": f"{name} ({file_}:{line})", "ncalls": nc,
                        "tottime": tt, "cumtime": ct})
        key = "cumtime" if self.sort.startswith("cum") else "tottime"
        rows.sort(key=lambda r: r[key], reverse=True)
        self.stats_ = rows
        self.total_time_ = float(sum(r["tottime"] for r in rows))

    def report(self, limit=20):
        buf = _io.StringIO()
        stats = pstats.Stats(self._pr, stream=buf)
        stats.sort_stats(self.sort).print_stats(limit)
        return buf.getvalue()


class ParallelSimulation:
    """Thin, public wrapper over :mod:`stochpylib._parallel`.

    Results are identical for any ``n_jobs``/``backend`` at a fixed
    ``(random_state, n, chunk_size)`` -- worker-count invariant by construction
    -- but are **not** equal to a legacy single-stream serial run at the same
    seed (each chunk draws its own independent child stream).

    On Windows (and anywhere ``backend="process"``/``"auto"`` picks a process
    pool), ``simulate``/``fn`` must be importable at module level and the
    driving script needs an ``if __name__ == "__main__":`` guard.
    """

    def __init__(self, n_jobs=-1, backend="auto", chunk_size=50_000):
        self.n_jobs = n_jobs
        self.backend = backend
        self.chunk_size = int(chunk_size)

    def map(self, fn, tasks):
        """Run ``fn(task)`` over ``tasks``, results in order (like the stdlib
        ``map()``/``Pool.map()`` convention -- one positional argument per task;
        ``fn`` itself is passed through unwrapped, so it stays picklable for
        ``backend="process"``)."""
        return _parallel.execute(fn, [(t,) for t in tasks], self.n_jobs, self.backend)

    def run(self, simulate, n, random_state=None):
        """``simulate(n_chunk, rng) -> ndarray``; returns the chunk outputs
        concatenated in order."""
        sizes = _parallel.chunk_sizes(n, self.chunk_size)
        children = self._spawn(random_state, len(sizes))
        chunks = _parallel.execute(simulate, list(zip(sizes, children)),
                                   self.n_jobs, self.backend)
        return np.concatenate([np.atleast_1d(c) for c in chunks])

    def estimate(self, simulate, n, random_state=None):
        """``simulate(n_chunk, rng) -> MCResult``; pools the chunks exactly."""
        sizes = _parallel.chunk_sizes(n, self.chunk_size)
        children = self._spawn(random_state, len(sizes))
        chunks = _parallel.execute(simulate, list(zip(sizes, children)),
                                   self.n_jobs, self.backend)
        pooled = self.pool(chunks)
        pooled.extras["n_jobs"] = _parallel.resolve_n_jobs(self.n_jobs)
        pooled.extras["backend"] = self.backend
        return pooled

    @staticmethod
    def pool(results):
        return _parallel.pool_mc_results(results)

    @staticmethod
    def _spawn(random_state, n):
        from stochpylib import _rng
        return _rng.spawn(random_state, n)

    def __repr__(self):
        return f"ParallelSimulation(n_jobs={self.n_jobs}, backend={self.backend!r})"


class GPUBackend:
    """Array backend abstraction over numpy / cupy / torch / jax.

    ``device="auto"`` picks cupy (if importable with a visible CUDA device),
    then torch (if ``torch.cuda.is_available()``), else numpy -- it never
    auto-selects CPU torch/jax (use them explicitly for that). ``standard_normal``
    draws with numpy's ``Generator`` by default and transfers the result, so every
    backend reproduces the same numbers to float tolerance; ``native_rng=True``
    uses the device's own RNG instead (faster, not cross-backend reproducible).
    """

    def __init__(self, device="auto", torch_device=None, native_rng=False):
        self.native_rng = native_rng
        self._torch_device = torch_device or "cpu"
        self.name = self._resolve(device)
        self.is_gpu = self.name in ("cupy",) or (
            self.name == "torch" and self._torch_device != "cpu")

    def _resolve(self, device):
        if device != "auto":
            return device
        if _backends.cupy_has_device():
            return "cupy"
        if _backends.is_installed("torch"):
            torch = _backends.import_backend("torch")
            if torch.cuda.is_available():
                self._torch_device = "cuda"
                return "torch"
        return "numpy"

    @staticmethod
    def available():
        return _backends.available_backends()

    # -- conversion --------------------------------------------------------
    def asarray(self, x):
        if self.name == "numpy":
            return np.asarray(x, dtype=float)
        if self.name == "cupy":
            cupy = _backends.import_backend("cupy")
            return cupy.asarray(x, dtype=cupy.float64)
        if self.name == "torch":
            torch = _backends.import_backend("torch")
            return torch.as_tensor(np.asarray(x, dtype=float), device=self._torch_device)
        if self.name == "jax":
            return _backends.jax_numpy().asarray(np.asarray(x, dtype=float))
        raise ValueError(f"unknown device {self.name!r}")

    def to_numpy(self, x):
        if self.name == "numpy":
            return np.asarray(x)
        if self.name == "cupy":
            return x.get()
        if self.name == "torch":
            return x.detach().cpu().numpy()
        if self.name == "jax":
            return np.asarray(x)
        raise ValueError(f"unknown device {self.name!r}")

    def _xp(self):
        if self.name == "numpy":
            return np
        if self.name == "cupy":
            return _backends.import_backend("cupy")
        if self.name == "torch":
            return _backends.import_backend("torch")
        if self.name == "jax":
            return _backends.jax_numpy()
        raise ValueError(f"unknown device {self.name!r}")

    def exp(self, x):
        return self._xp().exp(x)

    def log(self, x):
        return self._xp().log(x)

    def sqrt(self, x):
        return self._xp().sqrt(x)

    def maximum(self, a, b):
        return self._xp().maximum(a, b)

    def where(self, cond, a, b):
        return self._xp().where(cond, a, b)

    def cumsum(self, x, axis=None):
        xp = self._xp()
        return xp.cumsum(x, dim=axis) if self.name == "torch" else xp.cumsum(x, axis=axis)

    def sum(self, x, axis=None):
        return self._xp().sum(x, axis=axis) if self.name != "torch" else self._xp().sum(x, dim=axis)

    def mean(self, x, axis=None):
        return self._xp().mean(x, axis=axis) if self.name != "torch" else self._xp().mean(x, dim=axis)

    def std(self, x, ddof=0, axis=None):
        if self.name == "torch":
            torch = self._xp()
            n = x.numel() if axis is None else x.shape[axis]
            correction = ddof
            return torch.std(x, dim=axis, correction=correction) if axis is not None \
                else torch.std(x, correction=correction)
        return self._xp().std(x, axis=axis, ddof=ddof)

    def matmul(self, a, b):
        return self._xp().matmul(a, b)

    # -- RNG / simulation ---------------------------------------------------
    def standard_normal(self, shape, random_state=None):
        if self.native_rng and self.name != "numpy":
            if self.name == "torch":
                torch = _backends.import_backend("torch")
                gen = torch.Generator(device=self._torch_device)
                if random_state is not None:
                    gen.manual_seed(int(random_state))
                return torch.randn(*shape, generator=gen, device=self._torch_device,
                                   dtype=torch.float64)
            if self.name == "jax":
                key = _backends.jax_key(random_state)
                jax = _backends.jax_enable_x64()
                return jax.random.normal(key, shape)
            if self.name == "cupy":
                cupy = _backends.import_backend("cupy")
                rng = cupy.random.default_rng(random_state)
                return rng.standard_normal(shape)
        from stochpylib import _rng
        z = _rng.as_generator(random_state).standard_normal(shape)
        return self.asarray(z)

    def simulate_gbm(self, S0, mu, sigma, T, n_steps, n_paths, random_state=None):
        """Geometric Brownian motion paths, shape ``(n_paths, n_steps + 1)``,
        returned as a plain numpy array regardless of ``self.name``."""
        dt = T / n_steps
        z = self.standard_normal((n_paths, n_steps), random_state=random_state)
        drift = (mu - 0.5 * sigma ** 2) * dt
        vol = sigma * (dt ** 0.5)
        log_incr = drift + vol * z
        cum = self.cumsum(log_incr, axis=1)
        zeros_shape = (n_paths, 1)
        zeros = self.asarray(np.zeros(zeros_shape)) if self.name != "torch" else \
            self._xp().zeros(zeros_shape, dtype=cum.dtype, device=self._torch_device)
        log_paths = self._xp().concatenate([zeros, cum], axis=1) if self.name != "torch" \
            else self._xp().cat([zeros, cum], dim=1)
        paths = float(S0) * self.exp(log_paths)
        return np.asarray(self.to_numpy(paths), dtype=float)

    def mc_estimate(self, values):
        from stochpylib.montecarlo import MCResult

        arr = np.asarray(self.to_numpy(values), dtype=float).ravel()
        n = arr.size
        se = float(arr.std(ddof=1) / np.sqrt(n)) if n > 1 else float("nan")
        return MCResult(float(arr.mean()), se, n, "gpu-backend")

    def __repr__(self):
        return f"GPUBackend(name={self.name!r}, is_gpu={self.is_gpu})"


def JIT_compile(fn=None, *, backend="auto", cache=False, fallback=True, **options):
    """Decorator: JIT-compile ``fn`` with numba/jax when available, else run it
    unmodified in pure Python. Usable with or without call parens.

    ``backend``: ``"auto"`` (numba if usable, else python) | ``"numba"`` |
    ``"jax"`` | ``"none"``. On first-call compilation failure with
    ``fallback=True`` (the default), warns once and permanently falls back to
    the plain Python function (``wrapper.backend_`` records what actually ran).
    """
    def deco(func):
        return _JITWrapper(func, backend, cache, fallback, options)

    if fn is not None:
        return deco(fn)
    return deco


class _JITWrapper:
    """Callable wrapper returned by :func:`JIT_compile`. A class (not a bare
    function) so ``backend_``/``py_func`` can be real attributes."""

    def __init__(self, func, backend, cache, fallback, options):
        functools.update_wrapper(self, func)
        self.py_func = func
        self._backend_choice = backend
        self._cache = cache
        self._fallback = fallback
        self._options = options
        self._compiled = None
        self.backend_ = None

    def _compile(self):
        chosen = self._backend_choice
        if chosen == "auto":
            chosen = "numba" if _backends.is_installed("numba") else "none"
        if chosen == "none":
            self.backend_ = "python"
            self._compiled = self.py_func
            return
        try:
            if chosen == "numba":
                numba = _backends.import_backend("numba")
                self._compiled = numba.njit(cache=self._cache, **self._options)(self.py_func)
            elif chosen == "jax":
                jax = _backends.jax_enable_x64()
                self._compiled = jax.jit(self.py_func, **self._options)
            else:
                raise ValueError(f"unknown backend {chosen!r}")
            self.backend_ = chosen
        except ImportError as exc:
            if not self._fallback:
                raise
            warnings.warn(f"JIT_compile: {chosen} unavailable ({exc}); "
                          "falling back to plain Python", RuntimeWarning, stacklevel=3)
            self.backend_ = "python"
            self._compiled = self.py_func

    def __call__(self, *args, **kwargs):
        if self._compiled is None:
            self._compile()
        try:
            return self._compiled(*args, **kwargs)
        except Exception:
            if self.backend_ == "python" or not self._fallback:
                raise
            warnings.warn(f"JIT_compile: {self.backend_} call failed; "
                          "falling back to plain Python", RuntimeWarning, stacklevel=2)
            self.backend_ = "python"
            self._compiled = self.py_func
            return self.py_func(*args, **kwargs)


class VectorizedOps:
    """Static-method numeric helpers (log-domain stability, rolling stats,
    pairwise distances, chunked evaluation)."""

    @staticmethod
    def logsumexp(a, axis=None, b=None):
        a = np.asarray(a, dtype=float)
        amax = np.max(a, axis=axis, keepdims=True)
        amax = np.where(np.isfinite(amax), amax, 0.0)
        if b is not None:
            terms = np.asarray(b, dtype=float) * np.exp(a - amax)
            s = np.sum(terms, axis=axis, keepdims=True)
        else:
            s = np.sum(np.exp(a - amax), axis=axis, keepdims=True)
        out = np.log(s) + amax
        return np.squeeze(out, axis=axis) if axis is not None else out.reshape(())

    @staticmethod
    def softmax(a, axis=-1):
        a = np.asarray(a, dtype=float)
        amax = np.max(a, axis=axis, keepdims=True)
        e = np.exp(a - amax)
        return e / np.sum(e, axis=axis, keepdims=True)

    @staticmethod
    def log1mexp(x):
        """Numerically stable ``log(1 - exp(x))`` for ``x <= 0``."""
        x = np.asarray(x, dtype=float)
        return np.where(x > -np.log(2.0), np.log(-np.expm1(x)), np.log1p(-np.exp(x)))

    @staticmethod
    def rolling_mean(x, window):
        x = np.asarray(x, dtype=float)
        c = np.cumsum(np.insert(x, 0, 0.0))
        return (c[window:] - c[:-window]) / window

    @staticmethod
    def rolling_std(x, window, ddof=1):
        x = np.asarray(x, dtype=float)
        m = VectorizedOps.rolling_mean(x, window)
        c2 = np.cumsum(np.insert(x ** 2, 0, 0.0))
        mean_sq = (c2[window:] - c2[:-window]) / window
        var = np.maximum(mean_sq - m ** 2, 0.0) * window / max(window - ddof, 1)
        return np.sqrt(var)

    @staticmethod
    def pairwise_distances(X, Y=None, metric="euclidean"):
        X = np.asarray(X, dtype=float)
        Y = X if Y is None else np.asarray(Y, dtype=float)
        if metric in ("euclidean", "sqeuclidean"):
            xx = np.sum(X ** 2, axis=1)[:, None]
            yy = np.sum(Y ** 2, axis=1)[None, :]
            d2 = np.maximum(xx + yy - 2.0 * X @ Y.T, 0.0)
            return d2 if metric == "sqeuclidean" else np.sqrt(d2)
        if metric == "cityblock":
            return np.sum(np.abs(X[:, None, :] - Y[None, :, :]), axis=2)
        raise ValueError(f"unknown metric {metric!r}")

    @staticmethod
    def batched_mahalanobis(X, mean, cov):
        X = np.asarray(X, dtype=float)
        mean = np.asarray(mean, dtype=float)
        L = np.linalg.cholesky(np.asarray(cov, dtype=float))
        diff = (X - mean).T
        y = np.linalg.solve(L, diff)
        return np.sqrt(np.sum(y ** 2, axis=0))

    @staticmethod
    def chunked_apply(fn, X, chunk_size):
        X = np.asarray(X)
        out = []
        for i in range(0, len(X), chunk_size):
            out.append(fn(X[i:i + chunk_size]))
        return np.concatenate(out) if out and np.ndim(out[0]) > 0 else np.asarray(out)


class MemoryPool:
    """Reuse fixed-``(shape, dtype)`` ndarray buffers instead of reallocating,
    with LRU eviction once ``max_bytes`` is exceeded (``None`` = unbounded)."""

    def __init__(self, max_bytes=None):
        self.max_bytes = max_bytes
        self._free = {}       # key -> list[ndarray]
        self._order = []      # LRU order of keys with free buffers
        self.hits_ = 0
        self.misses_ = 0
        self.bytes_in_use_ = 0
        self.bytes_pooled_ = 0

    @staticmethod
    def _key(shape, dtype):
        return (tuple(shape), np.dtype(dtype))

    def get(self, shape, dtype=float, zero=False):
        key = self._key(shape, dtype)
        bucket = self._free.get(key)
        if bucket:
            arr = bucket.pop()
            if not bucket:
                del self._free[key]
            if key in self._order:
                self._order.remove(key)
            self.bytes_pooled_ -= arr.nbytes
            self.hits_ += 1
        else:
            arr = np.empty(shape, dtype=dtype)
            self.misses_ += 1
        if zero:
            arr.fill(0)
        self.bytes_in_use_ += arr.nbytes
        return arr

    def release(self, arr):
        key = self._key(arr.shape, arr.dtype)
        self._free.setdefault(key, []).append(arr)
        if key in self._order:
            self._order.remove(key)
        self._order.append(key)
        self.bytes_in_use_ -= arr.nbytes
        self.bytes_pooled_ += arr.nbytes
        self._evict()

    def _evict(self):
        if self.max_bytes is None:
            return
        while self.bytes_pooled_ > self.max_bytes and self._order:
            key = self._order.pop(0)
            bucket = self._free.get(key)
            if bucket:
                arr = bucket.pop(0)
                self.bytes_pooled_ -= arr.nbytes
                if not bucket:
                    del self._free[key]

    def clear(self):
        self._free.clear()
        self._order.clear()
        self.bytes_pooled_ = 0

    class _Borrow:
        def __init__(self, pool, shape, dtype, zero):
            self.pool, self.shape, self.dtype, self.zero = pool, shape, dtype, zero

        def __enter__(self):
            self.arr = self.pool.get(self.shape, self.dtype, self.zero)
            return self.arr

        def __exit__(self, *exc):
            self.pool.release(self.arr)
            return False

    def borrow(self, shape, dtype=float, zero=False):
        return MemoryPool._Borrow(self, shape, dtype, zero)

    def __repr__(self):
        return (f"MemoryPool(hits={self.hits_}, misses={self.misses_}, "
                f"bytes_pooled={self.bytes_pooled_})")
