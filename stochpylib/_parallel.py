"""Shared parallel-execution machinery backing ``utils.performance.ParallelSimulation``
and the ``n_jobs=``/``parallel_backend=`` retrofit on montecarlo/financial_stochastics/
advanced_mcmc.

Dependency-free (numpy + stdlib ``concurrent.futures`` only); ``MCResult`` is imported
lazily inside functions to avoid a cycle with ``stochpylib.montecarlo``.
"""

import functools
import multiprocessing
import os
import pickle
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor

import numpy as np

from stochpylib import _rng


def resolve_n_jobs(n_jobs):
    """``-1`` -> all CPUs; otherwise ``max(1, int(n_jobs))``."""
    if n_jobs == -1:
        return os.cpu_count() or 1
    return max(1, int(n_jobs))


def _is_picklable(fn):
    try:
        pickle.dumps(fn)
        return True
    except Exception:
        return False


def execute(fn, tasks, n_jobs=1, backend="thread"):
    """Run ``fn(*task)`` for each ``task`` in ``tasks``, results in task order.

    ``backend``: ``"serial" | "thread" | "process" | "auto"``. ``"auto"`` picks
    ``"process"`` when ``fn`` is picklable and ``n_jobs > 1``, else ``"thread"``.
    ``n_jobs == 1`` always runs serially regardless of ``backend``.
    """
    n_jobs = resolve_n_jobs(n_jobs)
    tasks = list(tasks)
    if n_jobs <= 1 or backend == "serial" or len(tasks) <= 1:
        return [fn(*t) for t in tasks]

    if backend == "auto":
        backend = "process" if _is_picklable(fn) else "thread"

    if backend == "process":
        if not _is_picklable(fn):
            raise ValueError(
                "stochpylib: fn is not picklable for backend='process' "
                "(define it at module level, or pass backend='thread')")
        ctx = multiprocessing.get_context("spawn")
        with ProcessPoolExecutor(max_workers=n_jobs, mp_context=ctx) as ex:
            futures = [ex.submit(fn, *t) for t in tasks]
            return [f.result() for f in futures]
    elif backend == "thread":
        with ThreadPoolExecutor(max_workers=n_jobs) as ex:
            futures = [ex.submit(fn, *t) for t in tasks]
            return [f.result() for f in futures]
    else:
        raise ValueError(f"unknown backend {backend!r}; expected "
                          "'serial', 'thread', 'process' or 'auto'")


def chunk_sizes(n, chunk_size, even=False):
    """Split ``n`` into chunk sizes depending only on ``(n, chunk_size)`` -- never on
    the worker count, so results are worker-count invariant. ``even=True`` rounds
    each chunk up to an even number (antithetic pairing)."""
    n = int(n)
    chunk_size = max(1, int(chunk_size))
    if even:
        chunk_size += chunk_size % 2
    sizes = []
    remaining = n
    while remaining > 0:
        c = min(chunk_size, remaining)
        if even and c % 2 and remaining > c:
            c -= 1
        sizes.append(c)
        remaining -= c
    return sizes


def pool_mc_results(results):
    """Exactly pool a list of ``MCResult`` chunks into one, from each chunk's own
    mean/SE/n (no access to raw samples needed)."""
    from stochpylib.montecarlo import MCResult

    ns = np.array([r.n_samples for r in results], dtype=float)
    means = np.array([r.estimate for r in results], dtype=float)
    N = float(ns.sum())
    pooled_mean = float(np.sum(ns * means) / N)

    ss = 0.0
    for r, n_i, m_i in zip(results, ns, means):
        se_i = r.std_error
        if n_i > 1 and se_i == se_i and se_i is not None:  # not NaN
            s2_i = (se_i ** 2) * n_i
            ss += (n_i - 1) * s2_i
        ss += n_i * (m_i - pooled_mean) ** 2
    pooled_se = float(np.sqrt(ss / (N - 1) / N)) if N > 1 else float("nan")

    extras = dict(results[0].extras) if results[0].extras else {}
    extras["n_chunks"] = len(results)
    return MCResult(pooled_mean, pooled_se, int(N), results[0].method, extras)


def parallelizable(n_arg, even=False, default_chunk=50_000, reject_if=None):
    """Decorate a function/method that takes ``(<n_arg>, random_state=...)`` and
    returns a single ``MCResult``, adding keyword-only ``n_jobs=``/``parallel_backend=``.

    ``n_jobs=None`` (the default) calls the wrapped function unchanged --
    byte-identical to the pre-retrofit behavior. Otherwise the requested ``n``
    is split into fixed-size chunks (independent of worker count), each chunk
    reproducibly re-seeded off the caller's ``random_state`` via
    :func:`stochpylib._rng.spawn`, run through :func:`execute`, and pooled with
    :func:`pool_mc_results`. ``reject_if(bound_arguments) -> str | None`` can
    raise (via a returned message) when ``n_jobs`` is incompatible with other
    arguments (e.g. QMC scrambling, which loses its low-discrepancy guarantee
    if chopped into independently-seeded chunks).
    """
    import inspect

    def deco(fn):
        sig = inspect.signature(fn)

        @functools.wraps(fn)
        def wrapper(*args, n_jobs=None, parallel_backend="thread", **kwargs):
            if n_jobs is None:
                return fn(*args, **kwargs)

            bound = sig.bind(*args, **kwargs)
            bound.apply_defaults()
            if reject_if is not None:
                reason = reject_if(bound.arguments)
                if reason:
                    raise ValueError(f"n_jobs is not supported here: {reason}")

            n_total = bound.arguments[n_arg]
            random_state = bound.arguments.get("random_state", None)
            sizes = chunk_sizes(n_total, default_chunk, even=even)
            children = _rng.spawn(random_state, len(sizes))

            def _run_chunk(size, child):
                call_kwargs = dict(bound.arguments)
                call_kwargs[n_arg] = size
                call_kwargs["random_state"] = child
                return fn(**call_kwargs)

            tasks = list(zip(sizes, children))
            results = execute(lambda s, c: _run_chunk(s, c), tasks,
                              n_jobs=n_jobs, backend=parallel_backend)
            pooled = pool_mc_results(results)
            pooled.extras["n_jobs"] = resolve_n_jobs(n_jobs)
            pooled.extras["backend"] = parallel_backend
            return pooled

        wrapper.__doc__ = (wrapper.__doc__ or "") + (
            "\n\nParallel: pass n_jobs=<int> (-1 for all CPUs) to split the run into "
            "independently-seeded chunks pooled into one MCResult; n_jobs=None (default) "
            "runs the original single-stream implementation unchanged.")
        return wrapper
    return deco
