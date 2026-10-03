"""Shared target wrapper and sampler base class for :mod:`stochpylib.advanced_mcmc`.

Every sampler subclasses :class:`MCMCSampler`, which owns chain bookkeeping (warmup,
thinning, multiple chains) and delegates the actual transition kernel to a ``_step``
hook. Every target log-density is wrapped in :class:`LogDensity`, which supplies a
finite-difference gradient/Hessian fallback when no analytic one is given -- the library
has no autodiff, so gradient-based samplers work either way.
"""

import numpy as np

from stochpylib.advanced_mcmc._common import (
    _broadcast_init,
    _numeric_grad,
    _numeric_hessian,
    _rng,
    _spawn_rngs,
)

__all__ = ["LogDensity", "MCMCSampler"]


class LogDensity:
    """Callable log-density target with an optional analytic gradient/Hessian.

    ``log_prob(theta) -> float``. ``grad(theta) -> (dim,)`` uses ``grad_log_prob`` when
    given, otherwise a central finite difference of ``log_prob`` with step ``fd_eps``.
    ``hessian(theta)`` uses an analytic ``hessian`` callable when given, otherwise a
    finite difference of ``grad`` (cheaper than differencing ``log_prob`` twice) if a
    gradient is available, else a finite difference of ``log_prob`` itself. Values that
    are NaN or ``+inf`` are mapped to ``-inf`` so samplers can treat any invalid point as
    zero density without special-casing. ``n_evals``/``n_grad_evals`` count calls.
    """

    def __init__(self, log_prob, grad_log_prob=None, hessian=None, fd_eps=1e-6, args=()):
        if isinstance(log_prob, LogDensity):
            other = log_prob
            log_prob = other.log_prob
            grad_log_prob = other.grad_log_prob if grad_log_prob is None else grad_log_prob
            hessian = other._hessian_fn if hessian is None else hessian
            fd_eps = other.fd_eps
            args = other.args
        self.log_prob = log_prob
        self.grad_log_prob = grad_log_prob
        self._hessian_fn = hessian
        self.fd_eps = float(fd_eps)
        self.args = tuple(args)
        self.n_evals = 0
        self.n_grad_evals = 0

    def __call__(self, theta):
        self.n_evals += 1
        theta = np.asarray(theta, dtype=float)
        try:
            v = float(self.log_prob(theta, *self.args))
        except (ValueError, FloatingPointError, ArithmeticError):
            return -np.inf
        if v != v or v == np.inf:
            return -np.inf
        return v

    @property
    def has_gradient(self):
        return self.grad_log_prob is not None

    def grad(self, theta):
        theta = np.asarray(theta, dtype=float)
        self.n_grad_evals += 1
        if self.grad_log_prob is not None:
            return np.asarray(self.grad_log_prob(theta, *self.args), dtype=float)
        return _numeric_grad(self.__call__, theta, self.fd_eps)

    def value_and_grad(self, theta):
        return self(theta), self.grad(theta)

    def hessian(self, theta):
        theta = np.asarray(theta, dtype=float)
        if self._hessian_fn is not None:
            return np.asarray(self._hessian_fn(theta, *self.args), dtype=float)
        if self.grad_log_prob is not None:
            return _numeric_hessian_of_grad(self.grad, theta, self.fd_eps * 10)
        return _numeric_hessian(self.__call__, theta, max(self.fd_eps * 100, 1e-4))

    @classmethod
    def from_distribution(cls, dist):
        """Wrap a library ``Distribution``/``MultivariateDistribution`` as a log-density.

        Univariate distributions take a scalar; multivariate ones take the whole vector.
        Density is clipped away from zero before taking the log so points outside the
        support map to a large negative (not ``-inf``, which would break gradients).
        """
        is_univariate = hasattr(dist, "support")

        def log_prob(theta):
            theta = np.asarray(theta, dtype=float)
            if is_univariate:
                x = theta[0] if theta.ndim >= 1 else theta
                p = np.clip(np.atleast_1d(np.asarray(dist.pdf(x), dtype=float)), 1e-300, None)
                return float(np.sum(np.log(p)))
            p = np.clip(float(np.asarray(dist.pdf(theta))), 1e-300, None)
            return float(np.log(p))

        return cls(log_prob)


def _numeric_hessian_of_grad(grad_fn, x, eps):
    """Central difference of a vector gradient function -> symmetric Hessian."""
    x = np.asarray(x, dtype=float)
    n = len(x)
    H = np.empty((n, n))
    steps = eps * np.maximum(1.0, np.abs(x))
    for i in range(n):
        dx = np.zeros(n)
        dx[i] = steps[i]
        gp = grad_fn(x + dx)
        gm = grad_fn(x - dx)
        H[:, i] = (gp - gm) / (2 * steps[i])
    return 0.5 * (H + H.T)


def _run_chain_worker(sampler, init, r, d):
    """Module-level (hence picklable, unlike a closure) worker for
    ``MCMCSampler.sample(..., n_jobs=)``'s ``backend="process"`` path."""
    draws, logps, acc_rate = sampler._run_chain(init, r, d)
    return draws, logps, acc_rate, sampler


class MCMCSampler:
    """Base class for every MCMC sampler: chain bookkeeping around a ``_step`` kernel.

    Subclasses implement ``_step(state, rng, adapt) -> (state, accept_stat)`` and may
    override ``_init_state``, ``_start_chain`` (reset per-chain adaptation state) and
    ``_end_warmup`` (freeze adaptation before the kept draws begin). ``sample`` runs
    ``n_warmup`` adaptation-only iterations per chain (discarded), then ``n_samples``
    kept iterations (every ``thin``-th accepted into ``chains_``), one chain at a time
    with independent child RNGs so multi-chain runs are reproducible and embarrassingly
    parallel in spirit.
    """

    def __init__(self, log_prob, n_samples=1000, n_warmup=None, n_chains=1, thin=1,
                 grad_log_prob=None):
        self.target = LogDensity(log_prob, grad_log_prob)
        self.n_samples = int(n_samples)
        if self.n_samples < 1:
            raise ValueError("n_samples must be >= 1")
        self.n_warmup = int(n_samples // 2) if n_warmup is None else int(n_warmup)
        if self.n_warmup < 0:
            raise ValueError("n_warmup must be >= 0")
        self.n_chains = int(n_chains)
        if self.n_chains < 1:
            raise ValueError("n_chains must be >= 1")
        self.thin = int(thin)
        if self.thin < 1:
            raise ValueError("thin must be >= 1")

    def _init_state(self, theta, rng):
        return {"theta": theta, "logp": self.target(theta)}

    def _start_chain(self, dim, rng):
        pass

    def _end_warmup(self):
        pass

    def _step(self, state, rng, adapt):
        raise NotImplementedError

    def _run_chain(self, init, rng, dim):
        """Run one full chain (warmup + kept draws) and return
        ``(draws, logps, acceptance_rate)``. Split out of :meth:`sample` so it
        can run either inline (serial) or on a worker (``n_jobs=``)."""
        state = self._init_state(init, rng)
        if not np.isfinite(state["logp"]):
            raise ValueError("theta_init has zero density under log_prob")
        self._start_chain(dim, rng)
        for _ in range(self.n_warmup):
            state, _ = self._step(state, rng, True)
        self._end_warmup()
        draws = np.empty((self.n_samples, dim))
        logps = np.empty(self.n_samples)
        n_acc = 0.0
        n_total = self.n_samples * self.thin
        for i in range(n_total):
            state, a = self._step(state, rng, False)
            n_acc += float(a)
            if (i + 1) % self.thin == 0:
                draws[i // self.thin] = state["theta"]
                logps[i // self.thin] = state["logp"]
        return draws, logps, n_acc / n_total

    def sample(self, theta_init, random_state=None, n_jobs=None, backend="thread"):
        """Run every chain and set ``chains_``/``log_probs_``/``acceptance_rates_``.

        ``n_jobs=None`` (default) runs chains one after another in-process --
        byte-identical to the pre-V0.20.0 behavior. ``n_jobs=`` an int runs
        chains on separate workers (each on its own deep-copied sampler, so
        per-chain adaptation state never leaks across chains); the numeric
        chains themselves are identical either way since each chain already
        draws from its own independent child RNG.
        """
        rng = _rng(random_state)
        inits = _broadcast_init(theta_init, self.n_chains)
        dim = inits.shape[1]
        rngs = _spawn_rngs(rng, self.n_chains)

        if n_jobs is None:
            results = [self._run_chain(inits[c], rngs[c], dim) for c in range(self.n_chains)]
        else:
            import copy

            from stochpylib._parallel import execute

            tasks = [(copy.deepcopy(self), inits[c], rngs[c], dim) for c in range(self.n_chains)]
            raw = execute(_run_chain_worker, tasks, n_jobs=n_jobs, backend=backend)
            results = [(d, l, a) for d, l, a, _ in raw]
            # Match the serial path: `self` ends with the last chain's adaptation state, read from
            # the sampler each worker mutated.
            last_sampler = raw[-1][3]
            for key, value in last_sampler.__dict__.items():
                if key != "target":
                    self.__dict__[key] = value

        chains = np.stack([r[0] for r in results])
        logps = np.stack([r[1] for r in results])
        acc = np.array([r[2] for r in results])
        self.chains_ = chains
        self.log_probs_ = logps
        self.acceptance_rates_ = acc
        self.acceptance_rate_ = float(acc.mean())
        self.dim_ = dim
        return self

    def get_chains(self):
        if not hasattr(self, "chains_"):
            raise RuntimeError("sample() must be called first")
        return self.chains_

    def get_samples(self):
        chains = self.get_chains()
        return chains.reshape(-1, self.dim_)

    def summary(self):
        from stochpylib.advanced_mcmc.diagnostics import TraceAnalysis
        return TraceAnalysis(self.get_chains()).summary()

    def __repr__(self):
        base = (f"{self.__class__.__name__}(n_samples={self.n_samples}, "
                f"n_warmup={self.n_warmup}, n_chains={self.n_chains})")
        if hasattr(self, "acceptance_rate_"):
            base = base[:-1] + f", acceptance_rate={self.acceptance_rate_:.3g})"
        return base
