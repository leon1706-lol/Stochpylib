"""Core Levy machinery: the base process, stable processes, the spectral
representation, subordination, and the Levy-Khintchine triplet.

Everything is native numpy/scipy; the stable samplers delegate to the
library's validated Chambers-Mallows-Stuck machinery in
``stochpylib.distributions.heavy_tail`` (scipy.stats stays a test oracle).
"""

import numpy as np

__all__ = [
    "LevyProcess", "StableProcess", "AlphaStableDistribution",
    "SpectrallyPositive", "SubordinatedProcess", "LevyKhintchine",
]


class LevyProcess:
    """Base class: Brownian component plus compound-Poisson jumps.

    Represents the Levy process with characteristic function
    ``exp(t * (i*u*b - 0.5*sigma^2*u^2 + rate*(jump_cf(u) - 1)))``.
    Subclasses override ``_increment(dt, rng)`` for non-CPP dynamics;
    the default implementation simulates drift + diffusion + Poisson jumps.
    ``jump_sampler`` is any callable ``(n, rng) -> array`` of jump sizes.
    """

    def __init__(self, b=0.0, sigma=0.0, jump_rate=0.0, jump_sampler=None,
                 jump_cf=None):
        if sigma < 0 or jump_rate < 0:
            raise ValueError("sigma and jump_rate must be non-negative")
        self.b = float(b)
        self.sigma = float(sigma)
        self.jump_rate = float(jump_rate)
        self.jump_sampler = jump_sampler
        self.jump_cf = jump_cf

    def _increment(self, dt, rng):
        inc = self.b * dt
        if self.sigma > 0:
            inc += self.sigma * np.sqrt(dt) * rng.standard_normal()
        if self.jump_rate > 0 and self.jump_sampler is not None:
            n = rng.poisson(self.jump_rate * dt)
            if n:
                inc += float(np.sum(self.jump_sampler(n, rng)))
        return inc

    def simulate(self, T, n_steps, n_paths=1, random_state=None):
        """Paths on ``[0, T]``; returns array of shape ``(n_paths, n_steps+1)``."""
        rng = np.random.default_rng(random_state)
        dt = float(T) / n_steps
        incs = np.array([[self._increment(dt, rng) for _ in range(n_steps)]
                         for _ in range(n_paths)])
        paths = np.empty((n_paths, n_steps + 1))
        paths[:, 0] = 0.0
        np.cumsum(incs, axis=1, out=paths[:, 1:])
        return paths

    def characteristic_function(self, u, t):
        """Characteristic function of ``X_t`` (Levy-Khintchine, CPP form)."""
        u = np.asarray(u, dtype=float)
        phi = 1j * u * self.b - 0.5 * self.sigma ** 2 * u ** 2
        if self.jump_rate > 0 and self.jump_cf is not None:
            phi = phi + self.jump_rate * (self.jump_cf(u) - 1.0)
        return np.exp(t * phi)


class StableProcess(LevyProcess):
    """Alpha-stable Levy process with self-similarity index 1/alpha.

    Increments are alpha-stable with skewness ``beta`` in [-1, 1], sampled by
    the library's validated Chambers-Mallows-Stuck machinery. For alpha < 1
    and beta = 1 the process is spectrally positive (nondecreasing).
    """

    def __init__(self, alpha=1.5, beta=0.0, loc=0.0, scale=1.0):
        if not 0.0 < alpha <= 2.0:
            raise ValueError("alpha must lie in (0, 2]")
        if not -1.0 <= beta <= 1.0:
            raise ValueError("beta must lie in [-1, 1]")
        self.alpha = float(alpha)
        self.beta = float(beta)
        self.loc = float(loc)
        self.scale = float(scale)
        super().__init__(b=loc, sigma=0.0)

    def _increment(self, dt, rng):
        from stochpylib.distributions import StableDistribution

        s = StableDistribution(alpha=self.alpha, beta=self.beta, loc=0.0,
                               scale=self.scale * dt ** (1.0 / self.alpha))
        return float(np.asarray(s.rvs(1, random_state=rng)).ravel()[0])

    def characteristic_function(self, u, t):
        # S1-parameterization stable exponent, scale gamma, loc delta:
        # exp(t * (i*u*delta - gamma^alpha |u|^alpha (1 - i*beta*sign(u)*tan(pi*alpha/2))))
        u = np.asarray(u, dtype=float)
        if self.alpha == 1.0:
            skew = 2.0 / np.pi * self.beta * np.log(np.abs(u))
        else:
            skew = self.beta * np.sign(u) * np.tan(np.pi * self.alpha / 2.0)
        phi = 1j * u * self.loc - (self.scale * np.abs(u)) ** self.alpha \
            * (1.0 - 1j * skew)
        return np.exp(t * phi)


class AlphaStableDistribution:
    """Adapter exposing the library's validated alpha-stable distribution
    (``distributions.heavy_tail``) as a Levy increment law.

    Delegates every method to :class:`stochpylib.distributions.AlphaStable`;
    this class exists so the Levy module owns its dependency surface and the
    conformance tests can pin the spec name.
    """

    def __init__(self, alpha=2.0, loc=0.0, scale=1.0):
        from stochpylib.distributions import AlphaStable

        self._impl = AlphaStable(alpha, loc, scale)
        self.alpha = float(alpha)
        self.loc = float(loc)
        self.scale = float(scale)

    def rvs(self, n, random_state=None):
        return self._impl.rvs(n, random_state)

    def cf(self, u):
        return self._impl.cf(u)

    def pdf(self, x):
        return self._impl.pdf(x)

    def cdf(self, x):
        return self._impl.cdf(x)

    def fit(self, data):
        return self._impl.fit(data)


class SpectrallyPositive(StableProcess):
    """Spectrally-positive alpha-stable process (beta = 1), 0 < alpha < 1.

    Nondecreasing paths; the canonical purely-jump Levy driver behind
    subordinators. Increments carry no Gaussian component and no negative
    jumps by construction.
    """

    def __init__(self, alpha=0.5, scale=1.0):
        if not 0.0 < alpha < 1.0:
            raise ValueError("alpha must lie in (0, 1) for the spectrally "
                             "positive process")
        super().__init__(alpha=alpha, beta=1.0, loc=0.0, scale=scale)


class SubordinatedProcess:
    """A base process time-changed by a subordinator: X_t = B_{T_t}.

    ``base`` must provide ``_increment(dt, rng)`` (Brownian motion by
    convention; any :class:`LevyProcess` works). ``subordinator`` is any
    :class:`~stochpylib.levy_processes.subordinators.Subordinator`.
    """

    def __init__(self, base=None, subordinator=None):
        if subordinator is None:
            raise ValueError("a subordinator is required")
        self.base = base if base is not None else BrownianMotion()
        self.subordinator = subordinator

    def _base_increment(self, dt, rng):
        return self.base._increment(dt, rng)

    def sample(self, times, random_state=None):
        """Values of X at ``times``: base increments evaluated at the
        subordinator's realized time increments (one coupled path)."""
        times = np.asarray(times, dtype=float)
        rng = np.random.default_rng(random_state)
        dts = np.diff(np.concatenate(([0.0], times)))
        taus = np.array([self.subordinator._increment(dt, rng) for dt in dts])
        xs = np.array([self._base_increment(t, rng) for t in taus])
        return np.cumsum(xs)

    def simulate(self, T, n_steps, n_paths=1, random_state=None):
        rng = np.random.default_rng(random_state)
        dt = float(T) / n_steps
        paths = np.empty((n_paths, n_steps + 1))
        paths[:, 0] = 0.0
        for p in range(n_paths):
            taus = np.array([self.subordinator._increment(dt, rng)
                             for _ in range(n_steps)])
            xs = np.array([self._base_increment(t, rng) for t in taus])
            paths[p, 1:] = np.cumsum(xs)
        return paths


class BrownianMotion:
    """Standard Brownian motion (the default base of SubordinatedProcess)."""

    def _increment(self, dt, rng):
        return rng.standard_normal() * np.sqrt(dt) if dt > 0 else 0.0


class LevyKhintchine:
    """Levy-Khintchine triplet (b, sigma^2, nu) with its characteristic exponent.

    Supports the Brownian part ``(b, sigma)`` plus either a compound-Poisson
    jump component (``jump_rate`` + ``jump_cf``) or a nonnegative subordinator
    component (``laplace_exponent``: psi(lam) with E[exp(-lam T_1)] =
    exp(-psi(lam)), e.g. ``lam**alpha`` for a stable subordinator). The
    exponent is ``i*u*b - 0.5*sigma^2*u^2 + rate*(jump_cf(u)-1)`` or, for the
    subordinator form, ``i*u*b - 0.5*sigma^2*u^2 + psi(-i*u)`` evaluated
    analytically when ``psi`` is a pure power law.
    """

    def __init__(self, b=0.0, sigma=0.0, jump_rate=0.0, jump_cf=None,
                 laplace_exponent=None):
        if sigma < 0 or jump_rate < 0:
            raise ValueError("sigma and jump_rate must be non-negative")
        self.b = float(b)
        self.sigma = float(sigma)
        self.jump_rate = float(jump_rate)
        self.jump_cf = jump_cf
        self.laplace_exponent = laplace_exponent

    def exponent(self, u):
        u = np.asarray(u, dtype=float)
        phi = 1j * u * self.b - 0.5 * self.sigma ** 2 * u ** 2
        if self.jump_rate > 0 and self.jump_cf is not None:
            phi = phi + self.jump_rate * (self.jump_cf(u) - 1.0)
        if self.laplace_exponent is not None:
            # psi(-i u) for a power-law Laplace exponent lam^alpha extends to
            # |u|^alpha * exp(-i * sign(u) * alpha * pi / 2) via the principal
            # complex power
            lam = -1j * u
            phi = phi + self.laplace_exponent(lam)
        return phi

    def characteristic_function(self, u, t):
        return np.exp(t * self.exponent(u))
