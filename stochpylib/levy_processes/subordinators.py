"""Subordinators: nondecreasing Levy processes used as stochastic time changes.

A subordinator T_t is a Levy process with nondecreasing sample paths. Used
directly (``GammaSubordinator`` for Variance-Gamma, ``InverseGaussianSubordinator``
for NIG) or through :class:`stochpylib.levy_processes.levy.SubordinatedProcess`
to time-change a Brownian motion. All sampling is native numpy/scipy with
``random_state=`` seeds; no scipy.stats wrapping.
"""

import numpy as np
from scipy import integrate, special

__all__ = [
    "Subordinator", "GammaSubordinator", "InverseGaussianSubordinator",
    "StableSubordinator", "TemperingSubordinator",
]


class Subordinator:
    """Base class for subordinators (nondecreasing Levy processes).

    Subclasses implement ``_increment(dt, rng)`` returning one increment over a
    time span ``dt``. The shared machinery provides path simulation and the
    standard conventions: ``random_state`` seeds, ``sample(times)`` returning
    the process values at ``times`` (including the origin), and
    ``increments(dt, n, random_state)``.
    """

    def _increment(self, dt, rng):
        raise NotImplementedError

    def sample(self, times, random_state=None):
        """Process values at ``times`` (array-like, must include 0 or start there).

        Returns an array of the same length as ``times`` with T_0 = 0.
        """
        times = np.asarray(times, dtype=float)
        if times.size == 0:
            return np.array([])
        rng = np.random.default_rng(random_state)
        dts = np.diff(np.concatenate(([0.0], times)))
        incs = np.array([self._increment(max(dt, 0.0), rng) for dt in dts])
        return np.cumsum(incs)

    def increments(self, dt, n, random_state=None):
        """``n`` i.i.d. increments over time span ``dt`` (as an array)."""
        rng = np.random.default_rng(random_state)
        return np.array([self._increment(dt, rng) for _ in range(n)]) \
            if n else np.array([])

    def simulate(self, T, n_steps, n_paths=1, random_state=None):
        """Simulate ``n_paths`` paths on ``[0, T]`` with ``n_steps`` steps.

        Returns an array of shape ``(n_paths, n_steps + 1)`` including the
        origin column.
        """
        rng = np.random.default_rng(random_state)
        dt = float(T) / n_steps
        incs = np.array([[self._increment(dt, rng) for _ in range(n_steps)]
                         for _ in range(n_paths)])
        paths = np.empty((n_paths, n_steps + 1))
        paths[:, 0] = 0.0
        np.cumsum(incs, axis=1, out=paths[:, 1:])
        return paths


class GammaSubordinator(Subordinator):
    """Gamma subordinator: T_t with Gamma(rate * t, scale) marginals.

    Mean ``rate * scale * t``, variance ``rate * scale**2 * t``. The time
    change behind the Variance-Gamma process.
    """

    def __init__(self, rate=1.0, scale=1.0):
        if rate <= 0 or scale <= 0:
            raise ValueError("rate and scale must be positive")
        self.rate = float(rate)
        self.scale = float(scale)

    def _increment(self, dt, rng):
        if dt <= 0:
            return 0.0
        return float(rng.gamma(shape=self.rate * dt, scale=self.scale))


class InverseGaussianSubordinator(Subordinator):
    """Inverse Gaussian subordinator with E[T_t] = t.

    Increments follow InvGaussian(mu=dt, lam=lam * dt) (the delta-unit
    parameterization, so the process has unit mean drift and variance
    ``t**2 / (lam * dt)`` per increment of size ``dt``). Delegates to the
    library's native InvGaussian sampler.
    """

    def __init__(self, lam=1.0):
        if lam <= 0:
            raise ValueError("lam must be positive")
        self.lam = float(lam)

    def _increment(self, dt, rng):
        if dt <= 0:
            return 0.0
        from stochpylib.distributions import InvGaussian

        return float(np.atleast_1d(InvGaussian(mu=dt, lam=self.lam * dt).rvs(1, random_state=rng))[0])


class StableSubordinator(Subordinator):
    """Positive (1/alpha)-stable subordinator, 0 < alpha < 1.

    Marginals have Laplace transform E[exp(-lam * T_t)] = exp(-t * lam**alpha).
    Increments are spectrally-positive alpha-stable, sampled by the library's
    validated Chambers-Mallows-Stuck machinery (beta=1) with the stable
    self-similarity scaling ``(dt)**(1/alpha)``.
    """

    def __init__(self, alpha=0.5):
        if not 0.0 < alpha < 1.0:
            raise ValueError("alpha must lie in (0, 1)")
        self.alpha = float(alpha)

    def _increment(self, dt, rng):
        if dt <= 0:
            return 0.0
        from stochpylib.distributions import StableDistribution

        # The library's stable sampler uses the S1 (Nolan) parameterization,
        # whose beta=1 Laplace transform carries an extra constant:
        # E[exp(-lam*X)] = exp(-lam**alpha / cos(pi*alpha/2)) for scale=1.
        # Rescaling by cos(pi*alpha/2)**(1/alpha) cancels that constant so
        # the subordinator matches its documented E[exp(-lam*T_t)] =
        # exp(-t*lam**alpha) exactly.
        norm = np.cos(np.pi * self.alpha / 2.0) ** (1.0 / self.alpha)
        s = StableDistribution(alpha=self.alpha, beta=1.0, loc=0.0,
                               scale=norm * dt ** (1.0 / self.alpha))
        return float(s.rvs(1, random_state=rng))


def upper_gamma_negative(s, x):
    """Upper incomplete gamma ``Gamma(s, x)`` for negative shape s in (-2, 0).

    scipy's ``gammaincc`` only accepts positive shapes; the recurrence
    ``Gamma(s+1, x) = s * Gamma(s, x) + x**s * exp(-x)`` lifts the argument
    to a positive shape (one step for s in (-1, 0), two for s in (-2, -1]).
    """
    if not -2.0 < s < 0.0:
        raise ValueError("s must lie in (-2, 0)")
    if s > -1.0:
        g1 = special.gammaincc(1.0 + s, x) * special.gamma(1.0 + s)
        return (g1 - x ** s * np.exp(-x)) / s
    # s in (-2, -1]: first lift s -> s+1 (still negative), then to s+2 > 0
    g2 = special.gammaincc(2.0 + s, x) * special.gamma(2.0 + s)
    g1 = (g2 - x ** (s + 1.0) * np.exp(-x)) / (s + 1.0)
    return (g1 - x ** s * np.exp(-x)) / s


class TemperingSubordinator(Subordinator):
    """Tempered stable subordinator (the CGMY/Y-tempering time change).

    Levy measure ``nu(dx) = C * exp(-lam * x) * x**(-1 - alpha) dx`` with
    ``0 < alpha < 1``, ``C > 0``, ``lam > 0``. Simulated by a compound-Poisson
    approximation that truncates jumps below ``jump_floor`` — the truncation is
    documented, the retained jump intensity above the floor is reported by
    ``truncation_mass`` and the analytic mean
    ``C * Gamma(1 - alpha) * lam**(alpha - 1) * t`` is preserved by
    re-injecting the truncated-out small-jump mean as drift.
    """

    def __init__(self, C=1.0, lam=5.0, alpha=0.5, jump_floor=1e-4):
        if not 0.0 < alpha < 1.0:
            raise ValueError("alpha must lie in (0, 1)")
        if C <= 0 or lam <= 0:
            raise ValueError("C and lam must be positive")
        self.C = float(C)
        self.lam = float(lam)
        self.alpha = float(alpha)
        self.jump_floor = float(jump_floor)
        self._grid = None

    def truncation_mass(self):
        """Retained jump intensity above ``jump_floor`` (per unit time) —
        the Poisson rate ``_increment`` actually samples from; the mass
        excluded below the floor is ``mean_rate() - _retained_mean_rate()``
        (see ``_shortfall_rate``, in mean rather than count terms since the
        excluded-jump count diverges as ``jump_floor -> 0``)."""
        return self.C * self.lam ** self.alpha \
            * upper_gamma_negative(-self.alpha, self.lam * self.jump_floor)

    def mean_rate(self):
        """Analytic full mean per unit time: C * Gamma(1-alpha) * lam**(alpha-1)."""
        return self.C * special.gamma(1.0 - self.alpha) \
            * self.lam ** (self.alpha - 1.0)

    def _retained_intensity(self, dt):
        # Lambda(eps) = C * lam^alpha * Gamma(-alpha, lam*eps) (upper inc. gamma)
        return self.C * dt * self.lam ** self.alpha \
            * upper_gamma_negative(-self.alpha, self.lam * self.jump_floor)

    def _retained_mean_rate(self):
        # C * lam^(alpha-1) * Gamma(1-alpha, lam*eps) — positive shape, safe
        return self.C * self.lam ** (self.alpha - 1.0) \
            * special.gammaincc(1.0 - self.alpha, self.lam * self.jump_floor) \
            * special.gamma(1.0 - self.alpha)

    def _shortfall_rate(self):
        # mean carried by the truncated-out small jumps (lower inc. gamma)
        return self.mean_rate() - self._retained_mean_rate()

    def _jump_quantile_grid(self):
        if self._grid is not None:
            return self._grid
        x_max = max(self.jump_floor * 4000.0, 10.0 / self.lam)
        # log-spaced grid: the density ~ x**(-1-alpha) is singular at the
        # left endpoint, so a linear grid under-resolves it near jump_floor
        # and (via a naive Riemann sum) grossly overweights that region,
        # biasing the sampled jump-size distribution toward small jumps.
        # Log spacing plus cumulative-trapezoid quadrature tracks the true
        # density to within a fraction of a percent at 8192 points.
        x = np.logspace(np.log10(self.jump_floor), np.log10(x_max), 8192)
        logd = -self.lam * x + (-1.0 - self.alpha) * np.log(x)
        d = np.exp(logd - logd.max())
        cdf = integrate.cumulative_trapezoid(d, x, initial=0.0)
        cdf /= cdf[-1]
        self._grid = (x, cdf)
        return self._grid

    def _increment(self, dt, rng):
        if dt <= 0:
            return 0.0
        n_jumps = rng.poisson(self._retained_intensity(dt))
        x, cdf = self._jump_quantile_grid()
        jumps = np.interp(rng.random(n_jumps), cdf, x)
        # re-inject the truncated-out mean so E[increment] matches the analytic
        # tempered-stable mean exactly (small-jump compensation)
        return float(jumps.sum() + dt * self._shortfall_rate())
