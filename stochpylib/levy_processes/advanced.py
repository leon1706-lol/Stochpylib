"""Advanced stochastic processes: point processes (Hawkes, Cox, renewal,
semi-Markov), branching, Gaussian random fields, and random measures.

Simulation is by thinning (Ogata) for the intensity-driven processes; the
univariate exponential Hawkes carries an exact recursive MLE with the
time-rescaling residual KS test. Everything is seeded via ``random_state=``.
"""

import numpy as np
from scipy import optimize, stats

__all__ = [
    "SemiMarkovProcess", "RenewalProcess", "BranchingProcess", "HawkesProcess",
    "MultivariateHawkes", "CoxProcess", "GaussianRandomField", "RandomMeasure",
]


class HawkesProcess:
    """Univariate Hawkes process with exponential kernel.

    Intensity ``lambda(t) = mu + sum_{t_i < t} alpha * exp(-beta (t - t_i))``.
    ``simulate`` uses Ogata's thinning algorithm; ``fit`` is the exact
    recursive maximum-likelihood estimator (L-BFGS-B on the 3-parameter
    log-likelihood); ``branching_ratio()`` returns ``alpha / beta`` (must be
    < 1 for stability); ``ks_residuals()`` applies the time-rescaling theorem
    (compensator-transformed arrivals are unit-rate Poisson) and returns the
    KS ``(statistic, p_value)``.
    """

    def __init__(self, mu=0.5, alpha=0.3, beta=1.0):
        if mu <= 0:
            raise ValueError("mu must be positive")
        if alpha < 0:
            raise ValueError("alpha must be non-negative")
        if beta <= 0:
            raise ValueError("beta must be positive")
        self.mu = float(mu)
        self.alpha = float(alpha)
        self.beta = float(beta)

    def intensity(self, t, events):
        """Intensity at time ``t`` given the event history ``events``."""
        events = np.asarray(events, dtype=float)
        past = events[events < t]
        if past.size == 0:
            return self.mu
        return float(self.mu + self.alpha * np.exp(-self.beta * (t - past)).sum())

    def simulate(self, T, random_state=None):
        """Ogata thinning on [0, T]; returns the sorted event-time array.

        The thinning bound is exact: after the last event the intensity is
        ``mu + a`` and decays monotonically until the next event.
        """
        rng = np.random.default_rng(random_state)
        events = []
        t = 0.0
        while t < T:
            a = self.alpha * float(
                np.exp(-self.beta * (t - np.asarray(events))).sum()) \
                if events else 0.0
            rate_max = self.mu + a
            t += rng.exponential(1.0 / rate_max)
            if t > T:
                break
            a = self.alpha * float(
                np.exp(-self.beta * (t - np.asarray(events))).sum()) \
                if events else 0.0
            if rng.random() < (self.mu + a) / rate_max:
                events.append(t)
        return np.array(events)

    @staticmethod
    def _R_recursive(events, beta):
        """R_i = sum_{j < i} exp(-beta (t_i - t_j)) via the O(n) recursion."""
        events = np.asarray(events, dtype=float)
        R = np.empty(events.size)
        R[0] = 0.0
        for i in range(1, events.size):
            R[i] = np.exp(-beta * (events[i] - events[i - 1])) * (1.0 + R[i - 1])
        return R

    def _log_likelihood(self, params, events, T):
        mu, alpha, beta = params
        if mu <= 0 or alpha < 0 or beta <= 0 or alpha >= beta:
            return -np.inf
        events = np.asarray(events, dtype=float)
        if events.size < 2:
            return -np.inf
        R = self._R_recursive(events, beta)
        ll = float(np.sum(np.log(mu + alpha * R)))
        ll += -mu * T + (alpha / beta) * float(
            np.sum(np.exp(-beta * (T - events)) - 1.0))
        return ll

    def fit(self, events, T=None, x0=None):
        """Exact recursive MLE of (mu, alpha, beta) on observed event times.

        ``T`` defaults to the last event time + 1 (the observation horizon
        matters for the likelihood; pass it explicitly when known).
        Stores ``params_``, ``mu_``/``alpha_``/``beta_``, and ``loglik_``.
        """
        events = np.asarray(events, dtype=float)
        T = float(T) if T is not None else float(events[-1] + 1.0)
        if x0 is None:
            rate = events.size / T
            x0 = (max(rate, 1e-3), 0.5 * rate, 2.0 * rate)
        res = optimize.minimize(
            lambda p: -self._log_likelihood(p, events, T), x0,
            method="L-BFGS-B",
            bounds=[(1e-6, None), (0.0, None), (1e-6, None)],
            options={"maxiter": 500})
        self.params_ = tuple(float(v) for v in res.x)
        self.mu_, self.alpha_, self.beta_ = self.params_
        self.loglik_ = -float(res.fun)
        self._T = T
        self._events = events
        return self

    def branching_ratio(self):
        """Stability ratio alpha / beta (< 1 required for a stationary process)."""
        if hasattr(self, "alpha_"):
            return self.alpha_ / self.beta_
        return self.alpha / self.beta

    def ks_residuals(self, events=None, T=None):
        """Time-rescaling KS test.

        The compensator increments ``Lambda(t_i) - Lambda(t_{i-1})`` (plus the
        tail gap to T) are i.i.d. Exp(1) under the model; returns the KS
        ``(statistic, p_value)`` against that law.
        """
        if events is None:
            if not hasattr(self, "params_"):
                raise ValueError("fit first or pass events explicitly")
            mu, alpha, beta = self.params_
            T = self._T
            events = self._events
        else:
            events = np.asarray(events, dtype=float)
            mu, alpha, beta = self.mu, self.alpha, self.beta
            if T is None:
                T = float(events[-1] + 1.0)
        if events is None or events.size < 2:
            raise ValueError("at least 2 events required")
        R = self._R_recursive(events, beta)
        dts = np.diff(np.concatenate(([0.0], events)))
        comp = mu * dts + (alpha / beta) * (1.0 - np.exp(-beta * dts)) \
            * np.concatenate(([0.0], 1.0 + R[:-1]))
        tail_gap = T - events[-1]
        tail = mu * tail_gap + (alpha / beta) * (1.0 + R[-1]) \
            * (1.0 - np.exp(-beta * tail_gap))
        gaps = np.concatenate((comp, [tail]))
        stat, p = stats.kstest(gaps, "expon", args=(0, 1))
        return float(stat), float(p)


class MultivariateHawkes:
    """Multivariate Hawkes process with exponential kernels.

    ``alpha_matrix[i, j]`` is the excitation of dimension i by an event on j;
    ``beta_vec`` the per-dimension decay rates. ``simulate`` uses thinning on
    the vector intensity; ``branching_matrix`` exposes ``alpha[i, j] / beta[j]``
    (stability requires its spectral radius < 1). MLE estimation is
    deliberately out of scope (documented limitation).
    """

    def __init__(self, mu, alpha_matrix, beta_vec):
        self.mu = np.atleast_1d(np.asarray(mu, dtype=float))
        self.alpha_matrix = np.atleast_2d(np.asarray(alpha_matrix, dtype=float))
        self.beta_vec = np.atleast_1d(np.asarray(beta_vec, dtype=float))
        d = self.mu.size
        if self.alpha_matrix.shape != (d, d) or self.beta_vec.size != d:
            raise ValueError("mu, alpha_matrix, beta_vec dimension mismatch")
        if np.any(self.mu <= 0) or np.any(self.beta_vec <= 0):
            raise ValueError("mu and beta entries must be positive")
        if np.any(self.alpha_matrix < 0):
            raise ValueError("alpha entries must be non-negative")

    def branching_matrix(self):
        """Excitation matrix ``alpha[i, j] / beta[j]``; stable iff its spectral
        radius is < 1."""
        return self.alpha_matrix / self.beta_vec[None, :]

    def simulate(self, T, random_state=None):
        """Thinning simulation; returns ``(times, dims)`` sorted by time."""
        rng = np.random.default_rng(random_state)
        d = self.mu.size
        times, dims = [], []
        event_times_by_dim = [[] for _ in range(d)]

        def excitation(t):
            a = np.zeros(d)
            for j in range(d):
                if event_times_by_dim[j]:
                    taus = t - np.asarray(event_times_by_dim[j])
                    a += self.alpha_matrix[:, j] * np.exp(
                        -self.beta_vec[j] * taus).sum()
            return a

        t = 0.0
        while t < T:
            rate = self.mu + excitation(t)
            rate_max = float(rate.sum())
            t += rng.exponential(1.0 / rate_max)
            if t > T:
                break
            rate = self.mu + excitation(t)
            j = int(rng.choice(d, p=rate / rate.sum()))
            times.append(t)
            dims.append(j)
            event_times_by_dim[j].append(t)
        return np.array(times), np.array(dims, dtype=int)


class CoxProcess:
    """Cox (doubly stochastic) Poisson process: conditional on the intensity
    path, event counts are Poisson with mean the integrated intensity.

    ``intensity_fn`` is a callable t -> lambda(t) >= 0; ``simulate`` thins a
    dominating constant rate and also returns the realized integrated
    intensity ``Lambda(T)`` so callers can verify E[N] = E[Lambda].
    """

    def __init__(self, intensity_fn, rate_cap=None):
        self.intensity_fn = intensity_fn
        self.rate_cap = rate_cap

    def simulate(self, T, random_state=None, n_grid=400):
        """Thinning simulation on [0, T].

        Returns ``(events, Lambda_T)``. The cap is auto-estimated on a grid
        when not given (times a 1.2 safety factor).
        """
        rng = np.random.default_rng(random_state)
        grid = np.linspace(0.0, T, n_grid)
        vals = np.array([float(self.intensity_fn(t)) for t in grid])
        if np.any(vals < 0):
            raise ValueError("intensity must be non-negative")
        cap = self.rate_cap if self.rate_cap is not None else 1.2 * vals.max()
        if cap <= 0:
            raise ValueError("intensity must be positive somewhere on [0, T]")
        events = []
        t = 0.0
        while t < T:
            t += rng.exponential(1.0 / cap)
            if t > T:
                break
            if rng.random() < float(self.intensity_fn(t)) / cap:
                events.append(t)
        trapezoid = np.trapezoid if hasattr(np, "trapezoid") else np.trapz
        lam_T = float(trapezoid(vals, grid))
        return np.array(events), lam_T


class RenewalProcess:
    """Renewal process with i.i.d. inter-arrival times from any library
    distribution exposing ``rvs(n, random_state)``."""

    def __init__(self, interarrival_dist):
        self.dist = interarrival_dist

    def simulate(self, T, random_state=None, batch=256):
        rng = np.random.default_rng(random_state)
        arrivals = []
        t = 0.0
        while True:
            gaps = np.atleast_1d(self.dist.rvs(batch, rng))
            for g in gaps:
                t += float(g)
                if t > T:
                    return np.array(arrivals)
                arrivals.append(t)

    def renewal_function(self, T, n_paths=2000, random_state=None):
        """Monte-Carlo estimate of m(T) = E[N(T)] with standard error."""
        rng = np.random.default_rng(random_state)
        counts = np.empty(n_paths)
        for p in range(n_paths):
            counts[p] = self.simulate(T, rng).size
        return float(counts.mean()), float(counts.std(ddof=1)
                                           / np.sqrt(n_paths))


class BranchingProcess:
    """Galton-Watson branching process with a custom offspring sampler.

    ``offspring_sampler(n, rng)`` returns n offspring counts. ``simulate``
    tracks the population per generation; ``extinction_probability`` estimates
    q over many runs.
    """

    def __init__(self, offspring_sampler):
        self.offspring_sampler = offspring_sampler

    def simulate(self, generations=20, z0=1, random_state=None):
        rng = np.random.default_rng(random_state)
        pop = np.empty(generations + 1, dtype=int)
        pop[0] = int(z0)
        for g in range(generations):
            if pop[g] == 0:
                pop[g + 1:] = 0
                break
            pop[g + 1] = int(self.offspring_sampler(pop[g], rng).sum())
        return pop

    def extinction_probability(self, generations=30, n_paths=2000,
                               random_state=None):
        rng = np.random.default_rng(random_state)
        extinct = 0
        for _ in range(n_paths):
            path = self.simulate(generations, 1, rng)
            if path[-1] == 0:
                extinct += 1
        return extinct / n_paths


class SemiMarkovProcess:
    """Semi-Markov process: an embedded Markov chain with state-dependent
    holding-time distributions (library distribution objects).

    ``transition_matrix[i, j]`` is the jump probability (diagonal ignored);
    ``holding_dists[i]`` the holding law when entering state i.
    """

    def __init__(self, transition_matrix, holding_dists):
        P = np.asarray(transition_matrix, dtype=float)
        P = P - np.diag(np.diag(P))
        row = P.sum(axis=1)
        if np.any(np.abs(row - 1.0) > 1e-8):
            raise ValueError("transition rows must sum to 1 (off-diagonal)")
        if len(holding_dists) != P.shape[0]:
            raise ValueError("one holding distribution per state required")
        self.P = P
        self.holding_dists = holding_dists

    def simulate(self, T, random_state=None, x0=0):
        """Returns ``(jump_times, states)`` including the initial state."""
        rng = np.random.default_rng(random_state)
        n_states = self.P.shape[0]
        t = 0.0
        state = int(x0)
        times, states = [0.0], [state]
        while t < T:
            hold = float(self.holding_dists[state].rvs(1, rng)[0])
            t += hold
            if t > T:
                break
            state = int(rng.choice(n_states, p=self.P[state]))
            times.append(t)
            states.append(state)
        return np.array(times), np.array(states, dtype=int)


class GaussianRandomField:
    """Gaussian random field with a prescribed power spectrum via FFT synthesis.

    ``spectrum(k)`` maps wave numbers to spectral power. ``sample`` draws white
    Gaussian noise in Fourier space, shapes it by ``sqrt(spectrum)``, and
    inverse-transforms; marginals are exactly Gaussian by construction and the
    empirical periodogram matches the prescribed spectrum.
    """

    def __init__(self, spectrum, shape=(256,), length=1.0):
        self.spectrum = spectrum
        self.shape = tuple(int(s) for s in shape)
        self.length = float(length)

    def sample(self, random_state=None):
        rng = np.random.default_rng(random_state)
        noise = rng.standard_normal(self.shape)
        f = np.fft.rfftn(noise)
        # wave numbers on the rfft grid
        k_r = [2.0 * np.pi * np.fft.fftfreq(n, d=self.length / n)[:f.shape[i]]
               for i, n in enumerate(self.shape)]
        if len(self.shape) == 1:
            kr = np.abs(k_r[0])
        else:
            grids = np.meshgrid(*k_r, indexing="ij")
            kr = np.sqrt(sum(g ** 2 for g in grids))
        spec = np.vectorize(self.spectrum)(np.where(kr == 0, 1e-12, kr))
        amp = np.sqrt(spec)
        amp[tuple(0 for _ in self.shape)] = 0.0
        return np.fft.irfftn(amp * f, s=self.shape, axes=range(len(self.shape)))


class RandomMeasure:
    """Independent random measures on the real line.

    ``kind='gamma'``: Gamma(shape_per_unit * length, scale) mass over
    intervals — additive over disjoint unions. ``kind='stable'``:
    spectrally-positive alpha-stable mass with Laplace transform
    ``exp(-length * lam**alpha)``. Both delegate to the library's own
    distribution machinery.
    """

    def __init__(self, kind="gamma", shape_per_unit=2.0, scale=1.0,
                 alpha=0.5):
        if kind not in ("gamma", "stable"):
            raise ValueError("kind must be 'gamma' or 'stable'")
        self.kind = kind
        self.shape_per_unit = float(shape_per_unit)
        self.scale = float(scale)
        self.alpha = float(alpha)

    def sample(self, interval, random_state=None):
        """Realized mass of the interval ``(a, b)``."""
        a, b = interval
        length = float(b - a)
        if length < 0:
            raise ValueError("interval must satisfy a <= b")
        if length == 0:
            return 0.0
        rng = np.random.default_rng(random_state)
        if self.kind == "gamma":
            return float(rng.gamma(self.shape_per_unit * length, self.scale))
        from stochpylib.distributions import StableDistribution

        # See TemperingSubordinator's cousin StableSubordinator for why this
        # cos(pi*alpha/2)**(1/alpha) rescaling is needed: it cancels the S1
        # parameterization's extra Laplace-transform constant so the sampled
        # mass matches the documented exp(-length * lam**alpha) transform.
        norm = np.cos(np.pi * self.alpha / 2.0) ** (1.0 / self.alpha)
        s = StableDistribution(alpha=self.alpha, beta=1.0, loc=0.0,
                               scale=norm * length ** (1.0 / self.alpha))
        return float(s.rvs(1, random_state=rng))
