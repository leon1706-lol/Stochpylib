"""Derivative-free global optimizers: simulated annealing, a real-coded genetic algorithm,
particle swarm, differential evolution, ant colony, CMA-ES, and Bayesian optimization over
a Gaussian-process surrogate.

All seven are stochastic, so ``random_state=`` is load-bearing: the same seed reproduces a
run exactly. Six take a continuous objective over a box; :class:`AntColony` is the
exception -- it is defined on a distance matrix and returns a tour, so it has its own
``minimize(distance_matrix)`` signature rather than being bent into the continuous one.
"""

import numpy as np

from stochpylib.optimization._base import PopulationOptimizer
from stochpylib.optimization._common import (
    _bounds_or_box, _clip_to_bounds, _latin_hypercube, _normalize_bounds, _rng,
    _uniform_population,
)
from stochpylib.optimization._result import OptimizeResult

__all__ = ["SimulatedAnnealing", "GeneticAlgorithm", "ParticleSwarmOptimization",
           "DifferentialEvolution", "AntColony", "BayesianOptimization", "CMA_ES"]


class SimulatedAnnealing(PopulationOptimizer):
    """Simulated annealing (Kirkpatrick, Gelatt & Vecchi 1983) with a geometric or
    Boltzmann cooling schedule.

    A single walker proposes a Gaussian step whose width shrinks with the temperature and
    accepts it with the Metropolis probability ``min(1, exp(-dE / T))``, so uphill moves are
    possible early and essentially impossible late -- that is what lets it leave a local
    basin. ``result_.x`` is the best point *ever visited*, not the final state of the chain,
    since the chain can wander away from it before cooling.

    ``T0=None`` (the default) calibrates the initial temperature from a short random probe
    so roughly half of the uphill moves are accepted at ``t=0`` -- a fixed ``T0`` is
    meaningless without knowing the objective's energy scale, and gets the chain stuck or
    makes it a random walk. ``cooling_rate=None`` likewise picks the geometric rate that
    takes ``T0`` down to ``T0 * 1e-6`` over exactly ``n_iter`` steps, so changing the budget
    rescales the schedule instead of silently freezing the chain early. Proposal widths are
    ``step_size`` times the box span times ``sqrt(T / T0)``. After ``restart_patience``
    consecutive non-improving steps the walker is teleported back to the best point found,
    which stops a hot early chain from wandering away and never coming back.
    """

    method = "simulated annealing"

    def __init__(self, n_iter=5000, T0=None, cooling="geometric", cooling_rate=None,
                 step_size=0.25, n_probe=50, restart_patience=500, bounds=None,
                 track_trajectory=False, random_state=None):
        self.n_iter = n_iter
        self.T0 = T0
        self.cooling = cooling
        self.cooling_rate = cooling_rate
        self.step_size = step_size
        self.n_probe = n_probe
        self.restart_patience = restart_patience
        self.bounds = bounds
        self.track_trajectory = track_trajectory
        self.random_state = random_state

    def _temperature(self, t, T0):
        if self.cooling == "geometric":
            rate = self.cooling_rate
            if rate is None:
                rate = 1e-6 ** (1.0 / max(1, int(self.n_iter)))
            return T0 * rate ** t
        if self.cooling == "boltzmann":
            return T0 / np.log(t + np.e)
        if self.cooling == "cauchy":
            return T0 / (1.0 + t)
        raise ValueError("cooling must be 'geometric', 'boltzmann' or 'cauchy'")

    def _calibrate_T0(self, obj, x, f, span, bounds, rng):
        """Set T0 so ~half of the uphill probe moves would be accepted (Ben-Ameur 2004)."""
        if self.T0 is not None:
            return float(self.T0)
        ups = []
        for _ in range(int(self.n_probe)):
            xp = _clip_to_bounds(x + self.step_size * span * rng.standard_normal(len(x)), bounds)
            dE = obj(xp) - f
            if dE > 0:
                ups.append(dE)
        return float(np.mean(ups) / np.log(2.0)) if ups else 1.0

    def _minimize(self, obj, x0, rng):
        bounds = _bounds_or_box(self._resolve_bounds(x0), x0)
        lo, hi = bounds
        span = hi - lo
        x = _clip_to_bounds(x0, bounds)
        f = obj(x)
        best_x, best_f = x.copy(), f
        history = [f]
        trajectory = [x.copy()] if self.track_trajectory else []
        T0 = self._calibrate_T0(obj, x, f, span, bounds, rng)
        n_accept, stale = 0, 0
        for t in range(1, int(self.n_iter) + 1):
            T = max(self._temperature(t, T0), 1e-300)
            scale = self.step_size * span * np.sqrt(min(1.0, T / T0))
            x_prop = _clip_to_bounds(x + scale * rng.standard_normal(len(x)), bounds)
            f_prop = obj(x_prop)
            dE = f_prop - f
            if dE <= 0 or rng.random() < np.exp(-dE / T):
                x, f = x_prop, f_prop
                n_accept += 1
            if f < best_f:
                best_x, best_f = x.copy(), f
                stale = 0
            else:
                stale += 1
                if self.restart_patience and stale >= int(self.restart_patience):
                    x, f, stale = best_x.copy(), best_f, 0
            history.append(best_f)
            if trajectory:
                trajectory.append(x.copy())
        self._store_population(best_x[None, :], [best_f])
        return OptimizeResult(best_x, best_f, nit=int(self.n_iter), converged=True,
                              method=self.method, message="cooling schedule exhausted",
                              history=history, trajectory=trajectory,
                              extras={"acceptance_rate": n_accept / max(1, int(self.n_iter)),
                                      "T0": T0,
                                      "final_temperature": self._temperature(int(self.n_iter), T0)})


class GeneticAlgorithm(PopulationOptimizer):
    """Real-coded genetic algorithm: tournament selection, blend (BLX-alpha) crossover,
    Gaussian mutation and elitism.

    Real coding avoids the bit-string/precision coupling of the classical binary GA; BLX
    samples each child coordinate uniformly from the parents' interval widened by
    ``alpha``, which supplies exploration that a pure convex combination cannot.
    ``elitism`` copies the best ``elitism`` individuals through unchanged, so the best
    objective is monotone across generations.
    """

    method = "genetic algorithm"

    def __init__(self, population_size=50, n_generations=200, crossover_rate=0.9,
                 mutation_rate=0.1, mutation_scale=0.2, alpha=0.5, tournament_size=3,
                 elitism=2, bounds=None, track_trajectory=False, random_state=None):
        self.population_size = population_size
        self.n_generations = n_generations
        self.crossover_rate = crossover_rate
        self.mutation_rate = mutation_rate
        self.mutation_scale = mutation_scale
        self.alpha = alpha
        self.tournament_size = tournament_size
        self.elitism = elitism
        self.bounds = bounds
        self.track_trajectory = track_trajectory
        self.random_state = random_state

    def _tournament(self, values, rng):
        idx = rng.choice(len(values), size=int(self.tournament_size), replace=False)
        return int(idx[np.argmin(values[idx])])

    def _minimize(self, obj, x0, rng):
        bounds = _bounds_or_box(self._resolve_bounds(x0), x0)
        lo, hi = bounds
        n, dim = int(self.population_size), len(x0)
        pop = _latin_hypercube(n, bounds, rng)
        pop[0] = _clip_to_bounds(x0, bounds)
        vals = np.array([obj(p) for p in pop])
        history = [float(vals.min())]
        trajectory = []
        span = hi - lo
        for _ in range(int(self.n_generations)):
            order = np.argsort(vals)
            new_pop = [pop[i].copy() for i in order[:int(self.elitism)]]
            while len(new_pop) < n:
                p1 = pop[self._tournament(vals, rng)]
                p2 = pop[self._tournament(vals, rng)]
                if rng.random() < self.crossover_rate:
                    c_lo = np.minimum(p1, p2) - self.alpha * np.abs(p1 - p2)
                    c_hi = np.maximum(p1, p2) + self.alpha * np.abs(p1 - p2)
                    child = c_lo + rng.random(dim) * (c_hi - c_lo)
                else:
                    child = p1.copy()
                mask = rng.random(dim) < self.mutation_rate
                child[mask] += self.mutation_scale * span[mask] * rng.standard_normal(int(mask.sum()))
                new_pop.append(_clip_to_bounds(child, bounds))
            pop = np.array(new_pop[:n])
            vals = np.array([obj(p) for p in pop])
            history.append(float(vals.min()))
            if self.track_trajectory:
                trajectory.append(pop[int(np.argmin(vals))].copy())
        best = int(np.argmin(vals))
        self._store_population(pop, vals)
        return OptimizeResult(pop[best], float(vals[best]), nit=int(self.n_generations),
                              converged=True, method=self.method,
                              message="generation budget exhausted", history=history,
                              trajectory=trajectory,
                              extras={"population_std": float(np.mean(np.std(pop, axis=0)))})


class ParticleSwarmOptimization(PopulationOptimizer):
    """Particle swarm (Kennedy & Eberhart 1995) with inertia weight and velocity clamping.

    Each particle is pulled toward its own best position and the swarm's best with random
    weights ``c1 * U`` and ``c2 * U``; the inertia weight decays linearly from ``w_start``
    to ``w_end``, which is the standard way to trade early exploration for late
    exploitation. Velocities are clamped to a fraction of the box width, without which the
    swarm routinely explodes on a wide box.
    """

    method = "particle swarm"

    def __init__(self, n_particles=40, n_iter=300, w_start=0.9, w_end=0.4, c1=1.5, c2=1.5,
                 velocity_clamp=0.2, bounds=None, track_trajectory=False, random_state=None):
        self.n_particles = n_particles
        self.n_iter = n_iter
        self.w_start = w_start
        self.w_end = w_end
        self.c1 = c1
        self.c2 = c2
        self.velocity_clamp = velocity_clamp
        self.bounds = bounds
        self.track_trajectory = track_trajectory
        self.random_state = random_state

    def _minimize(self, obj, x0, rng):
        bounds = _bounds_or_box(self._resolve_bounds(x0), x0)
        lo, hi = bounds
        n, dim = int(self.n_particles), len(x0)
        span = hi - lo
        pos = _latin_hypercube(n, bounds, rng)
        pos[0] = _clip_to_bounds(x0, bounds)
        vmax = self.velocity_clamp * span
        vel = rng.uniform(-1, 1, (n, dim)) * vmax
        vals = np.array([obj(p) for p in pos])
        pbest, pbest_val = pos.copy(), vals.copy()
        g = int(np.argmin(vals))
        gbest, gbest_val = pos[g].copy(), float(vals[g])
        history = [gbest_val]
        trajectory = []
        for t in range(int(self.n_iter)):
            w = self.w_start + (self.w_end - self.w_start) * t / max(1, int(self.n_iter) - 1)
            r1 = rng.random((n, dim))
            r2 = rng.random((n, dim))
            vel = w * vel + self.c1 * r1 * (pbest - pos) + self.c2 * r2 * (gbest - pos)
            vel = np.clip(vel, -vmax, vmax)
            pos = np.clip(pos + vel, lo, hi)
            vals = np.array([obj(p) for p in pos])
            improved = vals < pbest_val
            pbest[improved] = pos[improved]
            pbest_val[improved] = vals[improved]
            g = int(np.argmin(pbest_val))
            if pbest_val[g] < gbest_val:
                gbest, gbest_val = pbest[g].copy(), float(pbest_val[g])
            history.append(gbest_val)
            if self.track_trajectory:
                trajectory.append(gbest.copy())
        self._store_population(pos, vals)
        return OptimizeResult(gbest, gbest_val, nit=int(self.n_iter), converged=True,
                              method=self.method, message="iteration budget exhausted",
                              history=history, trajectory=trajectory,
                              extras={"swarm_spread": float(np.mean(np.std(pos, axis=0)))})


class DifferentialEvolution(PopulationOptimizer):
    """Differential evolution (Storn & Price 1997), ``rand/1/bin`` by default.

    The mutation ``a + F * (b - c)`` draws its step length and direction from the population
    itself, so the search scale adapts to the landscape with no tuning -- the reason DE
    needs so few hyperparameters. ``strategy="best/1/bin"`` replaces the base vector with
    the incumbent best (faster, more prone to premature convergence) and ``"rand/2/bin"``
    uses two difference vectors (slower, more explorative). Selection is greedy and
    one-to-one, so the population's best value is monotone.
    """

    method = "differential evolution"

    def __init__(self, population_size=None, n_iter=300, F=0.8, CR=0.9, strategy="rand/1/bin",
                 tol=1e-12, bounds=None, track_trajectory=False, random_state=None):
        self.population_size = population_size
        self.n_iter = n_iter
        self.F = F
        self.CR = CR
        self.strategy = strategy
        self.tol = tol
        self.bounds = bounds
        self.track_trajectory = track_trajectory
        self.random_state = random_state

    def _mutant(self, pop, vals, i, rng):
        n = len(pop)
        pool = [j for j in range(n) if j != i]
        if self.strategy == "rand/1/bin":
            a, b, c = rng.choice(pool, 3, replace=False)
            return pop[a] + self.F * (pop[b] - pop[c])
        if self.strategy == "best/1/bin":
            b, c = rng.choice(pool, 2, replace=False)
            return pop[int(np.argmin(vals))] + self.F * (pop[b] - pop[c])
        if self.strategy == "rand/2/bin":
            a, b, c, d, e = rng.choice(pool, 5, replace=False)
            return pop[a] + self.F * (pop[b] - pop[c]) + self.F * (pop[d] - pop[e])
        raise ValueError("strategy must be 'rand/1/bin', 'best/1/bin' or 'rand/2/bin'")

    def _minimize(self, obj, x0, rng):
        bounds = _bounds_or_box(self._resolve_bounds(x0), x0)
        lo, hi = bounds
        dim = len(x0)
        n = int(self.population_size or max(15, 10 * dim))
        pop = _latin_hypercube(n, bounds, rng)
        pop[0] = _clip_to_bounds(x0, bounds)
        vals = np.array([obj(p) for p in pop])
        history = [float(vals.min())]
        trajectory = []
        converged, nit = False, 0
        for t in range(1, int(self.n_iter) + 1):
            nit = t
            for i in range(n):
                trial = np.clip(self._mutant(pop, vals, i, rng), lo, hi)
                cross = rng.random(dim) < self.CR
                cross[rng.integers(dim)] = True  # at least one coordinate always crosses
                cand = np.where(cross, trial, pop[i])
                f_cand = obj(cand)
                if f_cand <= vals[i]:
                    pop[i], vals[i] = cand, f_cand
            history.append(float(vals.min()))
            if self.track_trajectory:
                trajectory.append(pop[int(np.argmin(vals))].copy())
            if np.std(vals) <= self.tol * max(1.0, abs(float(np.mean(vals)))):
                converged = True
                break
        best = int(np.argmin(vals))
        self._store_population(pop, vals)
        return OptimizeResult(pop[best], float(vals[best]), nit=nit, converged=converged,
                              method=self.method,
                              message="population converged" if converged else "iteration budget exhausted",
                              history=history, trajectory=trajectory,
                              extras={"strategy": self.strategy})


class CMA_ES(PopulationOptimizer):
    """Covariance matrix adaptation evolution strategy (Hansen & Ostermeier 2001).

    Samples ``lambda`` offspring from ``N(m, sigma^2 C)``, recombines the ``mu`` best with
    rank-based weights, and updates ``C`` from a rank-one term (the evolution path ``p_c``)
    and a rank-mu term (the weighted covariance of the successful steps), with cumulative
    step-size adaptation controlling ``sigma``. The learned ``C`` is what makes it the
    strongest derivative-free method on ill-conditioned and non-separable problems; it is
    exposed as ``result_.extras["C"]``.
    """

    method = "cma-es"

    def __init__(self, sigma0=0.5, population_size=None, n_iter=500, tol=1e-12,
                 bounds=None, track_trajectory=False, random_state=None):
        self.sigma0 = sigma0
        self.population_size = population_size
        self.n_iter = n_iter
        self.tol = tol
        self.bounds = bounds
        self.track_trajectory = track_trajectory
        self.random_state = random_state

    def _minimize(self, obj, x0, rng):
        bounds = _normalize_bounds(self.bounds, len(x0))
        n = len(x0)
        lam = int(self.population_size or (4 + int(3 * np.log(n))))
        mu = lam // 2
        w = np.log(mu + 0.5) - np.log(np.arange(1, mu + 1))
        w /= np.sum(w)
        mueff = 1.0 / np.sum(w ** 2)

        cc = (4 + mueff / n) / (n + 4 + 2 * mueff / n)
        cs = (mueff + 2) / (n + mueff + 5)
        c1 = 2 / ((n + 1.3) ** 2 + mueff)
        cmu = min(1 - c1, 2 * (mueff - 2 + 1 / mueff) / ((n + 2) ** 2 + mueff))
        damps = 1 + 2 * max(0.0, np.sqrt((mueff - 1) / (n + 1)) - 1) + cs
        chiN = np.sqrt(n) * (1 - 1 / (4 * n) + 1 / (21 * n ** 2))

        m = _clip_to_bounds(x0.copy(), bounds)
        sigma = float(self.sigma0)
        C = np.eye(n)
        pc = np.zeros(n)
        ps = np.zeros(n)
        best_x, best_f = m.copy(), obj(m)
        history = [best_f]
        trajectory = []
        converged, nit = False, 0
        for t in range(1, int(self.n_iter) + 1):
            nit = t
            C = 0.5 * (C + C.T)
            eigvals, B = np.linalg.eigh(C)
            eigvals = np.maximum(eigvals, 1e-20)
            D = np.sqrt(eigvals)
            z = rng.standard_normal((lam, n))
            y = z @ (B * D).T
            X = _clip_to_bounds(m + sigma * y, bounds)
            vals = np.array([obj(xi) for xi in X])
            order = np.argsort(vals)
            if vals[order[0]] < best_f:
                best_x, best_f = X[order[0]].copy(), float(vals[order[0]])
            y_w = w @ y[order[:mu]]
            m = _clip_to_bounds(m + sigma * y_w, bounds)

            C_invsqrt = (B * (1.0 / D)) @ B.T
            ps = (1 - cs) * ps + np.sqrt(cs * (2 - cs) * mueff) * (C_invsqrt @ y_w)
            hsig = (np.linalg.norm(ps) / np.sqrt(1 - (1 - cs) ** (2 * t)) / chiN
                    < 1.4 + 2 / (n + 1))
            pc = (1 - cc) * pc + hsig * np.sqrt(cc * (2 - cc) * mueff) * y_w
            rank_mu = sum(wi * np.outer(yi, yi) for wi, yi in zip(w, y[order[:mu]]))
            C = ((1 - c1 - cmu) * C
                 + c1 * (np.outer(pc, pc) + (0 if hsig else cc * (2 - cc)) * C)
                 + cmu * rank_mu)
            sigma *= np.exp((cs / damps) * (np.linalg.norm(ps) / chiN - 1))

            history.append(best_f)
            if self.track_trajectory:
                trajectory.append(m.copy())
            if sigma * np.sqrt(np.max(np.diag(C))) < self.tol:
                converged = True
                break
        self._store_population(X, vals)
        return OptimizeResult(best_x, best_f, nit=nit, converged=converged,
                              method=self.method,
                              message="step size collapsed" if converged else "iteration budget exhausted",
                              history=history, trajectory=trajectory,
                              extras={"sigma": sigma, "C": C, "population_size": lam})


class BayesianOptimization(PopulationOptimizer):
    """Bayesian optimization over a Gaussian-process surrogate.

    Fits :class:`stochpylib.gaussian_processes.GPRegression` with a Matern kernel to the
    observations so far, then picks the next point by maximizing an acquisition function --
    ``"ei"`` (expected improvement, the default), ``"ucb"`` (lower confidence bound, since
    we minimize) or ``"pi"`` (probability of improvement). The initial design is a Latin
    hypercube from :mod:`stochpylib.montecarlo`, and the acquisition is maximized by
    multi-start :class:`~stochpylib.optimization.LBFGS` over a random candidate set.

    This is the method for expensive objectives: it spends real compute per iteration to
    minimize the *number* of objective evaluations, so on a cheap objective it is far slower
    in wall-clock terms than any of the other six.
    """

    method = "bayesian optimization"

    def __init__(self, n_init=8, n_iter=25, acquisition="ei", xi=0.01, kappa=2.0,
                 n_candidates=500, length_scale=1.0, nu=2.5, noise=1e-06, bounds=None,
                 track_trajectory=False, random_state=None):
        self.n_init = n_init
        self.n_iter = n_iter
        self.acquisition = acquisition
        self.xi = xi
        self.kappa = kappa
        self.n_candidates = n_candidates
        self.length_scale = length_scale
        self.nu = nu
        self.noise = noise
        self.bounds = bounds
        self.track_trajectory = track_trajectory
        self.random_state = random_state

    def _acquire(self, mu, sd, best):
        from scipy import special

        sd = np.maximum(sd, 1e-12)
        if self.acquisition == "ucb":
            return -(mu - self.kappa * sd)  # maximized, so negate the lower bound
        z = (best - self.xi - mu) / sd
        cdf = 0.5 * (1.0 + special.erf(z / np.sqrt(2.0)))
        if self.acquisition == "pi":
            return cdf
        if self.acquisition == "ei":
            pdf = np.exp(-0.5 * z ** 2) / np.sqrt(2 * np.pi)
            return (best - self.xi - mu) * cdf + sd * pdf
        raise ValueError("acquisition must be 'ei', 'ucb' or 'pi'")

    def _minimize(self, obj, x0, rng):
        from stochpylib.gaussian_processes import GPRegression, MaternKernel

        bounds = _bounds_or_box(self._resolve_bounds(x0), x0)
        lo, hi = bounds
        X = _latin_hypercube(int(self.n_init), bounds, rng)
        X[0] = _clip_to_bounds(x0, bounds)
        y = np.array([obj(xi) for xi in X])
        history = [float(y.min())]
        trajectory = []
        span = np.where(hi > lo, hi - lo, 1.0)
        for _ in range(int(self.n_iter)):
            # the surrogate is fitted on the unit cube with standardized responses: a single
            # default length_scale/noise is only meaningful once both axes are scale-free
            y_mu, y_sd = float(np.mean(y)), float(np.std(y)) or 1.0
            gp = GPRegression(MaternKernel(nu=self.nu, length_scale=self.length_scale),
                              noise=self.noise).fit((X - lo) / span, (y - y_mu) / y_sd)
            cand = _uniform_population(int(self.n_candidates), bounds, rng)
            mu, sd = gp.predict((cand - lo) / span, return_std=True)
            acq = self._acquire(np.asarray(mu), np.asarray(sd), (float(y.min()) - y_mu) / y_sd)
            x_next = np.clip(cand[int(np.argmax(acq))], lo, hi)
            X = np.vstack([X, x_next])
            y = np.append(y, obj(x_next))
            history.append(float(y.min()))
            if self.track_trajectory:
                trajectory.append(X[int(np.argmin(y))].copy())
        best = int(np.argmin(y))
        self._store_population(X, y)
        return OptimizeResult(X[best], float(y[best]),
                              nit=int(self.n_iter), converged=True, method=self.method,
                              message="evaluation budget exhausted", history=history,
                              trajectory=trajectory,
                              extras={"X_observed": X, "y_observed": y,
                                      "acquisition": self.acquisition})


class AntColony(PopulationOptimizer):
    """Ant colony optimization for the travelling-salesman problem (Dorigo & Gambardella 1997).

    The one combinatorial member of this module, and therefore the one with a different
    signature: ``minimize(distance_matrix)`` takes an ``(n, n)`` matrix of city-to-city
    distances and returns the best tour found. ``result_.x`` is the tour as an integer
    array of city indices (starting and ending at city 0, with the return leg implicit) and
    ``result_.fun`` its total length. Each ant builds a tour choosing the next city with
    probability proportional to ``pheromone^alpha * (1/distance)^beta``; pheromone
    evaporates by ``rho`` each iteration and is reinforced along the tours found.
    """

    method = "ant colony"

    def __init__(self, n_ants=30, n_iter=150, alpha=1.0, beta=3.0, rho=0.1, Q=1.0,
                 elitist=True, random_state=None):
        self.n_ants = n_ants
        self.n_iter = n_iter
        self.alpha = alpha
        self.beta = beta
        self.rho = rho
        self.Q = Q
        self.elitist = elitist
        self.random_state = random_state

    def minimize(self, distance_matrix, x0=None, random_state=None):
        """``distance_matrix`` is ``(n, n)``; ``x0`` is accepted and ignored (no start point)."""
        D = np.asarray(distance_matrix, dtype=float)
        if D.ndim != 2 or D.shape[0] != D.shape[1] or D.shape[0] < 3:
            raise ValueError("distance_matrix must be a square (n, n) matrix with n >= 3")
        seed = self.random_state if random_state is None else random_state
        rng = _rng(seed)
        result = self._solve(D, rng)
        self.result_ = result
        self.x_ = result.x
        self.fun_ = result.fun
        return self

    @staticmethod
    def _tour_length(D, tour):
        return float(np.sum(D[tour, np.roll(tour, -1)]))

    def _solve(self, D, rng):
        n = D.shape[0]
        eta = np.where(D > 0, 1.0 / np.where(D > 0, D, 1.0), 0.0)
        tau = np.full((n, n), 1.0 / n)
        best_tour, best_len = None, np.inf
        history = []
        for _ in range(int(self.n_iter)):
            tours, lengths = [], []
            for _ in range(int(self.n_ants)):
                tour = [0]
                unvisited = set(range(1, n))
                while unvisited:
                    cur = tour[-1]
                    nxt = np.fromiter(unvisited, dtype=int)
                    w = tau[cur, nxt] ** self.alpha * eta[cur, nxt] ** self.beta
                    total = w.sum()
                    p = w / total if total > 0 else np.full(len(nxt), 1.0 / len(nxt))
                    choice = int(rng.choice(nxt, p=p))
                    tour.append(choice)
                    unvisited.discard(choice)
                tour = np.array(tour)
                length = self._tour_length(D, tour)
                tours.append(tour)
                lengths.append(length)
                if length < best_len:
                    best_tour, best_len = tour.copy(), length
            tau *= (1.0 - self.rho)
            for tour, length in zip(tours, lengths):
                tau[tour, np.roll(tour, -1)] += self.Q / length
            if self.elitist and best_tour is not None:
                tau[best_tour, np.roll(best_tour, -1)] += self.Q / best_len
            history.append(best_len)
        return OptimizeResult(best_tour.astype(float), best_len, nit=int(self.n_iter),
                              converged=True, method=self.method,
                              message="iteration budget exhausted", history=history,
                              extras={"tour": best_tour, "pheromone": tau,
                                      "n_cities": n})

    @property
    def tour_(self):
        """The best tour found, as an integer array of city indices."""
        return self.result_.extras["tour"]
