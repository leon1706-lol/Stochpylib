"""Stochastic approximation and sample-based optimization: the Robbins-Monro root-finding
recursion and its Kiefer-Wolfowitz and SPSA gradient-free descendants, the cross-entropy
method, and sample-average approximation.

These are the methods for objectives you can only observe *with noise*: none of them
assumes a deterministic evaluation, and all of them take ``random_state=``. Where the
answer is itself a Monte Carlo quantity -- ``SAA``'s optimality gap, ``CEM``'s final
objective estimate -- it is reported as a :class:`stochpylib.montecarlo.MCResult`, so the
point estimate always arrives with its standard error and confidence interval.
"""

import numpy as np

from stochpylib.montecarlo import MCResult
from stochpylib.optimization._base import Optimizer, PopulationOptimizer
from stochpylib.optimization._common import _clip_to_bounds, _normalize_bounds, _rng
from stochpylib.optimization._result import OptimizeResult

__all__ = ["SAA", "StochasticApprox", "RobbinsMonro", "KieferWolfowitz", "SPSA", "CEM"]


class StochasticApprox(Optimizer):
    """Generic stochastic approximation: ``x_{n+1} = x_n - a_n * H(x_n, noise)``.

    The engine the rest of this file specializes. ``H`` is whatever noisy direction the
    caller supplies via ``minimize(fun, x0, direction=...)``; by default it is the
    objective's gradient, so with a noiseless objective this is just decaying-step gradient
    descent. The step sequence is the standard ``a_n = a / (n + A)^alpha`` with
    ``alpha in (0.5, 1]`` -- the range for which ``sum a_n = inf`` and ``sum a_n^2 < inf``,
    which is exactly what the convergence theorem needs.

    ``averaging=True`` returns the Polyak-Ruppert average of the iterates rather than the
    last one, which attains the optimal asymptotic rate for ``alpha < 1``.
    """

    method = "stochastic approximation"

    def __init__(self, a=1.0, A=0.0, alpha=0.602, n_iter=2000, averaging=False,
                 average_from=0.5, bounds=None, track_trajectory=False, random_state=None):
        self.a = a
        self.A = A
        self.alpha = alpha
        self.n_iter = n_iter
        self.averaging = averaging
        self.average_from = average_from
        self.bounds = bounds
        self.track_trajectory = track_trajectory
        self.random_state = random_state

    def _step_size(self, n):
        return self.a / (n + self.A) ** self.alpha

    def _direction(self, obj, x, n, rng):
        return obj.grad(x)

    def minimize(self, fun, x0=None, grad=None, hess=None, args=(), direction=None,
                 random_state=None):
        """``direction(x, n, rng) -> (dim,)`` overrides the default noisy-gradient step."""
        self._direction_fn = direction
        return super().minimize(fun, x0, grad, hess, args, random_state)

    def _minimize(self, obj, x0, rng):
        bounds = _normalize_bounds(getattr(self, "bounds", None), len(x0))
        x = _clip_to_bounds(x0, bounds)
        history = [obj(x)]
        trajectory = [x.copy()] if self.track_trajectory else []
        iterates = [x.copy()]
        fn = getattr(self, "_direction_fn", None)
        direction_evals = 0
        for n in range(1, int(self.n_iter) + 1):
            before = obj.n_evals
            d = fn(x, n, rng) if fn is not None else self._direction(obj, x, n, rng)
            direction_evals += obj.n_evals - before
            x = _clip_to_bounds(x - self._step_size(n) * np.asarray(d, dtype=float), bounds)
            iterates.append(x.copy())
            history.append(obj(x))
            if trajectory:
                trajectory.append(x.copy())
        x_final = x
        if self.averaging:
            start = int(self.average_from * len(iterates))
            x_final = np.mean(np.array(iterates[start:]), axis=0)
        return OptimizeResult(x_final, obj(x_final), nit=int(self.n_iter), converged=True,
                              method=self.method, message="step budget exhausted",
                              history=history, trajectory=trajectory,
                              extras={"final_step_size": self._step_size(int(self.n_iter)),
                                      "averaged": bool(self.averaging),
                                      "direction_evals": direction_evals})


class RobbinsMonro(StochasticApprox):
    """The Robbins-Monro (1951) recursion for solving ``E[g(x, noise)] = target``.

    Root finding, not minimization: ``minimize`` is inherited but the natural entry point is
    :meth:`solve`, which takes a noisy observation callable ``g(x, rng)`` and drives its
    expectation to ``target``. Applied to a gradient observation it *is* stochastic gradient
    descent; the distinction matters because the classical convergence and asymptotic
    normality results (rate ``n^(-alpha/2)``, or ``n^(-1/2)`` with Polyak-Ruppert
    averaging) are stated for the root-finding form.
    """

    method = "robbins-monro"

    def __init__(self, a=1.0, A=0.0, alpha=1.0, target=0.0, n_iter=2000, averaging=False,
                 average_from=0.5, bounds=None, track_trajectory=False, random_state=None):
        super().__init__(a=a, A=A, alpha=alpha, n_iter=n_iter, averaging=averaging,
                         average_from=average_from, bounds=bounds,
                         track_trajectory=track_trajectory, random_state=random_state)
        self.target = target

    def solve(self, g, x0, random_state=None):
        """``g(x, rng) -> float or (dim,)``; drives ``E[g(x)]`` to ``target``."""
        seed = self.random_state if random_state is None else random_state
        rng = _rng(seed)
        bounds = _normalize_bounds(self.bounds, len(np.atleast_1d(x0)))
        x = _clip_to_bounds(np.atleast_1d(np.asarray(x0, dtype=float)).copy(), bounds)
        iterates, history = [x.copy()], []
        for n in range(1, int(self.n_iter) + 1):
            obs = np.atleast_1d(np.asarray(g(x, rng), dtype=float)) - self.target
            x = _clip_to_bounds(x - self._step_size(n) * obs, bounds)
            iterates.append(x.copy())
            history.append(float(np.max(np.abs(obs))))
        x_final = x
        if self.averaging:
            start = int(self.average_from * len(iterates))
            x_final = np.mean(np.array(iterates[start:]), axis=0)
        self.result_ = OptimizeResult(x_final, history[-1] if history else 0.0,
                                      nit=int(self.n_iter), converged=True,
                                      method=self.method, message="step budget exhausted",
                                      history=history,
                                      extras={"root": x_final, "target": self.target})
        self.x_ = self.result_.x
        self.fun_ = self.result_.fun
        return self

    @property
    def root_(self):
        """The recovered root of ``E[g(x)] = target``."""
        return self.result_.extras["root"]


class KieferWolfowitz(StochasticApprox):
    """Kiefer-Wolfowitz (1952): stochastic approximation with a *finite-difference* gradient.

    Uses ``2 * dim`` noisy objective evaluations per iteration to build a central-difference
    gradient with a shrinking perturbation ``c_n = c / n^gamma``, then takes a Robbins-Monro
    step. The ``2 * dim`` cost per iteration is precisely what :class:`SPSA` removes.
    """

    method = "kiefer-wolfowitz"

    def __init__(self, a=0.5, c=0.1, A=0.0, alpha=0.602, gamma=0.101, n_iter=1000,
                 averaging=False, average_from=0.5, bounds=None, track_trajectory=False,
                 random_state=None):
        super().__init__(a=a, A=A, alpha=alpha, n_iter=n_iter, averaging=averaging,
                         average_from=average_from, bounds=bounds,
                         track_trajectory=track_trajectory, random_state=random_state)
        self.c = c
        self.gamma = gamma

    def _perturbation(self, n):
        return self.c / n ** self.gamma

    def _direction(self, obj, x, n, rng):
        c_n = self._perturbation(n)
        g = np.empty(len(x))
        for i in range(len(x)):
            e = np.zeros(len(x))
            e[i] = c_n
            g[i] = (obj(x + e) - obj(x - e)) / (2 * c_n)
        return g


class SPSA(StochasticApprox):
    """Simultaneous perturbation stochastic approximation (Spall 1992).

    Perturbs *every* coordinate at once with an independent Rademacher vector, so the
    gradient estimate costs **two objective evaluations per iteration regardless of
    dimension** -- the same asymptotic convergence rate as Kiefer-Wolfowitz at ``1/dim`` of
    the cost. Defaults follow Spall's practical guidelines (``alpha=0.602``,
    ``gamma=0.101``, ``A`` around 10% of the iteration budget).
    """

    method = "spsa"

    def __init__(self, a=0.2, c=0.1, A=None, alpha=0.602, gamma=0.101, n_iter=1000,
                 averaging=False, average_from=0.5, bounds=None, track_trajectory=False,
                 random_state=None):
        super().__init__(a=a, A=A, alpha=alpha, n_iter=n_iter, averaging=averaging,
                         average_from=average_from, bounds=bounds,
                         track_trajectory=track_trajectory, random_state=random_state)
        self.c = c
        self.gamma = gamma

    def _step_size(self, n):
        A = 0.1 * int(self.n_iter) if self.A is None else self.A
        return self.a / (n + A) ** self.alpha

    def _perturbation(self, n):
        return self.c / n ** self.gamma

    def _direction(self, obj, x, n, rng):
        c_n = self._perturbation(n)
        delta = rng.choice([-1.0, 1.0], size=len(x))  # Rademacher, as Spall requires
        diff = obj(x + c_n * delta) - obj(x - c_n * delta)
        return diff / (2 * c_n) / delta


class CEM(PopulationOptimizer):
    """The cross-entropy method (Rubinstein 1997) with a diagonal Gaussian sampling density.

    Each iteration samples ``population_size`` points from ``N(mu, diag(sigma^2))``, keeps
    the ``elite_frac`` best, and moves ``mu``/``sigma`` toward that elite set's moments with
    smoothing factor ``smoothing``. ``extra_variance`` adds a decaying floor to ``sigma``,
    without which the sampling density collapses onto the first good region found -- the
    classic failure mode of CEM. ``result_.extras["estimate"]`` is an
    :class:`stochpylib.montecarlo.MCResult` for the final elite objective, since that value
    is a Monte Carlo average, not an exact evaluation.
    """

    method = "cross-entropy method"

    def __init__(self, population_size=100, n_iter=100, elite_frac=0.2, smoothing=0.7,
                 sigma0=1.0, extra_variance=0.1, variance_decay=0.95, tol=1e-10,
                 bounds=None, track_trajectory=False, random_state=None):
        self.population_size = population_size
        self.n_iter = n_iter
        self.elite_frac = elite_frac
        self.smoothing = smoothing
        self.sigma0 = sigma0
        self.extra_variance = extra_variance
        self.variance_decay = variance_decay
        self.tol = tol
        self.bounds = bounds
        self.track_trajectory = track_trajectory
        self.random_state = random_state

    def _minimize(self, obj, x0, rng):
        bounds = _normalize_bounds(getattr(self, "bounds", None), len(x0))
        n = int(self.population_size)
        n_elite = max(2, int(round(self.elite_frac * n)))
        mu = _clip_to_bounds(x0.copy(), bounds)
        sigma = np.full(len(x0), float(self.sigma0))
        history, trajectory = [obj(mu)], []
        best_x, best_f = mu.copy(), history[0]
        converged, nit = False, 0
        elite_vals = np.array([best_f])
        for t in range(1, int(self.n_iter) + 1):
            nit = t
            X = _clip_to_bounds(mu + sigma * rng.standard_normal((n, len(mu))), bounds)
            vals = np.array([obj(xi) for xi in X])
            elite = X[np.argsort(vals)[:n_elite]]
            elite_vals = np.sort(vals)[:n_elite]
            if elite_vals[0] < best_f:
                best_x, best_f = elite[0].copy(), float(elite_vals[0])
            mu = self.smoothing * elite.mean(axis=0) + (1 - self.smoothing) * mu
            floor = self.extra_variance * self.variance_decay ** t
            sigma = (self.smoothing * elite.std(axis=0) + (1 - self.smoothing) * sigma) + floor
            history.append(best_f)
            if self.track_trajectory:
                trajectory.append(mu.copy())
            if np.max(sigma) < self.tol:
                converged = True
                break
        self._store_population(elite, elite_vals)
        estimate = MCResult(float(np.mean(elite_vals)),
                            float(np.std(elite_vals, ddof=1) / np.sqrt(len(elite_vals)))
                            if len(elite_vals) > 1 else float("nan"),
                            len(elite_vals), "cross-entropy elite mean")
        return OptimizeResult(best_x, best_f, nit=nit, converged=converged,
                              method=self.method,
                              message="sampling density collapsed" if converged else "iteration budget exhausted",
                              history=history, trajectory=trajectory,
                              extras={"mu": mu, "sigma": sigma, "estimate": estimate})


class SAA(Optimizer):
    """Sample-average approximation: replace ``E[F(x, xi)]`` by a fixed-sample average and
    solve that deterministic problem.

    ``minimize(F, x0, sampler=...)`` draws ``n_samples`` scenarios once, builds
    ``(1/N) sum F(x, xi_i)`` and hands it to ``inner`` (an :class:`Optimizer`, L-BFGS by
    default). Because the scenarios are fixed, the surrogate is deterministic and any
    deterministic optimizer applies -- that is the whole point of SAA over stochastic
    approximation.

    The statistically interesting output is the **optimality gap**: the SAA solution's true
    expected cost minus the optimal value is bounded in expectation by the gap estimated
    from ``n_batches`` independent replications (Mak, Morton & Wood 1999). It is reported as
    an :class:`stochpylib.montecarlo.MCResult` in ``result_.extras["gap"]``, with the
    standard error and confidence interval the library's estimator contract requires.
    """

    method = "sample-average approximation"

    def __init__(self, n_samples=500, n_batches=10, batch_size=100, inner=None,
                 bounds=None, random_state=None):
        self.n_samples = n_samples
        self.n_batches = n_batches
        self.batch_size = batch_size
        self.inner = inner
        self.bounds = bounds
        self.random_state = random_state

    def minimize(self, F, x0=None, sampler=None, args=(), random_state=None):
        """``F(x, xi) -> float`` is the per-scenario cost; ``sampler(size, rng) -> scenarios``."""
        if sampler is None:
            raise ValueError("SAA requires sampler(size, rng) returning scenarios")
        self._F = F
        self._sampler = sampler
        return super().minimize(lambda x: 0.0, x0, args=args, random_state=random_state)

    def _solve_batch(self, x0, scenarios, bounds):
        from stochpylib.optimization.second_order import LBFGS

        inner = self.inner if self.inner is not None else LBFGS(max_iter=300)
        mean_cost = lambda x: float(np.mean([self._F(x, s) for s in scenarios]))
        fitted = inner.minimize(mean_cost, x0)
        x = _clip_to_bounds(fitted.result_.x, bounds)
        return x, mean_cost

    def _minimize(self, obj, x0, rng):
        bounds = _normalize_bounds(self.bounds, len(x0))
        scenarios = self._sampler(int(self.n_samples), rng)
        x_hat, mean_cost = self._solve_batch(x0, scenarios, bounds)
        f_hat = mean_cost(x_hat)

        # gap replications: each batch gives (cost of x_hat) - (optimal cost) on its own
        # scenarios, an upward-biased but consistent estimator of the true optimality gap
        gaps = []
        for _ in range(int(self.n_batches)):
            batch = self._sampler(int(self.batch_size), rng)
            x_b, cost_b = self._solve_batch(x_hat, batch, bounds)
            gaps.append(cost_b(x_hat) - cost_b(x_b))
        gaps = np.asarray(gaps, dtype=float)
        gap = MCResult(float(np.mean(gaps)),
                       float(np.std(gaps, ddof=1) / np.sqrt(len(gaps))) if len(gaps) > 1
                       else float("nan"),
                       len(gaps), "SAA optimality gap")
        return OptimizeResult(x_hat, f_hat, nit=1 + int(self.n_batches), converged=True,
                              method=self.method, message="SAA problem solved",
                              history=[f_hat],
                              extras={"gap": gap, "n_scenarios": int(self.n_samples),
                                      "gap_replications": gaps})

    @property
    def gap_(self):
        """The estimated optimality gap as an ``MCResult`` (estimate + SE + CI)."""
        return self.result_.extras["gap"]
