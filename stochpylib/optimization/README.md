# stochpylib.optimization

Stochastic and numerical optimization: first-order and adaptive-step gradient methods,
quasi-Newton and trust-region second-order methods, derivative-free global metaheuristics,
stochastic approximation, and constrained solvers — 32 spec names across five submodules,
natively on numpy/scipy. Every optimizer subclasses `Optimizer`, is driven by
`minimize(fun, x0)` returning `self`, and reports a single `OptimizeResult` (point, value,
effort counters, convergence status). Objectives are wrapped in `Objective`, which supplies
finite-difference gradients and Hessians so second-order methods work without analytic
derivatives. Where an answer is itself a Monte Carlo quantity — `SAA`'s optimality gap,
`CEM`'s elite objective — it is returned as a `stochpylib.montecarlo.MCResult` rather than
as a bare number.

**Status:** implemented & tested (32/32 spec names).

## Files

- `_result.py` — `OptimizeResult`, the single result type: the solution `x`, its objective
  `fun`, the effort counters `nit`/`nfev`/`njev`, `converged` plus a human-readable
  `message`, the final gradient `jac`, `hess_inv` where a method maintains one, the
  per-iteration `history`, an opt-in `trajectory`, and a per-method `extras` dict. `float()`
  yields the objective value; `x_scalar` unwraps a one-dimensional solution.

- `_common.py` — the shared numerics. Armijo backtracking and the Nocedal & Wright
  strong-Wolfe bracket/zoom line searches, the single `_converged` stopping predicate every
  optimizer uses, bound normalization/projection, `_modified_cholesky_solve` (ridge
  inflation until the factorization succeeds, so a Newton direction descends even in
  negative curvature), and the Latin-hypercube initial design — which delegates to
  `stochpylib.montecarlo.LatinHypercubeSampling` rather than reimplementing it.
  Finite-difference derivatives and RNG plumbing are re-exported from
  `stochpylib.advanced_mcmc._common`, and the truncated-CG solve from
  `stochpylib.numerical_methods._common`.

- `_base.py` — `Objective` (callable wrapper with analytic-or-finite-difference
  gradient/Hessian, evaluation counters, `negated()`, and the convention that NaN/`-inf` map
  to `+inf` so any invalid point is simply never accepted), and the three optimizer bases:
  `Optimizer` (owns `minimize`/`maximize`, result construction and the `_minimize` hook),
  `PopulationOptimizer` (adds bounds handling and the final `population_`), and
  `ConstrainedOptimizer` (normalizes constraints and reports `violation`/`feasible`).

- `gradient.py` — `GradientDescent` (Armijo line search; the only deterministic member),
  `StochasticGD` (mini-batch with momentum/Nesterov and a decaying step), and the adaptive
  family matching its published update rule term for term: `AdaGrad` (Duchi 2011),
  `RMSProp` (Tieleman & Hinton 2012), `Adadelta` (Zeiler 2012), `AdamOptimizer`
  (Kingma & Ba 2015, with bias correction), `NADAM` (Dozat 2016) and `AMSGrad`
  (Reddi 2018). All eight share one descent loop and differ only in an `_update` hook.

- `second_order.py` — `NewtonMethod` (damped, modified-Cholesky safeguarded), `BFGS`
  (maintains the inverse Hessian, exposed as `result_.hess_inv` and usable as an asymptotic
  covariance estimate), `LBFGS` (two-loop recursion, `O(memory * dim)`),
  `ConjugateGradient` (Fletcher-Reeves or PR+ with periodic restarts), `TrustRegion`
  (dogleg or Steihaug-CG subproblem with radius adaptation), and `LevenbergMarquardt`
  (nonlinear least squares on a *residual* function, Marquardt-scaled damping). All match
  the converged solution of the corresponding `scipy.optimize.minimize`/`least_squares`
  method.

- `metaheuristic.py` — `SimulatedAnnealing` (auto-calibrated initial temperature, best-point
  restarts), `GeneticAlgorithm` (tournament selection, BLX-alpha crossover, elitism),
  `ParticleSwarmOptimization` (decaying inertia, velocity clamping), `DifferentialEvolution`
  (`rand/1/bin`, `best/1/bin`, `rand/2/bin`), `CMA_ES` (rank-one + rank-mu covariance
  adaptation with cumulative step-size control), `BayesianOptimization` (EI/UCB/PI over a
  `stochpylib.gaussian_processes.GPRegression` Matern surrogate), and `AntColony` (the one
  combinatorial method: TSP on a distance matrix).

- `stochastic_optim.py` — `StochasticApprox` (the `a_n = a/(n+A)^alpha` engine, with
  Polyak-Ruppert averaging), `RobbinsMonro` (root-finding form, `solve(g, x0)`),
  `KieferWolfowitz` (finite-difference gradient, `2 * dim` evaluations per iteration),
  `SPSA` (Rademacher perturbation, two evaluations per iteration at any dimension), `CEM`
  (cross-entropy method with a variance floor), and `SAA` (sample-average approximation with
  a replicated optimality-gap estimate).

- `constrained.py` — `PenaltyMethod` (quadratic penalty), `AugmentedLagrangian` (method of
  multipliers; the one to reach for when a *feasible* point is what you need),
  `LagrangianRelaxation` (subgradient dual ascent, reports a valid lower bound),
  `ActiveSet` (primal active-set QP solver on linear constraints), and `InteriorPoint`
  (primal log-barrier, relaxed outside the region so a trial point never yields a NaN
  gradient).

## Conventions

- **`minimize()` returns `self`.** The same shape as `.fit(data) -> self` elsewhere in the
  library: the fitted optimizer carries `result_` (the `OptimizeResult`) plus `x_`/`fun_`
  shortcuts. `maximize()` negates the objective and reports the true maximum.
- **One result type, never a parallel one.** Everything returns `OptimizeResult`; Monte
  Carlo quantities inside it (`SAA.gap_`, `CEM`'s elite estimate) are
  `stochpylib.montecarlo.MCResult`, so a point estimate always arrives with its standard
  error and confidence interval.
- **Every stochastic optimizer takes `random_state=`.** Passing the same seed reproduces a
  run exactly, on the constructor or as a `minimize()` override.
- **A failed run returns, it does not raise.** Exhausting `max_iter`, a stalled line search,
  a diverging fixed-step method and a non-finite gradient all set `converged=False` with an
  explanatory `message`, so the best point found stays inspectable. Only genuine usage
  errors (a missing `x0`, an infeasible interior-point start, an unknown strategy name)
  raise.
- **`AntColony` has a different signature on purpose.** It is combinatorial:
  `minimize(distance_matrix)` takes an `(n, n)` matrix and `tour_` is the city order. Bending
  it into `minimize(fun, x0)` would have been a worse API than documenting the exception.
- **scipy is the oracle, never the backend.** Nothing here wraps `scipy.optimize`; every
  algorithm is implemented from its published update rule, and the test suite compares
  against scipy independently.

## Known limitations

- **The metaheuristics are single-run, not restarted.** `CMA_ES`, `DifferentialEvolution`
  and `SimulatedAnnealing` regularly land in a local basin on strongly multimodal problems
  (Rastrigin in 5+ dimensions); the published restart schemes (IPOP/BIPOP-CMA) are not
  implemented, so wrap them in your own restart loop when the global optimum matters.
- **`RMSProp` settles in an `O(learning_rate)` neighbourhood.** Its step is normalized to
  roughly the learning rate regardless of gradient size, so without decay it oscillates
  around the minimum rather than converging to it. `AdaGrad`'s non-decaying accumulator has
  the opposite failure mode: it can stall before arriving.
- **`Adadelta` is slow by construction.** The update-RMS accumulator starts at zero, so the
  first steps are of order `sqrt(epsilon)`; the paper's `learning_rate=1.0` needs far more
  iterations than Adam on a well-conditioned problem.
- **`InteriorPoint` handles equalities by penalty, not by a KKT step.** They are satisfied
  only to `O(1/mu_eq)`, and `x0` must be strictly feasible for the inequalities. Use
  `AugmentedLagrangian` when equalities dominate.
- **`LagrangianRelaxation` does not guarantee feasibility.** It returns the least-infeasible
  primal iterate (the usual recovery heuristic); its genuine output is the dual lower bound
  in `extras["dual_bound"]`. Its `ctol` default is correspondingly looser (`1e-4`), because
  subgradient dual ascent converges at `O(1/k)`.
- **`ActiveSet` is a QP solver.** It builds `G`/`c` from the Hessian and gradient at `x0`, so
  a non-quadratic objective is solved as its second-order model there, not globally.
- **`BayesianOptimization` maximizes its acquisition by random candidate search.** A
  `n_candidates` uniform sample is scored rather than the acquisition being optimized
  properly, which is adequate in low dimension and degrades above roughly ten.
- **No autodiff.** Missing gradients and Hessians are finite differences, so derivative-free
  Hessians cost `O(dim^2)` evaluations and carry `~1e-4` accuracy; supply `grad=`/`hess=`
  whenever you have them.

Spec: vault `Modules/optimization.md` (private). Tests: `tests/optimization/tests.py`
(oracles: `scipy.optimize`, closed-form quadratic/constrained solutions, published global
minima of the standard test functions, brute-force TSP enumeration, and the library's own
`statistics`/`distributions` MLE fits) and `tests/optimization/e2e.py` (API sweep).
