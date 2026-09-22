"""Tests for stochpylib.optimization: first-order and adaptive-step gradient methods,
quasi-Newton/trust-region/Levenberg-Marquardt second-order methods, derivative-free global
metaheuristics, stochastic approximation and sample-average approximation, and constrained
solvers.

Oracles: scipy.optimize (minimize with BFGS/L-BFGS-B/CG/Newton-CG/trust-ncg/SLSQP,
least_squares, differential_evolution, dual_annealing, linprog, rosen/rosen_der/rosen_hess),
closed-form solutions of quadratic and equality-constrained problems, published global
minima of the standard Rastrigin/Ackley/Sphere/Beale/Himmelblau test functions, brute-force
enumeration of small TSP instances, and the library's own
stochpylib.statistics/stochpylib.distributions MLE fits as independent answers to the same
estimation problem. All randomness is seeded; statistical assertions are set at >= 3 Monte
Carlo standard errors.
"""

import itertools

import numpy as np
import pytest
from scipy import optimize as sopt

from stochpylib import distributions, montecarlo, statistics
from stochpylib.optimization import (
    AMSGrad, ActiveSet, AdaGrad, Adadelta, AdamOptimizer, AntColony, AugmentedLagrangian,
    BFGS, BayesianOptimization, CEM, CMA_ES, ConjugateGradient, ConstrainedOptimizer,
    DifferentialEvolution, GeneticAlgorithm, GradientDescent, InteriorPoint,
    KieferWolfowitz, LBFGS, LagrangianRelaxation, LevenbergMarquardt, NADAM, NewtonMethod,
    Objective, OptimizeResult, Optimizer, ParticleSwarmOptimization, PenaltyMethod,
    PopulationOptimizer, RMSProp, RobbinsMonro, SAA, SPSA, SimulatedAnnealing,
    StochasticApprox, StochasticGD, TrustRegion,
)
from stochpylib.optimization._common import (
    _backtracking_armijo, _converged, _modified_cholesky_solve, _normalize_bounds,
    _strong_wolfe,
)
from stochpylib.optimization.constrained import _relaxed_log

_RNG = np.random.default_rng(0)

# a well-conditioned quadratic with a closed-form minimizer, used throughout
_A = np.array([[3.0, 1.0], [1.0, 2.0]])
_B = np.array([1.0, -1.0])
_STAR = np.linalg.solve(_A, _B)


def _quad(x):
    return 0.5 * float(x @ _A @ x) - float(_B @ x)


def _quad_grad(x):
    return _A @ x - _B


def _quad_hess(x):
    return _A


def rastrigin(x):
    x = np.asarray(x, dtype=float)
    return float(10 * len(x) + np.sum(x ** 2 - 10 * np.cos(2 * np.pi * x)))


def ackley(x):
    x = np.asarray(x, dtype=float)
    n = len(x)
    return float(-20 * np.exp(-0.2 * np.sqrt(np.sum(x ** 2) / n))
                 - np.exp(np.sum(np.cos(2 * np.pi * x)) / n) + 20 + np.e)


def sphere(x):
    return float(np.sum(np.asarray(x, dtype=float) ** 2))


def beale(x):
    # global minimum f(3, 0.5) = 0
    a, b = float(x[0]), float(x[1])
    return ((1.5 - a + a * b) ** 2 + (2.25 - a + a * b ** 2) ** 2
            + (2.625 - a + a * b ** 3) ** 2)


# ========================================================================= objective

def test_objective_finite_difference_gradient_matches_analytic():
    fd = Objective(sopt.rosen)
    exact = Objective(sopt.rosen, sopt.rosen_der)
    for x in ([0.7, 1.3], [-1.2, 1.0], [2.0, 2.0]):
        x = np.asarray(x, dtype=float)
        assert np.max(np.abs(fd.grad(x) - exact.grad(x))) < 1e-6


def test_objective_finite_difference_hessian_matches_analytic():
    exact = Objective(sopt.rosen, sopt.rosen_der, sopt.rosen_hess)
    from_grad = Objective(sopt.rosen, sopt.rosen_der)
    from_fun = Objective(sopt.rosen)
    x = np.array([0.7, 1.3])
    # differencing an analytic gradient is an order of magnitude tighter than
    # differencing the objective twice -- that is why hess() prefers it
    assert np.max(np.abs(from_grad.hess(x) - exact.hess(x))) < 1e-6
    assert np.max(np.abs(from_fun.hess(x) - exact.hess(x))) < 1e-4


def test_objective_maps_invalid_values_to_positive_infinity():
    obj = Objective(lambda x: np.nan)
    assert obj(np.zeros(2)) == np.inf
    assert Objective(lambda x: -np.inf)(np.zeros(2)) == np.inf
    assert Objective(lambda x: 1.0 / 0.0 if False else float("inf"))(np.zeros(1)) == np.inf

    def blows_up(x):
        raise ValueError("outside the domain")

    assert Objective(blows_up)(np.zeros(1)) == np.inf


def test_objective_counts_evaluations_and_rewraps_idempotently():
    obj = Objective(_quad, _quad_grad)
    obj(np.zeros(2))
    obj(np.ones(2))
    obj.grad(np.zeros(2))
    assert (obj.n_evals, obj.n_grad_evals) == (2, 1)
    rewrapped = Objective(obj)
    assert rewrapped.has_gradient and rewrapped.fun is obj.fun
    assert rewrapped.n_evals == 0  # counters are per wrapper, not shared


def test_objective_negated_flips_value_gradient_and_hessian():
    obj = Objective(_quad, _quad_grad, _quad_hess)
    neg = obj.negated()
    x = np.array([0.3, -0.7])
    assert neg(x) == pytest.approx(-obj(x))
    assert np.allclose(neg.grad(x), -obj.grad(x))
    assert np.allclose(neg.hess(x), -obj.hess(x))


def test_optimizer_maximize_reports_the_true_maximum():
    f = lambda z: -float((z[0] - 2) ** 2 + (z[1] + 1) ** 2)
    opt = BFGS().maximize(f, [0.0, 0.0])
    assert np.allclose(opt.x_, [2.0, -1.0], atol=1e-6)
    assert opt.fun_ == pytest.approx(0.0, abs=1e-12)
    assert opt.fun_ == pytest.approx(f(opt.x_), abs=1e-12)


def test_optimizer_repr_and_result_accessors():
    opt = BFGS()
    assert repr(opt) == "BFGS(unfitted)"
    opt.minimize(_quad, [3.0, 3.0], grad=_quad_grad)
    assert "BFGS(fun=" in repr(opt)
    assert opt.to_result() is opt.result_
    assert isinstance(opt.result_, OptimizeResult)
    assert float(opt.result_) == pytest.approx(opt.fun_)
    assert isinstance(opt, Optimizer)


def test_optimize_result_scalar_accessor_and_repr():
    res = BFGS().minimize(lambda x: float((x[0] - 4.0) ** 2), [0.0]).result_
    assert res.x_scalar == pytest.approx(4.0, abs=1e-6)
    assert "OptimizeResult(fun=" in repr(res)
    two_d = BFGS().minimize(_quad, [1.0, 1.0], grad=_quad_grad).result_
    with pytest.raises(ValueError):
        two_d.x_scalar


def test_minimize_without_x0_is_a_usage_error():
    with pytest.raises(ValueError):
        BFGS().minimize(_quad)
    with pytest.raises(ValueError):
        CMA_ES().minimize(sphere)  # population methods need x0 or per-coordinate bounds


# ========================================================================== gradient

_GRADIENT_CLASSES = [GradientDescent, StochasticGD, AdaGrad, RMSProp, Adadelta,
                     AdamOptimizer, NADAM, AMSGrad]


@pytest.mark.parametrize("cls", _GRADIENT_CLASSES)
def test_every_gradient_method_solves_the_closed_form_quadratic(cls):
    kw = {"max_iter": 20000}
    if cls is Adadelta:
        kw["learning_rate"] = 5.0  # the paper's 1.0 has a very slow start (see its docstring)
    opt = cls(**kw).minimize(_quad, [5.0, 5.0], grad=_quad_grad)
    # RMSProp normalizes the step to ~learning_rate regardless of gradient size, so it
    # settles in an O(learning_rate) neighbourhood rather than at the minimum
    tol = 1e-2 if cls is RMSProp else 1e-4
    assert np.max(np.abs(opt.x_ - _STAR)) < tol


def test_gradient_descent_line_search_matches_scipy_on_rosenbrock_objective():
    opt = GradientDescent(max_iter=20000).minimize(sopt.rosen, [-1.2, 1.0],
                                                   grad=sopt.rosen_der)
    ref = sopt.minimize(sopt.rosen, [-1.2, 1.0], jac=sopt.rosen_der, method="BFGS")
    # steepest descent crawls along Rosenbrock's valley: it gets the objective to ~1e-8
    # but needs thousands of iterations to match a quasi-Newton method's x
    assert np.allclose(opt.x_, [1.0, 1.0], atol=1e-3)
    assert np.allclose(opt.x_, ref.x, atol=1e-3)
    assert opt.fun_ < 1e-6
    assert opt.result_.nit > 100


def test_gradient_descent_without_line_search_diverges_and_says_so():
    opt = GradientDescent(learning_rate=10.0, line_search=False,
                          max_iter=200).minimize(_quad, [5.0, 5.0], grad=_quad_grad)
    assert not opt.result_.converged
    assert opt.result_.message == "objective diverged"


def test_amsgrad_equals_adam_with_amsgrad_flag():
    a = AMSGrad(max_iter=300, learning_rate=0.05).minimize(_quad, [5.0, 5.0], grad=_quad_grad)
    b = AdamOptimizer(max_iter=300, learning_rate=0.05, amsgrad=True).minimize(
        _quad, [5.0, 5.0], grad=_quad_grad)
    assert np.allclose(a.x_, b.x_, atol=1e-12)


def test_adam_bias_correction_makes_the_first_step_full_sized():
    # without bias correction m1/sqrt(v1) would be ~beta-scaled; with it the first step
    # is almost exactly the learning rate in magnitude, coordinate-wise
    lr = 0.05
    opt = AdamOptimizer(learning_rate=lr, max_iter=1, track_trajectory=True).minimize(
        _quad, [5.0, 5.0], grad=_quad_grad)
    first_step = np.abs(opt.result_.trajectory[1] - opt.result_.trajectory[0])
    assert np.allclose(first_step, lr, rtol=1e-6)


def test_stochastic_gd_minibatch_gradient_recovers_the_ols_solution():
    rng = np.random.default_rng(42)
    n, p = 300, 3
    X = rng.normal(size=(n, p))
    beta = np.array([1.5, -2.0, 0.5])
    y = X @ beta + rng.normal(0, 0.2, n)

    def grad_batch(b, idx):
        r = X[idx] @ b - y[idx]
        return X[idx].T @ r / len(idx)

    opt = StochasticGD(learning_rate=0.1, batch_size=32, n_samples=n, decay=0.01,
                       grad_sample=grad_batch, momentum=0.9, max_iter=4000,
                       random_state=1).minimize(
        lambda b: float(np.mean((X @ b - y) ** 2) / 2), np.zeros(p))
    ols = statistics.linear_regression(X, y, fit_intercept=False)
    assert np.max(np.abs(opt.x_ - ols.coef_)) < 0.02


def test_stochastic_gd_same_seed_is_reproducible_and_different_seeds_are_not():
    rng = np.random.default_rng(11)
    X = rng.normal(size=(200, 2))
    y = X @ np.array([1.0, -1.0]) + rng.normal(0, 1.0, 200)
    grad_batch = lambda b, idx: X[idx].T @ (X[idx] @ b - y[idx]) / len(idx)
    kw = dict(learning_rate=0.05, batch_size=8, n_samples=200, grad_sample=grad_batch,
              max_iter=100)
    f = lambda b: float(np.mean((X @ b - y) ** 2) / 2)
    a = StochasticGD(random_state=5, **kw).minimize(f, np.zeros(2))
    b = StochasticGD(random_state=5, **kw).minimize(f, np.zeros(2))
    c = StochasticGD(random_state=6, **kw).minimize(f, np.zeros(2))
    assert np.array_equal(a.x_, b.x_)
    assert not np.array_equal(a.x_, c.x_)


def test_bounds_are_respected_by_every_gradient_method():
    bounds = [(-0.25, 0.25), (-0.25, 0.25)]
    for cls in _GRADIENT_CLASSES:
        opt = cls(bounds=bounds, max_iter=500).minimize(_quad, [0.0, 0.0], grad=_quad_grad)
        assert np.all(opt.x_ >= -0.25 - 1e-12) and np.all(opt.x_ <= 0.25 + 1e-12)


def test_history_and_trajectory_lengths_agree_with_iteration_count():
    opt = AdamOptimizer(max_iter=50, gtol=0.0, ftol=0.0, xtol=0.0,
                        track_trajectory=True).minimize(_quad, [5.0, 5.0], grad=_quad_grad)
    assert len(opt.result_.history) == opt.result_.nit + 1
    assert len(opt.result_.trajectory) == opt.result_.nit + 1
    # trajectory is opt-in: it is O(nit * dim)
    assert AdamOptimizer(max_iter=10).minimize(_quad, [5.0, 5.0],
                                               grad=_quad_grad).result_.trajectory == []


# ====================================================================== second order

@pytest.mark.parametrize("cls", [NewtonMethod, BFGS, LBFGS, ConjugateGradient, TrustRegion])
def test_second_order_methods_match_scipy_minimize_on_rosenbrock(cls):
    opt = cls().minimize(sopt.rosen, [-1.2, 1.0], grad=sopt.rosen_der, hess=sopt.rosen_hess)
    ref = sopt.minimize(sopt.rosen, [-1.2, 1.0], jac=sopt.rosen_der, method="BFGS",
                        options={"gtol": 1e-10})
    assert np.allclose(opt.x_, ref.x, atol=1e-5)
    assert np.allclose(opt.x_, [1.0, 1.0], atol=1e-5)
    assert opt.result_.converged


@pytest.mark.parametrize("cls", [NewtonMethod, BFGS, LBFGS, ConjugateGradient, TrustRegion])
def test_second_order_methods_solve_the_quadratic_exactly(cls):
    opt = cls().minimize(_quad, [7.0, -4.0], grad=_quad_grad, hess=_quad_hess)
    # ConjugateGradient stops on the shared ftol rather than on an exact-arithmetic
    # termination, so it lands a few ulps looser than the Hessian-based methods
    tol = 1e-7 if cls is ConjugateGradient else 1e-9
    assert np.max(np.abs(opt.x_ - _STAR)) < tol


@pytest.mark.parametrize("cls", [NewtonMethod, BFGS, LBFGS, TrustRegion])
def test_second_order_methods_work_without_analytic_derivatives(cls):
    opt = cls().minimize(_quad, [7.0, -4.0])
    assert np.max(np.abs(opt.x_ - _STAR)) < 1e-5


def test_bfgs_inverse_hessian_estimates_the_true_inverse_hessian():
    opt = BFGS().minimize(_quad, [7.0, -4.0], grad=_quad_grad)
    exact = np.linalg.inv(_A)
    # the secant updates approximate the inverse Hessian; they equal it only in exact
    # arithmetic after dim independent curvature pairs, so this is a relative claim
    assert np.max(np.abs(opt.result_.hess_inv - exact)) / np.max(np.abs(exact)) < 0.01


def test_bfgs_hess_inv_reproduces_logistic_regression_standard_errors():
    rng = np.random.default_rng(7)
    n, p = 500, 3
    X = rng.normal(size=(n, p))
    Xd = np.column_stack([np.ones(n), X])
    beta = np.array([0.5, -1.2, 0.8, 0.3])
    y = (rng.random(n) < 1.0 / (1.0 + np.exp(-Xd @ beta))).astype(float)
    nll = lambda b: float(np.sum(np.logaddexp(0.0, Xd @ b) - y * (Xd @ b)))
    nll_grad = lambda b: Xd.T @ (1.0 / (1.0 + np.exp(-Xd @ b)) - y)
    opt = BFGS().minimize(nll, np.zeros(p + 1), grad=nll_grad)
    ref = statistics.logistic_regression(X, y)
    assert np.max(np.abs(opt.x_ - ref.coef_)) < 1e-6
    se = np.sqrt(np.diag(opt.result_.hess_inv))
    assert np.max(np.abs(se - ref.std_errors_) / ref.std_errors_) < 0.05


def test_lbfgs_memory_one_still_converges_and_stores_no_matrix():
    opt = LBFGS(memory=1).minimize(sopt.rosen, [-1.2, 1.0], grad=sopt.rosen_der)
    assert np.allclose(opt.x_, [1.0, 1.0], atol=1e-4)
    assert opt.result_.hess_inv is None


def test_conjugate_gradient_terminates_within_dim_steps_on_a_quadratic():
    rng = np.random.default_rng(20)
    dim = 6
    M = rng.normal(size=(dim, dim))
    G = M @ M.T + dim * np.eye(dim)
    c = rng.normal(size=dim)
    f = lambda x: 0.5 * float(x @ G @ x) + float(c @ x)
    g = lambda x: G @ x + c
    opt = ConjugateGradient(max_iter=200).minimize(f, np.zeros(dim), grad=g)
    assert np.max(np.abs(opt.x_ + np.linalg.solve(G, c))) < 1e-7
    # with restarts every dim steps, a quadratic is solved in a small multiple of dim
    assert opt.result_.nit <= 3 * dim


@pytest.mark.parametrize("beta", ["fletcher-reeves", "polak-ribiere"])
def test_conjugate_gradient_both_beta_rules_reach_the_same_minimum(beta):
    opt = ConjugateGradient(beta=beta, max_iter=3000).minimize(
        sopt.rosen, [-1.2, 1.0], grad=sopt.rosen_der)
    assert np.allclose(opt.x_, [1.0, 1.0], atol=1e-4)


def test_conjugate_gradient_rejects_an_unknown_beta_rule():
    with pytest.raises(ValueError):
        ConjugateGradient(beta="hestenes").minimize(_quad, [3.0, 3.0], grad=_quad_grad)


@pytest.mark.parametrize("subproblem", ["steihaug", "dogleg"])
def test_trust_region_both_subproblem_solvers_match_scipy_trust_ncg(subproblem):
    opt = TrustRegion(subproblem=subproblem).minimize(
        sopt.rosen, [-1.2, 1.0], grad=sopt.rosen_der, hess=sopt.rosen_hess)
    ref = sopt.minimize(sopt.rosen, [-1.2, 1.0], jac=sopt.rosen_der, hess=sopt.rosen_hess,
                        method="trust-ncg", options={"gtol": 1e-10})
    assert np.allclose(opt.x_, ref.x, atol=1e-5)


def test_newton_method_escapes_negative_curvature_at_a_saddle():
    # f = x^2 - y^2 has a saddle at the origin; the raw Newton step stays there forever,
    # the modified-Cholesky safeguard must move downhill in y
    f = lambda x: float(x[0] ** 2 - x[1] ** 2 + 0.25 * x[1] ** 4)
    g = lambda x: np.array([2 * x[0], -2 * x[1] + x[1] ** 3])
    h = lambda x: np.array([[2.0, 0.0], [0.0, -2.0 + 3 * x[1] ** 2]])
    opt = NewtonMethod(max_iter=50).minimize(f, [0.0, 1e-3], grad=g, hess=h)
    assert opt.result_.converged
    assert abs(abs(float(opt.x_[1])) - np.sqrt(2.0)) < 1e-6   # the true minima at y = +-sqrt2
    assert opt.fun_ == pytest.approx(-1.0, abs=1e-9)


def test_modified_cholesky_solve_returns_a_descent_direction_for_indefinite_hessians():
    H = np.array([[2.0, 0.0], [0.0, -2.0]])
    g = np.array([1.0, -1.0])
    d = _modified_cholesky_solve(H, g)
    assert float(np.dot(g, d)) < 0


def test_levenberg_marquardt_matches_scipy_least_squares():
    t = np.linspace(0, 3, 40)
    truth = 2.5 * np.exp(-0.7 * t)
    residual = lambda p: p[0] * np.exp(-p[1] * t) - truth
    opt = LevenbergMarquardt().minimize(residual, [1.0, 1.0])
    ref = sopt.least_squares(residual, [1.0, 1.0], method="lm")
    assert np.allclose(opt.x_, ref.x, atol=1e-6)
    assert np.allclose(opt.x_, [2.5, 0.7], atol=1e-6)
    assert opt.fun_ == pytest.approx(0.5 * float(np.sum(ref.fun ** 2)), abs=1e-12)
    assert opt.result_.extras["residual"].shape == t.shape


def test_levenberg_marquardt_with_analytic_jacobian_matches_the_finite_difference_run():
    t = np.linspace(0, 2, 25)
    truth = 1.8 * np.exp(-1.1 * t)
    residual = lambda p: p[0] * np.exp(-p[1] * t) - truth
    jac = lambda p: np.column_stack([np.exp(-p[1] * t), -p[0] * t * np.exp(-p[1] * t)])
    a = LevenbergMarquardt().minimize(residual, [1.0, 1.0])
    b = LevenbergMarquardt().minimize(residual, [1.0, 1.0], jac=jac)
    assert np.allclose(a.x_, b.x_, atol=1e-7)
    assert b.result_.extras["jacobian"].shape == (len(t), 2)


def test_levenberg_marquardt_recovers_noisy_regression_coefficients():
    rng = np.random.default_rng(21)
    t = np.linspace(0, 5, 120)
    truth = np.array([3.0, 0.45])
    y = truth[0] * np.exp(-truth[1] * t) + rng.normal(0, 0.02, len(t))
    opt = LevenbergMarquardt().minimize(lambda p: p[0] * np.exp(-p[1] * t) - y, [1.0, 1.0])
    ref = sopt.least_squares(lambda p: p[0] * np.exp(-p[1] * t) - y, [1.0, 1.0], method="lm")
    assert np.allclose(opt.x_, ref.x, atol=1e-6)
    assert np.max(np.abs(opt.x_ - truth)) < 0.05


# ===================================================================== metaheuristic

_STOCHASTIC_CONTINUOUS = [
    (SimulatedAnnealing, {"n_iter": 400}),
    (GeneticAlgorithm, {"n_generations": 20, "population_size": 20}),
    (ParticleSwarmOptimization, {"n_iter": 20, "n_particles": 15}),
    (DifferentialEvolution, {"n_iter": 20, "population_size": 20}),
    (CMA_ES, {"n_iter": 30}),
    (CEM, {"n_iter": 20, "population_size": 30}),
    (SPSA, {"n_iter": 60}),
    (BayesianOptimization, {"n_init": 4, "n_iter": 4, "n_candidates": 60}),
]


@pytest.mark.parametrize("cls,kw", _STOCHASTIC_CONTINUOUS,
                         ids=[c.__name__ for c, _ in _STOCHASTIC_CONTINUOUS])
def test_same_random_state_reproduces_a_run_exactly(cls, kw):
    f = lambda x: rastrigin(x)
    a = cls(bounds=(-5.0, 5.0), random_state=123, **kw).minimize(f, np.ones(3))
    b = cls(bounds=(-5.0, 5.0), random_state=123, **kw).minimize(f, np.ones(3))
    assert np.array_equal(a.x_, b.x_)
    assert a.fun_ == b.fun_


@pytest.mark.parametrize("cls,kw", [
    (GeneticAlgorithm, {"n_generations": 250}),
    (ParticleSwarmOptimization, {"n_iter": 400}),
    (DifferentialEvolution, {"n_iter": 300}),
    (CMA_ES, {"sigma0": 5.0, "n_iter": 800}),
    (SimulatedAnnealing, {"n_iter": 20000}),
], ids=["ga", "pso", "de", "cma", "sa"])
def test_metaheuristics_find_the_published_ackley_global_minimum(cls, kw):
    opt = cls(bounds=(-32.0, 32.0), random_state=0, **kw).minimize(ackley, np.full(5, 3.0))
    assert opt.fun_ < 0.05  # Ackley's global minimum is exactly 0 at the origin
    assert np.max(np.abs(opt.x_)) < 0.05


@pytest.mark.parametrize("cls,kw", [
    (GeneticAlgorithm, {"n_generations": 300}),
    (ParticleSwarmOptimization, {"n_iter": 400}),
    (DifferentialEvolution, {"n_iter": 400}),
    (CMA_ES, {"sigma0": 2.0, "n_iter": 800}),
], ids=["ga", "pso", "de", "cma"])
def test_metaheuristics_improve_massively_over_the_start_on_rastrigin(cls, kw):
    start = np.full(5, 3.0)
    opt = cls(bounds=(-5.12, 5.12), random_state=0, **kw).minimize(rastrigin, start)
    # Rastrigin-5 has ~10^5 local minima; a single run is not expected to hit the global
    # one, but must beat the start by orders of magnitude
    assert opt.fun_ < 0.1 * rastrigin(start)


def test_differential_evolution_matches_scipy_on_a_smooth_multimodal_function():
    opt = DifferentialEvolution(bounds=[(-4.5, 4.5), (-4.5, 4.5)], n_iter=400,
                                random_state=0).minimize(beale, [0.0, 0.0])
    ref = sopt.differential_evolution(beale, [(-4.5, 4.5), (-4.5, 4.5)], seed=0)
    assert opt.fun_ == pytest.approx(ref.fun, abs=1e-6)
    assert np.allclose(opt.x_, [3.0, 0.5], atol=1e-3)


def test_simulated_annealing_matches_scipy_dual_annealing_quality_on_beale():
    opt = SimulatedAnnealing(bounds=[(-4.5, 4.5), (-4.5, 4.5)], n_iter=40000,
                             random_state=0).minimize(beale, [0.0, 0.0])
    ref = sopt.dual_annealing(beale, [(-4.5, 4.5), (-4.5, 4.5)], seed=0)
    assert opt.fun_ < ref.fun + 1e-3
    assert np.allclose(opt.x_, [3.0, 0.5], atol=1e-2)


@pytest.mark.parametrize("strategy", ["rand/1/bin", "best/1/bin", "rand/2/bin"])
def test_differential_evolution_every_strategy_solves_the_sphere(strategy):
    opt = DifferentialEvolution(strategy=strategy, bounds=(-10.0, 10.0), n_iter=400,
                                random_state=3).minimize(sphere, np.full(4, 5.0))
    assert opt.fun_ < 1e-8
    assert opt.result_.extras["strategy"] == strategy


def test_differential_evolution_rejects_an_unknown_strategy():
    with pytest.raises(ValueError):
        DifferentialEvolution(strategy="rand/3/exp", n_iter=2,
                              bounds=(-1.0, 1.0)).minimize(sphere, [0.5, 0.5])


def test_genetic_algorithm_elitism_makes_the_best_objective_monotone():
    opt = GeneticAlgorithm(elitism=2, n_generations=60, bounds=(-5.12, 5.12),
                           random_state=4).minimize(rastrigin, np.full(3, 2.0))
    history = np.asarray(opt.result_.history)
    assert np.all(np.diff(history) <= 1e-12)


@pytest.mark.parametrize("cooling", ["geometric", "boltzmann", "cauchy"])
def test_simulated_annealing_every_cooling_schedule_descends(cooling):
    opt = SimulatedAnnealing(cooling=cooling, n_iter=4000, bounds=(-10.0, 10.0),
                             random_state=2).minimize(sphere, np.full(3, 5.0))
    assert opt.fun_ < sphere(np.full(3, 5.0)) / 100
    assert 0.0 <= opt.result_.extras["acceptance_rate"] <= 1.0


def test_simulated_annealing_rejects_an_unknown_cooling_schedule():
    with pytest.raises(ValueError):
        SimulatedAnnealing(cooling="linear", n_iter=10,
                           bounds=(-1.0, 1.0)).minimize(sphere, [0.5])


def test_simulated_annealing_calibrates_its_temperature_to_the_objective_scale():
    # the same landscape scaled by 1000 must produce a ~1000x larger T0, otherwise a
    # fixed T0 would make one of the two runs greedy and the other a random walk
    small = SimulatedAnnealing(n_iter=500, bounds=(-5.0, 5.0),
                               random_state=9).minimize(sphere, np.full(3, 3.0))
    big = SimulatedAnnealing(n_iter=500, bounds=(-5.0, 5.0),
                             random_state=9).minimize(lambda x: 1000 * sphere(x),
                                                      np.full(3, 3.0))
    ratio = big.result_.extras["T0"] / small.result_.extras["T0"]
    assert ratio == pytest.approx(1000.0, rel=1e-6)


def test_particle_swarm_clamps_velocity_and_stays_inside_the_box():
    opt = ParticleSwarmOptimization(bounds=[(-1.0, 1.0)] * 4, n_iter=100,
                                    random_state=5).minimize(sphere, np.zeros(4))
    assert np.all(np.abs(opt.population_) <= 1.0 + 1e-12)
    assert np.all(np.abs(opt.x_) <= 1.0 + 1e-12)
    assert isinstance(opt, PopulationOptimizer)


def test_cma_es_learns_an_anisotropic_covariance_on_an_ill_conditioned_problem():
    # a 1000:1 axis ratio: the learned C must reflect it, which is the whole point of CMA
    scale = np.array([1.0, 1000.0])
    f = lambda x: float(np.sum((x * scale) ** 2))
    opt = CMA_ES(sigma0=1.0, n_iter=400, random_state=0).minimize(f, np.array([1.0, 1.0]))
    C = opt.result_.extras["C"]
    assert C[0, 0] / C[1, 1] > 100
    assert opt.fun_ < 1e-8


def test_cma_es_matches_differential_evolution_on_a_smooth_problem():
    cma = CMA_ES(sigma0=2.0, n_iter=800, random_state=0).minimize(beale, [0.0, 0.0])
    de = DifferentialEvolution(bounds=[(-4.5, 4.5)] * 2, n_iter=300,
                               random_state=0).minimize(beale, [0.0, 0.0])
    assert cma.fun_ == pytest.approx(de.fun_, abs=1e-6)
    assert np.allclose(cma.x_, [3.0, 0.5], atol=1e-4)


@pytest.mark.parametrize("acquisition", ["ei", "ucb", "pi"])
def test_bayesian_optimization_finds_branin_with_few_evaluations(acquisition):
    def branin(x):
        a, b, c, r, s, t = 1.0, 5.1 / (4 * np.pi ** 2), 5 / np.pi, 6.0, 10.0, 1 / (8 * np.pi)
        return float(a * (x[1] - b * x[0] ** 2 + c * x[0] - r) ** 2
                     + s * (1 - t) * np.cos(x[0]) + s)

    opt = BayesianOptimization(n_init=10, n_iter=30, acquisition=acquisition,
                               bounds=[(-5.0, 10.0), (0.0, 15.0)],
                               random_state=0).minimize(branin, [0.0, 5.0])
    assert opt.fun_ < 0.7  # Branin's global minimum is 0.397887
    assert opt.result_.nfev <= 45  # the point of BO is a small evaluation budget
    assert opt.result_.extras["X_observed"].shape == (40, 2)


def test_bayesian_optimization_rejects_an_unknown_acquisition():
    with pytest.raises(ValueError):
        BayesianOptimization(n_init=3, n_iter=1, acquisition="thompson",
                             bounds=(-1.0, 1.0)).minimize(sphere, [0.5, 0.5])


def test_ant_colony_matches_brute_force_on_a_small_tsp():
    rng = np.random.default_rng(3)
    pts = rng.uniform(0, 10, (8, 2))
    D = np.linalg.norm(pts[:, None, :] - pts[None, :, :], axis=-1)
    length = lambda tour: float(np.sum(D[list(tour), list(np.roll(tour, -1))]))
    best = min(itertools.permutations(range(1, 8)), key=lambda p: length((0,) + p))
    opt = AntColony(n_ants=20, n_iter=80, random_state=0).minimize(D)
    assert opt.fun_ == pytest.approx(length((0,) + best), rel=1e-12)
    assert sorted(opt.tour_.tolist()) == list(range(8))
    assert opt.result_.extras["n_cities"] == 8


def test_ant_colony_rejects_a_non_square_distance_matrix():
    with pytest.raises(ValueError):
        AntColony().minimize(np.ones((3, 4)))
    with pytest.raises(ValueError):
        AntColony().minimize(np.ones((2, 2)))


def test_ant_colony_pheromone_concentrates_on_the_best_tour():
    rng = np.random.default_rng(6)
    pts = rng.uniform(0, 10, (7, 2))
    D = np.linalg.norm(pts[:, None, :] - pts[None, :, :], axis=-1)
    opt = AntColony(n_ants=15, n_iter=60, random_state=1).minimize(D)
    tau = opt.result_.extras["pheromone"]
    tour = opt.tour_
    on_tour = tau[tour, np.roll(tour, -1)]
    assert np.mean(on_tour) > 3 * np.mean(tau)


# =================================================================== stochastic optim

def test_stochastic_approx_step_sequence_satisfies_the_convergence_conditions():
    sa = StochasticApprox(a=1.0, A=0.0, alpha=0.7, n_iter=10)
    partial = {N: float(np.sum([sa._step_size(n) for n in range(1, N + 1)]))
               for N in (10 ** 4, 10 ** 5, 10 ** 6)}
    # sum a_n diverges like N^(1-alpha), so a 10x longer sum grows by 10^0.3 ~ 2
    assert partial[10 ** 5] / partial[10 ** 4] == pytest.approx(10 ** 0.3, rel=0.05)
    assert partial[10 ** 6] / partial[10 ** 5] == pytest.approx(10 ** 0.3, rel=0.05)
    # sum a_n^2 converges (2*alpha = 1.4 > 1), so it is bounded no matter how long
    squares = {N: float(np.sum([sa._step_size(n) ** 2 for n in range(1, N + 1)]))
               for N in (10 ** 4, 10 ** 6)}
    assert squares[10 ** 6] < 1.02 * squares[10 ** 4] < 10.0


def test_stochastic_approx_with_a_custom_direction_solves_the_quadratic():
    opt = StochasticApprox(a=1.0, alpha=0.7, n_iter=4000, averaging=True).minimize(
        _quad, [5.0, 5.0], direction=lambda x, n, rng: _quad_grad(x))
    assert np.max(np.abs(opt.x_ - _STAR)) < 1e-3
    assert opt.result_.extras["averaged"] is True


def test_robbins_monro_finds_the_root_of_a_noisy_regression_function():
    rm = RobbinsMonro(a=1.0, alpha=1.0, n_iter=20000, random_state=0).solve(
        lambda x, rng: (x - 3.0) + rng.normal(0.0, 1.0, size=x.shape), [0.0])
    # at alpha=1 the iterate is asymptotically normal with sd ~ sigma / sqrt(n)
    se = 1.0 / np.sqrt(20000)
    assert abs(float(rm.root_[0]) - 3.0) < 4 * se
    assert rm.result_.extras["target"] == 0.0


def test_robbins_monro_error_shrinks_at_the_root_n_rate():
    errs = []
    for n in (2000, 8000, 32000):
        reps = [abs(float(RobbinsMonro(a=1.0, alpha=1.0, n_iter=n, random_state=s).solve(
            lambda x, rng: (x - 3.0) + rng.normal(0.0, 1.0, size=x.shape),
            [0.0]).root_[0]) - 3.0) for s in range(12)]
        errs.append(float(np.mean(reps)))
    # a 4x longer run should roughly halve the error; allow a generous factor
    assert errs[1] < 0.8 * errs[0]
    assert errs[2] < 0.8 * errs[1]


def test_robbins_monro_with_a_nonzero_target():
    rm = RobbinsMonro(a=1.0, alpha=1.0, target=5.0, n_iter=20000, random_state=1).solve(
        lambda x, rng: 2.0 * x + rng.normal(0.0, 0.5, size=x.shape), [0.0])
    assert float(rm.root_[0]) == pytest.approx(2.5, abs=0.05)


def test_spsa_uses_two_objective_evaluations_per_iteration_at_any_dimension():
    for dim in (2, 5, 20):
        opt = SPSA(n_iter=100, random_state=0).minimize(sphere, np.ones(dim))
        assert opt.result_.extras["direction_evals"] == 200


def test_kiefer_wolfowitz_uses_two_evaluations_per_coordinate_per_iteration():
    for dim in (2, 5, 8):
        opt = KieferWolfowitz(n_iter=100).minimize(sphere, np.ones(dim))
        assert opt.result_.extras["direction_evals"] == 200 * dim


@pytest.mark.parametrize("cls", [KieferWolfowitz, SPSA])
def test_gradient_free_stochastic_approximation_solves_the_quadratic(cls):
    opt = cls(a=0.5, n_iter=4000, random_state=0).minimize(_quad, [5.0, 5.0])
    assert np.max(np.abs(opt.x_ - _STAR)) < 1e-2


def test_spsa_tolerates_a_noisy_objective():
    rng = np.random.default_rng(31)
    noisy = lambda x: _quad(x) + rng.normal(0.0, 0.05)
    opt = SPSA(a=0.4, c=0.2, n_iter=8000, averaging=True, random_state=2).minimize(
        noisy, [5.0, 5.0])
    assert np.max(np.abs(opt.x_ - _STAR)) < 0.15


def test_cem_solves_multimodal_problems_and_reports_a_monte_carlo_estimate():
    opt = CEM(n_iter=150, sigma0=3.0, random_state=0).minimize(rastrigin, np.full(4, 2.0))
    assert opt.fun_ < 1e-4
    est = opt.result_.extras["estimate"]
    assert isinstance(est, montecarlo.MCResult)
    lo, hi = est.confidence_interval()
    assert lo <= float(est) <= hi
    assert est.n_samples == len(opt.population_fun_)


def test_cem_extra_variance_prevents_premature_collapse():
    collapsed = CEM(n_iter=60, sigma0=2.0, extra_variance=0.0, random_state=7).minimize(
        rastrigin, np.full(4, 3.0))
    floored = CEM(n_iter=60, sigma0=2.0, extra_variance=0.2, random_state=7).minimize(
        rastrigin, np.full(4, 3.0))
    assert np.max(floored.result_.extras["sigma"]) > np.max(collapsed.result_.extras["sigma"])


def test_saa_newsvendor_matches_the_closed_form_critical_fractile():
    # demand ~ Exp(mean=20), unit cost 1, price 3 -> q* = -20 ln(1 - 2/3) = 20 ln 3
    cost = lambda q, d: float(1.0 * q[0] - 3.0 * min(q[0], d))
    sampler = lambda size, rng: rng.exponential(20.0, size=size)
    saa = SAA(n_samples=8000, n_batches=12, batch_size=800, random_state=0).minimize(
        cost, [10.0], sampler=sampler)
    assert float(saa.x_[0]) == pytest.approx(20 * np.log(3), rel=0.05)


def test_saa_reports_an_optimality_gap_with_standard_error_and_interval():
    cost = lambda q, d: float(1.0 * q[0] - 3.0 * min(q[0], d))
    sampler = lambda size, rng: rng.exponential(20.0, size=size)
    saa = SAA(n_samples=8000, n_batches=16, batch_size=800, random_state=0).minimize(
        cost, [10.0], sampler=sampler)
    gap = saa.gap_
    assert isinstance(gap, montecarlo.MCResult)
    reps = saa.result_.extras["gap_replications"]
    assert gap.std_error == pytest.approx(np.std(reps, ddof=1) / np.sqrt(len(reps)))
    lo, hi = gap.confidence_interval()
    assert lo < float(gap) < hi
    assert float(gap) >= 0.0  # the batch gap estimator is upward biased, never negative


def test_saa_solves_a_stochastic_quadratic_to_the_analytic_optimum():
    # min E[(x - xi)^2] with xi ~ N(4, 1) has minimizer 4 regardless of the variance
    cost = lambda x, xi: float((x[0] - xi) ** 2)
    sampler = lambda size, rng: rng.normal(4.0, 1.0, size=size)
    saa = SAA(n_samples=4000, n_batches=4, batch_size=400, random_state=2).minimize(
        cost, [0.0], sampler=sampler)
    se = 1.0 / np.sqrt(4000)
    assert abs(float(saa.x_[0]) - 4.0) < 4 * se


def test_saa_without_a_sampler_is_a_usage_error():
    with pytest.raises(ValueError):
        SAA().minimize(lambda x, xi: 0.0, [0.0])


# ======================================================================== constrained

_EQ = [{"type": "eq", "fun": lambda x: np.array([x[0] + x[1] - 1.0])}]
_INEQ = [{"type": "ineq", "fun": lambda x: np.array([x[0] + x[1] - 1.0])}]
_CIRCLE = lambda x: float(x[0] ** 2 + x[1] ** 2)


@pytest.mark.parametrize("cls", [PenaltyMethod, AugmentedLagrangian, LagrangianRelaxation,
                                 InteriorPoint])
def test_constrained_methods_solve_the_equality_problem_with_a_known_solution(cls):
    opt = cls(constraints=_EQ).minimize(_CIRCLE, [2.0, -1.0])
    assert np.allclose(opt.x_, [0.5, 0.5], atol=1e-3)
    assert opt.fun_ == pytest.approx(0.5, abs=1e-3)
    assert opt.result_.converged
    assert opt.result_.extras["feasible"] or opt.result_.extras["violation"] < 1e-3
    assert isinstance(opt, ConstrainedOptimizer)


@pytest.mark.parametrize("cls", [PenaltyMethod, AugmentedLagrangian, LagrangianRelaxation,
                                 InteriorPoint])
def test_constrained_methods_solve_the_inequality_problem_and_match_scipy_slsqp(cls):
    opt = cls(constraints=_INEQ).minimize(_CIRCLE, [2.0, 2.0])
    ref = sopt.minimize(_CIRCLE, [2.0, 2.0], method="SLSQP",
                        constraints=[{"type": "ineq", "fun": lambda x: x[0] + x[1] - 1.0}])
    assert np.allclose(opt.x_, ref.x, atol=1e-3)
    assert opt.fun_ == pytest.approx(ref.fun, abs=1e-3)


def test_augmented_lagrangian_is_feasible_at_a_far_smaller_penalty_than_the_penalty_method():
    al = AugmentedLagrangian(constraints=_EQ).minimize(_CIRCLE, [2.0, -1.0])
    pm = PenaltyMethod(constraints=_EQ).minimize(_CIRCLE, [2.0, -1.0])
    assert al.result_.extras["violation"] < pm.result_.extras["violation"]
    assert al.result_.extras["mu"] <= pm.result_.extras["mu"]


def test_augmented_lagrangian_recovers_the_analytic_multiplier():
    # stationarity of L = f - lambda.c gives 2x = lambda, so lambda = 2 * 0.5 = 1
    opt = AugmentedLagrangian(constraints=_EQ).minimize(_CIRCLE, [2.0, -1.0])
    assert float(opt.result_.extras["lambda_eq"][0]) == pytest.approx(1.0, abs=1e-3)


def test_lagrangian_relaxation_dual_bound_is_a_valid_lower_bound():
    opt = LagrangianRelaxation(constraints=_EQ).minimize(_CIRCLE, [2.0, -1.0])
    # weak duality: the dual value never exceeds the constrained optimum (0.5)
    assert opt.result_.extras["dual_bound"] <= 0.5 + 1e-8
    # and for this convex problem the gap closes
    assert abs(opt.result_.extras["duality_gap"]) < 1e-3


def test_lagrangian_relaxation_never_returns_a_feasible_but_unoptimized_start():
    # x0 = (2, -1) satisfies x+y=1 exactly but has objective 5, not 0.5
    opt = LagrangianRelaxation(constraints=_EQ).minimize(_CIRCLE, [2.0, -1.0])
    assert opt.fun_ < 1.0


def test_interior_point_requires_a_strictly_feasible_start():
    with pytest.raises(ValueError):
        InteriorPoint(constraints=_INEQ).minimize(_CIRCLE, [0.0, 0.0])


def test_interior_point_iterates_stay_inside_the_feasible_region():
    opt = InteriorPoint(constraints=_INEQ).minimize(_CIRCLE, [3.0, 3.0])
    assert opt.x_[0] + opt.x_[1] >= 1.0 - 1e-6
    assert opt.result_.extras["duality_gap_bound"] < 1e-6


def test_relaxed_log_barrier_is_finite_continuous_and_agrees_with_minus_log():
    delta = 1e-3
    inside = np.array([1.0, 0.5, 0.01])
    assert np.allclose(_relaxed_log(inside, delta), -np.log(inside))
    outside = np.array([-1.0, -0.5, 0.0])
    assert np.all(np.isfinite(_relaxed_log(outside, delta)))
    # continuity at the switch point
    assert _relaxed_log(np.array([delta * (1 + 1e-12)]), delta)[0] == pytest.approx(
        _relaxed_log(np.array([delta]), delta)[0], abs=1e-6)
    # and monotone decreasing, so an infeasible point is pushed back
    assert _relaxed_log(np.array([-1.0]), delta)[0] > _relaxed_log(np.array([0.0]), delta)[0]


def test_active_set_matches_scipy_on_a_textbook_quadratic_program():
    # Nocedal & Wright example 16.4: optimum (1.4, 1.7)
    G = np.array([[2.0, 0.0], [0.0, 2.0]])
    c = np.array([-2.0, -5.0])
    A_ub = np.array([[1.0, -2.0], [-1.0, -2.0], [-1.0, 2.0], [1.0, 0.0], [0.0, 1.0]])
    b_ub = np.array([-2.0, -6.0, -2.0, 0.0, 0.0])
    qp = lambda x: 0.5 * float(x @ G @ x) + float(c @ x)
    opt = ActiveSet(A_ub=A_ub, b_ub=b_ub).minimize(qp, [2.0, 0.0])
    ref = sopt.minimize(qp, [2.0, 0.0], method="SLSQP",
                        constraints=[{"type": "ineq", "fun": lambda x: A_ub @ x - b_ub}])
    assert np.allclose(opt.x_, [1.4, 1.7], atol=1e-8)
    assert np.allclose(opt.x_, ref.x, atol=1e-5)
    assert opt.result_.converged
    assert opt.result_.extras["violation"] < 1e-9


def test_active_set_handles_equality_constraints_and_reports_its_working_set():
    G = np.eye(2) * 2.0
    c = np.zeros(2)
    qp = lambda x: float(x @ x)
    opt = ActiveSet(A_eq=np.array([[1.0, 1.0]]), b_eq=np.array([1.0])).minimize(
        qp, [2.0, -1.0])
    assert np.allclose(opt.x_, [0.5, 0.5], atol=1e-9)
    assert opt.result_.extras["working_set"] == []
    assert opt.result_.extras["violation"] < 1e-12


def test_active_set_unconstrained_reduces_to_the_newton_step():
    opt = ActiveSet().minimize(_quad, [4.0, 4.0], grad=_quad_grad, hess=_quad_hess)
    assert np.max(np.abs(opt.x_ - _STAR)) < 1e-12
    # with finite-difference derivatives the same step is accurate to FD precision
    assert np.max(np.abs(ActiveSet().minimize(_quad, [4.0, 4.0]).x_ - _STAR)) < 1e-6


def test_constraint_type_must_be_eq_or_ineq():
    with pytest.raises(ValueError):
        PenaltyMethod(constraints=[{"type": "leq", "fun": lambda x: x}]).minimize(
            _CIRCLE, [1.0, 1.0])


def test_constrained_methods_accept_a_bare_constraint_dict():
    opt = AugmentedLagrangian(constraints=_EQ[0]).minimize(_CIRCLE, [2.0, -1.0])
    assert np.allclose(opt.x_, [0.5, 0.5], atol=1e-4)


def test_constrained_solvers_handle_a_nonlinear_constraint():
    # min x+y on the unit circle x^2+y^2 = 1 -> (-1/sqrt2, -1/sqrt2), f = -sqrt 2
    cons = [{"type": "eq", "fun": lambda x: np.array([x[0] ** 2 + x[1] ** 2 - 1.0])}]
    opt = AugmentedLagrangian(constraints=cons).minimize(
        lambda x: float(x[0] + x[1]), [-0.3, -0.9])
    assert opt.fun_ == pytest.approx(-np.sqrt(2.0), abs=1e-5)
    assert np.allclose(opt.x_, [-1 / np.sqrt(2)] * 2, atol=1e-5)


# ============================================================================ helpers

def test_normalize_bounds_accepts_scalar_and_per_coordinate_forms():
    lo, hi = _normalize_bounds((-1.0, 2.0), 3)
    assert np.allclose(lo, -1.0) and np.allclose(hi, 2.0) and len(lo) == 3
    lo, hi = _normalize_bounds([(-1.0, 1.0), (0.0, 5.0)], 2)
    assert np.allclose(lo, [-1.0, 0.0]) and np.allclose(hi, [1.0, 5.0])
    assert _normalize_bounds(None, 4) is None
    with pytest.raises(ValueError):
        _normalize_bounds([(1.0, 0.0)], 1)
    with pytest.raises(ValueError):
        _normalize_bounds([(-1.0, 1.0), (0.0, 5.0)], 3)


def test_converged_predicate_fires_on_each_of_its_three_criteria():
    ok, msg = _converged(np.zeros(2), np.zeros(2), 1.0, 1.0, np.zeros(2), 1e-8, 1e-8, 1e-8)
    assert ok and msg == "gradient below gtol"
    ok, msg = _converged(np.zeros(2), np.zeros(2), 1.0, 2.0, np.ones(2), 1e-8, None, 1e-8)
    assert ok and msg == "step below xtol"
    ok, msg = _converged(np.ones(2), np.zeros(2), 1.0, 1.0, np.ones(2), None, 1e-8, 1e-8)
    assert ok and msg == "objective change below ftol"
    ok, _ = _converged(np.ones(2), np.zeros(2), 1.0, 5.0, np.ones(2), 1e-8, 1e-8, 1e-8)
    assert not ok


def test_armijo_and_wolfe_line_searches_produce_a_decrease():
    obj = Objective(sopt.rosen, sopt.rosen_der)
    x = np.array([-1.2, 1.0])
    f0, g0 = obj(x), obj.grad(x)
    alpha, x_new, f_new = _backtracking_armijo(obj, x, -g0, f0, g0)
    assert alpha > 0 and f_new < f0
    alpha, x_new, f_new, g_new = _strong_wolfe(obj, x, -g0, f0, g0)
    assert alpha > 0 and f_new < f0
    # the strong-Wolfe curvature condition, which Armijo alone does not guarantee
    assert abs(float(np.dot(g_new, -g0))) <= 0.9 * abs(float(np.dot(g0, -g0))) + 1e-9


def test_line_searches_report_failure_on_an_ascent_direction_at_a_flat_point():
    obj = Objective(lambda x: 0.0)
    x = np.zeros(2)
    alpha, _, _ = _backtracking_armijo(obj, x, np.ones(2), 0.0, np.zeros(2))
    assert alpha == 0.0


def test_latin_hypercube_initial_design_comes_from_the_montecarlo_module():
    from stochpylib.optimization._common import _latin_hypercube

    rng = np.random.default_rng(0)
    design = _latin_hypercube(16, (np.array([0.0, -2.0]), np.array([1.0, 2.0])), rng)
    ref = np.asarray(montecarlo.LatinHypercubeSampling(
        dim=2, n=16, random_state=np.random.default_rng(0)).generate())
    assert design.shape == (16, 2)
    assert np.allclose(design[:, 0], ref[:, 0])
    assert np.all(design[:, 0] >= 0.0) and np.all(design[:, 0] <= 1.0)
    # exactly one point per stratum in every coordinate -- the defining LHS property,
    # measured against the requested box rather than the realized range
    lo, hi = np.array([0.0, -2.0]), np.array([1.0, 2.0])
    for j in range(2):
        strata = np.floor((design[:, j] - lo[j]) / (hi[j] - lo[j]) * 16).astype(int)
        assert sorted(np.clip(strata, 0, 15).tolist()) == list(range(16))


# ======================================================================= cross module

def test_optimization_reproduces_the_distributions_module_gamma_mle():
    from scipy import special

    data = np.asarray(distributions.Gamma(3.0, 2.0).rvs(2000, random_state=11), dtype=float)

    def nll(theta):
        k, s = np.exp(theta)  # log-parameterized, so the optimizer sees no constraints
        return float(-np.sum((k - 1) * np.log(data) - data / s
                             - special.gammaln(k) - k * np.log(s)))

    opt = NewtonMethod().minimize(nll, np.log([1.0, 1.0]))
    fitted = distributions.Gamma(1.0, 1.0).fit(data)
    assert np.exp(opt.x_[0]) == pytest.approx(fitted.shape, rel=1e-5)
    assert np.exp(opt.x_[1]) == pytest.approx(fitted.scale, rel=1e-5)


@pytest.mark.parametrize("cls", [BFGS, LBFGS, NewtonMethod, TrustRegion, ConjugateGradient])
def test_every_second_order_method_reproduces_the_statistics_module_logistic_mle(cls):
    rng = np.random.default_rng(13)
    n, p = 400, 3
    X = rng.normal(size=(n, p))
    Xd = np.column_stack([np.ones(n), X])
    y = (rng.random(n) < 1.0 / (1.0 + np.exp(-Xd @ np.array([0.4, -1.0, 0.7, 0.2])))
         ).astype(float)
    nll = lambda b: float(np.sum(np.logaddexp(0.0, Xd @ b) - y * (Xd @ b)))
    nll_grad = lambda b: Xd.T @ (1.0 / (1.0 + np.exp(-Xd @ b)) - y)
    opt = cls().minimize(nll, np.zeros(p + 1), grad=nll_grad)
    ref = statistics.logistic_regression(X, y)
    assert np.max(np.abs(opt.x_ - ref.coef_)) < 1e-5


def test_bayesian_optimization_surrogate_is_the_gaussian_processes_module():
    from stochpylib.gaussian_processes import GPRegression

    opt = BayesianOptimization(n_init=6, n_iter=6, n_candidates=100,
                               bounds=[(-2.0, 2.0)] * 2, random_state=0).minimize(
        sphere, [1.5, 1.5])
    X, y = opt.result_.extras["X_observed"], opt.result_.extras["y_observed"]
    assert len(X) == len(y) == 12
    # the recorded observations must be exactly the objective evaluated at those points
    assert np.allclose(y, [sphere(xi) for xi in X])
    assert GPRegression is not None


def test_library_code_never_imports_scipy_stats_or_scipy_optimize():
    import pathlib
    import re

    root = pathlib.Path(__file__).resolve().parents[2] / "stochpylib" / "optimization"
    banned = re.compile(r"^\s*(?:from|import)\s+scipy\.(?:stats|optimize)", re.MULTILINE)
    seen = 0
    for path in sorted(root.glob("*.py")):
        src = path.read_text(encoding="utf-8")
        assert not banned.search(src), path.name  # scipy is the oracle, never a backend
        assert not re.search(r"^\s*from\s+scipy\s+import\s+.*(stats|optimize)",
                             src, re.MULTILINE), path.name
        seen += 1
    assert seen >= 9


# ========================================================================= quickstart

def test_quickstart_example_runs():
    from stochpylib import optimization as opt

    # the exact snippet advertised in the module README and the root quickstart
    rosen = lambda x: float(100 * (x[1] - x[0] ** 2) ** 2 + (1 - x[0]) ** 2)
    fit = opt.BFGS().minimize(rosen, [-1.2, 1.0])
    assert np.allclose(fit.x_, [1.0, 1.0], atol=1e-4)
    assert fit.result_.converged

    swarm = opt.ParticleSwarmOptimization(bounds=[(-5.12, 5.12)] * 3,
                                          random_state=0).minimize(rastrigin, np.ones(3))
    assert swarm.fun_ < 1e-6

    solved = opt.AugmentedLagrangian(
        constraints=[{"type": "eq", "fun": lambda x: np.array([x[0] + x[1] - 1.0])}]
    ).minimize(lambda x: float(x[0] ** 2 + x[1] ** 2), [2.0, -1.0])
    assert np.allclose(solved.x_, [0.5, 0.5], atol=1e-6)
    assert solved.result_.extras["feasible"]
