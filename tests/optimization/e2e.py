"""End-to-end API sweep for stochpylib.optimization: one realistic exercise per public
name. ``test_every_public_name_is_exercised`` fails the moment a name ships without one;
``test_exercise[<name>]`` runs each exercise as its own pytest case."""

import numpy as np
import pytest

from stochpylib import optimization as mod

EXERCISES = {}


def exercise(name):
    def deco(fn):
        EXERCISES[name] = fn
        return fn
    return deco


_RNG = np.random.default_rng(0)

_A = np.array([[3.0, 1.0], [1.0, 2.0]])
_B = np.array([1.0, -1.0])
_STAR = np.linalg.solve(_A, _B)
_X0 = np.array([2.0, 2.0])


def _quad(x):
    return 0.5 * float(x @ _A @ x) - float(_B @ x)


def _quad_grad(x):
    return _A @ x - _B


def _quad_hess(x):
    return _A


def _sphere(x):
    return float(np.sum(np.asarray(x, dtype=float) ** 2))


_EQ = [{"type": "eq", "fun": lambda x: np.array([x[0] + x[1] - 1.0])}]
_INEQ = [{"type": "ineq", "fun": lambda x: np.array([x[0] + x[1] - 1.0])}]


def _finite(a):
    return bool(np.all(np.isfinite(np.asarray(a, dtype=float))))


def _near_star(opt, tol=1e-2):
    return bool(np.max(np.abs(opt.x_ - _STAR)) < tol)


# ------------------------------------------------------------------- base and result

@exercise("Objective")
def _objective():
    obj = mod.Objective(_quad, _quad_grad)
    assert obj(_X0) == pytest.approx(_quad(_X0))
    assert _finite(obj.grad(_X0)) and _finite(obj.hess(_X0))
    assert obj.has_gradient and obj.n_evals >= 1


@exercise("Optimizer")
def _optimizer_base():
    opt = mod.BFGS().minimize(_quad, _X0, grad=_quad_grad)
    assert isinstance(opt, mod.Optimizer)
    assert opt.to_result() is opt.result_


@exercise("PopulationOptimizer")
def _population_base():
    opt = mod.DifferentialEvolution(bounds=(-3.0, 3.0), n_iter=10, population_size=12,
                                    random_state=0).minimize(_sphere, _X0)
    assert isinstance(opt, mod.PopulationOptimizer)
    assert opt.population_.shape == (12, 2)
    assert len(opt.population_fun_) == 12


@exercise("ConstrainedOptimizer")
def _constrained_base():
    opt = mod.PenaltyMethod(constraints=_EQ, n_outer=4).minimize(_sphere, _X0)
    assert isinstance(opt, mod.ConstrainedOptimizer)
    assert "violation" in opt.result_.extras and "feasible" in opt.result_.extras


@exercise("OptimizeResult")
def _optimize_result():
    res = mod.LBFGS(max_iter=50).minimize(_quad, _X0, grad=_quad_grad).result_
    assert isinstance(res, mod.OptimizeResult)
    assert float(res) == pytest.approx(res.fun)
    assert res.nfev > 0 and len(res.history) >= 1


# ---------------------------------------------------------------------------- gradient

@exercise("GradientDescent")
def _gradient_descent():
    opt = mod.GradientDescent(max_iter=500).minimize(_quad, _X0, grad=_quad_grad)
    assert _near_star(opt)


@exercise("StochasticGD")
def _stochastic_gd():
    opt = mod.StochasticGD(learning_rate=0.05, momentum=0.9, max_iter=800,
                           random_state=1).minimize(_quad, _X0, grad=_quad_grad)
    assert _near_star(opt)


@exercise("AdamOptimizer")
def _adam():
    opt = mod.AdamOptimizer(max_iter=1500).minimize(_quad, _X0, grad=_quad_grad)
    assert _near_star(opt)


@exercise("AdaGrad")
def _adagrad():
    opt = mod.AdaGrad(max_iter=1500).minimize(_quad, _X0, grad=_quad_grad)
    assert _near_star(opt)


@exercise("RMSProp")
def _rmsprop():
    opt = mod.RMSProp(max_iter=1500).minimize(_quad, _X0, grad=_quad_grad)
    assert _near_star(opt, tol=0.05)


@exercise("Adadelta")
def _adadelta():
    opt = mod.Adadelta(learning_rate=5.0, max_iter=2000).minimize(_quad, _X0,
                                                                  grad=_quad_grad)
    assert _near_star(opt)


@exercise("NADAM")
def _nadam():
    opt = mod.NADAM(max_iter=1500).minimize(_quad, _X0, grad=_quad_grad)
    assert _near_star(opt)


@exercise("AMSGrad")
def _amsgrad():
    opt = mod.AMSGrad(max_iter=1500).minimize(_quad, _X0, grad=_quad_grad)
    assert _near_star(opt)


# ----------------------------------------------------------------------- second order

@exercise("NewtonMethod")
def _newton():
    opt = mod.NewtonMethod().minimize(_quad, _X0, grad=_quad_grad, hess=_quad_hess)
    assert _near_star(opt, tol=1e-8) and opt.result_.converged


@exercise("BFGS")
def _bfgs():
    opt = mod.BFGS().minimize(_quad, _X0, grad=_quad_grad)
    assert _near_star(opt, tol=1e-8)
    assert opt.result_.hess_inv.shape == (2, 2)


@exercise("LBFGS")
def _lbfgs():
    opt = mod.LBFGS(memory=5).minimize(_quad, _X0, grad=_quad_grad)
    assert _near_star(opt, tol=1e-8)


@exercise("ConjugateGradient")
def _conjugate_gradient():
    opt = mod.ConjugateGradient().minimize(_quad, _X0, grad=_quad_grad)
    assert _near_star(opt, tol=1e-6)


@exercise("TrustRegion")
def _trust_region():
    opt = mod.TrustRegion().minimize(_quad, _X0, grad=_quad_grad, hess=_quad_hess)
    assert _near_star(opt, tol=1e-8)
    assert opt.result_.extras["radius"] > 0


@exercise("LevenbergMarquardt")
def _levenberg_marquardt():
    t = np.linspace(0, 2, 20)
    truth = 2.0 * np.exp(-0.8 * t)
    opt = mod.LevenbergMarquardt().minimize(lambda p: p[0] * np.exp(-p[1] * t) - truth,
                                            [1.0, 1.0])
    assert np.allclose(opt.x_, [2.0, 0.8], atol=1e-5)


# ----------------------------------------------------------------------- metaheuristic

@exercise("SimulatedAnnealing")
def _simulated_annealing():
    opt = mod.SimulatedAnnealing(n_iter=2000, bounds=(-5.0, 5.0),
                                 random_state=0).minimize(_sphere, np.full(3, 3.0))
    assert opt.fun_ < 0.5
    assert 0.0 <= opt.result_.extras["acceptance_rate"] <= 1.0


@exercise("GeneticAlgorithm")
def _genetic_algorithm():
    opt = mod.GeneticAlgorithm(population_size=30, n_generations=60, bounds=(-5.0, 5.0),
                               random_state=0).minimize(_sphere, np.full(3, 3.0))
    assert opt.fun_ < 0.05


@exercise("ParticleSwarmOptimization")
def _particle_swarm():
    opt = mod.ParticleSwarmOptimization(n_particles=20, n_iter=80, bounds=(-5.0, 5.0),
                                        random_state=0).minimize(_sphere, np.full(3, 3.0))
    assert opt.fun_ < 1e-8


@exercise("DifferentialEvolution")
def _differential_evolution():
    opt = mod.DifferentialEvolution(population_size=20, n_iter=80, bounds=(-5.0, 5.0),
                                    random_state=0).minimize(_sphere, np.full(3, 3.0))
    assert opt.fun_ < 1e-8


@exercise("CMA_ES")
def _cma_es():
    opt = mod.CMA_ES(sigma0=1.0, n_iter=200, random_state=0).minimize(_sphere,
                                                                      np.full(3, 3.0))
    assert opt.fun_ < 1e-8
    assert opt.result_.extras["C"].shape == (3, 3)


@exercise("BayesianOptimization")
def _bayesian_optimization():
    opt = mod.BayesianOptimization(n_init=6, n_iter=8, n_candidates=80,
                                   bounds=[(-2.0, 2.0)] * 2,
                                   random_state=0).minimize(_sphere, [1.5, 1.5])
    assert opt.fun_ < 1.0
    assert opt.result_.extras["X_observed"].shape == (14, 2)


@exercise("AntColony")
def _ant_colony():
    rng = np.random.default_rng(2)
    pts = rng.uniform(0, 10, (6, 2))
    D = np.linalg.norm(pts[:, None, :] - pts[None, :, :], axis=-1)
    opt = mod.AntColony(n_ants=10, n_iter=25, random_state=0).minimize(D)
    assert sorted(opt.tour_.tolist()) == list(range(6))
    assert opt.fun_ > 0


# -------------------------------------------------------------------- stochastic optim

@exercise("StochasticApprox")
def _stochastic_approx():
    opt = mod.StochasticApprox(a=1.0, alpha=0.7, n_iter=1500, averaging=True).minimize(
        _quad, _X0, grad=_quad_grad)
    assert _near_star(opt, tol=0.05)


@exercise("RobbinsMonro")
def _robbins_monro():
    rm = mod.RobbinsMonro(a=1.0, alpha=1.0, n_iter=3000, random_state=0).solve(
        lambda x, rng: (x - 3.0) + rng.normal(0.0, 1.0, size=x.shape), [0.0])
    assert abs(float(rm.root_[0]) - 3.0) < 0.1


@exercise("KieferWolfowitz")
def _kiefer_wolfowitz():
    opt = mod.KieferWolfowitz(a=0.5, n_iter=800, random_state=0).minimize(_quad, _X0)
    assert _near_star(opt, tol=0.05)
    assert opt.result_.extras["direction_evals"] == 800 * 2 * 2


@exercise("SPSA")
def _spsa():
    opt = mod.SPSA(a=0.5, n_iter=1500, random_state=0).minimize(_quad, _X0)
    assert _near_star(opt, tol=0.05)
    assert opt.result_.extras["direction_evals"] == 1500 * 2


@exercise("CEM")
def _cem():
    opt = mod.CEM(population_size=40, n_iter=80, sigma0=2.0,
                  random_state=0).minimize(_quad, _X0)
    assert _near_star(opt, tol=1e-3)
    assert np.isfinite(float(opt.result_.extras["estimate"]))


@exercise("SAA")
def _saa():
    cost = lambda x, xi: float((x[0] - xi) ** 2)
    sampler = lambda size, rng: rng.normal(4.0, 1.0, size=size)
    opt = mod.SAA(n_samples=400, n_batches=3, batch_size=100,
                  random_state=0).minimize(cost, [0.0], sampler=sampler)
    assert abs(float(opt.x_[0]) - 4.0) < 0.3
    assert np.isfinite(float(opt.gap_))


# ------------------------------------------------------------------------- constrained

@exercise("PenaltyMethod")
def _penalty_method():
    opt = mod.PenaltyMethod(constraints=_EQ).minimize(_sphere, [2.0, -1.0])
    assert np.allclose(opt.x_, [0.5, 0.5], atol=1e-3)


@exercise("AugmentedLagrangian")
def _augmented_lagrangian():
    opt = mod.AugmentedLagrangian(constraints=_EQ).minimize(_sphere, [2.0, -1.0])
    assert np.allclose(opt.x_, [0.5, 0.5], atol=1e-5)
    assert opt.result_.extras["feasible"]


@exercise("LagrangianRelaxation")
def _lagrangian_relaxation():
    opt = mod.LagrangianRelaxation(constraints=_EQ).minimize(_sphere, [2.0, -1.0])
    assert np.allclose(opt.x_, [0.5, 0.5], atol=1e-3)
    assert opt.result_.extras["dual_bound"] <= 0.5 + 1e-8


@exercise("InteriorPoint")
def _interior_point():
    opt = mod.InteriorPoint(constraints=_INEQ).minimize(_sphere, [3.0, 3.0])
    assert np.allclose(opt.x_, [0.5, 0.5], atol=1e-3)


@exercise("ActiveSet")
def _active_set():
    qp = lambda x: float(x @ x)
    opt = mod.ActiveSet(A_eq=np.array([[1.0, 1.0]]), b_eq=np.array([1.0])).minimize(
        qp, [2.0, -1.0])
    assert np.allclose(opt.x_, [0.5, 0.5], atol=1e-8)
    assert opt.result_.extras["working_set"] == []


def test_every_public_name_is_exercised():
    assert set(EXERCISES) == set(mod.__all__), (
        set(mod.__all__) - set(EXERCISES), set(EXERCISES) - set(mod.__all__))


@pytest.mark.parametrize("name", sorted(mod.__all__))
def test_exercise(name):
    EXERCISES[name]()
