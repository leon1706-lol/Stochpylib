"""Stochastic and numerical optimization: first-order and adaptive-step gradient methods,
quasi-Newton and trust-region second-order methods, derivative-free global metaheuristics,
stochastic approximation, and constrained solvers -- 32 spec names across five submodules,
natively on numpy/scipy. Every optimizer subclasses
:class:`~stochpylib.optimization._base.Optimizer`, is driven by ``minimize(fun, x0)``
returning ``self``, and reports a single
:class:`~stochpylib.optimization._result.OptimizeResult` (point, value, effort counters,
convergence status). Objectives are wrapped in
:class:`~stochpylib.optimization._base.Objective`, which supplies finite-difference
gradients and Hessians so second-order methods work without analytic derivatives. Where an
answer is itself a Monte Carlo quantity -- ``SAA``'s optimality gap, ``CEM``'s elite
objective -- it is returned as a :class:`stochpylib.montecarlo.MCResult` rather than as a
bare number. Library code never wraps ``scipy.optimize``; every algorithm here is
implemented from the published update rules, and scipy is the test suite's independent
oracle only.
"""

from stochpylib.optimization._base import (
    ConstrainedOptimizer,
    Objective,
    Optimizer,
    PopulationOptimizer,
)
from stochpylib.optimization._result import OptimizeResult
from stochpylib.optimization.gradient import (
    AMSGrad,
    AdaGrad,
    Adadelta,
    AdamOptimizer,
    GradientDescent,
    NADAM,
    RMSProp,
    StochasticGD,
)
from stochpylib.optimization.second_order import (
    BFGS,
    ConjugateGradient,
    LBFGS,
    LevenbergMarquardt,
    NewtonMethod,
    TrustRegion,
)
from stochpylib.optimization.metaheuristic import (
    AntColony,
    BayesianOptimization,
    CMA_ES,
    DifferentialEvolution,
    GeneticAlgorithm,
    ParticleSwarmOptimization,
    SimulatedAnnealing,
)
from stochpylib.optimization.stochastic_optim import (
    CEM,
    KieferWolfowitz,
    RobbinsMonro,
    SAA,
    SPSA,
    StochasticApprox,
)
from stochpylib.optimization.constrained import (
    ActiveSet,
    AugmentedLagrangian,
    InteriorPoint,
    LagrangianRelaxation,
    PenaltyMethod,
)

__all__ = [
    "AMSGrad",
    "ActiveSet",
    "AdaGrad",
    "Adadelta",
    "AdamOptimizer",
    "AntColony",
    "AugmentedLagrangian",
    "BFGS",
    "BayesianOptimization",
    "CEM",
    "CMA_ES",
    "ConjugateGradient",
    "ConstrainedOptimizer",
    "DifferentialEvolution",
    "GeneticAlgorithm",
    "GradientDescent",
    "InteriorPoint",
    "KieferWolfowitz",
    "LBFGS",
    "LagrangianRelaxation",
    "LevenbergMarquardt",
    "NADAM",
    "NewtonMethod",
    "Objective",
    "OptimizeResult",
    "Optimizer",
    "ParticleSwarmOptimization",
    "PenaltyMethod",
    "PopulationOptimizer",
    "RMSProp",
    "RobbinsMonro",
    "SAA",
    "SPSA",
    "SimulatedAnnealing",
    "StochasticApprox",
    "StochasticGD",
    "TrustRegion",
]

assert len(__all__) == len(set(__all__)) == 37
