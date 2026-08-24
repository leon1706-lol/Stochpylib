"""Copulas: dependence modeling with elliptical, archimedean, empirical and
vine copulas behind one shared interface."""

from stochpylib.copulas._base import BaseCopula
from stochpylib.copulas._utils import (
    as_u_matrix, kendall_tau_estimate, pseudo_obs, spearman_rho_estimate,
    student_t_ppf,
)
from stochpylib.copulas.elliptical import GaussianCopula, StudentTCopula
from stochpylib.copulas.archimedean import (
    AliMikhailHaqCopula,
    BB1Copula,
    BB7Copula,
    ClaytonCopula,
    FrankCopula,
    GumbelCopula,
    JoeCopula,
    PlackettCopula,
)
from stochpylib.copulas.empirical import (
    BetaCopula,
    CheckerboardCopula,
    EmpiricalCopula,
)
from stochpylib.copulas.pair import PairCopulaConstruction
from stochpylib.copulas.vine import (
    CVine,
    DVine,
    RVine,
    VineCopula,
    VineStructureSelect,
)
from stochpylib.copulas.methods import (
    CopulaFit,
    CopulaSample,
    conditional_copula,
    copula_density,
    kendall_tau,
    spearman_rho,
    tail_dependence,
)

__all__ = [
    "BaseCopula",
    # elliptical
    "GaussianCopula", "StudentTCopula",
    # archimedean + specials
    "ClaytonCopula", "FrankCopula", "GumbelCopula", "JoeCopula",
    "AliMikhailHaqCopula", "PlackettCopula", "BB1Copula", "BB7Copula",
    # empirical
    "EmpiricalCopula", "CheckerboardCopula", "BetaCopula",
    # vines
    "PairCopulaConstruction", "CVine", "DVine", "RVine",
    "VineStructureSelect", "VineCopula",
    # methods
    "CopulaFit", "CopulaSample", "kendall_tau", "spearman_rho",
    "tail_dependence", "copula_density", "conditional_copula",
]
