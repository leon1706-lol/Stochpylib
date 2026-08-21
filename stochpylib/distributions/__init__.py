"""Probability distributions: the common ``Distribution`` interface from
``_base`` plus every concrete distribution, re-exported here so users import
from the module level (``from stochpylib.distributions import Normal``) per
ARCHITECTURE.md.
"""

from stochpylib.distributions._base import (
    Distribution,
    MultivariateDistribution,
)
from stochpylib.distributions.discrete import (
    Bernoulli,
    BetaBinomial,
    Binomial,
    ConwayMaxwellPoisson,
    DiscreteUniform,
    Geometric,
    Hypergeometric,
    Multinomial,
    NegBinomial,
    Poisson,
    ZipfDistribution,
)
from stochpylib.distributions.continuous import (
    Beta,
    Cauchy,
    Chi2,
    Exponential,
    F,
    Frechet,
    Gamma,
    GEV,
    Gumbel,
    InvGamma,
    InvGaussian,
    Kumaraswamy,
    Laplace,
    LogNormal,
    Maxwell,
    Nakagami,
    Normal,
    Pareto,
    Rayleigh,
    Rice,
    Student_t,
    Uniform,
    VonMises,
    Weibull,
    GPareto,
)
from stochpylib.distributions.multivariate import (
    Dirichlet,
    InverseWishart,
    MultivariateNormal,
    MultivariatePareto,
    MultivariateT,
    Wishart,
)
from stochpylib.distributions.heavy_tail import (
    AlphaStable,
    LevyDistribution,
    StableDistribution,
    SubExponential,
    SubGaussian,
)

__all__ = [
    # base
    "Distribution",
    "MultivariateDistribution",
    # discrete
    "Bernoulli",
    "BetaBinomial",
    "Binomial",
    "ConwayMaxwellPoisson",
    "DiscreteUniform",
    "Geometric",
    "Hypergeometric",
    "Multinomial",
    "NegBinomial",
    "Poisson",
    "ZipfDistribution",
    # continuous
    "Beta",
    "Cauchy",
    "Chi2",
    "Exponential",
    "F",
    "Frechet",
    "Gamma",
    "GEV",
    "GPareto",
    "Gumbel",
    "InvGamma",
    "InvGaussian",
    "Kumaraswamy",
    "Laplace",
    "LogNormal",
    "Maxwell",
    "Nakagami",
    "Normal",
    "Pareto",
    "Rayleigh",
    "Rice",
    "Student_t",
    "Uniform",
    "VonMises",
    "Weibull",
    # multivariate
    "Dirichlet",
    "InverseWishart",
    "MultivariateNormal",
    "MultivariatePareto",
    "MultivariateT",
    "Wishart",
    # heavy tail
    "AlphaStable",
    "LevyDistribution",
    "StableDistribution",
    "SubExponential",
    "SubGaussian",
]
