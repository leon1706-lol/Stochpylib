"""Distribution-free methods: kernel/series density estimation, empirical
distribution/CDF/characteristic-function estimators and Glivenko-Cantelli
bounds, Owen's empirical likelihood, resampling and rank-based hypothesis
tests, rank/distance dependence measures, and local/isotonic/spline
regression -- 31 spec names across five submodules, natively on numpy/scipy.
Density estimators satisfy the full 13-method distribution contract
(:class:`~stochpylib.nonparametric._base.NonparametricDensity` subclasses
:class:`stochpylib.distributions.Distribution`); hypothesis tests and
dependence measures return :class:`stochpylib.statistics.TestResult`/
:class:`~stochpylib.statistics.EstimateResult` rather than defining new
result types.
"""

from stochpylib.nonparametric._base import (
    DependenceMeasure,
    NonparametricDensity,
    NonparametricRegressor,
    NonparametricTest,
)
from stochpylib.nonparametric.density import (
    AdaptiveKDE,
    KernelDensityEstimate,
    LogsplineEstimator,
    NearestNeighborDensity,
    OrthogonalSeriesDensity,
)
from stochpylib.nonparametric.empirical import (
    EmpiricalCDF,
    EmpiricalCharFn,
    EmpiricalDistribution,
    EmpiricalLikelihood,
    GlivenkoCantelli,
)
from stochpylib.nonparametric.tests import (
    AndersenDarling,
    AndersonDarling,
    BootstrapTest,
    CramerVonMises,
    FriedmanTest,
    KruskalWallis,
    MoodTest,
    PermutationTest,
    RunsTest,
    SignTest,
    WaldWolfowitz,
)
from stochpylib.nonparametric.correlation import (
    BrownianCorrelation,
    DistanceCorrelation,
    HoeffdingD,
    KendallTau,
    RankCorrelation,
    SpearmanCorrelation,
)
from stochpylib.nonparametric.regression import (
    GPR_Nonparametric,
    IsotonicRegression,
    LocalPolynomialReg,
    QuantileRegression,
    SplineRegression,
)

__all__ = [
    "AdaptiveKDE",
    "AndersenDarling",
    "AndersonDarling",
    "BootstrapTest",
    "BrownianCorrelation",
    "CramerVonMises",
    "DependenceMeasure",
    "DistanceCorrelation",
    "EmpiricalCDF",
    "EmpiricalCharFn",
    "EmpiricalDistribution",
    "EmpiricalLikelihood",
    "FriedmanTest",
    "GPR_Nonparametric",
    "GlivenkoCantelli",
    "HoeffdingD",
    "IsotonicRegression",
    "KendallTau",
    "KernelDensityEstimate",
    "KruskalWallis",
    "LocalPolynomialReg",
    "LogsplineEstimator",
    "MoodTest",
    "NearestNeighborDensity",
    "NonparametricDensity",
    "NonparametricRegressor",
    "NonparametricTest",
    "OrthogonalSeriesDensity",
    "PermutationTest",
    "QuantileRegression",
    "RankCorrelation",
    "RunsTest",
    "SignTest",
    "SpearmanCorrelation",
    "SplineRegression",
    "WaldWolfowitz",
]

assert len(__all__) == len(set(__all__)) == 36
