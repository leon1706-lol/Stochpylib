"""Geostatistics and spatial modeling: variogram models and fitting, kriging (simple,
ordinary, universal, co-, indicator, disjunctive), covariance-driven random fields
(exact Cholesky/circulant-embedding sampling, Brownian/fractional-Brownian sheets),
spatial point processes (Poisson, Thomas, Matern cluster, log-Gaussian Cox) with Ripley's K
and the pair correlation function, and spatial autocorrelation tests (Moran's I, Geary's C,
Getis-Ord, nearest-neighbour distance) -- 32 spec names across five submodules, natively on
numpy/scipy.

Predictions reuse the library's result types (``timeseries.ForecastResult`` for kriging
mean/std, ``statistics.TestResult`` for the autocorrelation and nearest-neighbour tests);
``SpatialFunction`` is the one new result type, for K/pair-correlation summary functions.
``spatial_statistics.GaussianRandomField`` is a separate, covariance-driven class from
``levy_processes.GaussianRandomField`` (FFT spectral synthesis) -- see its docstring.
Nothing wraps ``scipy.stats``; it is the test suite's oracle only. ``SARModel``/``CARModel``
(lattice models) and ``SpatialWeights``/``SpatialFunction`` are documented extras beyond the
32 spec names.
"""

from stochpylib.spatial_statistics._result import SpatialFunction
from stochpylib.spatial_statistics.weights import SpatialWeights
from stochpylib.spatial_statistics.variogram import (
    ExperimentalVariogram,
    Nugget,
    Range,
    Semivariogram,
    Sill,
    SpatialCovariance,
    Variogram,
    VariogramFitting,
)
from stochpylib.spatial_statistics.random_fields import (
    BrownianSheet,
    FractionalBrownianSheet,
    GaussianRandomField,
    MaternField,
    OrnsteinUhlenbeckField,
)
from stochpylib.spatial_statistics.kriging import (
    CoKriging,
    DisjunctiveKriging,
    IndicatorKriging,
    Kriging,
    OrdinaryKriging,
    SimpleKriging,
    UniversalKriging,
)
from stochpylib.spatial_statistics.point_processes import (
    InhomogeneousPoisson,
    LogGaussianCox,
    MaternCluster,
    PairCorrelation,
    PoissonPointProcess,
    RipleyK,
    SpatialPointProcess,
    ThomasProcess,
)
from stochpylib.spatial_statistics.tests import GearyC, MoransI, NNDistanceTest, SpatialAutocorrelation
from stochpylib.spatial_statistics.lattice import CARModel, SARModel

__all__ = [
    "BrownianSheet",
    "CARModel",
    "CoKriging",
    "DisjunctiveKriging",
    "ExperimentalVariogram",
    "FractionalBrownianSheet",
    "GaussianRandomField",
    "GearyC",
    "IndicatorKriging",
    "InhomogeneousPoisson",
    "Kriging",
    "LogGaussianCox",
    "MaternCluster",
    "MaternField",
    "MoransI",
    "NNDistanceTest",
    "Nugget",
    "OrdinaryKriging",
    "OrnsteinUhlenbeckField",
    "PairCorrelation",
    "PoissonPointProcess",
    "Range",
    "RipleyK",
    "SARModel",
    "Semivariogram",
    "Sill",
    "SimpleKriging",
    "SpatialAutocorrelation",
    "SpatialCovariance",
    "SpatialFunction",
    "SpatialPointProcess",
    "SpatialWeights",
    "ThomasProcess",
    "UniversalKriging",
    "Variogram",
    "VariogramFitting",
]

assert len(__all__) == len(set(__all__)) == 36
