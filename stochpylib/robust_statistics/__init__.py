"""Outlier-resistant estimators: robust location (trimmed/winsorized means,
median, Hodges-Lehmann, L/M/R-estimators), robust scale (MAD, Qn, Sn, IQR,
Huber/biweight/tau M-scales), high-breakdown regression (Theil-Sen, Siegel,
RANSAC, FAST-LTS, FAST-S+MM, Huber), robust covariance (FAST-MCD, MVE, OGK,
robust correlation), covariance shrinkage (Ledoit-Wolf, OAS, constant
correlation), and resampling (i.i.d./wild/block/stationary bootstraps) -- 28
spec names across five submodules, natively on numpy/scipy. Location/scale
estimators and regressors reuse :class:`stochpylib.statistics.EstimateResult`/
:class:`~stochpylib.statistics.RegressionResult` as result objects rather
than defining new ones; this module's own additions are the four shared base
classes below.
"""

from stochpylib.robust_statistics._base import (
    Resampler,
    RobustCovarianceEstimator,
    RobustEstimator,
    RobustRegressor,
)
from stochpylib.robust_statistics.location import (
    HodgesLehmann,
    L_Estimator,
    M_Estimator,
    Median,
    R_Estimator,
    TrimmedMean,
    WinsorizedMean,
)
from stochpylib.robust_statistics.scale import (
    IQR_Scale,
    MedianAbsoluteDeviation,
    Qn_Estimator,
    RobustStd,
    Sn_Estimator,
)
from stochpylib.robust_statistics.regression import (
    HuberRegression,
    LTS_Regression,
    MMRegression,
    RANSACRegression,
    SiegalRegression,
    SiegelRegression,
    TheilSenRegression,
)
from stochpylib.robust_statistics.covariance import (
    MCD,
    MVE,
    OGK,
    CovShrinkage,
    RobustCorrelation,
    RobustCovariance,
)
from stochpylib.robust_statistics.bootstrap import (
    BlockBootstrap,
    RobustBootstrap,
    StationaryBootstrap,
    WildBootstrap,
)

__all__ = [
    "BlockBootstrap",
    "CovShrinkage",
    "HodgesLehmann",
    "HuberRegression",
    "IQR_Scale",
    "LTS_Regression",
    "L_Estimator",
    "MCD",
    "MMRegression",
    "MVE",
    "M_Estimator",
    "Median",
    "MedianAbsoluteDeviation",
    "OGK",
    "Qn_Estimator",
    "RANSACRegression",
    "R_Estimator",
    "Resampler",
    "RobustBootstrap",
    "RobustCorrelation",
    "RobustCovariance",
    "RobustCovarianceEstimator",
    "RobustEstimator",
    "RobustRegressor",
    "RobustStd",
    "SiegalRegression",
    "SiegelRegression",
    "Sn_Estimator",
    "StationaryBootstrap",
    "TheilSenRegression",
    "TrimmedMean",
    "WildBootstrap",
    "WinsorizedMean",
]

assert len(__all__) == len(set(__all__)) == 33
