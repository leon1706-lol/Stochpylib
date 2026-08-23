"""Time-series toolkit: linear models, volatility, state space, latent regimes,
changepoints, spectral analysis, and forecasting.

Public names (spec: ``Modules/timeseries.md``) are re-exported at module level per
ARCHITECTURE.md::

    from stochpylib.timeseries import ARIMA, GARCH, KalmanFilter

Conventions: fluent ``fit(...)`` returning self; ``forecast(horizon=...)`` returning a
:class:`ForecastResult`; ``random_state=`` for all simulation paths. Estimation is
native (numpy/scipy); ``statsmodels`` is a dev-only test oracle.
"""

from stochpylib.timeseries._result import ForecastResult, TestResult

from stochpylib.timeseries.linear_models import (
    AR, ARFIMA, ARIMA, ARMA, MA, SARIMA, VECM, VAR, VARMA,
)

from stochpylib.timeseries.volatility_models import (
    ARCH, APARCH, DCC_GARCH, EGARCH, FIGARCH, GARCH, GJRGARCH, IGARCH, MGARCH,
    TGARCH,
)

from stochpylib.timeseries.state_space import (
    ExtendedKalmanFilter, KalmanFilter, KalmanSmoother, ParticleFilter,
    RaoBlackwellFilter, StateSpaceModel, UnscentedKalmanFilter,
)

from stochpylib.timeseries.latent import (
    HiddenMarkovModel, MixtureAutoregressive, RegimeSwitching, SwitchingRegression,
)

from stochpylib.timeseries.changepoint import (
    BayesianChangePoint, BinarySegmentation, BottomUp, ChangePointDetection, PELT,
)
from stochpylib.timeseries.changepoint import BOCPDResult, ChangePointResult

from stochpylib.timeseries.spectral import (
    CWTTransform, DWTTransform, Hilbert, IDWTTransform, Periodogram, PowerSpectrum,
    SpectralAnalysis, STFT, WaveletTransform,
)

from stochpylib.timeseries.decomposition import (
    DecompositionResult, HPFilter, SeasonalDecomposition, STLDecomposition,
    TrendFilter, X11Decomposition,
)

from stochpylib.timeseries import tests as _tests_module
from stochpylib.timeseries.tests import (
    adf_test, arch_test, durbin_watson, granger_causality, johansen_test,
    kpss_test, ljung_box, pp_test,
)

from stochpylib.timeseries.forecasting import (
    backtesting, confidence_bands, cross_validation_ts, forecast, predict,
)

__all__ = [
    # linear models
    "AR", "MA", "ARMA", "ARIMA", "SARIMA", "ARFIMA", "VARMA", "VAR", "VECM",
    # volatility
    "ARCH", "GARCH", "IGARCH", "TGARCH", "GJRGARCH", "EGARCH", "APARCH", "FIGARCH",
    "MGARCH", "DCC_GARCH",
    # state space
    "StateSpaceModel", "KalmanFilter", "KalmanSmoother", "ExtendedKalmanFilter",
    "UnscentedKalmanFilter", "ParticleFilter", "RaoBlackwellFilter",
    # latent
    "HiddenMarkovModel", "SwitchingRegression", "RegimeSwitching",
    "MixtureAutoregressive",
    # changepoint
    "ChangePointDetection", "BayesianChangePoint", "BinarySegmentation", "BottomUp",
    "PELT",
    # spectral
    "SpectralAnalysis", "Periodogram", "PowerSpectrum", "WaveletTransform",
    "CWTTransform", "DWTTransform", "STFT", "Hilbert",
    # decomposition
    "SeasonalDecomposition", "STLDecomposition", "X11Decomposition", "TrendFilter",
    "HPFilter",
    # tests submodule functions
    "adf_test", "kpss_test", "pp_test", "ljung_box", "durbin_watson", "arch_test",
    "granger_causality", "johansen_test",
    # forecasting
    "forecast", "predict", "confidence_bands", "backtesting", "cross_validation_ts",
    # result types
    "ForecastResult", "TestResult", "ChangePointResult", "BOCPDResult",
]
