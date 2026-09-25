"""Design of experiments: classical factorial, screening and response-surface designs,
exact optimal designs, space-filling designs for computer experiments, response-surface
and surrogate models, and the analysis of designed experiments -- 29 spec names across
five submodules, natively on numpy/scipy.

Every design class subclasses :class:`~stochpylib.experimental_design.DesignGenerator`
and returns a :class:`~stochpylib.experimental_design.Design` from ``generate()``: the run
matrix in coded (``[-1, 1]``) or unit-cube (``[0, 1]``) units plus its factor names,
natural-unit bounds and properties (resolution, alpha, efficiencies, discrepancy, ...).
Optimal designs share the :class:`~stochpylib.experimental_design.OptimalDesign` exchange
engine. Surrogates subclass :class:`MetaModel` and ``fit(X, y)`` returning ``self``;
analyses reuse the library's result types (``statistics.TestResult``/``RegressionResult``,
``montecarlo.MCResult`` for Sobol indices) instead of defining new ones. Nothing wraps
``scipy.stats`` or ``scipy.optimize``; they are the test suite's oracles only.
"""

from stochpylib.experimental_design._base import DesignGenerator, OptimalDesign
from stochpylib.experimental_design._design import Design
from stochpylib.experimental_design.classical import (
    BoxBehnken,
    CCD,
    FractionalFactorial,
    FullFactorial,
    GraecoLatin,
    LatinSquare,
    Plackett_Burman,
)
from stochpylib.experimental_design.optimal import (
    A_OptimalDesign,
    BayesianDesign,
    D_OptimalDesign,
    G_OptimalDesign,
    I_OptimalDesign,
    T_OptimalDesign,
)
from stochpylib.experimental_design.space_filling import (
    LatinHypercubeDesign,
    MaximinLHD,
    MinimaxDesign,
    OrthogonalArrayDesign,
    UniformDesign,
)
from stochpylib.experimental_design.response_surface import (
    KrigingSurrogate,
    MetaModel,
    PolynomialChaos,
    RSM_ANOVA,
    ResponseSurface,
)
from stochpylib.experimental_design.analysis import (
    ANOVA_DOE,
    InteractionPlot,
    MainEffects,
    NormalPlot,
    SensitivityIndex,
    SobolIndex,
)

__all__ = [
    "ANOVA_DOE",
    "A_OptimalDesign",
    "BayesianDesign",
    "BoxBehnken",
    "CCD",
    "D_OptimalDesign",
    "Design",
    "DesignGenerator",
    "FractionalFactorial",
    "FullFactorial",
    "G_OptimalDesign",
    "GraecoLatin",
    "I_OptimalDesign",
    "InteractionPlot",
    "KrigingSurrogate",
    "LatinHypercubeDesign",
    "LatinSquare",
    "MainEffects",
    "MaximinLHD",
    "MetaModel",
    "MinimaxDesign",
    "NormalPlot",
    "OptimalDesign",
    "OrthogonalArrayDesign",
    "Plackett_Burman",
    "PolynomialChaos",
    "RSM_ANOVA",
    "ResponseSurface",
    "SensitivityIndex",
    "SobolIndex",
    "T_OptimalDesign",
    "UniformDesign",
]

assert len(__all__) == len(set(__all__)) == 32
