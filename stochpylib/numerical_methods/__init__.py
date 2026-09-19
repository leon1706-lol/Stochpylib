"""Numerical analysis backbone: Gauss quadrature families and adaptive/composite
integration, ODE solvers from explicit Euler through embedded Dormand-Prince and
implicit BDF, SDE path solvers, native linear algebra (matrix exponential/logarithm,
Cholesky, eigendecomposition, SVD, QR, Schur), root finding, interpolation (splines,
PCHIP, barycentric Lagrange, Chebyshev, NURBS), and PDE tools (finite difference,
finite element, a FEniCS-style adapter, boundary elements, spectral methods) -- 38
spec names across six submodules, natively on numpy/scipy (scipy.integrate/optimize/
interpolate/linalg are test oracles only; scipy.special/numpy.linalg are used as
numerical building blocks).
"""

from stochpylib.numerical_methods._result import (
    ODESolution,
    QuadratureResult,
    RootResult,
    SDESolution,
)
from stochpylib.numerical_methods.integration import (
    AdaptiveQuadrature,
    CubatureRule,
    GaussChebyshev,
    GaussHermite,
    GaussLegendre,
    MonteCarloIntegration,
    NumericalIntegration,
)
from stochpylib.numerical_methods.interpolation import (
    BarycentricLagrange,
    Chebyshev,
    CubicHermite,
    Interpolation,
    NURBS,
    SplineInterpolation,
)
from stochpylib.numerical_methods.linear_algebra import (
    CholeskyDecomp,
    EigenDecomp,
    MatrixExponential,
    MatrixLogarithm,
    QRDecomp,
    SVD,
    Schur,
)
from stochpylib.numerical_methods.ode_sde import (
    Adams_Bashforth,
    BDF,
    DormandPrince,
    EulerMethod,
    Euler_Maruyama_SDE,
    Milstein_SDE,
    RungeKutta4,
)
from stochpylib.numerical_methods.pde import (
    BoundaryElement,
    FEniCS_Interface,
    FiniteDifference,
    FiniteElement,
    Mesh,
    SpectralMethod,
)
from stochpylib.numerical_methods.root_solve import (
    Bisection,
    Brent,
    FixedPoint,
    NewtonRaphson,
    RootFinding,
    Secant,
)

__all__ = [
    "Adams_Bashforth",
    "AdaptiveQuadrature",
    "BDF",
    "BarycentricLagrange",
    "Bisection",
    "BoundaryElement",
    "Brent",
    "CholeskyDecomp",
    "Chebyshev",
    "CubatureRule",
    "CubicHermite",
    "DormandPrince",
    "EigenDecomp",
    "EulerMethod",
    "Euler_Maruyama_SDE",
    "FEniCS_Interface",
    "FiniteDifference",
    "FiniteElement",
    "FixedPoint",
    "GaussChebyshev",
    "GaussHermite",
    "GaussLegendre",
    "Interpolation",
    "Mesh",
    "Milstein_SDE",
    "MonteCarloIntegration",
    "MatrixExponential",
    "MatrixLogarithm",
    "NURBS",
    "NewtonRaphson",
    "NumericalIntegration",
    "ODESolution",
    "QRDecomp",
    "QuadratureResult",
    "RootFinding",
    "RootResult",
    "RungeKutta4",
    "SDESolution",
    "SVD",
    "Schur",
    "Secant",
    "SpectralMethod",
    "SplineInterpolation",
]
