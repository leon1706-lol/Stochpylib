# stochpylib.numerical_methods

*Numerical analysis backbone*

**Design-completeness score:** 8/10 — Quadrature, FEM, FDM — solid backbone for simulation

Status: **planned** — not yet implemented.

## Submodules

### `numerical_methods.integration`

- `NumericalIntegration`
- `AdaptiveQuadrature`
- `GaussLegendre`
- `GaussHermite`
- `GaussChebyshev`
- `MonteCarloIntegration`
- `CubatureRule`

### `numerical_methods.ode_sde`

- `EulerMethod`
- `RungeKutta4`
- `DormandPrince`
- `Adams_Bashforth`
- `BDF`
- `Euler_Maruyama_SDE`
- `Milstein_SDE`

### `numerical_methods.linear_algebra`

- `MatrixExponential`
- `MatrixLogarithm`
- `CholeskyDecomp`
- `EigenDecomp`
- `SVD`
- `QRDecomp`
- `Schur`

### `numerical_methods.root_solve`

- `RootFinding`
- `Bisection`
- `NewtonRaphson`
- `Brent`
- `Secant`
- `FixedPoint`

### `numerical_methods.interpolation`

- `Interpolation`
- `SplineInterpolation`
- `CubicHermite`
- `BarycentricLagrange`
- `Chebyshev`
- `NURBS`

### `numerical_methods.pde`

- `FiniteDifference`
- `FiniteElement`
- `FEniCS_Interface`
- `BoundaryElement`
- `SpectralMethod`

---

[[Module-Map]] · [[Ratings]] · [[ARCHITECTURE]]
