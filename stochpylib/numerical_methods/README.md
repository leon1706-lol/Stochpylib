# stochpylib.numerical_methods

Numerical analysis backbone: Gauss quadrature families (Legendre/Hermite/Chebyshev via
Golub-Welsch), adaptive Gauss-Kronrod and Simpson quadrature, Romberg extrapolation,
tensor/Smolyak-sparse cubature, and a Monte-Carlo facade; ODE solvers from explicit
Euler through embedded Dormand-Prince, multistep Adams-Bashforth(-Moulton), and
implicit BDF; Euler-Maruyama/Milstein SDE path solvers; native linear algebra (Pade
matrix exponential, inverse-scaling matrix logarithm, Cholesky, Jacobi/QR
eigendecomposition, one-sided-Jacobi SVD, Householder/Gram-Schmidt/Givens QR,
Francis-shift real and complex Schur); root finding (bisection, Brent, secant, Newton,
fixed point); interpolation (cubic splines, PCHIP-style cubic Hermite, barycentric
Lagrange, Chebyshev series, NURBS); and PDE tools (finite difference, finite element,
a FEniCS-style adapter with a native fallback, boundary elements, spectral methods).
38 public names across six submodules, natively on numpy/scipy — no wrapper
dependencies.

**Status:** implemented & tested (38/38 spec names).

## Files

- `_common.py` — private helpers: seeded RNG, scalar/vectorized-integrand wrapper,
  infinite-interval change of variables, tridiagonal (Thomas) solve, finite-difference
  gradient/Jacobian, matrix-free conjugate gradient, Householder vectors, Hessenberg
  reduction, the published Gauss-Kronrod 7-15 node/weight tables.
- `_result.py` — `QuadratureResult` (value + error estimate), `RootResult`,
  `ODESolution` (grid + callable dense output, linear or cubic-Hermite), `SDESolution`
  (grid + per-path states + driving Brownian increments).
- `integration.py` — `GaussLegendre`/`GaussHermite`/`GaussChebyshev` (Golub-Welsch),
  `AdaptiveQuadrature` (QUADPACK-style adaptive Gauss-Kronrod, or adaptive Simpson;
  infinite/singular endpoints via a change of variables), `NumericalIntegration`
  (dispatcher: adaptive/trapezoid/simpson/romberg/gauss_legendre/gauss_kronrod/
  monte_carlo), `MonteCarloIntegration` (delegates sampling to
  `stochpylib.montecarlo`), `CubatureRule` (tensor-product or Smolyak sparse grids).
- `ode_sde.py` — `EulerMethod`, `RungeKutta4`, `DormandPrince` (embedded RK5(4)7FM,
  adaptive step, cubic-Hermite dense output), `Adams_Bashforth` (orders 1-5, optional
  PECE corrector), `BDF` (implicit, orders 1-5, Newton corrector, order ramp on
  startup — order 1 is backward Euler), `Euler_Maruyama_SDE`, `Milstein_SDE`
  (diagonal/commutative noise).
- `linear_algebra.py` — `MatrixExponential` (degree-13 Pade scaling-and-squaring,
  or eigendecomposition/Taylor fallbacks), `MatrixLogarithm` (inverse
  scaling-and-squaring via Denman-Beavers square roots + a Gauss-Legendre
  partial-fraction Pade evaluation), `CholeskyDecomp`, `EigenDecomp` (cyclic Jacobi for
  symmetric matrices, Hessenberg + shifted QR for general), `SVD` (one-sided Jacobi),
  `QRDecomp` (Householder/modified Gram-Schmidt/Givens), `Schur` (Francis
  double-shift QR for the real form, single complex-Wilkinson-shift QR for the
  complex form).
- `root_solve.py` — `Bisection`, `Brent` (Brent-Dekker), `Secant`, `NewtonRaphson`
  (scalar or vector, analytic or finite-difference derivative/Jacobian, damped),
  `FixedPoint` (plain, Aitken, or Steffensen acceleration), `RootFinding` (dispatcher
  + `find_bracket` expanding-interval search + `all_roots` grid scan).
- `interpolation.py` — `SplineInterpolation` (natural/clamped/not-a-knot cubic
  spline), `CubicHermite` (PCHIP/Fritsch-Carlson slopes by default, or given slopes),
  `BarycentricLagrange` (+ Chebyshev-point construction), `Chebyshev` (series via the
  Clenshaw recurrence; derivative/integral/roots via the colleague matrix), `NURBS`
  (Cox-de Boor recursion; `circle()`/`interpolate()` constructors), `Interpolation`
  (facade: linear/nearest/polynomial/cubic/pchip/spline_natural).
- `pde.py` — `Mesh` (1-D interval / 2-D triangulated rectangle), `FiniteDifference`
  (Fornberg stencils, 1-D/2-D Poisson, 1-D heat with explicit/implicit/Crank-Nicolson
  schemes, a Black-Scholes PDE pricer with an American early-exercise projection, 1-D
  advection with upwind/Lax-Wendroff), `FiniteElement` (1-D P1/P2 Galerkin, 2-D P1 on
  triangles), `FEniCS_Interface` (solves natively via `FiniteElement` by default;
  `.export_mesh()` writes DOLFIN-XML/XDMF; `.to_fenics()` lazily hands the mesh to a
  real dolfinx/dolfin install if present, else raises a clear `ImportError`),
  `BoundaryElement` (2-D interior Laplace, constant elements), `SpectralMethod`
  (Fourier derivative/Poisson/heat on periodic domains, Chebyshev-collocation BVP
  solver).

## Conventions

- **Scalar vs vectorized integrands/functions.** Every quadrature and interpolation
  entry point defaults to a plain scalar callable (`f(x) -> float`), wrapped
  internally with `np.vectorize`; pass `vectorized=True` when your function already
  accepts/returns arrays for a real speedup.
- **Fluent linear algebra.** `MatrixExponential`/`MatrixLogarithm`/`CholeskyDecomp`/
  `EigenDecomp`/`SVD`/`QRDecomp`/`Schur` all follow `.compute()` (or, for Cholesky,
  compute in `__init__`) returning `self`, with `_`-suffixed result attributes — the
  same fluent-fit convention as the rest of the library.
- **Result objects carry uncertainty or diagnostics.** `QuadratureResult` never ships
  a value without an error estimate and a `converged` flag; `ODESolution`/
  `SDESolution` always carry the grid and diagnostics (`n_evals`, `n_steps`,
  `success`) alongside the trajectory.
- **Native algorithms, `numpy.linalg` as an explicit fast path.** Every linear-algebra
  class implements its own algorithm (Golub-Welsch, Jacobi, Householder QR, Francis
  QR, Denman-Beavers, Padé) and additionally exposes `method="numpy"` where that is a
  meaningful alternative; `scipy.integrate/optimize/interpolate/linalg` are test
  oracles only, never imported in this package.
- **`random_state=`** on every stochastic method (`Euler_Maruyama_SDE`,
  `Milstein_SDE`, `MonteCarloIntegration`), matching the rest of the library; SDE
  solvers additionally accept a shared `brownian=` array so strong-convergence
  studies can compare schemes on the identical driving path.

## Known limitations

- **`DormandPrince`'s dense output is cubic Hermite** (using the stored slopes at
  accepted steps), not DP's own built-in 5th-order interpolant — 3rd/4th-order
  accurate between grid points, which is what `t_eval` uses.
- **`BDF` is fixed-step** (no local error control / step-size adaptation) — pick `h`
  or `n_steps` conservatively for stiff problems; order 1 is backward Euler.
- **`Milstein_SDE` supports diagonal (commutative) noise only** — general
  correlated/multiplicative noise needs `Euler_Maruyama_SDE(noise="general")`
  (strong order 0.5) instead.
- **`FiniteElement`'s 2-D path is P1-only** on a triangular mesh; 1-D supports P1/P2.
- **`BoundaryElement` solves the 2-D interior Laplace equation only** (constant
  elements, direct BEM) — not a general elliptic/Helmholtz BEM.
- **`FEniCS_Interface.to_fenics()` is best-effort** — it hands off mesh construction
  to a real dolfinx/dolfin install when present (never a runtime dependency: a clear
  `ImportError` otherwise) and has not been exercised against a real FEniCS install in
  this repository's CI; `.solve()` (the native `FiniteElement` fallback) is the
  supported, tested path for actually solving a problem.
- **`MatrixLogarithm` requires no eigenvalue on `(-inf, 0]`** (principal branch only);
  raises `ValueError` otherwise.
- **`Schur`'s QR iteration caps at `30 * n` sweeps** before returning whatever it has
  (no exception) — pathological matrices may not fully deflate in that budget.

Spec: vault `Modules/numerical_methods.md` (private). Tests:
`tests/numerical_methods/tests.py` (oracles) and `tests/numerical_methods/e2e.py`
(API sweep).
