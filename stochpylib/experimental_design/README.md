# stochpylib.experimental_design

Design of experiments: classical factorial, screening and response-surface designs, exact
optimal designs, space-filling designs for computer experiments, response-surface and
surrogate models, and the analysis of designed experiments — 29 spec names across five
submodules, natively on numpy/scipy. Every design class returns a `Design` (the run matrix
plus factor names, units, bounds and the design's properties) from `generate()`; surrogates
and analyses are fitted with `.fit(...)` returning `self` and reuse the library's result
types (`statistics.TestResult`/`RegressionResult`, `montecarlo.MCResult`).

**Status:** implemented & tested (29/29 spec names).

```python
from stochpylib.experimental_design import (CCD, FractionalFactorial, NormalPlot,
                                            ResponseSurface, SobolIndex)

ff = FractionalFactorial(5, p=1).generate()        # 2^(5-1), resolution V
ff.properties["resolution"], ff.properties["aliases"]["AB"]   # 5, ["CDE"]

ccd = CCD(2, center=5).generate()                  # rotatable central composite
X = ccd.to_natural([(150, 200), (1.0, 3.0)])       # coded -> natural units
fit = ResponseSurface(order=2, bounds=[(150, 200), (1.0, 3.0)]).fit(X, y)
fit.canonical_analysis()["nature"], fit.stationary_point()

S = SobolIndex(n_samples=4096, random_state=0).analyze(model, bounds=[(-3.14, 3.14)] * 3)
S.first_order_[0].confidence_interval()            # each index is an MCResult
```

## Files

- `_design.py` — `Design`, the one result type: `points` (`(n_runs, n_factors)`),
  `factor_names`, `space` (`"coded"` = `[-1, 1]`, `"unit"` = `[0, 1]`, `"natural"`),
  `bounds`, `kind`, `properties`, `run_order`. `np.asarray(design)` gives the points;
  `to_natural`/`to_coded`, `randomized`, `with_center_points`, `model_matrix`, and the
  quality metrics `d_efficiency`, `min_distance`, `fill_distance`, `discrepancy`,
  `max_abs_correlation`.
- `_base.py` — `DesignGenerator` (seed resolution, `generate()` storing `design_`) and
  `OptimalDesign`, the multi-start modified-Fedorov point-exchange engine over a candidate
  grid, batched over candidates through eigendecompositions of the information matrices.
- `_common.py` — model-term expansion (`"linear"`, `"interaction"`, `"quadratic"`,
  `"cubic"`, `"full"`, exponent tuples or a callable), GF(q) arithmetic for prime powers
  (behind Plackett-Burman, Graeco-Latin squares and orthogonal arrays), mutually orthogonal
  Latin squares, the CD/WD/MD/L2-star discrepancies, distance metrics, rank/correlation
  helpers.
- `classical.py` — `FullFactorial` (Yates order, mixed levels), `FractionalFactorial`
  (explicit generators or a maximum-resolution / minimum-aberration search; defining
  relation, resolution, word-length pattern, alias structure, `fold_over()`),
  `Plackett_Burman` (Sylvester and Paley constructions — the cyclic ones reproduce the
  published N = 12/20/24 generator rows), `CCD` (rotatable/orthogonal/face/inscribed alpha,
  optional fractional cube), `BoxBehnken` (3–7 factors), `LatinSquare` and `GraecoLatin`
  (with `analyze(y)` ANOVA).
- `optimal.py` — `D_OptimalDesign`, `A_OptimalDesign`, `G_OptimalDesign`,
  `I_OptimalDesign` (exact uniform moment matrix for polynomial models), `T_OptimalDesign`
  (Atkinson-Fedorov model discrimination), `BayesianDesign` (`log det(F'F + R)` for linear
  models; pseudo-Bayesian D-optimality over prior draws for nonlinear ones).
- `space_filling.py` — `LatinHypercubeDesign` (delegates to
  `montecarlo.LatinHypercubeSampling`; centred, maximin- or correlation-screened),
  `MaximinLHD` (Morris-Mitchell simulated annealing on `phi_p`), `MinimaxDesign`
  (fill-distance minimization on a Sobol reference set), `UniformDesign` (good lattice
  points minimizing a discrepancy, threshold-accepting fallback), `OrthogonalArrayDesign`
  (Bush construction over GF(s); `lhs=True` gives Tang's OA-based Latin hypercube).
- `response_surface.py` — `MetaModel` (surrogate base; instantiated directly it selects
  the best candidate by k-fold CV), `ResponseSurface` (OLS through
  `statistics.linear_regression`; stationary point, canonical analysis, steepest ascent,
  box-constrained `optimize` via `optimization.DifferentialEvolution`), `RSM_ANOVA`
  (sequential linear/interaction/quadratic SS, lack of fit vs pure error, PRESS-based
  predicted R²), `PolynomialChaos` (Legendre/Hermite bases, least-squares or tensor Gauss
  quadrature via `numerical_methods`, analytic mean/variance/Sobol indices),
  `KrigingSurrogate` (universal kriging on `gaussian_processes` kernels, GLS trend,
  trend-aware variance, expected improvement, sequential design).
- `analysis.py` — `ANOVA_DOE` (contrast SS for orthogonal two-level designs, Type II SS
  otherwise), `MainEffects`, `InteractionPlot`, `NormalPlot` (Daniel plot + Lenth's
  PSE/ME/SME), `SensitivityIndex` (Morris, SRC, PRCC, correlation), `SobolIndex`
  (Saltelli 2010 first-order, Jansen total-order, Saltelli 2002 second-order, bootstrap
  standard errors as `MCResult`s).

## Conventions

- **Coded, unit and natural spaces.** Classical and optimal designs are coded to
  `[-1, 1]` (an m-level factor is `linspace(-1, 1, m)`), space-filling designs live in
  `[0, 1]^d`; `Design.to_natural(bounds)` maps either onto real units.
- **`generate()` returns a `Design`, `.fit()` returns `self`.** Every stochastic
  generator takes `random_state=` (constructor, or a `generate()` override) and reproduces
  a design exactly from a seed. Two-level designs come in Yates standard order; call
  `Design.randomized()` for a run order.
- **Factor names skip `I`** (`A B C D E F G H J K …`): `I` is the identity word of a
  defining relation. Effect labels concatenate letters (`"AC"`); model-term labels use
  `"A*C"`.
- **No new result types beyond `Design`.** ANOVA tables are `statistics.TestResult`s,
  response-surface fits carry a `statistics.RegressionResult`, Sobol indices are
  `montecarlo.MCResult`s.
- **Delegation, not reimplementation.** Latin hypercubes come from `montecarlo`, OLS from
  `statistics`, kernels/Cholesky/hyperparameters from `gaussian_processes`, Gauss rules
  from `numerical_methods`, box optimization from `optimization`. Nothing wraps
  `scipy.stats` or `scipy.optimize` (a test enforces it).

## Known limitations

- **`InteractionPlot` and `NormalPlot` are data-only at their core.** They compute the
  cell means, effect quantiles and Lenth thresholds plus a text table (`to_text()`);
  `to_figure()` renders them via `stochpylib.viz` (a lazy import, so this module itself
  stays free of any plotting dependency).
- **Construction coverage.** `BoxBehnken` is tabulated for 3–7 factors; `GraecoLatin`
  covers every order not congruent to 2 mod 4 (orders 2 and 6 are impossible; 10, 14, …
  exist but are not constructed); `Plackett_Burman` skips orders with no Sylvester/Paley
  construction here (N = 36, 52, …) to the next constructible N.
- **`FractionalFactorial`'s generator search** is exhaustive up to 200 000 generator
  sets and a seeded random search of 20 000 beyond that, so very large fractions are not
  guaranteed minimum aberration.
- **Exchange algorithms find local optima.** The optimal designs are multi-start point
  exchange over a finite candidate grid, not a global search; raise `n_starts` or refine
  `levels` for hard problems.
- **`MinimaxDesign`'s fill distance is measured on a Sobol reference set**, so it is a
  close lower bound on the true value.
- **`PolynomialChaos` uses an isoprobabilistic Hermite transform** for inputs that are
  neither `Uniform` nor `Normal`; convergence can be slower than with the family's own
  Askey polynomials.
- **`KrigingSurrogate` fits its hyperparameters on OLS-detrended data**, then solves the
  trend by GLS — the usual two-step approximation, not a joint concentrated likelihood.
- **No autodiff.** `BayesianDesign` differentiates nonlinear models by central finite
  differences unless a `jacobian=` is supplied.

Spec: vault `Modules/experimental_design.md` (private). Tests:
`tests/experimental_design/tests.py` (oracles: statsmodels OLS/ANOVA,
`scipy.stats.qmc.discrepancy`, `scipy.stats.sobol_indices`, `scipy.stats.norm`,
brute-force enumeration of exact optimal designs, published constructions and textbook
data — Montgomery's filtration-rate 2^4, Plackett & Burman's generator rows, the Ishigami
function's analytic Sobol indices, Box-Lucas' nonlinear optimal design) and
`tests/experimental_design/e2e.py` (API sweep).
