# stochpylib.spatial_statistics

Geostatistics and spatial modeling: variogram models and fitting, kriging (simple,
ordinary, universal, co-, indicator, disjunctive), covariance-driven random fields (exact
Cholesky/circulant-embedding sampling, plus Brownian/fractional-Brownian sheets), spatial
point processes (Poisson, Thomas, Matern cluster, log-Gaussian Cox) with Ripley's K and the
pair correlation function, and spatial autocorrelation tests (Moran's I, Geary's C,
Getis-Ord, nearest-neighbour distance) — 32 spec names across five submodules, natively on
numpy/scipy. `SARModel`/`CARModel` (lattice models) and `SpatialWeights`/`SpatialFunction`
are documented extras beyond the spec.

**Status:** implemented & tested (32/32 spec names).

```python
import numpy as np
from stochpylib.spatial_statistics import (
    ExperimentalVariogram, VariogramFitting, OrdinaryKriging, MoransI, SpatialWeights,
    RipleyK,
)

coords = np.random.default_rng(0).uniform(0, 10, size=(80, 2))
values = np.random.default_rng(1).normal(size=80)

ev = ExperimentalVariogram(bins=12).fit(coords, values)
vf = VariogramFitting(model=["spherical", "exponential", "gaussian"]).fit(ev)

ok = OrdinaryKriging(variogram=vf.variogram_).fit(coords, values)
mean, std = ok.predict([[5.0, 5.0]], return_std=True)   # mirrors GPRegression.predict

k = RipleyK(coords, window=((0, 10), (0, 10)), n_simulations=99, random_state=2)
k.pvalue                                                  # CSR envelope test

W = SpatialWeights.knn(coords, k=6)
mi = MoransI(values, W)                                   # -> statistics.TestResult
mi.statistic, mi.pvalue
```

## Files

- `_common.py` — `_as_coords`/`_as_values`, `_box_window` (normalizes a window to `(d, 2)`
  bounds plus its volume), `_ball_volume` (d-dimensional ball volume via `scipy.special`),
  `_lag_bins`. Pairwise/nearest-neighbour distances are reused (not duplicated) from
  `experimental_design._common`.
- `_result.py` — `SpatialFunction`, the one new result type: `r`/`estimate`/`theoretical`
  plus an optional Monte Carlo envelope (`lower`/`upper`/`pvalue`) and `L()` (Besag's
  variance-stabilizing transform).
- `weights.py` — `SpatialWeights` (`knn`, `distance_band`, `inverse_distance`, `kernel`,
  `lattice` constructors; `row_standardize`, `symmetrize`, the Cliff & Ord `S0`/`S1`/`S2`
  moment sums).
- `variogram.py` — `Variogram` (abstract; `+` nests structures), `Semivariogram`
  (spherical/exponential/gaussian/matern/cubic/linear/power/nugget, with a general-nu Matern
  via `scipy.special.kv`), `SpatialCovariance` (`to_kernel()`/`from_kernel()` bridge to
  `gaussian_processes`), `ExperimentalVariogram` (Matheron/Cressie-Hawkins/Dowd estimators,
  optional directional binning), `VariogramFitting` (weighted least squares, or the best of
  several models by AIC), `Nugget`/`Sill`/`Range` curve-based estimators.
- `kriging.py` — `Kriging`/`OrdinaryKriging`/`UniversalKriging` share one gamma-Lagrange
  linear system (so unbounded power/linear variograms work); `SimpleKriging` uses the
  covariance form directly; `CoKriging` uses the Markov Model 1 simplification (see
  Conventions); `IndicatorKriging` (order-relation-corrected ccdf); `DisjunctiveKriging`
  (Gaussian anamorphosis, Hermite expansion via `numerical_methods.GaussHermite`).
- `random_fields.py` — `GaussianRandomField` (covariance-driven; Cholesky on scattered
  points, exact circulant embedding on grids, or `method="spectral"` delegating to
  `levy_processes.GaussianRandomField`'s FFT synthesis), `MaternField` (general-nu),
  `OrnsteinUhlenbeckField` (exact per-axis AR(1) on separable grids), `BrownianSheet`,
  `FractionalBrownianSheet` (exact via the separable Kronecker Cholesky structure).
- `point_processes.py` — `SpatialPointProcess` base (`sample`/`intensity`/`K`/`pcf`/`fit`,
  Diggle's `K**(1/4)` minimum-contrast helper), `PoissonPointProcess`,
  `InhomogeneousPoisson` (Lewis-Shedler thinning; log-linear MLE), `ThomasProcess`,
  `MaternCluster`, `LogGaussianCox`; `RipleyK`/`PairCorrelation` are plain functions
  returning a `SpatialFunction`, with translation, quadrature-evaluated isotropic
  ("ripley"), and border edge corrections.
- `tests.py` — `MoransI`, `GearyC` (Cliff & Ord normality/randomization variances, optional
  permutation p-value), `SpatialAutocorrelation` (global or local Moran/Geary/Getis-Ord),
  `NNDistanceTest` (Clark-Evans R with the Donnelly edge correction, or the G-function CSR
  envelope test). All four are plain functions returning a `statistics.TestResult`.
- `lattice.py` — `SARModel`/`CARModel` (extras): concentrated-likelihood spatial lag /
  conditional autoregressive models, log-det via the weights matrix's eigenvalues, standard
  errors from the numerical Hessian (`statistics._common._numeric_hessian`).

## Conventions

- **`fit(...)` returns `self`.** Every stochastic method takes `random_state=`.
- **No new result types beyond `SpatialFunction`.** Kriging predictions reuse
  `timeseries.ForecastResult` (`predict_result(X)`); autocorrelation and nearest-neighbour
  tests reuse `statistics.TestResult`.
- **`Semivariogram`'s sill is the total sill (nugget included)**; `partial_sill` is
  `sill - nugget`. `range` is the *practical* range for spherical/exponential/gaussian/cubic
  (gamma reaches ~95% of the partial sill there) but the raw correlation length scale for
  `"matern"` (no natural single practical-range rescaling for general nu). `"linear"`/
  `"power"` reinterpret `sill` as the slope coefficient (unbounded models have no sill).
- **Windows are boxes**: `((x0, x1), (y0, y1), ...)`, any dimension. Point processes and
  `RipleyK`/`PairCorrelation`'s exact isotropic correction assume a 2-D rectangle only
  where noted; translation and border corrections work in any dimension.
- **`spatial_statistics.GaussianRandomField` is a separate class from
  `levy_processes.GaussianRandomField`.** The levy one is FFT spectral synthesis on a grid
  from a prescribed power spectrum; this one is covariance-driven (Cholesky/circulant) and
  only calls into the levy class when `method="spectral"` is explicitly requested. `spl show
  GaussianRandomField` lists both owners; use `spl show <module>.GaussianRandomField` (or
  the qualified import) to disambiguate.
- **`CoKriging` uses the Markov Model 1 simplification**: the cross-covariance is
  proportional to the primary's own correlation function (`rho * sqrt(sill1*sill2) *
  corr1(h)`), guaranteeing a valid model with no separate cross-variogram fit, at the cost
  of assuming the two variables share a spatial range — a standard, documented
  geostatistical simplification (Journel 1999), not the fully general linear model of
  coregionalization.
- **Cluster-process fitting is by minimum contrast on K, not maximum likelihood** —
  `mu` (offspring count) is not identifiable from K alone (K only depends on the clustering
  shape, not the overall intensity), so it is derived afterward from the fitted `kappa` and
  the observed count: `mu_hat = (n / |W|) / kappa_hat`.

## Known limitations

- **`VariogramFitting` bounds `range` at 3x the maximum observed lag and `partial_sill`
  (or, for `"linear"`/`"power"`, the slope) at 5x the maximum observed gamma level.**
  Weakly-structured (near-nugget) data leaves an unidentifiable flat ridge in that space;
  without a cap the optimizer can report an arbitrarily large, physically meaningless
  range (Probleme.md #119). A fit sitting at either bound signals "not resolved by this
  data", not a real practical range.
- **Box windows only** — no polygonal/irregular observation windows.
- **`RipleyK`/`PairCorrelation`'s `correction="ripley"` (exact isotropic edge correction)
  and directional `ExperimentalVariogram` binning are 2-D only.** The isotropic correction
  is evaluated by angular quadrature (180 points) rather than the closed-form corner
  geometry — accurate to ~1/180 of the circle, not bit-exact.
- **`CoKriging` supports one secondary variable** under the Markov Model 1 assumption (see
  Conventions); it is not the general multivariable linear model of coregionalization.
- **Dense global kriging/random-field systems are O(n^3).** Use `n_neighbors=` (local
  kriging) or `sample_grid`'s circulant embedding for larger problems.
- **No space-time kriging or random fields** — every model here is purely spatial.
- **`DisjunctiveKriging`'s Gaussian anamorphosis is estimated from data ranks.** For
  strongly spatially correlated data the effective sample size is well below `n`, so the
  anamorphosis (and properties like "reduces exactly to simple kriging under an identity
  map") converge slowly with `n`; verified in tests with a domain large relative to the
  variogram range.
- **`InhomogeneousPoisson`/`ThomasProcess`/`MaternCluster`/`LogGaussianCox` fit by numerical
  MLE or minimum contrast**, not a closed form — results depend on the optimizer's local
  search (multi-start, deterministically seeded) and on the empirical `RipleyK` used as the
  fitting target.
- **`SARModel`'s log-det uses the real part of `W`'s (possibly complex) eigenvalues** for a
  general asymmetric weights matrix — exact for the common symmetric/row-standardized case,
  a practical approximation otherwise. `CARModel` requires a symmetric `W`.

Spec: vault `Modules/spatial_statistics.md` (private). Tests:
`tests/spatial_statistics/tests.py` (oracles: `scipy.special.kv`/`scipy.stats.poisson`,
brute-force pairwise formulas, cross-module identities against `gaussian_processes`,
`experimental_design.KrigingSurrogate`, `levy_processes.GaussianRandomField`, and
`scipy.integrate` numerical checks) and `tests/spatial_statistics/e2e.py` (API sweep).
