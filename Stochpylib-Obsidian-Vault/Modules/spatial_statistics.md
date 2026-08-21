# stochpylib.spatial_statistics

*Geostatistics & spatial modeling*

**Design-completeness score:** 8/10 — Kriging + variogram + point processes — solid but could add CAR/SAR

Status: **planned** — not yet implemented.

## Submodules

### `spatial_statistics.random_fields`

- `GaussianRandomField`
- `MaternField`
- `OrnsteinUhlenbeckField`
- `BrownianSheet`
- `FractionalBrownianSheet`

### `spatial_statistics.kriging`

- `Kriging`
- `OrdinaryKriging`
- `UniversalKriging`
- `SimpleKriging`
- `CoKriging`
- `IndicatorKriging`
- `DisjunctiveKriging`

### `spatial_statistics.variogram`

- `Variogram`
- `Semivariogram`
- `SpatialCovariance`
- `ExperimentalVariogram`
- `VariogramFitting`
- `Nugget`
- `Sill`
- `Range`

### `spatial_statistics.point_processes`

- `SpatialPointProcess`
- `PoissonPointProcess`
- `ThomasProcess`
- `MaternCluster`
- `LogGaussianCox`
- `InhomogeneousPoisson`
- `RipleyK()`
- `PairCorrelation()`

### `spatial_statistics.tests`

- `MoransI()`
- `GearyC()`
- `SpatialAutocorrelation()`
- `NNDistanceTest()`

---

[[Module-Map]] · [[Ratings]] · [[ARCHITECTURE]]
