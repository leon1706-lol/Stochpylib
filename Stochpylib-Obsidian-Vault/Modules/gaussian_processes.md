# stochpylib.gaussian_processes

*Gaussian process models & kernels*

**Design-completeness score:** 10/10 — Full kernel zoo, sparse GP, regression & classification — research-grade

Status: **planned** — not yet implemented.

## Submodules

### `gaussian_processes.models`

- `GaussianProcess`
- `GPRegression`
- `GPClassification`
- `GPTimeSeriesModel`
- `SparseGaussianProcess`
- `InducingPointGP`
- `DeepGP`

### `gaussian_processes.kernels`

- `RBFKernel`
- `MaternKernel`
- `PeriodicKernel`
- `LinearKernel`
- `PolynomialKernel`
- `RationalQuadraticKernel`
- `WhiteNoiseKernel`
- `SpectralMixtureKernel`
- `NeuralNetworkKernel`
- `ArcCosineKernel`

### `gaussian_processes.kernel_ops`

- `KernelComposition`
- `KernelSum`
- `KernelProduct`
- `KernelPower`
- `StationaryKernel`
- `NonStationaryKernel`
- `kernel_matrix()`
- `kernel_grad()`

### `gaussian_processes.inference`

- `ExactInference`
- `LaplacePropagation`
- `ExpectationPropagation`
- `VariationalInference`
- `FITC`
- `VFE`
- `SparseVFE`

### `gaussian_processes.hyperparams`

- `MarginalLikelihood`
- `ARD`
- `optimize_hyperparams()`
- `cross_validate_gp()`

---

[[Module-Map]] · [[Ratings]] · [[ARCHITECTURE]]
