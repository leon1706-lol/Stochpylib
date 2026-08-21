# stochpylib.utils

*Infrastructure & utilities*

**Design-completeness score:** 9/10 — GPU backend + parallel simulation + reproducibility = production ready

Status: **planned** — not yet implemented.

## Submodules

### `utils.random`

- `set_seed()`
- `random_state()`
- `Generator`
- `SeedSequence`
- `spawn_generator()`

### `utils.performance`

- `Benchmark`
- `Profiler`
- `ParallelSimulation`
- `GPUBackend`
- `JIT_compile()`
- `VectorizedOps`
- `MemoryPool`

### `utils.reproducibility`

- `Reproducibility`
- `RandomStream`
- `VersionLock`
- `EnvironmentCapture`
- `ExperimentLogger`

### `utils.data`

- `fit()`
- `goodness_of_fit()`
- `moment_matching()`
- `ecdf()`
- `DataValidation`
- `outlier_detection()`
- `missing_imputation()`

### `utils.io`

- `to_dict()`
- `from_dict()`
- `to_json()`
- `from_json()`
- `to_pickle()`
- `Serialization`
- `Configuration`
- `Logging`
- `summary()`

### `utils.compat`

- `numpy_interface()`
- `scipy_interface()`
- `pandas_interface()`
- `torch_interface()`
- `jax_interface()`

---

[[Module-Map]] · [[Ratings]] · [[ARCHITECTURE]]
