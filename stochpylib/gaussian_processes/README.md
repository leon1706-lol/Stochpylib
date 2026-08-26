# stochpylib.gaussian_processes

A composable kernel zoo with exact, sparse and approximate-classification
inference: 10 covariance functions combined via `+` / `*` / `**` operator
overloading, exact GP regression, FITC/VFE sparse approximations, Laplace/EP/VI
classification, and L-BFGS hyperparameter optimization over the flattened
parameter tree.

**Status:** implemented & tested (36/36 spec names).

## Files

- `kernels/` — RBF (incl. ARD), Matérn closed forms (nu in {0.5, 1.5, 2.5}),
  Periodic, Linear, Polynomial, RationalQuadratic, WhiteNoise, SpectralMixture,
  NeuralNetwork (RW eq. 4.29–4.31), ArcCosine — all callable and composable.
- `kernel_ops.py` — `KernelSum`/`KernelProduct`/`KernelPower`/`KernelComposition`
  with flattened `part<i>__<name>` parameter trees for optimizers;
  `kernel_matrix`, finite-difference `kernel_grad`.
- `models.py` — `ExactInference` (jittered Cholesky + analytic log-ML),
  `GaussianProcess`, `GPRegression`, `GPTimeSeriesModel`, and the spec-facing
  `GPClassification` facade over the classification engines.
- `inference.py` — binary classification engines: `LaplacePropagation`
  (RW Alg. 3.1, logit/probit), `ExpectationPropagation` (**experimental**, may
  not converge — prefer Laplace or VI), `VariationalInference`
  (Jaakkola-Jordan bound, logit only).
- `sparse.py` — FITC (`InducingPointGP`) and VFE/Titsias SGPR
  (`SparseVFE`, `SparseGaussianProcess`) in a **whitened parameterization**: all
  solves through jittered Cholesky factors; verified against a brute-force
  Titsias reference and the M=T-is-exact identity to machine precision
  (`development/Probleme.md` [23]).
- `deep_gp.py` — documented two-layer composition (sparse latent → observed).
- `hyperparams.py` — `MarginalLikelihood`, log-space L-BFGS
  `optimize_hyperparams()` walking the kernel parameter tree, `ARD`,
  `cross_validate_gp`.

## Conventions

- Models expose fluent `.fit()` returning self; predictions return means plus
  pointwise std (full covariance with `full_cov=True`); forecasts wrap in the
  shared `ForecastResult`; everything runs on native numpy/scipy.

## Known limitations

- EP is experimental ([20]) — documented convergence issues, not recommended
  for production use.
- GP regression means revert to the prior under long-horizon extrapolation —
  trust the growing predictive std, not the point forecast.
- Sparse approximations converge to exact as inducing points cover the domain;
  small M on long ranges underfits silently (check the log-ML).
- DeepGP is a composition recipe with shared-kernel hyperparameters, not an
  independently parameterized multi-layer model.

Spec: vault `Modules/gaussian_processes.md` (private). Tests:
`tests/gaussian_processes/tests.py`.
