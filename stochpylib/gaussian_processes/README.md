# stochpylib.gaussian_processes — GP models, kernels & inference

**Status: implemented and tested** (36/36 spec names).

Layout:

- `kernels/` — the covariance zoo: RBF (incl. ARD), Matérn closed forms (nu ∈
  {0.5, 1.5, 2.5}), Periodic, Linear, Polynomial, RationalQuadratic, WhiteNoise,
  SpectralMixture, NeuralNetwork (RW eq. 4.29–4.31), ArcCosine — all callable and
  composable via `+` / `*` / `**` operator overloading.
- `kernel_ops.py` — `KernelSum`/`KernelProduct`/`KernelPower`/`KernelComposition` with
  flattened `part<i>__<name>` parameter trees for optimizers; `kernel_matrix`,
  finite-difference `kernel_grad`.
- `models.py` — `ExactInference` (jittered Cholesky + analytic LML), `GaussianProcess`,
  `GPRegression`, `GPTimeSeriesModel`, and the spec-facing `GPClassification` facade
  over the classification engines.
- `inference.py` — binary classification engines: `LaplacePropagation` (RW Alg. 3.1,
  logit/probit), `ExpectationPropagation` (**experimental**, may not converge — prefer
  Laplace or VI), `VariationalInference` (Jaakkola-Jordan bound, logit only).
- `sparse.py` — FITC (`InducingPointGP`) and VFE/Titsias SGPR (`SparseVFE`,
  `SparseGaussianProcess`) in a **whitened parameterization**: all solves through
  jittered Cholesky factors, posterior precision eigenvalues ≥ 1 by construction.
- `deep_gp.py` — documented two-layer composition (sparse latent → observed).
- `hyperparams.py` — `MarginalLikelihood`, L-BFGS log-space
  `optimize_hyperparams()` walking the kernel parameter tree, `ARD`, `cross_validate_gp`.

Conventions: kernels are composable operators; models expose fluent `.fit()` returning
self; predictions return means plus pointwise std (or full covariance with
`full_cov=True`); forecasts wrap in the shared `ForecastResult`; everything runs on
native numpy/scipy.

Known limitations: EP is experimental ([20]); GPR means revert to the prior under
long-horizon extrapolation — trust the growing predictive std, not the point forecast;
sparse approximations converge to exact as inducing points cover the domain (M = T
reproduces exact to machine precision).

Spec: vault `Modules/gaussian_processes.md` (private). Tests: `tests/gaussian_processes/tests.py`.
