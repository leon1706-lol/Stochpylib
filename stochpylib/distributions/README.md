# stochpylib.distributions — 80+ probability distributions

**Status: work in progress — code exists but the module is NOT done.**

What exists:

- `_base.py` — `Distribution` / `MultivariateDistribution` base classes defining the common
  interface (`.pdf()/.pmf()`, `.cdf()`, `.ppf()`, `.rvs()`, `.mean()`, `.var()`, `.fit()`, …).
- `discrete.py` (11 classes), `continuous.py` (25), `multivariate.py` (6), `heavy_tail.py` (5).

What is missing before this counts as implemented:

1. An `__init__.py` that re-exports all classes (currently nothing is importable from
   `stochpylib.distributions` directly — only submodules like
   `stochpylib.distributions.continuous`).
2. Tests in `tests/distributions/tests.py` (none exist yet). Per ARCHITECTURE.md, verify every
   class satisfies the common interface contract.
3. Update `development/Implementation-Checklist.md`, `CHANGELOG.md`, and module status.

Spec: vault `Modules/distributions.md` (private).
