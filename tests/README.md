# tests/

The test suite: one deterministic file per package module plus a cross-module
library suite, living outside the installed package on purpose (tests used to
ship inside the wheel — `development/Probleme.md` [3]).

## Conventions

- Two files per module, mirroring the layout: `tests/<module>/tests.py` (the
  oracle suite) and `tests/<module>/e2e.py` (the API sweep). pytest picks them
  up via `python_files = ["tests.py", "e2e.py"]` in `pyproject.toml`; each folder — and
  `tests/` itself — carries an `__init__.py`, so every suite imports as
  `tests.<module>.tests`, rooted at the repo (not just at directory name,
  which would let a module named after a stdlib package, e.g. `statistics`,
  shadow it — `development/Probleme.md` #63).
- Deterministic everywhere: fixed seeds on every stochastic path.
- Independent oracles: `scipy.stats`, `statsmodels` and `lifelines` are used
  *only* in tests — never in library code. Statistical assertions are set at
  >= 3 standard errors so results are stable while staying meaningful.
- Doctests run via a dedicated `test_doctests_pass` case in each suite.

## Layout

- `tests/<module>/tests.py` — one oracle suite per implemented module (twenty-three
  today, one per name in `stochpylib.__all__`).
- `tests/<module>/e2e.py` — the end-to-end API sweep of the same module: an
  `EXERCISES` registry with one realistic exercise per name in the module's
  `__all__`, each run as its own `test_exercise[<name>]` case, plus
  `test_every_public_name_is_exercised`, which fails the moment a name ships
  without an exercise. Fast (seconds to ~2 min per module); heavy statistical
  validation stays in `tests.py`. CI runs each module's pair as its own
  `smoke (<module>)` job.
- `tests/library/tests.py` — the cross-module suite:
  - **Spec-name conformance** for all 794 implemented public names (generated from
    `development/Implementation-Checklist.md` via `_extract_spec_names.py`, cached in
    `_spec_names.json`), plus every documented extra beyond the checklist (e.g.
    `MCResult`, `BaseCopula`, GP kernel base/ops, and each module's own result-object
    types — see each module's README for its own extras).
  - **The sanctioned multivariate method-contract deviation** (`.pdf()` not `.pmf()`,
    no `.mgf()`/`.cf()`).
  - **Cross-module workflows**: reliability MC on library Weibull, t-copula margins
    through library Student_t, ARIMA vs. GP forecasting agreement, `CopulaFit`
    round trips, a conjugate Bayesian posterior cross-checked against an MCMC
    sampler, robust regression/correlation/covariance against their
    `statistics`/`copulas` counterparts, nonparametric tests against
    `copulas`/`statistics`/`gaussian_processes`, spatial kriging against
    `gaussian_processes`/`experimental_design`, viz plots recomputed against the
    models underneath them, and `utils.fit`/`goodness_of_fit`/`ecdf`/
    `outlier_detection`/`set_seed`/`to_json`/`ParallelSimulation` exercised across
    several modules.
  - **Import-discipline guards**: no `stochpylib/**/*.py` file imports `scipy.stats`;
    `matplotlib` is imported only inside `stochpylib/viz/_mpl.py` function bodies;
    pandas/torch/jax/jaxlib/numba/cupy only inside `stochpylib/utils/_backends.py`
    function bodies; `np.random.default_rng`/`.seed`/`RandomState(` with a
    non-constant or missing argument appear only in `stochpylib/_rng.py` (plus the
    CLI demo/self-check suites and `utils/random.py`'s own `numpy_global=` path).
- `tests/docs/tests.py` — the documentation-consistency suite: every number the
  docs claim (test counts, versions, spec-name tables, links, checklist
  progress) is recomputed from reality; a drifted doc fails the suite.
- `tests/cli/tests.py` — the `spl` CLI suite: PyPI awareness (`--version`,
  `--list`), `spl update` (validation, dry-run, prompts, editable refusal),
  `spl info`/`show`/`demo`/`cite` — all with **mocked PyPI responses and mocked
  subprocess**, so no test ever touches the network or runs pip.

Run everything from the repo root:

```bash
pytest tests/ -v
```

The package also ships an embedded smoke suite runnable from any pip install:
`spl --test` (291 checks), which includes the per-module conformance and
cross-module spot checks. The live pass count lives only in the root README
badge — deliberately no second copy here to go stale.

`tests/viz/backend_mpl.py` is a third file in that module, testing the optional real-
matplotlib backend; `python_files = ["tests.py", "e2e.py"]` in `pyproject.toml` keeps it
out of the main `pytest tests/` run (and out of the docs suite's skip count), and only the
`viz-matplotlib` CI job — which installs matplotlib first — runs it directly by path.
`tests/utils/backend_optional.py` is the same pattern for `utils`' optional pandas/
torch/jax/numba interop, run only by the `utils-optional` CI job; `tests/utils/_workers.py`
holds the picklable module-level functions the `backend="process"` test cases need (a
closure or lambda can never be pickled).
