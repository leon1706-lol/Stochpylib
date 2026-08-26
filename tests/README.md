# tests/

The test suite: one deterministic file per package module plus a cross-module
library suite, living outside the installed package on purpose (tests used to
ship inside the wheel — `development/Probleme.md` [3]).

## Conventions

- One file per module, mirroring the layout: `tests/<module>/tests.py` (e.g.
  `probability/tests.py`). Files are named `tests.py` so pytest picks them up
  via `python_files = ["tests.py"]` in `pyproject.toml`; each folder carries an
  `__init__.py` so same-named files import as distinct modules.
- Deterministic everywhere: fixed seeds on every stochastic path.
- Independent oracles: `scipy.stats`, `statsmodels` and `lifelines` are used
  *only* in tests — never in library code. Statistical assertions are set at
  >= 3 standard errors so results are stable while staying meaningful.
- Doctests run via a dedicated `test_doctests_pass` case in each suite.

## Layout

- `tests/<module>/tests.py` — one suite per implemented module (nine today:
  probability, distributions, montecarlo, timeseries, gaussian_processes,
  copulas, survival, queueing, information_theory).
- `tests/library/tests.py` — the cross-module suite: spec-name conformance for
  all 317 implemented public names (generated from
  `development/Implementation-Checklist.md` via `_extract_spec_names.py`, cached
  in `_spec_names.json`), pinned documented extras (`MCResult`,
  `DigitalNetBase2`, timeseries result objects, GP kernel base/ops,
  `BaseCopula`), the sanctioned multivariate method-contract deviation, and
  end-to-end workflows spanning modules (reliability MC on library Weibull,
  t-copula margins through the library Student_t, ARIMA vs GP forecasting
  agreement, `CopulaFit` refit round trips, Sobol-QMC vs crude consistency).
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
`spl --test` (139 checks), which includes the per-module conformance and
cross-module spot checks. The live pass count lives only in the root README
badge — deliberately no second copy here to go stale.
