# tests/ — test suite

One file per package module, mirroring the layout: `tests/<module>/tests.py`
(e.g. `probability/tests.py`). Files are named `tests.py` so pytest picks them up via
`python_files = ["tests.py"]` in `pyproject.toml`.

Run everything from the repo root:

```bash
pytest tests/ -v
```

Doctests are included via a dedicated `test_doctests_pass` case in each suite.
Tests live outside the installed package on purpose (they used to ship inside the wheel —
see development/Probleme.md [3]). Each `tests/<module>/` folder contains an `__init__.py`
so same-named `tests.py` files import as distinct modules.

Current coverage: `probability` + `distributions` + `montecarlo` + `timeseries` +
`gaussian_processes` + `copulas` (329 passed / 2 skipped).
The package also ships an embedded smoke suite runnable from any pip install: `spl --test`
(122 checks).
