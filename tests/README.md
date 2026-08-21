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
see development/Probleme.md [3]).

Current coverage: `probability` (25 passing). `distributions` has code but no tests yet.
