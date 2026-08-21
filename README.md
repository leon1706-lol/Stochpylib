# stochpylib

![Python](https://img.shields.io/badge/python-3.10%2B-blue)
![License](https://img.shields.io/badge/license-MIT-green)
![Tests](https://img.shields.io/badge/tests-25%20passing-brightgreen)

A Python library for probability, distributions, time series, Gaussian processes, stochastic
processes, financial stochastics, and a long tail of statistical/numerical tooling — aiming to
cover in one package what's usually stitched together from `scipy.stats`, `statsmodels`, `pymc`,
`arch`, `lifelines`, and `copulas`.

The full target design — every planned module, submodule, and public function — lives in the
[design spec vault](Stochpylib-Obsidian-Vault/README.md). This README covers what's actually
built and how to work on it.

## Status

Early development — one module implemented so far:

- **`stochpylib.probability`** — sample spaces, events, Bayes' theorem, combinatorics,
  independence checks.

See [`development/Implementation-Checklist.md`](development/Implementation-Checklist.md) for
exact progress against the full spec (currently 21 / 794 public names).

## Install

```bash
pip install stochpylib
```

## Quick example

```python
from stochpylib.probability import bayes_theorem, total_probability

# Classic disease-screening example: 1% prevalence, 99% sensitivity, 5% false-positive rate.
p_positive = total_probability((0.99, 0.01), (0.05, 0.99))
p_disease_given_positive = bayes_theorem(0.01, 0.99, p_positive)
print(round(p_disease_given_positive, 4))  # 0.1667
```

## Development

```bash
pip install -e ".[dev]"
pytest tests/ -v
```

Tests live in `tests/<module>/tests.py` — one file per module, outside the installed package.
25/25 passing as of the `probability` module.

### Tools used

- **[setuptools](https://setuptools.pypa.io/)** — build backend (`pyproject.toml`).
- **[pytest](https://pytest.org/)** — test runner, including doctests.
- **[build](https://pypa-build.readthedocs.io/)** — sdist/wheel builds.
- **GitHub Actions** — CI (`.github/workflows/ci.yml`, test matrix across Python 3.10–3.13) and
  PyPI publishing (`.github/workflows/publish.yml`, triggered on `vX.Y.Z` tags via PyPI's Trusted
  Publisher/OIDC flow — no stored API tokens).
- **NumPy / SciPy** — declared runtime dependencies for the library as a whole, per the spec's
  baseline (not all implemented modules need them yet — `probability` currently doesn't).

## Project layout

- [`stochpylib/`](stochpylib/) — the actual package, one subpackage per module
  (e.g. `stochpylib/probability/`).
- [`tests/`](tests/) — tests, mirroring the package structure one folder per module.
- [`Stochpylib-Obsidian-Vault/`](Stochpylib-Obsidian-Vault/README.md) — the design spec: target
  module map, API conventions, ratings, quickstart examples for the *full* planned library.
- [`development/`](development/) — build history and process docs, separate from the design spec:
  - [`Development.md`](development/Development.md) — package layout decisions and workflow notes.
  - [`CHANGELOG.md`](development/CHANGELOG.md) — append-only log of build phases.
  - [`Probleme.md`](development/Probleme.md) — bugs found during implementation, with a
    severity score (1–10) and Fixed/Open status for each.
  - [`Implementation-Checklist.md`](development/Implementation-Checklist.md) — every
    module/submodule/public name in the spec, checked off as it's actually implemented and
    tested.

## Contributing

This is early-stage and the spec is large (23 modules, ~794 public names), so the most useful
contributions right now are implementing one module at a time against its
[`Modules/<name>.md`](Stochpylib-Obsidian-Vault/Modules) spec file. Before opening a PR:

1. Read the target module's spec in `Stochpylib-Obsidian-Vault/Modules/<name>.md` and the
   cross-cutting conventions in
   [`ARCHITECTURE.md`](Stochpylib-Obsidian-Vault/ARCHITECTURE.md) (e.g. the common
   `.pdf()/.cdf()/.rvs()/.fit()` interface every distribution must expose).
2. Add tests in `tests/<module>/tests.py` and make sure `pytest tests/ -v` is green.
3. Update `development/Implementation-Checklist.md` to check off what you implemented, and add an
   entry to `development/CHANGELOG.md` (and `development/Probleme.md` if you found a bug).
4. Follow the full wrap-up checklist in
   [`Essential-Tasks.md`](Stochpylib-Obsidian-Vault/Essential-Tasks.md) — it covers spec
   alignment, vault regeneration, and the handoff log.

## License

[MIT](LICENSE) © Francis Engert
