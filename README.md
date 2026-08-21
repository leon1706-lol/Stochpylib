# stochpylib

![Python](https://img.shields.io/badge/python-3.10%2B-blue)
![License](https://img.shields.io/badge/license-MIT-green)
![Tests](https://img.shields.io/badge/tests-182%20passing-brightgreen)

A Python library for probability, distributions, time series, Gaussian processes, stochastic
processes, financial stochastics, and a long tail of statistical/numerical tooling — aiming to
cover in one package what's usually stitched together from `scipy.stats`, `statsmodels`, `pymc`,
`arch`, `lifelines`, and `copulas`.

The full target design — every planned module, submodule, and public function — lives in the
[design spec vault](Stochpylib-Obsidian-Vault/README.md). This README covers what's actually
built and how to work on it.

## Status

Early development — three modules implemented so far:

- **`stochpylib.probability`** — sample spaces, events, Bayes' theorem, combinatorics,
  independence checks.
- **`stochpylib.distributions`** — 47 distributions (discrete, continuous, multivariate,
  heavy-tailed) behind one common interface: `.pdf()/.cdf()/.ppf()/.rvs()/.fit()`, moments,
  `.entropy()/.mgf()/.cf()/.ks_test()`.
- **`stochpylib.montecarlo`** — quasi-random sequences (Sobol/Halton/Faure/Niederreiter),
  crude/QMC/importance/rejection/stratified estimators, variance-reduction techniques
  (antithetic, control variates, LHS, conditioned MC, rejection control), and applications
  (integration, option pricing, VaR/ES, reliability, sensitivity).

A terminal entry point ships with the package: `spl --help` gives an overview of everything
the library offers (with quick-start code), `spl --version` prints the installed version, and
`spl --test` runs a built-in self-check (106 checks) against any installation — no pytest or
source checkout needed.

See [`development/Implementation-Checklist.md`](development/Implementation-Checklist.md) for
exact progress against the full spec (currently 106 / 794 public names).

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
182 passed / 2 skipped (intentional convention skips) as of the `montecarlo` module.

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
contributions right now are implementing one module at a time against its target API spec.
Start with [`CONTRIBUTING.md`](CONTRIBUTING.md) — it covers dev setup, the ground rules (no
`scipy.stats` wrapping, common distribution interface), the
[versioning & deprecation policy](CONTRIBUTING.md#versioning--deprecation-policy), and the PR
checklist. Bug reports and feature ideas go through the issue templates; security issues
privately per [`SECURITY.md`](SECURITY.md). This project follows
[`CODE_OF_CONDUCT.md`](CODE_OF_CONDUCT.md).

Each tagged release `vX.Y.Z` is published to PyPI by CI and gets a matching GitHub Release
with per-tag changelog notes.

Before opening a PR:

1. Read the target module's spec — open an issue first if you need the API contract.
2. Add tests in `tests/<module>/tests.py` and make sure `pytest tests/ -v` is green.
3. Update `development/Implementation-Checklist.md` to check off what you implemented, and add an
   entry to `development/CHANGELOG.md` (and `development/Probleme.md` if you found a bug).
4. Follow the full wrap-up checklist in
   [`Essential-Tasks.md`](Stochpylib-Obsidian-Vault/Essential-Tasks.md) — it covers spec
   alignment, vault regeneration, and the handoff log.

## License

[MIT](LICENSE) © Francis Engert
