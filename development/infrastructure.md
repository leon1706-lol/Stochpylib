# infrastructure

Build, packaging, CI and release infrastructure for stochpylib. This project is
deliberately infrastructure-light — pure Python/NumPy/SciPy, no compilers, no
GPU, no Docker — so the entire operational surface is: a `pip`-installable
package, one console command (`spl`), and three GitHub Actions workflows.

- **Runtime deps**: NumPy + SciPy, nothing else. `scipy.stats`/`statsmodels`/
  `lifelines`/`pytest` are dev-only extras (`pip install -e ".[dev]"`), never
  runtime dependencies.
- **Packaging**: `pyproject.toml` (setuptools backend). Version lives in
  `pyproject.toml` *and* `stochpylib/__init__.py` — both must be bumped
  together (semver policy in [`CONTRIBUTING.md`](../CONTRIBUTING.md)).

## Local setup

```bash
git clone https://github.com/leon1706-lol/Stochpylib.git
cd Stochpylib
pip install -e ".[dev]"     # runtime deps + pytest
pytest tests/ -v            # full suite must be green before you change anything
spl --version               # verify the editable install
spl --test                  # embedded self-check (136 checks), no pytest needed
```

The full suite is heavy (statistical convergence tests; tens of minutes on a
laptop — `timeseries` and `copulas` dominate). CI runs the same suite, so a
locally slow but green run is normal; a red run is not.

## The `spl` CLI

Registered by every install (PyPI wheel or `pip install -e .`) via
`[project.scripts]` in `pyproject.toml` → `stochpylib/cli.py`:

| Command | What it does |
|---|---|
| `spl` / `spl --help` | Full library inventory: modules, public functions, all distribution classes (generated dynamically from the package, never stale), the common interface, quick-start snippet |
| `spl --version` | Installed version (pip metadata, falling back to the in-code `__version__`) |
| `spl --test` | The embedded self-check suite (`stochpylib/selftest.py`, 136 checks): package sanity, per-module spec conformance, one closed-form spot check per distribution family, MC convergence sanity, cross-module workflows. Exits non-zero on any failure; works after any plain `pip install` |

## GitHub Actions

| Workflow | Trigger | Jobs |
|---|---|---|
| `.github/workflows/ci.yml` | every push / PR | Test matrix: Python 3.10–3.13 × ubuntu-latest + windows-latest; installs `-e ".[dev]" lifelines`, runs `pytest tests/ -v`, then smoke-verifies the install (`spl --version`, `spl --test`) |
| `.github/workflows/publish.yml` | `v*` tag | Job 1: full pytest. Job 2 (`environment: pypi`, `id-token: write`): build sdist + wheel, install the wheel and smoke-verify (`spl --version`, `spl --test`), publish to PyPI via Trusted Publisher (OIDC — no API tokens stored anywhere) |
| `.github/workflows/release.yml` | `v*` tag | Creates the matching GitHub Release with auto-generated changelog notes |

## Release process

1. Bump the version in `pyproject.toml` **and** `stochpylib/__init__.py`
   (semver — deprecation policy in `CONTRIBUTING.md`).
2. Tag and push:
   ```bash
   git tag vX.Y.Z && git push origin vX.Y.Z
   ```
3. CI runs the full matrix, builds + smoke-verifies the wheel, publishes to
   PyPI, and opens the GitHub Release automatically.

One-time prerequisite: configure the Trusted Publisher on pypi.org → your
project → Publishing (pointing at this repo + `publish.yml`). No other manual
step exists — deliberately no release on every push to `main`, only on
explicit version tags.

## Wheel hygiene

Tests live outside the package (`tests/`) precisely so they never ship in the
wheel (`development/Probleme.md` [3]); `spl --test` exists so any end user can
verify an install without pytest or a source checkout. After any packaging
change, build and inspect:

```bash
python -m build
python -c "import zipfile,glob; print('\n'.join(zipfile.ZipFile(glob.glob('dist/*.whl')[0]).namelist()))"
```
