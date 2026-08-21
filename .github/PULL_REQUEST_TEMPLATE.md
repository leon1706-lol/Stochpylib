# What does this PR change?

<!-- One or two sentences: what and why. Reference the issue number if one exists. -->

## Type of change

- [ ] Bug fix (non-breaking, fixes an issue)
- [ ] New feature / new public names (backward compatible)
- [ ] Breaking change (public API — must be flagged per CONTRIBUTING.md versioning policy)
- [ ] Docs / packaging / CI only

## Checklist

- [ ] Tests added or extended in `tests/<module>/tests.py`; `pytest tests/ -v` is green
- [ ] For distributions: every class satisfies the common interface
      (`.pdf/.cdf/.ppf/.rvs/.mean/.var/.skewness/.kurtosis/.entropy/.mgf/.cf/.fit/.ks_test`)
- [ ] No `scipy.stats` wrapping in library code (`tests/` may use it as reference oracle)
- [ ] Determinism: tests use fixed seeds; no reliance on wall-clock time
- [ ] Docs updated where relevant: README / sub-readmes (`stochpylib/*/README.md`,
      `tests/README.md`) / `development/CHANGELOG.md`
- [ ] Bugs found & fixed along the way documented in `development/Probleme.md`
      (severity 1–10 + status)
- [ ] Version bumped in `pyproject.toml` + `stochpylib/__init__.py` if the public surface
      changed (semver — see CONTRIBUTING.md)
- [ ] `spl --version` / `spl --test` still work after a fresh `pip install -e .`

## Notes for maintainers

<!-- Maintainers will sync the private spec vault and release notes; nothing needed here
     unless you were given vault access. -->
