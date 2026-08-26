# Contributing to stochpylib

Thanks for your interest! This project is early-stage (currently 317/794 planned public names
across nine implemented modules),
so the most useful contributions implement or improve **one module at a time**.

## Development setup

```bash
git clone <your fork>
cd Stochpylib
pip install -e ".[dev]"
pytest tests/ -v          # must be green before you start changing things
```

Verify your installation any time with the shipped CLI:

```bash
spl --version   # prints the installed version
spl --test      # embedded self-check suite (139 checks, no pytest needed)
```

Python >= 3.10 required. NumPy/SciPy are the only runtime dependencies.

## Ground rules

1. **No `scipy.stats` wrapping in library code.** Distributions are implemented from scratch;
   `scipy.special/optimize/integrate` may be used as numerical building blocks.
   `scipy.stats` *is* the test oracle — use it freely in `tests/`.
2. **One common interface.** Every distribution exposes `.pdf()/.pmf()`, `.cdf()`, `.ppf()`,
   `.rvs()`, `.mean()`, `.var()`, `.skewness()`, `.kurtosis()`, `.entropy()`, `.mgf()`,
   `.cf()`, `.fit()`, `.ks_test()` — either closed-form or via the generic fallbacks in
   `stochpylib/distributions/_base.py`. Multivariate classes intentionally raise
   `NotImplementedError` where no standard generalization exists.
3. **Tests live outside the package**, one file per module: `tests/<module>/tests.py`.
4. **Spec first.** Every module has a target-API spec. Specs are maintained in a private vault;
   ask in an issue before implementing so we can hand you the relevant API contract and agree
   on constructor/method signatures up front.

## Workflow

1. Open or claim an issue describing what you'll implement.
2. Implement + add tests in `tests/<module>/tests.py` (extend the existing file if present).
3. Run `pytest tests/ -v` — green required.
4. Debug manually beyond unit tests: exercise a realistic end-to-end case against the real
   implementation.
5. Update docs you touched: README / sub-readmes (`stochpylib/*/README.md`, `tests/`,
   `development/`) and `development/CHANGELOG.md` (append-only, one entry per chunk of work).
   Found and fixed a bug? Add an entry to `development/Probleme.md` in the established
   **Problem → Fix → Verification** format with a severity (1–10) and status from the legend
   at the top of that file.
6. Open the PR using the provided template.

Maintainers sync the internal design-spec tracker and release notes; you don't need access to
the private vault to contribute code.

## Versioning & deprecation policy

This project follows [Semantic Versioning](https://semver.org/) — `MAJOR.MINOR.PATCH`:

- **PATCH** — bug fixes, docs, performance. No public-API changes.
- **MINOR** — new modules/classes/functions and backward-compatible additions.
- **MAJOR** — breaking changes to the public API.

While in `0.x` (pre-1.0), the API is still stabilizing: minor versions may contain breaking
changes, but every breaking change is flagged loudly at the top of its changelog entry and
only touches APIs introduced during `0.x`.

**Deprecation policy (from 1.0.0 onward):**

1. Deprecated names emit a `DeprecationWarning` at call time and are marked `.. deprecated::`
   in their docstrings, with the replacement named explicitly.
2. Every deprecation appears under a `Deprecated` heading in `development/CHANGELOG.md`.
3. Deprecated APIs are kept for **at least two minor releases or six months**, whichever is
   longer, and are removed only in a **major** release.
4. Removing a deprecated API without following this cycle is considered a bug — please report it.

Each tagged release `vX.Y.Z` gets a GitHub Release whose notes summarize the changes
(auto-generated from PRs; maintainers align highlights with the changelog).

## Pull requests

Keep PRs scoped to one module or one fix. CI runs the full suite across Python 3.10–3.13;
a red CI blocks merge. Squash-merge is fine; write commit messages that describe the *why*.

## License

By contributing you agree that your contributions are licensed under the MIT license of this
repository.

## Conduct

Please read [`CODE_OF_CONDUCT.md`](CODE_OF_CONDUCT.md) — we enforce it in all project spaces.
