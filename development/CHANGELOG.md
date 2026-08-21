# Changelog

Append-only. One entry per meaningful chunk of work, in chronological order.

## Phase 1 — Vault digitization

Converted the React design-spec component (module tree, ratings, quickstart examples) into the
Obsidian vault at `Stochpylib-Obsidian-Vault/`: `Module-Map.md`, one `Modules/<name>.md` per
top-level module (23 modules, ~120 submodules, ~794 public names), `Ratings.md`,
`Quickstart-Examples.md`, `Dependencies.md`, `ARCHITECTURE.md`, `Essential-Tasks.md`. Replaced
prior stale vault content that had documented an unrelated project.

## Phase 2 — Naming and package scaffold

Discovered `stochpy` and several close variants (`pystoch`, `stochpy-toolkit`, ...) were already
taken on PyPI by unrelated packages. Renamed the project to `stochpylib` (PyPI distribution name
== Python import name) and propagated the rename through every vault file. Scaffolded the real
package: `pyproject.toml` (setuptools backend, license left as a placeholder), `.gitignore`,
package `README.md`, `.github/workflows/ci.yml` (test matrix) and `publish.yml` (tag-triggered
PyPI Trusted Publisher / OIDC, no stored token).

## Phase 3 — First module: `stochpylib.probability`

Implemented all 21 public names from `Modules/probability.md` across `basics.py`
(sample spaces, events, Bayes' theorem), `combinatorics.py` (factorial through derangements,
exact integer arithmetic), and `independence.py` (independence/conditional-independence checks).
Wrote tests and doctests; see [Probleme.md](Probleme.md) for the math errors found and fixed
while writing them. Verified the package builds (`python -m build`) and installs/imports
correctly from a clean venv.

## Phase 4 — Test relocation, progress tracking, dev-folder setup

Moved tests out of the package (`stochpylib/probability/tests.py` →
`tests/probability/tests.py`) so test files no longer ship inside the built wheel. Added
`Implementation-Checklist.md` to the vault — a single checkbox list spanning every
module/submodule/public name, checked off as things actually get implemented (currently 21/794).
Updated `Essential-Tasks.md` to require keeping that checklist current, running the full test
suite, regenerating the vault's code-graph notes, and appending a `HANDOFF.MD` entry as part of
wrapping up any task. Created this `development/` folder (this file, `CHANGELOG.md`,
`Probleme.md`) to track build history and bugs separately from the design spec.

## Phase 5 — Second module: `stochpylib.distributions` + `spl` CLI

Finished the distributions module whose class code existed but was undelivered: added
`distributions/__init__.py` exporting all 47 classes (+ 2 base classes) and wired it into
`stochpylib/__init__.py`. Audited every method of every class against scipy.stats references,
fixing four library bugs along the way (see Probleme.md [5]–[8]): GPareto pdf leaked probability
below its support; Rice pdf overflowed to NaN for large x (now computed in log space); discrete
`ppf` overshot bounded supports and returned the wrong atom; `MultivariateDistribution.fit` had
a broken signature. `StableDistribution` gained exact closed-form delegation for alpha=2
(Gaussian) and alpha=1,beta=0 (Cauchy), plus a Chambers–Mallows–Leckie sampler for all
alpha != 1 — validated empirically against the closed-form characteristic function (the
alpha=1, beta!=0 corner keeps a slow-but-correct inverse-CDF fallback; see Probleme [9]).
Added the full test suite `tests/distributions/tests.py` (interface-contract matrix over all
13 spec methods × 47 classes, scipy cross-checks, fit round-trips, stable-sampler validation);
tests folders are now packages so same-named `tests.py` files collect cleanly. New console
script `spl` (`stochpylib.cli`): `spl --version` prints the installed version;
`spl --test` runs the embedded self-check suite (`stochpylib.selftest`, 101 checks) that ships
in the wheel, so any pip install can be verified without pytest or a source checkout.
Full suite: 143 passed / 2 skipped. Progress: 81/794 public names.
