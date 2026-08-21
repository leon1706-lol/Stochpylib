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

## Phase 8 - Third module: `stochpylib.montecarlo`

Implemented the full simulation & variance-reduction module (25/25 spec names) natively on
numpy/scipy.special only. `quasi_random`: Halton, Faure (per-coordinate Pascal^j powers),
Sobol and Niederreiter base-2 digital nets driven by programmatically-verified primitive /
irreducible GF(2) generator polynomials with canonical odd initial values, plus the general
`DigitalNetBase2` engine (spec alias `DigitalNet`) and a `LowDiscrepancy` facade; seeded
digital-shift scrambling. Two construction bugs were caught by exactness checks before
shipping: dimension 1 must be plain van der Corput (the x+1 polynomial's generic recurrence
corrupts it), and a Gray-code single-flip walk enumerates points in gray order - replaced
with direct bit decomposition in natural order (see Probleme.md [10]). Known limitation:
exact (t,m,s)-net balance in dimensions >= 2 is within +-1 rather than certified; upgrading
to published Joe-Kuo direction-number tables is an open item ([11]). Statistical quality:
KS p ~ 1 per dimension at n=4096 and discrepancy ~50x better than pseudo-random.
`simulation`: crude/QMC/stratified/importance (self-normalized with ESS)/rejection estimators
returning a shared `MCResult`; `variance_reduction`: antithetic (incl. European call/put),
control variates (optional non-uniform sampler hook), LHS, orthogonal sampling, stratified
grid, conditioned MC, Hesterberg rejection control; `applications`: integration class,
pi estimation, GBM option pricing (validated against an internal Black-Scholes oracle),
historical VaR/ES (`RiskResult`), reliability via library distributions, correlation-based
sensitivity. Manual debug session priced a European call three ways (SE reduction 1.15x
antithetic, 1.84x control-variate vs crude; all within 3 SE of closed form) and ran VaR99/ES
on a simulated book. Tests: tests/montecarlo/tests.py (57 cases); embedded selftest extended
to 106 checks. Full suite: 182 passed / 2 skipped. Progress: 106/794 public names.

## Phase 6 - Open-source hygiene

Added the community/policy layer: CONTRIBUTING.md (dev setup, ground rules, semver +
deprecation policy: deprecations warn via DeprecationWarning, documented in the changelog,
kept >= 2 minor releases or 6 months, removed only in major releases post-1.0),
CODE_OF_CONDUCT.md (Contributor Covenant 2.1), SECURITY.md (private reporting channels,
72h acknowledgment, scope notes for a local numerical library), GitHub issue templates
(bug report + feature request YAML forms with spl --version pre-flight) and a PR template
checklist mirroring the wrap-up rules. New .github/workflows/release.yml creates a GitHub
Release with auto-generated per-tag changelog notes on every vX.Y.Z tag. Fixed a latent bug
found during this pass: publish.yml still ran pytest against the old in-package test location
(pytest stochpylib/) instead of tests/ - tag builds would have failed CI; it now runs the real
suite and additionally smoke-verifies the built wheel via spl --version / spl --test before
publishing.

## Phase 7 - spl --help library overview

spl --help (and bare spl) now prints a full inventory of the installed library instead of
bare flag help: implemented modules with their public functions, all 47 distribution classes
(dynamic from the package's __all__, so it never goes stale), the common distribution
interface, a quick-start snippet for both modules, and pointers to the roadmap/docs. Covered
by test_cli_help_shows_library_overview; output kept ASCII-safe for legacy Windows consoles.
