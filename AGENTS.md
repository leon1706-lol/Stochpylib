# AGENTS.md — stochpylib Development Guide

Single entry point for any AI agent or contributor. Read before writing code.
`Stochpylib-Obsidian-Vault/Essential-Tasks.md` is the wrap-up checklist this
file defers to — part of the contract, not optional.

## 1. Overview

Complete stochastic-computing library — probability, distributions, Monte
Carlo, time series, GPs, copulas, survival, queueing, information theory,
Lévy processes, financial stochastics, statistics, random matrix theory,
advanced MCMC, numerical methods, optimization, design of experiments — native
on NumPy/SciPy, no wrapper deps. Thesis: one coherent package replaces scipy.stats + statsmodels
+ lifelines + copulas.

- **State:** 20 modules implemented (689 / 794 spec names), 3 remaining as spec.
- **Runtime deps:** NumPy, SciPy (`special`/`optimize`/`integrate` only).
  **Test deps:** pytest. Nothing else, ever.
- **Docs move with code:** a change not reflected in the relevant docs in the
  same task isn't done (§5 step 5).

## 2. Read First

- **`todo.md`** — owner's live planning canvas, current objective(s). Not a
  backlog; gets rewritten/cleared, so an empty file means *ask* (or fall back
  to `development/CHANGELOG.md`'s latest phase).
- **Sub-readmes** — every folder has one; read it before touching that folder.
- **`development/*.md`** — read all: architecture, infrastructure,
  project_structure, CHANGELOG, Probleme, Implementation-Checklist (the
  single source of truth for progress), Development.
- **`Stochpylib-Obsidian-Vault/`** (private, gitignored) — design spec:
  `Modules/<name>.md` (target API per module), `Module-Map.md`,
  `Dependencies.md`, `Quickstart-Examples.md`, `Ratings.md`,
  `ARCHITECTURE.md`, `Essential-Tasks.md`, `HANDOFF.MD` (append-only log).
  The generated code graph in `code/` can be stale — cross-check against
  source.
- `CONTRIBUTING.md` — dev setup, ground rules, semver policy.

## 3. Code Conventions

- **Distribution contract (load-bearing):** every distribution exposes
  `.pdf()/.cdf()/.ppf()/.rvs()/.mean()/.var()/.skewness()/.kurtosis()/.entropy()/.mgf()/.cf()/.fit()/.ks_test()`.
  One sanctioned deviation: multivariate classes use `.pdf()` not `.pmf()`
  and omit `.mgf()/.cf()`.
- `.fit(data)` returns `self`. Every stochastic method takes `random_state=`.
- Library code never wraps `scipy.stats`; it's the **test oracle** only.
  pandas/statsmodels/lifelines/pymc are allowed only in `tests/`.
- Estimators return shared result objects (`MCResult`, `ForecastResult`,
  `QueueResult`): point estimate + standard error + confidence interval.
- PascalCase classes, snake_case functions. Module `__all__` lists every
  public name exactly once. Version in `stochpylib/__init__.py` and
  `pyproject.toml` must match.
- **Comments: short and precise.** Keep only what the code can't say — a
  non-obvious *why*, an invariant, a real gotcha. Restated code or process
  narration ("found live", "see Probleme.md #N") → delete; that history lives
  in git log / CHANGELOG / Probleme.md. Genuinely important invariants may
  stay in full — judgment, not a cap.

## 4. Layout

```
stochpylib/                 # installable package
  __init__.py               # re-exports all subpackages, __version__
  cli.py / cli_pypi.py / cli_demo.py / selftest.py   # spl command
  <module>/{__init__.py, README.md, *.py}
tests/<module>/{tests.py, e2e.py}   # outside the package, one folder per module:
                                     # oracle suite + end-to-end sweep of every public name
development/                # dev docs
Stochpylib-Obsidian-Vault/  # design spec (private)
```

Every folder has a short `README.md` — hard requirement, the main README
links each one.

## 5. Workflow

**Nothing is done until verified:** new behavior needs its own tests *and* a
manual real repro — green unit tests alone have missed real bugs here.

1. **Before coding:** read the vault's `Modules/<name>.md`,
   `Quickstart-Examples.md`, `ARCHITECTURE.md`.
2. **Manual debug:** realistic end-to-end scratch script against the real
   implementation, not mocks.
3. **Tests:** `tests/<module>/tests.py` in the same task, scipy.stats etc. as
   independent oracles, statistical assertions ≥ 3 SE — **and**
   `tests/<module>/e2e.py`: one realistic exercise per public name (the
   `EXERCISES == __all__` guard fails otherwise). CI (`ci.yml`) runs
   `pytest tests/` over the whole tree plus one `smoke (<module>)` job per
   module; a new module must be added to the `module-smoke` matrix
   (`tests/docs` asserts it equals `stochpylib.__all__`).
4. **`pytest tests/ -v`** must be green; fix failures now, not later.
5. **Update docs** (style in §6):
   `stochpylib/<module>/README.md` · `tests/README.md` (test count) ·
   `development/CHANGELOG.md` · `development/Probleme.md` ·
   `development/Implementation-Checklist.md` (check off names, progress line) ·
   `development/architecture.md` (module map, flow diagram) ·
   `development/infrastructure.md` (selftest count) · `README.md` (badges,
   status table, Known Limitations, Mermaid diagrams, Roadmap, CLI counts) ·
   `stochpylib/README.md` (module table) · `stochpylib/selftest.py` (new
   module checks) · `stochpylib/cli.py` (any hardcoded counts; `--help` is
   generated from `__all__`) · `pyproject.toml` + `__init__.py` (version) ·
   `todo.md` (rewrite/clear when the objective changes).
   `tests/docs/tests.py` recomputes every number the docs claim — it fails
   on any drift.
6. **Git — ask, don't act:** stage only intended files, never secrets; propose
   the exact commit message (`vX.Y.Z: <short description>`) and wait. Same
   for anything with side effects outside the checkout (publish, tags).
7. **Vault handoff:** from `Stochpylib-Obsidian-Vault/`, run
   `scripts/generate_code_graph.py` and `scripts/regenerate_vault.py`
   (`--repo-root .. --vault . --append-handoff --agent <name> --summary ...`),
   then append to `HANDOFF.MD`. Private and gitignored — do it in a full
   wrap-up, but say so if you defer it.

## 6. Writing Docs — condensed, not narrated

Condense at write time; don't write long and trim later. A MUST, not a
preference.

- **`Probleme.md`:** every bug found. Continuous entry numbers, severity 1
  (cosmetic) – 10 (wrong numbers silently shipped), status 🟢 fixed / 🟡
  partial / 🔴 closed, then **Problem** / **Fix** / **Verification** at
  ~1-3 sentences each — finding, change, proof. No hypotheses tried, no
  investigation narrative, no restated code.
- **`CHANGELOG.md`:** `## Phase N — Title` + bold-led bullets, one per task
  or fix, ~2 sentences each (a bit more only if genuinely needed) — what
  changed and why it matters, not file lists or verification play-by-play.

## 7. CI, Release, CLI

- `ci.yml`: full pytest on push/PR, Python 3.10–3.13, ubuntu + windows
  (`fail-fast: false`); `smoke (<module>)` per module (oracle suite + e2e sweep
  + `spl demo`); `cross-suite` (library/docs/cli); `install-smoke` (wheel ships
  every spec name).
- `publish.yml` on `v*` tags: build, smoke-test (`spl --version`,
  `spl --test`), publish via Trusted Publisher. `release.yml` creates the
  GitHub Release. Bump both version files before tagging.
- `spl`: `--help` (inventory from `__all__`), `--version [--list]`,
  `--test` (embedded self-check), `update`, `info`, `show <Name>`,
  `demo [module]`, `cite`.

*Update when conventions change.*
