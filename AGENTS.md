# AGENTS.md — stochpylib Development Guide

Single entry point for any AI agent or contributor. Read before writing code.
`Stochpylib-Obsidian-Vault/Essential-Tasks.md` is the wrap-up checklist this
file defers to — part of the contract, not optional.

## 1. Overview

Complete stochastic-computing library — probability, distributions, Monte
Carlo, time series, GPs, copulas, survival, queueing, information theory,
Lévy processes, financial stochastics, statistics, random matrix theory,
advanced MCMC, numerical methods, optimization, design of experiments, spatial
statistics, visualization, utilities — native on NumPy/SciPy, no wrapper deps.
Thesis: one coherent package replaces scipy.stats + statsmodels + lifelines + copulas.

- **State:** all 23 planned modules implemented (794 / 794 spec names).
- **Runtime deps:** NumPy, SciPy (`special`/`optimize`/`integrate` only).
  **Test deps:** pytest. Nothing else, ever — except `matplotlib`, an optional,
  lazily-imported `viz` backend (`stochpylib/viz/_mpl.py` only, never at module import
  time), and pandas/torch/jax/numba/cupy, optional lazily-imported `utils` interop
  (`stochpylib/utils/_backends.py` only, never at module import time); `viz`'s default
  SVG rendering and every other module never require any of them installed.
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
  - **Cap:** one or two short sentences per comment; a longer one means the
    code needs a clearer name or a docstring, not a bigger comment.
  - **Never write:** section banners (`# ---- foo ----`, `# ==== foo`), labels
    over obvious steps (`# build the tree`), shape/type tags on assignments
    (`# (n, m)`), restated formulas the code already spells out, or
    regression/incident stories in tests (the test name and git log carry them).
  - **Keep:** functional pragmas (`# noqa`, `# pragma: no cover`, `# type:`),
    numerical-stability reasons, algorithm/paper identification, and the
    cause of any tolerance, seed or oracle quirk a reader would otherwise "fix".
  - Comments go above the line they explain, in lowercase-or-sentence prose
    without trailing periods on fragments; trailing comments only for a short
    unit/convention note.
  - Docstrings are API docs, not comments; this convention doesn't shrink them.

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
4. **`pytest tests/ -v`** must be green; fix failures now, not later. This sandbox's
   ~4GB RAM can't run it monolithically (verify in 1-3-module `run_in_background`
   batches instead — see `todo.md`'s environment note for the exact pattern). A
   backgrounded run already delivers its own completion notification automatically;
   **don't poll `ReadNotifications`/process-CPU in a tight loop while waiting** — that
   burns tokens for no new information. Either just continue the turn (the notification
   arrives on its own) or arm one `Monitor` on the run's own log/exit rather than
   repeated manual checks.
   **Waiting on a running test: arm a timer, don't poll.** Size one wait to the
   run's expected duration (`ScheduleWakeup`/`sleep` inside a single
   `Monitor` until-loop, or just the `run_in_background` completion notice) and
   do other useful work or end the turn meanwhile. Never re-check the log, CPU
   or process list every minute — each check costs tokens and adds no
   information. If the wait expires with the run still going, re-arm one
   longer timer (≥ the previous one), not a tighter one.
5. **Update docs** (style in §6):
   `stochpylib/<module>/README.md` · `tests/README.md` (test count) ·
   `development/CHANGELOG.md` · `development/Probleme.md` ·
   `development/Implementation-Checklist.md` (check off names, progress line) ·
   `development/architecture.md` (module map, flow diagram) ·
   `development/infrastructure.md` (selftest count) · `README.md` (status table,
   Known Limitations, Mermaid diagrams, Roadmap, CLI counts — the `tests`/`public names`
   badges are live and self-update via CI, see §7, so skip those two) ·
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
  + `spl demo`); `viz-matplotlib` (ubuntu + windows) installs matplotlib and runs
  `tests/viz/backend_mpl.py` (the only place that file runs) plus `viz`'s suites again;
  `utils-optional` (ubuntu + windows) installs CPU torch/pandas/numba/jax and runs
  `tests/utils/backend_optional.py` plus `utils`'s suites again (cupy/CUDA can't run
  on a hosted runner, so it stays untested there); `cross-suite` (library/docs/cli);
  `update-stats-badge` (push to `main` only, after every other job is green) recomputes
  `development/stats.json` (`development/scripts/update_stats.py`) and commits it with
  `[skip ci]` if it changed — the source README's `tests`/`public names` badges read live
  via a shields.io `dynamic/json` badge over raw.githubusercontent.com, so those two numbers
  never need a manual README edit; `install-smoke` (wheel ships every spec name).
- `publish.yml` on `v*` tags: build, smoke-test (`spl --version`,
  `spl --test`), publish via Trusted Publisher. `release.yml` creates the
  GitHub Release. Bump both version files before tagging.
- `spl`: `--help` (inventory from `__all__`), `--version [--list]`,
  `--test` (embedded self-check), `update`, `info`, `show <Name>`,
  `demo [module]`, `cite`.

*Update when conventions change.*
