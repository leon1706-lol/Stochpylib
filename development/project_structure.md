# project_structure

The annotated directory tree of the stochpylib repository — what every folder
and top-level file is for. Every folder carries its own short `README.md` as
an entry-point guide.

```
Stochpylib/
├── README.md                    # project front page: banner, quickstart, status tables, roadmap
├── CONTRIBUTING.md              # dev setup, ground rules, semver & deprecation policy, PR checklist
├── CODE_OF_CONDUCT.md           # Contributor Covenant 2.1
├── SECURITY.md                  # private vulnerability reporting (72 h acknowledgment)
├── LICENSE                      # MIT
├── pyproject.toml               # setuptools packaging: deps, dev extras, spl console script, pytest config
├── .gitattributes / .gitignore
├── .github/
│   ├── workflows/               # ci.yml (test matrix), publish.yml (PyPI on tags), release.yml (GitHub Release)
│   ├── ISSUE_TEMPLATE/          # bug report + feature request YAML forms (spl --version pre-flight)
│   └── PULL_REQUEST_TEMPLATE.md # checklist mirroring the wrap-up rules
├── stochpylib/                  # the installable package (guide: stochpylib/README.md)
│   ├── __init__.py              # re-exports all nine subpackages; __version__
│   ├── cli.py                   # the spl console command: --help inventory, --version [--list],
│   │                            #   --test, and the update/info/show/demo/cite subcommands
│   ├── cli_pypi.py              # PyPI metadata access behind spl --version/update (cache, offline-safe)
│   ├── cli_demo.py              # the nine live mini-examples behind spl demo
│   ├── selftest.py              # embedded 139-check self-check suite shipped in the wheel
│   ├── probability/             # sample spaces, Bayes, exact combinatorics, independence
│   ├── distributions/           # 47 classes behind the common interface (_base.py fallbacks)
│   ├── montecarlo/              # QMC sequences, estimators, variance reduction, applications
│   ├── timeseries/              # ARIMA/GARCH families, filters, changepoints, spectral, forecasting
│   ├── gaussian_processes/      # kernel zoo, exact/sparse/classification inference, hyperparams
│   ├── copulas/                 # elliptical/Archimedean/empirical, vines, methods dispatcher
│   ├── survival/                # KM/NA, parametric fits, Cox/AFT/additive/FineGray, competing risks
│   ├── queueing/                # single queues, birth-death, networks, discrete-event simulation
│   └── information_theory/      # entropy, divergences, mutual info, channels, coding
├── tests/                       # test suite OUTSIDE the package on purpose (guide: tests/README.md)
│   ├── <module>/tests.py        # one deterministic suite per implemented module (nine)
│   ├── library/                 # cross-module suite: spec conformance, pinned extras, workflows
│   │   ├── _extract_spec_names.py   # generates conformance lists from the checklist
│   │   └── _spec_names.json         # cached spec-name lists
│   ├── docs/tests.py            # doc-consistency suite: every doc claim recomputed from reality
│   └── cli/tests.py             # spl CLI suite (PyPI/update/info/show/demo/cite; mocked, offline)
├── development/                 # build history & process docs (guide: development/README.md)
│   ├── architecture.md          # system architecture: contracts, conventions, module map
│   ├── infrastructure.md        # packaging/CLI/CI/release runbook
│   ├── project_structure.md     # this file
│   ├── Development.md           # layout decisions & workflow notes
│   ├── CHANGELOG.md             # append-only per-phase build history
│   ├── Probleme.md              # bug audit log (Problem → Fix → Verification, severity, status)
│   ├── Implementation-Checklist.md  # every planned public name as a checkbox (317/794 done)
│   └── logo.png                 # project logo, referenced by the README banner
├── build/  dist/  *.egg-info/   # local build artifacts (gitignored, never edited by hand)
└── Stochpylib-Obsidian-Vault/   # the full design-spec vault (private, not part of the repo)
    ├── Modules/<name>.md        # per-module target API (23 modules)
    ├── ARCHITECTURE.md  Module-Map.md  Ratings.md  Quickstart-Examples.md  Dependencies.md
    ├── Essential-Tasks.md       # the standard wrap-up checklist for any task
    ├── HANDOFF.MD               # append-only agent/session handoff log
    ├── code/                    # auto-generated per-file/per-function code-graph notes
    └── scripts/                 # generate_code_graph.py, regenerate_vault.py
```

Two deliberate structural rules (history in [`Development.md`](Development.md)
and [`Probleme.md`](Probleme.md) [3]):

- **Tests live outside the package** — `tests/<module>/tests.py`, never
  colocated, so no test file or pytest import ever ships in the built wheel.
- **The package stays nested** under `stochpylib/` rather than flattened to the
  repo root — a normal nested package directory avoids non-standard
  `package-dir` mapping in `pyproject.toml` (tried and deliberately reverted).
