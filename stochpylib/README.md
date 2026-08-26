# stochpylib/

The installable package: one subpackage per library module behind a single
load-bearing contract — every distribution exposes the same method set, every
stochastic method takes `random_state=`, every Monte Carlo estimator returns a
shared result object. Nine subpackages live today (317/794 spec names):

| Subpackage | Spec names | What it owns | Guide |
|---|---|---|---|
| `probability/` | 21 | sample spaces, Bayes, exact combinatorics, independence | [README](probability/README.md) |
| `distributions/` | 60 | 47 distributions behind the common interface | [README](distributions/README.md) |
| `montecarlo/` | 25 | quasi-random sequences, estimators, variance reduction, applications | [README](montecarlo/README.md) |
| `timeseries/` | 61 | linear/volatility models, filters, changepoints, spectral analysis | [README](timeseries/README.md) |
| `gaussian_processes/` | 36 | composable kernel zoo, exact/sparse GP regression & classification | [README](gaussian_processes/README.md) |
| `copulas/` | 26 | elliptical/Archimedean/empirical copulas, vines, dependence measures | [README](copulas/README.md) |
| `survival/` | 28 | Kaplan-Meier, parametric fits, Cox/AFT/FineGray, competing risks | [README](survival/README.md) |
| `queueing/` | 29 | M/M/1 to Jackson networks, birth-death formulas, discrete-event simulation | [README](queueing/README.md) |
| `information_theory/` | 31 | entropy, divergences, mutual information, channels, coding | [README](information_theory/README.md) |

Package-level files:

- `__init__.py` — re-exports every subpackage; `__version__` lives here.
- `cli.py` — the `spl` console command: `--help` library inventory (generated
  from each module's `__all__`), `--version [--list]` with PyPI awareness,
  `--test` embedded self-check, and the `update` / `info` / `show` / `demo` /
  `cite` subcommands.
- `cli_pypi.py` — offline-safe PyPI metadata access (fetch, 24 h cache,
  version parsing, install-mode detection) behind `--version` and `update`.
- `cli_demo.py` — the nine live mini-examples behind `spl demo <module>`.
- `selftest.py` — the 139-check self-check suite shipped inside the wheel,
  runnable from any pip install without pytest or a source checkout.

Target API for every planned module lives in the private Obsidian vault
(`../Stochpylib-Obsidian-Vault/Modules/<name>.md`); exact progress against the
full design spec is tracked in
[`../development/Implementation-Checklist.md`](../development/Implementation-Checklist.md).

Run all tests from the repo root: `pytest tests/ -v`.
