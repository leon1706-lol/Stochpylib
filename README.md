<h1 align="center">stochpylib</h1>

<p align="center">
  <strong>Probability · Distributions · Monte Carlo — one coherent Python library.</strong><br>
  A growing, from-scratch toolkit of stochastic computing: native implementations behind one
  common interface, engineered to eventually replace stitching together
  <code>scipy.stats</code>, <code>statsmodels</code>, <code>pymc</code>, <code>arch</code>,
  <code>lifelines</code> and <code>copulas</code>.
</p>

---

<p align="center">
  <img src="https://img.shields.io/badge/python-3.10%2B-FF8C00?style=flat-square&labelColor=1A1A1A&logo=python&logoColor=white" alt="Python 3.10+">
  <img src="https://img.shields.io/badge/%F0%9F%93%84%20license-MIT-8B5CF6?style=flat-square&labelColor=1A1A1A" alt="License: MIT">
  <img src="https://img.shields.io/badge/tests-449%20passing-brightgreen?style=flat-square&labelColor=1A1A1A" alt="329 tests passing">  <a href="https://github.com/leon1706-lol/Stochpylib/actions/workflows/ci.yml"><img src="https://img.shields.io/github/actions/workflow/status/leon1706-lol/Stochpylib/ci.yml?branch=main&style=flat-square&labelColor=1A1A1A&label=CI&logo=githubactions&logoColor=white" alt="CI status"></a>
  <a href="https://pypi.org/project/stochpylib/"><img src="https://img.shields.io/pypi/v/stochpylib?style=flat-square&labelColor=1A1A1A&color=FF8C00&logo=pypi&logoColor=white" alt="PyPI version"></a>
  <img src="https://img.shields.io/badge/public%20names-287%20of%20794-FF8C00?style=flat-square&labelColor=1A1A1A" alt="229 of 794 spec names implemented">
</p>

<p align="center">
  <img src="https://img.shields.io/badge/NumPy-4B5563?style=flat-square&labelColor=1A1A1A&logo=numpy&logoColor=white" alt="NumPy">
  <img src="https://img.shields.io/badge/SciPy-4B5563?style=flat-square&labelColor=1A1A1A&logo=scipy&logoColor=white" alt="SciPy">
  <img src="https://img.shields.io/badge/pytest-4B5563?style=flat-square&labelColor=1A1A1A&logo=pytest&logoColor=white" alt="pytest">
  <img src="https://img.shields.io/badge/setuptools-4B5563?style=flat-square&labelColor=1A1A1A" alt="setuptools">
  <img src="https://img.shields.io/badge/GitHub%20Actions-4B5563?style=flat-square&labelColor=1A1A1A&logo=githubactions&logoColor=white" alt="GitHub Actions">
  <img src="https://img.shields.io/badge/spl%20CLI-black?style=flat-square&labelColor=1A1A1A&logo=gnu-bash&logoColor=white" alt="spl command-line interface">
</p>

---

stochpylib is not a wrapper around existing statistical libraries — every distribution and
algorithm is implemented from scratch, with `scipy.special/optimize/integrate` used only as raw
numerical building blocks and `scipy.stats` serving as the independent test oracle. At its core
is a single load-bearing contract: every distribution exposes the same method set
(`.pdf()/.cdf()/.ppf()/.rvs()/.mean()/.var()/.skewness()/.kurtosis()/.entropy()/.mgf()/.cf()/.fit()/.ks_test()`),
every stochastic method takes a `random_state=` seed, and every Monte Carlo estimator returns a
shared result object carrying its point estimate together with an honest standard error and
confidence interval. Around that contract, seven modules are live today: a **probability
engine** (sample spaces, Bayes' theorem, exact-integer combinatorics, independence
testing), **47 distributions** across discrete/continuous/multivariate/heavy-tailed
families — including stable laws with Chambers–Mallows–Leckie sampling and numerically
inverted characteristic functions — a **Monte Carlo suite** spanning quasi-random
sequences (Sobol, Halton, Faure, Niederreiter), variance-reduction techniques (antithetic,
control variates, Latin hypercube, conditioned MC, rejection control), and applications
from option pricing validated against Black–Scholes to reliability analysis driven by the
library's own distribution objects, a **time-series toolkit** (ARIMA family, GARCH family,
Kalman/particle filters, HMMs, changepoints, spectral analysis), **Gaussian processes**
(a composable kernel zoo with exact, sparse and approximate-classification inference),
**survival analysis**
(Kaplan-Meier, Cox regression, parametric censored fits, competing risks),
**queueing theory** (M/M/1 to Jackson networks with discrete-event simulation) and **copulas**
(elliptical, Archimedean and empirical families plus C-/D-/R-vine dependence models).
The thesis this project exists to test: a complete stochastic-computing stack can live in
one coherent, well-tested package — the roadmap takes it onward through Lévy
processes, MCMC and beyond (23 modules, ~794 public names planned).

## Table of Contents

- [Quickstart](#quickstart)
- [Current Status](#current-status)
- [Download](#download)
- [Getting Started](#getting-started)
- [Requirements](#requirements)
- [CLI Reference](#cli-reference)
- [Project Layout](#project-layout)
- [Development Documentation](#development-documentation)
- [Test Suite](#test-suite)
- [Release Process](#release-process)
- [Roadmap](#roadmap)
- [Contributing](#contributing)
- [License](#license)

## Quickstart

```python
from stochpylib.probability import bayes_theorem, total_probability

# Classic disease-screening example: 1% prevalence, 99% sensitivity, 5% false-positive rate.
p_positive = total_probability((0.99, 0.01), (0.05, 0.99))
p_disease_given_positive = bayes_theorem(0.01, 0.99, p_positive)
print(round(p_disease_given_positive, 4))  # 0.1667
```

```python
from stochpylib.distributions import Normal, Weibull
from stochpylib.montecarlo import SobolSequence, AntitheticVariates

d = Normal(0.0, 1.0)
d.pdf(0.0); d.cdf(1.96); d.ppf(0.975); d.rvs(100, random_state=0)

fitted = Weibull.fit(lifetimes)          # maximum likelihood from data
stat, p_value = fitted.ks_test(data)     # goodness of fit

pts = SobolSequence(dim=5).generate(10_000)                    # low-discrepancy points
price = AntitheticVariates(n_simulations=100_000).price_european_call(
    S=100, K=100, T=1, r=0.05, sigma=0.2)                      # option pricing
```

```python
from stochpylib.gaussian_processes import GPRegression, RBFKernel

gp = GPRegression(kernel=RBFKernel(length_scale=1.0), noise=0.01).fit(X_train, y_train)
mu, sigma = gp.predict(X_test, return_std=True)     # mean + uncertainty

from stochpylib.copulas import CopulaFit

fit = CopulaFit().fit(returns_2d)                   # dependence modeling
simulated = fit.best_.sample(10_000)                # best family by AIC
```

## Current Status

Early development — eight modules implemented so far:

| Module | Public names | What's inside |
|---|---|---|
| `stochpylib.probability` | 21 | sample spaces, events, conditional probability, Bayes' theorem, combinatorics (factorial … derangements, Stirling, Bell, Catalan), independence checks |
| `stochpylib.distributions` | 60 | 47 distributions (discrete, continuous, multivariate, heavy-tailed) behind the common interface |
| `stochpylib.montecarlo` | 25 | quasi-random sequences, crude/QMC/importance/rejection/stratified estimators, variance reduction, applications |
| `stochpylib.timeseries` | 61 | ARIMA/SARIMA/ARFIMA/VAR/VECM, GARCH family, Kalman/EKF/UKF/particle filters, HMM & regime switching, changepoints, spectral analysis, classical diagnostics |
| `stochpylib.gaussian_processes` | 36 | composable kernel zoo (10 kernels with +/*/² operators), exact GP regression, FITC/VFE sparse approximations, Laplace/EP/VI classification, hyperparameter optimization, DeepGP |
| `stochpylib.copulas` | 26 | elliptical (Gaussian/t), Archimedean (Clayton/Gumbel/Frank/Joe/AMH/BB1/BB7) + Plackett, empirical (Empirical/Checkerboard/Beta), C-/D-/R-vines with AIC pair selection & rotations, CopulaFit dispatcher, dependence measures |
| `stochpylib.survival` | 28 | Kaplan-Meier/Nelson-Aalen/life tables, parametric censored fits (Weibull/Exponential/LogNormal/LogLogistic/Gompertz), Cox PH (Breslow/Efron), stratified Cox, Weibull AFT, Aalen additive, Fine-Gray competing risks, log-rank family, Aalen-Johansen CIF |
| `stochpylib.queueing` | 29 | M/M/1 through M/G/1 priority queues (closed-form), Jackson/closed/BCMP networks with product-form and MVA, Erlang B/C/Engset blocking formulas, discrete-event simulation engine |

Exact progress against the full design spec lives in
[`development/Implementation-Checklist.md`](development/Implementation-Checklist.md)
(currently **287 / 794 public names**).

## Download

If you just want to *use* stochpylib rather than develop on it, no source checkout is needed:

```bash
pip install stochpylib
spl --help        # overview of everything the library offers
```

> The PyPI release lands with the first tagged version (`v0.1.0`); until then the badge above
> will show "not found". For local development from this repository, `pip install -e .`
> registers the same `spl` command straight from source instead:

```bash
git clone https://github.com/leon1706-lol/Stochpylib.git
cd Stochpylib
pip install -e .
```

## Getting Started

For local development (this repo cloned, a virtual environment active):

```bash
pip install -e ".[dev]"     # runtime deps + pytest
pytest tests/ -v            # full test suite must be green before you start changing things
spl --version               # verify your editable install
spl --test                  # embedded self-check (133 checks), no pytest needed
```

Then implement or improve one module at a time and run the wrap-up procedure described in
[`CONTRIBUTING.md`](CONTRIBUTING.md).

## Requirements

- **Python ≥ 3.10**
- **NumPy** and **SciPy** (the only runtime dependencies)
- **pytest** for the development extras (`pip install -e ".[dev]"`)
- No compilers, no GPU, no other system packages — pure Python/NumPy/SciPy by design

## CLI Reference

Every install (PyPI wheel or `pip install -e .`) registers one console command, `spl`:

### `spl --help`

Prints a full inventory of the installed library: which modules are available, all public
functions per module, every distribution class (generated dynamically from the package, so it
never goes stale), the common distribution interface, and a runnable quick-start snippet.
Running bare `spl` shows the same thing.

### `spl --version`

```bash
$ spl --version
0.1.0
```

Prints the installed version — reads pip package metadata, falling back to the in-code version
when not installed through pip.

### `spl --test`

Runs the embedded self-check suite shipped inside the wheel (**133 checks**): package sanity and per-module spec
conformance, one closed-form spot check per distribution family, Monte Carlo
convergence sanity, cross-module workflows. This works
after any `pip install` — no pytest, no source checkout — making it the quickest way to verify
an installation. Exits non-zero on any failure.

## Project Layout

Every folder carries its own short `README.md` as an entry-point guide — this table links
them all:

| Folder | Guide | What lives there |
|---|---|---|
| `stochpylib/` | [package guide](stochpylib/README.md) | the installable package: module subpackages, `cli.py`, `selftest.py` |
| `stochpylib/probability/` | [module guide](stochpylib/probability/README.md) | core probability engine — complete, tested |
| `stochpylib/distributions/` | [module guide](stochpylib/distributions/README.md) | 47 distributions behind one common interface — complete, tested |
| `stochpylib/montecarlo/` | [module guide](stochpylib/montecarlo/README.md) | quasi-random sequences, estimators, variance reduction, applications |
| `stochpylib/timeseries/` | [module guide](stochpylib/timeseries/README.md) | linear/volatility models, filters, changepoints, spectral analysis |
| `stochpylib/gaussian_processes/` | [module guide](stochpylib/gaussian_processes/README.md) | composable kernels, exact/sparse GP regression & classification |
| `stochpylib/copulas/` | [module guide](stochpylib/copulas/README.md) | elliptical/Archimedean/empirical copulas, vines, dependence measures |
| `stochpylib/survival/` | [module guide](stochpylib/survival/README.md) | Kaplan-Meier, parametric fits, Cox/AFT/FineGray regression, competing risks |
| `tests/` | [suite guide](tests/README.md) | one deterministic test file per module, outside the installed package |
| `development/` | [dev-docs guide](development/README.md) | build history, bug audit log, progress checklist |
| `.github/` | — | CI / PyPI-publish / GitHub-Release workflows, issue & PR templates |
| `Stochpylib-Obsidian-Vault/` | — | the full design-spec vault; maintained privately, not part of this repo |

Workflow details: CI runs the full test matrix (Python 3.10–3.13) on every push; pushing a
`vX.Y.Z` tag builds + smoke-verifies the wheel, publishes to PyPI via Trusted Publisher
(OIDC, no stored tokens), and opens the matching GitHub Release automatically.

## Development Documentation

Everything a contributor or maintainer needs lives in committed docs:

- [`CONTRIBUTING.md`](CONTRIBUTING.md) — dev setup, ground rules, **semver & deprecation policy**, PR checklist
- [`CODE_OF_CONDUCT.md`](CODE_OF_CONDUCT.md) — Contributor Covenant 2.1
- [`SECURITY.md`](SECURITY.md) — private vulnerability reporting (72 h acknowledgment)
- [`development/Development.md`](development/Development.md) — layout decisions & workflow notes
- [`development/CHANGELOG.md`](development/CHANGELOG.md) — append-only log, one entry per build phase
- [`development/Probleme.md`](development/Probleme.md) — bug audit log in Problem → Fix →
  Verification format with a status legend (30 entries so far)
- [`development/Implementation-Checklist.md`](development/Implementation-Checklist.md) — every planned public name as a checkbox

## Test Suite

```bash
pytest tests/ -v
```

**447 passed / 2 skipped** as of the `survival` module. Tests are deterministic
(fixed seeds everywhere), live outside the installed package, and use `scipy.stats`,
`statsmodels` and brute-force references as independent oracles. Statistical assertions are
set at ≥ 3 standard errors so results are stable while staying meaningful. Additionally,
`spl --test` re-verifies any installation in seconds.

## Release Process

Releases are fully automated from tags:

1. Update the version in `pyproject.toml` **and** `stochpylib/__init__.py` (semver — see the
   policy in [`CONTRIBUTING.md`](CONTRIBUTING.md))
2. Tag and push:
   ```bash
   git tag v0.1.0 && git push origin v0.1.0
   ```
3. CI runs the full test matrix, builds sdist + wheel, smoke-verifies the wheel
   (`spl --version`, `spl --test`) and publishes to PyPI via Trusted Publisher (OIDC — no API
   tokens stored anywhere); a second workflow creates the matching GitHub Release with
   auto-generated changelog notes

Prerequisite for step 3: configure the Trusted Publisher once under pypi.org → your project →
Publishing.

## Roadmap

Fifteen modules remain on the spec (in rough implementation order):
queueing theory, information theory, Lévy processes,
financial stochastics, advanced MCMC, Bayesian inference, statistics, nonparametric
methods, robust statistics, numerical methods, random matrix theory, spatial statistics,
optimization, experimental design, visualization, and utilities. Each lands with the same bar: native implementations, the shared
interface conventions, full tests against independent oracles, and honest documentation of
deviations.

## Contributing

Issues and PRs welcome! Start with [`CONTRIBUTING.md`](CONTRIBUTING.md) — it covers the ground
rules (no `scipy.stats` wrapping in library code, the common interface contract, where tests
live) and the versioning/deprecation policy. Bug reports go through the issue templates;
security issues privately per [`SECURITY.md`](SECURITY.md). This project follows
[`CODE_OF_CONDUCT.md`](CODE_OF_CONDUCT.md).

## License

[MIT](LICENSE) © Leon Schwarzkopf

---

<p align="center">
  Built by <strong>Leon Schwarzkopf</strong>, <a href="mailto:leonschwarzkopf08@gmail.com">leonschwarzkopf08@gmail.com</a>
</p>

---

<div align="center">
  <sub>stochpylib</sub>
</div>
