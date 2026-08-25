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

## Phase 9 - Resolving the two documented open items ([9]/[12], [11])

Closed both known limitations. (1) Joe-Kuo direction numbers: embedded the standard 64-dim x
30-col Sobol table (_direction_numbers.py), extracted from the scipy.stats.qmc oracle via the
x_{2^b} = v_b identity and verified dyadic + bitwise round-trip before embedding; SobolSequence
uses it by default, new generate_block(m) API returns the aligned first-2^m block including the
origin point - exactly balanced in every dimension and set-identical to scipy's block (scipy
enumerates along the Gray walk; we output natural order). GF(2) machinery stays as fallback
beyond dim 64 and for custom/Niederreiter nets. Root cause of the old +-1 imbalance identified:
it belongs to the origin-skipped streaming window, not to the net itself. (2) alpha=1 skewed
stable sampling: a twelve-variant empirical hunt for a matching closed-form CML formula failed
(best residual 0.077 vs noise 0.003; shift-fitting proved structural mismatch), so implemented
a cached numerical quantile table instead - exact Gil-Pelaez CDF inside the reliable window
(|x-loc| <= 25 sigma) refined by monotone PCHIP, with exact power-law tail asymptotics
(1-F ~ c(1+beta)/(pi x)) grafted beyond down to q=1e-9; draws are O(1) lookups after a ~5 s
per-parameter-set warmup (class-level cache). Empirical cf matches at MC-noise level; central
quantile error <= ~1e-3 scale. New regression tests in both suites; full suite: 185 passed /
2 skipped. Probleme [11] -> Fixed, [12] added -> Fixed.

## Phase 10 - Fourth module: `stochpylib.timeseries`

Implemented the complete time-series toolkit (61/61 spec names across nine submodules),
natively on numpy/scipy with statsmodels as a dev-only test oracle - formally resolving
the wrap-vs-reimplement question recorded in the vault's Dependencies.md. Highlights:
Hannan-Rissanen + CSS estimation for the ARMA family; SARIMA seasonal lag structures;
ARFIMA fractional filtering (fixed-width binomial window); VAR OLS / VARMA CSS / VECM
reduced-rank regression; the GARCH family via Gaussian QMLE (ARCH/GARCH/IGARCH/TGARCH/
GJRGARCH/EGARCH/APARCH/FIGARCH plus CCC-MGARCH and scalar two-step DCC); Kalman filter +
RTS smoother, EKF, UKF, bootstrap particle filter and a GPB(1) Rao-Blackwell mixture
filter; Gaussian HMM (Baum-Welch + Viterbi), Markov-switching regression/AR and mixture
AR; PELT/binseg/bottom-up changepoint detection plus Adams-MacKay BOCPD; periodogram,
Welch PSD, Morlet CWT, DWT/IDWT (haar/db2, perfect reconstruction), STFT, Hilbert
transform; ADF/KPSS/PP/Ljung-Box/DW/ARCH-LM/Granger/Johansen diagnostics; forecasting
dispatchers, confidence bands, walk-forward backtesting and rolling-origin CV.
Conventions introduced: fluent .fit() returning self and ForecastResult result objects.
Eleven construction/logic bugs were caught by smoke tests before shipping and are logged
in Probleme.md [13]-[19] (integrated-model forecast seeding, FIGARCH filter sign, KPSS
interpolation direction, ADF explicit-lag semantics, particle-filter shape defense, BOCPD
reset-hypothesis predictive, DWT normalization/synthesis pair). Oracle checks: AR/VAR
coefficients match statsmodels exactly (shared OLS), ADF/Ljung-Box statistics to 1e-8;
GARCH recovery on known DGPs within tolerance. Manual session exercised the full
diagnose-fit-forecast flow on simulated ARMA+GARCH data. Suite: 236 passed / 2 skipped;
selftest extended to 111 checks. Progress: 167/794 public names. Version bumped to 0.2.0.

## Phase 11 - Fifth module: `stochpylib.gaussian_processes`

Implemented the full GP module (36/36 spec names across five submodules), natively on
numpy/scipy with no new runtime dependencies. Kernel zoo: 10 covariance functions (RBF,
Matérn closed forms, Periodic per-dim, Linear, Polynomial, RationalQuadratic, WhiteNoise,
SpectralMixture, NeuralNetwork, ArcCosine) all callable and composable via operator
overloading (+/*/×²) — the load-bearing ARCHITECTURE convention now fully realized.
kernel_ops: KernelSum/Product/Power/Composition with flattened parameter trees for
optimization; kernel_matrix and finite-difference kernel_grad. Models: ExactInference
(Cholesky solve + LML), GPRegression, GaussianProcess base, GPTimeSeriesModel.
Classification: LaplacePropagation (RW Alg 3.1, logit/probit links, predictive probit
correction), ExpectationPropagation (experimental — documented convergence issues,
Probleme [20]), VariationalInference (Jaakkola-Jordan bound, logit only). Sparse:
FITC/VFE/SparseVFE with Titsias closed-form posterior over inducing variables; verified
against exact GP predictions. DeepGP: documented two-layer composition (sparse latent →
observed). Hyperparams: MarginalLikelihood, optimize_hyperparams (L-BFGS on log-ML),
ARD initializer, cross_validate_gp. Seven construction bugs caught by smoke tests
(Probleme [13]-[19] from timeseries plus [21] DWT normalization, [22] NN kernel formula).
Manual session: composed kernels, optimized hyperparameters, compared sparse vs exact.
Tests: tests/gaussian_processes/tests.py (28 cases); selftest extended to 117 checks.
Full suite: 264 passed / 2 skipped. Progress: 203/794 public names.

## Phase 12 - Library audit: completing gaussian_processes delivery + stability fixes

Went through the whole library auditing implemented-vs-spec per
Implementation-Checklist.md. Found that the Phase 11 wrap-up had been committed with the
documentation steps unfinished and three spec names silently missing, plus two real
defects the existing tests could not see:

- **Delivery gaps closed:** `GPClassification` (spec-facing binary classifier facade over
  Laplace/EP/VI engines), `SparseGaussianProcess` (alias of VFE/Titsias SGPR) and
  `InducingPointGP` (alias of FITC) added - GP module now truly 36/36 spec names; module
  wired into `stochpylib/__init__.py` (was invisible from the package root);
  selftest extended with a GP section to the documented 117 checks; spl --help inventory
  extended to montecarlo/timeseries/gaussian_processes and its roadmap line no longer
  lists implemented modules as planned.
- **Probleme [21]:** removed a broken duplicate FITC/VFE copy inside inference.py whose
  predict path raised AttributeError (`_predict_core` never existed); sparse.py is now
  the single source.
- **Probleme [23]:** rewrote the sparse engines in the whitened parameterization after
  finding the old raw-inverse-of-unjittered-Kuu posterior exploded for larger inducing
  counts (deviation up to ~157) and logged invalid values in the LML. Verified against a
  brute-force Titsias reference, the M=T identity (equals exact GP to ~1e-12) and monotone
  M-convergence; replaced the weak corr>0.30 test assertion that had encoded the defect.
- **Probleme [24]:** fixed `BaseKernel.diag` calling `_matrix(X)` without Y (crashed for
  Matern/Periodic/RQ/NN/ArcCosine/SpectralMixture on any exact-GP predict) and gave
  KernelProduct/KernelPower exact diag overrides (composed-kernel predictions used to
  crash). Found by the manual debug session on the first composed-kernel prediction.
- Backfilled Probleme entries [20] (EP convergence caveat) and [22] (NN kernel formula)
  that CHANGELOG Phase 11 referenced but never wrote.
- Manual session (13 checks, all pass): composed RBF*Periodic regression with CI coverage,
  hyperparameter optimization improving LML, CV, sparse-vs-exact for both aliases,
  GPClassification accuracy + calibration, GPTimeSeriesModel forecast calibration +
  honest std growth, DeepGP smoke, SpectralMixture PSD.
- Tests: tests/gaussian_processes/tests.py grown from 28 to 42 cases; CLI overview test
  extended. Full suite: **278 passed / 2 skipped**; spl --test 117 checks OK.
- Docs: checklist updated to 203/794 with GP fully checked off; new
  stochpylib/gaussian_processes/README.md; root/package/tests/dev readmes refreshed;
  vault Modules/gaussian_processes.md deviations corrected; code graph regenerated;
  HANDOFF backfilled for Phase 11 and appended for this phase.

## Phase 13 - Sixth module: `stochpylib.copulas`
Implemented dependence modeling end to end (26/26 spec names across five submodules),
natively on numpy/scipy.special/optimize/integrate with scipy.stats as test oracle only.
Elliptical: exact recursive-integration CDF for any dimension (validated vs a bivariate
normal quadrature oracle to ~1e-16), tau-based correlation + profile-MLE degrees of
freedom. Archimedean: generator framework (CDF = psi(sum phi)), exact bivariate
densities via psi'', Genest-MacKay tau, tau-inversion fits (closed forms + cached
numeric curves; Frank via Debye D1), Marshall-Olkin/Kanter fast samplers plus generator-
derivative conditional inversion; BB1/BB7 two-parameter tails; Plackett odds-ratio family.
Empirical: e.c.d.f., checkerboard with multilinear CDF, Bernstein/Beta smoothing.
Vines: one recursive edge machinery behind C/D/R structures, AIC pair-family selection
with rotations, Disshmann-style MST R-vine selection, peel-order sequential Rosenblatt
sampler. Methods: CopulaFit dispatcher, CopulaSample, tail_dependence, copula_density,
conditional_copula, kendall_tau, spearman_rho.
Six construction defects were caught and fixed during validation (Probleme [25]-[30]):
wrong elliptical chain-rule CDF, wrong conditional transform family in samplers,
archimedean generator algebra errors, O(n^2) Kendall-tau memory blowup, Frank/Joe tau-
inversion hangs, vine rotation-h/away-head/mirror-side/stale-cache cluster.
Manual session (10 checks ALL PASS): t-copula df recovery + analytic-vs-empirical tail
dependence, CopulaFit ranking on clayton data, 5-d RVine fit/sample/refit with pairwise
tau recovery corr=1.00, Gumbel upper-tail estimation.
Tests: tests/copulas/tests.py (51 cases); selftest extended to 122 checks; spl --help
gained the copulas block. Full suite: 329 passed / 2 skipped. Version bumped to 0.3.0.
Progress: 229/794 public names.

## Phase 14 - V0.3.1 audit: spec conformance suite + cross-module tests + doc sync

Library-wide audit requested for V0.3.1: verify every implemented module is
complete per spec, documentation reflects reality, and end-to-end coverage
exists. Outcome:
- Implementation state confirmed complete (229/794 names across six modules);
  the only contract deviations are the documented multivariate ones (7 classes
  expose pdf instead of pmf and omit scalar-argument mgf/cf).
- New cross-module suite tests/library/tests.py (20 cases): spec-name
  conformance generated from development/Implementation-Checklist.md via
  tests/library/_extract_spec_names.py (lists cached in _spec_names.json),
  pinned documented extras (MCResult, DigitalNetBase2, timeseries result
  objects, GP kernel base/ops, BaseCopula), and end-to-end workflows spanning
  modules: reliability_mc on library Weibull vs closed form, t-copula margins
  through library Student_t (KS), ARIMA vs GPTimeSeriesModel short-horizon
  agreement, CopulaFit->sample->refit round trip, Sobol-QMC vs crude estimator
  consistency.
- spl --test extended from 122 to 130 checks: per-module export conformance,
  distributions contract spot check, reliability closed-form and t-copula df
  recovery - all runnable from any pip install without pytest.
- Documentation sync: CONTRIBUTING.md selftest count (101 -> 130), root README
  spl --version example and Quickstart gained GP/copulas snippets, README
  selftest description now mentions conformance/cross-module checks,
  tests/README documents tests/library/, vault ARCHITECTURE.md status updated
  to 229/794 across six modules.
Suite: 349 passed / 2 skipped. Version 0.3.1 (tests + docs only; no API
changes).

## Phase 15 - Seventh module: `stochpylib.survival`
Implemented survival and reliability analysis end to end (28/28 spec names across
six submodules), natively on numpy/scipy.special/optimize/integrate. Nonparametric:
Kaplan-Meier (Greenwood CI loglog/linear), Nelson-Aalen, actuarial life tables,
EmpiricalSurvival, BreslowEstimator. Parametric: Weibull/Exponential (closed-form
rate MLE)/LogNormal/LogLogistic/Gompertz censored likelihood with AIC. Regression:
Cox PH with vectorised suffix-sum risk sums (Breslow/Efron tie handling),
concordance index, Breslow baseline; StratifiedCox with per-stratum baselines;
Weibull AcceleratedFailureTime; AalenAdditiveModel with dN_i(u)-response LS and
stabilisation guards; FineGrayModel weighted-Cox scoring. Log-rank family:
LogRankTest, WilcoxonSurvival(Gehan-Breslow), TaroneWareTest, PetoTest,
FlemingHarrington(rho,gamma). Competing risks: CauseSpecificHazard,
Aalen-Johansen CIF with exact sum+KM=1 identity, CompetingRisksModel facade.
Functions wrappers bridge data fits and distribution objects via uniform
predict() surface. lifelines added as dev-only test oracle extra (importorskip
cross-checks for KM/NA/Cox). CI: windows-latest added to matrix.
Tests: tests/survival/tests.py (37 cases incl 3 lifelines oracles); selftest
extended to 136 checks; spl --help gained the survival block.
Suite: 387 passed / 2 skipped. Version bumped to 0.4.0.
Progress: 257/794 public names.

## Phase 16 - Eighth module: `stochpylib.queueing`
Implemented queueing theory and networks end to end (29/29 spec names across
five submodules), natively on numpy/scipy. Single queues: M/M/1 closed-form,
M/M/c with Erlang-C via birth-death module, M/M/inf (no-wait limit), M/D/1
Pollaczek-Khinchine, M/G/1 P-K formula with second-moment input, GI/G/1
Kingman heavy-traffic approximation, MG1PriorityQueue non-preemptive two-class.
Birth-death: general steady-state solver, Erlang B/C formulas, Engset formula.
Networks: JacksonNetwork (open, traffic equations via linear solve),
OpenNetwork alias, ClosedNetwork/GordonNewell mean-value analysis,
BCMP theorem types 1 and 3, ProductFormNetwork base class.
Simulation: DiscreteEventSim event-calendar engine with warmup filtering,
SimStats collecting wait/sojourn/service times and time-averaged populations;
QueueSimulation facade comparing analytical vs simulated results.
Analysis: LittleLaw solver, traffic_intensity, mean_waiting_time,
mean_queue_length, server_utilization, WaitingTimeDistribution (exact CDF).
Tests: tests/queueing/tests.py (43 cases); selftest extended to 136 checks;
spl --help gained the queueing block.
Suite: 447 passed / 2 skipped. Version bumped to 0.5.0.
Progress: 287/794 public names.
