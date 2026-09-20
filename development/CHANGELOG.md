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

## Phase 17 - V0.5.1 audit: bug fixes, edge-case tests, doc sync

Library-wide V0.5.1 audit covering all eight modules. Outcome:
- Implementation confirmed complete (257/794 spec names across 8 modules).
- Four bugs found and fixed in the survival module:
  (a) SurvivalFitter._step_evaluate defaulted to 1.0 for all callers,
      returning H=1.0 instead of H=0.0 before the first event for cumulative
      hazard step functions; added default parameter.
  (b) CumulativeHazard integration grid started at times.min()*0.5 instead
      of near zero, missing accumulated hazard below query range.
  (c) HazardFunction rejected library distribution objects lacking a .hazard()
      method; added generic pdf/(1-cdf) fallback.
  (d) Gompertz exp(b*t) overflowed for large b*t products; clipped exponent.
- One test bug fixed: ARIMA(1,1,0) trend-continuation test checked for the
  slope (~0.05) instead of the forecast level (~9.0).
- Added 20 new edge-case and cross-module tests across survival, queueing,
  copulas, GP, and library suites.
- Documentation synced: CONTRIBUTING selftest count, ARCHITECTURE status line,
  root README version example and quickstart snippets.
Suite: 406 passed / 2 skipped. Version 0.5.1 (bug fixes + tests only; no new
features or API changes).

## Phase 18 - Ninth module: `stochpylib.information_theory`
Implemented information-theoretic measures end to end (31/31 spec names across
five submodules), natively on numpy/scipy. Entropy: Shannon (discrete),
JointEntropy, ConditionalEntropy, CrossEntropy, TsallisEntropy (q-generalised),
RenyiEntropy (alpha-generalised, converges to Shannon at alpha=1),
DifferentialEntropy (histogram-based continuous), MaxEntropy (bounded
optimisation). Divergences: KLDivergence/RelativeEntropy, JensenShannonDivergence
(symmetric/bounded), WassersteinDistance (scipy 1-D), HellingerDistance,
TotalVariation, ChiSquaredDivergence, AlphaDivergence. Mutual info:
MutualInformation (contingency table), NormalizedMutualInformation,
VariationOfInformation, ConditionalMutualInfo, InteractionInformation,
MultiInformation (total correlation). Channels: ChannelCapacity (BSC/BEC/Z),
InformationGain, TransferEntropy (lagged conditional MI with discretisation),
DirectedInformation, SymbolicTransferEntropy (ordinal-pattern encoding).
Coding: ShannonLimit (BSC capacity), HuffmanCode (optimal prefix-free tree),
TypicalSet (AEP membership test), AEP bounds.
Also backfilled Probleme.md entries [31]-[34] for V0.5.1 survival bug fixes.
Tests: tests/information_theory/tests.py (45 cases); selftest extended to
136 checks; spl --help gained the information_theory block.
Suite: 496 passed / 2 skipped. Version bumped to 0.6.0.
Progress: 288/794 public names.

## Phase 19 - V0.6.1 audit: InformationGain bug fix, Renyi bits fix, edge-case tests

V0.6.1 library-wide audit of information_theory module. Found and fixed:
(a) InformationGain computed H(Y) from raw categorical labels instead of
    frequency counts, producing wildly inflated IG values (6.3 instead of 0.01);
(b) RenyiEntropy(alpha=0) used natural log instead of log2, returning Hartley
    entropy in nats (1.3863) instead of bits (2.0).
Added 12 new edge-case tests: Renyi alpha=0 in bits, CMI compute returns float,
TE bias floor for independent data, MaxEntropy with mean constraint,
AlphaDivergence near alpha=1 approximates KL, InformationGain equals MI,
TypicalSet biased source detection, MultiInfo independent ~ 0, VI(X;X)=0,
InteractionInfo XOR negative, AEP bounds consistency.
Suite: tests/information_theory 55 cases; full suite 498 passed / 2 skipped.
Version 0.5.1 -> 0.6.1 (bug fixes + tests only).

## Phase 20 - V0.6.2 documentation overhaul: AQ-style restructure, corrected progress accounting, doc-consistency suite

Library-wide documentation cleanup phase (no API changes; the only library-code
change is the spl --help inventory fix, Probleme [38]). Outcome:

- **Progress accounting corrected:** the queueing section of
  Implementation-Checklist.md had shipped complete but was never checked off,
  and the headline figure had drifted to a mathematically impossible 288/794
  (Probleme [37]). True implemented total: **317/794 across nine modules**
  (21+60+25+61+36+26+28+29+31). Checklist queueing section checked off,
  progress line corrected, tests/library conformance test restored to the
  strict == 317 invariant.
- **Documentation restructured in the Aether-Quant house style** (dense
  ownership-first prose, index tables, honest known-limitations sections):
  all nine module READMEs rewritten to a common template (status line, Files,
  Conventions, Known limitations, spec/tests pointers) including a NEW
  stochpylib/information_theory/README.md (the only implemented module without
  one); stochpylib/README.md now carries the full nine-subpackage table with
  per-module guide links; tests/README.md documents the docs-consistency suite
  and defers the live pass count to the README badge; development/README.md
  converted to an index table.
- **New development docs:** development/architecture.md (objective, system-flow
  and tech-stack mermaid diagrams, module map, the common distribution
  contract, cross-cutting conventions, package layout rules, known design
  gaps), development/infrastructure.md (local setup, spl CLI, GitHub Actions
  workflows, PyPI Trusted-Publisher release pipeline, wheel hygiene),
  development/project_structure.md (annotated directory tree). Development.md
  reworked into the folder index; the repo banner now uses the new project
  logo (development/logo.png).
- **Main README rebuilt:** AQ-style banner (logo, tagline, identity + tech
  badge rows), Known Limitations section (PyPI lag: latest published release
  0.1.1 vs repo 0.6.1; 14 of 23 modules remaining; the sanctioned multivariate
  deviation; experimental EP; heavy suite), Current Status table extended with
  information_theory, every stale figure replaced (tests 522 passed / 2
  skipped, selftest 136 checks, version example 0.6.1, 317/794, roadmap lists
  exactly the 14 remaining modules), footer in the house style.
- **Doc-consistency suite added:** tests/docs/tests.py (14 cases) - the README
  test badge and pass/skip statement must match live pytest collection; spl
  --test check count must match every doc that quotes it; version strings must
  agree across pyproject/__init__/README; the Current Status table must cover
  every subpackage with spec-accurate name counts (from
  tests/library/_spec_names.json); all relative links and TOC anchors must
  resolve; stale claims ("288", "496", "133 checks", "0.1.0", ...) fail the
  suite; checklist progress line is recomputed from its own sections;
  Probleme numbering must stay continuous; the logo must exist and be
  referenced. Runs in CI unchanged (ci.yml executes the full pytest tree).
- **Bugs found by the new suite and fixed:** spl --help was missing the
  queueing and information_theory inventory blocks despite the Phase 16/18
  changelog claims (Probleme [38]); tmp_fix_checklist.py leftover scaffolding
  removed from the repo root.
- **Probleme backfill:** entries [35] (V0.6.1 InformationGain counts) and [36]
  (V0.6.1 Renyi alpha=0 bits) were referenced by Phase 19 but never written;
  now recorded. New entries [37]/[38] for this phase's finds.
- **Manual session (all pass):** every README quickstart snippet run verbatim
  (Bayes 0.1667; Weibull fit recovery 2.0/10.0 -> 2.065/10.314 with KS p=0.92;
  Sobol block + antithetic call 10.488 +- 0.033; GP predict shapes; CopulaFit
  -> GaussianCopula on t margins), BSC capacity exact against the closed form,
  Huffman optimality within the [H, H+1] bound, M/M/1 closed form exact
  (L=4, Wq=4, rho=0.8), spl CLI surface end to end.

Suite: 524 collected - 522 passed / 2 skipped (the 2 permanent skips are the
VonMises/Kumaraswamy scipy cross-checks with no direct oracle mapping). Docs
only; no API changes; version stays 0.6.1.

## Phase 21 - V0.6.3 CI fix: version-literal test made bump-proof, README CLI TOC sub-links

Hotfix follow-up to the V0.6.2 documentation overhaul:

- **CI failure fixed (Probleme [39]):** test_top_level_package_wiring asserted
  the version as the literal "0.6.1"; the V0.6.2 bump broke it on CI's fresh
  install (metadata 0.6.2 != literal), failing one matrix job ~80 s in and
  fail-fast-cancelling the other seven. The test now asserts pip-metadata ==
  __version__ consistency instead of any literal. Lesson recorded: run the
  library suite AFTER refreshing the editable install when the version changed,
  not before.
- **README:** CLI Reference gained Aether-Quant-style Table-of-Contents
  sub-links to the individual command sections (spl --help, spl --version,
  spl --test), anchor-checked by the docs suite.
- Version bumped to 0.6.3 (test-infra fix + docs only; no API changes).

## Phase 22 - V0.6.4 CLI expansion: PyPI-aware versioning, spl update, info/show/demo/cite

The spl CLI grew from three flags to a full command surface (no library-API
changes; new package modules cli_pypi.py and cli_demo.py):

- **PyPI awareness (spl --version):** prints the installed version plus the
  latest version published on PyPI with the relationship stated (update
  available / up to date / installed newer-unreleased). Non-blocking and
  offline-safe by construction: 4 s timeout, 24 h on-disk cache, clear
  "PyPI check unavailable" degradation, STOCHPYLIB_SKIP_UPDATE_CHECK=1 kills
  all PyPI traffic. New cli_pypi.py keeps the logic pure and injectable
  (fetch/cache/parse/install-mode detection via direct_url.json).
- **spl --version --list:** lists every version ever published on PyPI in
  release order, marking the installed one (* installed) and the newest
  (latest); always fetches fresh.
- **spl update [--vers X] [--yes] [--dry-run] [--force]:** switches the pip
  package to any published version (upgrade, downgrade, pin). Validates the
  target against PyPI's real release list, refuses unknown versions, refuses
  editable/source installs without --force, prints the exact pip command, and
  asks for confirmation unless --yes; --dry-run executes nothing.
- **spl info:** environment report (install mode, python/platform, numpy/scipy
  versions, module inventory with per-module public-name counts).
- **spl show <Name>:** qualified path + signature + docstring of any public
  name across all modules, with difflib suggestions and non-zero exit on a
  miss (case-insensitive fallback first).
- **spl demo [module]:** nine fast deterministic mini-examples, one per
  implemented module, run live against the installation (Bayes screening,
  Weibull fit + KS, Sobol + antithetic call vs Black-Scholes, AR(1) fit +
  forecast, GP regression with uncertainty, copula AIC selection,
  Kaplan-Meier, M/M/1 closed form, entropy + Huffman); bare spl demo lists
  them. New cli_demo.py.
- **spl cite:** plain-text + BibTeX citation, versioned with the installed
  release.
- **spl --help:** module inventory now generated from each module's __all__
  (per-module public-name counts, total, 9-of-23 header) so it cannot go
  stale; the distributions block keeps the full dynamic class-name listing;
  subcommand epilog added.
- **selftest extended 136 -> 139 checks:** version_key numeric ordering,
  update_available status matrix, install_mode classification (all offline).
- **Tests: new tests/cli/tests.py (50 cases)** - PyPI parsing/cache-TTL/
  offline/skip-env paths with urlopen mocked to raise on any call; --list
  rendering; update validation/dry-run/prompt/editable/pip-failure paths with
  subprocess and input mocked; info/show/demo/cite output contracts; a
  sweep asserting every public name of every module resolves through
  spl show; --help regression. No test touches the network or pip.
  Probleme [40] records the sentinel-injection flaw caught during
  development (_meta=None briefly allowed a real pip call in a test).
- **Docs synced:** README CLI Reference rewritten (all seven sections + TOC
  sub-links), badge/test counts (572 passed / 2 skipped of 574), infrastructure.md
  CLI table, tests/README.md suite layout, CONTRIBUTING/selftest counts,
  stochpylib/README. Version bumped to 0.6.4.
- **Manual session (all pass, live):** spl --version --list against real PyPI
  (latest published 0.1.1; 2 releases), update dry-run/unknown-version/
  editable-refusal paths, info, show (hit, case-fallback, suggestions), demo
  runs (M/M/1 exact, MC call 10.41+-0.05 vs BS 10.45, Huffman optimal), cite,
  help inventory.

Suite: 574 collected - 572 passed / 2 skipped. Version 0.6.4.

## Phase 23 - V0.7.0 levy_processes: Lévy-Khintchine core, jump-diffusion pricing, subordinators, advanced point processes, SDE solvers (33 names)

The tenth module: `stochpylib.levy_processes`, five submodules, 33 public
names, no new runtime dependencies.

- **`levy.py`:** `LevyProcess` (Brownian + compound-Poisson base with a
  Lévy-Khintchine characteristic function), `StableProcess` (alpha-stable
  Lévy motion via the library's validated Chambers-Mallows-Stuck sampler),
  `AlphaStableDistribution` (adapter over `distributions.AlphaStable`),
  `SpectrallyPositive` (nondecreasing alpha-stable), `SubordinatedProcess`
  (`X_t = B_{T_t}` for any base process + subordinator), `LevyKhintchine`
  (the `(b, sigma^2, nu)` triplet, both compound-Poisson and pure-subordinator
  jump forms).
- **`subordinators.py`:** `Subordinator` base (path simulation machinery),
  `GammaSubordinator` (Variance-Gamma time change), `InverseGaussianSubordinator`
  (NIG time change, `E[T_t]=t`), `StableSubordinator` (positive
  1/alpha-stable, `E[exp(-lam T_t)] = exp(-t lam**alpha)`), `TemperingSubordinator`
  (CGMY/tempered-stable time change via truncated compound-Poisson with
  analytic mean compensation).
- **`jump_diffusion.py`:** `JumpDiffusion` base, `MertonJumpDiffusion`
  (lognormal jumps, closed-form call series), `KouJumpDiffusion`
  (double-exponential jumps, Carr-Madan Fourier call price), `BatesModel`
  (Heston stochastic volatility + lognormal jumps, MC pricing),
  `VarianceGammaProcess`, `CGMYProcess`, `NormalInverseGaussianProcess`.
- **`advanced.py`:** `HawkesProcess` (exponential-kernel self-exciting
  process: Ogata thinning, exact recursive MLE, `branching_ratio()`,
  time-rescaling `ks_residuals()`), `MultivariateHawkes`, `CoxProcess`,
  `RenewalProcess`, `BranchingProcess` (Galton-Watson), `SemiMarkovProcess`,
  `GaussianRandomField` (FFT spectral synthesis), `RandomMeasure` (gamma or
  spectrally-positive stable mass on intervals).
- **`sde.py`:** `SDE` definition (finite-difference derivatives for the
  higher-order schemes), `Euler_Maruyama` (strong order 0.5), `Milstein`
  (strong order 1.0), `Runge_Kutta_SDE` (derivative-free Milstein/Platen,
  strong order 1.0), `StochasticTaylor` (Kloeden-Platen strong order 1.5),
  `WeakApproximation` (Talay-Tubaro weak order 2), `StrongApproximation`
  (strong-error convergence studies across step sizes).
- **Eleven bugs found and fixed while writing `tests/levy_processes/tests.py`**
  (Probleme.md #41-51), on top of the ten already fixed during initial
  implementation: the Carr-Madan pricer's log-strike convention (`log(K/S0)`
  vs the required absolute `log(K)`, which had made every `KouJumpDiffusion.call_price`
  wrong by an order of magnitude); the truncated-jump quantile sampler's
  linear grid (biased `TemperingSubordinator`/`CGMYProcess` simulated means
  by up to ~50% — replaced with a log-spaced grid + cumulative-trapezoid
  quadrature); `CoxProcess.simulate` crashing on numpy >= 2.x (`np.trapz`
  removed); `StableSubordinator`/`RandomMeasure(kind="stable")` not matching
  their documented Laplace transform (an uncancelled S1-parameterization
  constant); `Runge_Kutta_SDE` actually being a weak-order stochastic-Heun
  scheme (empirical strong order ~0.5, not the documented 1.0) — replaced
  with the derivative-free Milstein/Platen scheme; `StochasticTaylor`'s
  multiple stochastic integrals not matching Kloeden-Platen (empirical order
  ~1.0, not 1.5); `StrongApproximation` sharing one already-advanced RNG
  between the "exact" and "approximate" solvers, so the two paths were never
  driven by the same Brownian path and the measured "error" never shrank
  with step size; `WeakApproximation`'s three-point increment distribution
  using uniform 1/3 weights instead of the required 1/6-2/3-1/6, doubling
  `E[dW**2]` and biasing `E[X_T]` by an amount that did not shrink with
  refinement; and `HawkesProcess.ks_residuals()` always raising when called
  with no arguments right after `.fit()` (the fitted events were never
  stored). All ten are independently reproduced and regression-tested.
- **`tests/levy_processes/tests.py` (40 new tests):** subordinator
  monotonicity/mean/Laplace-transform checks, Lévy-Khintchine
  characteristic-function consistency (manual-formula and empirical-CF
  cross-checks), jump-diffusion pricing vs Black-Scholes in the no-jump
  limit plus Monte Carlo cross-checks at matched risk-neutral drift, Hawkes
  simulate/fit/branching-ratio/KS-residuals, Cox/renewal/branching/
  semi-Markov moment checks, Gaussian random fields and random measures, and
  SDE strong/weak convergence-order studies (EM=0.5, Milstein=1.0, RK=1.0,
  Taylor=1.5, weak-2 vs the exact GBM mean).
- **`selftest.py` extended 139 -> 145 checks:** `levy_processes` CONFORM spot
  check plus `TemperingSubordinator` mean, `GammaSubordinator` mean, Kou
  zero-jump vs Black-Scholes, Hawkes `branching_ratio`, and an
  Euler-Maruyama GBM terminal-mean check.
- **`spl demo levy_processes`** added to `cli_demo.py` (Kou call price via
  Carr-Madan vs Monte Carlo, plus a `TemperingSubordinator` sample path); also
  fixed `cli_demo.DEMO_MODULES` being declared in `__all__` without ever
  being defined (Probleme.md #49).
- **Docs synced:** README badges/status table/architecture diagrams/roadmap,
  `stochpylib/README.md` module table, `development/architecture.md` module
  map + diagrams, `development/infrastructure.md` and `tests/README.md`
  selftest-count mentions, `stochpylib/levy_processes/README.md` (new),
  `development/Implementation-Checklist.md` (33/33, progress line to
  350/794), `development/Probleme.md` (#41-51). Version bumped to 0.7.0 in
  both `pyproject.toml` and `stochpylib/__init__.py`.

Suite: 614 collected - 612 passed / 2 skipped. Version 0.7.0.

## Phase 24 — CI hotfix: statsmodels 0.15.0 dropped `AutoReg`'s `old_names`

- **Fixed `test_ar_recovery_and_statsmodels_exact`:** dropped the
  `old_names=False` kwarg from the `statsmodels.tsa.ar_model.AutoReg` oracle
  call — statsmodels 0.15.0 removed the long-deprecated parameter, breaking
  all 8 CI matrix jobs on push (1 real failure, 7 cancelled by fail-fast).
  The default was already `False` on the `statsmodels>=0.14` floor, so
  dropping it is a no-op there and fixes 0.15.0. Verified against both
  versions (Probleme.md #52).

Suite: 614 collected - 612 passed / 2 skipped. Version 0.7.0.

## Phase 25 — V0.8.0 financial_stochastics: option pricing, Greeks, stochastic/local vol, rate models, risk, credit, portfolio (50 names)

The eleventh module: `stochpylib.financial_stochastics`, seven submodules,
50 public names, no new runtime dependencies.

- **`option_pricing.py`:** `BlackScholes` (closed form + Greeks properties
  matching the vault quickstart), `BlackScholes_American` (Barone-Adesi-
  Whaley quadratic approximation, falling back to a 1000-step `BinomialTree`
  on request), `BinomialTree`/`TrinomialTree` (CRR / Kamrad-Ritchken
  lattices, European or American), `MonteCarloOptionPricing` (plain,
  antithetic, control-variate, and Sobol-QMC estimators; path-dependent and
  arithmetic/geometric Asian payoffs), `LongstaffSchwartz` (American
  least-squares Monte Carlo, Laguerre or monomial basis, ITM-only
  regression), `FourierOptionPricing` (Carr-Madan or Fang-Oosterlee COS
  inversion of any characteristic function, reusing `levy_processes`'s
  `carr_madan_call`).
- **`greeks.py`:** closed-form `Delta`/`Gamma`/`Vega`/`Theta`/`Rho`/`Vanna`/
  `Volga`; `Greeks_FD` (central finite differences around any pricer, not
  just closed-form ones — works on `BinomialTree`, MC point estimates, etc.);
  `Greeks_MC` (pathwise, likelihood-ratio/score-function, and common-random-
  number bump estimators, each returning `MCResult`s).
- **`stochastic_vol.py`:** `HestonModel` (Albrecher "little trap"
  characteristic function with an analytic `xi->0` fallback, QE or
  full-truncation-Euler simulation, Carr-Madan pricing, bounded
  least-squares calibration), `SABRModel` (Hagan 2002 implied vol),
  `RoughHeston` (El Euch-Rosenbaum fractional Riccati equation via a
  fractional Adams predictor-corrector scheme), `RoughBergomi` (exact joint-
  Gaussian Cholesky simulation of the Riemann-Liouville fBM), `LocalVol`
  (Crank-Nicolson PDE or Monte Carlo), `Dupire` (Gatheral's total-implied-
  variance local-vol formula), `LVSV` (local-stochastic vol with a
  histogram-binned leverage-function particle calibration), `VarianceSwap`
  (Heston closed form, static-replication from a vol surface, or realized
  Monte Carlo).
- **`rate_models.py`:** `VasicekModel`, `CIRProcess` (exact noncentral-
  chi-square transition simulation, Feller-condition check), `HullWhiteModel`
  and `HoLeeModel` (constant-theta mode delegating to closed forms, or a
  curve-fitted mode reproducing an arbitrary input discount curve exactly),
  `G2ppModel` (Brigo-Mercurio two-factor Gaussian, exact bivariate-OU
  simulation), `BlackKarasinski` (log-OU short rate, Monte Carlo-only
  pricing — no closed form), `LMM` (spot-measure log-Euler LIBOR Market
  Model with a proper reset-timing convention: forward `F_j` stays
  stochastic until its own fixing, then freezes), `HJM` (Gaussian one-factor
  Musiela-parametrization simulation, exactly reproducing the input forward
  curve by no-arbitrage construction).
- **`risk.py`:** `HistoricalVaR` (optional age-weighting, overlapping-sum
  multi-period horizons, KDE-based standard error), `ParametricVaR`
  (normal/Student-t/Cornish-Fisher/EWMA/GARCH — the GARCH path reuses
  `timeseries.GARCH`), `ValueAtRisk` (facade + Kupiec POF backtest),
  `ExpectedShortfall`, `ConditionalVaR` (+ Rockafellar-Uryasev CVaR
  portfolio optimization via `scipy.optimize.linprog`), `StressTest`,
  `ScenarioAnalysis` (historical, multivariate-normal Monte Carlo, or a
  custom sampler).
- **`credit.py`:** `DefaultIntensity` (piecewise-constant hazard curve),
  `CDSPricing` (premium/protection legs with accrual, par-spread solving,
  sequential bootstrapping), `MertonCreditModel` (structural PD,
  distance-to-default, equity-implied calibration via `scipy.optimize.fsolve`),
  `CreditMigration` (cohort-MLE transition-matrix fit, generator via a
  numpy-only eigendecomposition matrix logarithm with IRW regularization),
  `CreditRiskModel` (independent-default portfolio loss, Vasicek
  single-factor quantile), `CopulaCreditModel` (Gaussian/Student-t
  copula-dependent defaults, reusing `copulas.elliptical` by setting
  `correlation_`/`df_` directly — those classes have no constructor path for
  a fixed correlation matrix).
- **`portfolio.py`:** `CovarianceEstimation` (sample/EWMA/Ledoit-Wolf 2004
  shrinkage), `MeanVariance` (closed-form unconstrained, SLSQP long-only),
  `PortfolioOptimization` (fluent `.fit()`/`.optimize()` over
  max-Sharpe/min-variance/target-return/max-utility objectives),
  `BlackLitterman` (He-Litterman posterior), `RiskParity` (Spinu 2013 convex
  cyclical coordinate descent).
- **Ten bugs found and fixed while writing `tests/financial_stochastics/tests.py`
  (Probleme.md #53-#62 — nine library bugs plus the pre-existing `ci.yml`
  `runs-on` hardcoding that silently skipped real Windows coverage):** the
  COS Fourier method's truncation range used the raw second moment instead
  of the variance, pricing a K=100 call at `2.7e19` instead of `10.45`;
  `RoughHeston`'s characteristic function integrated the Riccati equation's
  *derivative* instead of its solution, so `H=0.5` never converged to exact
  Heston; `SABRModel`'s `z/x(z)` skew factor was inverted, flipping the
  smile's skew direction for any nonzero `rho`; `HullWhiteModel`'s
  constant-theta zero-coupon-bond price fed the raw Hull-White drift
  constant into the Vasicek delegation instead of dividing by `kappa`;
  `CreditMigration.generator()`'s regularization clipped the diagonal along
  with the negative off-diagonals it was meant to fix, corrupting every
  generator; `RiskParity.optimize()` renormalized weights every
  coordinate-descent sweep, preventing convergence to equal risk
  contributions; the Ledoit-Wolf shrinkage estimator was missing a factor of
  `n` (verified against `sklearn.covariance.ledoit_wolf`), saturating
  shrinkage at 1.0 almost regardless of sample size; Cornish-Fisher expected
  shortfall integrated the wrong tail, returning a negative ES;
  `ScenarioAnalysis` crashed on any pricer taking a non-stochastic parameter
  alongside a stochastic one. All ten are independently reproduced and
  regression-tested.
- **`tests/financial_stochastics/tests.py` (110 tests, collected via
  parametrization):** closed-form cross-checks (Black-Scholes, put-call
  parity, Cornish-Fisher, Merton PD) against hand-derived formulas and
  `scipy.stats`/`scipy.linalg`; lattice-to-closed-form convergence; Monte
  Carlo within a stated number of standard errors of its closed-form or
  semi-analytic counterpart throughout (antithetic/control-variate/QMC
  variance-reduction ordering checked directly); limiting-case reductions
  (Heston `xi->0` to Black-Scholes, rough Heston `H=0.5` to classical
  Heston, G2++ `eta->0` to Hull-White, Hull-White constant-theta to
  Vasicek, LVSV `xi=0` to LocalVol); an exact CDS bootstrap round-trip; a
  full wiring/quickstart/reproducibility section. Manual debug session
  (AGENTS.md 5.2) priced a call three ways, compared Heston semi-analytic
  vs QE Monte Carlo, printed a SABR smile, compared three rate-curve
  parametrizations, computed VaR/ES on synthetic Student-t returns,
  bootstrapped a CDS curve, and compared Longstaff-Schwartz against a
  binomial American put — all before the automated suite was finalized.
- **`selftest.py` extended 145 -> 152 checks:** `financial_stochastics`
  CONFORM spot check plus six `FIN:` checks (Black-Scholes reference value
  and put-call parity, binomial-to-Black-Scholes convergence, Heston
  `xi->0` vs Black-Scholes, Hull-White-vs-Vasicek ZCB agreement, RiskParity
  equal risk contributions).
- **`spl demo financial_stochastics`** added to `cli_demo.py`; `cli.py`
  gained module-overview blurbs for both `financial_stochastics` and the
  previously-undocumented `levy_processes`, and its stale roadmap epilog
  (still listing both as "planned") was corrected.
- **CI (`ci.yml`):** fixed the `runs-on: ubuntu-latest` hardcoding that made
  the `windows-latest` matrix cell run on Ubuntu the whole time
  (Probleme.md #62); added `finance-smoke` (fast-feedback subset run) and
  `install-smoke` (wheel-build verification that the new subpackage actually
  ships) jobs. `publish.yml`'s wheel smoke step now also runs
  `spl demo financial_stochastics`.
- **Docs synced:** README badges/status table/architecture diagrams/roadmap,
  `stochpylib/README.md` module table, `development/architecture.md` module
  map + diagrams, `development/infrastructure.md`, `development/
  project_structure.md`, `development/Development.md`, `CONTRIBUTING.md`,
  `AGENTS.md`, `tests/README.md`, `stochpylib/financial_stochastics/README.md`
  (new), `development/Implementation-Checklist.md` (50/50, progress line to
  400/794), `development/Probleme.md` (#53-#62). Version bumped to 0.8.0 in
  both `pyproject.toml` and `stochpylib/__init__.py`.

Suite: 729 collected - 727 passed / 2 skipped. Version 0.8.0.

## Phase 26 — V0.9.0 statistics: descriptive, estimation, hypothesis tests, regression, multivariate (48 names)

The twelfth module: `stochpylib.statistics`, five submodules, 48 public
names, no new runtime dependencies.

- **`descriptive.py`:** `mean`/`median`/`mode`/`variance`/`std` (weighted and
  trimmed variants), `quantile` (all nine Hyndman-Fan interpolation types),
  `iqr`, `skewness`/`kurtosis` (biased and bias-adjusted), `covariance`,
  `correlation` (Pearson/Spearman/Kendall tau-b, tie-corrected, with a
  significance test), `describe`.
- **`estimation.py`:** `MLE` (any `Distribution` subclass or a
  `loglik(params, data)` callable, Hessian-based standard errors), `MOM`
  (closed forms for 8 named distributions plus a least-squares fallback),
  `bayesian_estimator` (normal-normal/beta-binomial/gamma-Poisson/
  gamma-exponential conjugate families, or a generic grid posterior),
  `confidence_interval` (mean/proportion — wald/wilson/agresti-coull/
  clopper-pearson/variance/median/mean-difference/correlation),
  `bootstrap_ci` (percentile/basic/normal/BCa with jackknife acceleration),
  `jackknife`, `delta_method`, `profile_likelihood` (Wilks' theorem).
- **`hypothesis.py`:** `z_test`/`t_test`/`chi2_test`/`f_test` (variance-ratio
  or nested-model), `ANOVA` (one-way, Welch, or two-way Type-II sums of
  squares via nested-model comparison — coding-independent), `MANOVA`
  (Wilks/Pillai/Hotelling-Lawley/Roy with their SAS-style F-approximations),
  `mann_whitney`/`wilcoxon` (exact null distributions via dynamic
  programming over Python big integers, or tie-corrected asymptotic
  normal), `ks_test`, `shapiro_wilk` (a from-scratch reimplementation of
  Royston 1992's Algorithm AS R94), `levene`/`bartlett`, `tukey_hsd` (a
  from-scratch studentized-range CDF/quantile via log-space Gauss-Legendre
  double quadrature), `bonferroni` (bonferroni/holm/sidak/holm-sidak/
  fdr_bh).
- **`regression.py`:** `linear_regression` (OLS/WLS, nonrobust or HC0-HC3
  sandwich standard errors), `glm` (IRLS; gaussian/binomial/poisson/gamma/
  inverse_gaussian/negative_binomial families x 8 links), `logistic_regression`/
  `poisson_regression` (GLM facades), `ridge` (closed-form SVD, optional
  GCV grid search), `lasso`/`elastic_net` (cyclic coordinate descent,
  warm-started regularization paths), `quantile_regression` (exact linear
  program + Koenker-Bassett kernel-sandwich standard errors).
- **`multivariate.py`:** `PCA`, `factor_analysis` (maximum likelihood via
  profiled Jöreskog concentration, or iterated principal axis; varimax/
  quartimax rotation), `canonical_correlation` (with Wilks
  sequential-dimensionality tests), `discriminant_analysis` (LDA/QDA by
  direct Gaussian Bayes classification), `cluster_analysis` (k-means++ +
  Lloyd, or agglomerative Lance-Williams linkage), `MDS` (classical
  Torgerson eigendecomposition, or SMACOF stress majorization — metric or
  non-metric via isotonic regression).
- **Seven bugs found and fixed while writing `tests/statistics/tests.py`
  and the manual debug session (Probleme.md #64-#69, plus infrastructure
  bug #63):** `tests/` lacked an `__init__.py`, so pytest named
  `tests/statistics/tests.py`'s package after its bare directory name,
  clobbering the stdlib `statistics` module for the whole session; `glm`'s
  IRLS took the reciprocal of an already-correct link derivative, silently
  converging to the wrong coefficients past one iteration; `_mu_bounds`
  clipped the Gaussian family's fitted mean to be positive, breaking
  identity-link IRLS; family bounds and link-domain bounds were conflated,
  breaking Gaussian-with-log-link; OLS/GLM AIC/BIC over-counted parameters
  by one relative to `statsmodels`' convention; the Gaussian+identity GLM
  log-likelihood used the SE-purpose Pearson dispersion instead of
  `statsmodels`' concentrated (n-denominator) variance; the ML
  factor-analysis discrepancy function included a spurious term for the
  already-exactly-absorbed top eigenvalues. All seven are independently
  reproduced and regression-tested.
- **`tests/statistics/tests.py` (118 tests, 15s):** closed-form and
  independent-oracle cross-checks against `scipy.stats` (descriptive
  stats, all rank tests, Shapiro-Wilk, Levene/Bartlett, Tukey HSD,
  KS) and `statsmodels` (z/proportion tests, Welch ANOVA, two-way
  Type-II ANOVA, MANOVA, OLS/WLS/robust-SE/logistic/Poisson/GLM
  family-link grid/quantile regression, factor analysis, canonical
  correlation) and `scipy.cluster` (hierarchical linkage, k-means); a
  from-scratch studentized-range CDF verified against
  `scipy.stats.studentized_range` to ~1e-11; bootstrap/median confidence
  interval coverage checked over 150-200 seeded replicates against the
  binomial standard error; a full wiring/quickstart/reproducibility/
  stdlib-non-shadowing section. Manual debug session (AGENTS.md §5.2) ran
  a realistic treatment/control analysis end to end — normality check to
  pick a test, ANOVA + Tukey + Holm, HC3-robust OLS, logistic regression,
  a lasso path, PCA + k-means, a BCa bootstrap ratio CI, Gamma MLE, and
  Bayesian conjugate updating — before the automated suite was finalized.
- **`selftest.py` extended 152 -> 160 checks:** `statistics` CONFORM spot
  check plus seven `STAT:` checks (t-test type-I control, OLS slope
  recovery, ANOVA power, PCA variance ordering, bootstrap CI coverage,
  Holm monotonicity, studentized-range boundary value).
- **`spl demo statistics`** added to `cli_demo.py`; `cli.py` gained a
  module-overview blurb and its roadmap epilog dropped `statistics` from
  the planned list.
- **CI (`ci.yml`):** added `stats-smoke` (fast-feedback subset run,
  mirroring `finance-smoke`) and extended `install-smoke`'s wheel
  assertion to cover `statistics.__all__` and the stdlib-non-shadowing
  guarantee. `publish.yml`'s wheel smoke step now also runs
  `spl demo statistics`.
- **Docs synced:** README badges/status table/architecture diagrams/
  roadmap, `stochpylib/README.md` module table, `development/
  architecture.md` module map + diagrams, `development/infrastructure.md`,
  `development/project_structure.md`, `development/Development.md`,
  `development/README.md` (also corrected a pre-existing stale 350/794
  count), `CONTRIBUTING.md`, `AGENTS.md`, `tests/README.md`,
  `stochpylib/statistics/README.md` (new), `development/
  Implementation-Checklist.md` (48/48, progress line to 448/794),
  `development/Probleme.md` (#63-#69). Version bumped to 0.9.0 in both
  `pyproject.toml` and `stochpylib/__init__.py`.

Suite: 850 collected - 848 passed / 2 skipped. Version 0.9.0.

## Phase 27 — V0.10.0 random_matrix (23 names), per-module CI smoke jobs, end-to-end API sweeps for every module

The thirteenth module plus the testing infrastructure the V0.10.0 brief asked for.

- **`stochpylib.random_matrix` (23/23 spec names, four submodules):** `ensembles.py`
  (GOE/GUE/GSE with a quaternion self-dual GSE, `WignerMatrix` over any library
  distribution, `WishartMatrix`/`InverseWishart` sampled through
  `distributions.Wishart`/`InverseWishart`, `CUE`, and `MuresanMatrix` — a name the spec
  lists without defining, implemented as the missing Ginibre-type i.i.d. ensemble with the
  circular law); `empirical_spectra.py` (`WignerSemicircle`, `MarchenkoPastur` with the
  `gamma > 1` atom, and `TracyWidomDistribution` for beta = 1, 2, 4 from the Hastings-McLeod
  Painleve II solution, all three full `Distribution` subclasses; Dumitriu-Edelman
  `BetaEnsemble` for any beta and the MANOVA `JacobiEnsemble` with the Wachter law);
  `random_rotations.py` (`HaarMeasure` on O(n)/U(n)/Sp(n) via Mezzadri's QR and a
  quaternionic Gram-Schmidt); `statistics.py` (`EigenvalueSpacing` with the unfolding-free
  mean gap ratio, `LevelRepulsion` by maximum likelihood in the generalized-surmise family,
  `EigenvalueDistribution`, `LargestEigenvalue` with Tracy-Widom scaling — Ramirez-Rider-Virag
  for the Hermitian ensembles, Johnstone-Ma for Wishart — `BulkSpectrum` with the Dyson-Mehta
  number variance, `SpectralEdge` with Edelman's hard-edge law).
- **Tracy-Widom conventions pinned numerically:** beta = 4 tables use the classical
  `F4(s / sqrt 2)` convention (mean -2.3069), and the beta-ensemble edge variable needs the
  factor `2^(1/6)` to reach it — verified to KS p = 0.37 on 1500 tridiagonal draws before
  being written into `LargestEigenvalue`.
- **`tests/random_matrix/tests.py` (71 tests):** closed-form moments (Catalan/Narayana
  numbers), published Tracy-Widom moments and cdf values, `scipy.stats.ortho_group` /
  `unitary_group` Haar oracles, semicircle universality across three entry laws, `<r>` for
  GOE/GUE/GSE/Poisson, Tracy-Widom edge KS tests, Edelman hard edge, the full distribution
  contract on the three limit laws.
- **End-to-end API sweeps (`tests/<module>/e2e.py`, all 13 modules, 486 exercises):** one
  realistic exercise per public name, run as its own pytest case, with a guard that fails the
  moment a name ships without one (`pyproject.toml` now collects `e2e.py`). Writing them
  surfaced **ten shipped bugs** (Probleme #71-#80): `VonMises.fit` never returning, six
  survival fitters without `predict`, a vine `NameError`, `KernelComposition` rejecting
  weights, `SpectralMixtureKernel(dimension=)`, stable increments crashing on NumPy >= 2.5,
  Hull-White/Ho-Lee unreachable `fit`, `VECM.forecast`, a duplicated intercept in the
  switching-AR models, and the unimplemented EKF/UKF smoothers — all fixed with regression
  tests in the modules' own suites. The new wheel check then caught an eleventh (#81): three
  `queueing` spec names (`ErlangBFormula`/`ErlangCFormula`/`EngsetFormula`) had never been
  exported and the conformance test had skipped `queueing` and `information_theory`; and
  running the sweeps ahead of the oracle suites exposed a twelfth (#82): the BB1/BB7
  Kendall-tau curve cache ignored `delta`, so their fits inverted tau on the wrong curve and
  `kendall_tau()` went stale process-wide after any fit.
- **CI restructured (`ci.yml`):** `finance-smoke`/`stats-smoke` replaced by a
  `smoke (<module>)` matrix job for every module (oracle suite + e2e sweep + `spl demo`),
  `fail-fast: false` on both matrices so one red cell no longer cancels the rest, a
  `cross-suite` job for library/docs/cli, and an `install-smoke` that verifies every spec
  name of every module against the built wheel. `tests/docs` now asserts the matrix equals
  `stochpylib.__all__` and that every module has both `tests.py` and `e2e.py`;
  `publish.yml` runs every module demo against the wheel.
- **CLI/selftest:** `spl demo random_matrix`, `--help` blurb, roadmap epilog; `spl --test`
  160 -> 168 checks (conformance + seven `RMT:` checks).
- **Docs synced:** README (badges, status table, diagrams, roadmap, test counts), package/
  tests/development docs, `stochpylib/random_matrix/README.md` (new), checklist 471/794,
  vault status. Version 0.10.0.

Suite: 1449 collected - 1447 passed / 2 skipped. Version 0.10.0.

## Phase 28 — V0.11.0 advanced_mcmc: samplers from Metropolis to NUTS/RMHMC, slice/tempering/SMC/particle/transdimensional methods, diagnostics, variational inference (35 names)

The fourteenth module: state-of-the-art MCMC and variational inference, natively on
numpy/scipy.special only, across six submodules plus a shared `LogDensity`/`MCMCSampler`
base (`_base.py`).

- **`standard.py`:** `MetropolisHastings` (random-walk or fully general/asymmetric
  proposal with `proposal_log_density` correction), `IndependenceSampler`, `GibbsSampler`
  (exact conditionals or Metropolis-within-Gibbs, systematic/random scan), `AdaptiveMetropolis`
  (Haario, Saksman & Tamminen 2001), `RobustAdaptiveMetropolis` (Vihola 2012 Cholesky-factor
  adaptation).
- **`gradient_based.py`:** `HamiltonianMonteCarlo` (leapfrog, dual-averaging step size,
  Stan-style windowed diagonal mass adaptation), `NoUTurnSampler` (multinomial trajectory
  sampling with biased progressive sampling, the classic position-momentum stopping
  criterion), `MALA`, `MMALA` (simplified manifold MALA, SoftAbs-regularised metric
  default), `RiemannianHMC` (generalized leapfrog via fixed-point iteration), `NeutraHMC`
  (fits a `NormalizingFlows` approximation and runs NUTS/HMC in its base space — proven
  exact regardless of flow fit quality, since the flow is only a reparameterization).
- **`slice_sampling.py`:** `Stepping`/`Doubling` (Neal 2003 interval strategies),
  `SliceSampling` (coordinate-wise with shrinkage), `EllipticalSliceSampling` (Murray,
  Adams & MacKay 2010, exact for a Gaussian prior), `Polar_Slice` (Gibbsian polar slice
  sampler, Schar-Habeck-Rudolf 2023).
- **`advanced.py`:** `ReplicaExchange` (general population-MCMC engine) and
  `ParallelTempering` (tempered-ladder specialization with a pilot-phase ladder
  adaptation toward equal swap rates); `SequentialMonteCarlo` (adaptive-tempering SMC
  with bisection-chosen annealing schedule, systematic resampling, RW-MH rejuvenation,
  and marginal-likelihood estimation); `ParticleMCMC` (particle marginal MH via
  `timeseries.ParticleFilter`); `ReversibleJumpMCMC` (Green 1995, default
  Gaussian-auxiliary dimension-matching move) and `TransdimensionalMCMC` (Carlin & Chib
  1995 product-space sampler with fitted Gaussian pseudo-priors).
- **`diagnostics.py`:** `Rhat` (split/rank/classic), `ESS` (bulk/tail/mean/sd, Stan-style
  Geyer-paired estimator), `GelmanRubin`, `PSRF` (Brooks & Gelman multivariate),
  `geweke_test`, `raftery_lewis` (+ `RafteryLewisResult`), `autocorr_time`
  (Sokal/Geyer), `TraceAnalysis` (bundles all diagnostics per parameter).
- **`variational.py`:** `MeanFieldVI`, `ADVI` (mean-field or full-rank, with
  unconstraining bound transforms), `BlackBoxVI` (score-function gradients, no target
  gradient needed), `NormalizingFlows` (planar flows, Rezende & Mohamed 2015, with
  hand-derived reverse-mode gradients — no autodiff anywhere in the stack), `SteinVI`
  (Stein variational gradient descent).
- **Numerical decisions worth recording:** NUTS uses the classic Hoffman-Gelman position-
  momentum stopping criterion (an earlier attempt at a `rho`-accumulated "generalized"
  criterion terminated trees almost immediately — the classic criterion is simpler and
  provably correct here); momentum must be sampled as `p ~ N(0, mass)` where `mass =
  1/inv_mass` (the adapted quantity), not `inv_mass` directly — this bug caused a runaway
  mass-adaptation collapse across warmup windows before being caught; `NormalizingFlows`
  needed weight decay, per-sample and per-step gradient clipping, a hard cap on `||u||`,
  and an invertibility safety margin (`denom >= margin`, not just `>= 0`) to train
  stably — even plain small-step gradient ascent on the raw ELBO diverges to NaN without
  them, a genuine (documented) planar-flow pathology, not an optimizer artifact; its
  forward/backward pass is vectorized over the whole Monte Carlo batch for a ~10-50x
  speedup, verified to match the original per-sample loop to floating-point precision.
- **Trans-dimensional correctness note:** `ReversibleJumpMCMC`/`TransdimensionalMCMC`
  compare densities across models of different dimension, so (unlike single-model MCMC)
  every `log_posteriors[k]` must include *all* normalizing constants — an early manual
  test with unnormalized densities gave a wildly wrong Bayes factor (P(M2) off by an
  order of magnitude) purely from this omission, not a sampler bug; fixed in the test
  and called out prominently in the module README.
- **`tests/advanced_mcmc/tests.py` (48 tests):** closed-form posteriors (conjugate
  Gaussian models, a hand-written Kalman-filter oracle for `ParticleMCMC`, conjugate
  linear-regression Bayes factors for the two trans-dimensional samplers), `scipy.stats`
  KS tests, and independent hand-derived diagnostic formulas, plus exact finite-difference
  gradient checks for the planar flow (both the per-sample and batched code paths).
- **Every Essential-Tasks.md wrap-up item completed:** `tests/advanced_mcmc/e2e.py` (39
  exercises, one per public name); `selftest.py` extended 168 -> 177 checks (`MCMC:`
  block); `stochpylib/advanced_mcmc/README.md` written; `development/CHANGELOG.md`,
  `Probleme.md`, `Implementation-Checklist.md` (35/35, progress line 506/794),
  `architecture.md`, `infrastructure.md` updated; root `README.md` (badges, status
  table, Known Limitations, architecture diagrams, roadmap, CLI reference counts,
  demo prose), `stochpylib/README.md`, `tests/README.md`, `CONTRIBUTING.md`,
  `AGENTS.md`, `development/{Development,README,project_structure}.md` all synced;
  `tests/docs/tests.py` and `tests/library/tests.py` updated (spec-name conformance,
  wiring, a cross-module integration test sampling a library `Gamma` distribution
  through `SliceSampling`); `spl demo advanced_mcmc` added to `cli_demo.py`; `ci.yml`
  `module-smoke` matrix extended.
- **Two bugs found and fixed while testing (see `Probleme.md` #83-84):**
  `NoUTurnSampler._step` incremented the old single-chain divergence counter but never
  the newer per-chain list used to compute the public `divergences_` attribute, so NUTS
  always reported zero divergences regardless of how unstable the trajectory actually
  was; and the rank-normalization transform (`Rhat`/`ESS` "rank"/"bulk" variants) used
  `N - 3/4` instead of Blom's `N + 1/4` in its denominator, producing values fractionally
  above 1 for the extreme ranks and `NaN` from `ndtri` — surfaced as `Rhat(method="rank")
  == inf` on perfectly good iid chains.

Suite: 1540 collected - 1538 passed / 2 skipped. Version 0.11.0.

## Phase 29 — V0.12.0 numerical_methods: quadrature, ODE/SDE solvers, linear algebra, root finding, interpolation, PDE tools (38 names)

The fifteenth module: the numerical analysis backbone the rest of the library has been
building on top of implicitly (finite differences, quadrature, matrix decompositions),
now a first-class module with native implementations across six submodules.

- **`integration.py`:** `GaussLegendre`/`GaussHermite`/`GaussChebyshev` (all via the
  Golub-Welsch symmetric-tridiagonal-eigenvalue construction, not looked up from
  `numpy.polynomial`); `AdaptiveQuadrature` (QUADPACK-style adaptive Gauss-Kronrod 7-15
  with a heap of worst-error intervals, or adaptive Simpson; infinite/half-infinite
  limits via a change of variables); `NumericalIntegration` (dispatcher over
  adaptive/trapezoid/simpson/romberg/gauss_legendre/gauss_kronrod/monte_carlo);
  `MonteCarloIntegration` (a `numerical_methods`-native facade delegating sampling to
  `stochpylib.montecarlo.crude_mc`/`quasi_montecarlo` — a deliberate same-name-different-
  module pairing, precedented by `random_matrix.InverseWishart` vs
  `distributions.InverseWishart`); `CubatureRule` (tensor-product or Smolyak sparse
  grids over Gauss-Legendre/Gauss-Hermite/Clenshaw-Curtis 1-D rules).
- **`ode_sde.py`:** `EulerMethod`, `RungeKutta4` (classic or a custom explicit Butcher
  tableau), `DormandPrince` (embedded RK5(4)7FM with FSAL, Hairer-Norsett-Wanner initial
  step heuristic, classic PI-free step control, cubic-Hermite dense output — not DP's
  own 5th-order interpolant), `Adams_Bashforth` (explicit orders 1-5, RK4-started,
  optional PECE Adams-Moulton corrector), `BDF` (implicit orders 1-5, Newton-corrected,
  order ramp on startup — order 1 is backward Euler); `Euler_Maruyama_SDE` (diagonal or
  general/correlated noise) and `Milstein_SDE` (diagonal noise, numeric or analytic
  diffusion derivative) SDE path solvers, both taking a shared `brownian=` array so
  strong-convergence studies compare schemes on the identical driving path.
- **`linear_algebra.py`:** `MatrixExponential` (Higham 2005 degree-13 Pade with scaling-
  and-squaring, plus eigendecomposition/Taylor fallbacks), `MatrixLogarithm` (inverse
  scaling-and-squaring via Denman-Beavers matrix square roots and a Gauss-Legendre
  partial-fraction Pade evaluation of `log(I+X)`), `CholeskyDecomp` (native
  Cholesky-Banachiewicz with optional escalating jitter), `EigenDecomp` (cyclic Jacobi
  for symmetric matrices, Hessenberg reduction + shifted QR with inverse-iteration
  eigenvectors for general matrices), `SVD` (one-sided Hestenes Jacobi), `QRDecomp`
  (Householder, modified Gram-Schmidt, or Givens), `Schur` (Francis implicit
  double-shift QR for the real form, single complex-Wilkinson-shift QR for the complex
  form).
- **`root_solve.py`:** `Bisection`, `Brent` (Brent-Dekker: inverse quadratic
  interpolation / secant with a bisection safety net), `Secant`, `NewtonRaphson`
  (scalar or vector, analytic or finite-difference derivative/Jacobian, backtracking
  damped), `FixedPoint` (plain, Aitken delta-squared, or Steffensen acceleration),
  `RootFinding` (dispatcher + `find_bracket` expanding-interval search + `all_roots`
  grid-scan-and-refine).
- **`interpolation.py`:** `SplineInterpolation` (natural/clamped/not-a-knot cubic
  spline, second-derivative formulation), `CubicHermite` (Fritsch-Carlson PCHIP slopes
  by default, or given slopes), `BarycentricLagrange` (+ Chebyshev-point construction
  for Runge-phenomenon-free interpolation), `Chebyshev` (series via the Clenshaw
  recurrence; derivative/integral by the standard coefficient recurrences; roots via the
  colleague matrix), `NURBS` (Cox-de Boor basis recursion; `circle()`/`interpolate()`
  constructors), `Interpolation` (facade: linear/nearest/polynomial/cubic/pchip/
  spline_natural).
- **`pde.py`:** `Mesh` (1-D interval / 2-D triangulated rectangle), `FiniteDifference`
  (Fornberg arbitrary-stencil weights, 1-D/2-D Poisson via Thomas/conjugate-gradient
  solves, 1-D heat with explicit/implicit/Crank-Nicolson theta-schemes, a Black-Scholes
  PDE pricer with an American early-exercise projection, 1-D advection with
  upwind/Lax-Wendroff), `FiniteElement` (1-D P1/P2 Galerkin, 2-D P1 on triangles),
  `FEniCS_Interface` (solves natively via `FiniteElement` by default; `.export_mesh()`
  writes DOLFIN-XML or XDMF; `.to_fenics()` lazily hands the mesh to a real
  dolfinx/dolfin install when present, else a clear `ImportError` — FEniCS is never a
  runtime dependency), `BoundaryElement` (2-D interior Laplace, constant elements, the
  `-(1/2*pi)*ln(r)` fundamental solution), `SpectralMethod` (Fourier
  derivative/Poisson/heat on periodic domains via FFT, Chebyshev-collocation BVP solver
  via Trefethen's differentiation matrix).
- **Every algorithm is native**; `numpy.linalg` (eigh/eig/qr/svd/solve) is used as an
  explicit `method="numpy"` fast path in the linear-algebra classes, and
  `scipy.integrate/optimize/interpolate/linalg` remain test oracles only, never
  imported inside `stochpylib/`.
- **Five bugs found and fixed while testing (see `Probleme.md` #85-89):**
  `FiniteElement.evaluate()` silently discarded the P2 midpoint DOFs (linear-only
  lookup), erasing all of P2's extra accuracy until a P1-vs-P2 convergence-rate
  comparison caught two identical error curves; the complex-Schur QR step never
  updated the coupling block above a deflated trailing part, so reconstruction broke
  as soon as any eigenvalue had deflated; `Chebyshev.integral()`'s `T_1` coefficient
  used the general recurrence instead of its required special case, giving a definite
  integral off by exactly half the constant term; the DOLFIN-XML mesh exporter wrote
  NumPy 2.x's `repr()` (`"np.float64(0.0)"`) instead of a plain float string, breaking
  the reader round-trip — the same class of NumPy-2.x formatting break this project
  has hit before (Probleme #41-51); and `spl --help`'s column-padding logic used `>`
  instead of `>=`, so a module name of exactly 17 characters (`numerical_methods`
  itself, the first module name to ever hit that exact length) got zero separator
  characters before its summary.
- **`tests/numerical_methods/tests.py` (85 tests):** `scipy.integrate`/`optimize`/
  `interpolate`/`linalg` and `numpy.polynomial` as independent oracles, closed forms,
  step-halving order verification for every ODE/SDE scheme, an SDE strong-convergence
  study (Euler-Maruyama order ~0.5, Milstein order ~1.0, log-log slope fit), the
  Robertson stiff-system benchmark against `scipy.integrate.solve_ivp(method="Radau")`,
  P1/P2 finite-element convergence rates, a Black-Scholes PDE price against the
  library's own closed form and against `BinomialTree`'s American exercise, a
  FEniCS-stub-module injection test for `.to_fenics()`, and cross-module checks against
  `financial_stochastics.BlackScholes` and `levy_processes.Euler_Maruyama`.
- **Every Essential-Tasks.md wrap-up item completed:** `tests/numerical_methods/e2e.py`
  (43 exercises, one per public name); `selftest.py` extended 177 -> 187 checks
  (`NUM:` block); `stochpylib/numerical_methods/README.md` written;
  `development/CHANGELOG.md`, `Probleme.md`, `Implementation-Checklist.md` (38/38,
  progress line 544/794), `architecture.md`, `infrastructure.md` updated; root
  `README.md` (badges, status table, Known Limitations, architecture diagrams,
  roadmap, CLI reference counts, demo prose), `stochpylib/README.md`, `tests/README.md`,
  `CONTRIBUTING.md`, `AGENTS.md`, `development/{Development,README,project_structure}.md`
  all synced; `tests/docs/tests.py` and `tests/library/tests.py` updated (spec-name
  conformance, wiring, a cross-module integration test pricing Black-Scholes through
  both a finite-difference PDE and a Gauss-Hermite risk-neutral expectation); `spl demo
  numerical_methods` added to `cli_demo.py`; `ci.yml` `module-smoke` matrix extended.

Suite: 1673 collected - 1671 passed / 2 skipped. Version 0.12.0.

## Phase 30 — V0.13.0 bayesian: priors/posteriors, conjugate engine, Bayesian models, model selection, posterior approximations (25 names)

- **`bayesian.core`**: `Prior`/`prior()` (a library distribution, a product-prior list,
  `"flat"` improper, or custom logpdf/sampler) and `Likelihood`/`likelihood()` (ten
  built-in exponential families — bernoulli/binomial/poisson/exponential/gamma/normal/
  normal_variance/normal_unknown_var/categorical/mvnormal — or a custom `loglik`).
  `ConjugateFamily`/`conjugate_prior()` implement closed-form update/predictive/evidence
  for all ten families, each cross-checked against `scipy.stats` or a fine numerical
  grid. `posterior()`/`evidence()` default to `method="auto"` (conjugate when the pair
  matches, else a numerical grid for dim <= 2, else Laplace/MCMC), with `"laplace"`,
  `"vi"`, `"importance"`, `"smc"`, and `"mcmc"` (slice/NUTS/adaptive-Metropolis, via
  `advanced_mcmc`) as explicit alternatives. `bayes_update()` is sequential-equals-batch
  conjugate updating; `posterior_predictive()` is closed form when conjugate, else a
  Monte Carlo `EmpiricalPredictive` (a posterior-predictive distribution satisfying the
  full 13-method distribution contract via a Gaussian KDE).
- **`bayesian.computation`**: `LaplacePosterior` (MAP + inverse-negative-Hessian
  covariance), `EP_Posterior` (Gaussian expectation propagation with Gauss-Hermite
  moment matching per site — works for *any* analytically evaluable site, not only
  Gaussian-conjugate ones, verified exact on a linear-Gaussian model and matching a long
  NUTS run on a probit/logistic model), `ImportanceSamplingPosterior` (self-normalized
  IS with Pareto-smoothed weights, a default Laplace-centred multivariate-t proposal).
  `MFVariational` **delegates to `advanced_mcmc.MeanFieldVI`/`ADVI`** instead of
  reimplementing VI, resolving the design-scorecard note that `bayesian.computation`'s
  variational inference was thin relative to `advanced_mcmc.variational` (Ratings.md).
- **`bayesian.selection`**: `AIC`/`BIC` (+ AICc, matched to `statsmodels.OLS` exactly),
  `DIC`, `WAIC`, `LOO_CV` (Pareto-smoothed importance sampling by default — a native
  Zhang-Stephens 2009 generalized-Pareto fit plus the Vehtari/Simpson/Gelman/Yao/Gabry
  smoothing recipe — cross-checked against exact leave-one-out refits of a conjugate
  model), `TICfit` (the sandwich-covariance generalization of AIC), and `bayes_factor`
  (from log-evidences, `(prior, likelihood)` tuples, or `method="bic"` on two fitted
  models, statsmodels results included) — all return the shared `ICResult`.
- **`bayesian.models`**: `BayesianLinear` (conjugate Normal-Inverse-Gamma regression,
  matching OLS exactly under a near-flat prior, its log-evidence formula verified
  against a fine-grid numerical integral), `BayesianLogistic` (`method=`
  laplace/ep/mcmc/vi, MAP matching `statsmodels.Logit` to 1e-4), `NaiveBayes`
  (gaussian/bernoulli/multinomial posterior-mean parameters), `HierarchicalModel`
  (normal-normal Gibbs, known-per-group or shared-unknown-sigma modes, half-Cauchy/
  inverse-gamma/fixed tau priors), `MixtureModel` (collapsed-Gibbs Bayesian mixtures,
  gaussian NIG/NIW or 1-D poisson-gamma), `BayesianNetwork` (discrete, exact variable
  elimination, ancestral sampling, Dirichlet-smoothed MLE fit, BDeu structure score),
  `DirichletProcess` (stick-breaking, CRP, and a Neal-2000-Algorithm-3 collapsed-Gibbs
  DP mixture).
- **Two real numerical bugs surfaced and fixed in existing distributions** while testing
  conjugate posteriors with realistic sample sizes (Probleme.md #90-92): `NegBinomial.pmf`,
  `Gamma.pdf`, and `BetaBinomial.pmf` all overflowed to `NaN` for the large shape/rate
  parameters a conjugate posterior naturally produces after a few hundred observations —
  fixed by moving each to log-space (`gammaln`/`betaln`/`xlog1py`) before exponentiating.
- **`tests/bayesian/tests.py` (~55 test functions, several parametrized):** `scipy.stats`
  (beta/gamma/nbinom/betabinom/t/invgamma/lomax/multivariate_normal), `statsmodels`
  (OLS/Logit), brute-force enumeration (the sprinkler `BayesianNetwork` example), fine
  numerical grids/1-D quadrature, and long `advanced_mcmc` runs as independent oracles.
- **Every Essential-Tasks.md wrap-up item completed:** `tests/bayesian/e2e.py` (32
  exercises, one per public name); `selftest.py` extended 187 -> 199 checks (`BAYES:`
  block); `stochpylib/bayesian/README.md` written; `development/{CHANGELOG,Probleme,
  Implementation-Checklist,architecture}.md` updated (progress line 569/794); root
  `README.md` (badges, status table, Known Limitations, architecture diagrams, roadmap,
  CLI reference counts, demo prose), `stochpylib/README.md`, `tests/README.md`,
  `CONTRIBUTING.md`, `AGENTS.md`, `development/{Development,README,project_structure}.md`
  all synced; `tests/docs/tests.py` and `tests/library/tests.py` updated (spec-name
  conformance, wiring, a cross-module integration test agreeing with
  `statistics.bayesian_estimator` and an `advanced_mcmc` sampler); `spl demo bayesian`
  added to `cli_demo.py`; `ci.yml` `module-smoke` matrix extended.

Suite: 1758 collected - 1756 passed / 2 skipped. Version 0.13.0.
