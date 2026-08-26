# Problems

Bugs and infrastructure issues found while building `stochpylib`, how they were fixed
(or why they're still open), a severity rating (1 = cosmetic, 10 = critical /
wrong-numbers-silently-shipped), and a status. Ordered by entry number (oldest first);
the numbering is continuous with no gaps. Entry #12 resolves the open remainder of #9,
and #10 documents two bugs that were caught during development and never shipped —
both are kept for the lessons they encode.

**Status legend:**
- 🟢 `fixed` — code changed and verified (by tests against independent oracles,
  statistical assertions with fixed seeds, or direct inspection); nothing pending.
- 🟡 `partial` — a fix shipped but verification is incomplete, or a real known caveat
  remains.
- 🔴 `closed` — no code fix applied: declined/won't-fix, a non-goal, or moot.

Every entry follows: **Problem** (what was wrong) → **Fix** (what changed) →
**Verification** (how it was confirmed). This project's standing verification convention:
library code never wraps `scipy.stats`; instead `scipy.stats` is the *test oracle*, so most
numerical fixes are verified by cross-checks against it plus deterministic statistical
assertions (fixed seeds, tolerances ≥ 3 standard errors).

---

### 1. `independence.py` doctest/test examples were mathematically wrong

**Severity:** 4/10 · **Status:** 🟢 `fixed` (pre-V0.0.1)

**Problem:** The docstring examples for `is_independent()`, `pairwise_independence()` and
`conditional_independence()` used event pairs that happened to be independent on a
4-outcome uniform space while the docstrings/tests asserted dependence (`False`). The
function logic was correct throughout — only the hand-picked example data was wrong.

**Fix:** Replaced the example event sets with genuinely dependent ones
(`event(1, 2)` vs `event(1, 2, 3)`).

**Verification:**
- Originally surfaced as 2 hard doctest failures on first `pytest` run; after the fix the
  examples were re-checked by hand and the suite went green.

---

### 2. `derangement()` used floating-point summation despite the "exact arithmetic" claim

**Severity:** 3/10 · **Status:** 🟢 `fixed` (pre-V0.0.1)

**Problem:** First implementation computed D(n) via the floating-point series
`n! * sum((-1)^k / k!)` then rounded. Float64 precision runs out well before `n!`
overflows, so large inputs would silently produce a wrong integer — contradicting the
module's exact-arithmetic promise.

**Fix:** Rewrote using the exact integer recurrence
`D(n) = (n-1) * (D(n-1) + D(n-2))`, `D(0)=1`, `D(1)=0`.

**Verification:**
- No floating point remains in the function (inspection); the derangement unit tests and
  the module's exact-arithmetic doctests pass for small and large `n`.

---

### 3. Colocated `tests.py` shipped inside the built wheel despite an exclusion rule

**Severity:** 3/10 · **Status:** 🟢 `fixed` (pre-V0.0.1)

**Problem:** `[tool.setuptools.exclude-package-data]` only filters *data* files; regular
`.py` modules are always packaged. The exclusion rule was a silent no-op and
`python -m build` shipped `stochpylib/probability/tests.py` to end users.

**Fix:** Relocated tests out of the package entirely to `tests/<module>/tests.py` at the
repo root, making the problem moot rather than patching around setuptools semantics.

**Verification:**
- Wheel contents inspected before and after relocation (`python -m build` + archive
  listing): no test modules ship anymore; this layout is now the documented convention
  (see `tests/README.md`).

---

### 4. Vault generator scripts referenced an unrelated prior project

**Severity:** 4/10 · **Status:** 🟢 `fixed` (pre-V0.0.1)

**Problem:** `generate_code_graph.py`'s `IGNORE_DIRS` listed another project's vault
folder name, so running it would have walked into this repo's own vault and generated
bogus "code" notes for Markdown files. `regenerate_vault.py` hardcoded a wrong title and
skipped "the vault" by comparing directory *names* instead of resolved paths — which
never matched.

**Fix:** Corrected `IGNORE_DIRS` to the real vault folder name plus build/cache dirs;
fixed the title and switched the skip check to resolved-path comparison.

**Verification:**
- Both scripts ran end-to-end; `Project-Map.md` correctly excludes the vault and carries
  the right title.

---

### 5. `GPareto.pdf` leaked probability below its support

**Severity:** 6/10 · **Status:** 🟢 `fixed` (68c4712)

**Problem:** For `shape > 0` the pdf formula was masked only by `t > 0` (with
`t = 1 + shape*z`), not by support membership `z >= 0`. Points *below* `loc` returned
large positive densities — mass was created outside the support.

**Fix:** Mask extended to `(z >= 0) & (t > 0)`.

**Verification:**
- Caught by the scipy cross-check audit (`pdf(-1)`: ours 4.69 vs reference 0.0); the
  dedicated `test_gpareto_support_mask` regression test now asserts exactly this case,
  and the full scipy comparison passes at rtol 1e-6.

---

### 6. `Rice.pdf` overflowed to NaN for large x, poisoning numeric moments

**Severity:** 5/10 · **Status:** 🟢 `fixed` (68c4712)

**Problem:** The density factored `i0e(y) * exp(y)` to undo the Bessel exponential
scaling. The factors cancel analytically but overflow numerically (`exp(y)` → inf for
y > ~709 while the Gaussian factor underflows), producing NaN that quadrature probes in
generic moment fallbacks picked up — `skewness()` returned NaN despite Rice having
finite moments of all orders.

**Fix:** Pdf computed in log space via `log I0(y) = y + log(i0e(y))`.

**Verification:**
- Matches scipy.stats.rice to machine precision at test points including x = 800
  (previously NaN); interface-contract `skewness()` now agrees with scipy's to 1e-12.

---

### 7. Discrete `ppf` overshot bounded supports and returned the wrong atom

**Severity:** 5/10 · **Status:** 🟢 `fixed` (68c4712)

**Problem:** `_ppf_discrete`'s bracket-expansion loop returned `high` as soon as the
expanding index crossed the support bound — before the binary search could find the true
smallest atom. E.g. `DiscreteUniform(0, 9).ppf(0.9)` returned 9 although `cdf(8) = 0.9`.

**Fix:** Expansion clamps to `high` and falls through to the binary search, tracking the
last known-below bracket.

**Verification:**
- The scipy cross-check suite compares `ppf` at q ∈ {0.2, 0.55, 0.9} for every bounded-
  support discrete class (DiscreteUniform, Binomial, BetaBinomial, Hypergeometric) —
  all match exactly.

---

### 8. `MultivariateDistribution.fit` had a broken instance-method signature

**Severity:** 2/10 · **Status:** 🟢 `fixed` (68c4712)

**Problem:** Declared `def fit(cls, data): raise NotImplementedError` without
`@classmethod`, so instance calls raised a confusing `TypeError` instead of the intended
clear error.

**Fix:** Decorated with `@classmethod`.

**Verification:**
- Latent only (every current multivariate class overrides `fit` properly); confirmed the
  base-class call path now raises `NotImplementedError` and the multivariate convention
  tests still pass.

---

### 9. `StableDistribution.rvs` unusably slow; exact special cases routed through numerics

**Severity:** 4/10 · **Status:** 🟢 `fixed` (2dfc156, corner completed by #12)

**Problem:** Generic inverse-CDF sampling evaluated the Gil-Pelaez numerical CDF once per
brentq iteration per sample (each evaluation a full Fourier-inversion quadrature), making
`rvs(300)` take minutes. alpha=2 (always Gaussian) and alpha=1/beta=0 (Cauchy) were also
needlessly routed through numerics.

**Fix:** alpha=2 delegates exactly to Gaussian closed forms, alpha=1/beta=0 to Cauchy;
all alpha != 1 use a Chambers–Mallows–Leckie sampler whose constants were determined
empirically by matching our closed-form characteristic function at Monte-Carlo noise
level across symmetric and skewed parameter sets.

**Verification:**
- `test_stable_alpha_two_delegates_to_gaussian` and `test_stable_cauchy_special_case`
  assert exact equality with scipy closed forms; the CML sampler cf-match test locks the
  validated constants in (max deviation < 0.02 ≈ MC noise).
- The remaining alpha=1/beta≠0 corner is tracked separately and resolved in #12.

---

### 10. Base-2 digital-net engine: two construction bugs caught by exactness checks

**Severity:** 7/10 · **Status:** 🟢 `fixed` (never shipped — caught pre-release, 2dfc156)

**Problem:** (a) Dimension 1 was routed through the generic direction-number recurrence
for its degree-1 polynomial (x+1); the recurrence doubles even integers there and
corrupts the stream — dimension 1 must be plain van der Corput with m_j = 1 at every
level. (b) Point generation used a single-XOR Gray walk on the natural index counter;
that enumerates points in *Gray-code order*, not natural order (a natural-order increment
flips many bits at once), so dim-1 output did not equal van der Corput in sequence
position.

**Fix:** Row 0 short-circuits to pure van der Corput; generation replaced with direct bit
decomposition over natural indices (`x_i = XOR of V[:, b] over the set bits b of i`),
vectorized.

**Verification:**
- Dim-1 equals van der Corput bitwise over the first 16 points; the canonical 2-D prefix
  ((1/2,1/2), (1/4,3/4), (3/4,1/4)) matches published tables; both locked in by tests.

---

### 11. Base-2 nets were not certified exact (t,m,s)-nets under simple initial values

**Severity:** 2/10 · **Status:** 🟢 `fixed` (unreleased — ships with next tag)

**Problem:** Direction numbers came from GF(2) polynomials with simple canonical odd
initial values rather than published optimized tables. Half-interval balance in
dimensions ≥ 2 could be off by ±1 sample at n = 2^m. Root-caused later: part of the
imbalance belongs to any origin-skipped streaming window, but the underlying net also
wasn't certified.

**Fix:** Embedded the standard Joe–Kuo lineage direction-number table
(`_direction_numbers.py`, 64 dims × 30 columns — the same data scipy.stats.qmc uses).
It was extracted from the oracle via the `x_{2^b} = v_b` identity and verified dyadic
plus bitwise round-trip *before* being embedded. `SobolSequence` uses it by default; new
`generate_block(m)` returns the aligned first-2^m block **including the origin**, which
is exactly balanced in every dimension. Streaming `generate(n)` still skips the origin,
so its windows remain ±1 misaligned by construction (documented). GF(2) machinery
remains as fallback beyond dimension 64 and for custom/Niederreiter nets.

**Verification:**
- `test_generate_block_exact_net_balance`: perfect halves/eighths in every dimension at
  n = 2^8 and 2^10.
- `test_block_matches_scipy_set_and_gray_order`: block set-identical to scipy's and
  equal under the Gray-code position mapping.
- Pre-embedding checks: every column dyadic, column 0 identically 1/2.

---

### 12. StableDistribution alpha=1, beta≠0 sampling was unusably slow

**Severity:** 4/10 · **Status:** 🟢 `fixed` (unreleased — ships with next tag)

**Problem:** The alpha=1 skewed corner fell back to per-sample numerical root finding on
the Gil-Pelaez CDF (~seconds for a handful of draws). Two attempts at adopting a
published closed-form Chambers–Mallows–Leckie variant failed validation against our own
characteristic function (best residual 0.077 vs noise level ~0.003 across twelve
structural variants; shift-fitting proved structural mismatch, not a location offset).

**Fix:** Cached numerical quantile table per parameter set: the exact Gil-Pelaez CDF is
evaluated on a grid inside the numerically reliable window (|x − loc| ≤ 25σ), refined
through monotone PCHIP, with the exact power-law tail asymptotics of alpha=1 stables
(`1-F(x) ~ c(1+β)/(πx)` as x → ∞, symmetric on the left) grafted on beyond it down to
q = 1e-9. Draws are a vectorized table lookup. Warmup ~5 s per parameter set
(class-level cache), afterwards O(1) per draw.

**Verification:**
- Empirical characteristic function of 20k draws matches the closed form within 0.012
  (MC noise level) for β = ±{0.5, 0.7}.
- Central quantile error vs root-refined truth ≤ ~1e-3 · scale (asserted over random q).
- Seeded determinism asserted; cache reuse makes repeat instances instant.

---

### 13. Integrated-model forecasts seeded the recursion with raw levels

**Severity:** 7/10 - **Status:** fixed (unreleased; ships with 0.2.0)

**Problem:** ARIMA.forecast built its recursion history from self._y (raw levels) while
the fitted coefficients lived on the *differenced* series - an ARIMA(1,1,0) on a
0.5-slope trend forecast steps of +14.5 instead of +0.5. MA/ARMA lacked the transformed-
series attribute entirely.

**Fix:** Base class gained _fit_series (the series CSS actually fit); forecast seeding,
innovation alignment (T = len(_fit_series)) and SARIMA's inverse differencing order
(seasonal-undo before regular-undo) all corrected.

**Verification:**
- ARIMA(1,1,0) on a 0.5-slope trend now forecasts slope 0.498 (was 14.5).
- Full statsmodels-oracle suite green.

---

### 14. FIGARCH applied the integration kernel instead of the differencing filter

**Severity:** 5/10 - **Status:** fixed (unreleased; ships with 0.2.0)

**Problem:** FIGARCH.fit built weights via frac_diff_weights(-d), i.e. the (1-B)^-d
integration kernel, so the filtered series was MORE persistent than the input instead of
whitened (lag-1 correlation 1.000 in the smoke test).

**Fix:** Filter uses frac_diff_weights(+d); the inverse kernel is only used when mapping
forecasts back to levels.

**Verification:**
- Round-trip identity: integrate white noise with (1-B)^-0.35, re-filter with
  (1-B)^+0.35 -> lag-1 correlation ~0 (asserted in the suite).

---

### 15. KPSS p-values interpolated against a descending table

**Severity:** 4/10 - **Status:** fixed (unreleased; ships with 0.2.0)

**Problem:** np.interp(stat, cvs, levels) was called with the critical-value array in
descending order; np.interp requires ascending x, so returned p-values were garbage
(white noise reported p=0.01).

**Fix:** Arrays reordered ascending ([CV10%, CV5%, CV1%] against [0.10, 0.05, 0.01]).

**Verification:**
- White noise p >= 0.10; random walk p <= 0.01; trend-stationary with regression='t'
  p >= 0.05. The statistic itself matches statsmodels within the Newey-West lag tolerance.

---

### 16. ADF ignored an explicit max_lag and AIC-searched anyway

**Severity:** 3/10 - **Status:** fixed (unreleased; ships with 0.2.0)

**Problem:** Passing max_lag=2 still ran the AIC selection over lags {0,1,2}, so the
reported statistic could correspond to any smaller lag - breaking exact comparison with
statsmodels' autolag=None semantics.

**Fix:** An explicit max_lag is now the exact augmentation order; AIC selection only
applies for the default None (Schwert cap).

**Verification:**
- Statistic equals statsmodels adfuller(maxlag=2, autolag=None) to 1e-8 on identical data
  (locked in by test_adf_stat_matches_statsmodels_fixed_lag).

---

### 17. ParticleFilter broadcast user log-pdfs into an n x n matrix

**Severity:** 4/10 - **Status:** fixed (unreleased; ships with 0.2.0)

**Problem:** An observation log-pdf returning shape (n, 1) was added directly to the (n,)
weight vector, broadcasting into an (n, n) matrix - nine-million-element state explosions
and crashes for perfectly reasonable user callables.

**Fix:** The observation log-density output is flattened defensively before use.

**Verification:**
- The smoke test uses exactly such an (n, 1) callable; the filter tracks a local level at
  correlation 0.978.

---

### 18. BOCPD changepoint hypothesis reused per-run predictives, pinning P(change) = hazard

**Severity:** 6/10 - **Status:** fixed (unreleased; ships with 0.2.0)

**Problem:** In the Adams-MacKay recursion, the reset hypothesis's predictive density was
computed per old run length (copied from the growth terms) instead of using the NIG prior
predictive. Since the reset term then agreed with every growth term, the posterior
probability of change collapsed to exactly the hazard rate forever - the detector could
never fire.

**Fix:** The reset hypothesis now uses the NIG *prior* predictive (Student-t with the
base hyperparameters); growth terms keep their per-run predictives.

**Verification:**
- On a two-block mean-shift series the posterior probability of change spikes above 0.5
  at the true boundary and the detected-point list contains it.

---

### 19. DWT reconstruction failed for db2: unnormalized taps plus non-transpose synthesis

**Severity:** 5/10 - **Status:** fixed (unreleased; ships with 0.2.0)

**Problem:** Two stacked issues in the wavelet pair. (a) The Daubechies-4 scaling taps
were missing the sqrt(2) normalization (sum h^2 = 2 instead of 1), so the analysis
operator was not orthonormal. (b) The synthesis pass applied the analysis filters
directly instead of their transpose, which happens to work only for symmetric filters
like Haar.

**Fix:** Taps normalized (sum h^2 = 1, verified numerically); synthesis applies the same
filters at the same circulant taps (the transpose of the analysis operator).

**Verification:**
- Perfect reconstruction to 1e-15 for both haar and db2 over random 1024-sample inputs
  at level 4 (test_dwt_idwt_perfect_reconstruction).

---

### 20. ExpectationPropagation does not always converge to informative posteriors

**Severity:** 3/10 · **Status:** 🟡 `partial` (documented experimental caveat)

**Problem:** The damped EP implementation can converge to a degenerate solution for some
datasets: predictive probabilities collapse toward 0.5 instead of separating the classes.
The tilted-moment machinery itself is exact (Gauss-Hermite quadrature of
``p(y|f) N(f | cavity)``), but the fixed-point iteration is not guaranteed to reach an
informative fixed point from its prior initialization.

**Fix:** None applied — the class carries an explicit **experimental** warning and steers
users to :class:`LaplacePropagation` (reliable) or :class:`VariationalInference`
(logit-link alternative). The spec-facing ``GPClassification`` facade defaults to Laplace.

**Verification:**
- Laplace and VI classification accuracy tests (>0.90 / >0.78 on a separated 2-D
  Gaussian mixture) pass; EP remains usable as a smoke-tested engine.

---

### 21. Broken duplicate FITC/VFE copy inside ``inference.py`` crashed on predict

**Severity:** 6/10 · **Status:** 🟢 `fixed`

**Problem:** Besides the canonical implementation in ``sparse.py``, ``inference.py``
carried a second, older copy of ``_SparseRegressionBase``/``FITC``/``VFE``/
``SparseVFE`` whose ``FITC.predict`` called a nonexistent ``self._predict_core`` — any
direct import of the sparse engines from ``inference.py`` raised ``AttributeError`` at
prediction time. The package ``__init__`` imported from ``sparse.py`` only, so the test
suite never touched the broken copy.

**Fix:** Duplicate removed; ``inference.py`` now contains only the three classification
engines, with a module docstring pointing to ``sparse.py`` as the single source of the
sparse-regression engines.

**Verification:**
- New regression test ``test_inference_module_is_classification_only`` asserts the
  sparse names no longer exist there while the classification engines do; full GP suite
  green against the canonical implementations.

---

### 22. NeuralNetworkKernel initially used an incorrect closed form

**Severity:** 4/10 · **Status:** 🟢 `fixed` (caught pre-release during Phase 11 construction)

**Problem:** The first draft of the neural-network covariance did not implement the
Rasmussen & Williams eq. 4.29–4.31 construction correctly (bias-augmented inputs and
the exact arc-sine integral), producing a matrix that failed PSD spot checks at larger
variance/bias combinations.

**Fix:** Rewritten as the standard depth-1 NN kernel:
bias-augment each input with ``sqrt(bias_variance)``, then
``k(x, x') = variance * (2/pi) * asin(2 u'v / sqrt((1+2u'u)(1+2v'v)))``.

**Verification:**
- Symmetry + PSD across the whole kernel zoo (min eigenvalue > -1e-8,
  ``test_kernel_symmetry_and_psd[NN]``); positive definiteness by construction.

---

### 23. Sparse GP posterior used raw inverses of near-singular Kuu — predictions exploded

**Severity:** 7/10 · **Status:** 🟢 `fixed`

**Problem:** The sparse engines (FITC/VFE) computed ``Kuu_inv = np.linalg.inv(Kuu)``
on the *unjittered* inducing-point kernel and inverted the unwhitened posterior
precision ``A = Kuu_inv + Kuf Lam^-1 Kfu^T``. RBF ``Kuu`` becomes numerically singular
once inducing points number a few dozen (eigenvalues down to ~1e-16), so predictive
means blew up (max deviation from the exact GP grew from ~0.9 at M=10 to ~157 at M=24)
and the LML evaluated ``log`` of negative values (RuntimeWarning, garbage bound). The
old test suite encoded the defect as a weak ``corr > 0.30`` assertion plus a
copy-pasted comment referencing the unrelated Sobol-table problem [11].

**Fix:** Rewrote ``sparse.py`` in the **whitened parameterization**: everything runs
through jittered Cholesky solves of ``Luu``; the posterior lives on whitened inducing
values with precision ``I + V Lam^-1 V^T`` (eigenvalues >= 1 by construction); the LML
uses the closed-form Titsias SGPR bound (identical formula serves the FITC
pseudo-evidence); sparse classes gained a ``log_marginal_likelihood()`` method for
``MarginalLikelihood``/``optimize_hyperparams`` parity.

**Verification:**
- Matches a brute-force Titsias reference exactly on well-conditioned inputs.
- M = T identity: with Z = X the sparse posterior equals the exact GP mean/std to
  ~1e-12 (locked in by ``test_sparse_equals_exact_gp_when_inducing_points_are_training_points``).
- Monotone M-convergence locked in by ``test_sparse_stable_for_many_inducing_points``
  (deviation < 0.1 at M=20, < 1e-6 at M=120 where the old code produced garbage).
- Old weak correlation assertion replaced by ``corr > 0.999`` and max-dev < 0.05.

---

### 24. ``BaseKernel.diag`` crashed for most kernels and all product/power compositions

**Severity:** 7/10 · **Status:** 🟢 `fixed`

**Problem:** Two stacked defects in the diagonal path that every exact-GP *predict*
call exercises. (a) ``BaseKernel.diag(X)`` invoked ``self._matrix(_as_2d(X))`` without
the second argument, so every kernel relying on the inherited implementation (Matern,
Periodic, RationalQuadratic, NeuralNetwork, ArcCosine, SpectralMixture) raised
``TypeError`` whenever ``diag`` was reached. (b) ``KernelProduct`` and ``KernelPower``
had no ``diag`` override at all, so *any* GP prediction using the headline composability
convention (``k1 * k2``, ``k ** 2``) crashed — fits worked, predictions did not, and the
existing tests only ever evaluated full matrices.

**Fix:** ``BaseKernel.diag`` now passes ``Y=None`` explicitly (every ``_matrix``
implementation already handles ``None`` since ``__call__(X)`` depends on it);
``KernelProduct.diag`` returns the elementwise product of part diagonals and
``KernelPower.diag`` the powered base diagonal (both exact and cheaper than the full
matrix).

**Verification:**
- ``test_diag_matches_full_matrix_all_kernels`` locks diag == diag(K) for the entire
  zoo incl. ARD RBF.
- ``test_composite_diag_and_predict_with_product_kernel`` predicts end-to-end through
  an ``RBF * Periodic`` composition and checks product/power diag identities.
- Surfaced by the manual debug session (composed-kernel workflow crashed at first
  predict); full suite green after the fix.

---

### 25. Elliptical copula CDF factorized densities instead of integrating them

**Severity:** 8/10 - **Status:** fixed (unreleased; ships with 0.3.0)

**Problem:** The first Gaussian/t-copula CDF used a chain rule of the form
F(z) = prod P(Z_k <= z_k | Z_<k = z_<k), treating conditionals as pinned at the
realized limits. That factorizes DENSITIES, not distribution functions - values
came out badly wrong (e.g. C(.3,.4)=0.159 vs oracle 0.209).

**Fix:** Replaced with exact recursive 1-D integration over truncated
conditionals (scipy.integrate.quad per level, Schur-complement state updates;
multivariate-t conditionals stay t with nu+1). No quadrature-free shortcut
exists for general d.

**Verification:**
- Matches an independent bivariate-normal quadrature oracle to ~1e-16 on a grid;
  boundary identities C(u,1)=u and independence product exact; monotone in every
  coordinate (locked in tests).

---

### 26. Copula samplers inverted the wrong conditional transform

**Severity:** 9/10 - **Status:** fixed (unreleased; ships with 0.3.0)

**Problem:** Sequential samplers treated P(U<=w|V=v) as if it were the ratio
C(v,w)/C(v,1) - true only in special cases, false for Archimedean and Plackett
families. Sampled margins came out non-uniform (means ~0.33-0.43) and sampled
Kendall's tau far off theory (Frank: 0.27 vs 0.46).

**Fix:** Archimedean families now invert the generator-derivative conditional
psi'(phi(u)+phi(w))/psi'(phi(u)) on a fixed w-grid (exact, vectorized);
Plackett uses its closed-form dC/dv ratio. The wrong quadratic-inversion
shortcut was removed entirely.

**Verification:**
- All families: sampled margins uniform within MC noise and empirical Kendall's
  tau within 5 standard errors of theory at n=12000 (locked in by tests).

---

### 27. Archimedean generator algebra errors across five families

**Severity:** 7/10 - **Status:** fixed (unreleased; ships with 0.3.0)

**Problem:** Several closed-form primitives were wrong: BB1/BB7 used generator
exponent conventions inconsistent with their documented CDFs; BB7's
phi/phi-inverse pair was mutually inconsistent; Joe's phi' carried a wrong sign
and scale; Frank's and Clayton's psi'' had leftover algebra slips. Grid-mass
integration exposed them (Joe's 'density' integrated to -1.17e6).

**Fix:** All primitives rederived from the documented CDFs; a validation harness
compares each family's CDF against independent textbook formulas (~1e-15),
checks psi(phi(u))=u round-trips, and compares bivariate densities against
finite differences of the CDF plus grid mass in [0.96, 0.98].

**Verification:**
- Full family matrix passes the harness; densities strictly positive with unit-
  approximating mass; tau formulas (Genest-MacKay) match known closed forms
  (Clayton theta/(theta+2), Gumbel 1-1/theta).

---

### 28. Dependence-measure plumbing: O(n^2) memory and unbounded inversion cost

**Severity:** 4/10 - **Status:** fixed (unreleased; ships with 0.3.0)

**Problem:** (a) kendall_tau_estimate materialized n x n sign matrices -
2.98 GB allocation failure at n=20000. (b) Frank/Joe tau inversion probed the
Genest-MacKay integral at extreme theta where the substituted integrand is
near-singular - quad hung for minutes inside vine pair selection. (c) The
Student-t profile MLE re-ran the full marginal transform per scalar-MLE
iteration (6036 loglik evaluations per fitted pair at n=1500, ~25 s).

**Fix:** (a) Fenwick-tree inversion counting, O(n log n) time / O(n) memory,
verified equal to scipy.stats.kendalltau with and without ties. (b) Frank uses
the exact Debye-D1 relation; bounds tightened (Clayton<=300, Gumbel<=200,
Joe<=60); remaining numeric inversions run over a cached per-class monotone
tau(theta) curve with bracketed local refinement. (c) Coarse nu grid (14 pts)
plus ONE bounded refinement; t-copula density vectorized (was one slogdet per
row). Net effect: fitted pair cost at n=1500 dropped from ~24.5 s to ~3.8 s.

**Verification:**
- tau estimator equals scipy on tie-free and heavy-tie data to 1e-10;
  df recovery stays accurate after the coarse-grid change (nu=3.86 on nu=4
  ground truth); full vine fit of 10 edges dropped from >15 min to ~2 min.

---

### 29. Vine sampler cluster: shallow introductions, mirrored sides, stale cache

**Severity:** 8/10 - **Status:** fixed (unreleased; ships with 0.3.0)

**Problem:** Four stacked defects made simulated vines inconsistent with their
own fitted pairs. (a) Introduction order used shallow-tree edges, so variables
were drawn from single-sibling conditionals instead of their full conditionals.
(b) Mid-realization fallback drew conditioning leaves marginally without
marking them done, so later plan steps re-introduced and OVERWROTE them.
(c) realize()/columns() assumed mirror sides ('a'<->'b') instead of following
each edge's stored away-side, deep trees conditioned on the wrong sibling
column (margins overshooting fit taus, e.g. 0.487 vs 0.264 ground truth).
(d) _col_cache persisted from fit(), serving stale n=1200 arrays to post-fit
calls on larger samples (phantom KS failures).

**Fix:** Planner rewritten as R-vine-matrix-style PEELING (deepest-first greedy:
every variable introduced through an edge whose entire leaf set minus that
variable is already drawn; seed tried exhaustively). Edges store their actual
away-sides; realization and column recomputation follow stored sides. Column
cache is call-local. Validation switched to what theory guarantees: Rosenblatt
KS-uniformity restricted to realization-diagonal edges, adjacent-margin taus
equal to effective rotated pair taus, refit stability.

**Verification:**
- D/C/R-vines on 5-d Gaussian data: diagonal-edge KS <= 0.0066 at n=40k;
  all tree-1 margin taus within MC noise of their rotated pairs; pairwise tau
  recovery corr=0.99 (max dev 0.046); refit-on-own-samples stable; d=3 C-vine
  cross-checked against a brute-force numeric conditional sampler.

---

### 30. Rotated pair-copula conventions were internally inconsistent

**Severity:** 6/10 - **Status:** fixed (unreleased; ships with 0.3.0)

**Problem:** h-functions for rotations 90/270 swapped the arguments of the base
conditional, rot180's docstring showed the wrong CDF sign, and the fitting
transform handed the base copula columns that did not correspond to its
rotation's density convention. Consequences: AIC selected rotations against the
wrong likelihoods, and sampled rotated pairs broke Rosenblatt calibration.

**Fix:** Single consistent set: C90 = v - Q(1-u,v), C180 = u+v-1+Q(1-u,1-v),
C270 = u - Q(u,1-v); h-functions verified as exactly dC_rot/dv; density
transforms c90=c_Q(1-u,v), c180=c_Q(1-u,1-v), c270=c_Q(u,1-v) aligned between
fitting and evaluation.

**Verification:**
- For all four rotations: analytic h matches central-difference dC/dv to <1e-4
  and base-density-at-rotated-columns matches the mixed partial of the rotated
  CDF to ~1e-6 (regression-tested).

---

### 31. SurvivalFitter._step_evaluate defaulted to 1.0 for all callers

**Severity:** 6/10 - **Status:** fixed (unreleased; ships with V0.5.1)

**Problem:** The shared step-function evaluator hardcoded `default=1.0` for
entries before the first step, which is correct for survival functions but
wrong for cumulative hazards (should be H=0 before first event) and CIFs
(should be 0). Nelson-Aalen with no events returned H=1.0 instead of 0.

**Fix:** Added a ``default`` parameter; all cumulative-hazard and CIF callers
pass ``default=0.0``.

**Verification:**
- NA with zero events returns H=0; CIF predict returns 0 before first event;
  KM still returns S=1 (regression-tested).

---

### 32. CumulativeHazard integration grid started too late

**Severity:** 5/10 - **Status:** fixed (unreleased; ships with V0.5.1)

**Problem:** The parametric-model integration grid started at
times.min()*0.5, missing accumulated hazard between 0 and that point.
WeibullSurvival(shape=1, scale=2) at t=2 returned H=0.50 instead of 1.00.

**Fix:** Grid starts at 1e-8 (near zero); also fixed shape mismatch in the
Riemann sum (h[:-1] * diff(grid) instead of h * diff(grid)).

**Verification:**
- WeibullSurvival(shape=1, scale=2) at t=2 returns H=1.0000 exactly;
  regression-tested.

---

### 33. HazardFunction rejected library distribution objects

**Severity:** 4/10 - **Status:** fixed (unreleased; ships with V0.5.1)

**Problem:** HazardFunction wrapper required source objects to expose
.hazard() or .hazard_(), but none of the 47 library distributions implement
those methods — they only have .pdf()/.cdf(). Wrapping Exponential(0.5)
raised TypeError.

**Fix:** Added generic fallback computing hazard as pdf(t)/(1-cdf(t)) from
any object exposing pdf and cdf callables.

**Verification:**
- HazardFunction(source=Exponential(0.5)).predict([3]) returns 0.5 exactly
  (constant hazard for exponential); regression-tested.

---

### 34. Gompertz exp(b*t) overflowed for large b*t products

**Severity:** 3/10 - **Status:** fixed (unreleased; ships with V0.5.1)

**Problem:** GompertzSurvival._survival computed exp(a/b*(1-exp(b*t)))
without clipping; for large b*t products the inner exponential overflowed to
inf, producing NaN after the outer multiplication.

**Fix:** Clipped inner exponent argument to [-700, 0].

**Verification:**
- GompertzSurvival(a=.5, b=1e-15).survival([1,2]) equals exp(-.5*[1,2])
  to machine precision; regression-tested.

---

### 35. InformationGain computed H(Y) from raw labels instead of frequency counts

**Severity:** 6/10 - **Status:** fixed (unreleased; ships with V0.6.1)

**Problem:** InformationGain.fit passed the raw categorical label array to the
entropy estimator instead of its frequency counts. H(Y) was then computed over
n distinct "symbols" each seen once, producing wildly inflated gains (6.3 bits
instead of 0.01 for near-independent data).

**Fix:** Compute H(Y) from np.unique(y, return_counts=True) counts before
subtracting the conditional entropy.

**Verification:**
- IG equals MutualInformation on the same (x, y) pair for independent and
  dependent datasets; regression-tested (12 new edge-case tests in
  tests/information_theory/tests.py).

---

### 36. RenyiEntropy(alpha=0) used natural log instead of log2

**Severity:** 3/10 - **Status:** fixed (unreleased; ships with V0.6.1)

**Problem:** The alpha=0 corner (Hartley/max-entropy case, log of the support
size) used np.log instead of np.log2, returning nats (1.3863 for a uniform
4-symbol source) where every other Renyi order returns bits.

**Fix:** Use log2 at alpha=0; the alpha->1 Shannon limit path was already
correct.

**Verification:**
- RenyiEntropy(alpha=0) on a uniform 4-symbol source returns exactly 2.0 bits;
  monotone convergence to Shannon as alpha -> 1 regression-tested.

---

### 37. Implementation-Checklist queueing section never checked off; headline progress figure arithmetically wrong

**Severity:** 4/10 - **Status:** fixed (V0.6.2)

**Problem:** The queueing module shipped complete in Phase 16 (29/29 spec
names, full test file, module README), but its Implementation-Checklist section
kept every box unchecked. The checklist's own arithmetic chain then broke:
Phase 16 reported 287/794 (257 + 29 is 286, not 287), and Phase 18 added
information_theory's 31 names to that wrong base and reported 288. The true
implemented total across the nine shipped modules is 317/794
(21+60+25+61+36+26+28+29+31). Every document quoting 288 inherited the error,
and the library conformance test had been weakened to >= 280 to accommodate
it.

**Fix:** Checked off the entire queueing checklist section (72 boxes), set the
progress line to 317/794, restored the conformance test to the exact
== 317 invariant with queueing in the implemented tuple, and propagated the
correct figure through README, development docs and the new
tests/docs/tests.py consistency suite (which now recomputes the checklist
totals from the section headers and fails on any drift).

**Verification:**
- tests/docs/tests.py::test_implementation_checklist_progress_line_matches_reality
  recomputes 317 from the checklist sections against tests/library/
  _spec_names.json and passes; full docs suite green.

---

### 38. spl --help inventory missing the queueing and information_theory blocks

**Severity:** 3/10 - **Status:** fixed (V0.6.2)

**Problem:** CHANGELOG Phases 16 and 18 both claim "spl --help gained the
<module> block", but cli.py::_implemented_overview() never received those
blocks - the inventory stopped at survival. Any pip install's primary
self-description silently omitted the two newest modules.

**Fix:** Added the queueing and information_theory inventory blocks (guarded by
the same hasattr pattern as the other seven).

**Verification:**
- New tests/docs/tests.py::test_spl_help_invents_every_module asserts every
  implemented module name appears in spl --help output; manual session
  confirmed all nine blocks render.

---

### 39. test_top_level_package_wiring hardcoded the version literal, breaking CI on every bump

**Severity:** 3/10 - **Status:** fixed (V0.6.3)

**Problem:** The library wiring test asserted stochpylib.__version__ == "0.6.1"
as a string literal. The V0.6.2 version bump (docs-only phase) left the test
green on the dev machine only because the editable install's stale metadata
still reported 0.6.1; on CI's fresh install the metadata read 0.6.2, the test
failed ~80 s into pytest, and GitHub Actions' default fail-fast cancelled the
other seven matrix jobs (one real failure, seven "Cancelled" entries).

**Fix:** The test now asserts consistency instead of a literal: pip metadata
version must equal the in-code __version__ (skipped gracefully when running
from source without an install). Version literals belong only in
pyproject.toml/__init__.py, which the tests/docs suite already cross-checks.

**Verification:**
- tests/library + tests/docs green locally after refreshing the editable
  install; CI run on the V0.6.3 commit green across the full matrix.
