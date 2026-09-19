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

---

### 40. Test-injection parameter where None meant "do the real thing" let a unit test run live pip

**Severity:** 5/10 - **Status:** fixed (unreleased; ships with V0.6.4)

**Problem:** The first draft of `spl update`'s tests passed `_meta=None` to
simulate an unreachable PyPI, but `cmd_update` interpreted `None` as "not
provided" and performed a real forced PyPI fetch - then executed a **real**
`pip install stochpylib==0.1.1` subprocess inside the test run. The
development environment survived only because the pip invocation failed
against the editable install; the design flaw itself is the finding: a unit
test mutated (or tried to mutate) the machine's package state.

**Fix:** Sentinel-based injection: `_meta` now defaults to `_UNSET`
("perform the real fetch"); an explicitly passed `None` means "simulate
offline". The shipped test suite passes `_meta`/`_run`/`_input`/`_mode`
on every path, so no test can reach the network or pip. Rule recorded: an
injection parameter whose default triggers real side effects must use a
sentinel, never `None`.

**Verification:**
- Full tests/cli suite green with zero network access (urlopen monkeypatched
  to raise AssertionError on any call in the fetch tests); manual session
  confirms the real paths still work (dry-run, unknown-version rejection,
  editable refusal).

---

### 41. `KouJumpDiffusion.call_price` used log-moneyness where the Carr-Madan formula needs the absolute log-strike

**Severity:** 9/10 · **Status:** 🟢 `fixed` (V0.7.0)

**Problem:** `carr_madan_call()` set `k = math.log(K / S0)`, but `cf_log_price(u)`
is the characteristic function of the *absolute* log-price `ln S_T` (it embeds
`ln S0` in its own drift term). Feeding the formula a relative log-moneyness
while the CF already carries `S0` left an uncancelled `S0` factor inside the
Fourier phase, producing wildly wrong prices (a Kou call priced at 99.05
against a Black-Scholes-adjacent Monte Carlo value of ~9-12, and the same
failure reproduced with a plain GBM characteristic function, proving the bug
lived in `carr_madan_call` itself, not the Kou-specific CF).

**Fix:** `k = math.log(K)` (the absolute log-strike, matching the absolute
log-price CF), with a docstring note spelling out the convention so it can't
regress silently.

**Verification:**
- Reproduced with a bare GBM CF against the library's own Black-Scholes
  closed form (`carr_madan_call` returned 99.05 vs the exact 10.45); after the
  fix it returns 10.450583... matching to 1e-6.
- `tests/levy_processes/tests.py` pins `KouJumpDiffusion.call_price` against
  Black-Scholes in the zero-jump limit and against `call_price_mc` at
  `mu=r` (matching risk-neutral measures) within a few Monte-Carlo standard
  errors.

---

### 42. `TemperingSubordinator`/`CGMYProcess` truncated-jump sampler used a linear grid, biasing the mean ~50%

**Severity:** 9/10 · **Status:** 🟢 `fixed` (V0.7.0)

**Problem:** `_jump_quantile_grid()` (and CGMY's `_jump_grid()`) built the
truncated Lévy-density quantile grid on `np.linspace(jump_floor, x_max, 8192)`
and integrated it with a plain `cumsum` (an implicit left-Riemann sum). The
density `x**(-1-alpha)` is singular at `jump_floor`, and the linear grid's
spacing (~2.4e-4 for `jump_floor=1e-4`) is *coarser* than the singularity
scale itself, so the Riemann sum massively overweighted the near-floor region
for the 0th moment (measured: 366.4 vs the true 192.2, almost 2x) while barely
moving the 1st moment (0.787 vs 0.773) — the ratio of the two (the sampled
mean jump size) came out ~47% too small, and the retained-jump Poisson count
(computed separately via the exact `upper_gamma_negative` analytic formula)
stayed correct, so only the sampled jump *sizes* were biased low. This is
exactly the "TemperingSubordinator mean off by ~48%" issue left open in the
prior session.

**Fix:** Log-spaced grid (`np.logspace`) plus `scipy.integrate.cumulative_trapezoid`
for the CDF — verified against `scipy.integrate.quad` to within 0.01% at 1024
points, vs. the old scheme's ~2x error at 8192. Applied identically to both
`TemperingSubordinator._jump_quantile_grid` and `CGMYProcess._jump_grid`
(same bug, same fix). Also removed unreachable dead code left after the
`upper_gamma_negative` two-step lift (an `if s > -1.0: return ...` followed
by unconditional-but-unreachable duplicate statements).

**Verification:**
- `TemperingSubordinator(C=1, lam=5, alpha=0.5)`: simulated mean now 0.0790
  vs analytic `mean_rate()*dt` 0.0793 (0.3% off, was 47% off).
- `tests/levy_processes/tests.py` asserts the simulated mean against
  `mean_rate()` for both classes at several standard errors.

---

### 43. `CoxProcess.simulate` crashed on numpy >= 2.x (`np.trapz` removed)

**Severity:** 6/10 · **Status:** 🟢 `fixed` (V0.7.0)

**Problem:** `np.trapz` was deprecated in numpy 2.0 and is gone entirely by
2.5 (`AttributeError: module 'numpy' has no attribute 'trapz'`); every call
to `CoxProcess.simulate` crashed on a current numpy install.
`jump_diffusion.carr_madan_call` already guarded this exact case with
`np.trapezoid(...) if hasattr(np, "trapezoid") else np.trapz(...)`, but
`advanced.py`'s `CoxProcess.simulate` used the bare, unguarded call.

**Fix:** Same `hasattr` fallback as `carr_madan_call`.

**Verification:** `CoxProcess(lambda t: 3.0).simulate(2.0, random_state=0)`
now runs; `tests/levy_processes/tests.py` exercises `CoxProcess` directly
(the whole test file would otherwise fail to collect on this numpy version).

---

### 44. `StableSubordinator` and `RandomMeasure(kind="stable")` didn't match their documented Laplace transform

**Severity:** 8/10 · **Status:** 🟢 `fixed` (V0.7.0)

**Problem:** Both sampled `StableDistribution(alpha, beta=1.0, scale=dt**(1/alpha))`
and claimed `E[exp(-lam*T_t)] = exp(-t*lam**alpha)`. The library's stable
sampler uses the S1 (Nolan) parameterization, whose actual one-sided Laplace
transform at `scale=1` is `exp(-lam**alpha / cos(pi*alpha/2))` — an extra
multiplicative constant the docstring's formula doesn't have. Measured at
`alpha=0.6`: Monte-Carlo `E[exp(-1.5*T_1)]` came out 0.2185 against the
documented 0.4095 (off by roughly a factor of 2, worse at smaller alpha).

**Fix:** Rescale by `cos(pi*alpha/2)**(1/alpha)` before applying the
`dt**(1/alpha)` (or `length**(1/alpha)`) self-similarity scaling — this
constant exactly cancels the S1 parameterization's extra factor, derived from
`E[exp(-lam*(c*X))] = exp(-(lam*c)**alpha / cos(pi*alpha/2))` and solving
`c**alpha / cos(pi*alpha/2) = 1`. Applied to both `StableSubordinator._increment`
and `RandomMeasure.sample` (kind="stable") — same underlying sampler, same
bug, same fix. (`StableProcess`/`SpectrallyPositive` were checked and are
*not* affected: their `characteristic_function` is self-consistent with the
S1 sampler by construction, and self-similarity under the `dt**(1/alpha)`
scaling holds for any fixed constant, so nothing there claims the "clean"
`exp(-t*lam**alpha)` normalization.)

**Fix verification:** `alpha=0.6`, `lam=1.5`, `t=0.7`: Monte-Carlo Laplace
transform now 0.4091 vs analytic 0.4095 (was 0.2185).

**Verification:**
- `tests/levy_processes/tests.py` checks the Monte-Carlo Laplace transform of
  both `StableSubordinator` and `RandomMeasure(kind="stable")` against the
  documented closed form.

---

### 45. `Runge_Kutta_SDE` was a stochastic-Heun scheme mislabeled "strong order 1.0" — actually order 0.5

**Severity:** 8/10 · **Status:** 🟢 `fixed` (V0.7.0)

**Problem:** The scheme averaged drift and diffusion at a predictor stage
(`0.5*(a1+a2)*dt + 0.5*(b1+b2)*dW`) — a stochastic-Heun / trapezoidal update.
This is a *weak*-order-2 construction; it has no Itô correction term
(`b*b'*(dW^2-dt)`-shaped) and so cannot exceed strong order 0.5. A
strong-error convergence study (same driving Brownian path at each step
count — see #47) measured its empirical order at ~0.49 on a GBM test SDE,
statistically indistinguishable from Euler-Maruyama's ~0.50, and its absolute
error was consistently *larger* than EM's at every step size tested.

**Fix:** Replaced with the derivative-free Milstein (Platen) scheme —
`H = x + a*dt + b*sqrt(dt)`; `x_new = x + a*dt + b*dW + 0.5/sqrt(dt) * (b(H)-b(x)) * (dW**2-dt)`
(Kloeden-Platen 1992, eq. 11.1.4) — which approximates the Milstein
correction term by a finite difference of `b` alone (no explicit derivative
needed) and genuinely reaches strong order 1.0.

**Verification:** Convergence study on GBM (`n_steps` in
16..256, 40k paths, shared Brownian path against the exact solution):
empirical order 0.996 (was ~0.49). `tests/levy_processes/tests.py` asserts
the fitted log-log order is > 0.85 and that RK's error is well below EM's at
matched step counts.

---

### 46. `StochasticTaylor`'s multiple stochastic integrals were wrong — order 1.0 instead of documented 1.5

**Severity:** 8/10 · **Status:** 🟢 `fixed` (V0.7.0)

**Problem:** The prior session's rewrite of this scheme (Probleme #1) fixed
the grossest errors but still had incorrect formulas for two of the multiple
Itô integrals: `J01` (meant to be the double integral
`I_(1,0) = int_0^dt int_0^s dW(u) ds`, jointly Gaussian with `dW` — mean 0,
`Var = dt**3/3`, `Cov(dW, ·) = dt**2/2`) was coded as `dt**1.5 * z / sqrt(3)`,
dropping the `0.5*dt*dW` correlated component and the leading `0.5` on the
independent-noise term entirely; and `J111` (the triple same-integrand
integral, `(dW**3 - 3*dt*dW)/6`, a *deterministic* function of `dW` and `dt`
with no independent randomness) had a spurious extra `+ dt**1.5*z/sqrt(3)`
term appended, and the coefficient it was multiplied by in the update also
didn't match the Kloeden-Platen formula's grouping. Net effect: a
convergence study (shared Brownian path, see #47) measured empirical strong
order ~1.00, not the documented 1.5.

**Fix:** Rewrote to match Kloeden-Platen (1992, eq. 10.4.3) term-by-term:
`dZ = 0.5*dt*dW + 0.5/sqrt(3)*dt**1.5*z` for `I_(1,0)`; `(dW*dt - dZ)` for its
complement `I_(0,1)`; `((1/3)*dW**2 - dt)*dW` (undivided, multiplied by its
own correctly-grouped coefficient `0.5*b*(b*b''+b'**2)`) for the triple
integral's contribution — no independent randomness in that last term.

**Verification:** Same GBM convergence study as #45: empirical order 1.501
(was ~1.00), and absolute error roughly 100x smaller than Milstein's at the
finest step size tested. `tests/levy_processes/tests.py` asserts the fitted
order is > 1.2.

---

### 47. `StrongApproximation` shared one already-advanced RNG between the exact and approximate solvers

**Severity:** 9/10 · **Status:** 🟢 `fixed` (V0.7.0)

**Problem:** `rng = np.random.default_rng(random_state)` was created once per
step size and passed to *both* `exact_solver(..., rng)` and
`solver(..., rng)` in sequence. `np.random.default_rng` returns an existing
`Generator` unchanged, so the exact solver consumed the stream first and the
approximate solver then drew a *different, independent* set of Brownian
increments — the two paths being "compared" never shared a driving path at
all. The measured "strong error" was therefore just the statistical
difference between two unrelated GBM paths (~30 in absolute terms on a
Wiener a jumps~100-scale process) and did not shrink as the step count
increased — this is the root cause behind #45 and #46 initially reporting
order ~0 (no trend at all) before the RNG-sharing bug was found and fixed
first.

**Fix:** Reseed a *fresh* generator from the same `random_state` value before
each of the exact and approximate calls (`np.random.default_rng(random_state)`
called twice), so both see the identical driving Brownian path at every step
size. Documented that `random_state` must therefore be a reusable seed
(int/array/`None`), not a live `Generator`, for the convergence study to be
meaningful.

**Verification:** Same GBM study: Euler-Maruyama's fitted order became 0.50
(previously the "error" did not decrease with step size at all — see the
raw numbers in #45/#46 above, ~30 and flat/increasing). `tests/levy_processes/tests.py`
uses `StrongApproximation` for the EM/Milstein/RK/Taylor order checks.

---

### 48. `WeakApproximation`'s three-point increment distribution had the wrong weights, doubling `E[dW**2]`

**Severity:** 9/10 · **Status:** 🟢 `fixed` (V0.7.0)

**Problem:** The weak-order-2 scheme requires the discrete increment
`dW in {-sqrt(3*dt), 0, +sqrt(3*dt)}` with probabilities `{1/6, 2/3, 1/6}` (so
its first four moments match `N(0, dt)` exactly). The code drew
`r = rng.integers(0, 3, n_paths)` and mapped each of the three outcomes with
**equal** probability 1/3, giving `E[dW] = 0` (fine, symmetric) but
`E[dW**2] = 2*dt` — twice the required variance. On a GBM test SDE this
produced an `E[X_T]` bias of ~2.14 (against a value of ~105, so ~2%) that did
**not** shrink as `n_steps` increased from 10 to 50 — a flat, non-vanishing
bias is the signature of a moment-matching error, not ordinary discretization
error, and is exactly what CONTINUATION.md's "expected at this coarse
resolution; converges with refinement" note got wrong: it does not converge,
because the per-step second moment is wrong at every resolution.

**Fix:** Sample via `u = rng.random(n_paths)` thresholded at `1/6` and `5/6`
so the three outcomes get probabilities `1/6, 2/3, 1/6` as required.

**Verification:** GBM check (`mu=0.05`, `sigma=0.2`, `T=1`, 100k paths):
bias now 0.025-0.14 across `n_steps` in 5..50 (was a flat ~2.14, ~1-2 Monte
Carlo standard errors instead of ~30). `tests/levy_processes/tests.py`
asserts `E[X_T]` against the exact GBM mean within a small multiple of the
Monte-Carlo standard error.

---

### 49. `stochpylib/cli_demo.py` declared `DEMO_MODULES` in `__all__` without defining it

**Severity:** 2/10 · **Status:** 🟢 `fixed` (V0.7.0)

**Problem:** `__all__ = ["DEMO_MODULES", "run_demo", "list_demos"]` but the
module only ever defined `DEMOS` (the dict); `from stochpylib.cli_demo import
DEMO_MODULES` would have raised `ImportError` for any caller that actually
used the declared public name. Nothing in the shipped test suite imported it
by name, so it went unnoticed.

**Fix:** Added `DEMO_MODULES = tuple(DEMOS)` alongside the existing `DEMOS`
dict, and added the `levy_processes` demo entry to `DEMOS` at the same time
(Probleme.md entries here are for the V0.7.0 `levy_processes` module wrap-up;
this one was found while touching the same file to add that demo).

**Verification:** `from stochpylib.cli_demo import DEMO_MODULES` now succeeds
and includes all ten implemented modules; `spl demo levy_processes` runs.

---

### 50. `HawkesProcess.ks_residuals()` always raised when called after `.fit()` with no arguments

**Severity:** 6/10 · **Status:** 🟢 `fixed` (V0.7.0)

**Problem:** The documented usage is `HawkesProcess().fit(events).ks_residuals()`
— call the time-rescaling KS test on the same data just fitted, without
re-passing it. The `events is None` branch read `mu, alpha, beta` and `T`
back from the fitted model but never recovered `events` itself, leaving it
`None`; the very next line (`if events is None or events.size < 2: raise
ValueError(...)`) then always fired. The zero-argument call path documented
in the class docstring (`ks_residuals()` "applies the time-rescaling
theorem... returns the KS (statistic, p_value)") never worked.

**Fix:** `fit()` now also stores `self._events = events`; `ks_residuals()`
recovers it (`events = self._events`) in the `events is None` branch instead
of leaving it unset.

**Verification:** `HawkesProcess(mu=0.5, alpha=0.3, beta=1.0).simulate(2000.0,
random_state=42)` then `.fit(...).ks_residuals()` now returns `(0.018, 0.77)`
— a high p-value, as expected for a well-specified model — instead of
raising. `tests/levy_processes/tests.py` calls `ks_residuals()` with no
arguments immediately after `.fit()`.

---

### 51. `TemperingSubordinator.truncation_mass()` docstring described the opposite of what it computes

**Severity:** 3/10 · **Status:** 🟢 `fixed` (V0.7.0)

**Problem:** Docstring said "Expected Levy mass excluded below `jump_floor`",
but the formula (`C * lam**alpha * upper_gamma_negative(-alpha, lam*floor)`)
is exactly `_retained_intensity(1)` — the upper incomplete gamma integrates
the Lévy density from `lam*floor` to infinity, i.e. the intensity of jumps
*above* the floor (what `_increment` actually draws its Poisson count from),
not the excluded mass below it. Purely a documentation defect — no code path
used the wrong value numerically — but a user reading the docstring would
draw the wrong conclusion about what the method reports.

**Fix:** Reworded both the class and method docstrings to describe the
retained-intensity semantics accurately, and pointed to
`mean_rate() - _retained_mean_rate()` (the actual excluded-mass-in-mean-terms
quantity, since the excluded jump *count* diverges as `jump_floor -> 0` and
so has no finite "mass" of its own to report).

**Verification:** Docstring-only change; `tests/levy_processes/tests.py`
still asserts `truncation_mass() > 0.0`.

---

### 52. CI failed: `AutoReg(..., old_names=False)` broke against statsmodels 0.15.0

**Severity:** 6/10 · **Status:** 🟢 `fixed` (V0.7.0)

**Problem:** `test_ar_recovery_and_statsmodels_exact` passed `old_names=False`
to `statsmodels.tsa.ar_model.AutoReg` — a long-deprecated legacy-compat flag
that statsmodels 0.15.0 (released after this repo's last CI run) removed
entirely, raising `TypeError: unexpected keyword argument 'old_names'`. All 8
CI matrix jobs showed red: one (3.11, windows-latest) failed for real, the
other 7 were cancelled by the default `fail-fast` matrix strategy once it did.

**Fix:** Dropped the kwarg — `old_names` already defaulted to `False` in
0.14.x, so the call is unchanged for the supported floor (`statsmodels>=0.14`)
and now works on 0.15.0 too, where the parameter is gone.

**Verification:** Passes locally (statsmodels 0.14.6). Cross-checked in an
isolated venv against statsmodels 0.15.0: full `tests/timeseries/` suite
(52 tests) green, confirming no other 0.15.0 incompatibility in that module —
`adfuller`'s new `result_object=` return-shape opt-in raises only a
`FutureWarning` on 0.15.0 (no `result_object` kwarg exists yet on the 0.14
floor, so it's intentionally left unset rather than pinned either way).

---

### 53. `FourierOptionPricing`'s COS method priced wildly wrong (off by ~1e19)

**Severity:** 8/10 · **Status:** 🟢 `fixed` (V0.8.0)

**Problem:** The Fang-Oosterlee COS implementation computed the truncation
range's variance as the raw second moment of `ln S_T` (`E[z^2]`) instead of
the true variance (`E[z^2] - E[z]^2`) — for `S0=100`, `E[z] ~ ln(100) ~ 4.6`,
so the "variance" came out ~21 instead of the true ~0.04, blowing the
truncation interval `[a,b]` up by orders of magnitude. Compounding this, the
call/put payoff integration bounds used `0`/`b` (or `a`/`0`) instead of
`ln(K)`/`b` (or `a`/`ln(K)`), and a spurious extra `(b-a)/2` scale factor was
applied on top of the already-included `2/(b-a)` in the coefficients. Net
result: a K=100 call priced at `2.7e19` instead of `10.45`.

**Fix:** Compute both cumulants (`c1`, `c2`) via finite differences of the
characteristic function at `u=0` (`c1 = -i*phi'(0)`, `c2 = -phi''(0) - c1^2`),
use `kappa = ln(K)` as the payoff-support boundary in the `chi`/`psi`
integrals, and drop the spurious scale factor.

**Verification:** `test_fourier_bs_cf_reproduces_bs` — Carr-Madan and COS
agree with closed-form Black-Scholes to `1e-4` across `K` in `[80,120]`; both
methods now agree with each other to `1e-6`.

---

### 54. `RoughHeston.characteristic_function` integrated the Riccati derivative instead of its solution

**Severity:** 8/10 · **Status:** 🟢 `fixed` (V0.8.0)

**Problem:** El Euch-Rosenbaum's formula is
`log phi = iu(...) + kappa*theta * I^1[h](T) + v0 * I^(1-alpha)[h](T)`, where
both integral terms integrate the *solution* `h` of the fractional Riccati
equation. The implementation instead integrated `F(h)` (`h`'s time
derivative) in both terms, and the `H=0.5` fast path returned `F(h(T))`
instead of `h(T)`. The characteristic function did not converge to the exact
classical Heston cf even as `H -> 0.5` with `n_steps -> infinity` — it
converged to a different, wrong limit (`-0.1243-0.6401j` vs the exact
`-0.1522-0.6064j` at `u=5, T=1`).

**Fix:** Integrate the `h` array (not `Fh`) for both `I^1` and `I^(1-alpha)`;
the `H=0.5` special case now returns `h[-1]` (the solution's value at `T`),
not `F(h(T))`.

**Verification:** `test_rough_heston_h_half_equals_heston` — cf matches
classical Heston to `1e-5` at `H=0.5` across `u in {1,2,5,10}`, and to `1e-8`
at `u=0` for `phi(0)=1`; call price matches to `1e-5`.

---

### 55. `SABRModel.implied_vol`'s `z/x(z)` skew factor was inverted (and its small-`z` branch sign-flipped)

**Severity:** 8/10 · **Status:** 🟢 `fixed` (V0.8.0)

**Problem:** Hagan (2002)'s formula multiplies by `z/x(z)`; the
implementation computed `x(z)/z` (the reciprocal) in the main branch, and the
small-`z` Taylor expansion used `1 - 0.5*rho*z` instead of the correct
`1 + 0.5*rho*z`. Both ratios tend to 1 as `z -> 0`, so the `nu -> 0` limit
test (which only exercises small `z`) passed regardless — but for realistic
`nu`, the smile's skew direction came out backwards: `rho=-0.6` produced an
*upward*-sloping smile (0.048 at K=90 rising to 0.074 at K=110) instead of
downward.

**Fix:** Compute `x(z)` once, then divide `z` by it (not the reverse);
correct the small-`z` branch to `1 + 0.5*rho*z`.

**Verification:** `test_sabr_smile_shape` — `rho<0` now produces a strictly
lower vol at higher strikes; `test_sabr_beta1_nu0_is_lognormal` (unaffected
by the fix, still passes) and `test_sabr_mc_vs_hagan` (Monte Carlo price
against the corrected Hagan-implied-vol Black price, within 4 SE) both green.

---

### 56. `CreditMigration.generator()`'s IRW regularization corrupted the diagonal

**Severity:** 7/10 · **Status:** 🟢 `fixed` (V0.8.0)

**Problem:** The Israel-Rosenthal-Wei regularization is supposed to zero out
*negative off-diagonal* entries of the matrix logarithm (an artifact of a
non-exactly-embeddable transition matrix) and move the removed mass onto the
diagonal. The implementation instead clipped every entry in each row —
diagonal included — to `>= 0` via `np.maximum(G[i,:], 0.0)`, destroying the
(legitimately negative) diagonal entirely. `expm(generator())` no longer
reproduced the input transition matrix even when regularization should have
been a no-op (all off-diagonals already non-negative): diagonal entries came
back `> 1` and rows no longer summed to a valid distribution.

**Fix:** Only zero strictly negative off-diagonal entries, subtracting the
same (negative) amount from that row's diagonal so the generator's zero-row-
sum constraint is preserved exactly.

**Verification:** `test_credit_migration_powers_and_generator` —
`scipy.linalg.expm(generator())` matches the input `P` to `1e-6`, and every
row of `generator()` sums to `0` to `1e-10`.

---

### 57. `RiskParity.optimize()` renormalized weights every coordinate-descent sweep, preventing convergence to equal risk contributions

**Severity:** 6/10 · **Status:** 🟢 `fixed` (V0.8.0)

**Problem:** Spinu's (2013) convex risk-parity objective is solved over
*unnormalized* `w > 0` (the log-barrier term fixes the scale); the simplex
weights are obtained by normalizing once, after convergence. The
implementation normalized `w` to sum to 1 inside the loop after every single
coordinate update, which changes every other coordinate's fixed-point
equation mid-sweep and prevents the iteration from ever reaching the true
equal-risk-contribution point — risk contributions differed by up to ~30%
instead of matching to numerical precision.

**Fix:** Removed the per-sweep renormalization; normalize `w` exactly once,
after the convergence loop exits.

**Verification:** `test_risk_parity_equal_contributions_and_budgets` — risk
contributions equal to `1e-6` for equal budgets, proportional to custom
budgets to `1e-4`, and the diagonal-covariance closed form (`w_i ~ 1/sigma_i`)
matches to `1e-4`.

---

### 58. Ledoit-Wolf shrinkage estimator's `pi_hat` was missing a factor of `n`, saturating shrinkage at 1.0

**Severity:** 6/10 · **Status:** 🟢 `fixed` (V0.8.0)

**Problem:** Ledoit-Wolf (2004) section 2 defines
`b_bar^2 = (1/n^2) sum_t ||x_t x_t' - S||_F^2` — the sum of per-observation
squared deviations divided by `n^2` (an average-of-`n`-terms estimator of an
`O(1/n)` asymptotic quantity). The implementation divided by `n` only, making
the numerator roughly `n` times too large relative to the target-distance
denominator and pushing the `min(pi_hat, delta2)` clip to `delta2` (full
shrinkage, `shrinkage_ = 1.0`) for essentially any sample size, defeating the
whole point of an *adaptive* shrinkage intensity.

**Fix:** Divide `pi_hat` by `n**2`, matching the paper's normalization.

**Verification:** Cross-checked against `sklearn.covariance.ledoit_wolf` as
an independent oracle on identical data — both now agree on `shrinkage=1.0`
for a genuinely small, noisy `n=15` draw (confirming that specific case was
correct all along) and on sensible partial shrinkage (~0.003 at `n=5000`,
~0.28 at `n=20`) elsewhere. `test_ledoit_wolf_shrinkage_bounds_psd_and_small_n_improvement`
averages squared Frobenius error over 200 independent small-`n` draws and
confirms LW beats the sample covariance in expectation (not guaranteed on
every single draw — a single draw can still legitimately collapse to full
shrinkage and lose to that draw's sample covariance, which is why the test
averages rather than comparing one draw).

---

### 59. Cornish-Fisher Expected Shortfall integrated the wrong tail, returning a negative ES

**Severity:** 7/10 · **Status:** 🟢 `fixed` (V0.8.0)

**Problem:** `ES(alpha) = (1/(1-alpha)) * integral_alpha^1 VaR_u du` averages
the Cornish-Fisher VaR over confidence levels `u` in the *loss* tail
(`u` near 1). The implementation integrated `u` from `~0` to `1-alpha`
instead — the opposite (gain) tail — producing a large *negative* expected
shortfall for a risk measure that should always be `>= VaR > 0` for a
right-skewed loss distribution.

**Fix:** Integrate over `u in [alpha, 1)` instead of `[0, 1-alpha]`.

**Verification:** `test_cornish_fisher_zero_moments_is_normal_and_nonmonotone_raises` —
with zero skew/kurtosis the Cornish-Fisher ES now matches the closed-form
normal ES to `1e-4` (previously it returned a negative number with the wrong
sign entirely).

---

### 60. `ScenarioAnalysis.from_mc`/`from_historical` crashed when the pricer took non-stochastic parameters

**Severity:** 5/10 · **Status:** 🟢 `fixed` (V0.8.0)

**Problem:** Both methods assumed every key in `base_factors` was a
stochastic factor shocked 1:1 by a column of the MC draws / historical factor
returns. Any pricer taking a fixed parameter alongside a stochastic one (e.g.
`lambda f: f["S"] - f["K"]` with only `S` random) raised `IndexError` the
moment `base_factors` had more keys than the shock array had columns.

**Fix:** Added an explicit `factors=` argument naming which `base_factors`
keys are stochastic (defaulting to all of them, preserving the old
single-factor behavior); every other key stays fixed at its base value.

**Verification:** `test_scenario_analysis_mc_linear_var_and_from_historical`
exercises a two-key pricer (`S`, `K`) with only `S` stochastic.

---

### 61. `MonteCarloOptionPricing.price()`'s antithetic/control-variate `n_samples` reported half the requested paths

**Severity:** 3/10 · **Status:** 🟢 `fixed` (V0.8.0)

**Problem:** `montecarlo.applications.option_pricing_mc` (the library's
existing antithetic MC pricer) reports `n_samples` as the total number of
paths *requested*, even though the standard-error calculation internally
uses half that many antithetic pairs. The new `MonteCarloOptionPricing.price`
deviated from that convention, reporting `n_samples` as the pair count
(`n_paths // 2`) instead, inconsistent with the rest of the library's MC
result objects.

**Fix:** Override `n_samples` to the originally requested `n_paths` after
building the `MCResult`, matching `option_pricing_mc`'s convention.

**Verification:** `test_mc_plain_within_se` asserts `res.n_samples == 200_000`
for a `n_paths=200_000` antithetic call.

---

### 62. `ci.yml`'s `os` matrix never actually ran on Windows

**Severity:** 4/10 · **Status:** 🟢 `fixed` (V0.8.0)

**Problem:** `.github/workflows/ci.yml` declared a `strategy.matrix.os:
[ubuntu-latest, windows-latest]` but the job's `runs-on: ubuntu-latest` was a
hardcoded literal instead of `${{ matrix.os }}` — every matrix cell, Windows
included, actually ran on Ubuntu. AGENTS.md §7 has claimed Windows CI
coverage since the matrix was added; it was never real.

**Fix:** `runs-on: ${{ matrix.os }}`.

**Verification:** Next push's Actions run shows both `ubuntu-latest` and
`windows-latest` job instances actually executing on their named runners
(visible in the Actions UI's runner labels), not just in the matrix
definition.

---

### 63. `tests/` had no `__init__.py`, so `tests/statistics/tests.py` shadowed the stdlib `statistics` module

**Severity:** 6/10 · **Status:** 🟢 `fixed` (V0.9.0)

**Problem:** With no `tests/__init__.py`, pytest's default (and its
`importlib`) import mode both name `tests/statistics/tests.py`'s package
after its directory alone (`statistics`), registering `tests/statistics/
__init__.py` in `sys.modules["statistics"]` — clobbering the stdlib
`statistics` module for the rest of the interpreter session the moment the
new module's tests were added.

**Fix:** Added an empty `tests/__init__.py`, anchoring every suite's import
name at the repo root (`tests.<module>.tests`) so no subpackage name can
collide with a stdlib module again.

**Verification:** `pytest --collect-only -q tests/` collects the same count
before/after; `tests/statistics/tests.py::test_stdlib_statistics_module_not_shadowed`
asserts `import statistics` still resolves to the stdlib module inside a
test run.

---

### 64. New `statistics.glm()`'s IRLS used the reciprocal of the link derivative

**Severity:** 8/10 · **Status:** 🟢 `fixed` (V0.9.0)

**Problem:** Each link's `d(eta)/d(mu)` was correctly authored in
`_LINKS`, but the IRLS loop then computed `deta_dmu = 1.0 / that_value` —
inverting it a second time into `d(mu)/d(eta)`. Every non-trivial GLM fit
(anything beyond one lucky iteration) converged to the wrong coefficients
silently, with no error raised.

**Fix:** Removed the spurious reciprocal; the tuple element is used
directly as `d(eta)/d(mu)`.

**Verification:** `test_glm_family_link_vs_statsmodels` (9 family/link
pairs) and `test_glm_negative_binomial_vs_statsmodels` now match
`statsmodels.GLM` coefficients to 1e-4 after full convergence (previously
diverged by whole units after more than one iteration).

---

### 65. `statistics.glm()` clipped Gaussian-family fitted means to be positive

**Severity:** 7/10 · **Status:** 🟢 `fixed` (V0.9.0)

**Problem:** `_mu_bounds()` defaulted every family except binomial to
`(1e-10, inf)`, including `"gaussian"` — whose response mean is legitimately
unbounded. Every Gaussian/identity IRLS iteration silently clipped negative
fitted values to `1e-10`, corrupting the working response and preventing
convergence to the OLS-equivalent solution.

**Fix:** `_mu_bounds()` returns `(-inf, inf)` for `"gaussian"`.

**Verification:** `test_glm_family_link_vs_statsmodels[gaussian-identity]`
matches `statsmodels.GLM(family=Gaussian())` to 1e-8 (previously off by
whole units after convergence).

---

### 66. `statistics.glm()` conflated family bounds with link-domain bounds

**Severity:** 5/10 · **Status:** 🟢 `fixed` (V0.9.0)

**Problem:** Fixing #65 by unbounding the Gaussian family broke
Gaussian-with-log-link (`log(mu)` needs `mu > 0` regardless of family,
gaussian included) — `_mu_bounds()` had conflated the family's natural
response range with the link function's domain into one lookup.

**Fix:** Split into `_LINK_BOUNDS` (per-link domain) and `_mu_bounds(family,
link)` (their intersection), so a log link always forces positivity even
under an otherwise-unrestricted family.

**Verification:** `test_glm_family_link_vs_statsmodels[gaussian-log]`
converges and matches `statsmodels` to 1e-7 (previously raised `SVD did not
converge`).

---

### 67. `statistics` OLS/GLM AIC and BIC over-counted parameters by one

**Severity:** 3/10 · **Status:** 🟢 `fixed` (V0.9.0)

**Problem:** `linear_regression`'s and `glm`'s AIC/BIC added `+1` to the
parameter count for the estimated noise variance/dispersion, following the
textbook `-2*llf + 2*(k+1)` form. `statsmodels.OLS`/`statsmodels.GLM` do not
add that `+1` (dispersion is not counted as a free AIC parameter in their
convention), so results differed by exactly `2` (AIC) or `log(n)` (BIC).

**Fix:** Dropped the `+1`; `aic_ = -2*loglik_ + 2*p`, matching
`statsmodels` exactly.

**Verification:** `test_linear_regression_ols_vs_statsmodels` and
`test_glm_family_link_vs_statsmodels` assert `aic_`/`bic_` equal
`statsmodels`' to within numerical tolerance (previously off by exactly 2.0
in every case).

---

### 68. `statistics.glm()`'s Gaussian+identity log-likelihood used the wrong dispersion

**Severity:** 3/10 · **Status:** 🟢 `fixed` (V0.9.0)

**Problem:** `loglik_` was always evaluated at the Pearson dispersion
(`SSR/df_resid`, the unbiased estimator used for standard errors). For
Gaussian family with the identity link specifically, `statsmodels.GLM`
reports the *concentrated* log-likelihood, using the MLE (biased,
`SSR/n`) variance instead — a documented statsmodels-specific convention,
not shared by any other family/link combination.

**Fix:** `glm()` now uses `SSR/n` for `loglik_` only in the
Gaussian-and-identity case, keeping the Pearson dispersion for standard
errors and every other family/link.

**Verification:** `test_glm_family_link_vs_statsmodels[gaussian-identity]`
matches `statsmodels`' `llf` to 1e-6 (previously off by ~0.013 nats).

---

### 69. `statistics.factor_analysis(method="ml")`'s discrepancy function had a spurious term

**Severity:** 6/10 · **Status:** 🟢 `fixed` (V0.9.0)

**Problem:** The concentrated ML objective should sum
`(w_i - log(w_i) - 1)` over only the `p - n_factors` *smallest* eigenvalues
of `Psi^-1/2 R Psi^-1/2` (the top `n_factors` are absorbed by the loadings
with zero residual by construction). The implementation added an extra,
mathematically invalid term for the top eigenvalues as well, biasing every
fitted uniqueness and loadings matrix.

**Fix:** Removed the spurious top-eigenvalue term; the objective now sums
only over the discarded (smallest) eigenvalues.

**Verification:** `test_factor_analysis_ml_matches_statsmodels_covariance`
reconstructs `L @ L.T + diag(Psi)` within 1e-3 of `statsmodels.Factor
(method="ml")`'s fitted covariance (previously off by up to 0.67 on a unit
correlation matrix).

---

### 70. `test_factor_analysis_ml_matches_statsmodels_covariance` was flaky across platforms

**Severity:** 3/10 · **Status:** 🟢 `fixed` (V0.9.0)

**Problem:** The test's fixed seed happens to draw a near-Heywood case (one
communality ~1; `statsmodels`' own BFGS warns "Fitting did not converge").
Near that boundary the ML objective's minimum sits on a flat/near-singular
ridge, so two independent BFGS runs converge to visibly different `Psi`/
loadings with almost identical objective value — comparing the raw fitted
covariance matrix (`atol=1e-3`) is sensitive to exactly which point on that
ridge each platform's BLAS/LAPACK backend lands on. CI failed on
`windows-latest` (Python 3.10 and 3.11) with a max cell diff of 1.1e-3,
just over the threshold, while `ubuntu-latest` and this machine's local
Windows run both passed.

**Fix:** Added a primary assertion that the actual minimized quantity — the
concentrated ML discrepancy `F(Psi)` — is no worse (within `1e-4`) than
`statsmodels`' own `F(Psi)`, which is what is genuinely pinned down at the
optimum regardless of which point on a flat ridge either optimizer picks.
The raw-covariance check is kept as a secondary sanity net at a looser
`atol=5e-3`.

**Verification:** Reproduced the exact CI seed locally: `F_mine` exceeds
`F_ref` by `3.6e-5`, comfortably inside the new `1e-4` tolerance (3x
headroom) — the objective-value check is robust where the raw-parameter
comparison was not. Full `tests/statistics`/`tests/library`/`tests/docs`/
`tests/cli` suite re-verified green (215 passed) after the change.

---

### 71. `distributions.VonMises.fit` crashed on every dataset (Bessel overflow)

**Severity:** 7/10 · **Status:** 🟢 `fixed` (V0.10.0)

**Problem:** The concentration solve bracketed `kappa` up to `1e4`, where
`special.i1(k) / special.i0(k)` is `inf / inf = NaN`, so `brentq` aborted with "function
value at x=10000.0 is NaN" — `VonMises.fit` never returned. Found by the new
end-to-end API sweep (`tests/distributions/e2e.py`).

**Fix:** Use the exponentially scaled Bessel functions `i1e / i0e` — the same ratio with no
overflow.

**Verification:** `test_von_mises_fit_recovers_parameters` recovers `(mu=0.3, kappa)` for
`kappa` in {0.5, 2, 20} from 5000 draws (kappa within 5 %).

---

### 72. Six survival fitters inherited the abstract `predict()` and raised

**Severity:** 6/10 · **Status:** 🟢 `fixed` (V0.10.0)

**Problem:** `LifeTable` and the five parametric fitters (`WeibullSurvival`,
`ExponentialSurvival`, `LogNormalSurvival`, `LogLogisticSurvival`, `GompertzSurvival`) never
overrode `SurvivalFitter.predict`, so the documented uniform `predict(times)` surface raised
`NotImplementedError` for them. Found by `tests/survival/e2e.py`.

**Fix:** Parametric `predict` returns `survival_(times)`; `LifeTable.predict` evaluates the
actuarial survival as a right-continuous step function at the interval edges.

**Verification:** `test_every_fitter_exposes_predict_as_survival` — parametric `predict ==
survival_`, life-table `predict` equals `survival_` at the edges and is 1 before the first.

---

### 73. `copulas` vine `loglik(data, raw=False)` raised `NameError`

**Severity:** 5/10 · **Status:** 🟢 `fixed` (V0.10.0)

**Problem:** `vine.py` referenced `as_u_matrix` without importing it, so evaluating a fitted
vine's log-likelihood or AIC on copula-scale data (`raw=False`) crashed. Found by
`tests/copulas/e2e.py`.

**Fix:** Import `as_u_matrix` from `copulas._utils`.

**Verification:** `test_vine_loglik_accepts_copula_scale_data` — finite `loglik`/`aic` on
sampled uniforms, agreeing with the `raw=True` path within noise.

---

### 74. `gaussian_processes.KernelComposition` rejected every non-unit weight

**Severity:** 6/10 · **Status:** 🟢 `fixed` (V0.10.0)

**Problem:** A weight `w != 1` built `KernelProduct([k, w])`, but the shared composite base
rejected the scalar part ("part 1 is not a kernel") even though `KernelProduct` documents
and evaluates scalar factors; `set_params` also indexed over all parts including scalars.
`KernelComposition(..., weights=[0.5, 2.0])` and `3.0 * RBFKernel()` were unusable. Found by
`tests/gaussian_processes/e2e.py`.

**Fix:** `_CompositeBase` accepts scalar parts where the subclass opts in
(`KernelProduct._allow_scalars = True`); parameter get/set map over kernel parts only.

**Verification:** `test_kernel_composition_supports_scalar_weights` — weighted matrix and
diagonal equal the hand-built sum, nested `part0__part0__length_scale` round-trips, `KernelSum`
still rejects scalars.

---

### 75. `SpectralMixtureKernel(dimension=d)` failed its own length check

**Severity:** 4/10 · **Status:** 🟢 `fixed` (V0.10.0)

**Problem:** Passing `dimension` overwrote the `q` component vectors (`means`, `scales`) with
`d`-length vectors and then raised "weights, means and scales must have equal length" for any
`d != q`. Each component is isotropic across inputs, so the reshape was meaningless.

**Fix:** `dimension` is stored as information only; component vectors stay length `q`.

**Verification:** `test_spectral_mixture_dimension_keeps_components` — `q=3, dimension=2`
constructs and evaluates a symmetric kernel matrix on 2-D inputs.

---

### 76. `levy_processes` stable increments crashed on NumPy >= 2.5

**Severity:** 8/10 · **Status:** 🟢 `fixed` (V0.10.0)

**Problem:** `StableProcess`, `StableSubordinator` and `RandomMeasure(kind="stable")` drew
increments with `float(s.rvs(1, random_state=rng))`; the `alpha=2` Gaussian-delegated sampler
returns a length-1 array, and NumPy >= 2.5 raises `TypeError: only 0-dimensional arrays can be
converted to Python scalars` instead of the old deprecation warning. Any Brownian-based
subordination (`SubordinatedProcess(base=StableProcess(alpha=2))`) crashed. Found by
`tests/levy_processes/e2e.py`.

**Fix:** Extract the scalar with `np.asarray(...).ravel()[0]` in all three places.

**Verification:** `test_gaussian_stable_increments_are_scalars` — increments are Python
floats, subordinated sampling and the stable random measure run.

---

### 77. `HullWhiteModel` / `HoLeeModel` could not be constructed for `fit()`

**Severity:** 5/10 · **Status:** 🟢 `fixed` (V0.10.0)

**Problem:** The constructors demanded either a `discount_curve` or both `r0` and `theta`, so
the documented fluent `HullWhiteModel(kappa, sigma).fit(maturities, discount_factors)` path
raised before `fit` could run. Found by `tests/financial_stochastics/e2e.py`.

**Fix:** A bare constructor now yields an unfitted model; `theta()`/`zcb_price()` raise a
clear `RuntimeError` until a curve is fitted or given.

**Verification:** `test_curve_models_support_fluent_fit_from_bare_constructor` — bare
construction, guarded errors, then `fit` reproduces the input discount factors to 1e-9.

---

### 78. `timeseries.VECM.forecast` raised `AttributeError: 'VECM' object has no attribute 'k'`

**Severity:** 7/10 · **Status:** 🟢 `fixed` (V0.10.0)

**Problem:** `fit` kept the system dimension in a local `k`; `forecast` read `self.k`. Every
VECM forecast crashed. Found by `tests/timeseries/e2e.py`.

**Fix:** `fit` stores `self.k`.

**Verification:** `test_vecm_forecast_after_fit` — a 4-step forecast of a cointegrated pair
has shape `(4, 2)` and is finite.

---

### 79. `RegimeSwitching` / `MixtureAutoregressive` fitted a duplicated intercept

**Severity:** 7/10 · **Status:** 🟢 `fixed` (V0.10.0)

**Problem:** `lag_matrix` already returns `[1, y_{t-1}, ..., y_{t-p}]`; both AR variants passed
it through a fit that prepends another constant column, giving a perfectly collinear design
with `p + 2` coefficients per regime, `ar_coefficients_` of length `p + 1` (the second
intercept masquerading as an AR term), split intercepts, and a `predict()` that rejected `p`
lagged values. Found by `tests/timeseries/e2e.py`.

**Fix:** Drop `lag_matrix`'s constant before the switching fit (`RegimeSwitching`) and use
the lag design directly (`MixtureAutoregressive`).

**Verification:** `test_switching_ar_models_have_a_single_intercept` — `p + 1` coefficients per
regime, `ar_coefficients_` of length `p`, `predict` on `(n, p)` lags works.

---

### 80. `ExtendedKalmanFilter.smooth()` / `UnscentedKalmanFilter.smooth()` were unimplemented

**Severity:** 5/10 · **Status:** 🟢 `fixed` (V0.10.0)

**Problem:** Both public smoothers raised `NotImplementedError` (the EKF one claiming
"requires iterated methods", which is not the case for the standard extended RTS pass). Found
by `tests/timeseries/e2e.py`.

**Fix:** Both filters now store the one-step predictions and the cross-covariance
`Cov(x_{t-1|t-1}, x_{t|t-1})` (linearized for the EKF, sigma-point for the UKF) and run the
shared backward RTS recursion of Sarkka (2008).

**Verification:** `test_nonlinear_smoothers_reduce_to_rts_on_a_linear_model` — on a linear
model both agree with the exact `KalmanSmoother` to 1e-3; on a monotone nonlinear model the
smoother MSE is below the filter MSE for both.

---

### 81. Three `queueing` spec names were missing and the conformance test never looked

**Severity:** 4/10 · **Status:** 🟢 `fixed` (V0.10.0)

**Problem:** The checklist lists `ErlangBFormula()`, `ErlangCFormula()` and `EngsetFormula()`
but `queueing` only exported the snake_case functions, and `tests/library`'s
`test_spec_names_present` parametrization omitted both `queueing` and `information_theory`,
so the gap shipped through three releases. Found by the new `install-smoke` wheel check.

**Fix:** Spec-named aliases (`ErlangBFormula = erlang_b_formula`, ...) exported alongside the
snake_case API; `queueing` and `information_theory` added to the conformance parametrization
(with their documented extras pinned).

**Verification:** `test_spec_names_present[queueing]` / `[information_theory]` pass; the wheel
check in `ci.yml` asserts every spec name of every module against the installed copy.

---

### 82. BB1/BB7 Kendall-tau curve cache ignored `delta`

**Severity:** 7/10 · **Status:** 🟢 `fixed` (V0.10.0)

**Problem:** The Archimedean `tau(theta)` curve is cached per class, keyed only on the theta
bounds. For the two-parameter BB1/BB7 families tau also depends on `delta`, so `fit()` — which
scans a delta grid — inverted tau on the curve built for the *first* delta, and afterwards
`kendall_tau()` on any BB1/BB7 instance in the process returned that stale curve's value
(`BB1Copula(1.4, 1.7).kendall_tau()` gave 0.44 instead of 0.65). Surfaced as an
order-dependent failure of `test_archimedean_sampler_margins_and_tau[bb1/bb7]` once the e2e
sweep fitted BB1/BB7 earlier in the same session.

**Fix:** The cache is a per-class dict keyed on `_tau_cache_key()` (BB1/BB7: `delta`). For
those families `_invert_tau` root-finds on the exact integral directly (a curve per delta
would cost 48 integrals each), with a short geometric scan for the sign change because
`tau(theta)` is numerically unreliable at the tiny lower bound.

**Verification:** `test_two_parameter_tau_cache_is_keyed_on_delta` — after a fit, a fresh
instance's `kendall_tau()` equals the exact integral to 1e-9 and the fitted copula's tau
matches the empirical tau to 1e-3; BB1/BB7 fits now recover the generating tau (5 s / 4 s).
