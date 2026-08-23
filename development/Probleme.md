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
